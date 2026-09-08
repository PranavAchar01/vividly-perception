"""Evaluation runner internals. The script `scripts/eval/eval_runner.py` is a thin
--config shell; everything testable lives here.

Execution model: resolve the frozen subset, then per condition: materialise the images and
their ground truth, detect and estimate pose with structural Timing, serialise predictions,
score COCOeval bbox and OKS keypoints, take latency percentiles, write the sink row and the
deltas against the clean row, then emit results.json and results.md.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vividly.eval.metrics import MetricBlock, coco_bbox_eval, coco_keypoint_eval
from vividly.models.base import Detections, Poses, Timing

LATENCY_STAGES = ("detect", "pose", "e2e")


@dataclass(frozen=True)
class ImageRef:
    image_id: int
    file_name: str


def load_subset(subset_file: Path) -> list[ImageRef]:
    """The frozen, committed subset. Never a glob."""
    d = json.loads(Path(subset_file).read_text())
    return [ImageRef(i, f) for i, f in zip(d["image_ids"], d["file_names"], strict=True)]


def subset_fingerprint(subset_file: Path) -> str:
    return json.loads(Path(subset_file).read_text())["fingerprint"]


def host_info() -> dict:
    """The host description recorded next to every latency number."""
    import platform
    import subprocess
    import sys

    cpu = "unknown"
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    elif sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    info = {
        "cpu": cpu,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


HOST = host_info()


def latency_stats(timings: list[Timing]) -> dict[str, float]:
    """p50/p95/mean per stage, in ms per call and ms per item.

    A row missing these keys is a hard failure downstream (assert_row_complete),
    because reporting latency alongside accuracy is only real if it is enforced.
    """
    out: dict[str, float] = {}
    for stage in LATENCY_STAGES:
        ms = np.array([t.ms for t in timings if t.stage == stage], dtype=float)
        items = np.array([t.n_items for t in timings if t.stage == stage], dtype=float)
        if ms.size == 0:
            # An absent stage is stated, never implied: n_calls = 0 says "this stage
            # never ran", while the NaN percentiles below make assert_row_complete
            # reject the row rather than let a NaN reach results.json (NaN is not
            # legal JSON, and json.dumps emits it anyway).
            out[f"{stage}_p50_ms"] = float("nan")
            out[f"{stage}_p95_ms"] = float("nan")
            out[f"{stage}_n_calls"] = 0
            continue
        out[f"{stage}_p50_ms"] = float(np.percentile(ms, 50))
        out[f"{stage}_p95_ms"] = float(np.percentile(ms, 95))
        out[f"{stage}_mean_ms"] = float(ms.mean())
        out[f"{stage}_ms_per_item"] = float(ms.sum() / max(items.sum(), 1))
        out[f"{stage}_n_calls"] = int(ms.size)
    return out


def _missing(value: object) -> bool:
    """True when a cell cannot stand next to an accuracy number.

    None and "" are the obvious cases. NaN and inf are the load-bearing ones: a
    condition where no image yields a detection leaves the pose stage with zero
    Timings, latency_stats emits NaN percentiles for it, and a NaN row used to sail
    through the gate and land in results.json, which then stops being strict JSON.
    Every latency cell is built with `float(...)` in latency_stats, so the float
    branch is the one that sees them.
    """
    if value is None or value == "":
        return True
    return isinstance(value, float) and not math.isfinite(value)


def assert_row_complete(row: dict) -> None:
    missing = [
        k
        for k in (
            "detect_p50_ms",
            "detect_p95_ms",
            "pose_p50_ms",
            "pose_p95_ms",
            "e2e_p50_ms",
            "e2e_p95_ms",
            "host_cpu",
            "det_provenance",
            "pose_provenance",
        )
        if _missing(row.get(k))
    ]
    if missing:
        raise AssertionError(
            f"eval row '{row.get('condition')}' is missing or non-finite at {missing}. "
            "Latency and provenance are recorded next to every accuracy number."
        )


def to_coco_detections(image_id: int, d: Detections) -> list[dict]:
    out = []
    for (x1, y1, x2, y2), s in zip(d.xyxy, d.scores, strict=True):
        out.append(
            {
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            }
        )
    return out


def to_coco_keypoints(image_id: int, d: Detections, p: Poses) -> list[dict]:
    out = []
    for i in range(len(p)):
        kps = p.keypoints[i]
        scs = p.scores[i]
        flat: list[float] = []
        for (x, y), s in zip(kps, scs, strict=True):
            flat += [float(x), float(y), 2.0 if s > 0.0 else 0.0]
        out.append(
            {
                "image_id": image_id,
                "category_id": 1,
                "keypoints": flat,
                # instance score: detector box score if aligned, else mean kpt confidence
                "score": float(d.scores[i]) if i < len(d) else float(scs.mean()),
            }
        )
    return out


def materialize_condition(
    cond: dict, images: list[ImageRef], cfg: dict, repo_root: Path
) -> tuple[list[Path], Path]:
    """Return (per-image paths aligned with `images`, GT annotation file).

    One condition kind is supported: `passthrough`, the images as the dataset ships them
    with the annotation file the registry resolves for that split. A condition that asks
    for anything else is refused rather than silently scored against the wrong pixels.
    """
    from vividly.data.registry import default_registry

    reg = default_registry()
    name = cfg["dataset"]["registry_name"]
    split = cfg["dataset"]["split"]
    ann_file = reg.annotation_file(name, split)
    # image_dir() confines layout.images[split] to the licence bucket the same way
    # annotation_file() confines the annotation path. The earlier dataset_root() /
    # image_rel() join could read imagery through a symbolic link or a '..' segment in
    # layout.images that escaped the bucket, which is what POLICY.md, rule 6 forbids.
    image_root = reg.image_dir(name, split)

    if cond["kind"] == "passthrough":
        return [image_root / r.file_name for r in images], ann_file

    raise ValueError(
        f"condition {cond.get('name')!r} declares kind {cond['kind']!r}; the only supported "
        "kind is 'passthrough'. A condition that rewrites the pixels needs its own "
        "materialiser and its own ground truth, and neither is registered here."
    )


def run_condition(
    cond: dict, images: list[ImageRef], det, pose, cfg: dict, repo_root: Path
) -> tuple[dict, MetricBlock, MetricBlock]:
    """One row of the delta table."""
    import cv2

    paths, gt_json = materialize_condition(cond, images, cfg, repo_root)
    det.reset_timings()
    pose.reset_timings()
    bbox_preds: list[dict] = []
    kpt_preds: list[dict] = []
    e2e: list[Timing] = []
    for ref, path in zip(images, paths, strict=True):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        t0 = time.perf_counter()
        d = det.detect_one(img)
        p = pose.estimate(img, d.xyxy)
        # One e2e Timing per image. Owned by the runner, not by a model, because e2e
        # spans two models plus the decode, so no single backend can measure it.
        e2e.append(Timing("e2e", (time.perf_counter() - t0) * 1000.0, 1, "runner"))
        bbox_preds += to_coco_detections(ref.image_id, d)
        kpt_preds += to_coco_keypoints(ref.image_id, d, p)
    img_ids = [r.image_id for r in images]
    m_box = coco_bbox_eval(gt_json, bbox_preds, img_ids)
    m_kpt = coco_keypoint_eval(gt_json, kpt_preds, img_ids)
    row = {
        "condition": cond["name"],
        "kind": cond["kind"],
        "n_images": len(images),
        "det_AP": m_box.values["AP"],
        "det_AP50": m_box.values["AP50"],
        "det_AR100": m_box.values["AR@100"],
        "kpt_AP": m_kpt.values["AP"],
        "kpt_AP50": m_kpt.values["AP50"],
        "kpt_AR": m_kpt.values["AR"],
        **latency_stats(det.timings + pose.timings + e2e),
        "host_cpu": HOST["cpu"],
        "det_provenance": det.provenance.label(),
        "pose_provenance": pose.provenance.label(),
        "gt_file": str(gt_json),
    }
    assert_row_complete(row)
    return row, m_box, m_kpt


def results_markdown(rows: list[dict], cfg: dict, fingerprint: str) -> str:
    hdr = (
        "| condition | det AP | Δ det AP | kpt AP | Δ kpt AP | det p50 ms | pose p50 ms | e2e p95 ms |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['det_AP']:.3f} | {r.get('delta_det_AP', 0.0):+.3f} "
            f"| {r['kpt_AP']:.3f} | {r.get('delta_kpt_AP', 0.0):+.3f} "
            f"| {r['detect_p50_ms']:.1f} | {r['pose_p50_ms']:.1f} | {r['e2e_p95_ms']:.1f} |"
        )
    foot = (
        f"\n\nmodels: {rows[0]['det_provenance']} / {rows[0]['pose_provenance']}\n"
        f"host:   {HOST['cpu']}, onnxruntime {HOST.get('onnxruntime', '?')}, "
        f"{(HOST.get('ort_providers') or ['?'])[-1]}\n"
        f"subset: {Path(cfg['dataset']['subset_file']).stem} fingerprint {fingerprint}\n"
        "NOTE:   eval-smoke provenance. Not a Vividly accuracy claim.\n"
    )
    return hdr + "\n".join(lines) + foot


def run(cfg: dict, repo_root: Path) -> Path:
    """Score every declared condition on the frozen subset. Returns the output directory."""
    from vividly.eval.sinks import build_sink
    from vividly.models.factory import build_detector, build_pose

    subset_file = repo_root / cfg["dataset"]["subset_file"]
    images = load_subset(subset_file)
    fp = subset_fingerprint(subset_file)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / cfg["logging"]["out_root"] / f"{cfg['run_name']}_{ts}_{fp[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = build_sink(cfg, out_dir)
    sink.log_config(cfg)

    det = build_detector(cfg["detector"])
    pose = build_pose(cfg["pose"])
    det.warmup()
    pose.warmup()

    rows: list[dict] = []
    clean_row: dict | None = None
    for cond in cfg["conditions"]:
        row, _mb, _mk = run_condition(cond, images, det, pose, cfg, repo_root)
        if cond["name"] == "clean":
            clean_row = row
        if clean_row is not None:
            row["delta_det_AP"] = row["det_AP"] - clean_row["det_AP"]
            row["delta_kpt_AP"] = row["kpt_AP"] - clean_row["kpt_AP"]
        rows.append(row)
        sink.log_row(row)
        print(
            f"  [{row['condition']:>18s}] det AP {row['det_AP']:.3f}  "
            f"kpt AP {row['kpt_AP']:.3f}  e2e p50 {row['e2e_p50_ms']:.0f} ms"
        )

    results = {
        "run_name": cfg["run_name"],
        "subset_fingerprint": fp,
        "host": HOST,
        "config": cfg,
        "rows": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    (out_dir / "results.md").write_text(results_markdown(rows, cfg, fp))
    sink.log_summary({"rows": rows, "subset_fingerprint": fp})
    sink.close()
    return out_dir
