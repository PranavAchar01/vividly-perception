#!/usr/bin/env python
r"""I1: dump a v2 detector's dev-val predictions as COCO detections JSON + sha sidecar.

    # Pod side, inside the trainer's own venv, cwd = the trainer repo root.
    # TRAINING.md carries the concrete pod invocation, data locations included.
    cd "$TRAINER" && .venv/bin/python \
        "$REPO/scripts/detector/dump_detections_v2.py" run \
        -c configs/rtv4/rtv4_s_person_v2_noteacher.yml \
        -r "$CKPT" \
        --out "$OUT"

    # Mac side, normalising a results file produced by another tool:
    uv run python scripts/detector/dump_detections_v2.py convert \
        --in raw_results.json --out outputs/eyes_b1/dumps/dets_345c30a5.json \
        --checkpoint outputs/pod_backup/b1_best_stg2_345c30a5.pth

Everything under `outputs/` is a run output and is not in version control.

===============================================================================
Read this first: `--test-only` does not emit detections (verified 2026-09-05)
===============================================================================
Reconnaissance of both pinned trainers (clones at the PINS.yaml SHAs, read on the Mac):

  RT-DETRv4 @ 55fefaae  train.py  --test-only -> DetSolver.val()
  DEIM      @ 09d35d53  train.py  --test-only -> DetSolver.val()

and in both repos `DetSolver.val()` (`engine/solver/det_solver.py` in each) writes
exactly one artefact:

    torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

`COCOeval.eval` is the accumulate output: the precision/recall/scores ndarrays
over the whole IoU x recall x area x maxDets grid. It contains no boxes, no image
ids and no per-detection scores, so **no per-detection file exists to convert**.
The detections do get built, in `CocoEvaluator.prepare_for_coco_detection()` in rtv4
`engine/data/dataset/coco_eval.py` (DEIM identical), in exactly pycocotools'
results-list format,

    {"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}

but that list is a local in `CocoEvaluator.update()`, handed to `loadRes` and then
discarded (rtv4 `coco_eval.py`, same class). Nothing persists it.

Hence `run` mode below, which re-walks the trainer's own eval loop
(`evaluate()` in `engine/solver/det_engine.py`) with the trainer's own config, dataloader,
EMA weights and postprocessor, and writes that same list to disk. It is not a
re-implementation of the model: every tensor comes from the trainer's objects.

`convert` mode exists for the other direction, normalising a results list that
another tool already produced, and it refuses an `eval.pth` with a pointer back
to this header rather than emitting a plausible-looking empty file.

Recorded outcome: `run` mode executed on the pod on 2026-09-05 and produced the four
dumps behind the I1 table. All four sidecars record `mode: run`, `score_floor` 0.0,
`limit` null and 1,800 of 1,800 images, 540,000 rows apiece
(research/eyes-instrument-grading.md). The `--compare-ap` watch item is closed, and it was
closed on the full split rather than by a `--limit` smoke run: the `345c30a5` identity dump
was re-graded on the Mac
with pycocotools at AP 0.8355529764 against the training log's `best_stat` of
0.8355529763877473, a delta of exactly 0.0 (same page). A dump whose re-grade does not
match its checkpoint's `best_stat` on the full split is a stop, not a rounding matter.

Sidecar: `<out>.sidecar.json` carries {checkpoint_sha256, config, date, ...}.
Every accuracy number carries its checkpoint hash (POLICY.md, rule 10): no AP,
precision or FP number leaves this pipeline without that sha256 beside it.
scripts/eval/detector_precision_at_thr.py reads the sidecar automatically and
reports the hash as unknown when it is missing.

Exit codes: 0 ok, 3 config/prerequisite problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_CONFIG = 3

RESULT_KEYS = ("image_id", "category_id", "bbox", "score")


# --------------------------------------------------------------------------- util


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_head(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def validate_results(results: list[Any], source: str) -> list[dict[str, Any]]:
    """Every entry must be a pycocotools results-list record. Fail loud, fail early."""
    if not isinstance(results, list):
        raise ValueError(f"{source}: expected a list of detections, got {type(results).__name__}")
    for i, rec in enumerate(results[:512]):  # a bad dump is bad in its first records
        if not isinstance(rec, dict):
            raise ValueError(f"{source}[{i}]: expected a dict, got {type(rec).__name__}")
        missing = [k for k in RESULT_KEYS if k not in rec]
        if missing:
            raise ValueError(f"{source}[{i}]: missing {missing}; keys present: {sorted(rec)}")
        if len(rec["bbox"]) != 4:
            raise ValueError(f"{source}[{i}]: bbox must be [x, y, w, h], got {rec['bbox']!r}")
    return results


def write_dump(
    results: list[dict[str, Any]],
    out: Path,
    meta: dict[str, Any],
) -> tuple[Path, Path]:
    """Write the detections list and its sidecar; return both paths."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results) + "\n")
    side = out.with_suffix(out.suffix + ".sidecar.json")
    payload = {
        **meta,
        "n_detections": len(results),
        "n_images_with_detections": len({r["image_id"] for r in results}),
        "detections_sha256": sha256_file(out),
        "format": "pycocotools results list: image_id/category_id/bbox[xywh]/score",
        "written_by": "scripts/detector/dump_detections_v2.py",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    side.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}  ({len(results)} detections)")
    print(f"wrote {side}")
    print(f"checkpoint_sha256 = {payload.get('checkpoint_sha256')}")
    return out, side


# ------------------------------------------------------------------------ convert


def _load_any(path: Path) -> Any:
    if path.suffix in (".pth", ".pt"):
        import torch  # local: the Mac venv has torch, the DEIM venv has it too

        return torch.load(path, map_location="cpu", weights_only=False)
    with path.open() as fh:
        return json.load(fh)


def cmd_convert(args: argparse.Namespace) -> int:
    src: Path = args.input
    if not src.is_file():
        print(f"FATAL: no such file: {src}", file=sys.stderr)
        return EXIT_CONFIG

    raw = _load_any(src)

    # The single most likely mistake: pointing this at the trainer's eval.pth.
    if isinstance(raw, dict) and {"precision", "recall"} <= set(raw):
        print(
            f"FATAL: {src} is a COCOeval.accumulate() dump (keys {sorted(raw)[:6]}...), not "
            "detections. --test-only writes only that file; it holds no boxes. Produce real "
            "detections with `dump_detections_v2.py run` on the pod; see this script's header.",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    if isinstance(raw, dict):
        for key in ("annotations", "detections", "results"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            print(
                f"FATAL: {src} is a dict with keys {sorted(raw)[:8]} and no detections list "
                "under 'annotations'/'detections'/'results'.",
                file=sys.stderr,
            )
            return EXIT_CONFIG

    try:
        results = validate_results(raw, str(src))
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    meta: dict[str, Any] = {
        "mode": "convert",
        "source": str(src),
        "source_sha256": sha256_file(src),
        "config": args.config,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": (
            args.checkpoint_sha256
            if args.checkpoint_sha256
            else (sha256_file(args.checkpoint) if args.checkpoint else None)
        ),
        "note": args.note,
    }
    if meta["checkpoint_sha256"] is None:
        print(
            "WARNING: no --checkpoint / --checkpoint-sha256 given. The sidecar will carry a null "
            "sha256, and the rule that every accuracy number carries its checkpoint hash "
            "(POLICY.md, rule 10) forbids quoting any number from it.",
            file=sys.stderr,
        )
    write_dump(results, args.out, meta)
    return EXIT_OK


# ---------------------------------------------------------------------------- run


def _unflip_boxes_xyxy(boxes, width: float):
    """Un-mirror boxes predicted on an hflipped image. Boxes are xyxy in original pixels.

    The postprocessor multiplies normalised cxcywh by `orig_target_sizes.repeat(1, 2)`
    (the box-scaling step in rtv4 `engine/rtv4/postprocessor.py`), and `orig_size` is
    stored **[w, h]** by the dataset in `engine/data/dataset/coco_dataset.py`, so width is
    index 0, and the mirror is about that same width.
    """
    flipped = boxes.clone()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return flipped


def cmd_run(args: argparse.Namespace) -> int:
    """Re-walk the trainer's own eval loop and persist the detections it discards.

    Mirrors, in order: `train.py` (YAMLConfig plus HGNetv2 pretrained=False),
    `_solver.BaseSolver.eval()` (val_dataloader + resume), `det_solver.val()`
    (`module = ema.module if ema else model`), `det_engine.evaluate()`
    (samples -> model -> postprocessor(outputs, orig_target_sizes)) and
    `coco_eval.prepare_for_coco_detection()` (xyxy -> xywh, .tolist()).
    """
    # `python /abs/path/to/this_script.py` puts the script's directory on sys.path, not
    # the cwd, so the run-from-trainer-root contract below needs the cwd added by hand
    # (first observed live on the pod, 2026-09-05).
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        import torch
        from engine.core import YAMLConfig  # the trainer's package, not this repository's
        from engine.solver import TASKS
    except ImportError as exc:  # pragma: no cover - pod-only path
        print(
            f"FATAL: {exc}. `run` mode must execute from the trainer repository root with "
            "that trainer's virtual environment; TRAINING.md gives the invocation in full.",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    ckpt = Path(args.resume)
    if not ckpt.is_file():
        print(f"FATAL: no such checkpoint: {ckpt}", file=sys.stderr)
        return EXIT_CONFIG

    cfg = YAMLConfig(args.config, resume=str(ckpt))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # as train.py does before eval

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()  # builds val_dataloader + evaluator, loads `resume` into model/ema

    module = solver.ema.module if solver.ema else solver.model
    module.eval()
    postprocessor = solver.postprocessor
    device = solver.device

    results: list[dict[str, Any]] = []
    n_images = 0
    with torch.no_grad():
        for samples, targets in solver.val_dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

            if args.hflip:
                samples = torch.flip(samples, dims=[3])
            outputs = module(samples)
            preds = postprocessor(outputs, orig_target_sizes)

            for target, pred in zip(targets, preds, strict=True):
                boxes = pred["boxes"]  # xyxy in original-image pixels
                if args.hflip:
                    boxes = _unflip_boxes_xyxy(boxes, float(target["orig_size"][0].item()))
                scores = pred["scores"]
                labels = pred["labels"]
                keep = scores >= args.score_floor
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if boxes.numel() == 0:
                    continue
                xywh = torch.stack(
                    [
                        boxes[:, 0],
                        boxes[:, 1],
                        boxes[:, 2] - boxes[:, 0],
                        boxes[:, 3] - boxes[:, 1],
                    ],
                    dim=1,
                ).tolist()
                img_id = int(target["image_id"].item())
                results.extend(
                    {
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [float(v) for v in box],
                        "score": float(score),
                    }
                    for box, score, label in zip(
                        xywh, scores.tolist(), labels.tolist(), strict=True
                    )
                )
            n_images += len(targets)
            if args.limit and n_images >= args.limit:
                print(f"--limit {args.limit}: stopping after {n_images} images (smoke, not a run)")
                break

    validate_results(results, "run")
    meta: dict[str, Any] = {
        "mode": "run",
        "config": args.config,
        "config_sha256": sha256_file(Path(args.config)) if Path(args.config).is_file() else None,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "trainer_repo": str(Path.cwd()),
        "trainer_git_head": git_head(Path.cwd()),
        "weights_used": "ema.module" if solver.ema else "model",
        "hflip": bool(args.hflip),
        "score_floor": args.score_floor,
        "limit": args.limit or None,
        "n_images": n_images,
        "ann_file": cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {}).get("ann_file"),
        "note": args.note,
    }
    write_dump(results, args.out, meta)

    if args.compare_ap:  # pragma: no cover - pod-only path
        evaluator = solver.evaluator
        evaluator.cleanup()
        coco_eval = evaluator.coco_eval["bbox"]
        coco_eval.cocoDt = evaluator.coco_gt.loadRes(results)
        coco_eval.params.imgIds = sorted({r["image_id"] for r in results})
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        print(
            "Parity check: the AP above is over the dumped boxes. On a full split it must "
            "match this checkpoint's `best_stat` in the training log."
        )
    return EXIT_OK


# ---------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="pod-side: walk the trainer's eval loop and dump boxes")
    run.add_argument("-c", "--config", required=True, help="the trainer's own yml")
    run.add_argument("-r", "--resume", required=True, help="checkpoint (.pth) to score")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument(
        "--hflip", action="store_true", help="TTA dump: mirror in, un-mirror boxes out"
    )
    run.add_argument("--score-floor", type=float, default=0.0, help="drop boxes below this first")
    run.add_argument("--limit", type=int, default=0, help="smoke only: stop after N images")
    run.add_argument("--compare-ap", action="store_true", help="re-grade the dump and print AP")
    run.add_argument("--note", default="")
    run.set_defaults(func=cmd_run)

    conv = sub.add_parser("convert", help="normalise an existing results list + write the sidecar")
    conv.add_argument("--in", dest="input", required=True, type=Path)
    conv.add_argument("--out", required=True, type=Path)
    conv.add_argument("--checkpoint", type=Path, default=None, help="hashed into the sidecar")
    conv.add_argument("--checkpoint-sha256", default=None, help="when the .pth is not local")
    conv.add_argument("--config", default=None)
    conv.add_argument("--note", default="")
    conv.set_defaults(func=cmd_convert)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
