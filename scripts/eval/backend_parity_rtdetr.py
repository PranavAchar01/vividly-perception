#!/usr/bin/env python
r"""G1 gate: does the `rtdetr_onnx` pipeline backend reproduce the I2 grader, box for box?

    uv run python scripts/eval/backend_parity_rtdetr.py

No extraction runs with a new detector backend until this passes. The numbers for the eyes
model of record, every AP and every precision-at-threshold row in research/, were graded by
`scripts/eval/eval_detector_v2_onnx.py`. A pipeline backend that is even slightly
different is a second, ungraded detector wearing the name of the model of record, and the
drift shows up as tracks that never quite match the evaluated behaviour, months later, with
no way left to tell which side moved.

The two paths compared here share the ONNX file and the onnxruntime session shape, so the
only things that can differ are the ones this backend had to re-implement:

  A  the backend, built through the factory exactly as `_build_perception` and the
     perception loop in `scripts/run_pipeline.py` build and call it, fed the frame as
     `cv2.imread(path)`, that is BGR, the way `sources/replay.py` hands frames over;
  B  the grader's own `preprocess()`, imported from `scripts/eval/eval_detector_v2_onnx.py`
     and not re-typed here, fed the file the way the grader feeds it (a path, opened with
     PIL), then run through the same three-output session and cut at the same score floor.

Path A converts BGR->RGB and resizes in PIL space to get there; path B never leaves RGB.
If those two are not the same tensor, for instance if the backend had reached for
`cv2.resize(..., INTER_LINEAR)`, scores move by ~0.15 and this gate fails. It is
meant to fail in that case: the fix is the backend, never a looser bar here.

The bar (all four, or the verdict is fail):

    identical detection counts on >= N-1 of N frames
    median |box coordinate delta| <= 0.01 px
    max    |box coordinate delta| <= 1.00 px
    max    |score delta|          <= 1e-3

Those are near-exact on purpose. Two paths sharing a graph, a session and a preprocessing
recipe should agree bitwise; the small slack is for fp32 noise, not for a different resize.

Frames. `outputs/eyes_b1/parity_sample/` holds 20 dev-val frames but no expected outputs
(only SHA256SUMS.txt), so it is a byte-fixture for `onnx_parity_v2.py`, not a graded set.
This gate therefore samples deterministically from the full 1,800-frame pod-byte dev-val
tree, `outputs/eyes_b1/devval_frames_podbytes/`: sorted relative paths, evenly spaced,
so the same N frames are drawn on every run and the sample spans all 30 cameras rather
than clustering in the first one. Both trees are run outputs under `outputs/`, which is
not in version control.

Exit codes: 0 pass, 1 fail, 3 config/prerequisite problem.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CONFIG = 3

REPO = Path(__file__).resolve().parents[2]
GRADER = Path(__file__).resolve().with_name("eval_detector_v2_onnx.py")

DEFAULT_ONNX = REPO / "outputs/eyes_b1/b1_best_stg2_345c30a5.onnx"
DEFAULT_FRAMES = REPO / "outputs/eyes_b1/devval_frames_podbytes"
FALLBACK_FRAMES = REPO / "outputs/eyes_b1/parity_sample"

# The bar. Named constants so the verdict line can print the number it enforced.
MAX_COUNT_MISMATCHES = 1
MEDIAN_BOX_DELTA_PX = 0.01
MAX_BOX_DELTA_PX = 1.0
MAX_SCORE_DELTA = 1e-3


def load_grader() -> ModuleType:
    """Import the I2 grader by path. `scripts/` is not a package.

    Same loader shape as `_load_script` in `tests/test_detector_precision.py`, including the
    sys.modules insert before exec_module.
    """
    spec = importlib.util.spec_from_file_location("eval_detector_v2_onnx", GRADER)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load the grader from {GRADER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pick_frames(root: Path, n: int) -> tuple[list[Path], int]:
    """N frames plus the tree's total, deterministic and spread.

    Sorted relative paths, evenly spaced indices: `np.linspace(...).round()` over the
    sorted list rather than a random generator, so the sample is a property of the
    directory and not of an unrecorded seed. The total is returned with the sample so the
    caller can report it without walking the tree a second time.
    """
    all_frames = sorted(root.rglob("*.jpg"), key=lambda p: p.relative_to(root).as_posix())
    if not all_frames:
        raise FileNotFoundError(f"no .jpg frames under {root}")
    if len(all_frames) <= n:
        return all_frames, len(all_frames)
    idx = np.unique(np.linspace(0, len(all_frames) - 1, n).round().astype(int))
    return [all_frames[i] for i in idx], len(all_frames)


def reference_detections(
    grader: ModuleType, session: Any, out_names: list[str], path: Path, score_thr: float
) -> tuple[np.ndarray, np.ndarray]:
    """The grader's own path: its `preprocess`, its session call, its score cut.

    Mirrors the per-image inference loop in `eval_detector_v2_onnx.py`. The grader takes
    width/height from the COCO GT record; here they come from the file itself, which is
    the same number (the GT records the image's true size) and keeps this gate runnable
    without an annotations file.
    """
    from PIL import Image

    blob = grader.preprocess(path)
    with Image.open(path) as im:
        width, height = im.size
    sizes = np.array([[int(width), int(height)]], dtype=np.int64)
    outs = session.run(out_names, {"images": blob, "orig_target_sizes": sizes})
    named = dict(zip(out_names, outs, strict=True))
    boxes = np.asarray(named["boxes"][0], dtype=np.float64)
    scores = np.asarray(named["scores"][0], dtype=np.float64)
    keep = scores >= score_thr
    return boxes[keep], scores[keep]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--frames", type=Path, default=None, help=f"default: {DEFAULT_FRAMES}")
    ap.add_argument("--n", type=int, default=50, help="frames to compare (default 50)")
    ap.add_argument("--score-thr", type=float, default=0.3, help="shared threshold, both paths")
    ap.add_argument("--out-json", type=Path, default=None, help="optional machine-readable copy")
    args = ap.parse_args(argv)

    if not args.onnx.is_file():
        print(f"FATAL: no such ONNX: {args.onnx}", file=sys.stderr)
        return EXIT_CONFIG
    root = args.frames
    if root is None:
        root = DEFAULT_FRAMES if DEFAULT_FRAMES.is_dir() else FALLBACK_FRAMES
    if not root.is_dir():
        print(f"FATAL: not a directory: {root}", file=sys.stderr)
        return EXIT_CONFIG

    import cv2
    import onnxruntime as ort

    from vividly.models.factory import build_detector

    grader = load_grader()
    try:
        frames, n_available = pick_frames(root, args.n)
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    # A: through the factory, exactly as run_pipeline.py builds it from a config dict.
    detector = build_detector(
        {
            "backend": "rtdetr_onnx",
            "onnx": str(args.onnx),
            "input_size": [640, 640],
            "device": "cpu",
            "score_thr": args.score_thr,
        }
    )
    # B: the grader's session, built the way eval_detector_v2_onnx.py builds its own,
    # on the CPU execution provider.
    ref_session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    ref_out_names = [o.name for o in ref_session.get_outputs()]

    rows: list[dict[str, Any]] = []
    all_box_deltas: list[float] = []
    backend_ms: list[float] = []
    for path in frames:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"FATAL: cv2 could not decode {path}", file=sys.stderr)
            return EXIT_CONFIG
        t0 = time.perf_counter()
        got = detector.detect_one(image)
        backend_ms.append((time.perf_counter() - t0) * 1000.0)
        ref_boxes, ref_scores = reference_detections(
            grader, ref_session, ref_out_names, path, args.score_thr
        )

        n_a, n_b = len(got), len(ref_boxes)
        # Both lists are the postprocessor's score-descending top-300 cut at the same
        # floor, so rank k on one side IS rank k on the other. Comparing by rank is the
        # strict test; an IoU match would hide a systematic sub-pixel shift.
        m = min(n_a, n_b)
        if m:
            d_box = np.abs(got.xyxy[:m].astype(np.float64) - ref_boxes[:m])
            d_score = np.abs(got.scores[:m].astype(np.float64) - ref_scores[:m])
            flat = d_box.ravel()
            all_box_deltas.extend(flat.tolist())
            row = {
                "frame": path.relative_to(root).as_posix(),
                "n_backend": n_a,
                "n_reference": n_b,
                "max_box_px": float(flat.max()),
                "median_box_px": float(np.median(flat)),
                "max_score": float(d_score.max()),
            }
        else:
            row = {
                "frame": path.relative_to(root).as_posix(),
                "n_backend": n_a,
                "n_reference": n_b,
                "max_box_px": 0.0,
                "median_box_px": 0.0,
                "max_score": 0.0,
            }
        rows.append(row)

    n_frames = len(rows)
    mismatched = [r for r in rows if r["n_backend"] != r["n_reference"]]
    max_box = max((r["max_box_px"] for r in rows), default=0.0)
    max_score = max((r["max_score"] for r in rows), default=0.0)
    median_box = statistics.median(all_box_deltas) if all_box_deltas else 0.0
    n_dets = sum(r["n_backend"] for r in rows)

    # ---- table
    print(f"onnx      : {args.onnx}")
    print(f"frames    : {root}  (n={n_frames} of {n_available}, deterministic even spacing)")
    print(f"score_thr : {args.score_thr}  (both paths)")
    print("backend   : rtdetr_onnx via vividly.models.factory.build_detector")
    print(f"reference : {GRADER.relative_to(REPO)} preprocess() + CPUExecutionProvider")
    print()
    head = f"{'frame':<34} {'n_be':>5} {'n_ref':>5} {'max_box_px':>11} {'med_box_px':>11} {'max_dscore':>11}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['frame']:<34} {r['n_backend']:>5} {r['n_reference']:>5} "
            f"{r['max_box_px']:>11.6f} {r['median_box_px']:>11.6f} {r['max_score']:>11.3e}"
        )
    print("-" * len(head))
    print(
        f"{'total':<34} {n_dets:>5} {sum(r['n_reference'] for r in rows):>5} "
        f"{max_box:>11.6f} {median_box:>11.6f} {max_score:>11.3e}"
    )
    print()
    print(
        f"frames with identical counts : {n_frames - len(mismatched)}/{n_frames} "
        f"(bar: >= {n_frames - MAX_COUNT_MISMATCHES})"
    )
    print(f"median |box delta| (px)      : {median_box:.6f}  (bar: <= {MEDIAN_BOX_DELTA_PX})")
    print(f"max    |box delta| (px)      : {max_box:.6f}  (bar: <= {MAX_BOX_DELTA_PX})")
    print(f"max    |score delta|         : {max_score:.3e}  (bar: <= {MAX_SCORE_DELTA:.0e})")
    print(
        f"backend latency              : mean {np.mean(backend_ms):.1f} ms/frame, "
        f"p95 {np.percentile(backend_ms, 95):.1f} ms/frame (CPU, decode excluded)"
    )

    checks = {
        "counts": len(mismatched) <= MAX_COUNT_MISMATCHES,
        "median_box_px": median_box <= MEDIAN_BOX_DELTA_PX,
        "max_box_px": max_box <= MAX_BOX_DELTA_PX,
        "max_score_delta": max_score <= MAX_SCORE_DELTA,
    }
    ok = all(checks.values())
    failed = sorted(k for k, v in checks.items() if not v)
    verdict = (
        f"G1 parity verdict: {'pass' if ok else 'fail'} "
        f"({n_frames} frames, {n_dets} detections at score_thr={args.score_thr}; "
        f"counts {n_frames - len(mismatched)}/{n_frames}, median {median_box:.6f} px, "
        f"max {max_box:.6f} px, max dscore {max_score:.3e})" + ("" if ok else f"; failed: {failed}")
    )
    print()
    print(verdict)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "written_by": "scripts/eval/backend_parity_rtdetr.py",
                    "onnx": str(args.onnx),
                    "frames_root": str(root),
                    "score_thr": args.score_thr,
                    "n_frames": n_frames,
                    "n_detections": n_dets,
                    "summary": {
                        "frames_with_identical_counts": n_frames - len(mismatched),
                        "median_box_delta_px": median_box,
                        "max_box_delta_px": max_box,
                        "max_score_delta": max_score,
                    },
                    "bar": {
                        "max_count_mismatches": MAX_COUNT_MISMATCHES,
                        "median_box_delta_px": MEDIAN_BOX_DELTA_PX,
                        "max_box_delta_px": MAX_BOX_DELTA_PX,
                        "max_score_delta": MAX_SCORE_DELTA,
                    },
                    "checks": checks,
                    "verdict": "pass" if ok else "fail",
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.out_json}")

    return EXIT_OK if ok else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
