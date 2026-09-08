"""RT-DETRv4 person detector over ONNX Runtime. No torch, no mm stack, no rtmlib.

Same runtime shape as `rtmlib_backend` and `reid_backend`: onnxruntime on the CPU
execution provider, numpy in and numpy out. What differs is the graph. The eyes model of
record (RT-DETRv4-S, checkpoint 345c30a5, committed at `final_models/eyes/model.onnx`) is
a DETR-family model exported by `tools/deployment/export_onnx.py` at trainer pin 55fefaae,
so it takes two inputs and emits three outputs:

    images            (N, 3, 640, 640) float32   the letterbox-free 0..1 RGB batch
    orig_target_sizes (N, 2)           int64     [width, height] in original pixels

and returns

    labels            (N, 300) int64     class index per query
    boxes             (N, 300, 4) float32  xyxy, already in original pixels
    scores            (N, 300) float32   score per query, descending

There is no anchor decode, no NMS and no letterbox unpadding to do here: the exported
graph wraps `postprocessor.deploy()`, which multiplies its normalised cxcywh by
`orig_target_sizes.repeat(1, 2)` (the box-scaling step in RT-DETRv4
`engine/rtv4/postprocessor.py`, set out in the preprocessing note in the module docstring
of `scripts/eval/onnx_parity_v2.py`). Hand it the original width and height and the boxes
come back in the frame the caller passed in, which is exactly the coordinate convention
`base.py` requires.

Preprocessing is the trainer's validation pipeline, not a guess. The RT-DETRv4
`configs/base/dataloader.yml` `val_dataloader` is `Resize size=[640, 640]` then
`ConvertPILImage(dtype='float32', scale=True)` and nothing else:

  * plain resize to a fixed 640x640, with no letterbox and no aspect ratio preserved;
  * `/ 255.` only, with no ImageNet mean and standard deviation normalisation anywhere
    in this pipeline;
  * the resize happens in PIL space (torchvision v2 `Resize` over a PIL input calls
    `Image.resize((w, h), BILINEAR)`), which is an antialiased triangle filter whose
    support scales with the reduction factor. `cv2.resize(..., INTER_LINEAR)` is not
    that: it samples 2x2 neighbourhoods with no prefilter, and substituting it here
    moves a 1920x1080 frame to 640x640 with pixel differences of up to 0.4 out of 1.0 in
    the 0..1 tensor, and shifts detection scores by about 0.15 (measured on the parity
    sample). That is why this module reaches for Pillow rather than staying inside cv2:
    parity with the repository's I2 grader (`preprocess` in
    `scripts/eval/eval_detector_v2_onnx.py`) beats dependency tidiness, and
    `scripts/eval/backend_parity_rtdetr.py` is the gate that keeps it honest.

Colour is a contract, and the two sides disagree by default. `base.py` says images cross
this boundary as OpenCV BGR, and the perception loop in `scripts/run_pipeline.py` hands
`frame.image` straight from a cv2 decode. The training and evaluation path is RGB: the
grader opens the file with `Image.open(...).convert("RGB")`. So the conversion from BGR to
RGB belongs here, inside the backend, once, where it cannot be forgotten. Feeding this
graph BGR does not raise, it just quietly costs accuracy. Callers keep handing BGR; that
is the contract.

Two deliberate departures from `RtmlibDetector`, both about not corrupting the numbers:

  * No box clipping. `rtmlib_backend._forward_one` clips to the frame because letterboxed
    RTMDet coordinates can land far outside it; this postprocessor already emits original
    pixels, and a real person standing at the frame edge legitimately regresses a box a
    fraction of a pixel past it (seen on the parity sample: y2 = 1079.33 on a 1080-row
    frame). Clipping would move those edges, break I2 parity, and change AP. Nothing
    downstream needs in-bounds boxes: `run_pipeline._observations_from` reads centres and
    extents, and the pose stage crops through an affine warp that pads.
  * One image per `session.run`. The graph's batch axis is dynamic, so batching would
    run, but fp32 accumulation order in a batched GEMM is not the single-image one, and
    this backend's whole claim is bit-identity with the grader. The loop is billed as one
    Timing for the whole `detect` call, as in `rtmlib_backend.detect`.

Import discipline follows `reid_backend`: cv2, Pillow and onnxruntime are imported inside
functions, so importing this module costs numpy alone and a missing weights file is a
`FileNotFoundError` on a box with no inference stack rather than a `ModuleNotFoundError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vividly.models.base import Detections, ModelProvenance, Shippability, TimedModel

#: The exported graph's static spatial dims, (w, h). Both are 640 and the export fixes
#: them; a config asking for anything else is rejected in the constructor rather than
#: allowed to fail deep inside `.run()` with a shape message nobody can act on.
INPUT_SIZE: tuple[int, int] = (640, 640)

#: The graph's I/O names, checked by name and not by position at construction.
INPUT_IMAGES = "images"
INPUT_SIZES = "orig_target_sizes"
OUTPUT_NAMES: tuple[str, str, str] = ("labels", "boxes", "scores")

#: person_v2 is single-class (`num_classes: 1`), so every query this graph emits carries
#: label 0, verified against the export of record, all 300 queries on all 20 parity-sample
#: frames. The filter below is therefore a no-op today and a guard tomorrow: if a
#: multi-class graph is ever pointed at this key, non-person queries are dropped rather
#: than returned as people with a class_id the rest of the repo assumes is 0.
PERSON_LABEL = 0

#: How many queries the postprocessor emits per image, whatever the image holds
#: (`num_queries: 300`, `num_classes: 1` -> `topk(300)` returns every query). The tail runs
#: down to ~1e-4 with arbitrary box regressions, which is why `score_thr` is not optional
#: cosmetics: at 0.0 this backend would hand the tracker 300 boxes of noise per frame.
NUM_QUERIES = 300

#: Default operating point. `research/eyes-instrument-grading.md` grades this model
#: at 0.3 / 0.4 / 0.5; 0.3 is the loosest of the three and the one the precision tables
#: lead with. It is a default, not a fixed operating point; the pipeline config sets the
#: real one.
DEFAULT_SCORE_THR = 0.3


def preprocess_bgr(image: np.ndarray, input_size: tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    """One BGR frame -> the `images` tensor, (1, 3, H, W) float32 in 0..1, RGB order.

    Pure function, so the geometry and the colour order are testable with no session and
    no weights. Behaviourally identical to `scripts/eval/eval_detector_v2_onnx.py`'s
    `preprocess`, which starts from `Image.open(path).convert("RGB")` instead of a decoded
    BGR array; cv2 and Pillow decode this repo's JPEGs to byte-identical RGB, so the two
    paths agree exactly and `scripts/eval/backend_parity_rtdetr.py` asserts it.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 BGR, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(
            f"image must be uint8 BGR (base.py: 'HWC, uint8, BGR'), got dtype {image.dtype}. "
            "A float image here would silently be rescaled twice."
        )

    import cv2
    from PIL import Image

    width, height = input_size
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # PIL's own bilinear, not cv2's: see the module docstring. `Image.fromarray` copies,
    # so the caller's frame is never touched.
    resized = np.asarray(
        Image.fromarray(rgb).resize((width, height), Image.BILINEAR), dtype=np.uint8
    )
    return (resized.transpose(2, 0, 1).astype(np.float32) / 255.0)[None, ...]


def orig_target_sizes(image: np.ndarray) -> np.ndarray:
    """The `orig_target_sizes` tensor for one frame: (1, 2) int64, **[width, height]**.

    The order is fixed by the trainer, not a free choice. The trainer stores `orig_size`
    as [w, h] (the field the dataset builds in `engine/data/dataset/coco_dataset.py`) and
    the deployed postprocessor repeats it to scale xyxy, so [h, w] here would transpose
    every box on a non-square frame and still return finite, plausible-looking numbers.
    """
    height, width = image.shape[:2]
    return np.array([[int(width), int(height)]], dtype=np.int64)


class RTDETRv4Detector(TimedModel):
    """RT-DETRv4 person detector over ONNX Runtime. Implements the PersonDetector protocol.

    The provenance defaults describe the eyes model of record this backend was written for
    and are copied from the records on disk, not inferred: architecture and licence from the
    `repos.rtdetrv4` entry in `third_party/PINS.yaml` (RT-DETRv4 @ 55fefaae, Apache-2.0),
    training corpus and init lineage from the "The run" section of
    `research/eyes-rtdetrv4-training-record.md` (person_v2 over
    `configs/datasets/physicalai_smartspaces.yaml`, TRAIN_OK / CC BY 4.0, initialised from
    `rtv4_s_coco.pth`, whose lineage is ImageNet -> COCO train2017 with no Objects365, the
    `checkpoints.rtv4_s_coco` entry in `third_party/PINS.yaml`). Every one of them is an
    overridable keyword argument, because POLICY.md, rule 8 makes provenance a recorded property
    of a checkpoint and a different `.onnx` behind the same key is a different checkpoint.

    `shippable` defaults to "counsel" and no argument makes it "yes" by accident: Counsel
    item C-1 (the `repos.rtdetrv4.license_notes` entry in `third_party/PINS.yaml`) records
    that the released RT-DETRv4-S weights this run initialises from were themselves trained
    with a DINOv3 teacher. Whether that inheritance matters is a counsel question, not a
    question this code answers.

    Construction order is deliberate, and matches `reid_backend`: argument validation, then
    the file-existence check, then the onnxruntime import, then the graph-shape checks. A
    missing `.onnx` is a `FileNotFoundError` on every platform, including a CI box with no
    onnxruntime at all, where importing first would mask it as a `ModuleNotFoundError`.
    """

    def __init__(
        self,
        onnx: str | Path,
        input_size: tuple[int, int] = INPUT_SIZE,
        device: str = "cpu",
        score_thr: float = DEFAULT_SCORE_THR,
        name: str = "rtdetrv4-s-person-onnx",
        sha256: str | None = None,
        train_data: Sequence[str] = ("physicalai_smartspaces/person_v2",),
        pretrain_data: Sequence[str] = ("ImageNet-1k", "COCO train2017"),
        shippable: Shippability = "counsel",
        code_license: str = "Apache-2.0",
        notes: str = "",
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError(
                f"device={device!r} rejected. On the Mac, ONNX inference runs on the CPU "
                "execution provider: the CoreML EP crashes inside .run() on this repo's "
                "detector graphs (onnxruntime 1.28.0). Use 'cpu', or 'cuda' on a GPU box."
            )
        size = tuple(int(v) for v in input_size)
        if size != INPUT_SIZE:
            raise ValueError(
                f"input_size={input_size!r} rejected: the exported RT-DETRv4 graph has "
                f"STATIC {INPUT_SIZE[0]}x{INPUT_SIZE[1]} spatial dims. Feeding another size "
                "is a rebuild of the export, not a config change."
            )
        if not 0.0 <= float(score_thr) <= 1.0:
            raise ValueError(f"score_thr must be in [0, 1], got {score_thr!r}")
        path = Path(onnx)
        if not path.is_file():
            raise FileNotFoundError(
                f"RT-DETRv4 detector weights not found: {path}. This backend takes a LOCAL "
                "exported .onnx; unlike the rtmlib key it never downloads a zoo asset."
            )
        self.model_path = path
        self.score_thr = float(score_thr)
        if sha256 is None:
            # POLICY.md, rule 8: provenance is recorded from the loaded weights, and the tree
            # extractor writes prov.sha256 into every meta. rtmlib backends get hashed
            # externally via their `._m.onnx_model` attribute, which this class does not
            # have, so hash the file here, once, at construction (about 0.1 s for 40 MB).
            import hashlib

            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            sha256 = h.hexdigest()

        super().__init__(
            ModelProvenance(
                name=name,
                task="person_detection",
                weights_uri=str(path),
                sha256=sha256,
                train_data=tuple(train_data),
                pretrain_data=tuple(pretrain_data),
                code_license=code_license,
                shippable=shippable,
                input_size=size,  # ModelProvenance records (w, h); this graph is square
                runtime=f"onnxruntime-{device}",
                notes=notes or "provenance comes from this weight's model card (POLICY.md, rule 8)",
            )
        )

        import onnxruntime as ort

        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            available = set(ort.get_available_providers())
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "device='cuda' was requested but this onnxruntime build has no "
                    f"CUDAExecutionProvider (available: {sorted(available)}). Install "
                    "onnxruntime-gpu on the GPU box, or run with device='cpu'. Falling "
                    "back silently would report a GPU runtime in the provenance and time "
                    "the CPU."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(path), providers=providers)

        self._check_graph()

    # -- graph checks -------------------------------------------------------
    def _check_graph(self) -> None:
        """Fail at construction, not on frame one, if this is not the expected graph.

        Every check names what it wanted, because the alternative failure, a D-FINE or
        RTMDet export behind this key, produces a `.run()` error about tensor ranks that
        says nothing about which config line is wrong.
        """
        got_inputs = {i.name: i for i in self._session.get_inputs()}
        if set(got_inputs) != {INPUT_IMAGES, INPUT_SIZES}:
            raise ValueError(
                f"{self.model_path}: expected inputs {sorted((INPUT_IMAGES, INPUT_SIZES))}, "
                f"got {sorted(got_inputs)}. That is not an RT-DETRv4 deploy export."
            )
        got_outputs = [o.name for o in self._session.get_outputs()]
        if set(got_outputs) != set(OUTPUT_NAMES):
            raise ValueError(
                f"{self.model_path}: expected outputs {sorted(OUTPUT_NAMES)}, "
                f"got {sorted(got_outputs)}. That is not an RT-DETRv4 deploy export."
            )
        # (N, 3, H, W): only the batch axis may be dynamic, and H/W must match the export
        # this backend preprocesses for.
        shape = list(got_inputs[INPUT_IMAGES].shape)
        if len(shape) != 4:
            raise ValueError(
                f"{self.model_path}: '{INPUT_IMAGES}' must be (N, 3, H, W), got {shape}"
            )
        declared_h, declared_w = shape[2], shape[3]
        if (declared_w, declared_h) != INPUT_SIZE:  # INPUT_SIZE is (w, h)
            raise ValueError(
                f"{self.model_path}: '{INPUT_IMAGES}' declares {shape[2]}x{shape[3]} (h x w), "
                f"but this backend preprocesses to {INPUT_SIZE[1]}x{INPUT_SIZE[0]}. Re-export "
                "or use a backend written for that size."
            )

    # -- core ---------------------------------------------------------------
    def _forward_one(self, image: np.ndarray) -> Detections:
        blob = preprocess_bgr(image, self.provenance.input_size)
        sizes = orig_target_sizes(image)
        outs: list[Any] = self._session.run(
            list(OUTPUT_NAMES), {INPUT_IMAGES: blob, INPUT_SIZES: sizes}
        )
        labels, boxes, scores = (np.asarray(o) for o in outs)
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise NotImplementedError(
                f"unexpected RT-DETRv4 boxes shape {boxes.shape}; expected (1, N, 4) xyxy"
            )
        labels, boxes, scores = labels[0], boxes[0], scores[0]
        keep = (scores >= self.score_thr) & (labels == PERSON_LABEL)
        return Detections(
            xyxy=boxes[keep].astype(np.float32),
            scores=scores[keep].astype(np.float32),
            class_ids=np.zeros(int(keep.sum()), np.int32),
        )

    def detect(self, images: Sequence[np.ndarray]) -> list[Detections]:
        # One session.run per image on purpose (see the module docstring); one Timing for
        # the whole call, so ms_per_item stays comparable with the other detectors.
        return self._timed("detect", len(images), lambda: [self._forward_one(im) for im in images])

    def detect_one(self, image: np.ndarray) -> Detections:
        return self.detect([image])[0]

    def _warmup_once(self, image: np.ndarray) -> None:
        self._forward_one(image)
