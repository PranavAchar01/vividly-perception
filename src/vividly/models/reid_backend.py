"""ONNX Runtime appearance-embedding backend for the session-scoped re-identification
scope defined in POLICY.md, section 4.

Same runtime shape as `rtmlib_backend`: onnxruntime, `device='cpu'` by default, no torch
anywhere. That is not a stylistic echo. `torchreid` has a forbidden-import contract of its
own in `.importlinter` (torchreid-stays-out-of-vividly), and docs/LICENCE_ALLOWLIST.md
(re-ID section) puts it in the `curate` fork only, plus the by-hand install on the Linux
training pod that TRAINING.md records, so the dev-side path to an OSNet embedding is an
ONNX export consumed here, never the library.

What this module is not: it does not decide identity. It turns a person crop into a vector.
Galleries, thresholds, false-merge accounting and every other identity decision live above
it, because POLICY.md, section 4 keeps those behind human confirmation.

Preprocessing (constants below, and a property of the weights, not of this file):

    1. resize to 256x128 (HxW) with bilinear interpolation  -> cv2.resize(crop, (128, 256))
    2. BGR to RGB (channel reversal; crops arrive in OpenCV order)
    3. scale to 0..1 (divide by 255)
    4. subtract the ImageNet mean, divide by the ImageNet std, per RGB channel
    5. transpose HWC to CHW and stack into an NCHW float32 batch

That is the torchreid/OSNet recipe, and it is stated here rather than inferred because a
mismatched normalisation degrades embeddings quietly: every vector is still finite, still
unit norm, and still wrong. `preprocess_crops` is a pure function so the geometry is
testable with no session and no weights.

Two things this constructor refuses, before it opens anything:

  * a weight whose file path or declared lineage mentions LUPerson, SOLIDER or
    TransReID-SSL. POLICY.md, rule 11 makes those ship=never; the LUPerson terms are
    research only.
  * a construction that does not state `train_data` and `pretrain_data`. POLICY.md, rule 8:
    provenance is recorded, never assumed, and the model card is where it comes from.

Import discipline: cv2 and onnxruntime are darwin-marked in this repo's dev fork and absent
from linux CI, so both are imported inside functions. Importing this module costs numpy
(which `base` already needs) and nothing else, which is what lets the deny-list and the
preprocessing be tested on a box with no inference stack at all.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vividly.models.base import ModelProvenance, Shippability, TimedModel

#: Network input, (H, W). OSNet and every torchreid re-ID export use 256x128; a crop is
#: taller than it is wide because a person is.
INPUT_H: int = 256
INPUT_W: int = 128

#: ImageNet statistics in RGB channel order, applied after the 0..1 scaling.
IMAGENET_MEAN_RGB: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB: tuple[float, float, float] = (0.229, 0.224, 0.225)

_MEAN = np.asarray(IMAGENET_MEAN_RGB, np.float32)
_STD = np.asarray(IMAGENET_STD_RGB, np.float32)

#: Guards the division in `l2_normalise`. A zero row stays zero instead of becoming NaN:
#: a NaN embedding poisons every similarity it touches, a zero one simply matches nothing.
_L2_EPS = 1e-12

#: Lineage tokens that mean ship=never under POLICY.md, rule 11. The LUPerson repository
#: states that it "can only be used for research, commercial usage is forbidden"; SOLIDER
#: and TransReID-SSL both pretrain on it, as do the LUPerson-NL descendants. Matching is done
#: on a normalised string, so "TransReID-SSL", "transreid ssl" and "LUPerson-NL" all hit.
BANNED_LINEAGE_TOKENS: tuple[str, ...] = (
    "luperson",
    "solider",
    "transreidssl",
    "passreid",
    "personvit",
    "dukemtmc",
)

#: Banned for a different reason: not a lineage but a distribution channel. The
#: kadirnar/osnet Hugging Face repack is labelled GPL-3.0, and pulling from it would
#: launder that labelling into this tree whether or not the bytes are the MIT ones.
BANNED_SOURCE_TOKENS: tuple[str, ...] = ("kadirnar",)

_LINEAGE_REASON = (
    "LUPerson-derived re-ID weights (SOLIDER, TransReID-SSL, PASS-reID, PersonViT, "
    "LUPerson-NL descendants) and DukeMTMC-lineage weights are ship=never under the "
    "weight-provenance rule in POLICY.md, rule 11: the LUPerson terms are research only, "
    "commercial use forbidden, and DukeMTMC was retracted in 2019 over an IRB breach "
    "(research/reid-data-licences.md). "
    "This backend will not load them at all; if counsel ever clears a benchmark against "
    "one, it needs a quarantined path of its own, the way AGPL reference code does."
)
_SOURCE_REASON = (
    "the kadirnar/osnet repack is labelled GPL-3.0 and is banned outright, same bytes or "
    "not. OSNet weights come only from the canonical MIT torchreid distribution "
    "(KaiyangZhou/deep-person-reid), ImageNet-pretrained variant preferred."
)


def _normalise_lineage(text: str) -> str:
    """Lowercase and strip every non-alphanumeric, so separator games cannot dodge a token.

    Removal beats collapsing-to-underscore: `LU-Person`, `lu person`, `TransReIDSSL`, and
    `transreid_ssl` all reduce to the same separator-free string, which is why the tokens
    above are spelled separator-free too.
    """
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def assert_lineage_allowed(fields: Sequence[tuple[str, str]]) -> None:
    """Raise ValueError if any (field_name, text) pair names a weight forbidden by
    POLICY.md, rule 11.

    Checked against the file path as well as the declared provenance, because a weight
    living in `weights/luperson_pretrained/` is the same weight whatever the caller types
    into `train_data`. This is the model-layer half of that enforcement;
    `scripts/reid/export_osnet_onnx.py` refuses the same tokens at the point weights become an
    ONNX file, and neither is a provenance system: an innocently named file with a LUPerson
    lineage passes both and is still forbidden. The model card is the record.
    """
    checks = (
        (BANNED_LINEAGE_TOKENS, _LINEAGE_REASON),
        (BANNED_SOURCE_TOKENS, _SOURCE_REASON),
    )
    for field, text in fields:
        haystack = _normalise_lineage(text)
        for tokens, reason in checks:
            for token in tokens:
                if token in haystack:
                    raise ValueError(
                        f"refusing {token!r} in {field}={text!r}. {reason} "
                        "See docs/LICENCE_ALLOWLIST.md, re-ID section, and POLICY.md, rule 11."
                    )


def _check_crop(crop: Any, index: int) -> None:
    """Reject anything that is not an HxWx3 uint8 BGR crop, naming the offending index."""
    if not isinstance(crop, np.ndarray):
        raise TypeError(f"crops[{index}] must be a numpy array, got {type(crop).__name__}")
    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError(f"crops[{index}] must be HxWx3 BGR, got shape {crop.shape}")
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        raise ValueError(f"crops[{index}] has no pixels: shape {crop.shape}")
    if crop.dtype != np.uint8:
        raise ValueError(
            f"crops[{index}] must be uint8 BGR in 0..255, got dtype {crop.dtype}. "
            "Scaling to 0..1 happens inside preprocess_crops; doing it twice halves "
            "every embedding without raising anything."
        )


def preprocess_crops(crops: Sequence[np.ndarray]) -> np.ndarray:
    """N BGR uint8 crops -> one (N, 3, 256, 128) float32 NCHW batch. Pure function.

    Steps 1-5 of the recipe in the module docstring, in that order. An empty sequence
    returns an empty batch of the right rank rather than raising, so an idle frame is not
    a special case for the caller.
    """
    import cv2

    batch = np.empty((len(crops), 3, INPUT_H, INPUT_W), np.float32)
    for i, crop in enumerate(crops):
        _check_crop(crop, i)
        resized = cv2.resize(crop, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        rgb -= _MEAN
        rgb /= _STD
        batch[i] = rgb.transpose(2, 0, 1)
    return batch


def l2_normalise(features: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation of an (N, D) float32 matrix.

    Applied by this backend rather than baked into the export, because most re-ID exports
    emit raw features and normalising an already-normalised row is a no-op. A row whose
    norm is zero is returned unchanged (still zero) instead of dividing to NaN.
    """
    feats = np.asarray(features, np.float32)
    if feats.ndim != 2:
        raise ValueError(f"features must be (N, D), got shape {feats.shape}")
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return (feats / np.maximum(norms, _L2_EPS)).astype(np.float32)


def static_dim(shape: Sequence[Any], axis: int) -> int | None:
    """The declared size of `shape[axis]`, or None when the graph leaves it dynamic.

    onnxruntime reports a dimension as an int when it is fixed, and as a `dim_param`
    string (or None) when the export made it dynamic.
    """
    if axis >= len(shape):
        return None
    dim = shape[axis]
    return dim if isinstance(dim, int) and dim > 0 else None


def check_declared_input_shape(shape: Sequence[Any], where: str) -> None:
    """Raise ValueError when a graph cannot be fed the batch `preprocess_crops` builds.

    Dimensions the export left dynamic are accepted: a graph with a symbolic height takes
    256 happily. A dimension declared as a different fixed number is a mismatch that would
    otherwise surface as an onnxruntime shape error on the first real frame.
    """
    if len(shape) != 4:
        raise ValueError(
            f"{where}: expected a 4-D NCHW input, got a declared shape of {list(shape)}"
        )
    for axis, want, label in ((1, 3, "channels"), (2, INPUT_H, "height"), (3, INPUT_W, "width")):
        got = static_dim(shape, axis)
        if got is not None and got != want:
            raise ValueError(
                f"{where}: graph declares {label}={got}, but this backend feeds "
                f"{INPUT_H}x{INPUT_W} (HxW) 3-channel crops. Preprocessing is a property of "
                "the weights; re-export the model or add a backend for its recipe."
            )


def embedding_dim_from_output_shape(shape: Sequence[Any], where: str) -> int:
    """D from a declared (N, D) output shape. A dynamic D is an error, not a guess.

    A gallery has to be allocated before the first crop arrives, so the feature width has
    to be knowable from the graph. Batch may be dynamic; the feature dimension may not.
    """
    if len(shape) != 2:
        raise ValueError(
            f"{where}: expected a 2-D (N, D) embedding output, got a declared shape of "
            f"{list(shape)}. Pool and flatten inside the export, not out here."
        )
    dim = static_dim(shape, 1)
    if dim is None:
        raise ValueError(
            f"{where}: the embedding dimension is declared as {shape[1]!r} (dynamic). "
            "Export with a fixed feature width; only the batch axis may be dynamic."
        )
    return dim


class OnnxAppearanceEmbedder(TimedModel):
    """Appearance embeddings over ONNX Runtime. Implements the AppearanceEmbedder protocol.

    `train_data` and `pretrain_data` are required keyword arguments with no defaults on
    purpose. POLICY.md, rule 8 says provenance is recorded and never assumed, and for re-ID
    weights the pretraining lineage is the licence question, per POLICY.md, rule 11: a card
    that cannot state it does not get written and the checkpoint does not get used.
    `shippable` defaults to "counsel" for the same reason it does everywhere else in this
    repository, which is that shippability is not resolved in this repository and no
    checkpoint on disk is "yes" today.

    Construction order is deliberate and tested: deny-list, then plain argument validation,
    then the file-existence check, then the onnxruntime import. A missing weights file is
    a FileNotFoundError on every platform, including the CI fork with no onnxruntime, where
    importing first would mask it as ModuleNotFoundError. `sources/replay.py` orders its
    cv2 checks the same way and for the same reason.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        train_data: Sequence[str],
        pretrain_data: Sequence[str],
        shippable: Shippability = "counsel",
        name: str = "reid-embedder-onnx",
        sha256: str | None = None,
        code_license: str = "unknown",
        device: str = "cpu",
        batch_size: int | None = None,
        notes: str = "",
    ) -> None:
        train = tuple(train_data)
        pretrain = tuple(pretrain_data)
        path = Path(model_path)
        assert_lineage_allowed(
            [
                ("model_path", str(model_path)),
                # Again on the resolved path: a clean-looking symlink into a luperson/ tree
                # is exactly what a check on the argument alone waves through.
                ("resolved model_path", str(path.resolve())),
                ("name", name),
                ("train_data", " ".join(train)),
                ("pretrain_data", " ".join(pretrain)),
                ("notes", notes),
            ]
        )
        if device not in {"cpu", "cuda"}:
            raise ValueError(
                f"device={device!r} rejected. On the Mac, ONNX inference runs on the CPU "
                "execution provider: the CoreML EP crashes inside .run() on this repo's "
                "graphs (onnxruntime 1.28.0). Use 'cpu', or 'cuda' on a GPU box."
            )
        if batch_size is not None and batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1 or None (one call per batch), got {batch_size}"
            )
        if not path.exists():
            raise FileNotFoundError(f"appearance embedder weights not found: {path}")
        self.model_path = path

        super().__init__(
            ModelProvenance(
                name=name,
                task="appearance_embedding",
                weights_uri=str(path),
                sha256=sha256,
                train_data=train,
                pretrain_data=pretrain,
                code_license=code_license,
                shippable=shippable,
                input_size=(INPUT_W, INPUT_H),  # ModelProvenance records (w, h)
                runtime=f"onnxruntime-{device}",
                notes=notes or "provenance comes from this weight's model card (POLICY.md, rule 8)",
            )
        )

        import onnxruntime as ort

        providers = (
            ["CPUExecutionProvider"]
            if device == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(path), providers=providers)

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(
                f"{path}: expected one graph input, got {[i.name for i in inputs]}. An "
                "embedder takes a crop batch and nothing else."
            )
        if len(outputs) != 1:
            raise ValueError(
                f"{path}: expected one graph output, got {[o.name for o in outputs]}. A "
                "re-ID model exported in training mode emits classifier logits next to the "
                "features, and picking the first of two here would embed the wrong tensor. "
                "Export in eval mode, features only."
            )
        check_declared_input_shape(inputs[0].shape, str(path))
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._dim = embedding_dim_from_output_shape(outputs[0].shape, str(path))
        # A graph exported with a fixed batch axis dictates the chunk size; an explicit
        # batch_size overrides it (and caps peak memory on a large frame).
        self._batch_size = batch_size if batch_size is not None else static_dim(inputs[0].shape, 0)

    # -- protocol -----------------------------------------------------------
    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """(N, D) float32, L2-normalised, row i from crops[i]. See AppearanceEmbedder."""
        return self._timed("embed", max(len(crops), 1), lambda: self._forward(crops))

    # -- internals ----------------------------------------------------------
    def _forward(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros((0, self._dim), np.float32)
        feats = self._run(preprocess_crops(crops))
        if feats.shape != (len(crops), self._dim):
            raise ValueError(
                f"{self.model_path}: graph returned {feats.shape} for {len(crops)} crops, "
                f"expected {(len(crops), self._dim)}"
            )
        return l2_normalise(feats)

    def _run(self, batch: np.ndarray) -> np.ndarray:
        """Run the session, in chunks of `self._batch_size` when one is in force.

        The final chunk is padded up to the chunk size by repeating its last crop and the
        padding rows are sliced off again. Padding costs one duplicate forward at most and
        is what lets a fixed-batch export (the common shape of a torchreid ONNX dump) take
        a frame with any number of people in it.
        """
        if self._batch_size is None:
            return np.asarray(
                self._session.run([self._output_name], {self._input_name: batch})[0], np.float32
            )
        out: list[np.ndarray] = []
        for start in range(0, len(batch), self._batch_size):
            chunk = batch[start : start + self._batch_size]
            kept = len(chunk)
            if kept < self._batch_size:
                pad = np.repeat(chunk[-1:], self._batch_size - kept, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            got = self._session.run([self._output_name], {self._input_name: chunk})[0]
            out.append(np.asarray(got, np.float32)[:kept])
        return np.concatenate(out, axis=0)

    def _warmup_once(self, image: np.ndarray) -> None:
        """`shape` in TimedModel.warmup is a CROP size here, not a frame size; any size
        works, since preprocessing resizes everything to 256x128."""
        self._forward([image])
