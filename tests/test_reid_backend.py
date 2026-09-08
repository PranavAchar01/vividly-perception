"""AppearanceEmbedder protocol and the ONNX backend behind it.

Three groups of claims, and they are deliberately separable:

  * things that must hold with no inference stack at all: the LUPerson deny-list, the
    provenance discipline, the graph-contract checks, and the ordering that makes a missing
    weights file a FileNotFoundError rather than a ModuleNotFoundError. A Linux environment
    without the linux-harness group has no onnxruntime and no cv2, and these are the tests
    that still run there.
  * preprocessing geometry, which needs cv2 and no weights. Checked against hand-computed
    numbers, because a wrong normalisation produces embeddings that are finite, unit-norm
    and silently useless.
  * the full path over `tests/fixtures/reid_synth/tiny_embedder.onnx`, a 1.3 KB graph whose
    weights are arithmetic. Its zero-padded convolution makes a flat-colour crop's
    embedding hand-computable, so even the end-to-end assertions are statements about the
    recipe rather than comparisons against a recorded blob.

No weights are fetched, no host is contacted, and nothing in here is derived from any
dataset.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vividly.models import factory
from vividly.models.base import AppearanceEmbedder, ModelProvenance, Timing
from vividly.models.reid_backend import (
    BANNED_LINEAGE_TOKENS,
    BANNED_SOURCE_TOKENS,
    IMAGENET_MEAN_RGB,
    IMAGENET_STD_RGB,
    INPUT_H,
    INPUT_W,
    OnnxAppearanceEmbedder,
    check_declared_input_shape,
    embedding_dim_from_output_shape,
    l2_normalise,
    preprocess_crops,
    static_dim,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "reid_synth"
MODEL = FIXTURE / "tiny_embedder.onnx"

#: Provenance a caller must state to construct anything. The fixture is a synthetic graph,
#: so "synthetic" is the literal truth rather than a placeholder.
SYNTH_PROVENANCE: dict[str, Any] = {
    "train_data": ("synthetic",),
    "pretrain_data": ("none",),
    "name": "tiny-synth-embedder",
}


# ------------------------------------------------------------------ helpers
def _generator():
    """The committed generate.py, imported without executing anything that needs onnx."""
    spec = importlib.util.spec_from_file_location("reid_synth_generate", FIXTURE / "generate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _solid(bgr: tuple[int, int, int], h: int = 64, w: int = 32) -> np.ndarray:
    """One flat-colour BGR crop. Flat on purpose: see the fixture README."""
    crop = np.empty((h, w, 3), np.uint8)
    crop[:, :] = bgr
    return crop


def _normalised_channels(bgr: tuple[int, int, int]) -> list[float]:
    """The preprocessed RGB values of a flat BGR crop, computed by hand in float64."""
    rgb = (bgr[2], bgr[1], bgr[0])
    return [
        (value / 255.0 - mean) / std
        for value, mean, std in zip(rgb, IMAGENET_MEAN_RGB, IMAGENET_STD_RGB, strict=True)
    ]


@pytest.fixture
def embedder() -> OnnxAppearanceEmbedder:
    pytest.importorskip("onnxruntime", reason="onnxruntime is darwin-marked in the dev fork")
    pytest.importorskip("cv2", reason="cv2 is darwin-marked in the dev fork")
    return OnnxAppearanceEmbedder(MODEL, **SYNTH_PROVENANCE)


# =================================================== protocol conformance
class _StubEmbedder:
    """A conforming implementation that touches no inference stack, so the protocol itself
    is testable on a box with neither onnxruntime nor cv2."""

    def __init__(self) -> None:
        self.provenance = ModelProvenance(
            name="stub",
            task="appearance_embedding",
            weights_uri="memory://stub",
            sha256=None,
            train_data=("none",),
            pretrain_data=("none",),
            code_license="Apache-2.0",
            shippable="counsel",
            input_size=(INPUT_W, INPUT_H),
            runtime="stub",
        )

    def embed(self, crops):
        return np.zeros((len(crops), 4), np.float32)

    @property
    def embedding_dim(self) -> int:
        return 4

    def warmup(self, shape=(720, 1280), n=2) -> None:
        return None

    @property
    def timings(self) -> list[Timing]:
        return []

    def reset_timings(self) -> None:
        return None


def test_a_conforming_stub_is_an_appearance_embedder() -> None:
    assert isinstance(_StubEmbedder(), AppearanceEmbedder)


def test_carrying_provenance_and_timings_is_not_enough_to_be_an_embedder() -> None:
    """The shape every TimedModel in this repo already has, minus `embed`. Without this the
    conformance test above would pass for a detector, a tracker, or anything else here."""

    class _NotAnEmbedder:
        provenance = _StubEmbedder().provenance

        @property
        def embedding_dim(self) -> int:
            return 4

        @property
        def timings(self) -> list[Timing]:
            return []

        def reset_timings(self) -> None:
            return None

    assert not isinstance(_NotAnEmbedder(), AppearanceEmbedder)


def test_the_onnx_backend_satisfies_the_protocol(embedder: OnnxAppearanceEmbedder) -> None:
    assert isinstance(embedder, AppearanceEmbedder)
    assert embedder.provenance.task == "appearance_embedding"


def test_the_embedding_dimension_is_known_before_the_first_call(
    embedder: OnnxAppearanceEmbedder,
) -> None:
    """A gallery is allocated before a crop ever arrives, so D comes from the graph."""
    assert embedder.embedding_dim == _generator().EMBED_DIM == 8
    assert embedder.embed([_solid((10, 20, 30))]).shape[1] == embedder.embedding_dim


# =============================================== output shape and L2 norm
def test_embeddings_are_unit_norm_and_index_aligned(embedder: OnnxAppearanceEmbedder) -> None:
    crops = [_solid(c) for c in ((0, 0, 220), (0, 200, 0), (220, 0, 0), (255, 255, 255))]
    out = embedder.embed(crops)

    assert out.shape == (4, 8)
    assert out.dtype == np.float32
    assert np.linalg.norm(out, axis=1) == pytest.approx(1.0, rel=1e-5)
    # Four different colours must not collapse onto one vector, or every later test that
    # says "these two crops matched" would be vacuous.
    assert len({tuple(np.round(row, 5)) for row in out}) == 4


def test_a_batch_of_zero_returns_a_zero_by_d_matrix(embedder: OnnxAppearanceEmbedder) -> None:
    """Not a bare (0,) array: an idle frame still has to multiply against a gallery."""
    out = embedder.embed([])
    assert out.shape == (0, embedder.embedding_dim)
    assert out.dtype == np.float32
    assert (out @ np.zeros((8, 3), np.float32)).shape == (0, 3)


def test_a_crop_embeds_the_same_alone_as_it_does_inside_a_batch(
    embedder: OnnxAppearanceEmbedder,
) -> None:
    solo = embedder.embed([_solid((0, 200, 0))])
    batch = embedder.embed([_solid((7, 7, 7)), _solid((0, 200, 0)), _solid((255, 0, 0))])
    assert batch[1] == pytest.approx(solo[0], rel=1e-6)


def test_repeated_calls_are_bit_identical(embedder: OnnxAppearanceEmbedder) -> None:
    crops = [_solid((12, 34, 56)), _solid((200, 100, 50))]
    first, second = embedder.embed(crops), embedder.embed(crops)
    assert np.array_equal(first, second)


def test_l2_normalise_leaves_a_zero_row_at_zero_rather_than_nan() -> None:
    out = l2_normalise(np.array([[0.0, 0.0], [3.0, 4.0]], np.float32))
    assert not np.isnan(out).any()
    assert out[0].tolist() == [0.0, 0.0]
    assert out[1] == pytest.approx([0.6, 0.8])


def test_l2_normalise_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match=r"must be \(N, D\)"):
        l2_normalise(np.zeros((3,), np.float32))


# ==================================================== preprocessing geometry
def test_preprocessing_produces_the_documented_nchw_batch() -> None:
    pytest.importorskip("cv2")
    batch = preprocess_crops([_solid((0, 0, 0), 10, 400), _solid((1, 2, 3), 300, 7)])
    assert batch.shape == (2, 3, INPUT_H, INPUT_W) == (2, 3, 256, 128)
    assert batch.dtype == np.float32


def test_an_empty_crop_list_preprocesses_to_an_empty_batch() -> None:
    pytest.importorskip("cv2")
    assert preprocess_crops([]).shape == (0, 3, 256, 128)


def test_preprocessing_maps_every_pixel_exactly_when_no_resize_is_needed() -> None:
    """A crop already at 256x128 goes through bilinear resize unchanged (scale 1 maps
    sample centres onto themselves), so every output value is checkable against the recipe
    in closed form. This is the test that pins channel order and axis order at once: swap
    BGR for RGB, or H for W, and the comparison fails on most pixels."""
    pytest.importorskip("cv2")
    ys, xs = np.mgrid[0:INPUT_H, 0:INPUT_W]
    crop = np.stack(
        [
            (ys % 256).astype(np.uint8),
            ((xs * 2) % 256).astype(np.uint8),
            ((ys + xs) % 256).astype(np.uint8),
        ],
        axis=2,
    )  # channel 0 is blue, as OpenCV orders it

    batch = preprocess_crops([crop])

    expected = np.empty((3, INPUT_H, INPUT_W), np.float64)
    for out_c, in_c in ((0, 2), (1, 1), (2, 0)):  # R, G, B out of a B, G, R crop
        scaled = crop[:, :, in_c].astype(np.float64) / 255.0
        expected[out_c] = (scaled - IMAGENET_MEAN_RGB[out_c]) / IMAGENET_STD_RGB[out_c]
    assert batch[0] == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_one_hand_computed_pixel_pins_the_normalisation_constants() -> None:
    """BGR (0, 128, 255) is RGB (255, 128, 0). By hand:
    R: 255/255 = 1.0            -> (1.0        - 0.485) / 0.229 =  2.24890830
    G: 128/255 = 0.501960784... -> (0.50196078 - 0.456) / 0.224 =  0.20518207
    B:   0/255 = 0.0            -> (0.0        - 0.406) / 0.225 = -1.80444444
    """
    pytest.importorskip("cv2")
    batch = preprocess_crops([_solid((0, 128, 255))])
    got = [float(batch[0, c, 100, 40]) for c in range(3)]
    assert got == pytest.approx([2.24890830, 0.20518207, -1.80444444], rel=1e-6)
    assert got == pytest.approx(_normalised_channels((0, 128, 255)), rel=1e-6)


def test_resizing_does_not_transpose_height_and_width() -> None:
    """A 64x32 crop with a white top half must stay white on top after the resize to
    256x128. If the cv2.resize argument order were (INPUT_H, INPUT_W) the pattern would
    come out split down the middle instead, and every embedding would still look fine."""
    pytest.importorskip("cv2")
    crop = np.zeros((64, 32, 3), np.uint8)
    crop[:32] = 255

    batch = preprocess_crops([crop])
    white = _normalised_channels((255, 255, 255))
    black = _normalised_channels((0, 0, 0))

    for c in range(3):
        assert batch[0, c, :120, :] == pytest.approx(white[c], rel=1e-5)
        assert batch[0, c, 136:, :] == pytest.approx(black[c], rel=1e-5)


@pytest.mark.parametrize(
    ("crop", "match"),
    [
        (np.zeros((8, 8), np.uint8), "must be HxWx3"),
        (np.zeros((8, 8, 4), np.uint8), "must be HxWx3"),
        (np.zeros((0, 8, 3), np.uint8), "no pixels"),
        (np.zeros((8, 8, 3), np.float32), "must be uint8"),
    ],
)
def test_preprocessing_rejects_a_crop_that_is_not_hwc_bgr_uint8(
    crop: np.ndarray, match: str
) -> None:
    pytest.importorskip("cv2")
    with pytest.raises(ValueError, match=match) as exc:
        preprocess_crops([_solid((1, 1, 1)), crop])
    assert "crops[1]" in str(exc.value), "the message must name the offending index"


def test_preprocessing_rejects_a_non_array() -> None:
    pytest.importorskip("cv2")
    with pytest.raises(TypeError, match="must be a numpy array"):
        preprocess_crops([[[1, 2, 3]]])  # type: ignore[list-item]


# ============================================ hand-computed end to end
def test_a_flat_colour_crop_embeds_to_the_hand_computed_vector(
    embedder: OnnxAppearanceEmbedder,
) -> None:
    """The fixture's convolution is unpadded, so every window of a flat crop is identical
    and the whole graph collapses to `sum_c x_c * kernel_sum[o][c] + bias[o]`, then L2.
    Computing that from the generator's own weight recipe checks preprocessing, channel
    order, inference and normalisation in one assertion, without a recorded blob.

    The tolerance is set by float32 accumulation order: the graph sums 256x128 windows
    and the execution provider's convolution kernels differ between arm64 and x86, which
    moved one element by 1.3e-4 relative on a Linux runner. A channel swap or a wrong
    normalisation constant moves the vector by orders of magnitude more than that.
    """
    generate = _generator()
    sums, bias = generate.kernel_sums(), generate.conv_bias_values()
    colour = (32, 96, 200)
    x = _normalised_channels(colour)

    raw = [sum(x[c] * sums[o][c] for c in range(3)) + bias[o] for o in range(generate.EMBED_DIM)]
    norm = math.sqrt(sum(v * v for v in raw))
    expected = [v / norm for v in raw]

    got = embedder.embed([_solid(colour)])[0]
    assert got == pytest.approx(expected, rel=1e-3, abs=1e-5)


def test_a_black_crop_still_produces_a_finite_unit_vector(
    embedder: OnnxAppearanceEmbedder,
) -> None:
    """The degenerate input. It must not divide by zero, and it must not be NaN."""
    out = embedder.embed([_solid((0, 0, 0))])
    assert np.isfinite(out).all()
    assert float(np.linalg.norm(out[0])) == pytest.approx(1.0, rel=1e-5)


# ============================== the LUPerson deny-list (the weight-source rules)
@pytest.mark.parametrize(
    "filename",
    [
        "luperson_osnet.onnx",
        "LUPerson-NL_r50.onnx",
        "solider_swin_base.onnx",
        "SOLIDER.onnx",
        "transreid_ssl_vit.onnx",
        "TransReID-SSL-vit-base.onnx",
        "transreid ssl.onnx",
        "TransReIDSSL_vit.onnx",
        "LU-Person_r50.onnx",
        "lu person weights.onnx",
    ],
)
def test_a_banned_lineage_in_the_filename_is_refused(tmp_path: Path, filename: str) -> None:
    """Clean declared provenance on purpose: the filename alone has to be enough, because a
    caller who mislabels the lineage is the case this check exists for."""
    path = tmp_path / filename
    path.write_bytes(b"not a real graph")
    with pytest.raises(ValueError) as exc:
        OnnxAppearanceEmbedder(path, train_data=("COCO",), pretrain_data=("ImageNet-1k",))
    assert "model_path" in str(exc.value)
    assert "docs/LICENCE_ALLOWLIST.md" in str(exc.value)


@pytest.mark.parametrize(
    "field",
    ["train_data", "pretrain_data", "notes", "name"],
)
def test_a_banned_lineage_in_the_declared_provenance_is_refused(field: str) -> None:
    kwargs: dict[str, Any] = {
        "train_data": ("COCO",),
        "pretrain_data": ("ImageNet-1k",),
        "name": "osnet-x1-0-onnx",
        "notes": "",
    }
    kwargs[field] = ("LUPerson",) if field.endswith("_data") else "pretrained on LUPerson"
    with pytest.raises(ValueError, match="luperson"):
        OnnxAppearanceEmbedder(MODEL, **kwargs)


def test_a_banned_weight_is_refused_before_anything_is_opened(tmp_path: Path) -> None:
    """The deny-list runs first, so a LUPerson weight is refused whether or not the file
    exists. Otherwise a typo'd path would report FileNotFoundError and hide the real
    problem, and a fixed path would then load it."""
    with pytest.raises(ValueError, match="luperson"):
        OnnxAppearanceEmbedder(
            tmp_path / "nowhere" / "luperson.onnx",
            train_data=("unknown",),
            pretrain_data=("unknown",),
        )


def test_the_refusal_cites_the_policy_and_the_document(tmp_path: Path) -> None:
    path = tmp_path / "solider_base.onnx"
    path.write_bytes(b"x")
    with pytest.raises(ValueError) as exc:
        OnnxAppearanceEmbedder(path, train_data=("unknown",), pretrain_data=("unknown",))
    message = str(exc.value)
    assert "docs/LICENCE_ALLOWLIST.md" in message
    assert "ship=never" in message
    assert "research only" in message


def test_the_deny_list_holds_exactly_the_named_lineages() -> None:
    """Separator-free spellings on purpose: matching squashes separators out of the
    haystack, so `TransReID-SSL`, `transreid_ssl`, and `TransReIDSSL` all hit.
    2026-09-03 additions from research/reid-data-licences.md: the other LUPerson-lineage
    weights (PASS-reID, PersonViT) and the retracted DukeMTMC lineage."""
    assert BANNED_LINEAGE_TOKENS == (
        "luperson",
        "solider",
        "transreidssl",
        "passreid",
        "personvit",
        "dukemtmc",
    )


def test_the_gpl_labelled_repack_is_refused_too(tmp_path: Path) -> None:
    """A different rule from the lineage one: kadirnar/osnet may hold the MIT bytes, but it
    is labelled GPL-3.0 and the weight-source rules ban it outright.
    scripts/reid/export_osnet_onnx.py refuses the same token, and the union of both tuples
    here is that script's DENIED_LINEAGE_TOKENS."""
    assert BANNED_SOURCE_TOKENS == ("kadirnar",)
    path = tmp_path / "kadirnar_osnet_x1_0.onnx"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match=re.escape("GPL-3.0")):
        OnnxAppearanceEmbedder(path, train_data=("unknown",), pretrain_data=("unknown",))


def test_a_symlink_out_of_a_banned_tree_is_refused(tmp_path: Path) -> None:
    """The case a name check on the argument alone waves through: an innocent-looking link
    whose target lives in a luperson/ directory."""
    banned_dir = tmp_path / "luperson_pretrained"
    banned_dir.mkdir()
    real = banned_dir / "embedder.onnx"
    real.write_bytes(MODEL.read_bytes())
    link = tmp_path / "clean_name.onnx"
    link.symlink_to(real)

    with pytest.raises(ValueError, match="resolved model_path"):
        OnnxAppearanceEmbedder(link, train_data=("unknown",), pretrain_data=("unknown",))


def test_a_clean_lineage_is_not_refused(embedder: OnnxAppearanceEmbedder) -> None:
    """The deny-list must not be a substring trap: OSNet-from-torchreid is the one lineage
    the weight-source rules explicitly allow, and it has to construct."""
    assert embedder.provenance.name == "tiny-synth-embedder"
    ok = OnnxAppearanceEmbedder(
        MODEL,
        name="osnet_x1_0_imagenet-onnx",
        train_data=("none",),
        pretrain_data=("ImageNet-1k",),
        code_license="MIT",
    )
    assert ok.provenance.pretrain_data == ("ImageNet-1k",)


# ================================================== provenance and shippability
def test_shippable_defaults_to_counsel_and_assert_shippable_raises(
    embedder: OnnxAppearanceEmbedder,
) -> None:
    assert embedder.provenance.shippable == "counsel"
    with pytest.raises(RuntimeError, match="shippable='counsel'"):
        embedder.provenance.assert_shippable()


def test_a_no_checkpoint_raises_and_names_its_lineage() -> None:
    pytest.importorskip("onnxruntime")
    model = OnnxAppearanceEmbedder(
        MODEL,
        train_data=("Market-1501",),
        pretrain_data=("ImageNet-1k",),
        shippable="no",
        name="market-osnet-onnx",
    )
    with pytest.raises(RuntimeError) as exc:
        model.provenance.assert_shippable()
    assert "Market-1501" in str(exc.value)
    assert "eval reference numbers only" in str(exc.value)
    assert "ship=no" in model.provenance.label()


def test_provenance_records_the_input_size_and_runtime(embedder: OnnxAppearanceEmbedder) -> None:
    assert embedder.provenance.input_size == (INPUT_W, INPUT_H)  # (w, h), as the field says
    assert embedder.provenance.runtime == "onnxruntime-cpu"
    assert embedder.provenance.weights_uri == str(MODEL)


def test_train_data_and_pretrain_data_have_no_defaults() -> None:
    """POLICY.md, rule 11 made structural: you cannot construct one of these without stating the
    lineage, because for a re-ID weight the lineage is the licence question."""
    with pytest.raises(TypeError, match="train_data"):
        OnnxAppearanceEmbedder(MODEL)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="pretrain_data"):
        OnnxAppearanceEmbedder(MODEL, train_data=("COCO",))  # type: ignore[call-arg]


# ============================================ construction order and arguments
def test_a_missing_model_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="appearance embedder weights not found"):
        OnnxAppearanceEmbedder(
            tmp_path / "absent.onnx", train_data=("COCO",), pretrain_data=("ImageNet-1k",)
        )


def test_the_existence_check_runs_before_onnxruntime_is_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ReplaySource.open() orders its cv2 check the same way and for the same reason: on a
    Linux install without the harness group there is no onnxruntime, and importing first
    would turn a plain typo into a ModuleNotFoundError about a package the caller never
    mentioned."""
    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any):
        if name.split(".")[0] == "onnxruntime":
            raise AssertionError("onnxruntime was imported before the file existence check")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(FileNotFoundError):
        OnnxAppearanceEmbedder(
            tmp_path / "absent.onnx", train_data=("COCO",), pretrain_data=("ImageNet-1k",)
        )


@pytest.mark.parametrize("device", ["mps", "coreml", "MPS", ""])
def test_a_device_other_than_cpu_or_cuda_is_rejected(device: str) -> None:
    with pytest.raises(ValueError, match="rejected"):
        OnnxAppearanceEmbedder(MODEL, train_data=("COCO",), pretrain_data=("none",), device=device)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_non_positive_batch_size_is_rejected(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        OnnxAppearanceEmbedder(
            MODEL, train_data=("COCO",), pretrain_data=("none",), batch_size=batch_size
        )


def test_chunked_batches_match_one_call(embedder: OnnxAppearanceEmbedder) -> None:
    """A fixed-batch export is the common shape of a torchreid ONNX dump, so the chunk loop
    pads its short tail and slices the padding off again. Five crops through chunks of two
    exercises exactly that path, and must equal the single-call answer."""
    crops = [_solid((v, 255 - v, (v * 7) % 256)) for v in (0, 40, 90, 160, 250)]
    chunked = OnnxAppearanceEmbedder(MODEL, batch_size=2, **SYNTH_PROVENANCE)
    assert chunked.embed(crops) == pytest.approx(embedder.embed(crops), rel=1e-6)


# ================================================ graph-contract helpers (no session)
@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 3, 256, 128), 1),
        (("batch", 3, 256, 128), None),
        ((None, 3, 256, 128), None),
        ((0, 3, 256, 128), None),
    ],
)
def test_static_dim_reads_only_a_fixed_positive_dimension(
    shape: tuple[Any, ...], expected: int | None
) -> None:
    assert static_dim(shape, 0) == expected
    assert static_dim(shape, 9) is None


@pytest.mark.parametrize(
    "shape",
    [
        ("batch", 3, 256, 128),
        (8, 3, 256, 128),
        ("batch", 3, "height", "width"),
        ("batch", None, 256, 128),
    ],
)
def test_a_graph_that_can_take_our_batch_is_accepted(shape: tuple[Any, ...]) -> None:
    check_declared_input_shape(shape, "fixture")


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        (("batch", 3, 128, 64), "declares height=128"),
        (("batch", 3, 256, 256), "declares width=256"),
        (("batch", 1, 256, 128), "declares channels=1"),
        (("batch", 3, 256), "4-D NCHW"),
        (("batch", 3, 256, 128, 1), "4-D NCHW"),
    ],
)
def test_a_graph_expecting_a_different_recipe_is_refused(
    shape: tuple[Any, ...], match: str
) -> None:
    """Preprocessing is a property of the weights. A 128x64 export fed 256x128 crops is a
    silent accuracy loss on any runtime that resizes for you, so it is refused here."""
    with pytest.raises(ValueError, match=match):
        check_declared_input_shape(shape, "fixture")


def test_the_embedding_width_must_be_declared() -> None:
    assert embedding_dim_from_output_shape(("batch", 512), "fixture") == 512
    with pytest.raises(ValueError, match="dynamic"):
        embedding_dim_from_output_shape(("batch", "features"), "fixture")
    with pytest.raises(ValueError, match=r"2-D \(N, D\)"):
        embedding_dim_from_output_shape(("batch", 8, 1, 1), "fixture")


# ============================================================ timing discipline
def test_every_call_records_an_embed_timing(embedder: OnnxAppearanceEmbedder) -> None:
    embedder.embed([_solid((1, 2, 3)), _solid((4, 5, 6))])
    embedder.embed([])

    stages = [t.stage for t in embedder.timings]
    assert stages == ["embed", "embed"]
    assert [t.n_items for t in embedder.timings] == [2, 1]  # empty still bills one item
    assert all(t.backend == "onnxruntime-cpu" and t.ms >= 0.0 for t in embedder.timings)

    embedder.reset_timings()
    assert embedder.timings == []


def test_warmup_runs_the_graph_and_discards_its_own_timings(
    embedder: OnnxAppearanceEmbedder,
) -> None:
    embedder.warmup(shape=(120, 60), n=2)
    assert embedder.timings == []


# ============================================================ factory registration
def test_the_embedder_table_is_closed() -> None:
    assert sorted(factory.EMBEDDERS) == ["onnx"]
    with pytest.raises(KeyError) as exc:
        factory.build_embedder({"backend": "torchreid", "model_path": "x.onnx"})
    assert "['onnx']" in str(exc.value)
    assert "factory.py" in str(exc.value)


def test_no_embedder_is_a_valid_configuration() -> None:
    assert factory.build_embedder(None) is None
    assert factory.build_embedder({}) is None


def test_build_embedder_routes_to_the_onnx_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setitem(factory.EMBEDDERS, "onnx", lambda **kw: calls.append(kw) or "built")

    out = factory.build_embedder(
        {
            "backend": "onnx",
            "model_path": "weights/osnet.onnx",
            "train_data": ["none"],
            "pretrain_data": ["ImageNet-1k"],
        }
    )

    assert out == "built"
    assert calls == [
        {
            "model_path": "weights/osnet.onnx",
            "train_data": ["none"],
            "pretrain_data": ["ImageNet-1k"],
        }
    ], "the selector key must not reach the constructor"


def test_a_yaml_shaped_spec_binds_to_the_real_constructor() -> None:
    """Routing is not enough: the config keys must be the constructor's parameters, and a
    YAML list has to survive as a lineage tuple."""
    pytest.importorskip("onnxruntime", reason="onnxruntime is darwin-marked in the dev fork")
    built = factory.build_embedder(
        {
            "backend": "onnx",
            "model_path": str(MODEL),
            "train_data": ["synthetic"],
            "pretrain_data": ["none"],
            "name": "tiny-synth-embedder",
        }
    )
    assert built is not None
    assert built.provenance.train_data == ("synthetic",)


# ================================================================== hygiene
def test_the_backend_has_no_top_level_cv2_or_onnxruntime_import() -> None:
    """Both are darwin-marked in the dev fork and absent from a Linux install without the
    harness group, which still collects this module: the deny-list and the provenance checks
    have to work with no inference stack at all. AST, not grep, because a docstring naming
    onnxruntime is fine."""
    import vividly.models.reid_backend as backend

    tree = ast.parse(Path(backend.__file__).read_text(encoding="utf-8"))
    banned = {"cv2", "onnxruntime", "torch", "torchreid", "rtmlib"}
    offenders: list[str] = []
    for node in tree.body:  # module level only; function bodies are not walked
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] in banned]
        elif (
            isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in banned
        ):
            offenders.append(node.module)
    assert not offenders, f"imported at module level: {offenders}"


def test_importing_the_backend_loads_no_inference_stack() -> None:
    code = (
        "import sys; import vividly.models.reid_backend; "
        "print([m for m in ('cv2', 'onnxruntime', 'torch', 'torchreid') if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert proc.stdout.strip() == "[]"


# ================================================================== the fixture
def test_the_committed_fixture_matches_its_recorded_digest() -> None:
    """The check that runs everywhere, including where onnx is absent, which is the default
    development environment and continuous integration."""
    generate = _generator()
    assert MODEL.is_file()
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert digest == generate.EXPECTED_SHA256, (
        "tiny_embedder.onnx no longer matches EXPECTED_SHA256 in generate.py. Regenerate it "
        "with the command in that file's header and commit both together."
    )


def test_the_fixture_stays_tiny() -> None:
    total = sum(p.stat().st_size for p in FIXTURE.rglob("*") if p.is_file())
    assert MODEL.stat().st_size < 8 * 1024, "the graph is meant to be a couple of KB"
    assert total < 32 * 1024, f"reid_synth grew to {total} bytes"


def test_the_generator_header_documents_the_isolated_command() -> None:
    """The dev venv has no onnx, so the header is the instructions. If it drifts, the next
    person runs a bare `uv run` and gets ModuleNotFoundError with no idea why."""
    doc = _generator().__doc__ or ""
    for fragment in ("--isolated", "--no-project", "--with onnx", "--with numpy"):
        assert fragment in doc, f"generate.py header no longer mentions {fragment}"


def test_the_fixture_graph_matches_what_the_backend_expects() -> None:
    ort = pytest.importorskip("onnxruntime")
    generate = _generator()
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    (graph_input,) = session.get_inputs()
    (graph_output,) = session.get_outputs()

    assert graph_input.name == generate.INPUT_NAME
    assert graph_input.shape == [generate.BATCH_DIM, 3, INPUT_H, INPUT_W]
    check_declared_input_shape(graph_input.shape, "fixture")
    assert embedding_dim_from_output_shape(graph_output.shape, "fixture") == generate.EMBED_DIM


def test_the_fixture_regenerates_byte_for_byte(tmp_path: Path) -> None:
    """Full rebuild, byte for byte. Skips wherever `onnx` is absent, which is the default
    development environment; the rebuild is run by hand with the isolated command in
    generate.py's header (`--isolated --no-project`, so the curate fork's onnx is not on the
    path either), and verified so on 2026-08-07 with onnx 1.22.0."""
    pytest.importorskip("onnx", reason="onnx is absent from the default development environment")
    rebuilt = _generator().build(tmp_path)
    assert rebuilt.read_bytes() == MODEL.read_bytes()
