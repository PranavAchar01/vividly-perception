#!/usr/bin/env python3
"""Regenerate tests/fixtures/reid_synth/tiny_embedder.onnx, byte for byte.

    uv run --isolated --no-project --with onnx --with numpy \\
        python tests/fixtures/reid_synth/generate.py

That command, not a bare `uv run`. The dev venv has no `onnx` package and is not getting
one: onnxruntime executes graphs, it does not author them, and a graph-authoring
dependency in the shipped fork to build a 1.3 KB test fixture is not a trade this repo
makes. `--isolated --no-project` builds a throwaway environment for the two packages this
script needs and touches neither uv.lock nor .venv.

What the fixture is: a deterministic 8-dimensional embedder, small enough to commit, with
the input signature `OnnxAppearanceEmbedder` expects (NCHW float32, dynamic batch, 3 x 256
x 128) and a declared feature width, so the whole backend path is testable with no
download, no counsel gate, and no real re-ID weights on disk. Nothing here is trained on
anything; the weights are arithmetic. It is not a re-ID model and its outputs mean nothing.

    images [batch, 3, 256, 128]
      -> Conv  8 filters, 3x3, stride 2, no padding   -> [batch, 8, 127, 63]
      -> GlobalAveragePool                            -> [batch, 8, 1, 1]
      -> Flatten(axis=1)                              -> [batch, 8]

Padding is zero on purpose. Without it every window lies inside the crop, so for a crop of
one flat colour the embedding is exactly `sum_c(x_c * sum_kernel(W[o, c])) + b[o]` before
normalisation, and tests/test_reid_backend.py checks the backend against that hand
computation rather than against a recorded blob.

Byte-for-byte determinism, since the committed file's sha256 is asserted in the tests:

  * every weight is an exact binary fraction (eighths and sixteenths) of a small integer
    expression, so no float formatting or rounding is involved anywhere;
  * ir_version and the opset are pinned below rather than inherited from whatever `onnx`
    release uv resolved, which is the usual source of drift;
  * every node, initializer, input and output is explicitly named, and no doc_string is
    set, so the serialised protobuf has no version-dependent field in it.

If a future `onnx` release changes the encoding anyway, the digest moves: regenerate,
update EXPECTED_SHA256 below, and commit both in the same change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_NAME = "tiny_embedder.onnx"

#: Graph shape. INPUT_H/INPUT_W mirror vividly.models.reid_backend deliberately: the point
#: of the fixture is that the backend's declared-shape check passes on it.
EMBED_DIM = 8
IN_CHANNELS = 3
INPUT_H = 256
INPUT_W = 128
KERNEL = 3
STRIDE = 2

INPUT_NAME = "images"
OUTPUT_NAME = "embeddings"
BATCH_DIM = "batch"
WEIGHT_NAME = "conv.weight"
BIAS_NAME = "conv.bias"

#: Pinned, not inherited. Opset 13 with IR version 8 is accepted by every onnxruntime this
#: repo pins or plausibly will (1.28.0 today) and by every onnx release that can still
#: write it.
OPSET = 13
IR_VERSION = 8

PRODUCER_NAME = "vividly-reid-synth"
PRODUCER_VERSION = "1"
GRAPH_NAME = "tiny_embedder"

#: sha256 of the committed tiny_embedder.onnx. tests/test_reid_backend.py asserts it, which
#: is the check that survives on a box with no onnx package to regenerate with.
EXPECTED_SHA256 = "433ccc4c9efdae1d881326ccd2796ab7861b2a6c194daa61e678309b3b03cd57"


def conv_weight_values() -> list[float]:
    """The (8, 3, 3, 3) kernel, flattened in ONNX order (o, c, kh, kw).

    Pure Python and importable without onnx or numpy, so a test can recompute the expected
    embedding for a flat-colour crop from the same numbers the graph holds. Values land on
    eighths in [-0.375, 0.375], all exact in float32.

    The modulus is 7, not 9, and that is load-bearing. A 9-cycle over a 9-tap kernel gives
    every kernel the same residues in a different order, so every kernel sums to zero and
    the graph returns its bias for any input, which would be a fixture that passes a shape
    test and cannot fail an inference test. With 7 the per-channel sums differ, so distinct
    crops produce distinct embeddings, which is the property the tests actually rely on.
    """
    return [
        ((o * 27 + c * 9 + kh * KERNEL + kw) % 7 - 3) / 8.0
        for o in range(EMBED_DIM)
        for c in range(IN_CHANNELS)
        for kh in range(KERNEL)
        for kw in range(KERNEL)
    ]


def conv_bias_values() -> list[float]:
    """Per-output-channel bias, on sixteenths. Keeps a black crop off the origin, so no
    embedding is a zero row and the L2 normalisation always has something to divide by."""
    return [(o - 4) / 16.0 for o in range(EMBED_DIM)]


def kernel_sums() -> list[list[float]]:
    """`[o][c]` = the sum of W[o, c] over the 3x3 kernel. The hand-computation shortcut:
    with zero padding, a flat crop of channel values x gives feature[o] = sum_c x_c *
    kernel_sums()[o][c] + bias[o], identically at every output position."""
    flat = conv_weight_values()
    per_channel = KERNEL * KERNEL
    return [
        [
            sum(flat[(o * IN_CHANNELS + c) * per_channel : (o * IN_CHANNELS + c + 1) * per_channel])
            for c in range(IN_CHANNELS)
        ]
        for o in range(EMBED_DIM)
    ]


def build_model():  # returns an onnx.ModelProto; the type is unimportable up here
    """The ModelProto. `onnx` and `numpy` are imported here, never at module scope, so the
    constants and the weight recipe above stay importable in the dev venv (which has
    neither onnx nor a reason to grow one)."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(
        np.asarray(conv_weight_values(), np.float32).reshape(
            EMBED_DIM, IN_CHANNELS, KERNEL, KERNEL
        ),
        name=WEIGHT_NAME,
    )
    bias = numpy_helper.from_array(np.asarray(conv_bias_values(), np.float32), name=BIAS_NAME)

    conv = helper.make_node(
        "Conv",
        inputs=[INPUT_NAME, WEIGHT_NAME, BIAS_NAME],
        outputs=["feature_map"],
        name="conv",
        kernel_shape=[KERNEL, KERNEL],
        strides=[STRIDE, STRIDE],
        pads=[0, 0, 0, 0],
    )
    pool = helper.make_node(
        "GlobalAveragePool", inputs=["feature_map"], outputs=["pooled"], name="gap"
    )
    flatten = helper.make_node(
        "Flatten", inputs=["pooled"], outputs=[OUTPUT_NAME], name="flatten", axis=1
    )

    graph = helper.make_graph(
        nodes=[conv, pool, flatten],
        name=GRAPH_NAME,
        inputs=[
            helper.make_tensor_value_info(
                INPUT_NAME, TensorProto.FLOAT, [BATCH_DIM, IN_CHANNELS, INPUT_H, INPUT_W]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [BATCH_DIM, EMBED_DIM])
        ],
        initializer=[weight, bias],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name=PRODUCER_NAME,
        producer_version=PRODUCER_VERSION,
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def build(dest: Path) -> Path:
    """Write `<dest>/tiny_embedder.onnx` and return its path."""
    path = Path(dest) / MODEL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_model().SerializeToString())
    return path


def main() -> int:
    path = build(Path(__file__).resolve().parent)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    print(f"sha256 {digest}")
    if digest != EXPECTED_SHA256:
        print(f"NOTE: EXPECTED_SHA256 in this file says {EXPECTED_SHA256}; update it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
