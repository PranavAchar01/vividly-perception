# reid_synth: a 1.3 KB deterministic ONNX embedder

One tiny ONNX graph, `tiny_embedder.onnx`, used by `tests/test_reid_backend.py` to exercise
`vividly.models.reid_backend.OnnxAppearanceEmbedder` end to end with no download, no
counsel gate, and no real re-identification weights anywhere near the repo.

It is not a re-ID model. Its weights are arithmetic (see `conv_weight_values()` in
`generate.py`), it was trained on nothing, and its outputs mean nothing. What it has is the
right *shape*: an NCHW float32 input with a dynamic batch and a fixed 3 x 256 x 128 crop,
and a declared 8-wide feature output, which is exactly the contract the backend checks a
real export against.

Guardrails, stated because a fixture is still data: no real people, no recorded footage, no
dataset bytes of any kind, and no simulated or acted medical event. Nothing here needs an
entry in `../ATTRIBUTION.md`, because nothing here came from anywhere.

## The graph

    images [batch, 3, 256, 128]
      -> Conv  8 filters, 3x3, stride 2, pads 0   -> [batch, 8, 127, 63]
      -> GlobalAveragePool                        -> [batch, 8, 1, 1]
      -> Flatten(axis=1)                          -> [batch, 8]

    opset 13, ir_version 8, 1288 bytes
    sha256 433ccc4c9efdae1d881326ccd2796ab7861b2a6c194daa61e678309b3b03cd57

Zero padding is the load-bearing detail. Every 3x3 window lies wholly inside the crop, so a
crop of one flat colour produces the same value at every output position and the embedding
before normalisation is exactly

    feature[o] = sum_c x_c * kernel_sums()[o][c] + conv_bias_values()[o]

where `x_c` is the preprocessed value of channel c. `test_reid_backend.py` checks the
backend against that hand computation, which is what makes the test a statement about
preprocessing and inference rather than a comparison against a recorded blob.

## Regenerating

    uv run --isolated --no-project --with onnx --with numpy \
        python tests/fixtures/reid_synth/generate.py

Not a bare `uv run`. The dev venv has no `onnx` package and is not getting one: onnxruntime
runs graphs, it does not author them, and this repo does not add a graph-authoring
dependency to the shipped fork to build a 1.3 KB fixture. `--isolated --no-project` builds
a throwaway environment and touches neither `uv.lock` nor `.venv`.

Verified byte-for-byte reproducible with onnx 1.22.0 on 2026-08-07: regenerating into a
temp directory produced an identical 1288-byte file. The pinned `ir_version` and opset in
`generate.py` are what make that hold across onnx releases; if one ever changes the
encoding regardless, the digest moves, and `EXPECTED_SHA256` in `generate.py` is updated in
the same commit as the regenerated file.

`test_the_committed_fixture_matches_its_recorded_digest` is the check that runs everywhere,
including on a box with no `onnx` to regenerate with;
`test_the_fixture_regenerates_byte_for_byte` does the full rebuild and skips where `onnx`
is absent, which is every environment this repository declares except the `curate` extra on
macOS, where `pyproject.toml` declares `onnx>=1.16 ; sys_platform == 'darwin'`.
