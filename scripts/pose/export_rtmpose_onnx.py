#!/usr/bin/env python3
"""Export an RTMPose-m topdown checkpoint to ONNX (simcc_x/simcc_y heads).

mmdeploy is not used here: research/pose-stack.md records it as effectively dead. This
is the direct torch.onnx.export route: the graph takes a normalized 256x192 crop batch
and emits the two SimCC classification vectors; argmax/decode stays outside the graph
(same contract rtmlib uses). Parity-checked against the torch forward.

Run inside the vendored mmpose trainer environment, where $POSE_REPO is the vendored pose
repository at pin 759b39c1 and $REPO is this repository's checkout, so the recipe's base
configuration resolves. The recorded invocation, from TRAINING.md:

  cd "$POSE_REPO" && .venv/bin/python "$REPO/scripts/pose/export_rtmpose_onnx.py" \
      --config configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_person_v2_ft.py \
      --ckpt <file.pth> --out <file.onnx> --device cpu
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from mmpose.apis import init_model


class SimCCWrapper(torch.nn.Module):
    """backbone -> neck? -> head, returning (simcc_x, simcc_y)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.extract_feat(x)
        return self.model.head.forward(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = init_model(args.config, args.ckpt, device=args.device)
    model.eval()
    wrapper = SimCCWrapper(model).eval()

    dummy = torch.randn(1, 3, 256, 192, device=args.device)
    with torch.no_grad():
        ref_x, ref_y = wrapper(dummy)

    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["simcc_x", "simcc_y"],
        dynamic_axes={"input": {0: "batch"}, "simcc_x": {0: "batch"}, "simcc_y": {0: "batch"}},
        opset_version=17,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ox, oy = sess.run(None, {"input": dummy.cpu().numpy()})
    dx = float(np.abs(ox - ref_x.cpu().numpy()).max())
    dy = float(np.abs(oy - ref_y.cpu().numpy()).max())
    sha = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print(f"PARITY simcc_x max|d|={dx:.3e} simcc_y max|d|={dy:.3e}")
    print(f"ONNX_SHA256 {sha}")
    print(f"SHAPES simcc_x={ox.shape} simcc_y={oy.shape}")
    assert dx < 1e-4 and dy < 1e-4, "parity check failed"
    print("EXPORT_OK")


if __name__ == "__main__":
    main()
