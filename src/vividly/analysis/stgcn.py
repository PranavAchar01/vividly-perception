"""ST-GCN skeleton-action head — the LEARNED fall/action confirmer (scaffold).

A spatial-temporal graph convolutional network over the COCO-17 skeleton: graph
convolution mixes information across the body's joint graph, temporal convolution mixes it
across time, and a stack of such blocks feeds a linear classifier (fall / not-fall, or the
7-action set of the GajuuzZ pipeline). Reference: Yan/Xiong/Lin, ST-GCN, AAAI 2018.

STATUS ON THIS MACHINE. This is the architecture + a numpy forward pass, ready to load
weights and to be exported to ONNX — NOT a trained detector. There is no torch and no
fall-trained ST-GCN weights on this Mac, so it does not classify anything yet; the live
fall head is the geometry FallAnalyzer (transition + stillness), which is exactly the
research-endorsed method and needs no weights. To ACTIVATE this head:
  * PRODUCTION: train ST-GCN on a fall dataset (UR Fall / CAUCAFall / Le2i) on a GPU,
    export to ONNX, and run it via onnxruntime on the shared pose stream (~5-10 MB); or
  * convert the GajuuzZ Human-Falling-Detect-Tracks ST-GCN weights to this COCO-17 graph.
The PRODUCTION path for real weights is to export a trained torch ST-GCN to ONNX and run it
via onnxruntime. This numpy implementation is a reference/scaffold: `STGCN.from_npz(path)`
loads weights arranged in THIS module's own layout ({"blocks": [...], "fc_w", "fc_b"} — see
`_random_weights`), not an arbitrary torch state_dict, so a converter must emit that layout.
Until real weights are loaded, `STGCN(...).loaded` is False and callers treat it as absent.

The numpy forward mirrors the standard ST-GCN layer math (graph conv = A·X·W, then a dense
channel-mixing convolution over the time axis, BN, ReLU, residual), so shapes and the
layer plumbing are unit-testable without any training.
"""

from __future__ import annotations

import itertools

import numpy as np

# COCO-17 skeleton edges (the body graph). Symmetric; self-loops added in the adjacency.
COCO17_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)  # fmt: skip
NUM_JOINTS = 17


def normalized_adjacency(edges=COCO17_EDGES, n=NUM_JOINTS) -> np.ndarray:
    """Symmetric-normalized adjacency with self-loops: Â = D^-1/2 (A+I) D^-1/2."""
    a = np.eye(n, dtype=np.float32)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    deg = a.sum(1)
    dinv = np.diag((deg**-0.5).astype(np.float32))
    return (dinv @ a @ dinv).astype(np.float32)


def _bn(x: np.ndarray, p: dict, axis: int) -> np.ndarray:
    """Batch-norm in inference mode over channel `axis`, using stored running stats."""
    shape = [1] * x.ndim
    shape[axis] = -1
    g = p["weight"].reshape(shape)
    b = p["bias"].reshape(shape)
    m = p["running_mean"].reshape(shape)
    v = p["running_var"].reshape(shape)
    return g * (x - m) / np.sqrt(v + 1e-5) + b


class _STBlock:
    """One spatial-temporal block: graph conv (over joints) then temporal conv (over time)."""

    def __init__(self, w: dict, adjacency: np.ndarray, residual: bool) -> None:
        self.A = adjacency
        self.gcn_w = w["gcn_w"]  # (Cin, Cout)
        self.gcn_b = w.get("gcn_b")
        self.gcn_bn = w.get("gcn_bn")
        self.tcn_w = w["tcn_w"]  # (Cout, Cout, kt) dense channel-mixing temporal kernel
        self.tcn_bn = w.get("tcn_bn")
        self.residual = residual
        self.res_w = w.get("res_w")  # optional 1x1 to match channels

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (T, V, Cin)
        res = x
        # --- graph conv: aggregate neighbors then mix channels ---
        agg = np.einsum("vw,twc->tvc", self.A, x)  # (T,V,Cin)
        h = agg @ self.gcn_w  # (T,V,Cout)
        if self.gcn_b is not None:
            h = h + self.gcn_b
        if self.gcn_bn is not None:
            h = _bn(h, self.gcn_bn, axis=2)
        h = np.maximum(h, 0.0)
        # --- temporal conv: 1D conv over time, per joint/channel ---
        h = self._temporal(h)
        if self.tcn_bn is not None:
            h = _bn(h, self.tcn_bn, axis=2)
        if self.residual:
            if self.res_w is not None:
                res = res @ self.res_w
            h = h + res
        return np.maximum(h, 0.0)

    def _temporal(self, h: np.ndarray) -> np.ndarray:
        # h: (T,V,C). kernel self.tcn_w: (C, C, kt). "same" conv over T.
        _cin, _cout, kt = self.tcn_w.shape
        pad = kt // 2
        hp = np.pad(h, ((pad, pad), (0, 0), (0, 0)))
        out = np.zeros_like(h)
        for k in range(kt):
            # (T,V,Cin) @ (Cin,Cout) accumulated over the kernel taps
            out += hp[k : k + h.shape[0]] @ self.tcn_w[:, :, k].T
        return out


class STGCN:
    """Numpy ST-GCN over the COCO-17 graph. Loaded=False until real weights are supplied."""

    def __init__(
        self, num_classes: int = 2, channels=(3, 32, 64), weights: dict | None = None
    ) -> None:
        self.A = normalized_adjacency()
        self.num_classes = num_classes
        self.channels = channels
        self.loaded = weights is not None
        self._w = weights or self._random_weights(channels, num_classes)

    @classmethod
    def from_npz(cls, path: str) -> STGCN:
        """Load weights arranged in this module's layout (an object array under "blocks" plus
        "fc_w"/"fc_b"; see `_random_weights`). A converter from a trained model must emit that
        layout — this is NOT an arbitrary torch state_dict loader. Flips `loaded` to True."""
        d = dict(np.load(path, allow_pickle=True))
        if "blocks" not in d or "fc_w" not in d:
            raise ValueError(
                "npz missing this module's ST-GCN layout (need 'blocks', 'fc_w', 'fc_b'); "
                "convert the trained model into that layout, or export to ONNX and run it "
                "via onnxruntime instead."
            )
        meta = d.pop("__meta__", None)
        num_classes = int(meta.item()["num_classes"]) if meta is not None else 2
        if isinstance(d.get("blocks"), np.ndarray):
            d["blocks"] = list(d["blocks"])
        return cls(num_classes=num_classes, weights=d)

    def _random_weights(self, channels, num_classes) -> dict:
        # Small deterministic init so the forward pass RUNS (shape/plumbing tests) — this is
        # NOT trained and its outputs are meaningless until real weights are loaded.
        rng = np.random.default_rng(0)
        blocks = []
        for cin, cout in itertools.pairwise(channels):
            blocks.append(
                {
                    "gcn_w": rng.standard_normal((cin, cout)).astype(np.float32) * 0.1,
                    "gcn_b": np.zeros(cout, np.float32),
                    "tcn_w": (rng.standard_normal((cout, cout, 9)) * 0.1).astype(np.float32),
                    "res_w": (rng.standard_normal((cin, cout)) * 0.1).astype(np.float32)
                    if cin != cout
                    else None,
                }
            )
        return {
            "blocks": blocks,
            "fc_w": (rng.standard_normal((channels[-1], num_classes)) * 0.1).astype(np.float32),
            "fc_b": np.zeros(num_classes, np.float32),
        }

    def forward(self, skeleton: np.ndarray) -> np.ndarray:
        """skeleton: (T, 17, C) joint features over time -> class logits (num_classes,)."""
        x = np.asarray(skeleton, np.float32)
        if x.ndim != 3 or x.shape[1] != NUM_JOINTS:
            raise ValueError(f"expected (T,17,C), got {x.shape}")
        blocks = self._w["blocks"]
        for i, bw in enumerate(blocks):
            block = _STBlock(bw, self.A, residual=(i > 0))
            x = block(x)
        pooled = x.mean(axis=(0, 1))  # global average over time and joints -> (C,)
        return pooled @ self._w["fc_w"] + self._w["fc_b"]

    def predict_proba(self, skeleton: np.ndarray) -> np.ndarray:
        z = self.forward(skeleton)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()
