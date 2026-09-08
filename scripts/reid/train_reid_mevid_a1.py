#!/usr/bin/env python3
"""v2 re-ID Phase A: train OSNet-x1_0 on MEVID from an ImageNet-only initialisation.

    .venv/bin/python -u scripts/reid/train_reid_mevid_a1.py \
        --config configs/reid/train_mevid_a1.yaml

Runs on the Linux training pod (RTX 5090, torch >= 2.7 / cu128); the mac fork is torch-free
by design and cannot run this. Everything is logged to stdout because the pod captures
the stream, so there is no second log file to look for.

What re-identification means here, so nothing downstream over-reads the numbers: a
session-scoped appearance embedding over person crops, whose long-gap re-associations are
proposed to a person and never auto-bound. Appearance only: no pose, gait or body-shape
identity signal, and nothing face-adjacent (POLICY.md, section 4, and rule 2 on facial
recognition). Every weight this script produces is
`weight_class: prototype`: interim, internal, never quoted as clinical performance, never
deployed commercially without counsel.

Three evaluations are reported side by side over one feature extraction, so the deltas are
attributable to the protocol rather than to a re-run:

  1. general       - torchreid's standard protocol (euclidean, raw features), the number
                     that is comparable to published MEVID results.
  2. camera-debias - the per-camera mean embedding (over the pooled query+gallery test
                     features, grouped by camid) subtracted from every feature, then L2
                     renormalised, then cosine. Reported next to an L2+cosine baseline so
                     the debias delta is separated from the metric change.
  3. SC-approx     - an in-house definition, not the MEVID paper's same-clothes protocol.
                     Positives are same pid and same outfit_id; same-pid different-outfit
                     gallery tracklets are removed from the ranking as junk rather than
                     counted as negatives. Labelled as in-house everywhere it is printed.

Outputs under `out_dir`: torchreid's per-epoch checkpoints, `model-best.pth.tar` (best test
mAP when eval_freq > 0), `curves.csv`, `eval_summary.json`, `osnet_x1_0_mevid_a1.onnx` and
its `.sha256`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mevid_dataset import register_mevid  # noqa: E402

#: OSNet's native input geometry; batch is the only axis allowed to float, because a re-ID
#: batch is N crops of one fixed size and letting H/W vary silently changes the embedding.
INPUT_SHAPE = (1, 3, 256, 128)
INPUT_NAME = "images"
OUTPUT_NAME = "embeddings"
BATCH_AXIS = "batch"

#: Where torchreid parks a model-zoo download (torchreid/models/osnet.py:init_pretrained_weights).
TORCH_HOME_ENV = ("TORCH_HOME", "XDG_CACHE_HOME")
IMAGENET_INIT_FILENAME = "osnet_x1_0_imagenet.pth"

CURVE_COLUMNS = (
    "epoch",
    "loss_t",
    "loss_x",
    "acc",
    "lr",
    "test_map",
    "test_rank1",
)

REQUIRED_KEYS = (
    "root",
    "out_dir",
    "seed",
    "epochs",
    "seq_len",
    "sample_method",
    "batch_train",
    "lr",
    "weight_class",
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_PARITY = 7


class ConfigError(RuntimeError):
    """The YAML handed to --config is missing something or says something impossible."""


class ParityError(RuntimeError):
    """The exported ONNX graph does not reproduce the torch embedding."""


# ------------------------------------------------------------------ config
def load_config(path: Path) -> dict[str, Any]:
    """Read and validate the run config. No path or hyperparameter is ever inline."""
    if not path.is_file():
        raise ConfigError(f"--config {path} does not exist")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(cfg).__name__}")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"{path}: missing required key(s) {missing}")
    # The weight class is declared by the recipe, not inferred (POLICY.md, rule 6). MEVID is
    # a TRAIN_PROTOTYPE entry, so a run that forgets to say so would produce a checkpoint
    # with no later evidence that it is prototype-class.
    if cfg["weight_class"] != "prototype":
        raise ConfigError(
            f"{path}: weight_class is {cfg['weight_class']!r}. MEVID is a TRAIN_PROTOTYPE "
            "registry entry (configs/datasets/mevid.yaml); only prototype weights may be "
            "trained from it, and the recipe has to declare it."
        )
    return cfg


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def imagenet_init_path() -> Path:
    """Where torchreid caches `osnet_x1_0_imagenet.pth`, mirroring its own resolution."""
    torch_home = os.environ.get(TORCH_HOME_ENV[0])
    if torch_home is None:
        cache = os.environ.get(TORCH_HOME_ENV[1], str(Path("~/.cache").expanduser()))
        torch_home = str(Path(cache) / "torch")
    return Path(torch_home).expanduser() / "checkpoints" / IMAGENET_INIT_FILENAME


def preflight_imports(cfg: dict[str, Any]) -> None:
    """Fail in the first second, not after thirty epochs.

    Every one of these is needed at a point where losing the run would be expensive:
    tensorboard the moment `Engine.run` opens its SummaryWriter, gdown the moment
    `build_model(pretrained=True)` reaches for the ImageNet weights, onnxruntime only at
    the very end for the export parity check. On the pod they do not all arrive with the
    training extra, so the right place to discover that is before any GPU time is spent.
    """
    needed = [
        ("torch", "torch"),
        ("torchreid", "torchreid"),
        ("tensorboard", "torch.utils.tensorboard (torchreid's Engine opens a SummaryWriter)"),
        ("onnxruntime", "onnxruntime (final ONNX parity check)"),
    ]
    if bool(cfg.get("pretrained", True)):
        needed.append(("gdown", "gdown (torchreid fetches the ImageNet init with it)"))

    import importlib.util

    missing = [why for mod, why in needed if importlib.util.find_spec(mod) is None]
    if missing:
        raise ConfigError(
            "this interpreter is missing: "
            + ", ".join(missing)
            + ". They are all required before the run finishes, so the run is refused now "
            "rather than after training. Install them into the pod venv (the v2 `cuda` "
            "extra does not currently carry torchreid, gdown, tensorboard or onnxruntime) "
            "or set `pretrained: false` if the ImageNet init is genuinely not wanted."
        )


# ------------------------------------------------------------------ ranking
def _cmc_and_map(
    distmat: np.ndarray,
    match: np.ndarray,
    junk: np.ndarray,
    *,
    max_rank: int = 50,
) -> tuple[np.ndarray, float, int]:
    """CMC curve and mAP under caller-supplied match/junk masks.

    `match[q, g]` is True when gallery entry g is a correct answer for query q; `junk[q, g]`
    is True when g must be removed from the ranking rather than counted as a negative. The
    arithmetic is the standard Market-1501 formulation torchreid implements, restated here
    only because torchreid's version hard-codes the junk rule to same-pid-same-camera and
    the SC-approx protocol needs a different one.

    Queries left with no positive after removal are skipped, exactly as upstream does, and
    the returned count says how many contributed.
    """
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    order = np.argsort(distmat, axis=1)

    cmc_sum = np.zeros(max_rank, dtype=np.float64)
    ap_sum = 0.0
    valid = 0
    for q in range(num_q):
        keep = ~junk[q][order[q]]
        raw = match[q][order[q]][keep].astype(np.float64)
        if not raw.any():
            continue
        cmc = raw.cumsum()
        cmc[cmc > 1] = 1
        # A query whose kept gallery is shorter than max_rank still has a defined CMC past
        # its own list end: the hit state cannot change once the list is exhausted, so the
        # curve is held at its final value rather than the row being dropped (torchreid's
        # own python ranker would raise a ragged-array error here instead).
        taken = min(max_rank, cmc.size)
        padded = np.full(max_rank, cmc[taken - 1], dtype=np.float64)
        padded[:taken] = cmc[:taken]
        cmc_sum += padded
        positions = np.arange(1, raw.size + 1, dtype=np.float64)
        ap_sum += float((raw.cumsum() / positions * raw).sum() / raw.sum())
        valid += 1
    if valid == 0:
        raise RuntimeError("no query has a positive in the gallery under this protocol")
    return cmc_sum / valid, ap_sum / valid, valid


def general_masks(
    q_pids: np.ndarray, q_cams: np.ndarray, g_pids: np.ndarray, g_cams: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standard protocol: same pid matches; same pid on the same camera is junk."""
    same_pid = q_pids[:, None] == g_pids[None, :]
    same_cam = q_cams[:, None] == g_cams[None, :]
    return same_pid, same_pid & same_cam


def sc_approx_masks(
    q_pids: np.ndarray,
    q_cams: np.ndarray,
    q_outfits: np.ndarray,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
    g_outfits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """SC-approx protocol masks, an in-house definition, not the MEVID paper's.

    A positive is same pid and same outfit. A same-pid gallery tracklet wearing a different
    outfit is removed as junk rather than scored as a negative: counting it as a negative
    would reward an embedding for telling two outfits of one person apart, which is the
    opposite of what a session-scoped re-identification is asked to do.
    Same-pid-same-camera stays junk as in the standard protocol.
    """
    same_pid = q_pids[:, None] == g_pids[None, :]
    same_cam = q_cams[:, None] == g_cams[None, :]
    same_outfit = q_outfits[:, None] == g_outfits[None, :]
    match = same_pid & same_outfit
    junk = (same_pid & same_cam) | (same_pid & ~same_outfit)
    return match, junk


def camera_debias(features: Any, camids: np.ndarray) -> Any:
    """Subtract each camera's mean embedding, then L2-renormalise.

    The mean is computed over the pooled query and gallery test features for that camid, which
    is what the caller passes in: a per-split mean would leak the split boundary into the
    distance matrix. Cameras seen once contribute a mean equal to their single feature, so
    that feature collapses to zero and renormalises to an arbitrary direction; those are
    reported in the summary rather than silently dropped.
    """
    import torch

    out = features.clone()
    for cam in np.unique(camids):
        rows = torch.from_numpy(np.flatnonzero(camids == cam))
        out[rows] = out[rows] - out[rows].mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(out, p=2, dim=1)


# ------------------------------------------------------------------ engine
def build_engine_class(loss: str = "softmax") -> type:
    """Subclass the video engine for `loss` with curve recording and best-mAP checkpointing.

    `loss` is `softmax` (torchreid's VideoSoftmaxEngine: cross-entropy over the training
    identities with label smoothing, the library's own OSNet recipe) or `triplet`
    (VideoTripletEngine: the same softmax term plus a batch-hard triplet term). On MEVID
    the triplet engine collapsed twice, with and without a classifier-only warm start:
    hard mining on unnormalised features whose clothes-change positives sit far from the
    anchor is satisfied cheapest by shrinking every feature to one point, and the softmax
    term could not hold it. The softmax engine has no such path.

    torchreid 1.4.0's `Engine.train()` keeps its loss meters in a local `MetricMeter` and
    `Engine.run()` calls `save_model(epoch, rank1, save_dir)` with no `is_best`, so neither
    the per-epoch loss components nor a best checkpoint exist unless something reaches into
    the loop. These three overrides are that reach, and nothing else is changed.
    """
    from torchreid.engine import VideoSoftmaxEngine, VideoTripletEngine

    if loss not in ("softmax", "triplet"):
        raise ConfigError(f"loss must be softmax or triplet, got {loss!r}")
    base = VideoSoftmaxEngine if loss == "softmax" else VideoTripletEngine

    class CurveRecordingEngine(base):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *args: Any,
            curves_path: Path,
            save_dir: Path,
            keep_last: int = 2,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.curves_path = curves_path
            self.checkpoint_dir = save_dir
            self.keep_last = keep_last
            self._sums: dict[str, float] = {}
            self._batches = 0
            self._pending: dict[str, Any] | None = None
            self.best_map = float("-inf")
            self.best_epoch = -1
            self.last_eval: dict[str, Any] | None = None
            self._eval_epoch = -1
            with self.curves_path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(CURVE_COLUMNS)

        # -- curve plumbing ------------------------------------------------
        def _flush_pending(self) -> None:
            if self._pending is None:
                return
            row = [self._pending.get(col, "") for col in CURVE_COLUMNS]
            with self.curves_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(row)
            self._pending = None

        def forward_backward(self, data: Any) -> dict[str, float]:
            summary = super().forward_backward(data)
            for key, value in summary.items():
                # The softmax engine reports its cross-entropy as `loss`; it is the same
                # term the triplet engine reports as `loss_x`, so the curve column is shared.
                column = "loss_x" if key == "loss" else key
                self._sums[column] = self._sums.get(column, 0.0) + float(value)
            self._batches += 1
            return summary

        def train(self, *args: Any, **kwargs: Any) -> Any:
            # The previous epoch's row is flushed here rather than at the end of train(),
            # because run() may call test() after train() and that is where test_map comes
            # from. run() ends with a final test, so the last row is flushed by the caller.
            self._flush_pending()
            self._sums, self._batches = {}, 0
            lr = self.get_current_lr()  # read before Engine.train() steps the scheduler
            result = super().train(*args, **kwargs)
            n = max(self._batches, 1)
            self._pending = {
                "epoch": self.epoch + 1,
                "lr": f"{lr:.8g}",
                **{k: f"{v / n:.6f}" for k, v in self._sums.items()},
            }
            # torchreid's run() checkpoints only on eval epochs and once at the end, so a
            # 30-epoch run at eval_freq 10 leaves 27 epochs unsaved and a crash at epoch 29
            # costs the whole run. Save every epoch here instead. At an eval epoch this
            # writes before the test and run() then overwrites the same file with the
            # tested rank1, which is the order that keeps the recorded number correct.
            self.save_model(
                self.epoch,
                float((self.last_eval or {}).get("rank1", -1.0)),
                str(self.checkpoint_dir),
            )
            return result

        # -- eval capture + best checkpoint --------------------------------
        def _evaluate(self, *args: Any, **kwargs: Any) -> Any:
            rank1, m_ap = super()._evaluate(*args, **kwargs)
            self.last_eval = {"rank1": float(rank1), "map": float(m_ap)}
            self._eval_epoch = self.epoch
            if self._pending is not None:
                self._pending["test_map"] = f"{float(m_ap):.6f}"
                self._pending["test_rank1"] = f"{float(rank1):.6f}"
            return rank1, m_ap

        def save_model(
            self, epoch: int, rank1: float, save_dir: str, is_best: bool = False
        ) -> None:
            # "Best" is by test mAP, not rank-1: rank-1 on a tracklet benchmark this small
            # moves in visible steps and ties constantly, and mAP is the number the re-ID
            # decision is read off. Only a mAP measured for this same epoch can promote
            # a checkpoint: the per-epoch save above runs with a stale last_eval, and
            # letting that re-crown an old score would label the wrong file best.
            current = (self.last_eval or {}).get("map")
            if current is not None and self._eval_epoch == epoch and current > self.best_map:
                self.best_map, self.best_epoch, is_best = current, epoch + 1, True
            super().save_model(epoch, rank1, save_dir, is_best=is_best)
            self._prune_checkpoints()

        def _prune_checkpoints(self) -> None:
            """Keep the newest `keep_last` epoch checkpoints; model-best is never touched.

            Each file carries the optimizer and scheduler state as well as the weights, so
            30 of them is a few hundred MB of pod disk for no benefit: the only ones anyone
            reads are the newest (to resume) and the best (to export).
            """
            if self.keep_last <= 0 or not self.checkpoint_dir.is_dir():
                return
            # torchreid parks its checkpoints one level down, in <save_dir>/model/: the
            # directory name is the literal `model`, which is where the best checkpoint is
            # read from at the end of the run. Every subdirectory is scanned rather than
            # that one name, so the pruning does not depend on the name holding.
            for model_dir in (d for d in self.checkpoint_dir.iterdir() if d.is_dir()):
                saved = sorted(
                    model_dir.glob("model.pth.tar-*"),
                    key=lambda p: int(p.name.rsplit("-", 1)[-1]),
                )
                for stale in saved[: -self.keep_last]:
                    stale.unlink(missing_ok=True)

    return CurveRecordingEngine


# ------------------------------------------------------------------ features
def extract_split_features(engine: Any, loader: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    """Tracklet embeddings, pids and camids for one test loader, in loader order.

    Mirrors torchreid's own `Engine._evaluate` extraction so the features here and the ones
    behind the standard protocol number are the same tensors, not a lookalike.
    """
    import torch

    engine.set_model_mode("eval")
    feats, pids, cams = [], [], []
    with torch.no_grad():
        for data in loader:
            imgs, batch_pids, batch_cams = engine.parse_data_for_eval(data)
            if engine.use_gpu:
                imgs = imgs.cuda()
            out = engine.extract_features(imgs).cpu()
            feats.append(out)
            pids.extend(batch_pids.tolist())
            cams.extend(batch_cams.tolist())
    return torch.cat(feats, 0), np.asarray(pids), np.asarray(cams)


def outfit_array(dataset: Any, mode: str, expected: int) -> np.ndarray:
    """Outfit ids for a split, positionally aligned with its loader order.

    The test loaders are built with `shuffle=False`, so loader order is dataset order is
    metadata order. The length check is the tripwire for that assumption ever changing.
    """
    meta = dataset.meta_for(mode)
    if len(meta) != expected:
        raise RuntimeError(
            f"{mode} metadata has {len(meta)} rows but {expected} features were extracted; "
            "the loader is no longer in dataset order and the outfit alignment is unsafe"
        )
    return np.asarray([m.outfit_id for m in meta])


# ------------------------------------------------------------------ protocols
def report_protocol(
    label: str,
    distmat: np.ndarray,
    match: np.ndarray,
    junk: np.ndarray,
    *,
    note: str = "",
) -> dict[str, Any]:
    cmc, m_ap, valid = _cmc_and_map(distmat, match, junk)
    print(f"\n--- {label} ---")
    if note:
        print(f"    {note}")
    print(f"    mAP     : {m_ap:.2%}")
    for rank in (1, 5, 10):
        if rank <= cmc.size:
            print(f"    Rank-{rank:<3}: {cmc[rank - 1]:.2%}")
    print(f"    queries scored: {valid}")
    return {
        "label": label,
        "note": note,
        "mAP": m_ap,
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]) if cmc.size >= 5 else None,
        "rank10": float(cmc[9]) if cmc.size >= 10 else None,
        "queries_scored": valid,
    }


def run_all_protocols(engine: Any, dataset: Any, dataset_name: str) -> dict[str, Any]:
    """One feature extraction, three protocols, plus a self-check against torchreid."""
    import torch
    from torchreid import metrics

    loaders = engine.datamanager.test_loader[dataset_name]
    print("\n=> Extracting test features (query then gallery) ...")
    qf, q_pids, q_cams = extract_split_features(engine, loaders["query"])
    gf, g_pids, g_cams = extract_split_features(engine, loaders["gallery"])
    print(f"   query   {tuple(qf.shape)}   gallery {tuple(gf.shape)}")

    q_outfits = outfit_array(dataset, "query", qf.shape[0])
    g_outfits = outfit_array(dataset, "gallery", gf.shape[0])

    results: dict[str, Any] = {}

    # 1. general protocol, torchreid defaults (euclidean on raw features).
    distmat = metrics.compute_distance_matrix(qf, gf, "euclidean").numpy()
    match, junk = general_masks(q_pids, q_cams, g_pids, g_cams)
    results["general"] = report_protocol(
        "general protocol (torchreid standard: euclidean, raw features)", distmat, match, junk
    )

    # Self-check: the same distance matrix through torchreid's own ranker. A disagreement
    # means the in-house ranker used for SC-approx is wrong, and SC-approx has no external
    # result to check it against, so the two seconds are worth spending.
    ref_cmc, ref_map = metrics.evaluate_rank(distmat, q_pids, g_pids, q_cams, g_cams)
    delta = abs(float(ref_map) - results["general"]["mAP"])
    print(
        f"    self-check vs torchreid.metrics.evaluate_rank: mAP {float(ref_map):.2%} "
        f"(delta {delta:.2e}), rank-1 {float(ref_cmc[0]):.2%}"
    )
    results["general"]["torchreid_map"] = float(ref_map)
    results["general"]["torchreid_rank1"] = float(ref_cmc[0])
    results["general"]["selfcheck_map_delta"] = delta
    if delta > 1e-6:
        print("    WARNING: in-house ranker disagrees with torchreid; treat SC-approx as suspect")

    # 2a. L2 + cosine baseline, so the debias delta is not confounded with the metric change.
    qn = torch.nn.functional.normalize(qf, p=2, dim=1)
    gn = torch.nn.functional.normalize(gf, p=2, dim=1)
    dist_cos = metrics.compute_distance_matrix(qn, gn, "cosine").numpy()
    results["l2_cosine"] = report_protocol(
        "L2-normalised cosine baseline (no debias)",
        dist_cos,
        match,
        junk,
        note="the comparison point for camera-debias; same protocol, different metric",
    )

    # 2b. camera-debiased: per-camid mean over the pooled test features, then renormalise.
    pooled = torch.cat([qf, gf], 0)
    pooled_cams = np.concatenate([q_cams, g_cams])
    debiased = camera_debias(pooled, pooled_cams)
    qd, gd = debiased[: qf.shape[0]], debiased[qf.shape[0] :]
    dist_debias = metrics.compute_distance_matrix(qd, gd, "cosine").numpy()
    cam_counts = {int(c): int((pooled_cams == c).sum()) for c in np.unique(pooled_cams)}
    singleton_cams = sorted(c for c, n in cam_counts.items() if n < 2)
    results["camera_debias"] = report_protocol(
        "camera-debiased (per-camera mean subtracted, L2-renormalised, cosine)",
        dist_debias,
        match,
        junk,
        note=f"{len(cam_counts)} cameras; singleton cameras (mean == the feature): "
        f"{singleton_cams or 'none'}",
    )
    results["camera_debias"]["camera_counts"] = cam_counts
    results["camera_debias"]["singleton_cameras"] = singleton_cams

    # 3. SC-approx. Same features, same metric as the general protocol, different masks.
    sc_match, sc_junk = sc_approx_masks(q_pids, q_cams, q_outfits, g_pids, g_cams, g_outfits)
    results["sc_approx"] = report_protocol(
        "SC-approx protocol (in-house definition), not the paper protocol",
        distmat,
        sc_match,
        sc_junk,
        note=(
            "positives are same pid and same outfit_id; same-pid different-outfit gallery "
            "tracklets are removed as junk, not scored as negatives. This number is not "
            "comparable to any published MEVID same-clothes result."
        ),
    )
    return results


# ------------------------------------------------------------------ export
def export_onnx(model: Any, out: Path, opset: int) -> dict[str, Any]:
    """Export the embedding graph, hash it, and prove it reproduces the torch output.

    The model is deep-copied to CPU in eval mode first: OSNet returns the embedding `v`
    only when `self.training` is False (torchreid/models/osnet.py forward), so exporting a
    training-mode module would emit the classifier logits instead and the parity check
    would be comparing the wrong tensor to the wrong tensor.
    """
    import torch

    net = copy.deepcopy(model)
    if hasattr(net, "module"):  # unwrap DataParallel
        net = net.module
    net = net.cpu().eval()

    dummy = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        reference = net(dummy)
    if reference.ndim != 2:
        raise ParityError(
            f"expected a 2-D (batch, dim) embedding in eval mode, got {tuple(reference.shape)}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False keeps the TorchScript exporter, which honours `dynamic_axes` directly.
    # torch >= 2.9 defaults to the torch.export path; pinning it means the emitted graph
    # does not change under a torch bump. The kwarg does not exist before torch 2.5.
    kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(
        net,
        (dummy,),
        str(out),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: BATCH_AXIS}, OUTPUT_NAME: {0: BATCH_AXIS}},
        opset_version=opset,
        **kwargs,
    )
    if not out.is_file():
        raise ParityError(f"torch.onnx.export returned without writing {out}")

    digest = sha256_file(out)
    (out.parent / f"{out.name}.sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")

    # Parity on random input, not the zero dummy: a zero input passes through a broken
    # graph far too easily.
    import onnxruntime as ort

    probe = torch.randn(4, *INPUT_SHAPE[1:], dtype=torch.float32)
    with torch.no_grad():
        torch_out = net(probe).numpy()
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_out = session.run([OUTPUT_NAME], {INPUT_NAME: probe.numpy()})[0]
    if onnx_out.shape != torch_out.shape:
        raise ParityError(f"shape mismatch: torch {torch_out.shape} vs onnx {onnx_out.shape}")
    cos = (torch_out * onnx_out).sum(axis=1) / (
        np.linalg.norm(torch_out, axis=1) * np.linalg.norm(onnx_out, axis=1) + 1e-12
    )
    worst = float(cos.min())
    if worst <= 0.999:
        raise ParityError(
            f"torch/onnxruntime cosine similarity {worst:.6f} <= 0.999 on random input; "
            "the export does not reproduce the model and is not usable"
        )
    print(f"\n=> ONNX export: {out}")
    print(f"   sha256          : {digest}")
    print(f"   embedding dim   : {int(reference.shape[1])}  opset {opset}  dynamic batch")
    print(f"   parity (cos min): {worst:.6f}  (threshold > 0.999)  pass")
    return {
        "path": str(out),
        "sha256": digest,
        "bytes": out.stat().st_size,
        "opset": opset,
        "embedding_dim": int(reference.shape[1]),
        "parity_cosine_min": worst,
    }


# ------------------------------------------------------------------ run
def run(cfg: dict[str, Any], config_path: Path) -> int:
    preflight_imports(cfg)

    import torch
    import torchreid

    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    torchreid.utils.set_random_seed(seed)
    # cudnn determinism is best-effort: it pins algorithm selection, but atomics in some
    # backward kernels stay non-deterministic. The seed is recorded so a run is
    # reproducible in distribution, and that is what is claimed - not bitwise equality.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = str(cfg.get("device", "cuda"))
    use_gpu = device == "cuda" and torch.cuda.is_available()
    if device == "cuda" and not use_gpu:
        print("WARNING: device is 'cuda' but no CUDA device is visible; falling back to CPU")

    print("=" * 78)
    print("v2 re-ID Phase A - OSNet-x1_0 on MEVID (weight_class: prototype)")
    print("=" * 78)
    print(f"config          : {config_path}")
    print(f"started (utc)   : {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"torch           : {torch.__version__}   torchreid: {torchreid.__version__}")
    print(f"device          : {'cuda:' + torch.cuda.get_device_name(0) if use_gpu else 'cpu'}")
    print(f"seed            : {seed}")
    print(f"data root       : {cfg['root']}")
    print(f"out_dir         : {out_dir}")

    dataset_name = str(cfg.get("dataset_name", "mevid"))
    register_mevid(dataset_name)

    datamanager = torchreid.data.VideoDataManager(
        root=str(cfg["root"]),
        sources=dataset_name,
        height=int(cfg.get("height", 256)),
        width=int(cfg.get("width", 128)),
        transforms=list(cfg.get("transforms", ["random_flip", "random_erase"])),
        batch_size_train=int(cfg["batch_train"]),
        batch_size_test=int(cfg.get("batch_test", 8)),
        workers=int(cfg.get("workers", 4)),
        num_instances=int(cfg.get("num_instances", 4)),
        train_sampler=str(cfg.get("train_sampler", "RandomIdentitySampler")),
        seq_len=int(cfg["seq_len"]),
        sample_method=str(cfg["sample_method"]),
        use_gpu=use_gpu,
    )
    num_classes = datamanager.num_train_pids
    print(f"train identities: {num_classes}   train cameras: {datamanager.num_train_cams}")

    # torchreid 1.4.0 applies ONE sample_method to the train AND test datasets
    # (VideoDataManager passes it to every init_video_dataset call), so a `random` train
    # setting makes every evaluation a fresh random frame draw: the same weights measure a
    # different mAP on two consecutive passes, so no evaluation reproduces. Training wants
    # `random`, a new view of each tracklet every epoch, and evaluation wants a fixed,
    # span-covering draw, so the test datasets are switched after construction. Without
    # this the mid-run curve, the final test and the post-training protocols all measure
    # slightly different things and none of them reproduces.
    test_sample_method = str(cfg.get("test_sample_method", "evenly"))
    for split in ("query", "gallery"):
        datamanager.test_loader[dataset_name][split].dataset.sample_method = test_sample_method
    print(
        f"sample_method   : train={cfg['sample_method']}  test={test_sample_method} "
        "(test forced deterministic)"
    )

    loss_name = str(cfg.get("loss", "softmax"))
    # OSNet's forward returns the logits alone under `softmax` and (logits, embedding)
    # under `triplet` while training, and the embedding alone in eval mode either way; the
    # engine chosen below expects the matching shape.
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=num_classes,
        loss=loss_name,
        pretrained=bool(cfg.get("pretrained", True)),
    )
    # POLICY.md, rule 8: the backbone pretrain data is part of the provenance record, so
    # where the init came from and what it hashes to are logged, not assumed. torchreid pulls
    # `osnet_x1_0_imagenet.pth` from its model zoo into $TORCH_HOME/checkpoints on first
    # use; an absent file here means the run started from random weights.
    init_path = imagenet_init_path()
    if init_path.is_file():
        init_sha = sha256_file(init_path)
        print(f"imagenet init   : {init_path}")
        print(f"imagenet sha256 : {init_sha}")
    else:
        init_sha = None
        print(f"imagenet init   : not found at {init_path}; model is not ImageNet-initialised")
    if use_gpu:
        model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim=str(cfg.get("optim", "amsgrad")),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 5e-4)),
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler=str(cfg.get("lr_scheduler", "single_step")),
        stepsize=cfg.get("stepsize", 20),
        gamma=float(cfg.get("gamma", 0.1)),
        max_epoch=int(cfg["epochs"]),
    )

    curves_path = out_dir / "curves.csv"
    engine_kwargs: dict[str, Any] = {}
    if loss_name == "triplet":
        engine_kwargs = {
            "margin": float(cfg.get("margin", 0.3)),
            "weight_t": float(cfg.get("weight_t", 1.0)),
            "weight_x": float(cfg.get("weight_x", 1.0)),
        }
    print(f"loss            : {loss_name}   engine kwargs: {engine_kwargs or 'none'}")
    engine = build_engine_class(loss_name)(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        use_gpu=use_gpu,
        label_smooth=bool(cfg.get("label_smooth", True)),
        pooling_method=str(cfg.get("pooling_method", "avg")),
        curves_path=curves_path,
        save_dir=out_dir,
        keep_last=int(cfg.get("keep_last", 2)),
        **engine_kwargs,
    )

    # torchreid 1.4.0's Engine._evaluate truncates the CMC curve to min(50, num_gallery)
    # and then prints cmc[r-1] for every r in `ranks` without a bounds check, so a rank
    # larger than the gallery is an IndexError in the middle of a finished evaluation.
    # MEVID's gallery is 1,438 tracklets so the default is safe, but the ranks are read
    # from config anyway and clamped, because a subset run should not be able to throw
    # away an epoch's training on a print statement.
    ranks = [int(r) for r in cfg.get("ranks", [1, 5, 10, 20])]
    gallery_size = len(datamanager.test_loader[dataset_name]["gallery"].dataset)
    safe_ranks = [r for r in ranks if r <= min(50, gallery_size)] or [1]
    if safe_ranks != ranks:
        print(f"note: ranks {ranks} clamped to {safe_ranks} for a {gallery_size}-tracklet gallery")

    # Two-stepped transfer learning, torchreid's own OSNet fine-tuning recipe: for the first
    # `fixbase_epoch` epochs only the layers named in `open_layers` train and the pretrained
    # trunk stays frozen in eval mode, so the fresh classifier learns on ImageNet features
    # before the triplet term can move the trunk. Without it the first run of this recipe
    # collapsed by epoch 3: loss_t at the margin, loss_x at ln(num_classes), and rank-1
    # equal to rank-20 at the epoch-10 evaluation, which is a constant feature.
    open_layers = cfg.get("open_layers")
    engine.run(
        save_dir=str(out_dir),
        max_epoch=int(cfg["epochs"]),
        print_freq=int(cfg.get("print_freq", 20)),
        fixbase_epoch=int(cfg.get("fixbase_epoch", 0)),
        open_layers=list(open_layers) if open_layers else None,
        eval_freq=int(cfg.get("eval_freq", -1)),
        test_only=False,
        dist_metric=str(cfg.get("dist_metric", "euclidean")),
        ranks=safe_ranks,
    )
    engine._flush_pending()
    print(f"\n=> curves written: {curves_path}")
    if engine.best_epoch > 0:
        print(f"=> best test mAP {engine.best_map:.2%} at epoch {engine.best_epoch}")
        print(f"   best checkpoint: {out_dir / 'model' / 'model-best.pth.tar'}")

    dataset = datamanager.test_loader[dataset_name]["query"].dataset
    protocols = run_all_protocols(engine, dataset, dataset_name)

    summary = {
        "artefact": "reid-mevid-a1",
        "schema_version": 1,
        "finished_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": str(config_path),
        "weight_class": "prototype",
        "seed": seed,
        "epochs": int(cfg["epochs"]),
        "recipe": {
            key: cfg.get(key)
            for key in (
                "loss",
                "batch_train",
                "num_instances",
                "seq_len",
                "sample_method",
                "test_sample_method",
                "lr",
                "weight_decay",
                "fixbase_epoch",
                "open_layers",
                "lr_scheduler",
                "stepsize",
                "gamma",
                "margin",
                "weight_t",
                "weight_x",
                "label_smooth",
                "pooling_method",
            )
        },
        "torch": str(torch.__version__),
        "torchreid": str(torchreid.__version__),
        "num_train_pids": int(num_classes),
        "imagenet_init": {"path": str(init_path), "sha256": init_sha},
        "best": {"map": engine.best_map, "epoch": engine.best_epoch},
        "protocols": protocols,
        "onnx": None,
        "caveat": (
            "Prototype weights (POLICY.md, rule 6). Internal only; never quoted as clinical "
            "performance; SC-approx is an in-house protocol and is not comparable to any "
            "published MEVID same-clothes number."
        ),
    }
    summary_path = out_dir / "eval_summary.json"

    def write_summary() -> None:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\n=> summary written: {summary_path}")

    # The evaluation is written before the export: a failed parity check must not also
    # destroy the record of what the training run measured. On failure the summary keeps
    # the numbers, records why the export was rejected, and the process still exits
    # non-zero, because the ONNX file is not usable and nothing may declare success.
    write_summary()
    onnx_path = out_dir / str(cfg.get("onnx_name", "osnet_x1_0_mevid_a1.onnx"))
    # export_onnx records the path, digest, size, opset, embedding dimension and parity
    # value, and nothing else. The `onnx.exported_from` block in the shipped summary
    # (final_models/reid/eval_summary.json) was written by the separate re-export from
    # model-best.pth.tar after this in-process call had failed on a missing `onnx` package;
    # a run of this script writes the six fields above.
    try:
        summary["onnx"] = export_onnx(model, onnx_path, int(cfg.get("opset", 17)))
    except ParityError as exc:
        summary["onnx"] = {"path": str(onnx_path), "error": str(exc)}
        write_summary()
        raise
    write_summary()
    print("\nPROTOTYPE weights. Not clinical performance. Not deployable without counsel.")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_reid_mevid_a1.py",
        description=(
            "Train OSNet-x1_0 on MEVID from an ImageNet-only init, evaluate under three "
            "protocols, and export a parity-checked ONNX embedding graph."
        ),
    )
    parser.add_argument("--config", required=True, help="path to the run config YAML")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    try:
        return run(load_config(config_path), config_path)
    except ConfigError as exc:
        # Covers the preflight too: a missing tensorboard or onnxruntime is a configuration
        # problem with the environment, and it exits before any GPU time is spent.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except ParityError as exc:
        print(f"onnx parity failure: {exc}", file=sys.stderr)
        return EXIT_PARITY


if __name__ == "__main__":
    raise SystemExit(main())
