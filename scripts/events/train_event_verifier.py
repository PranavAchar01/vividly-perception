"""Train an interim event verifier on windowed track sequences.

Runs where torch lives (the pod's cuda venv; these models are tiny and train in
minutes on CPU too). Consumes extraction directories (real: extract_track_sequences,
synthetic: make_synth_convulsion) through vividly.verifier.features, trains a small
temporal CNN, and exports ONNX + a model card. The exported graph takes
(B, n_steps, n_channels) float32 and returns (B,) logits.

    uv run python scripts/events/train_event_verifier.py \
        --config configs/events/train_fall_verifier_v3.yaml

Honesty rails (POLICY.md rules 4, 6, 8 and 10):
  * vividly_weight_class = "prototype" is stamped into the records and the card;
    internal demonstration only, never quoted as clinical detection performance.
  * The holdout here is split by video key (windows of one video never straddle the
    split), but it is still the training distribution; the real test numbers for the
    fall verifier come only from the held-out datasets (ur_fall, multicam_fall,
    registry-enforced test_only) and are produced by a separate eval run.
  * Every metric row carries the checkpoint sha256, as POLICY.md, rule 10 requires.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vividly.verifier.augment import make_augmenter, make_geo_augmenter  # noqa: E402
from vividly.verifier.features import (  # noqa: E402
    WindowSource,
    WindowSpec,
    _event_track,
    channel_names,
    load_rows,
    normalize_events,
    tracks_from_rows,
    windows_from_extraction,
)

WEIGHT_CLASS = "prototype"  # POLICY.md, rule 4: interim event models are always prototype-class

# Negative-window classes, named exactly as the push-2 OOF diagnosis classified them
# (research/seizure-push-2-out-of-fold-gate.md item 2). A window is `positive` whenever its
# label is 1; otherwise:
#   seizure_free    (a) the recording carries no event at all: pat04__free /
#                       pat04__no-Sz2P in the rotation pool. 75.4% of the top-decile
#                       false-alarm mass sits here, which is what authorized weighting.
#   owner_offevent  (b) the event-owning track, outside the event.
#   nonowner_track  (c) any other track on a recording that does carry an event.
#   forced_negative (d) a source with `force_label` (the AJILE12 diet). Kept separate
#                       on purpose: those metas also carry empty events_t, and folding
#                       them into (a) would silently reweight the whole diet.
NEG_CLASSES: tuple[str, ...] = (
    "seizure_free",
    "owner_offevent",
    "nonowner_track",
    "forced_negative",
)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------- data
def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("task", "window", "sources", "train", "out_dir"):
        if key not in cfg:
            raise SystemExit(f"config {path} is missing `{key}`")
    return cfg


def _source_event_map(
    src_dir: Path, event_keys: tuple[str, str]
) -> dict[str, tuple[bool, int | None]]:
    """key -> (recording carries an event, id of the event-owning track).

    Resolves the event exactly the way windows_from_extraction does (events_t wins,
    else the two event_keys fields) and picks the owner with the same
    features._event_track the labeller uses, so the classes below cannot drift from
    the labels. Only called when `negative_class_weights` is configured.
    """
    out: dict[str, tuple[bool, int | None]] = {}
    for meta_path in sorted((src_dir / "meta").glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("events_t"):
            event: Any = [(float(e0), float(e1)) for e0, e1 in meta["events_t"]]
        else:
            e0, e1 = meta.get(event_keys[0]), meta.get(event_keys[1])
            event = None if e0 is None or e1 is None else (float(e0), float(e1))
        events = normalize_events(event)
        owner = None
        if events:
            tracks = tracks_from_rows(load_rows(src_dir / "sequences" / f"{meta['key']}.jsonl"))
            owner = _event_track(tracks, events)
        out[str(meta["key"])] = (bool(events), owner)
    return out


def build_dataset(
    cfg: dict[str, Any], *, classify_negatives: bool = False, with_sources: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, list[Any] | None]:
    """(X, y, groups, neg_class, sources): windows, labels, subject group, class, rows.

    ``neg_class`` is None unless ``classify_negatives`` and ``sources`` is None unless
    ``with_sources``; the default path does not read a single extra byte, so a run
    with neither `negative_class_weights` nor `geo_augment` is the same code it
    always was.

    ``sources`` (fall push v3 arm F3) is a per-row list holding the WindowSource of
    positive windows and None everywhere else: only positives are augmented, and
    keeping a copy of every negative's raw rows would cost about 4 times the memory
    for nothing.
    """
    spec = WindowSpec(**cfg["window"])
    xs, ys, gs, cs = [], [], [], []
    srcs: list[Any] = []
    for source in cfg["sources"]:
        src_dir = Path(source["dir"])
        if not src_dir.is_absolute():
            src_dir = REPO_ROOT / src_dir
        event_keys = tuple(source.get("event_keys", ("fall_start_t", "fall_end_t")))
        locations = source.get("locations")
        wins = windows_from_extraction(
            src_dir,
            spec,
            event_keys=event_keys,
            locations=tuple(locations) if locations else None,
            no_event_only=bool(source.get("no_event_only", False)),
            with_sources=with_sources,
        )
        force = source.get("force_label")
        ev_map: dict[str, tuple[bool, int | None]] = {}
        if classify_negatives and force is None:
            ev_map = _source_event_map(src_dir, event_keys)
        for w in wins:
            xs.append(w.x)
            label = int(force) if force is not None else w.label
            ys.append(label)
            if with_sources:
                srcs.append(w.source if label == 1 else None)
            if classify_negatives:
                if label == 1:
                    cs.append("positive")
                elif force is not None:
                    cs.append("forced_negative")
                else:
                    has_event, owner = ev_map[w.key]
                    if not has_event:
                        cs.append("seizure_free")
                    elif owner is not None and w.track_id == owner:
                        cs.append("owner_offevent")
                    else:
                        cs.append("nonowner_track")
            # Group by subject wherever the extraction records one, never by
            # recording. Two recordings of one patient share a bed, a room, a camera
            # and a body, so a recording-level holdout validates the model on someone
            # it has already trained on. Measured consequence (2026-08-28): val AUC
            # ran anti-correlated with held-out performance, with v2 `07c399be`, the
            # model of record, on 0.772 val while v6 `10cdbe78`, rejected as worse on
            # the exam, scored 0.983.
            gs.append(f"{source['dir']}::{w.location or w.key}")
        print(
            f"source {source['dir']}: {len(wins)} windows "
            f"({sum(1 for w in wins if w.label == 1)} positive before force_label)",
            flush=True,
        )
    if not xs:
        raise SystemExit("no windows produced; check the sources and extraction dirs")
    return (
        np.stack(xs).astype(np.float32),
        np.array(ys, dtype=np.int64),
        np.array(gs),
        np.array(cs) if classify_negatives else None,
        srcs if with_sources else None,
    )


def negative_weights(neg_class: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Per-window loss weight: `weights[class]` where named, 1.0 everywhere else.

    Weighting is a train-side lever; the caller applies this to the training rows
    only, and the val surface stays the unweighted patient-held-out AUC the arm-B grid
    was selected on, so a weighted arm and the baseline are compared on the same
    ruler (research/seizure-push-2-protocol.md item 2).
    """
    unknown = sorted(set(weights) - set(NEG_CLASSES))
    if unknown:
        raise SystemExit(
            f"negative_class_weights names unknown classes {unknown}; known: {list(NEG_CLASSES)}"
        )
    w = np.ones(len(neg_class), dtype=np.float32)
    for name, alpha in weights.items():
        alpha = float(alpha)
        if alpha <= 0:
            raise SystemExit(f"negative_class_weights[{name}] must be > 0, got {alpha}")
        w[neg_class == name] = alpha
    return w


def grouped_split(
    groups: np.ndarray,
    holdout_fraction: float,
    seed: int,
    holdout_groups: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train_mask, val_mask) split by group (subject), never by window.

    ``holdout_groups`` names the val groups outright: one explicit leave-one-subject-out
    fold. Prefer it over the random draw whenever the group count is small, because with
    four patients a random 20% holdout is one patient chosen by a seed, so the identity
    of the patient the model was judged on becomes an undocumented function of the seed. A group is
    matched if it equals the name or ends in "::<name>", so a config names the subject
    without repeating the source directory.
    """
    uniq = np.unique(groups)
    if holdout_groups:
        wanted = set(holdout_groups)
        val_groups = {g for g in uniq.tolist() if g in wanted or g.split("::")[-1] in wanted}
        missing = wanted - {g.split("::")[-1] for g in val_groups} - val_groups
        if missing:
            raise SystemExit(f"holdout_groups not present in the data: {sorted(missing)}")
        if len(val_groups) == len(uniq):
            raise SystemExit("holdout_groups covers every group; nothing left to train on")
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        n_val = max(1, round(len(uniq) * holdout_fraction))
        val_groups = set(uniq[:n_val].tolist())
    val_mask = np.array([g in val_groups for g in groups])
    return ~val_mask, val_mask


# --------------------------------------------------------------------- metrics
def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pr_at(y: np.ndarray, score: np.ndarray, thr: float) -> dict[str, float | int]:
    pred = score >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "threshold": thr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    import torch
    from torch import nn

    class TemporalVerifier(nn.Module):
        """Small temporal CNN: (B, T, C) -> (B,) logits. Mean+max pooled so the event
        can sit anywhere in the window."""

        def __init__(self, n_channels: int, hidden: int, blocks: int, dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            c = n_channels
            for _ in range(blocks):
                layers += [
                    nn.Conv1d(c, hidden, kernel_size=5, padding=2),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
                c = hidden
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(hidden * 2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.body(x.transpose(1, 2))
            z = torch.cat([z.mean(dim=-1), z.amax(dim=-1)], dim=1)
            return self.head(z).squeeze(-1)

    tcfg = cfg["train"]
    seed = int(tcfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    ncw = cfg.get("negative_class_weights") or {}
    gcfg = cfg.get("geo_augment")
    if ncw and gcfg:
        raise SystemExit(
            "negative_class_weights and geo_augment are not supported together: the "
            "weight vector is per original row and geo_augment appends rows"
        )
    X, y, groups, neg_class, sources = build_dataset(
        cfg, classify_negatives=bool(ncw), with_sources=bool(gcfg)
    )
    train_mask, val_mask = grouped_split(
        groups,
        float(cfg.get("holdout_fraction", 0.2)),
        seed,
        holdout_groups=cfg.get("holdout_locations"),
    )
    print(
        f"split: {len(np.unique(groups[train_mask]))} train groups / "
        f"{len(np.unique(groups[val_mask]))} val groups "
        f"({sorted({g.split('::')[-1] for g in np.unique(groups[val_mask])})} held out)",
        flush=True,
    )
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(
        f"dataset: {len(y)} windows ({n_pos} pos / {n_neg} neg), "
        f"{len(np.unique(groups))} videos; train {int(train_mask.sum())} / "
        f"val {int(val_mask.sum())}",
        flush=True,
    )

    device = tcfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = cfg.get("model", {})
    model = TemporalVerifier(
        # From the data, not len(CHANNELS): window.scene_channels appends columns.
        n_channels=int(X.shape[2]),
        hidden=int(mcfg.get("hidden", 64)),
        blocks=int(mcfg.get("blocks", 3)),
        dropout=float(mcfg.get("dropout", 0.1)),
    ).to(device)

    pos_weight = tcfg.get("pos_weight", "auto")
    if pos_weight == "auto":
        pos_weight = max(0.25, min(10.0, n_neg / max(1, n_pos)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    epochs = int(tcfg.get("epochs", 40))
    batch = int(tcfg.get("batch", 256))

    Xtr = X[train_mask]  # the numpy copy the augmenter redraws from each epoch
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(y[train_mask]).float()
    Xv = torch.from_numpy(X[val_mask]).to(device)
    yv = y[val_mask]

    # ---- lever 1: per-window loss weights on a named negative class (default OFF).
    # The loss is normalized by the batch's mean weight, so alpha is a pure
    # RE-weighting: total gradient magnitude (and hence the effective learning rate)
    # stays where the unweighted baseline put it, and the ablation measures the
    # reweighting rather than a hidden LR change.
    wt: torch.Tensor | None = None
    criterion_none = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), device=device), reduction="none"
    )
    weight_summary: dict[str, Any] | None = None
    if ncw:
        w_all = negative_weights(neg_class, ncw)
        wt = torch.from_numpy(w_all[train_mask])
        train_classes = neg_class[train_mask]
        weight_summary = {
            "weights": {k: float(v) for k, v in ncw.items()},
            "loss_form": "(elementwise_bce * w).sum() / w.sum() per batch",
            "class_counts_all": {
                c: int((neg_class == c).sum())
                for c in ("positive", *NEG_CLASSES)
                if int((neg_class == c).sum())
            },
            "class_counts_train": {
                c: int((train_classes == c).sum())
                for c in ("positive", *NEG_CLASSES)
                if int((train_classes == c).sum())
            },
            "n_weighted_train_windows": int((w_all[train_mask] != 1.0).sum()),
            "mean_train_weight": round(float(w_all[train_mask].mean()), 5),
        }
        print(f"negative_class_weights: {json.dumps(weight_summary)}", flush=True)

    # ---- lever 2: track-space augmentation of train rows only (default off).
    # Built after grouped_split and fed only Xtr, so no augmented view of a holdout
    # window can exist (the augmentation scope clause of POLICY.md, rule 4).
    acfg = cfg.get("augment")
    if acfg:
        acfg = dict(acfg)
        acfg.setdefault("seed", seed)
    augmenter = make_augmenter(acfg, channel_names(WindowSpec(**cfg["window"])))
    if augmenter is not None:
        print(f"augment: {json.dumps(acfg)} (train rows only, per-epoch draws)", flush=True)

    # ---- lever 3: camera-geometry augmentation (fall push v3 arm F3, default off).
    # Built from the raw rows of the train positive windows only, selected after
    # grouped_split so no augmented view of a holdout window can exist, and it appends
    # k rows per positive rather than replacing rows, which is what the protocol asks
    # for with K augmented draws per positive window. The val surface is untouched, so
    # the arm and the F0 control are read on one ruler.
    geo = None
    geo_summary: dict[str, Any] | None = None
    if gcfg:
        gcfg = dict(gcfg)
        gcfg.setdefault("seed", seed)
        wspec = WindowSpec(**cfg["window"])
        assert sources is not None
        pos_idx = np.flatnonzero(train_mask & (y == 1))
        missing = [i for i in pos_idx.tolist() if sources[i] is None]
        if missing:
            raise SystemExit(f"{len(missing)} train positives carry no raw rows")
        geo_sources: list[WindowSource] = [sources[i] for i in pos_idx.tolist()]
        geo = make_geo_augmenter(gcfg, wspec, geo_sources, X[pos_idx])
    if geo is not None:
        # Plumbing guard, run before a single epoch: every source window must
        # re-featurize to the byte-identical row the dataset already holds. If it does
        # not, the raw rows are not the ones behind X and every delta below would be
        # measuring the plumbing instead of the camera.
        bad = geo.verify_identity()
        if bad:
            raise SystemExit(
                f"{bad}/{len(geo.sources)} geo sources do not re-featurize to their "
                "recorded window; refusing to train on a mismatched augmentation"
            )
        geo_summary = {
            "config": gcfg,
            "n_source_windows": len(geo.sources),
            "n_extra_rows_per_epoch": len(geo.sources) * int(geo.spec.k),
            "applies_to": "train positives only, appended (labels all 1)",
            "identity_refeaturize_mismatches": 0,
        }
        print(f"geo_augment: {json.dumps(geo_summary)}", flush=True)

    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # The extra camera-geometry rows are all positives (their sources are), so the
    # label vector for them is a constant the augmenter never has to be told.
    yt_full = yt
    if geo is not None:
        yt_full = torch.cat([yt, torch.ones(len(geo.sources) * int(geo.spec.k))])

    best_auc, best_state, history = -1.0, None, []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        if augmenter is not None:
            # Fresh online draws every epoch, from the augmenter's own generator;
            # the torch RNG stream below is untouched, so the batch order is the
            # baseline's batch order.
            Xt = torch.from_numpy(augmenter.augment_epoch(Xtr, epoch))
        if geo is not None:
            base = Xt if augmenter is not None else torch.from_numpy(Xtr)
            Xt = torch.cat([base, torch.from_numpy(geo.augment_epoch(epoch))])
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            xb, yb = Xt[idx].to(device), yt_full[idx].to(device)
            optimizer.zero_grad()
            if wt is None:
                loss = criterion(model(xb), yb)
            else:
                wb = wt[idx].to(device)
                elem = criterion_none(model(xb), yb)
                loss = (elem * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(Xv)).cpu().numpy()
        auc = roc_auc(yv, scores)
        row = {
            "epoch": epoch,
            "train_loss": round(total / max(1, len(Xt)), 5),
            "val_auc": None if auc is None else round(auc, 5),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().to("cpu")

    ckpt_path = out_dir / "verifier.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg}, ckpt_path)
    onnx_path = out_dir / "verifier.onnx"
    dummy = torch.zeros(1, X.shape[1], X.shape[2], dtype=torch.float32)
    export_kwargs: dict[str, object] = {}
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6):
        # The torch>=2.6 default (dynamo) needs onnxscript, which none of the declared
        # environments in pyproject.toml carries. The legacy exporter handles this tiny
        # graph fine everywhere.
        export_kwargs["dynamo"] = False
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["windows"],
        output_names=["logits"],
        dynamic_axes={"windows": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        **export_kwargs,
    )
    onnx_sha = sha256_file(onnx_path)

    with torch.no_grad():
        val_logits = model(torch.from_numpy(X[val_mask])).numpy()
    val_scores = 1.0 / (1.0 + np.exp(-val_logits))
    # Temperature calibration, fitted only on the train-side holdout val (never any
    # held-out test set): a grid search for the T minimizing val NLL of
    # sigmoid(logit/T). Monotonic, so ranking metrics (AUC and PR) are untouched; this
    # exists to turn score-collapsed logits (the finalv1real failure mode,
    # research/event-verifier-v1-record.md) into thresholdable margins.
    temperature = 1.0
    degenerate_fit = False
    if len(val_logits) and 0 < yv.sum() < len(yv):
        yv_f = yv.astype(np.float64)
        best_nll = np.inf
        # Bounded away from the float-saturation regime: monotonic and AUC-neutral
        # is only true in exact math. Below T about 0.25 the division drives |logit|/T
        # past sigmoid's float range, scores collapse to exactly 0.0 and 1.0, and every
        # ranking metric downstream is destroyed by ties (the v3 T=0.02 incident,
        # 2026-08-27). A fit that still lands on a boundary is a val set too separable
        # to calibrate honestly: keep T=1.0 and flag it instead of shipping a boundary
        # artifact.
        grid = np.geomspace(0.25, 8.0, 141)
        for t_cand in grid:
            p = 1.0 / (1.0 + np.exp(-val_logits / t_cand))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            nll = float(-(yv_f * np.log(p) + (1 - yv_f) * np.log(1 - p)).mean())
            if nll < best_nll:
                best_nll, temperature = nll, float(t_cand)
        if temperature in (float(grid[0]), float(grid[-1])):
            degenerate_fit, temperature = True, 1.0
    report = {
        "task": cfg["task"],
        "vividly_weight_class": WEIGHT_CLASS,
        "disclosure": (
            "Interim prototype event verifier, admitted under the acted-event "
            "carve-out of POLICY.md, rule 4. Trained on public acted-fall footage, on "
            "synthetic motion, or on both. Internal demonstration only; no number here "
            "is clinical detection performance. Real test numbers come only from the "
            "registry-enforced held-out sets."
        ),
        "git_commit": _git_commit(),
        "config": cfg,
        # From the spec, never hand-assembled: this list is what names the model's
        # input columns, and the old copy here silently omitted RHYTHM_CHANNELS.
        "channels": list(channel_names(WindowSpec(**cfg["window"]))),
        "calibration": {
            "temperature": round(temperature, 4),
            "fitted_on": "train_holdout_val",
            "degenerate_fit": degenerate_fit,
            "note": "score = sigmoid(logit / temperature); monotonic in exact math, "
            "grid bounded [0.25, 8] against float saturation",
        },
        "n_windows": len(y),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_videos": len(np.unique(groups)),
        "resolved_pos_weight": float(pos_weight),
        # Both levers record themselves; null is the default-OFF path (the same code
        # the 12 conv_rotB baselines ran).
        "negative_class_weighting": weight_summary,
        "augment": acfg if augmenter is not None else None,
        "geo_augment": (
            None
            if geo is None or geo_summary is None
            else {**geo_summary, "n_fallback_rows_total": geo.n_fallback}
        ),
        "val_auc_best": None if best_auc < 0 else round(best_auc, 5),
        "val_operating_points": [pr_at(yv, val_scores, thr) for thr in (0.5, 0.8, 0.9)],
        "checkpoint": {"file": ckpt_path.name, "sha256": sha256_file(ckpt_path)},
        "onnx": {"file": onnx_path.name, "sha256": onnx_sha},
        "history": history,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "val_auc_best": report["val_auc_best"],
                "onnx_sha256": onnx_sha,
                "out_dir": str(out_dir),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
