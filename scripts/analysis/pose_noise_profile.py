#!/usr/bin/env python3
"""Per-joint confidence and jitter profile of the v2pose track trees (report-only).

    uv run python scripts/analysis/pose_noise_profile.py --trees-glob <tree glob>

Answers one question with recorded data only: for each COCO-17 joint, how confident is
the v2 pose model of record on this project's own footage, and how much does the joint move
between adjacent frames when it is confident? No model is loaded, nothing is scored,
nothing is trained; this reads the `kp` blocks that `scripts/events/extract_track_sequences.py` already wrote
under `emit_keypoints: true` and tabulates them. The record is
research/pose-noise-profile.md.

Why it exists. The limb-channel gate finding (research/seizure-limb-channel-gate.md)
measured one number on one tree, wrist+ankle confidence median 0.305 pooled over
wu_sahzu_tracks_v2kp, and that number settled a feature. It was measured on the earlier
pose weights and it pooled seven joints together. Every downstream gate (LIMB_CONF_MIN,
WindowSpec.kp_conf_min, TrackConfig.kp_conf_min, all defaulting near 0.2-0.3) is a
threshold on a distribution nobody had written down per joint. This writes it down.

Definitions (fixed here; each one is a choice, so each one is stated).

  confidence     the third element of a row's `kp[j]` entry, exactly as the extraction
                 rounded it (3 decimals). Non-finite entries are dropped and counted;
                 they are the estimator returning nothing, not a zero-confidence joint.
  frac >= gate   share of a joint's observations with conf >= `--conf-gate` (default
                 0.3 = features.KP_CONF_MIN = the extraction configs' tracks.kp_conf_min).
                 ">=" because that is the comparison posture_angle_deg and KP_CHANNELS
                 already apply.
  jitter         || (x, y)_{f+1} - (x, y)_f || / h, over rows of the same track whose
                 frame indices differ by exactly 1, where both endpoints clear the gate
                 and both boxes have h > 0. h is the mean of the two rows' box heights,
                 so the measure is scale-free and a person walking toward the camera does
                 not inflate it. Reported per frame; the per-second column multiplies by
                 the recording's fps, which is what makes trees at 15/20/25/30 fps
                 comparable. Pairs spanning a tracker gap (delta-frame > 1) are excluded;
                 across a gap the displacement is a re-acquisition, not jitter.
  kpc            the row's own scalar `kpc` field (mean confidence over all 17 entries,
                 as the extraction computed it), the channel the verifiers actually see.
  angle_valid    share of rows whose `angle` field is non-null, i.e. the share of rows
                 where posture_angle_deg found hips and shoulders above the gate. This is
                 the row-level population behind the `angle_valid` feature channel.

Quantiles come from fixed-width histograms (confidence 1/1000 of a unit, jitter 1/10000 of
a box-height) rather than from stored samples: the pooled corpus is ~3M rows x 17 joints
and the bin width is far below any difference this document reads. Values above the
jitter histogram's ceiling (2 box-heights per frame, 20 per second) land in an overflow bin
and are reported as ">ceiling".

Exam safety. This is data profiling, not model evaluation. No verifier, detector or pose
graph is executed, no window is scored, and no held-out metric is produced, so rows from
the WU-SAHZU exam patients are in scope as statistics the same way any other row is. The
per-tree table names how many recordings each tree contributes; the WU tree's exam
patients are listed in the doc rather than silently pooled.

What this instrument can no longer reproduce. The `*_tracks_v2pose` trees the two
`--trees-glob` and `--contrast-tree` defaults name came from extraction configurations
that are superseded and are not in this repository; the seven committed extraction
configurations are the `configs/events/extract_*_v2c.yaml` set, which writes
`*_tracks_v2c`. A run therefore needs the tree paths passed explicitly, and section 3's
pose-weight provenance column for the three trees without an `extraction.json` is
recorded in research/pose-noise-profile.md, not recomputed here.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# COCO-17, in the order the extraction writes them.
JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
# Groups used by the per-tree summary; every joint belongs to exactly one.
GROUPS: dict[str, tuple[int, ...]] = {
    "face (nose/eyes/ears)": (0, 1, 2, 3, 4),
    "shoulders": (5, 6),
    "elbows": (7, 8),
    "wrists": (9, 10),
    "hips": (11, 12),
    "knees": (13, 14),
    "ankles": (15, 16),
}
# The population the limb-channel gate finding recorded its 0.305 median over.
LIMB_JOINTS: tuple[int, ...] = (9, 10, 15, 16)
# WU-SAHZU patients allocated exam-side: the focal group (pat06/08/10/12) and the locked
# convulsive pair (pat03, pat07). Their rows are in scope here because this instrument
# produces statistics and never a score; naming them keeps that visible in the record.
EXAM_LOCATIONS: frozenset[str] = frozenset({"pat03", "pat06", "pat07", "pat08", "pat10", "pat12"})

CONF_BINS = 1000  # confidence resolution 0.001 over [0, 1]
JIT_BINS = 20000  # jitter resolution 0.0001 box-heights over [0, JIT_MAX]
JIT_MAX = 2.0  # a joint that moves more than 2 box-heights in one frame is an overflow
JIT_S_MAX = 20.0  # the same measure per second; ceiling in box-heights/s


class Hist:
    """Fixed-width histogram over [0, hi] with an overflow bin at the end."""

    def __init__(self, nbins: int, hi: float, shape: tuple[int, ...] = ()) -> None:
        self.nbins, self.hi = nbins, hi
        self.counts = np.zeros((*shape, nbins + 1), dtype=np.int64)

    def add(self, values: np.ndarray, row: int | None = None) -> None:
        v = values[np.isfinite(values)]
        if not len(v):
            return
        idx = np.minimum((v / self.hi * self.nbins).astype(np.int64), self.nbins)
        idx = np.maximum(idx, 0)
        target = self.counts if row is None else self.counts[row]
        np.add.at(target, idx, 1)

    def __iadd__(self, other: Hist) -> Hist:
        self.counts += other.counts
        return self

    def total(self, row: int | None = None) -> int:
        c = self.counts if row is None else self.counts[row]
        return int(c.sum())

    def quantile(self, q: float, row: int | None = None) -> float | None:
        """Bin-centre quantile; math.inf when the quantile lands in the overflow bin."""
        c = self.counts if row is None else self.counts[row]
        n = int(c.sum())
        if n == 0:
            return None
        cum = np.cumsum(c)
        i = int(np.searchsorted(cum, max(1.0, q * n)))
        if i >= self.nbins:
            return math.inf
        return (i + 0.5) * self.hi / self.nbins

    def frac_ge(self, x: float, row: int | None = None) -> float | None:
        c = self.counts if row is None else self.counts[row]
        n = int(c.sum())
        if n == 0:
            return None
        first = math.ceil(x / self.hi * self.nbins)
        return float(c[min(first, self.nbins) :].sum()) / n


class TreeStats:
    def __init__(self, name: str) -> None:
        self.name = name
        self.conf = Hist(CONF_BINS, 1.0, (len(JOINTS),))
        self.jit = Hist(JIT_BINS, JIT_MAX, (len(JOINTS),))
        self.jit_s = Hist(JIT_BINS, JIT_S_MAX, (len(JOINTS),))  # box-heights/second
        self.kpc = Hist(CONF_BINS, 1.0)
        self.n_rows = 0
        self.n_rows_no_kp = 0
        self.n_tracks = 0
        self.n_seqs = 0
        self.n_angle_valid = 0
        self.n_kp_nonfinite = 0
        self.fps: list[float] = []
        self.locations: set[str] = set()

    def merge(self, other: TreeStats) -> None:
        self.conf += other.conf
        self.jit += other.jit
        self.jit_s += other.jit_s
        self.kpc += other.kpc
        self.n_rows += other.n_rows
        self.n_rows_no_kp += other.n_rows_no_kp
        self.n_tracks += other.n_tracks
        self.n_seqs += other.n_seqs
        self.n_angle_valid += other.n_angle_valid
        self.n_kp_nonfinite += other.n_kp_nonfinite
        self.fps += other.fps
        self.locations |= other.locations


def scan_sequence(rows: list[dict[str, Any]], st: TreeStats, gate: float, fps: float) -> None:
    """Fold one sequence file's rows into the tree's histograms."""
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        st.n_rows += 1
        if r.get("angle") is not None:
            st.n_angle_valid += 1
        k = r.get("kpc")
        if k is not None:
            st.kpc.add(np.array([float(k)]))
        if not r.get("kp"):
            st.n_rows_no_kp += 1
            continue
        by_track[int(r["track"])].append(r)

    for rs in by_track.values():
        st.n_tracks += 1
        rs.sort(key=lambda r: int(r["f"]))
        kp = np.asarray([r["kp"] for r in rs], dtype=np.float64)  # (n, 17, 3)
        if kp.ndim != 3 or kp.shape[1:] != (len(JOINTS), 3):
            continue
        f = np.asarray([int(r["f"]) for r in rs])
        h = np.asarray([float(r["h"]) for r in rs])
        conf = kp[:, :, 2]
        st.n_kp_nonfinite += int((~np.isfinite(conf)).sum())
        for j in range(len(JOINTS)):
            st.conf.add(conf[:, j], row=j)

        if len(rs) < 2:
            continue
        adj = (f[1:] - f[:-1]) == 1
        hm = (h[1:] + h[:-1]) / 2.0
        ok_h = adj & np.isfinite(hm) & (hm > 0)
        if not ok_h.any():
            continue
        d = np.linalg.norm(kp[1:, :, :2] - kp[:-1, :, :2], axis=2)  # (n-1, 17)
        both = (conf[1:] >= gate) & (conf[:-1] >= gate)
        for j in range(len(JOINTS)):
            m = ok_h & both[:, j] & np.isfinite(d[:, j])
            if not m.any():
                continue
            vals = d[m, j] / hm[m]
            st.jit.add(vals, row=j)
            st.jit_s.add(vals * fps, row=j)


def scan_tree(tree: Path, gate: float) -> TreeStats:
    st = TreeStats(tree.name)
    for meta_path in sorted((tree / "meta").glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        seq = tree / "sequences" / f"{meta['key']}.jsonl"
        if not seq.is_file():
            continue
        st.n_seqs += 1
        fps = float(meta.get("fps") or 30.0)
        st.fps.append(fps)
        if meta.get("location"):
            st.locations.add(str(meta["location"]))
        rows = [
            json.loads(line)
            for line in seq.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        scan_sequence(rows, st, gate, fps)
    return st


def pose_provenance(tree: Path) -> dict[str, str]:
    """Where this tree's pose weights are attested: the run record, else its config."""
    ex = tree / "extraction.json"
    if ex.is_file():
        p = json.loads(ex.read_text(encoding="utf-8")).get("provenance", {}).get("pose") or {}
        # extract_track_sequences.py records {} for a pose model it did not load, so an
        # extraction written without one falls through to the configuration scan below.
        if p.get("sha256"):
            return {
                "sha8": p["sha256"][:8],
                "label": p.get("label", "unknown"),
                "attested_by": "extraction.json",
            }
    for cfg in sorted((REPO_ROOT / "configs" / "events").glob("extract_*.yaml")):
        text = cfg.read_text(encoding="utf-8")
        if f"out_dir: outputs/events/{tree.name}" in text:
            for line in text.splitlines():
                if line.strip().startswith("name: rtmpose"):
                    label = line.split("name:", 1)[1].strip()
                    return {
                        "sha8": label.rsplit("-", 1)[-1],
                        "label": label,
                        "attested_by": f"configs/events/{cfg.name} (no extraction.json on disk)",
                    }
    return {"sha8": "unknown", "label": "unknown", "attested_by": "none"}


def fmt(x: float | None, nd: int = 3, hi: float = JIT_MAX) -> str:
    if x is None:
        return "n/a"
    if math.isinf(x):
        return f">{hi:g}"
    return f"{x:.{nd}f}"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def group_stats(st: TreeStats, idxs: tuple[int, ...], gate: float) -> dict[str, Any]:
    """Merge a joint group's rows of the per-joint histograms, then read it."""
    conf = Hist(CONF_BINS, 1.0)
    jit = Hist(JIT_BINS, JIT_MAX)
    jit_s = Hist(JIT_BINS, JIT_S_MAX)
    for j in idxs:
        conf.counts += st.conf.counts[j]
        jit.counts += st.jit.counts[j]
        jit_s.counts += st.jit_s.counts[j]
    return {
        "n": conf.total(),
        "p10": conf.quantile(0.10),
        "median": conf.quantile(0.50),
        "p90": conf.quantile(0.90),
        "frac_ge": conf.frac_ge(gate),
        "jit_n": jit.total(),
        "jit_median": jit.quantile(0.50),
        "jit_p90": jit.quantile(0.90),
        "jit_s_median": jit_s.quantile(0.50),
    }


READING = """## 5. Reading

**Nothing here is a per-joint accuracy.** RTMPose emits a confidence for all 17 joints on
every observation, so `n obs` is the row count, not a detection count, and a 0.1 entry is
the estimator guessing rather than a joint that went missing. What the table ranks is how
often each joint is *worth reading*.

1. **The trustworthy end is the head and the shoulder girdle; the untrustworthy end is
   every distal joint.** Pooled medians fall monotonically outward from the torso: face
   0.553, shoulders 0.538, elbows 0.433, hips 0.413, wrists 0.373, ankles 0.371, knees
   0.360. The share clearing the 0.3 gate falls with them, 73.3% at the face to 56.9%
   at the knees. Roughly two in five wrist, knee and ankle observations in this corpus are
   below the gate that `posture_angle_deg` and `KP_CHANNELS` already apply, so any feature
   built on a distal joint is running on about half the frames it appears to have.
2. **Which tree the footage comes from matters more than which joint.** `wu_sahzu`, the
   seizure lane's own EMU footage and the only clinical set here, sits at face 0.457 /
   wrists 0.312 / knees 0.277, while every acted-fall tree sits between 0.55 and 0.85 on
   the same joints. gmdcsa24 and le2i clear the gate on 88-99% of observations at every
   joint; `wu_sahzu` clears it on 46-67%. Bedding, blankets, low light and a wall-mounted
   oblique view are the difference. A gate tuned on the fall trees is a different gate on
   the seizure tree.
3. **Jitter is small and, at the median, nearly joint-independent.** Conditioned on both
   endpoints clearing the gate, the median frame-to-frame displacement is 0.007-0.009
   box-heights everywhere, about 0.2-0.3 box-heights per second, well under the motion any
   event rule looks for. The joints separate only in the tail: p90 is 0.033-0.034 for
   shoulders and hips against 0.045-0.047 for wrists and ankles, i.e. the distal joints are
   not noisier most of the time, they are noisier *sometimes*, and that tail is where a
   spectral limb feature would live. Read those numbers as conditional: a joint that clears
   the gate 57% of the time contributes jitter only from its better half.
4. **Per tree the jitter ordering is a property of the footage, not of the estimator.**
   `wu_sahzu` is the quietest tree at every joint (0.004-0.007 bh/frame) because its
   subjects are in bed; the multicam trees are the loudest (0.013-0.019) because their
   subjects walk across the room. This is the same fact the seizure lane already pays for
   elsewhere, that a patient convulsing in bed barely moves their box, arriving from the
   opposite direction.
5. **The v2 pose model of record did not fix the limb-confidence problem.** On identical footage
   the wrist+ankle median moves from 0.303 (stock body7, `v2kp`) to 0.311 (v2 model of record,
   `v2pose`), and the share above 0.5 moves from 24.8% to 26.1%. The limb-channel gate
   finding therefore survives its own pose upgrade intact: sub-gate limb confidence on
   this footage is a property of the recordings, not of the weights, and re-extracting will
   not buy a limb feature the confidence it needs. The two independently recomputed rows
   also reproduce that document's headline (0.305 recorded, 0.303 recomputed here over a row
   population that differs by 0.7%), which is the cheapest available check that both
   instruments are measuring the same thing.
6. **What this does not say.** No number here is evidence for or against any model, gate
   value or feature; it is the distribution those decisions are taken against, which had
   never been written down per joint. Confidence is not calibrated, and 0.6 from RTMPose is
   not a 60% probability of correct placement, so these are comparisons within one
   estimator's own scale, never absolute accuracy. And jitter here is displacement, not
   error: a fast-moving joint and a badly-estimated one look the same to it.
"""


def contrast_section(v2pose: TreeStats, v2kp: TreeStats | None, gate: float) -> list[str]:
    """The wrist+ankle population the limb-channel gate finding settled a feature on."""
    L = ["## 4. Contrast with the one recorded body7 figure", ""]
    L.append(
        "`research/seizure-limb-channel-gate.md` measured **wrist+ankle "
        "confidence median 0.305, 25.1% > 0.5** over all 976,659 rows of "
        "`wu_sahzu_tracks_v2kp`, and that single pooled number is the measured cause it "
        "gives for the limb channels' 8/8-negative result. `v2kp` and `v2pose` are the same "
        "36 recordings through the same detector (`55c2c242`, sha `6a6e4d90`) and the same "
        "tracker; the only difference between them is the pose graph, stock body7 "
        "`5c0a4bf6` against the v2 model of record `6ba96285`. So the two computed rows below "
        "isolate the pose upgrade and nothing else."
    )
    L.append("")
    L.append(
        "| population (wrist + ankle, every track) | n obs | conf p10 | conf median | conf p90 | at or above 0.5 | at or above 0.8 | at or above gate |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    L.append(
        "| `wu_sahzu_tracks_v2kp`, as recorded in research/seizure-limb-channel-gate.md | "
        "976,659 rows x 4 | not recorded | 0.305 | not recorded | 25.1% | 1.8% | not recorded |"
    )
    for label, st in (
        ("`wu_sahzu_tracks_v2kp` (recomputed here)", v2kp),
        ("`wu_sahzu_tracks_v2pose` (v2 pose model of record)", v2pose),
    ):
        if st is None:
            continue
        g = group_stats(st, LIMB_JOINTS, gate)
        conf = Hist(CONF_BINS, 1.0)
        for j in LIMB_JOINTS:
            conf.counts += st.conf.counts[j]
        L.append(
            f"| {label} | {g['n']:,} | {fmt(g['p10'])} | **{fmt(g['median'])}** | "
            f"{fmt(g['p90'])} | {pct(conf.frac_ge(0.5))} | {pct(conf.frac_ge(0.8))} | "
            f"{pct(g['frac_ge'])} |"
        )
    L.append("")
    L.append(
        "`n obs` counts joint observations (rows x 4 joints), which is why the recorded row is "
        "given as `rows x 4`. The `v2kp` tree on disk today holds 983,104 rows against the "
        "976,659 research/seizure-limb-channel-gate.md counted, a 0.7% difference whose most "
        "likely cause is that "
        "the earlier count dropped the `h <= 0` rows `tracks_from_rows` discards. It is far "
        "too small to move a median, and the recomputed 0.303 against the recorded 0.305 is "
        "the check that both instruments are measuring the same population."
    )
    L.append("")
    L.append(
        "The columns marked at or above are inclusive where "
        "research/seizure-limb-channel-gate.md wrote "
        "`>`; at 0.001 bin width "
        "the two differ by less than the last digit shown."
    )
    L.append("")
    return L


def build_markdown(
    trees: list[TreeStats],
    pooled: TreeStats,
    prov: dict[str, dict],
    gate: float,
    contrast: TreeStats | None = None,
) -> str:
    L: list[str] = []
    shas = sorted({p["sha8"] for p in prov.values()})
    L.append("# Pose noise profile: the v2pose track trees, joint by joint")
    L.append("")
    L.append(
        f"Dated 2026-09-05. Scope: report only. Instrument: "
        f"`scripts/analysis/pose_noise_profile.py`. Every number below is a count over rows "
        f"that were already on disk; no model was loaded, no window was scored, nothing "
        f"was trained. **Pose weights: `{'`, `'.join(shas)}`** "
        f"(`rtmpose-m-person-v2`, the P0-e38 model of record of "
        f"research/pose-seed-floor-and-champion.md); all {len(trees)} trees were extracted "
        f"with that one graph, and the per-tree table records how each tree attests it."
    )
    L.append("")
    L.append(
        f"Gate used throughout: **conf at or above {gate}**, which is `features.KP_CONF_MIN`, the "
        f"`tracks.kp_conf_min` every v2pose extraction config carries and the gate "
        f"`posture_angle_deg` applies. Jitter is the per-frame displacement of a joint "
        f"between rows of one track whose frame indices differ by exactly 1, with both "
        f"endpoints above the gate, divided by the mean of the two box heights "
        f"(box-heights/frame; the per-second column makes 15/20/25/30 fps trees "
        f"comparable). "
        f"Definitions in full sit in the instrument's docstring."
    )
    L.append("")

    L.append("## 1. Pooled over every v2pose tree")
    L.append("")
    L.append(
        f"{pooled.n_rows:,} rows, {pooled.n_tracks:,} tracks and {pooled.n_seqs:,} recordings "
        f"across {len(trees)} trees."
    )
    L.append("")
    L.append(
        "| joint | n obs | conf p10 | conf median | conf p90 | at or above gate | jitter median (bh/frame) | jitter p90 | jitter median (bh/s) | n pairs |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for j, name in enumerate(JOINTS):
        L.append(
            f"| {name} | {pooled.conf.total(j):,} | {fmt(pooled.conf.quantile(0.10, j))} | "
            f"**{fmt(pooled.conf.quantile(0.50, j))}** | {fmt(pooled.conf.quantile(0.90, j))} | "
            f"{pct(pooled.conf.frac_ge(gate, j))} | {fmt(pooled.jit.quantile(0.50, j))} | "
            f"{fmt(pooled.jit.quantile(0.90, j))} | {fmt(pooled.jit_s.quantile(0.50, j), 3, JIT_S_MAX)} | "
            f"{pooled.jit.total(j):,} |"
        )
    L.append("")
    L.append("By group:")
    L.append("")
    L.append(
        "| group | conf p10 | conf median | conf p90 | at or above gate | jitter median (bh/frame) | jitter p90 |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for gname, idxs in GROUPS.items():
        g = group_stats(pooled, idxs, gate)
        L.append(
            f"| {gname} | {fmt(g['p10'])} | **{fmt(g['median'])}** | {fmt(g['p90'])} | "
            f"{pct(g['frac_ge'])} | {fmt(g['jit_median'])} | {fmt(g['jit_p90'])} |"
        )
    L.append("")

    L.append("## 2. Per tree, joint groups")
    L.append("")
    L.append("Confidence median (top line of each cell) and share at or above gate (bottom).")
    L.append("")
    head = "| tree | rows | tracks | recs | " + " | ".join(GROUPS) + " |"
    L.append(head)
    L.append("|---|---:|---:|---:|" + "---:|" * len(GROUPS))
    for st in trees:
        cells = []
        for idxs in GROUPS.values():
            g = group_stats(st, idxs, gate)
            cells.append(f"{fmt(g['median'])}<br>{pct(g['frac_ge'])}")
        L.append(
            f"| `{st.name}` | {st.n_rows:,} | {st.n_tracks:,} | {st.n_seqs} | "
            + " | ".join(cells)
            + " |"
        )
    L.append("")
    L.append("Jitter median, box-heights per frame (gate-passing adjacent pairs only):")
    L.append("")
    L.append("| tree | median fps | " + " | ".join(GROUPS) + " |")
    L.append("|---|---:|" + "---:|" * len(GROUPS))
    for st in trees:
        cells = []
        for idxs in GROUPS.values():
            g = group_stats(st, idxs, gate)
            cells.append(fmt(g["jit_median"]))
        med_fps = float(np.median(st.fps)) if st.fps else float("nan")
        L.append(f"| `{st.name}` | {med_fps:.2f} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 3. Scalar channels the verifiers actually read")
    L.append("")
    L.append(
        "`kpc` is the row's mean confidence over all 17 entries, which is feature channel "
        "`kpc`. `angle_valid` is the share of rows whose `angle` field is non-null, the row-level "
        "population behind feature channel `angle_valid` (and, because `angle_norm` is that "
        "same field over 90 with nulls filled to 0, behind `angle_norm` too)."
    )
    L.append("")
    L.append(
        "| tree | rows | kpc p10 | kpc median | kpc p90 | kpc at or above gate | angle_valid | pose weights (attested by) |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for st in trees:
        p = prov[st.name]
        av = st.n_angle_valid / st.n_rows if st.n_rows else None
        L.append(
            f"| `{st.name}` | {st.n_rows:,} | {fmt(st.kpc.quantile(0.10))} | "
            f"**{fmt(st.kpc.quantile(0.50))}** | {fmt(st.kpc.quantile(0.90))} | "
            f"{pct(st.kpc.frac_ge(gate))} | {pct(av)} | `{p['sha8']}` ({p['attested_by']}) |"
        )
    av = pooled.n_angle_valid / pooled.n_rows if pooled.n_rows else None
    L.append(
        f"| **pooled** | {pooled.n_rows:,} | {fmt(pooled.kpc.quantile(0.10))} | "
        f"**{fmt(pooled.kpc.quantile(0.50))}** | {fmt(pooled.kpc.quantile(0.90))} | "
        f"{pct(pooled.kpc.frac_ge(gate))} | {pct(av)} | not applicable |"
    )
    L.append("")

    wu = next((s for s in trees if s.name == "wu_sahzu_tracks_v2pose"), None)
    if wu is not None:
        L += contrast_section(wu, contrast, gate)
        exam = sorted(wu.locations & EXAM_LOCATIONS)
        if exam:
            L.append(
                f"**Exam-side rows are included, as statistics only.** The WU tree's "
                f"{len(wu.locations)} patients include the {len(exam)} allocated exam-side "
                f"({', '.join(exam)}). This instrument runs no model and produces no score, "
                f"so their rows are data profiling, not an exam look; no number on this page "
                f"is or implies a held-out metric."
            )
            L.append("")
    L.append(READING)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trees-glob", default="outputs/events/*_tracks_v2pose")
    ap.add_argument("--conf-gate", type=float, default=0.3)
    ap.add_argument(
        "--contrast-tree",
        default="outputs/events/wu_sahzu_tracks_v2kp",
        help="one extra tree scanned for the body7 contrast section; '' to skip",
    )
    ap.add_argument("--out", default="research/pose-noise-profile.md")
    ap.add_argument("--json-out", default=None, help="optional machine-readable sidecar")
    args = ap.parse_args()

    trees = sorted(REPO_ROOT.glob(args.trees_glob))
    if not trees:
        raise SystemExit(f"no trees matched {args.trees_glob}")

    stats: list[TreeStats] = []
    prov: dict[str, dict] = {}
    pooled = TreeStats("pooled")
    for t in trees:
        print(f"scanning {t.name} ...", flush=True)
        st = scan_tree(t, args.conf_gate)
        stats.append(st)
        prov[st.name] = pose_provenance(t)
        pooled.merge(st)
        print(
            f"  rows {st.n_rows:,} tracks {st.n_tracks:,} recs {st.n_seqs} "
            f"kpc_median {fmt(st.kpc.quantile(0.5))}",
            flush=True,
        )

    contrast: TreeStats | None = None
    if args.contrast_tree:
        ct = REPO_ROOT / args.contrast_tree
        if ct.is_dir():
            print(f"scanning contrast tree {ct.name} ...", flush=True)
            contrast = scan_tree(ct, args.conf_gate)
            print(f"  rows {contrast.n_rows:,} tracks {contrast.n_tracks:,}", flush=True)
        else:
            print(f"contrast tree {ct} not on disk - section 4 will omit it", flush=True)

    md = build_markdown(stats, pooled, prov, args.conf_gate, contrast)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    print(f"wrote {out}")

    if args.json_out:
        blob = {
            "conf_gate": args.conf_gate,
            "pose_weights": prov,
            "trees": {
                st.name: {
                    "n_rows": st.n_rows,
                    "n_tracks": st.n_tracks,
                    "n_recordings": st.n_seqs,
                    "n_rows_without_kp": st.n_rows_no_kp,
                    "n_nonfinite_kp_conf": st.n_kp_nonfinite,
                    "median_fps": float(np.median(st.fps)) if st.fps else None,
                    "locations": sorted(st.locations),
                    "angle_valid_frac": (st.n_angle_valid / st.n_rows) if st.n_rows else None,
                    "kpc": {
                        q: st.kpc.quantile(v)
                        for q, v in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))
                    },
                    "joints": {
                        JOINTS[j]: {
                            "n": st.conf.total(j),
                            "p10": st.conf.quantile(0.10, j),
                            "median": st.conf.quantile(0.50, j),
                            "p90": st.conf.quantile(0.90, j),
                            "frac_ge_gate": st.conf.frac_ge(args.conf_gate, j),
                            "jitter_pairs": st.jit.total(j),
                            "jitter_median_bh_per_frame": st.jit.quantile(0.50, j),
                            "jitter_p90_bh_per_frame": st.jit.quantile(0.90, j),
                            "jitter_median_bh_per_s": st.jit_s.quantile(0.50, j),
                        }
                        for j in range(len(JOINTS))
                    },
                }
                for st in [*stats, pooled, *([contrast] if contrast else [])]
            },
            "limb_joints_pooled": group_stats(pooled, LIMB_JOINTS, args.conf_gate),
            "limb_joints_wu_v2pose": (
                group_stats(_wu, LIMB_JOINTS, args.conf_gate)
                if (_wu := next((s for s in stats if s.name == "wu_sahzu_tracks_v2pose"), None))
                else None
            ),
            "limb_joints_contrast": (
                group_stats(contrast, LIMB_JOINTS, args.conf_gate) if contrast else None
            ),
        }
        Path(args.json_out).write_text(
            json.dumps(blob, indent=1, default=lambda o: None if o == math.inf else o),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
