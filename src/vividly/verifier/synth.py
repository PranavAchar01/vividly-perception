"""Synthetic convulsion motion generator (2026-08-26).

No public real-seizure video corpus was available when this generator was written (TUH and
CHB-MIT are EEG corpora), and acting one out is barred. The carve-out in rule 4 of
POLICY.md opens exactly one lane: synthetic convulsion motion generated in pose space and
track space, that is skeleton and trajectory arithmetic with no imagery of any kind, may
train the interim convulsion verifier. This module is that generator.

The lane now holds real footage. DATASETS.md registers `wu_sahzu_emu`, the held-out
seizure exam, and its train-side split `wu_sahzu_emu_trainsplit`, and the seizure model of
record is convulsion_v2_real (checkpoint 715d172c, ONNX 07c399be, testsplit window AUC
0.69841), trained on real WU-SAHZU tracks read as track and pose features
(final_models/seizure/README.md). This module produced the superseded convulsion_v1_synthreal
lineage (checkpoint 094ddddc, ONNX a95616f2, testsplit window AUC 0.55937) and remains the
generator for the synthetic negative-motion classes below.

What this is not
================
Every constant below (frequency bands, amplitude fractions, noise sigmas, envelope
shapes) is an invented modelling assumption anchored loosely on the clinical literature
(clonic jerking concentrates around 2 to 4 Hz and slows before stopping) and on the shape
of the burn-in's false firings, which were small box jitter with frequent direction
reversal on crowded footage. None of it is a patient. A verifier trained here
learns to separate sustained rhythmic oscillation from tracker noise, walking, fidgeting
and rocking, which is a motion signature and not a diagnosis, and no number it produces
may be quoted, internally or externally, as clinical detection performance. The weights
are prototype-class and are replaced when clinical footage exists (CQ-21, which gates
both the claim wording and what a learned event model may train on).

Mechanism
=========
Sequences are generated in continuous time and then sampled at the sequence's fps, so
a 4 Hz oscillation observed at 10 fps aliases exactly the way it would off a real
camera, and the verifier trains on what the pipeline would actually see. Output rows are
byte-compatible with scripts/events/extract_track_sequences.py
({f, t, track, cx, cy, w, h, angle, kpc}), so real extracted negatives and synthetic
sequences flow through one training path.

Positives: a person mostly down (high posture angle) or seated; after an onset delay,
box-relative oscillation of the centre (amplitude a fraction of box height, never
absolute pixels, because the burn-in showed absolute-pixel gates are scale artefacts) at a
frequency that wanders and slows toward the end, under a rise and decay envelope.
Negatives, same base motion model: still, subclinical tremor (too small), fidget
(non-rhythmic bursts), walk (translation), rock (slow periodic, below the clonic band),
jitter (white tracker noise, the burn-in's false-positive class).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

NEGATIVE_KINDS: tuple[str, ...] = ("still", "tremor", "fidget", "walk", "rock", "jitter")

ROW_KEYS: tuple[str, ...] = ("f", "t", "track", "cx", "cy", "w", "h", "angle", "kpc")


@dataclass(frozen=True)
class SynthConfig:
    """Ranges every per-sequence draw comes from. All amplitudes are fractions of box
    height; all pixel geometry is derived, so one config serves any frame scale."""

    frame_w: float = 320.0
    frame_h: float = 240.0
    fps_choices: tuple[float, ...] = (10.0, 15.0, 24.0, 25.0)
    duration_s: tuple[float, float] = (20.0, 90.0)
    onset_s: tuple[float, float] = (4.0, 20.0)
    event_s: tuple[float, float] = (10.0, 45.0)
    # Clonic band, capped under each sequence's Nyquist with margin at draw time.
    freq_hz: tuple[float, float] = (1.5, 6.0)
    freq_wander: float = 0.10  # random-walk scale on instantaneous frequency
    freq_slow: tuple[float, float] = (0.15, 0.40)  # fractional slowdown by event end
    amp_frac: tuple[float, float] = (0.03, 0.12)  # of box height; rule gate territory
    axis_ratio: tuple[float, float] = (0.3, 1.0)  # vertical amplitude vs lateral
    # Posture: lying (angle near 90) dominates; a third of positives are seated.
    lying_angle_deg: tuple[float, float] = (60.0, 90.0)
    seated_angle_deg: tuple[float, float] = (20.0, 50.0)
    seated_prob: float = 0.33
    angle_osc_deg: tuple[float, float] = (1.0, 6.0)
    # Box geometry (fractions of frame height).
    box_h_frac: tuple[float, float] = (0.25, 0.6)
    lying_aspect: tuple[float, float] = (1.4, 2.4)  # w/h when down
    upright_aspect: tuple[float, float] = (0.35, 0.65)
    # Tracker noise, as a box-relative sigma rather than absolute pixels, so one config
    # serves any box scale.
    noise_frac: tuple[float, float] = (0.004, 0.02)
    drift_frac_per_s: tuple[float, float] = (0.0, 0.01)
    angle_noise_deg: tuple[float, float] = (0.5, 3.0)
    kpc_base: tuple[float, float] = (0.55, 0.85)
    dropout_prob: tuple[float, float] = (0.0, 0.06)
    # Negatives.
    tremor_amp_frac: tuple[float, float] = (0.004, 0.015)  # below the amp band
    rock_hz: tuple[float, float] = (0.2, 1.2)  # below the clonic band
    walk_speed_frac_per_s: tuple[float, float] = (0.05, 0.25)  # of frame width
    fidget_burst_s: tuple[float, float] = (0.3, 1.5)
    jitter_frac: tuple[float, float] = (0.01, 0.05)  # the burn-in false-alarm class


@dataclass(frozen=True)
class SequenceParams:
    """Everything one sequence was generated from; written into its meta record."""

    kind: str  # "convulsion" or a NEGATIVE_KINDS member
    fps: float
    duration_s: float
    onset_s: float | None
    event_s: float | None
    freq_hz: float | None
    amp_frac: float | None
    posture: str
    base_angle_deg: float
    box_h_px: float
    noise_frac: float
    extra: dict[str, float] = field(default_factory=dict)


def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _base_geometry(
    rng: np.random.Generator, cfg: SynthConfig, *, down: bool
) -> tuple[float, float, float, float]:
    """(cx, cy, w, h) of the resting box, inside the frame with margin."""
    h = _u(rng, cfg.box_h_frac) * cfg.frame_h
    if down:
        h, aspect = h * 0.55, _u(rng, cfg.lying_aspect)
    else:
        aspect = _u(rng, cfg.upright_aspect)
    w = h * aspect
    cx = float(rng.uniform(w / 2 + 2, cfg.frame_w - w / 2 - 2))
    cy = float(rng.uniform(h / 2 + 2, cfg.frame_h - h / 2 - 2))
    return cx, cy, w, h


def _sample_times(rng: np.random.Generator, cfg: SynthConfig, fps: float) -> np.ndarray:
    duration = _u(rng, cfg.duration_s)
    n = max(2, round(duration * fps))
    return np.arange(n, dtype=np.float64) / fps


def _noise_walk(rng: np.random.Generator, n: int, sigma: float, drift: float) -> np.ndarray:
    """White noise + integrated drift, the tracker's two visible error modes."""
    white = rng.normal(0.0, sigma, n)
    slow = np.cumsum(rng.normal(0.0, drift, n))
    return white + slow


def _rows_from_arrays(
    t: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    angle: np.ndarray,
    kpc: np.ndarray,
    keep: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(len(t)):
        if not keep[i]:
            continue
        rows.append(
            {
                "f": int(i),
                "t": round(float(t[i]), 4),
                "track": 0,
                "cx": round(float(cx[i]), 2),
                "cy": round(float(cy[i]), 2),
                "w": round(float(w[i]), 2),
                "h": round(float(h[i]), 2),
                "angle": round(float(np.clip(angle[i], 0.0, 90.0)), 2),
                "kpc": round(float(np.clip(kpc[i], 0.05, 0.99)), 4),
            }
        )
    return rows


def _common_channels(
    rng: np.random.Generator, cfg: SynthConfig, t: np.ndarray, h_px: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """(noise_cx, noise_cy, kpc, noise_frac, keep) shared by every kind."""
    n = len(t)
    noise_frac = _u(rng, cfg.noise_frac)
    drift = _u(rng, cfg.drift_frac_per_s) * h_px / max(1.0, float(len(t)) / (t[-1] + 1e-9))
    sigma = noise_frac * h_px
    ncx = _noise_walk(rng, n, sigma, drift)
    ncy = _noise_walk(rng, n, sigma, drift)
    kpc = np.clip(_u(rng, cfg.kpc_base) + rng.normal(0.0, 0.05, n), 0.05, 0.99)
    keep = rng.uniform(0.0, 1.0, n) >= _u(rng, cfg.dropout_prob)
    keep[0] = True
    return ncx, ncy, kpc, noise_frac, keep


def convulsion_sequence(
    rng: np.random.Generator, cfg: SynthConfig | None = None
) -> tuple[list[dict[str, Any]], SequenceParams]:
    """One positive: onset of sustained rhythmic oscillation, box-relative amplitude,
    wandering-then-slowing frequency, on a lying or seated base posture."""
    cfg = cfg or SynthConfig()
    fps = float(rng.choice(cfg.fps_choices))
    t = _sample_times(rng, cfg, fps)
    n = len(t)
    seated = rng.uniform() < cfg.seated_prob
    down = not seated
    cx0, cy0, w0, h0 = _base_geometry(rng, cfg, down=down)
    base_angle = _u(rng, cfg.seated_angle_deg if seated else cfg.lying_angle_deg)

    onset = min(_u(rng, cfg.onset_s), max(0.0, float(t[-1]) - 8.0))
    event_len = min(_u(rng, cfg.event_s), float(t[-1]) - onset)
    # Frequency: drawn from the clonic band, capped under this fps's Nyquist with margin.
    f_hi = min(cfg.freq_hz[1], 0.45 * fps)
    f0 = float(rng.uniform(cfg.freq_hz[0], max(cfg.freq_hz[0] + 0.1, f_hi)))
    slow = _u(rng, cfg.freq_slow)
    amp_frac = _u(rng, cfg.amp_frac)
    axis = _u(rng, cfg.axis_ratio)
    angle_osc = _u(rng, cfg.angle_osc_deg)

    in_event = (t >= onset) & (t <= onset + event_len)
    prog = np.zeros(n)
    prog[in_event] = (t[in_event] - onset) / max(event_len, 1e-9)
    # Envelope: fast rise, slow decay. Instantaneous frequency wanders and slows.
    env = np.where(in_event, np.minimum(1.0, prog * 8.0) * (1.0 - 0.5 * prog**2), 0.0)
    f_inst = (
        f0
        * (1.0 - slow * prog)
        * (
            1.0
            + cfg.freq_wander * np.cumsum(rng.normal(0.0, 1.0 / math.sqrt(max(fps, 1.0)), n)) / 10.0
        )
    )
    phase = 2.0 * math.pi * np.cumsum(f_inst) / fps
    osc = env * amp_frac * h0 * np.sin(phase)

    ncx, ncy, kpc, noise_frac, keep = _common_channels(rng, cfg, t, h0)
    cx = cx0 + osc + ncx
    cy = cy0 + axis * env * amp_frac * h0 * np.sin(phase + rng.uniform(0, math.pi)) + ncy
    w = np.full(n, w0) + rng.normal(0.0, noise_frac * w0, n)
    h = np.full(n, h0) + rng.normal(0.0, noise_frac * h0, n)
    angle = (
        base_angle
        + angle_osc * env * np.sin(phase)
        + rng.normal(0.0, _u(rng, cfg.angle_noise_deg), n)
    )
    rows = _rows_from_arrays(t, cx, cy, w, h, angle, kpc, keep)
    params = SequenceParams(
        kind="convulsion",
        fps=fps,
        duration_s=float(t[-1]),
        onset_s=round(onset, 3),
        event_s=round(event_len, 3),
        freq_hz=round(f0, 3),
        amp_frac=round(amp_frac, 4),
        posture="seated" if seated else "lying",
        base_angle_deg=round(base_angle, 2),
        box_h_px=round(h0, 1),
        noise_frac=round(noise_frac, 4),
        extra={"freq_slow": round(slow, 3), "axis_ratio": round(axis, 3)},
    )
    return rows, params


def negative_sequence(
    rng: np.random.Generator, kind: str, cfg: SynthConfig | None = None
) -> tuple[list[dict[str, Any]], SequenceParams]:
    """One negative of the named kind. Shares the base geometry and noise model with
    the positives so the only separating signal is the motion structure itself."""
    cfg = cfg or SynthConfig()
    if kind not in NEGATIVE_KINDS:
        raise ValueError(f"kind must be one of {NEGATIVE_KINDS}, got {kind!r}")
    fps = float(rng.choice(cfg.fps_choices))
    t = _sample_times(rng, cfg, fps)
    n = len(t)
    down = kind in ("still", "tremor") and rng.uniform() < 0.5
    cx0, cy0, w0, h0 = _base_geometry(rng, cfg, down=down)
    if down:
        base_angle = _u(rng, cfg.lying_angle_deg)
    elif kind == "walk":
        base_angle = float(rng.uniform(0.0, 20.0))
    else:
        base_angle = _u(rng, cfg.seated_angle_deg)

    ncx, ncy, kpc, noise_frac, keep = _common_channels(rng, cfg, t, h0)
    dx = np.zeros(n)
    dy = np.zeros(n)
    extra: dict[str, float] = {}
    if kind == "tremor":
        f_hi = min(cfg.freq_hz[1], 0.45 * fps)
        f_tr = float(rng.uniform(cfg.freq_hz[0], max(cfg.freq_hz[0] + 0.1, f_hi)))
        amp = _u(rng, cfg.tremor_amp_frac) * h0  # subclinical: below the amp band
        dx = amp * np.sin(2 * math.pi * f_tr * t)
        extra = {"freq_hz": round(f_tr, 3), "amp_frac": round(amp / h0, 4)}
    elif kind == "rock":
        f_rock = _u(rng, cfg.rock_hz)
        amp = _u(rng, cfg.amp_frac) * h0  # full amplitude, wrong band
        dx = amp * np.sin(2 * math.pi * f_rock * t)
        extra = {"freq_hz": round(f_rock, 3), "amp_frac": round(amp / h0, 4)}
    elif kind == "walk":
        speed = _u(rng, cfg.walk_speed_frac_per_s) * cfg.frame_w
        heading = rng.uniform(0, 2 * math.pi)
        dx = np.cos(heading) * speed * t
        dy = np.sin(heading) * speed * t * 0.3
        extra = {"speed_px_s": round(speed, 1)}
    elif kind == "fidget":
        # Non-rhythmic bursts: a few short random-direction displacements.
        n_bursts = int(rng.integers(2, 7))
        for _ in range(n_bursts):
            b0 = rng.uniform(0.0, max(0.1, float(t[-1]) - 2.0))
            blen = _u(rng, cfg.fidget_burst_s)
            mask = (t >= b0) & (t <= b0 + blen)
            dx[mask] += rng.normal(0.0, 0.05 * h0)
            dy[mask] += rng.normal(0.0, 0.05 * h0)
        extra = {"n_bursts": float(n_bursts)}
    elif kind == "jitter":
        sigma = _u(rng, cfg.jitter_frac) * h0  # the burn-in false-alarm class
        dx = rng.normal(0.0, sigma, n)
        dy = rng.normal(0.0, sigma, n)
        extra = {"jitter_frac": round(sigma / h0, 4)}

    cx = np.clip(cx0 + dx + ncx, 0.0, cfg.frame_w)
    cy = np.clip(cy0 + dy + ncy, 0.0, cfg.frame_h)
    w = np.full(n, w0) + rng.normal(0.0, noise_frac * w0, n)
    h = np.full(n, h0) + rng.normal(0.0, noise_frac * h0, n)
    angle = base_angle + rng.normal(0.0, _u(rng, cfg.angle_noise_deg), n)
    rows = _rows_from_arrays(t, cx, cy, w, h, angle, kpc, keep)
    params = SequenceParams(
        kind=kind,
        fps=fps,
        duration_s=float(t[-1]),
        onset_s=None,
        event_s=None,
        freq_hz=None,
        amp_frac=None,
        posture="lying" if down else ("upright" if kind == "walk" else "seated"),
        base_angle_deg=round(base_angle, 2),
        box_h_px=round(h0, 1),
        noise_frac=round(noise_frac, 4),
        extra=extra,
    )
    return rows, params


def write_synth_dataset(
    out_dir: Path,
    *,
    n_positive: int,
    n_negative_per_kind: int,
    seed: int,
    cfg: SynthConfig | None = None,
) -> dict[str, Any]:
    """Materialize a labelled synthetic set in the extractor's output layout:
    sequences/<key>.jsonl + meta/<key>.json + extraction.json. Deterministic in seed."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(seed)
    (out_dir / "sequences").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    metas: list[dict[str, Any]] = []

    def _write(key: str, rows: list[dict[str, Any]], params: SequenceParams) -> None:
        (out_dir / "sequences" / f"{key}.jsonl").write_text(
            "\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n",
            encoding="utf-8",
        )
        meta = {
            "key": key,
            "synthetic": True,
            "kind": params.kind,
            "has_convulsion": params.kind == "convulsion",
            "convulsion_start_t": params.onset_s,
            "convulsion_end_t": (
                None if params.onset_s is None else round(params.onset_s + params.event_s, 3)
            ),
            "fps": params.fps,
            "n_rows": len(rows),
            "params": asdict(params),
        }
        (out_dir / "meta" / f"{key}.json").write_text(
            json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        metas.append(meta)

    for i in range(n_positive):
        rows, params = convulsion_sequence(rng, cfg)
        _write(f"synth_convulsion_{i:04d}", rows, params)
    for kind in NEGATIVE_KINDS:
        for i in range(n_negative_per_kind):
            rows, params = negative_sequence(rng, kind, cfg)
            _write(f"synth_{kind}_{i:04d}", rows, params)

    summary = {
        "synthetic": True,
        "generator": "vividly.verifier.synth",
        "seed": seed,
        "n_positive": n_positive,
        "n_negative_per_kind": n_negative_per_kind,
        "negative_kinds": list(NEGATIVE_KINDS),
        "n_sequences": len(metas),
        "n_rows_total": sum(m["n_rows"] for m in metas),
    }
    (out_dir / "extraction.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
