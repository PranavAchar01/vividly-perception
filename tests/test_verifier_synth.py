"""Synthetic convulsion generator tests (vividly.verifier.synth).

The claims that matter: the generator is deterministic in its seed, its rows are
byte-compatible with the track-sequence extractor's schema, positives carry sustained
band-limited oscillation exactly where their params say and nowhere else, and three of
the negative kinds lack that signature for the reason their names state (tremor: amplitude
below the band; rock: frequency below the band; jitter: no band at all).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vividly.verifier import synth

CFG_25 = synth.SynthConfig(fps_choices=(25.0,))


def _band_power(x: np.ndarray, fps: float, lo: float, hi: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fps)
    return float(spec[(freqs >= lo) & (freqs <= hi)].sum())


def _event_cx(rows: list[dict], params: synth.SequenceParams) -> np.ndarray:
    lo, hi = params.onset_s, params.onset_s + params.event_s
    return np.array([r["cx"] for r in rows if lo <= r["t"] <= hi])


def test_deterministic_in_seed() -> None:
    a, pa = synth.convulsion_sequence(np.random.default_rng(7))
    b, pb = synth.convulsion_sequence(np.random.default_rng(7))
    assert a == b and pa == pb
    c, _ = synth.convulsion_sequence(np.random.default_rng(8))
    assert c != a


def test_rows_match_extractor_schema() -> None:
    rows, _ = synth.convulsion_sequence(np.random.default_rng(0))
    for row in rows[:5]:
        assert tuple(row.keys()) == synth.ROW_KEYS
    ts = [r["t"] for r in rows]
    assert ts == sorted(ts)
    assert all(0.0 <= r["angle"] <= 90.0 for r in rows)
    assert all(0.0 < r["kpc"] < 1.0 for r in rows)


def test_positive_oscillates_in_band_during_event_only() -> None:
    rng = np.random.default_rng(3)
    rows, params = synth.convulsion_sequence(rng, CFG_25)
    assert params.kind == "convulsion" and params.freq_hz is not None
    event = _event_cx(rows, params)
    pre = np.array([r["cx"] for r in rows if r["t"] < params.onset_s - 0.5])
    in_band = _band_power(event, 25.0, 1.2, 7.0)
    below = _band_power(event, 25.0, 0.05, 1.0)
    assert in_band > 3.0 * below, "event window must be dominated by the clonic band"
    if len(pre) >= 25:
        pre_band = _band_power(pre, 25.0, 1.2, 7.0)
        # Per-sample band power before onset is far below the event's.
        assert in_band / len(event) > 5.0 * (pre_band / len(pre))


def test_positive_amplitude_is_box_relative() -> None:
    # The oscillation the verifier learns must scale with the person's box, not pixels.
    rng = np.random.default_rng(11)
    seen = []
    for _ in range(6):
        rows, params = synth.convulsion_sequence(rng, CFG_25)
        event = _event_cx(rows, params)
        if len(event) < 50:
            continue
        peak = float(np.percentile(np.abs(event - np.median(event)), 95))
        seen.append(peak / params.box_h_px)
    assert seen and all(0.005 < frac < 0.35 for frac in seen)


def test_rock_negative_sits_below_the_clonic_band() -> None:
    rng = np.random.default_rng(5)
    rows, params = synth.negative_sequence(rng, "rock", CFG_25)
    cx = np.array([r["cx"] for r in rows])
    assert params.extra["freq_hz"] < 1.3
    low = _band_power(cx, 25.0, 0.05, 1.3)
    high = _band_power(cx, 25.0, 1.5, 7.0)
    assert low > 3.0 * high


def test_tremor_negative_amplitude_is_subclinical() -> None:
    cfg = CFG_25
    rng = np.random.default_rng(9)
    _, params = synth.negative_sequence(rng, "tremor", cfg)
    assert params.extra["amp_frac"] < cfg.amp_frac[0]


def test_jitter_negative_has_no_dominant_narrowband() -> None:
    rng = np.random.default_rng(13)
    rows, _ = synth.negative_sequence(rng, "jitter", CFG_25)
    cx = np.array([r["cx"] for r in rows], dtype=np.float64)
    cx -= cx.mean()
    spec = np.abs(np.fft.rfft(cx)) ** 2
    assert spec[1:].max() < 0.3 * spec[1:].sum()


def test_unknown_negative_kind_refused() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        synth.negative_sequence(np.random.default_rng(0), "sprint")


def test_write_synth_dataset_layout_and_determinism(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    sa = synth.write_synth_dataset(out_a, n_positive=2, n_negative_per_kind=1, seed=42)
    sb = synth.write_synth_dataset(out_b, n_positive=2, n_negative_per_kind=1, seed=42)
    assert sa == sb
    assert sa["n_sequences"] == 2 + len(synth.NEGATIVE_KINDS)
    seq_a = sorted(p.name for p in (out_a / "sequences").glob("*.jsonl"))
    assert seq_a == sorted(p.name for p in (out_b / "sequences").glob("*.jsonl"))
    for name in seq_a:
        assert (out_a / "sequences" / name).read_bytes() == (
            out_b / "sequences" / name
        ).read_bytes()
    meta = json.loads((out_a / "meta" / "synth_convulsion_0000.json").read_text())
    assert meta["has_convulsion"] and meta["synthetic"]
    assert meta["convulsion_end_t"] > meta["convulsion_start_t"]
    neg = json.loads((out_a / "meta" / "synth_rock_0000.json").read_text())
    assert not neg["has_convulsion"] and neg["convulsion_start_t"] is None
