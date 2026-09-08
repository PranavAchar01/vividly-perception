"""Windowing and featurization tests (vividly.verifier.features).

The load-bearing claims: labels follow the two policies exactly (contain for falls,
overlap for convulsions, ambiguity trains nothing), features are box-relative and
fixed-shape with tracker gaps encoded in the present channel, the event is assigned
to the most-overlapping track and other tracks near it are excluded rather than
called negatives, and the synthetic generator's output flows through the same path.
"""

from __future__ import annotations

import numpy as np
import pytest

from vividly.verifier import synth
from vividly.verifier.features import (
    CHANNELS,
    SCENE_CHANNELS,
    FeatureError,
    WindowSpec,
    featurize_window,
    label_window,
    load_rows,
    tracks_from_rows,
    windows_for_video,
    windows_from_extraction,
)

SPEC = WindowSpec(window_s=16.0, stride_s=2.0, n_steps=64, policy="contain")


def _rows(track: int, t: np.ndarray, cy: np.ndarray, h: float = 100.0) -> list[dict]:
    return [
        {
            "f": i,
            "t": round(float(t[i]), 4),
            "track": track,
            "cx": 160.0,
            "cy": round(float(cy[i]), 2),
            "w": h * 0.5,
            "h": h,
            "angle": 10.0,
            "kpc": 0.8,
        }
        for i in range(len(t))
    ]


# --------------------------------------------------------------- labels
def test_contain_policy_full_inside_is_positive() -> None:
    assert label_window(0.0, 16.0, (5.0, 8.0), SPEC) == 1


def test_contain_policy_partial_overlap_is_ambiguous() -> None:
    assert label_window(0.0, 16.0, (14.0, 20.0), SPEC) is None
    # event touching the margin band is not "contained"
    assert label_window(0.0, 16.0, (15.8, 15.9), SPEC) is None


def test_contain_policy_disjoint_is_negative_and_no_event_is_negative() -> None:
    assert label_window(0.0, 16.0, (20.0, 25.0), SPEC) == 0
    assert label_window(0.0, 16.0, None, SPEC) == 0


def test_overlap_policy_thresholds() -> None:
    spec = WindowSpec(window_s=12.0, policy="overlap", overlap_min_s=4.0)
    assert label_window(0.0, 12.0, (8.0, 30.0), spec) == 1  # 4 s inside
    assert label_window(0.0, 12.0, (11.0, 30.0), spec) is None  # 1 s inside
    assert label_window(0.0, 12.0, (20.0, 30.0), spec) == 0
    # An event shorter than overlap_min_s counts by its own length.
    assert label_window(0.0, 12.0, (5.0, 6.0), spec) == 1


def test_bad_policy_refused() -> None:
    with pytest.raises(FeatureError, match="policy"):
        WindowSpec(policy="majority")


# --------------------------------------------------------------- tracks + features
def test_tracks_from_rows_groups_sorts_and_drops_bad_height() -> None:
    t = np.array([0.2, 0.0, 0.1])
    rows = _rows(3, t, np.array([1.0, 2.0, 3.0]))
    rows.append({**rows[0], "track": 1, "h": 0.0})
    tracks = tracks_from_rows(rows)
    assert [tr.track_id for tr in tracks] == [3]
    assert list(tracks[0].t) == [0.0, 0.1, 0.2]


def test_featurize_shape_channels_and_present_gap() -> None:
    fps = 10.0
    t = np.arange(0, 16.0, 1 / fps)
    keep = (t < 6.0) | (t > 10.0)  # a 4 s tracker gap mid-window
    tr = tracks_from_rows(_rows(0, t[keep], np.full(keep.sum(), 100.0)))[0]
    x = featurize_window(tr, 16.0, SPEC)
    assert x is not None and x.shape == (SPEC.n_steps, len(CHANNELS))
    present = x[:, CHANNELS.index("present")]
    # steps inside the gap are marked absent, edges present
    assert present[:20].min() == 1.0 and present[-18:].min() == 1.0
    gap_zone = present[int(64 * 6.5 / 16) : int(64 * 9.5 / 16)]
    assert gap_zone.max() == 0.0


def test_featurize_vertical_velocity_is_box_relative() -> None:
    fps = 25.0
    t = np.arange(0, 16.0, 1 / fps)
    cy = np.full(len(t), 50.0)
    drop = (t >= 10.0) & (t <= 11.0)
    cy[drop] = 50.0 + (t[drop] - 10.0) * 200.0  # falls 2 box-heights in 1 s
    cy[t > 11.0] = 250.0
    tr = tracks_from_rows(_rows(0, t, cy, h=100.0))[0]
    x = featurize_window(tr, 16.0, SPEC)
    assert x is not None
    vy = x[:, CHANNELS.index("vy")]
    assert vy.max() == pytest.approx(2.0, rel=0.2)  # box-heights per second
    assert abs(vy[:30]).max() < 0.2


def test_featurize_rejects_sub_coverage() -> None:
    t = np.arange(0.0, 3.0, 0.1)  # 3 s of data in a 16 s window
    tr = tracks_from_rows(_rows(0, t, np.full(len(t), 100.0)))[0]
    assert featurize_window(tr, 16.0, SPEC) is None


# --------------------------------------------------------------- per-video assembly
def test_windows_for_video_labels_owner_and_excludes_bystander() -> None:
    fps = 10.0
    t = np.arange(0, 40.0, 1 / fps)
    faller = _rows(0, t, np.full(len(t), 100.0))
    bystander = _rows(7, t, np.full(len(t), 60.0))
    tracks = tracks_from_rows(faller + bystander)
    event = (20.0, 22.0)
    wins = windows_for_video("v", tracks, event, SPEC)
    own = [w for w in wins if w.track_id == 0]
    other = [w for w in wins if w.track_id == 7]
    assert any(w.label == 1 for w in own)
    # bystander windows overlapping the event were excluded, not negated
    for w in other:
        assert w.label == 0
        assert label_window(w.t_end - SPEC.window_s, w.t_end, event, SPEC) == 0
    pos = [w for w in own if w.label == 1]
    for w in pos:  # a positive window really contains the event
        assert w.t_end - SPEC.window_s <= event[0] and event[1] <= w.t_end


def test_windows_for_video_no_event_is_all_negative() -> None:
    t = np.arange(0, 30.0, 0.1)
    tracks = tracks_from_rows(_rows(0, t, np.full(len(t), 100.0)))
    wins = windows_for_video("v", tracks, None, SPEC)
    assert wins and all(w.label == 0 for w in wins)


# --------------------------------------------------------------- synth integration
def test_windows_from_synth_extraction(tmp_path) -> None:
    synth.write_synth_dataset(tmp_path, n_positive=3, n_negative_per_kind=1, seed=5)
    spec = WindowSpec(window_s=12.0, stride_s=3.0, policy="overlap", overlap_min_s=4.0)
    wins = windows_from_extraction(
        tmp_path, spec, event_keys=("convulsion_start_t", "convulsion_end_t")
    )
    labels = {w.label for w in wins}
    assert labels == {0, 1}
    for w in wins:
        assert w.x.shape == (spec.n_steps, len(CHANNELS))
        assert w.x.dtype == np.float32


def test_windows_from_extraction_requires_meta(tmp_path) -> None:
    with pytest.raises(FeatureError, match="no meta records"):
        windows_from_extraction(tmp_path, SPEC)


def test_load_rows_round_trip(tmp_path) -> None:
    p = tmp_path / "seq.jsonl"
    p.write_text('{"f":0,"t":0.1,"track":2,"cx":1,"cy":2,"w":3,"h":4,"angle":null,"kpc":0.5}\n')
    rows = load_rows(p)
    tr = tracks_from_rows(rows)[0]
    assert tr.track_id == 2 and np.isnan(tr.angle[0])


def test_short_clip_yields_one_padded_window_with_the_fall() -> None:
    # UR Fall clips run 3-13 s; a 3.7 s clip must still produce a scoreable window.
    fps = 30.0
    t = np.arange(0, 3.7, 1 / fps)
    tracks = tracks_from_rows(_rows(0, t, np.full(len(t), 100.0)))
    spec = WindowSpec(window_s=8.0, stride_s=1.5, policy="contain")
    wins = windows_for_video("v", tracks, (1.3, 2.27), spec)
    assert len(wins) == 1
    assert wins[0].label == 1 and wins[0].t_end == pytest.approx(t[-1], abs=1e-3)


def test_multi_event_video_labels_both_falls() -> None:
    # multicam_fall chute04/chute23 stage two falls in one recording.
    t = np.arange(0, 60.0, 0.1)
    tracks = tracks_from_rows(_rows(0, t, np.full(len(t), 100.0)))
    spec = WindowSpec(window_s=8.0, stride_s=1.0, policy="contain")
    events = [(10.0, 11.0), (40.0, 41.5)]
    wins = windows_for_video("v", tracks, events, spec)
    pos_ends = sorted(w.t_end for w in wins if w.label == 1)
    assert any(te < 20 for te in pos_ends) and any(te > 40 for te in pos_ends)
    # windows between the falls (no overlap with either) are negatives
    assert any(w.label == 0 and 20 < w.t_end < 39 for w in wins)


def test_windows_from_extraction_prefers_events_t(tmp_path) -> None:
    import json as _json

    (tmp_path / "sequences").mkdir()
    (tmp_path / "meta").mkdir()
    t = np.arange(0, 30.0, 0.1)
    rows = _rows(0, t, np.full(len(t), 100.0))
    (tmp_path / "sequences" / "v.jsonl").write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    (tmp_path / "meta" / "v.json").write_text(
        _json.dumps({"key": "v", "events_t": [[5.0, 6.0], [20.0, 21.0]], "fps": 10.0})
    )
    spec = WindowSpec(window_s=8.0, stride_s=1.0, policy="contain")
    wins = windows_from_extraction(tmp_path, spec)
    assert sum(w.label == 1 for w in wins) >= 2


def test_windows_from_extraction_no_event_only_keeps_quiet_videos(tmp_path) -> None:
    # The quiet-clip selector for negative diets over mixed extractions (caucafall's
    # ADLs share subject locations with its falls, so `locations` cannot split them).
    import json as _json

    (tmp_path / "sequences").mkdir()
    (tmp_path / "meta").mkdir()
    t = np.arange(0, 30.0, 0.1)
    rows = _rows(0, t, np.full(len(t), 100.0))
    seq_text = "\n".join(_json.dumps(r) for r in rows) + "\n"
    for key, meta in (
        ("fall", {"key": "fall", "events_t": [[5.0, 6.0]], "fps": 10.0}),
        ("adl", {"key": "adl", "events_t": [], "fps": 10.0}),
    ):
        (tmp_path / "sequences" / f"{key}.jsonl").write_text(seq_text)
        (tmp_path / "meta" / f"{key}.json").write_text(_json.dumps(meta))
    spec = WindowSpec(window_s=8.0, stride_s=1.0, policy="contain")
    quiet = windows_from_extraction(tmp_path, spec, no_event_only=True)
    assert quiet and {w.key for w in quiet} == {"adl"}
    assert all(w.label == 0 for w in quiet)
    both = windows_from_extraction(tmp_path, spec)
    assert {w.key for w in both} == {"adl", "fall"}


# --------------------------------------------------------------- scene channels
def test_scene_channels_shape_density_and_births() -> None:
    spec = WindowSpec(
        window_s=16.0, stride_s=2.0, n_steps=64, policy="contain", scene_channels=True
    )
    fps = 10.0
    t_full = np.arange(0, 16.0, 1 / fps)
    rows = _rows(0, t_full, np.full(len(t_full), 100.0))
    # a second track born mid-window (a "fragment"), alive 8..16 s
    t_frag = t_full[t_full >= 8.0]
    rows += _rows(1, t_frag, np.full(len(t_frag), 100.0))
    tracks = tracks_from_rows(rows)
    wins = windows_for_video("v", tracks, None, spec)
    assert wins, "expected at least one window"
    x = wins[0].x
    assert x.shape == (spec.n_steps, len(CHANNELS) + len(SCENE_CHANNELS))
    density = x[:, len(CHANNELS) + SCENE_CHANNELS.index("scene_n_tracks")]
    births = x[:, len(CHANNELS) + SCENE_CHANNELS.index("scene_births")]
    # first half: 1 active track (1/8); second half: 2 (2/8)
    assert density[:30].max() == pytest.approx(1 / 8)
    assert density[-20:].min() == pytest.approx(2 / 8)
    # the fragment's birth at 8 s registers for SCENE_BIRTH_WINDOW_S afterwards

    def step_of(second: float) -> int:
        return int(spec.n_steps * second / spec.window_s)

    assert births[step_of(8.5)] == pytest.approx(1 / 4)
    assert births[step_of(12.5)] == 0.0
    # birth of the original track at t=0 is inside the first steps' lookback
    assert births[0] > 0


def test_scene_channels_refused_without_context() -> None:
    spec = WindowSpec(window_s=16.0, policy="contain", scene_channels=True)
    t = np.arange(0, 16.0, 0.1)
    tr = tracks_from_rows(_rows(0, t, np.full(len(t), 100.0)))[0]
    with pytest.raises(FeatureError, match="SceneContext"):
        featurize_window(tr, 16.0, spec)


def test_default_spec_keeps_eight_channels() -> None:
    t = np.arange(0, 16.0, 0.1)
    tr = tracks_from_rows(_rows(0, t, np.full(len(t), 100.0)))[0]
    x = featurize_window(tr, 16.0, SPEC)
    assert x is not None and x.shape[1] == len(CHANNELS)


# --------------------------------------------------------------- rhythmicity
# A convulsion is rhythmic motion in roughly 1-5 Hz; the normal in-bed movement it
# has to be told apart from carries the same amplitude with no periodicity. These
# tests pin that the channels actually separate the two, and that they refuse to
# answer when the capture rate cannot resolve the band.
def _rhythm_track(
    fs: float, seconds: float, *, hz: float | None, amp: float = 8.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, cx, cy): sinusoidal jerking at `hz`, or an aperiodic random walk."""
    t = np.arange(0.0, seconds, 1.0 / fs)
    rng = np.random.default_rng(seed)
    if hz is None:
        cy = 200.0 + np.cumsum(rng.normal(0.0, amp * 0.35, t.size))
        cx = 160.0 + np.cumsum(rng.normal(0.0, amp * 0.35, t.size))
    else:
        cy = 200.0 + amp * np.sin(2.0 * np.pi * hz * t)
        cx = 160.0 + amp * np.sin(2.0 * np.pi * hz * t + 0.4)
    return t, cx, cy


def test_rhythm_finds_the_frequency_it_was_given() -> None:
    from vividly.verifier.features import RHYTHM_FMAX_HZ, rhythm_features

    t, cx, cy = _rhythm_track(30.0, 12.0, hz=3.0)
    strength, freq_norm, band_ratio, valid = rhythm_features(t, cx, cy, 100.0)
    assert valid == 1.0
    assert freq_norm * RHYTHM_FMAX_HZ == pytest.approx(3.0, abs=0.3)
    # A pure oscillation puts nearly all of its non-DC power in one in-band line.
    assert strength > 0.5
    assert band_ratio > 0.9


def test_rhythm_separates_convulsive_from_restless_motion() -> None:
    """The whole point of the channels: same amplitude, different periodicity."""
    from vividly.verifier.features import rhythm_features

    rhythmic = rhythm_features(*_rhythm_track(30.0, 12.0, hz=3.0), 100.0)
    restless = rhythm_features(*_rhythm_track(30.0, 12.0, hz=None, seed=7), 100.0)
    assert rhythmic[3] == restless[3] == 1.0  # both measurable
    assert rhythmic[0] > 4.0 * restless[0]  # peak concentration
    assert rhythmic[2] > restless[2]  # in-band share


def test_rhythm_refuses_when_capture_rate_cannot_resolve_the_band() -> None:
    """Below 2 Hz the band's own floor is past Nyquist: report nothing, flag invalid.

    This is the live-rig case (a stream processed at a few fps), and the failure it
    prevents is a model reading a folded artefact as evidence of rhythm.
    """
    from vividly.verifier.features import rhythm_features

    t, cx, cy = _rhythm_track(1.5, 20.0, hz=3.0)
    assert rhythm_features(t, cx, cy, 100.0) == (0.0, 0.0, 0.0, 0.0)


def test_rhythm_refuses_degenerate_inputs() -> None:
    from vividly.verifier.features import RHYTHM_MIN_SAMPLES, rhythm_features

    t, cx, cy = _rhythm_track(30.0, 12.0, hz=3.0)
    short = slice(0, RHYTHM_MIN_SAMPLES - 1)
    assert rhythm_features(t[short], cx[short], cy[short], 100.0) == (0.0, 0.0, 0.0, 0.0)
    assert rhythm_features(t, cx, cy, 0.0) == (0.0, 0.0, 0.0, 0.0)


def test_rhythm_channels_are_opt_in_and_appended_last() -> None:
    from vividly.verifier.features import RHYTHM_CHANNELS, tracks_from_rows

    fs, seconds = 30.0, 20.0
    t, _cx, cy = _rhythm_track(fs, seconds, hz=3.0)
    tracks = tracks_from_rows(_rows(1, t, cy))
    tr = tracks[0]
    off = WindowSpec(window_s=12.0, stride_s=3.0, n_steps=64, policy="overlap")
    on = WindowSpec(window_s=12.0, stride_s=3.0, n_steps=64, policy="overlap", rhythm_channels=True)
    x_off = featurize_window(tr, 15.0, off)
    x_on = featurize_window(tr, 15.0, on)
    assert x_off is not None and x_on is not None
    assert x_off.shape == (64, len(CHANNELS))  # untouched: v1-v6 geometry is frozen
    assert x_on.shape == (64, len(CHANNELS) + len(RHYTHM_CHANNELS))
    np.testing.assert_allclose(x_on[:, : len(CHANNELS)], x_off, rtol=0, atol=0)
    # Window-level property: constant down the window, and it found the rhythm.
    tail = x_on[:, len(CHANNELS) :]
    assert np.all(tail == tail[0])
    assert tail[0, 3] == 1.0


# --------------------------------------------------------------- limb channels
def _kp_rows(
    track: int,
    t: np.ndarray,
    *,
    limb_hz: float = 3.0,
    bilateral: bool = True,
    h: float = 100.0,
    conf: float = 0.8,
    seed: int = 7,
) -> list[dict]:
    """Rows whose box never moves while the wrists and ankles oscillate.

    This is the seizure lane's actual failure case in miniature: a patient convulsing
    in bed thrashes their limbs inside a nearly stationary bounding box, so every
    box-derived signal, including RHYTHM_CHANNELS, sees a still person.
    """
    rng = np.random.default_rng(seed)
    cx, cy = 160.0, 200.0
    amp = 0.15 * h
    rows = []
    for i, ti in enumerate(t):
        d = amp * float(np.sin(2.0 * np.pi * limb_hz * float(ti)))
        # The quiet side of a one-sided motion still jitters; nothing in real footage
        # is perfectly static, and a zero-variance series has no defined correlation.
        quiet = float(rng.normal(0.0, 0.002 * h))
        other = d if bilateral else quiet
        kp = [[cx, cy, conf] for _ in range(17)]
        kp[5] = [cx - 0.20 * h, cy - 0.20 * h, conf]  # shoulders
        kp[6] = [cx + 0.20 * h, cy - 0.20 * h, conf]
        kp[11] = [cx - 0.15 * h, cy + 0.20 * h, conf]  # hips
        kp[12] = [cx + 0.15 * h, cy + 0.20 * h, conf]
        kp[9] = [cx - 0.35 * h, cy + d, conf]  # left wrist
        kp[10] = [cx + 0.35 * h, cy + other, conf]  # right wrist
        kp[15] = [cx - 0.25 * h, cy + 0.5 * h + d, conf]  # left ankle
        kp[16] = [cx + 0.25 * h, cy + 0.5 * h + other, conf]  # right ankle
        rows.append(
            {
                "f": i,
                "t": round(float(t[i]), 4),
                "track": track,
                "cx": cx,
                "cy": cy,
                "w": h * 0.5,
                "h": h,
                "angle": 10.0,
                "kpc": conf,
                "kp": kp,
            }
        )
    return rows


def _kp_track(**kw):
    from vividly.verifier.features import tracks_from_rows

    t = np.arange(0.0, 20.0, 1.0 / 30.0)
    return tracks_from_rows(_kp_rows(1, t, **kw))[0]


def test_limb_channels_are_opt_in_and_appended_last() -> None:
    from vividly.verifier.features import LIMB_CHANNELS

    tr = _kp_track()
    base = dict(window_s=12.0, stride_s=3.0, n_steps=64, policy="overlap")
    x_off = featurize_window(tr, 15.0, WindowSpec(**base))
    x_on = featurize_window(tr, 15.0, WindowSpec(**base, limb_channels=True))
    assert x_off is not None and x_on is not None
    assert x_off.shape == (64, len(CHANNELS))  # untouched: v1-v6 geometry is frozen
    assert x_on.shape == (64, len(CHANNELS) + len(LIMB_CHANNELS))
    np.testing.assert_allclose(x_on[:, : len(CHANNELS)], x_off, rtol=0, atol=0)
    tail = x_on[:, len(CHANNELS) :]
    assert np.all(tail == tail[0])  # a window property, constant down the window
    assert tail[0, LIMB_CHANNELS.index("limb_valid")] == 1.0


def test_limb_rhythm_finds_the_motion_the_box_centre_cannot() -> None:
    """The measured reason this feature exists (2026-08-28): centre rhythm is blind."""
    from vividly.verifier.features import RHYTHM_FMAX_HZ, limb_features, rhythm_features

    tr = _kp_track(limb_hz=3.0)
    sel = (tr.t >= 3.0) & (tr.t <= 15.0)
    t, h_ref = tr.t[sel], 100.0

    # The box never moved, so every centre-derived signal is dead, matching the recorded
    # results on the WU-SAHZU test split: the centre-rhythm channels alone score ROC 0.429,
    # 0.494 and 0.412 over 4,880 windows, 285 positive, and the second probe of the
    # box-shape signals spans 0.347 to 0.530 over 5,455 windows, 891 positive
    # (research/event-verifier-v1-record.md).
    assert rhythm_features(t, tr.cx[sel], tr.cy[sel], h_ref) == (0.0, 0.0, 0.0, 0.0)

    strength, freq, band_ratio, excursion, _bilateral, valid = limb_features(t, tr.kpts[sel], h_ref)
    assert valid == 1.0
    assert abs(freq * RHYTHM_FMAX_HZ - 3.0) < 0.05  # recovers the true limb frequency
    assert strength > 0.5 and band_ratio > 0.9  # a clean tone, essentially all in band
    assert excursion > 0.0


def test_limb_bilateral_separates_two_sided_from_one_sided_motion() -> None:
    from vividly.verifier.features import limb_features

    def _bilateral(**kw) -> float:
        tr = _kp_track(**kw)
        sel = (tr.t >= 3.0) & (tr.t <= 15.0)
        return limb_features(tr.t[sel], tr.kpts[sel], 100.0)[4]

    both = _bilateral(bilateral=True)
    one = _bilateral(bilateral=False)
    assert both > one, f"two-sided {both:.3f} should beat one-sided {one:.3f}"
    assert both > 0.9  # in-phase limbs agree almost perfectly


def test_limb_features_refuse_what_they_cannot_measure() -> None:
    from vividly.verifier.features import LIMB_CONF_MIN, limb_features

    dead = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    tr = _kp_track()
    sel = (tr.t >= 3.0) & (tr.t <= 15.0)
    t, kp = tr.t[sel], tr.kpts[sel]

    assert limb_features(t, None, 100.0) == dead  # a pre-v2kp tree carries no joints
    assert limb_features(t[:4], kp[:4], 100.0) == dead  # too few samples
    assert limb_features(t, kp, 0.0) == dead  # no scale to normalise against

    unsure = kp.copy()
    unsure[:, :, 2] = LIMB_CONF_MIN - 0.01  # the estimator was guessing throughout
    assert limb_features(t, unsure, 100.0) == dead


def test_pre_v2kp_rows_still_load_and_report_limb_valid_zero() -> None:
    """Every extraction before v2kp wrote a 9-field row; those trees must still load."""
    from vividly.verifier.features import LIMB_CHANNELS, tracks_from_rows

    t = np.arange(0.0, 20.0, 1.0 / 30.0)
    tr = tracks_from_rows(_rows(1, t, np.full(len(t), 200.0)))[0]
    assert tr.kpts is None
    spec = WindowSpec(window_s=12.0, stride_s=3.0, n_steps=64, policy="overlap", limb_channels=True)
    x = featurize_window(tr, 15.0, spec)
    assert x is not None
    assert x.shape == (64, len(CHANNELS) + len(LIMB_CHANNELS))
    assert np.all(x[:, len(CHANNELS) :] == 0.0)  # valid=0 forces every sibling to 0


def test_limb_conf_gate_rejects_joints_the_estimator_is_unsure_about() -> None:
    """The gate is the whole experiment: same motion, different confidence, different answer.

    Wrist and ankle confidences over every track in wu_sahzu_tracks_v2kp sit at a 0.305
    median (research/seizure-limb-channel-gate.md), a pooled figure that averages real
    people with spurious tracks; wrists alone on tracks with mean kpc at or above 0.5 sit
    at 0.579. A gate below the pooled median admits noise and the spectral channels
    measure the noise. A limb resolved at 0.4 must survive a 0.2 gate and be refused by a
    0.5 one, with `valid` carrying the refusal rather than a quietly wrong number.
    """
    from vividly.verifier.features import limb_features

    t = np.arange(0.0, 12.0, 1.0 / 30.0)
    rows = _kp_rows(1, t, conf=0.4)
    tr = tracks_from_rows(rows)[0]

    loose = limb_features(tr.t, tr.kpts, 100.0, 0.2)
    strict = limb_features(tr.t, tr.kpts, 100.0, 0.5)

    assert loose[5] == 1.0  # valid: 0.4 clears a 0.2 gate
    assert loose[0] > 0.5  # and the 3 Hz oscillation is found
    assert strict == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # 0.4 cannot clear a 0.5 gate


def test_window_spec_gate_reaches_the_feature_matrix() -> None:
    """A spec-level gate must actually change the emitted limb columns, not just validate."""
    from vividly.verifier.features import LIMB_CHANNELS, tracks_from_rows

    t = np.arange(0.0, 20.0, 1.0 / 30.0)
    tr = tracks_from_rows(_kp_rows(1, t, conf=0.4))[0]
    base = dict(window_s=12.0, stride_s=3.0, n_steps=64, policy="overlap", limb_channels=True)

    loose = featurize_window(tr, 10.0, WindowSpec(**base, limb_conf_min=0.2))
    strict = featurize_window(tr, 10.0, WindowSpec(**base, limb_conf_min=0.5))
    assert loose is not None and strict is not None

    limb = slice(-len(LIMB_CHANNELS), None)
    assert loose[0, limb][-1] == 1.0  # limb_valid
    assert np.all(strict[:, limb] == 0.0)  # gated out entirely
