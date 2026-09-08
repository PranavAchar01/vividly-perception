"""ONNX event-verifier backend tests: the acute-event verifier seam
(vividly.models.event_verifier).

The load-bearing claims: the backend implements the EventVerifier protocol and answers
"unavailable" (never a guess, never a raise) for unknown rules, missing tracks and thin
evidence; the scored window is anchored at the firing's onset_t; and, the claim that
keeps every threshold meaningful, the features fed to the graph are identical to what
the training extraction + featurization would produce for the same observations, proven
by running both paths on one trajectory and comparing arrays exactly.

No onnxruntime here: heads take anything with session.run(), so a recording fake stands
in. Real-graph loading (report.json + sha check) is covered through its refusal paths.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from vividly.models.event_verifier import (
    DEFAULT_PRE_PAD_S,
    SCENE_RETENTION_S,
    OnnxEventVerifier,
    VerifierBackendError,
    VerifierHead,
    load_head,
)
from vividly.rules.base import RuleFiring
from vividly.rules.verify import EventVerifier
from vividly.tracks import NUM_KEYPOINTS, TrackConfig, TrackObservation, TrackState
from vividly.verifier.features import WindowSpec, featurize_window, tracks_from_rows

KP_CONF_MIN = 0.3
FPS = 10.0


def kps(angle_deg: float, conf: float = 0.9) -> tuple[tuple[float, float, float], ...]:
    """COCO-17 keypoints whose torso tilts angle_deg from vertical."""
    out = [(0.0, 0.0, conf)] * NUM_KEYPOINTS
    dx = 40.0 * math.sin(math.radians(angle_deg))
    dy = 40.0 * math.cos(math.radians(angle_deg))
    out[5] = (-10.0, 0.0, conf)  # shoulders
    out[6] = (10.0, 0.0, conf)
    out[11] = (-10.0 + dx, dy, conf)  # hips
    out[12] = (10.0 + dx, dy, conf)
    return tuple(out)


def fall_track(track_id: int = 3) -> TrackState:
    """A synthetic collapse at t=10: upright drift, a 1 s drop of two box heights with
    the torso going horizontal, then 30 s of lying still. Values carry sub-rounding
    digits on purpose, because the parity test must exercise the extraction's rounding."""
    cfg = TrackConfig(kp_conf_min=KP_CONF_MIN, ring_size=2000)
    st = TrackState(track_id, cfg)
    t = 0.0
    while t < 45.0:
        if t < 10.0:  # upright, slight drift
            cy, angle, h = 100.0 + 0.3337 * t, 5.0, 80.111
        elif t <= 11.0:  # the drop
            frac = t - 10.0
            cy, angle, h = 100.0 + 160.0 * frac, 5.0 + 80.0 * frac, 80.111 - 20.0 * frac
        else:  # down and still
            cy, angle, h = 260.0, 85.0, 60.111
        st.update(
            TrackObservation(
                track_id=track_id,
                t=round(t, 4),
                cx=160.123456,
                cy=cy,
                w=40.667,
                h=h,
                keypoints=kps(angle),
            )
        )
        t += 1.0 / FPS
    return st


class RecordingSession:
    """Stands in for ort.InferenceSession: fixed logit out, feeds recorded."""

    def __init__(self, logit: float) -> None:
        self.logit = logit
        self.feeds: list[np.ndarray] = []

    def run(self, outputs, feeds):  # onnxruntime's calling shape
        assert outputs is None
        x = feeds["windows"]
        self.feeds.append(x)
        return [np.full((x.shape[0], 1), self.logit, dtype=np.float32)]


def head(session, **overrides) -> VerifierHead:
    kwargs = {
        "session": session,
        "spec": WindowSpec(window_s=8.0, stride_s=1.5, n_steps=64, policy="contain"),
        "task": "fall",
        "weight_class": "prototype",
        "sha8": "deadbeef",
    }
    kwargs.update(overrides)
    return VerifierHead(**kwargs)


def firing(rule_id: str = "fall_then_still_v0", onset_t: float = 10.0) -> RuleFiring:
    return RuleFiring(
        rule_id=rule_id,
        rule_version="0.1.0",
        track_id=3,
        onset_t=onset_t,
        detected_t=onset_t + 30.0,
        baseline=None,
        details={},
    )


# ----------------------------------------------------------------- protocol
def test_backend_satisfies_the_protocol_and_names_its_weights() -> None:
    v = OnnxEventVerifier({"fall_then_still_v0": head(RecordingSession(0.0))})
    assert isinstance(v, EventVerifier)
    assert v.verifier_version == "fall_then_still_v0:deadbeef"


def test_no_heads_is_refused() -> None:
    with pytest.raises(VerifierBackendError, match="at least one head"):
        OnnxEventVerifier({})


# ----------------------------------------------------------------- verdicts
def test_high_score_confirms_low_score_rejects() -> None:
    track = fall_track()
    confirm = OnnxEventVerifier({"fall_then_still_v0": head(RecordingSession(2.0))})
    verdict = confirm.verify(firing(), track)
    assert verdict.verdict == "confirmed"
    assert verdict.confidence == pytest.approx(1.0 / (1.0 + math.exp(-2.0)), abs=1e-5)

    reject = OnnxEventVerifier({"fall_then_still_v0": head(RecordingSession(-2.0))})
    verdict = reject.verify(firing(), track)
    assert verdict.verdict == "rejected"
    assert verdict.confidence == pytest.approx(1.0 / (1.0 + math.exp(2.0)), abs=1e-5)


def test_unknown_rule_and_missing_track_are_unavailable() -> None:
    v = OnnxEventVerifier({"fall_then_still_v0": head(RecordingSession(2.0))})
    assert v.verify(firing("egress_without_discharge_v0"), fall_track()).verdict == "unavailable"
    assert v.verify(firing(), None).verdict == "unavailable"


def test_thin_evidence_is_unavailable_not_a_vote() -> None:
    # Half a second of track against an 8 s window: featurize_window abstains.
    cfg = TrackConfig(kp_conf_min=KP_CONF_MIN, ring_size=2000)
    st = TrackState(3, cfg)
    for i in range(5):
        st.update(TrackObservation(track_id=3, t=i / FPS, cx=0.0, cy=0.0, w=40.0, h=80.0))
    v = OnnxEventVerifier({"fall_then_still_v0": head(RecordingSession(2.0))})
    verdict = v.verify(firing(onset_t=0.2), st)
    assert verdict.verdict == "unavailable"
    assert verdict.confidence is None


def test_window_far_before_the_ring_is_unavailable() -> None:
    v = OnnxEventVerifier({"fall_then_still_v0": head(RecordingSession(2.0))})
    assert v.verify(firing(onset_t=-500.0), fall_track()).verdict == "unavailable"


# ----------------------------------------------------------------- features
def test_model_input_shape_and_dtype() -> None:
    session = RecordingSession(2.0)
    v = OnnxEventVerifier({"fall_then_still_v0": head(session)})
    v.verify(firing(), fall_track())
    (x,) = session.feeds
    assert x.shape == (1, 64, 8)
    assert x.dtype == np.float32


def test_features_match_the_training_extraction_path_exactly() -> None:
    """The parity claim: rebuild the rows the extraction writer would have written for
    the same observations (same rounding), run the training featurization directly, and
    the backend must feed the graph that exact array."""
    track = fall_track()
    session = RecordingSession(2.0)
    spec = WindowSpec(window_s=8.0, stride_s=1.5, n_steps=64, policy="contain")
    v = OnnxEventVerifier({"fall_then_still_v0": head(session, spec=spec)})
    v.verify(firing(onset_t=10.0), track)
    (fed,) = session.feeds

    rows = []
    for (t, cx, cy), (_, w, h, kpc) in zip(track.positions(1e6), track.boxes(1e6), strict=True):
        angle = dict(track.posture_angles(1e6)).get(t)
        rows.append(
            {
                "t": round(t, 4),
                "track": track.track_id,
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "w": round(w, 2),
                "h": round(h, 2),
                "angle": None if angle is None else round(angle, 2),
                "kpc": None if math.isnan(kpc) else round(kpc, 4),
            }
        )
    t_end = 10.0 - DEFAULT_PRE_PAD_S + spec.window_s
    expected = featurize_window(tracks_from_rows(rows)[0], t_end, spec)
    assert expected is not None
    np.testing.assert_array_equal(fed[0], expected.astype(np.float32))


# ------------------------------------------------------------- scene channels
def scene_spec() -> WindowSpec:
    return WindowSpec(window_s=8.0, stride_s=1.5, n_steps=64, policy="contain", scene_channels=True)


class FakeBook:
    """TrackBook stand-in: just the tracks() mapping observe_book reads."""

    def __init__(self, states: list[TrackState]) -> None:
        self._states = {s.track_id: s for s in states}

    def tracks(self) -> dict[int, TrackState]:
        return dict(self._states)


def span_state(track_id: int, first: float, last: float) -> TrackState:
    """A track whose only relevant property is its observed span."""
    st = TrackState(track_id, TrackConfig())
    st.first_seen_t, st.last_seen_t = first, last
    return st


def test_scene_head_scores_ten_channels_from_the_observed_book() -> None:
    session = RecordingSession(2.0)
    v = OnnxEventVerifier({"fall_then_still_v0": head(session, spec=scene_spec())})
    tr = fall_track()
    v.observe_book(FakeBook([tr, span_state(9, 0.0, 45.0), span_state(10, 12.5, 45.0)]), now=45.0)
    assert v.verify(firing(), tr).verdict == "confirmed"
    (x,) = session.feeds
    assert x.shape == (1, 64, 10)
    # Window [8, 16]: two tracks span it whole, one is born at 12.5. Density steps
    # 2/8 -> 3/8 at the birth; the birth column holds 1/4 for SCENE_BIRTH_WINDOW_S.
    steps = 8.0 + (np.arange(64) + 0.5) * (8.0 / 64)
    dens, births = x[0, :, 8], x[0, :, 9]
    assert np.allclose(dens[steps < 12.5], 2.0 / 8.0)
    assert np.allclose(dens[steps >= 12.5], 3.0 / 8.0)
    assert np.allclose(births[steps < 12.5], 0.0)
    assert np.allclose(births[(steps >= 12.5) & (steps <= 15.5)], 1.0 / 4.0)


def test_scene_head_refuses_when_the_engine_never_handed_the_book() -> None:
    v = OnnxEventVerifier({"fall_then_still_v0": head(RecordingSession(2.0), spec=scene_spec())})
    with pytest.raises(VerifierBackendError, match="observe_book"):
        v.verify(firing(), fall_track())


def test_scene_memory_outlives_the_reap_but_not_retention() -> None:
    session = RecordingSession(2.0)
    v = OnnxEventVerifier({"fall_then_still_v0": head(session, spec=scene_spec())})
    tr = fall_track()
    # A visitor observed early, then gone from the book (reaped) by firing time: still
    # counted in the window they overlapped.
    v.observe_book(FakeBook([tr, span_state(9, 0.0, 14.0)]), now=14.0)
    v.observe_book(FakeBook([tr]), now=45.0)
    v.verify(firing(), tr)
    assert np.allclose(session.feeds[0][0, :, 8].max(), 2.0 / 8.0)
    # ...but pruned once SCENE_RETENTION_S passes since they were last seen.
    v.observe_book(FakeBook([tr]), now=14.0 + SCENE_RETENTION_S + 1.0)
    v.verify(firing(), tr)
    assert np.allclose(session.feeds[1][0, :, 8].max(), 1.0 / 8.0)


def test_plain_heads_ignore_the_scene_memory_entirely() -> None:
    session = RecordingSession(2.0)
    v = OnnxEventVerifier({"fall_then_still_v0": head(session)})
    v.observe_book(FakeBook([span_state(9, 0.0, 45.0)]), now=45.0)
    v.verify(firing(), fall_track())
    assert session.feeds[0].shape == (1, 64, 8)


# ----------------------------------------------------------------- load_head
def test_load_head_requires_a_report(tmp_path: Path) -> None:
    with pytest.raises(VerifierBackendError, match=r"report\.json"):
        load_head(tmp_path)


def test_verify_applies_the_calibration_temperature() -> None:
    """score = sigmoid(logit / T), matching the eval harness; a live path that skipped
    the division would put a calibrated model's thresholds on a different scale than
    every number measured for it."""
    track = fall_track()
    raw = OnnxEventVerifier({"fall_then_still_v0": head(RecordingSession(2.0))})
    calibrated = OnnxEventVerifier(
        {"fall_then_still_v0": head(RecordingSession(2.0), temperature=2.5)}
    )
    assert raw.verify(firing(), track).confidence == pytest.approx(
        1.0 / (1.0 + math.exp(-2.0)), abs=1e-6
    )
    assert calibrated.verify(firing(), track).confidence == pytest.approx(
        1.0 / (1.0 + math.exp(-2.0 / 2.5)), abs=1e-6
    )


def test_load_head_reads_the_calibration_block(tmp_path: Path) -> None:
    """The calibration block is parsed (and validated) before the graph is even opened:
    a nonsense temperature is refused with the report named, not silently defaulted."""
    # Only the invalid leg is exercised here: load_head raises on the calibration block
    # before it opens the graph, so onnxruntime is never reached and no import is needed.
    import hashlib

    blob = b"bytes that never reach onnxruntime"
    (tmp_path / "verifier.onnx").write_bytes(blob)
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "task": "convulsion",
                "vividly_weight_class": "prototype",
                "config": {"window": {"window_s": 12.0, "policy": "overlap"}},
                "onnx": {"file": "verifier.onnx", "sha256": hashlib.sha256(blob).hexdigest()},
                "calibration": {"temperature": -1.0},
            }
        )
    )
    with pytest.raises(VerifierBackendError, match="temperature"):
        load_head(tmp_path)


def test_load_head_names_git_lfs_pull_for_a_pointer_file(tmp_path: Path) -> None:
    """A checkout made without `git lfs pull` holds a three-line pointer where the graph
    should be. That is refused by name, before hashing, so the fix is in the message rather
    than a digest mismatch the reader has to decode."""
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 177690\n"
    )
    (tmp_path / "verifier.onnx").write_bytes(pointer)
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "task": "convulsion",
                "vividly_weight_class": "prototype",
                "config": {"window": {"window_s": 12.0, "policy": "overlap"}},
                "onnx": {"file": "verifier.onnx", "sha256": "0" * 64},
            }
        )
    )
    with pytest.raises(VerifierBackendError, match="git lfs pull"):
        load_head(tmp_path)


def test_load_head_refuses_a_sha_mismatch(tmp_path: Path) -> None:
    (tmp_path / "verifier.onnx").write_bytes(b"not the trained graph")
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "task": "fall",
                "vividly_weight_class": "prototype",
                "config": {"window": {"window_s": 8.0, "policy": "contain"}},
                "onnx": {"file": "verifier.onnx", "sha256": "0" * 64},
            }
        )
    )
    with pytest.raises(VerifierBackendError, match="sha256"):
        load_head(tmp_path)
