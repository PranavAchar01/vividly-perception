"""Deterministic trajectory generators for rule unit tests. No video, no RNG, no clock.

What these are. Closed-form trajectories: a few sinusoids, a ramp, an alternation,
evaluated on a fixed time grid. Every generator returns the same list of TrackObservation
batches every time it is called, on every machine, with no seed to get wrong, and
produces a simulated hour in well under a second because there is nothing in it but
arithmetic.

What these are not. They are not medical events, not recordings of people, and not
training or evaluation data. POLICY.md, rule 4 forbids simulated or acted medical events in
any training or evaluation set, including synthetic renders, and this material is neither:
these are abstract points moving on a plane, used to check that onset arithmetic, window
edges, cascade staging, debounce and track identity behave as specified. That the
cascade fixtures trace a fall-shaped or oscillation-shaped curve does not change what
they are: a threshold test needs a trajectory on both sides of the threshold, and a
closed-form curve is not an acted event, a rendering of a person, or data any model may
train or be evaluated on. docs/CALIBRATION.md says the same thing in its own words, and
the boundary that rule draws (unit-test trajectories against training and evaluation data)
is recorded for counsel in CQ-21. Nothing measured here is evidence about a person, and no
firing count from these fixtures is a false-alarm rate.

Scene. One seated person (PATIENT_TRACK_ID) plus, by default, one second person
(STAFF_TRACK_ID) walking a circular patrol. The patrol track predates the acute-event
re-scope (it existed for the removed proximity rule) and stays: a scene with a second,
always-moving body in it is the standing proof that neither event rule fires on ordinary
locomotion, and the dashboard's staff tag still needs a track to be tagged on. The patrol
never enters the EXIT_ZONE_POLYGON at the right-hand edge of the frame (x >= 1640; the
patrol stays below x = 1260), so in the egress scenes it is also the standing proof that a
person who walks around all day and never leaves is not a departure. A patient generator
may return None for an instant, meaning the person is not observed then (they left); the
scene keeps ticking on the patrol so the clock keeps advancing.

The caller tags the staff track: these generators produce observations, and
TrackBook.set_staff is what makes a track staff. That is deliberate, because the tag is a
manual dashboard action in v0 and nothing in the observation stream carries it.

Time. Observations are generated at `hz` (default 1.0, that is one batch per simulated
second) starting at `start_t`. The event fixtures are meaningful at higher rates too
(the dashboard scene runs them at 5 Hz); every generator takes `hz`. Simulated time only.
Nothing here sleeps.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from vividly.tracks import (
    LEFT_HIP,
    LEFT_SHOULDER,
    NUM_KEYPOINTS,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    Keypoints,
    TrackObservation,
)

# Scene geometry, in pixels on a nominal 1920x1080 source frame.
# The egress scenes' exit: a door region at the right-hand edge, and the point inside it
# a departing person walks to before the tracker loses them. Mirrors the zone declared in
# configs/pipeline/egress_demo.yaml; tests/test_egress.py pins the two equal.
EXIT_ZONE_POLYGON: tuple[tuple[float, float], ...] = (
    (1640.0, 380.0),
    (1920.0, 380.0),
    (1920.0, 700.0),
    (1640.0, 700.0),
)
EXIT_ZONE_NAME = "main door"
EXIT_POINT: tuple[float, float] = (1780.0, 540.0)
SEAT: tuple[float, float] = (960.0, 540.0)
PATIENT_TRACK_ID = 1
STAFF_TRACK_ID = 99
PATIENT_BOX: tuple[float, float] = (80.0, 110.0)
STAFF_BOX: tuple[float, float] = (70.0, 70.0)

# Staff patrol: a circle whose closest approach to SEAT is 100 px. Constant speed, so its
# steps never reverse direction and its posture never leaves the upright band: the patrol
# is a walking negative control for both event rules.
STAFF_CENTER: tuple[float, float] = (960.0, 940.0)
STAFF_LAP_RADIUS_PX = 300.0
STAFF_LAP_S = 600.0
STAFF_POSTURE_MID_DEG = 8.0
STAFF_POSTURE_AMP_DEG = 6.0

# Fidget: two incommensurate sinusoids per axis, with short periods relative to every rule
# window, so the motion is stationary over any window a rule looks at. The turning points
# of a slow sinusoid are the only direction reversals it has, which keeps the reversal
# fraction far below the convulsion threshold at any reasonable sampling rate.
_FIDGET_X = ((6.0, 37.0), (3.0, 13.0))
_FIDGET_Y = ((5.0, 23.0), (2.0, 11.0))
# Posture swings 0 to 10 degrees with a 97 s period: always upright, never down.
_FIDGET_POSTURE_MID_DEG = 5.0
_FIDGET_POSTURE_AMP_DEG = 5.0
_FIDGET_POSTURE_S = 97.0

# The collapse shape: where a fall starts from, where it lands, and how far the box
# centre drops on the way down. All comfortably on the firing side of the provisional
# thresholds (upright 6 < 30; down 78 > 60; drop 120 > 40) so the tests exercise the
# cascade, not threshold-edge float behaviour.
FALL_START_DEG = 6.0
FALL_END_DEG = 78.0
FALL_DROP_PX = 120.0

# The convulsion shape: the box centre alternates around the seat every sample. At any
# hz this reverses direction every step; amplitude is per half-swing.
CONVULSE_AMP_X_PX = 8.0
CONVULSE_AMP_Y_PX = 4.0
CONVULSE_POSTURE_MID_DEG = 35.0
CONVULSE_POSTURE_AMP_DEG = 15.0

# Torso geometry used to synthesise COCO-17 keypoints at a chosen angle.
TORSO_PX = 40.0
SHOULDER_PX = 20.0
KP_CONF = 0.9


def torso_keypoints(
    cx: float,
    cy: float,
    angle_deg: float,
    *,
    torso_px: float = TORSO_PX,
    shoulder_px: float = SHOULDER_PX,
    conf: float = KP_CONF,
) -> Keypoints:
    """COCO-17 keypoints whose torso axis sits at `angle_deg` from the image vertical.

    Only the four torso keypoints (5, 6, 11, 12) carry meaning; the rest sit at the box
    centre with the same confidence, because nothing in this layer reads them. Feeding the
    result back through vividly.tracks.posture_angle_deg returns angle_deg for angles in
    [0, 90], which tests assert directly.
    """
    theta = math.radians(angle_deg)
    shoulder_x = cx
    shoulder_y = cy - torso_px / 2.0
    hip_x = shoulder_x + torso_px * math.sin(theta)
    hip_y = shoulder_y + torso_px * math.cos(theta)
    half = shoulder_px / 2.0
    points = [(cx, cy, conf) for _ in range(NUM_KEYPOINTS)]
    points[LEFT_SHOULDER] = (shoulder_x - half, shoulder_y, conf)
    points[RIGHT_SHOULDER] = (shoulder_x + half, shoulder_y, conf)
    points[LEFT_HIP] = (hip_x - half, hip_y, conf)
    points[RIGHT_HIP] = (hip_x + half, hip_y, conf)
    return tuple(points)


def _sines(elapsed: float, terms: tuple[tuple[float, float], ...]) -> float:
    return sum(amp * math.sin(math.tau * elapsed / period) for amp, period in terms)


def _fidget_sample(elapsed: float, seat: tuple[float, float]) -> tuple[float, float, float]:
    """Seated small movements: (cx, cy, torso angle in degrees)."""
    cx = seat[0] + _sines(elapsed, _FIDGET_X)
    cy = seat[1] + _sines(elapsed, _FIDGET_Y)
    angle = _FIDGET_POSTURE_MID_DEG + _FIDGET_POSTURE_AMP_DEG * math.sin(
        math.tau * elapsed / _FIDGET_POSTURE_S
    )
    return cx, cy, angle


def _staff_sample(elapsed: float) -> tuple[float, float, float]:
    """Circular patrol at constant speed: (cx, cy, torso angle in degrees).

    Constant speed and constant turning: every step has the same length and consecutive
    steps never reverse, which is what makes the patrol a negative control for
    seizure_candidate_v0 in every scene it appears in.
    """
    phase = math.tau * elapsed / STAFF_LAP_S
    cx = STAFF_CENTER[0] + STAFF_LAP_RADIUS_PX * math.cos(phase)
    cy = STAFF_CENTER[1] + STAFF_LAP_RADIUS_PX * math.sin(phase)
    angle = STAFF_POSTURE_MID_DEG + STAFF_POSTURE_AMP_DEG * math.sin(phase)
    return cx, cy, angle


def _down_sample(
    elapsed: float, seat: tuple[float, float], drop_px: float
) -> tuple[float, float, float]:
    """On the ground and still: micro-wobble around the landing point, torso down."""
    cx = seat[0] + 0.2 * math.sin(math.tau * elapsed / 7.0)
    cy = seat[1] + drop_px + 0.2 * math.sin(math.tau * elapsed / 9.0)
    angle = FALL_END_DEG + 0.5 * math.sin(math.tau * elapsed / 11.0)
    return cx, cy, angle


def _n_ticks(minutes: float, hz: float) -> int:
    if minutes <= 0:
        raise ValueError(f"minutes must be > 0, got {minutes}")
    if hz <= 0:
        raise ValueError(f"hz must be > 0, got {hz}")
    return round(minutes * 60.0 * hz) + 1


def _scene(
    *,
    n_ticks: int,
    hz: float,
    start_t: float,
    patient_id: int,
    patient_fn: Callable[[float], tuple[float, float, float | None] | None],
    staff_id: int | None,
) -> list[list[TrackObservation]]:
    """One batch per tick, observations sorted by track_id. Pure loop, no sleeping.

    `patient_fn` may return None for an instant: the person is not observed then.
    """
    dt = 1.0 / hz
    batches: list[list[TrackObservation]] = []
    for i in range(n_ticks):
        elapsed = i * dt
        t = start_t + elapsed
        batch: list[TrackObservation] = []
        sample = patient_fn(elapsed)
        if sample is not None:
            cx, cy, angle = sample
            batch.append(
                TrackObservation(
                    track_id=patient_id,
                    t=t,
                    cx=cx,
                    cy=cy,
                    w=PATIENT_BOX[0],
                    h=PATIENT_BOX[1],
                    keypoints=None if angle is None else torso_keypoints(cx, cy, angle),
                )
            )
        if staff_id is not None:
            sx, sy, s_angle = _staff_sample(elapsed)
            batch.append(
                TrackObservation(
                    track_id=staff_id,
                    t=t,
                    cx=sx,
                    cy=sy,
                    w=STAFF_BOX[0],
                    h=STAFF_BOX[1],
                    keypoints=torso_keypoints(sx, sy, s_angle),
                )
            )
        batch.sort(key=lambda obs: obs.track_id)
        batches.append(batch)
    return batches


def _fall_interp(
    progress: float, seat: tuple[float, float], drop_px: float
) -> tuple[float, float, float]:
    """Linear upright-to-down transition at `progress` in [0, 1]."""
    angle = FALL_START_DEG + (FALL_END_DEG - FALL_START_DEG) * progress
    return seat[0], seat[1] + drop_px * progress, angle


def collapse_and_stay_down(
    *,
    lead_in_minutes: float = 6.0,
    down_minutes: float = 1.5,
    fall_s: float = 1.5,
    drop_px: float = FALL_DROP_PX,
    track_id: int = PATIENT_TRACK_ID,
    staff_track_id: int | None = STAFF_TRACK_ID,
    hz: float = 1.0,
    start_t: float = 0.0,
    seat: tuple[float, float] = SEAT,
) -> list[list[TrackObservation]]:
    """Fidget for lead_in_minutes, fall in fall_s, then stay down and still.

    The trajectory fall_then_still_v0 exists to fire on. The true fall onset is
    `start_t + lead_in_minutes * 60` (the last sample of the lead-in is the last upright
    sample the rule can find), which is what the reported onset_t is checked against,
    to within one or two sample intervals, because the rule can only anchor on samples.
    The lead-in matters: it is what proves the rule needs an upright phase, and its
    fidget posture (0-10 degrees) sits safely inside fall_upright_max_deg.
    """
    if fall_s <= 0:
        raise ValueError(f"fall_s must be > 0, got {fall_s}")
    lead_in_s = lead_in_minutes * 60.0

    def patient_fn(elapsed: float) -> tuple[float, float, float | None]:
        if elapsed < lead_in_s:
            return _fidget_sample(elapsed, seat)
        if elapsed < lead_in_s + fall_s:
            return _fall_interp((elapsed - lead_in_s) / fall_s, seat, drop_px)
        return _down_sample(elapsed, seat, drop_px)

    return _scene(
        n_ticks=_n_ticks(lead_in_minutes + fall_s / 60.0 + down_minutes, hz),
        hz=hz,
        start_t=start_t,
        patient_id=track_id,
        patient_fn=patient_fn,
        staff_id=staff_track_id,
    )


def collapse_then_recover(
    *,
    lead_in_minutes: float = 6.0,
    down_s: float = 8.0,
    fall_s: float = 1.5,
    recover_s: float = 1.5,
    tail_minutes: float = 2.0,
    drop_px: float = FALL_DROP_PX,
    track_id: int = PATIENT_TRACK_ID,
    staff_track_id: int | None = STAFF_TRACK_ID,
    hz: float = 1.0,
    start_t: float = 0.0,
    seat: tuple[float, float] = SEAT,
) -> list[list[TrackObservation]]:
    """The same fall, but the person gets up after down_s and resumes fidgeting.

    The false-positive control for the cascade: a trip, a dropped phone picked up
    off the floor, a knee-down moment. Stage 1 of fall_then_still_v0 happens in full and
    stage 2 never completes (down_s defaults to well under down_still_min_s), so the rule
    must stay silent on every batch of this scene. If it ever fires here, the
    confirmation stage has stopped confirming.
    """
    if fall_s <= 0 or recover_s <= 0 or down_s <= 0:
        raise ValueError(
            f"fall_s, recover_s and down_s must be > 0, got {fall_s}, {recover_s}, {down_s}"
        )
    lead_in_s = lead_in_minutes * 60.0
    up_at = lead_in_s + fall_s + down_s

    def patient_fn(elapsed: float) -> tuple[float, float, float | None]:
        if elapsed < lead_in_s:
            return _fidget_sample(elapsed, seat)
        if elapsed < lead_in_s + fall_s:
            return _fall_interp((elapsed - lead_in_s) / fall_s, seat, drop_px)
        if elapsed < up_at:
            return _down_sample(elapsed, seat, drop_px)
        if elapsed < up_at + recover_s:
            return _fall_interp(1.0 - (elapsed - up_at) / recover_s, seat, drop_px)
        return _fidget_sample(elapsed, seat)

    total_minutes = lead_in_minutes + (fall_s + down_s + recover_s) / 60.0 + tail_minutes
    return _scene(
        n_ticks=_n_ticks(total_minutes, hz),
        hz=hz,
        start_t=start_t,
        patient_id=track_id,
        patient_fn=patient_fn,
        staff_id=staff_track_id,
    )


def convulsive_episode(
    *,
    lead_in_minutes: float = 6.0,
    episode_s: float = 20.0,
    tail_minutes: float = 1.0,
    amp_x_px: float = CONVULSE_AMP_X_PX,
    amp_y_px: float = CONVULSE_AMP_Y_PX,
    track_id: int = PATIENT_TRACK_ID,
    staff_track_id: int | None = STAFF_TRACK_ID,
    hz: float = 5.0,
    start_t: float = 0.0,
    seat: tuple[float, float] = SEAT,
) -> list[list[TrackObservation]]:
    """Fidget, then oscillate around the seat for episode_s, then fidget again.

    The trajectory seizure_candidate_v0 exists to fire on. During the episode the centre
    alternates sign around the seat on every sample, so every consecutive step pair
    reverses direction (reversal fraction 1.0 inside the episode) at an amplitude far
    above the fidget's. The true episode onset is `start_t + lead_in_minutes * 60`;
    the rule's onset_t is only ever the start of its evidence window, so tests assert it
    lands within convulse_window_s of the truth, not on it.

    Default hz is 5.0, not 1.0: an alternation needs a sampling rate meaningfully above
    the rule's minimum sample count over its window, and 1 Hz over a 12 s window cannot
    carry 20 samples.
    """
    if episode_s <= 0:
        raise ValueError(f"episode_s must be > 0, got {episode_s}")
    lead_in_s = lead_in_minutes * 60.0
    dt = 1.0 / hz

    def patient_fn(elapsed: float) -> tuple[float, float, float | None]:
        if elapsed < lead_in_s or elapsed >= lead_in_s + episode_s:
            return _fidget_sample(elapsed, seat)
        sign = 1.0 if round(elapsed / dt) % 2 == 0 else -1.0
        cx = seat[0] + amp_x_px * sign
        cy = seat[1] - amp_y_px * sign
        angle = CONVULSE_POSTURE_MID_DEG + CONVULSE_POSTURE_AMP_DEG * sign
        return cx, cy, angle

    total_minutes = lead_in_minutes + episode_s / 60.0 + tail_minutes
    return _scene(
        n_ticks=_n_ticks(total_minutes, hz),
        hz=hz,
        start_t=start_t,
        patient_id=track_id,
        patient_fn=patient_fn,
        staff_id=staff_track_id,
    )


def normal_fidget(
    minutes: float = 60.0,
    *,
    track_id: int = PATIENT_TRACK_ID,
    staff_track_id: int | None = STAFF_TRACK_ID,
    hz: float = 1.0,
    start_t: float = 0.0,
    seat: tuple[float, float] = SEAT,
) -> list[list[TrackObservation]]:
    """An ordinary wait: someone shifts in their seat, a second person walks laps.

    This is the negative control for the whole rule layer, and it is only a control if
    nothing fires on it: no fall (posture never leaves the upright band, the centre never
    drops), and no convulsion (the fidget's slow sinusoids reverse direction only at
    their turning points and move around a pixel per second). A firing on this scene is
    a false positive by construction, which is the defect class that matters most.
    """
    return _scene(
        n_ticks=_n_ticks(minutes, hz),
        hz=hz,
        start_t=start_t,
        patient_id=track_id,
        patient_fn=lambda elapsed: _fidget_sample(elapsed, seat),
        staff_id=staff_track_id,
    )


def leave_and_return(
    minutes: float = 30.0,
    *,
    track_id: int = PATIENT_TRACK_ID,
    staff_track_id: int | None = STAFF_TRACK_ID,
    hz: float = 1.0,
    start_t: float = 0.0,
    seat: tuple[float, float] = SEAT,
    leave_at_minutes: float = 8.0,
    gap_s: float = 20.0,
) -> list[list[TrackObservation]]:
    """normal_fidget with the patient missing for gap_s, starting at leave_at_minutes.

    The gap defaults to 20 s, comfortably inside TrackConfig.reap_after_s (30 s), so the
    track survives and keeps its id, its baseline, its staff tag and its mock chart
    binding. It also defaults to starting after TrackConfig.baseline_window_s (300 s) has
    elapsed, so the baseline is already frozen when the person walks off and can be
    compared before and after.

    There is no re-identification anywhere in this layer. This fixture tests that a gap
    inside the reap window costs nothing; a gap longer than the reap window produces a new
    track id, and that is the honest answer rather than a bug to paper over.
    """
    if gap_s <= 0:
        raise ValueError(f"gap_s must be > 0, got {gap_s}")
    batches = normal_fidget(
        minutes,
        track_id=track_id,
        staff_track_id=staff_track_id,
        hz=hz,
        start_t=start_t,
        seat=seat,
    )
    gap_start = start_t + leave_at_minutes * 60.0
    gap_end = gap_start + gap_s
    return [
        [obs for obs in batch if not (obs.track_id == track_id and gap_start <= obs.t < gap_end)]
        for batch in batches
    ]


def depart_through_exit(
    *,
    seated_minutes: float = 3.0,
    walk_s: float = 10.0,
    absent_minutes: float = 12.0,
    track_id: int = PATIENT_TRACK_ID,
    staff_track_id: int | None = STAFF_TRACK_ID,
    hz: float = 1.0,
    start_t: float = 0.0,
    seat: tuple[float, float] = SEAT,
    exit_point: tuple[float, float] = EXIT_POINT,
    return_after_s: float | None = None,
    return_track_id: int | None = None,
) -> list[list[TrackObservation]]:
    """Sit, walk to the exit, vanish; optionally come back later under a NEW tracker id.

    The trajectory egress_without_discharge_v0 exists to fire on. The person fidgets in
    the seat for seated_minutes, walks in a straight line (upright, directionally
    coherent: neither event rule has anything to say about it) to `exit_point` over
    walk_s, and is then not observed again. The scene keeps running for absent_minutes on
    the patrol track alone so the pipeline clock advances past the reap window and the
    egress clock. The last sighting - the egress onset - is `start_t + seated_minutes * 60
    + walk_s` to within one sample interval, at `exit_point`.

    `walk_s = 0` means the person vanishes from the seat without walking anywhere: the
    mid-room loss the head must NOT read as an exit.

    With `return_after_s` (measured from the last sighting) and `return_track_id`, a track
    under the new id appears back in the seat and fidgets for the rest of the scene. That
    is a return as far as the tracker can tell, that is a new person, and only a human
    confirming the re-association (TrackBook.merge_identity) turns it into the same person
    coming back.

    POLICY.md, rule 4 is not in play: someone walking out of a room is not a medical event, and
    these are abstract points on a plane in any case.
    """
    if seated_minutes < 0 or walk_s < 0 or absent_minutes <= 0:
        raise ValueError("seated_minutes and walk_s must be >= 0 and absent_minutes > 0")
    if (return_after_s is None) != (return_track_id is None):
        raise ValueError("return_after_s and return_track_id must be given together")
    if return_after_s is not None and (return_after_s <= 0 or return_track_id == track_id):
        raise ValueError("return_after_s must be > 0 and return_track_id must be a new id")
    seated_s = seated_minutes * 60.0
    leave_s = seated_s + walk_s
    total_minutes = seated_minutes + walk_s / 60.0 + absent_minutes

    def leaver_fn(elapsed: float) -> tuple[float, float, float | None] | None:
        if elapsed < seated_s:
            return _fidget_sample(elapsed, seat)
        if walk_s > 0 and elapsed <= leave_s:
            progress = (elapsed - seated_s) / walk_s
            return (
                seat[0] + (exit_point[0] - seat[0]) * progress,
                seat[1] + (exit_point[1] - seat[1]) * progress,
                FALL_START_DEG,
            )
        return None

    batches = _scene(
        n_ticks=_n_ticks(total_minutes, hz),
        hz=hz,
        start_t=start_t,
        patient_id=track_id,
        patient_fn=leaver_fn,
        staff_id=staff_track_id,
    )
    if return_after_s is None or return_track_id is None:
        return batches

    back_at = leave_s + return_after_s

    def returner_fn(elapsed: float) -> tuple[float, float, float | None] | None:
        if elapsed < back_at:
            return None
        return _fidget_sample(elapsed - back_at, seat)

    returning = _scene(
        n_ticks=_n_ticks(total_minutes, hz),
        hz=hz,
        start_t=start_t,
        patient_id=return_track_id,
        patient_fn=returner_fn,
        staff_id=None,
    )
    merged: list[list[TrackObservation]] = []
    for batch, extra in zip(batches, returning, strict=True):
        joined = list(batch) + list(extra)
        joined.sort(key=lambda obs: obs.track_id)
        merged.append(joined)
    return merged
