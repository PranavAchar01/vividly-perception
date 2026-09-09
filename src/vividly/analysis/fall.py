"""Streaming fall detector: a fast collapse CONFIRMED by prolonged down-and-still.

The live-pipeline counterpart of vividly.rules.collapse.FallThenStillRule — the same
research-endorsed cascade (a fall transition, then sustained stillness on the ground), but
as a per-track streaming analyzer that consumes pose keypoints frame by frame, matching
SeizureAnalyzer so both heads share one pose backbone and one per-track history.

Two stages, BOTH required (a fall alone never alerts — someone who trips and gets up is
not an emergency):
  1. THE COLLAPSE. The torso goes from upright (angle-from-vertical <= upright_max_deg) to
     down (>= down_min_deg) within max_fall_s, and the body drops by at least drop_min of
     its standing height. The speed + drop gates separate a real fall from slowly reclining
     into a chair.
  2. THE CONFIRMATION. The person then stays down AND still (little centroid motion,
     normalized by body size) for hold_s. Movement on the ground is not down-and-still; if
     that motion is rhythmic, the seizure head owns it.

Everything is normalized by body scale (torso size / standing height), so thresholds are
resolution- and distance-independent — unlike the raw-pixel rule tuned for one camera.

Honest limitation (validated on the UR Fall dataset: sideways falls fire, a fall directly
TOWARD/AWAY from the camera is missed): the discriminator is the torso's angle in the image
plane, which barely changes when someone collapses straight toward or away from a single
camera. Side-on falls are caught cleanly; the toward/away blind spot needs a second camera
or a bounding-box-aspect complement, which is deferred to keep false positives near zero.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from vividly.analysis.seizure import L_HIP, L_SH, R_HIP, R_SH


@dataclass
class FallConfig:
    window_s: float = 20.0  # history kept (must cover max_fall_s + hold_s + slack)
    kp_score_thr: float = 0.3  # a torso keypoint counts as visible above this
    upright_max_deg: float = 35.0  # torso within this of vertical = upright
    down_min_deg: float = 55.0  # the collapse must PEAK at/beyond this (truly horizontal)
    max_fall_s: float = (
        2.0  # upright -> down must happen within this (fast = a fall, not a recline)
    )
    drop_min: float = 0.14  # body-center must drop >= this fraction of standing height
    hold_s: float = 4.0  # down-and-still must persist this long -> CONFIRMED fall
    move_max: float = (
        0.18  # max RMS body-center spread (frac of standing height) to count as "still"
    )
    # ^ accommodates the small settling/box-jitter of a real collapse (~0.12 measured on UR
    #   Fall) while rejecting active motion — thrashing or getting up spreads ~0.3+.
    stale_s: float = 1.5  # a gap larger than this breaks the contiguous down span
    min_samples: int = 8  # need at least this many samples in the down span
    gone_s: float = 10.0  # a latched fall is HELD through an absence this long (occlusion),
    #                       and only cleared once the subject has been gone longer (they left)


@dataclass
class FallState:
    state: str = "idle"  # "idle" | "falling" | "fall"
    torso_deg: float = 0.0  # current torso angle from vertical
    down_s: float = 0.0  # how long the person has been continuously down
    drop: float = 0.0  # measured collapse drop (fraction of standing height)
    held_s: float = 0.0  # how long the confirmed condition has held
    meta: dict = field(default_factory=dict)


def _mean_visible(kp: np.ndarray, sc: np.ndarray, idxs, thr: float):
    """Mean of the visible members of a keypoint pair; None if neither is visible."""
    vis = [i for i in idxs if sc[i] > thr]
    if not vis:
        return None
    return kp[vis].mean(axis=0)


def _torso(kp: np.ndarray, sc: np.ndarray, thr: float):
    """(angle-from-vertical in degrees, torso length px) or None if torso not visible."""
    sh = _mean_visible(kp, sc, (L_SH, R_SH), thr)
    hp = _mean_visible(kp, sc, (L_HIP, R_HIP), thr)
    if sh is None or hp is None:
        return None
    dx, dy = float(hp[0] - sh[0]), float(hp[1] - sh[1])
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return None
    # angle between the torso axis and the vertical (image-down) direction: 0 = upright,
    # 90 = horizontal / lying.
    ang = math.degrees(math.atan2(abs(dx), abs(dy)))
    return ang, length


class FallAnalyzer:
    """Streaming, per-track. Feed it (keypoints[17,2], scores[17], bbox[4], t) per frame."""

    def __init__(self, cfg: FallConfig | None = None) -> None:
        self.cfg = cfg or FallConfig()
        # each entry: (t, torso_deg, cx, cy, standing_h, torso_len)
        self._buf: deque[tuple[float, float, float, float, float, float]] = deque()
        self._absent_since: float | None = None
        # A collapse is a LATCHED event: detected once at the fall (when the pose is still
        # reliable), then persisted via the stable body-center height until the person clearly
        # gets back up — because the pose torso angle is too noisy on a floor-crumpled body to
        # re-derive "still down" frame by frame. (collapse_t, upright_cy, collapse_cy, stand_h).
        self._collapse: tuple[float, float, float, float] | None = None
        self._resolved_until: float = -1e18  # ignore collapses at/before this (already handled)

    def reset(self) -> None:
        self._buf.clear()
        self._absent_since = None
        self._collapse = None
        self._resolved_until = -1e18

    def _trim(self, now: float) -> None:
        cut = now - self.cfg.window_s
        while self._buf and self._buf[0][0] < cut:
            self._buf.popleft()

    def on_absent(self, now: float) -> FallState:
        """No trackable subject this frame. Critically, a LATCHED fall is HELD through an
        occlusion — a person crumpled on the floor is exactly who gets blocked by a caregiver
        or an object, and a person already down can never re-latch (there is no fresh
        upright->down transition to detect), so decaying the alert on a brief absence would
        silently drop a real fall. The alert persists until the subject has been gone longer
        than gone_s (they were helped up / moved away), at which point the state is cleared."""
        if self._absent_since is None:
            self._absent_since = now
        gone = now - self._absent_since
        if gone >= self.cfg.gone_s:
            self._buf.clear()
            self._collapse = None
            return FallState(state="idle")
        if self._collapse is not None:
            # Hold the latched fall while the subject is briefly out of view.
            collapse_t, upright_cy, collapse_cy, stand_h = self._collapse
            down_s = now - collapse_t
            st = FallState(down_s=down_s, drop=float((collapse_cy - upright_cy) / stand_h))
            if down_s >= self.cfg.hold_s:
                st.state = "fall"
                st.held_s = down_s
            else:
                st.state = "falling"
            return st
        return FallState(state="idle")

    def update(self, keypoints: np.ndarray, scores: np.ndarray, bbox: np.ndarray, now: float):
        self._absent_since = None
        cfg = self.cfg
        # Always record the body center from the box — even when the torso keypoints are too
        # low-confidence to give an angle (routine on a floor-crumpled person). The torso angle
        # is only needed to DETECT a collapse; a latched collapse is carried by the center
        # alone, so starving the buffer whenever the pose degrades would drop real falls.
        t = _torso(
            np.asarray(keypoints, np.float32), np.asarray(scores, np.float32), cfg.kp_score_thr
        )
        ang = t[0] if t is not None else float("nan")
        torso_len = t[1] if t is not None else float("nan")
        x1, y1, x2, y2 = (float(v) for v in bbox)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        standing_h = max(abs(y2 - y1), 1.0)
        self._buf.append((float(now), ang, cx, cy, standing_h, torso_len))
        self._trim(now)
        return self._evaluate(now)

    def _evaluate(self, now: float) -> FallState:
        cfg = self.cfg
        st = FallState()
        buf = list(self._buf)
        if len(buf) < 2:
            return st
        newest = buf[-1]
        st.torso_deg = newest[1]
        if now - newest[0] > cfg.stale_s:
            return st  # evidence too old to judge

        # Smooth the torso angle (rolling median). Pose on a floor-crumpled body is very noisy
        # — the angle bounces even between identical frames as the box drifts — so a collapse
        # is DETECTED once (below) and then LATCHED via the stable body-center height, never
        # re-derived from the per-frame torso.
        raw = [b[1] for b in buf]  # torso angle, may be NaN when the pose is too weak
        deg = [
            float(np.nanmedian(w))
            if np.any(~np.isnan(w := np.array(raw[max(0, i - 2) : i + 3])))
            else float("nan")
            for i in range(len(raw))
        ]

        # --- clear a latched collapse if the person clearly STOOD BACK UP -----------------
        # Judge recovery by the body-center HEIGHT, not the torso angle. cy grows downward, so
        # a smaller recent cy means they rose. "Recovered" = risen past halfway from the
        # collapsed height back toward the upright height.
        if self._collapse is not None:
            _ct, upright_cy, collapse_cy, _sh = self._collapse
            recovered_cy = upright_cy + 0.5 * (collapse_cy - upright_cy)
            recent_cy = [b[3] for b in buf if b[0] >= now - 0.6]
            if recent_cy and float(np.median(recent_cy)) < recovered_cy:
                self._resolved_until = now  # this fall is over; don't re-latch it
                self._collapse = None

        # --- detect a NEW collapse: a FAST upright->horizontal transition with a drop -------
        # Scan for the earliest qualifying transition newer than the last resolved one, so a
        # single fall latches once (not at every later torso noise-spike).
        if self._collapse is None:
            for c in range(1, len(buf)):
                if buf[c][0] <= self._resolved_until or not (deg[c] >= cfg.down_min_deg):
                    continue  # NaN torso (weak pose) fails >= and is skipped as a peak
                # Look back for the last CONFIDENT upright, counting only confident-frame time
                # toward max_fall_s: if the fall itself blurs the pose (a NaN blackout mid-
                # transition), that blackout must not disqualify a genuinely fast fall, nor let
                # a slow recline (continuous confident tilt, no blackout) sneak through.
                u = None
                conf_elapsed = 0.0
                for j in range(c - 1, -1, -1):
                    if not np.isnan(deg[j + 1]) and not np.isnan(deg[j]):
                        conf_elapsed += buf[j + 1][0] - buf[j][0]
                    if conf_elapsed > cfg.max_fall_s:
                        break
                    if deg[j] <= cfg.upright_max_deg:
                        u = j
                        break
                if u is None:
                    continue
                drop = (buf[c][3] - buf[u][3]) / buf[u][4]  # cy grows downward
                if drop < cfg.drop_min:
                    continue
                self._collapse = (buf[c][0], buf[u][3], buf[c][3], buf[u][4])
                break

        if self._collapse is None:
            return st  # no fast, dropping, upright->horizontal transition -> not a fall
        collapse_t, upright_cy, collapse_cy, stand_h = self._collapse
        st.drop = float((collapse_cy - upright_cy) / stand_h)
        down_s = now - collapse_t
        st.down_s = down_s

        # --- post-collapse STILLNESS: RMS spread of the body center over the recent window,
        # normalized by standing height. Spread over a fixed TIME is frame-rate independent
        # (per-frame step is not), and it ignores the collapse's own motion.
        recent = [b for b in buf if b[0] >= max(collapse_t, now - cfg.hold_s)]
        if len(recent) >= 2:
            mx = sum(b[2] for b in recent) / len(recent)
            my = sum(b[3] for b in recent) / len(recent)
            spread = (
                math.sqrt(sum((b[2] - mx) ** 2 + (b[3] - my) ** 2 for b in recent) / len(recent))
                / stand_h
            )
        else:
            spread = 1.0
        n_down = sum(1 for b in buf if b[0] >= collapse_t)
        st.meta = {"down_s": float(down_s), "spread": float(spread), "n_down": int(n_down)}

        confirmed = down_s >= cfg.hold_s and n_down >= cfg.min_samples and spread <= cfg.move_max
        if confirmed:
            st.state = "fall"
            st.held_s = down_s
        else:
            st.state = "falling"
        return st
