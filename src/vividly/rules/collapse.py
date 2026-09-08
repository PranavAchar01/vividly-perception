"""fall_then_still_v0: a fast fall, confirmed by a prolonged down-and-still period.

The acute-event cascade, and the visible signature of the events the product exists for:
sudden collapse (stroke, cardiac arrest, syncope) presents on an overhead camera as a fast
postural transition to the ground followed by the person staying down. Which of those it
is cannot be decided from geometry, and this rule does not try: it detects the event shape
and the trained verifier decides whether the event is real. The seam is
vividly.rules.verify and the ONNX backend behind it is vividly.models.event_verifier; the
fall head of record is fall_v3, onnx sha8 67954c04, recorded on
final_models/fall/README.md. Two stages, both required before anything fires:

  1. The fall. The torso goes from upright (<= fall_upright_max_deg) to down
     (>= fall_down_min_deg) within fall_max_s, and the box centre moves down the image by
     at least fall_drop_min_px across the transition. The speed gate is what separates a
     fall from the slow slide the removed drift rule used to chase: a slump over minutes
     is not an acute event and must never alert here.
  2. The confirmation. The person then stays down (torso at or above fall_down_min_deg)
     and still (mean per-step centre displacement <= down_move_max_px) for at least
     down_still_min_s, with fresh evidence (down_stale_s) and enough samples
     (down_min_samples). A fall alone never alerts: someone who trips and gets up, bends
     to pick something up, or kneels briefly produces stage 1 and not stage 2, and the
     rule stays silent. That asymmetry is the false-positive stance in code.

Abstentions, each one deliberate (precision over recall):
  * No upright sample inside collapse_search_s: the person entered the frame already on
    the ground, or went down slower than the window remembers. Different scenario, no
    verdict.
  * Posture blackout longer than down_stale_s inside the down span: occlusion (a crowd
    gathering, a responder kneeling over the person) breaks the evidence chain, and the
    rule reports nothing rather than guessing through it.
  * Movement above down_move_max_px during the down span: a person moving on the ground
    is not down-and-still; if the motion is rhythmic, seizure_candidate_v0 owns it.

An event, not a state: once the fall transition ages out of collapse_search_s, a person still
on the ground produces no further alert from this rule. One event, one alert (plus the
engine's refire window while the evidence still spans the fall). Re-alerting on an
unattended person lying on the floor is a workflow escalation problem, not a detection
problem, and it is out of scope for v0.

Pose dependence is the honest limitation: a track whose torso keypoints never clear
TrackConfig.kp_conf_min has no posture history and this rule stays silent on it forever.
There is no box-aspect-ratio fallback, because a box aspect ratio under a ceiling-mounted
top-view camera is not posture.
"""

from __future__ import annotations

import math
from itertools import pairwise

from vividly.rules.base import RuleConfig, RuleFiring, window_positions, window_postures
from vividly.tracks import TrackState


class FallThenStillRule:
    """Fast fall confirmed by down_still_min_s on the ground. Pure: only state is cfg."""

    rule_id = "fall_then_still_v0"
    rule_version = "0.1.0"

    def __init__(self, cfg: RuleConfig | None = None) -> None:
        self.cfg = RuleConfig() if cfg is None else cfg

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FallThenStillRule(fall_max_s={self.cfg.fall_max_s}, "
            f"down_still_min_s={self.cfg.down_still_min_s})"
        )

    def evaluate(self, track: TrackState, now: float) -> RuleFiring | None:
        """Fire when the full cascade holds; abstain on any missing or stale stage."""
        cfg = self.cfg
        postures = window_postures(track, now, cfg.collapse_search_s)
        if len(postures) < cfg.down_min_samples + 1:
            return None
        newest_t, newest_deg = postures[-1]
        if now - newest_t > cfg.down_stale_s:
            return None
        if newest_deg < cfg.fall_down_min_deg:
            return None

        # Stage 2 first, walking backwards: the contiguous down span ending now. A
        # blackout longer than down_stale_s is not evidence of staying down across the
        # gap, so the span stops at the near edge of the gap.
        down_start = len(postures) - 1
        for i in range(len(postures) - 2, -1, -1):
            sample_t, angle = postures[i]
            if postures[i + 1][0] - sample_t > cfg.down_stale_s:
                break
            if angle < cfg.fall_down_min_deg:
                break
            down_start = i
        down_t0 = postures[down_start][0]
        down_s = newest_t - down_t0
        n_down = len(postures) - down_start
        if down_s < cfg.down_still_min_s:
            return None
        if n_down < cfg.down_min_samples:
            return None

        # Still during the down span. Positions keep flowing when pose confidence drops,
        # so this list can be denser than the posture evidence above.
        points = [p for p in window_positions(track, now, cfg.collapse_search_s) if p[0] >= down_t0]
        if len(points) < 2:
            return None
        steps = [math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in pairwise(points)]
        mean_step_px = sum(steps) / len(steps)
        if mean_step_px > cfg.down_move_max_px:
            return None

        # Stage 1: the last upright sample before the down span. Transitional angles
        # (between upright and down) are skipped: they are the fall itself.
        upright_t: float | None = None
        upright_deg = 0.0
        for i in range(down_start - 1, -1, -1):
            sample_t, angle = postures[i]
            if angle <= cfg.fall_upright_max_deg:
                upright_t = sample_t
                upright_deg = angle
                break
        if upright_t is None:
            return None
        fall_s = down_t0 - upright_t
        if fall_s > cfg.fall_max_s:
            return None

        # The drop: nearest position at/before the upright sample against the first
        # at/after the down span opened. cy grows downward, so positive is the drop.
        all_points = window_positions(track, now, cfg.collapse_search_s)
        cy_pre: float | None = None
        cy_post: float | None = None
        for t, _cx, cy in all_points:
            if t <= upright_t:
                cy_pre = cy
            if cy_post is None and t >= down_t0:
                cy_post = cy
        if cy_pre is None or cy_post is None:
            return None
        drop_px = cy_post - cy_pre
        if drop_px < cfg.fall_drop_min_px:
            return None

        return RuleFiring(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            track_id=track.track_id,
            onset_t=upright_t,
            detected_t=now,
            baseline=track.baseline(),
            details={
                "fall_s": float(fall_s),
                "fall_drop_px": float(drop_px),
                "upright_deg": float(upright_deg),
                "down_deg_newest": float(newest_deg),
                "down_s": float(down_s),
                "down_mean_step_px": float(mean_step_px),
                "n_down_samples": float(n_down),
                "threshold_fall_max_s": float(cfg.fall_max_s),
                "threshold_fall_drop_min_px": float(cfg.fall_drop_min_px),
                "threshold_down_still_min_s": float(cfg.down_still_min_s),
                "threshold_down_move_max_px": float(cfg.down_move_max_px),
                "threshold_status": "PROVISIONAL (acute-event rules; see docs/CALIBRATION.md)",
            },
        )
