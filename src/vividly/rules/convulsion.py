"""seizure_candidate_v0: sustained, rhythmic, direction-reversing motion.

The acute-event seizure candidate. Tonic-clonic convulsion on an overhead camera is
rhythmic oscillation of the whole body: the box centre reverses direction nearly every
step, at an amplitude no seated fidget reaches, sustained for seconds. Three conditions,
all required by the false-positive stance:

  * Reversal. The fraction of consecutive step pairs whose directions reverse (negative
    dot product) must reach convulse_reversal_frac. Walking is directionally coherent
    (fraction near zero); seated fidgeting is slow sinusoidal drift that reverses only at
    its turning points (fraction well under the threshold at any reasonable sampling rate).
    This is the convulsion signature; amplitude alone is a person shifting in a chair.
  * Amplitude. Mean per-step centre displacement must reach convulse_amp_min_px, so pose
    jitter and sub-pixel tracker noise (which reverse direction constantly, at no
    amplitude) can never fire the rule on a motionless person.
  * Duration. The evidence must span convulse_min_s with convulse_min_samples samples and
    a newest sample no older than convulse_stale_s. A two-second shudder, a dropped bag,
    a chair scrape: none of them sustain.

Named a candidate deliberately. Geometry cannot distinguish a seizure from vigorous
rhythmic motion of other kinds (a child bouncing, aggressive fidgeting); the trained
verifier is what confirms the event before a nurse is interrupted. The seam is
vividly.rules.verify and the ONNX backend behind it is vividly.models.event_verifier; the
seizure head of record is convulsion_v2_real, onnx sha8 07c399be, recorded on
final_models/seizure/README.md. Posture is deliberately not consulted: seizures happen seated, on
the ground after a fall (fall_then_still_v0 abstains on that track because the down span
is not still, so this rule owns it), and mid-transfer.

Frequency note: at the 15 fps working point the sampler resolves oscillation up to
7.5 Hz, which covers the 2-6 Hz band clonic movement occupies. At the fixtures' lower
rates the alternation is coarser but the reversal fraction is the same statistic, which
is why the thresholds are expressed as fractions and pixels, never as Hz.
"""

from __future__ import annotations

import math
from itertools import pairwise

from vividly.rules.base import RuleConfig, RuleFiring, window_positions
from vividly.tracks import TrackState

# Steps shorter than this many pixels carry no usable direction; pairs touching one are
# excluded from the reversal statistic rather than counted either way.
MIN_STEP_PX = 1e-9


class ConvulsiveMotionRule:
    """Sustained rhythmic oscillation of the box centre. Pure: only state is cfg."""

    rule_id = "seizure_candidate_v0"
    rule_version = "0.1.0"

    def __init__(self, cfg: RuleConfig | None = None) -> None:
        self.cfg = RuleConfig() if cfg is None else cfg

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ConvulsiveMotionRule(convulse_reversal_frac={self.cfg.convulse_reversal_frac}, "
            f"convulse_amp_min_px={self.cfg.convulse_amp_min_px})"
        )

    def evaluate(self, track: TrackState, now: float) -> RuleFiring | None:
        """Fire when reversal, amplitude and duration all hold; abstain otherwise."""
        cfg = self.cfg
        points = window_positions(track, now, cfg.convulse_window_s)
        if len(points) < cfg.convulse_min_samples:
            return None
        if now - points[-1][0] > cfg.convulse_stale_s:
            return None
        span_s = points[-1][0] - points[0][0]
        if span_s < cfg.convulse_min_s:
            return None

        steps = [(b[1] - a[1], b[2] - a[2]) for a, b in pairwise(points)]
        magnitudes = [math.hypot(dx, dy) for dx, dy in steps]
        mean_step_px = sum(magnitudes) / len(magnitudes)
        if mean_step_px < cfg.convulse_amp_min_px:
            return None

        reversals = 0
        usable = 0
        for (a, mag_a), (b, mag_b) in pairwise(zip(steps, magnitudes, strict=True)):
            if mag_a < MIN_STEP_PX or mag_b < MIN_STEP_PX:
                continue
            usable += 1
            if a[0] * b[0] + a[1] * b[1] < 0.0:
                reversals += 1
        if usable == 0:
            return None
        reversal_frac = reversals / usable
        if reversal_frac < cfg.convulse_reversal_frac:
            return None

        return RuleFiring(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            track_id=track.track_id,
            onset_t=points[0][0],
            detected_t=now,
            baseline=track.baseline(),
            details={
                "reversal_frac": float(reversal_frac),
                "mean_step_px": float(mean_step_px),
                "span_s": float(span_s),
                "n_samples": float(len(points)),
                # Each reversal is roughly half an oscillation cycle, so this is a rough
                # rate for the reviewer, not a measurement (sampling-rate limited).
                "approx_oscillation_hz": float(reversals / (2.0 * span_s)) if span_s > 0 else 0.0,
                "threshold_convulse_reversal_frac": float(cfg.convulse_reversal_frac),
                "threshold_convulse_amp_min_px": float(cfg.convulse_amp_min_px),
                "threshold_convulse_min_s": float(cfg.convulse_min_s),
                "threshold_status": "PROVISIONAL (acute-event rules; see docs/CALIBRATION.md)",
            },
        )
