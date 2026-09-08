"""The event-verification seam.

The product statement for this layer: the model is trained to verify that a detected event
is real. The rules in this package are geometric candidates; the verifier is the trained
model that looks at the evidence for one firing and answers confirmed or rejected. This
module is the seam that model plugs into, holding the protocol, the verdict record, and the
abstaining implementation a deployment attaches when no trained head is wired. The engine
(vividly.rules.engine) dispatches through it before a firing can become an alert, which
is where the false-positive stance gets its teeth: a rejected firing never reaches the
store, and under the "require" policy an unverified firing does not either.

What lives where. Two trained heads are recorded, both prototype class (POLICY.md, rule 6):
fall_v3 (checkpoint 3d2d17ee, onnx 67954c04), admitted under the rule 4 interim carve-out (b)
for public licence-reviewed acted-fall data, and convulsion_v2_real (checkpoint 715d172c,
onnx 07c399be), trained on the registered clinical split wu_sahzu_emu_trainsplit. The pages
are final_models/fall/README.md and final_models/seizure/README.md. Pilot-derived training
and live alerting remain gated on the counsel question CQ-21 in docs/COUNSEL_QUESTIONS.md,
because POLICY.md, rule 4 bars simulated or acted medical events from any training set,
admitting them only as registered engineering-only test sets under carve-out (a) and as
interim prototype training data under carve-out (b), which makes the data source a question
for counsel. No trained verifier lives inside this package, since the import rule keeps it
free of model runtimes: the ONNX backend is vividly.models.event_verifier, which implements
the protocol declared here and answers "unavailable" itself for a firing whose rule id it
has no head for, egress firings included. AbstainingVerifier is the implementation this
module does carry, the one a deployment attaches when no head is wired at all. It answers
"unavailable" to everything: under the default "annotate" policy alerts flow with that
answer stamped on them, and under "require" the pipeline is fail-closed and alerts nothing.
Both behaviours are deliberate and tested.

Import rule: standard library plus pyyaml plus vividly.tracks only, same as the rest of
the package (see base.py).

Raw-signal rule (the 2026-08-31 review, research/v2-training-stack.md section 7c;
enforced by tests/test_alert_path_raw_signal.py). Everything a verifier sees, that is
keypoints, boxes and centers, arrives exactly as the models emitted it. No temporal
smoothing exists anywhere in the alert path, and none may be added: a 5-frame moving
average at 15 fps places an exact spectral null on 3.0 Hz, the center of the clinically
validated 2-6 Hz convulsive band, so the smoother would erase the seizure it exists to
confirm. The one legal filter inside a verifier's feature extractor is an explicit
band-pass (~1-7 Hz) that keeps the band; anything that smooths for display lives on the
dashboard's display path and never feeds back.

Verdicts. Three, closed set:
  * confirmed: the verifier judged the event real. Confidence is its score.
  * rejected: the verifier judged the event not real. The engine drops the firing
    (and counts it: RuleEngine.n_rejected).
  * unavailable: the verifier cannot judge (no model, missing evidence). Policy decides:
    "annotate" passes it through stamped, "require" drops it (counted:
    RuleEngine.n_unconfirmed).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from vividly.rules.base import RuleFiring
from vividly.tracks import TrackState

VERIFIER_VERDICTS: tuple[str, ...] = ("confirmed", "rejected", "unavailable")

VERIFY_POLICIES: tuple[str, ...] = ("annotate", "require")


@dataclass(frozen=True)
class VerifierVerdict:
    """One verifier's answer about one firing. Frozen: an answer is not editable."""

    verdict: str
    verifier_id: str
    verifier_version: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.verdict not in VERIFIER_VERDICTS:
            raise ValueError(f"verdict must be one of {VERIFIER_VERDICTS}, got {self.verdict!r}")
        if not self.verifier_id or not self.verifier_version:
            raise ValueError("verifier_id and verifier_version must be non-empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1] or None, got {self.confidence}")


@runtime_checkable
class EventVerifier(Protocol):
    """A model (or stand-in) that judges whether one candidate firing is a real event.

    Called by the engine with the firing AND the live TrackState it fired on, so a real
    backend can reach the evidence (recent positions, postures) without the firing having
    to carry it. Implementations must be safe to call twice with the same inputs.

    `track` is None for an egress firing (vividly.rules.egress): the person has, by
    definition, already been reaped, so there is no live state to hand over and the
    firing's own details are the evidence. A verifier that needs a TrackState must answer
    "unavailable" for those rather than raise.
    """

    verifier_id: str
    verifier_version: str

    def verify(self, firing: RuleFiring, track: TrackState | None) -> VerifierVerdict: ...


class AbstainingVerifier:
    """The abstaining implementation: answers "unavailable" to every firing.

    Not a mock and not dead code: this is the shipped default for a deployment that wants
    the fail-closed "require" policy with no head wired (nothing alerts), and the
    annotate-mode stamp that makes "no verifier judged this" visible on every alert record
    instead of implicit. A configured ONNX backend does not route through it: that backend
    answers "unavailable" itself for a rule id it has no head for.
    """

    verifier_id = "abstaining_v0"
    verifier_version = "0.1.0"

    def verify(self, firing: RuleFiring, track: TrackState | None) -> VerifierVerdict:
        return VerifierVerdict(
            verdict="unavailable",
            verifier_id=self.verifier_id,
            verifier_version=self.verifier_version,
        )


def annotate_firing(firing: RuleFiring, verdict: VerifierVerdict) -> RuleFiring:
    """A copy of the firing with the verdict stamped into its details.

    A copy, never a mutation: RuleFiring is frozen and the original may already be in
    someone else's hands. Detail keys are namespaced `verifier_*` so a rule's own keys
    can never collide with them.
    """
    details: dict[str, float | str] = {
        **firing.details,
        "verifier_verdict": verdict.verdict,
        "verifier_id": verdict.verifier_id,
        "verifier_version": verdict.verifier_version,
    }
    if verdict.confidence is not None:
        details["verifier_confidence"] = float(verdict.confidence)
    return replace(firing, details=details)
