"""Rule layer: acute-event detection.

Two detectors plus a debouncing engine with a verifier seam, sitting on top of
vividly.tracks. fall_then_still_v0 is the collapse cascade (a fast fall, confirmed by a
prolonged down-and-still period, which is the visible signature of stroke, cardiac arrest
and syncope, and only the trained verifier can tell those apart); seizure_candidate_v0 is
sustained convulsive motion. The prevention-era detectors (stillness, postural drift,
variance collapse, staff proximity) were removed when the layer was re-scoped to acute
events: this layer detects events, never gradual deterioration, and its stated priority is
avoiding false positives.

The third head of the product, egress_without_discharge_v0 (vividly.rules.egress), is
a state machine over identity rather than a motion rule: a person who left through a
configured exit zone, stayed away, was not tagged discharged and was not confirmed to have
come back. It is an attachment of the engine (RuleEngine(..., egress=EgressMonitor(...)))
because its subject is a track that has already been reaped; its firings take the same
debounce and verifier path as the two rules.

Import rule: standard library plus pyyaml plus vividly.tracks only. No numpy, no cv2, no
model runtime, so it imports cleanly on a Linux CI runner that has none of them installed.

Every threshold lives in configs/pipeline/rules.yaml, every one of them is PROVISIONAL
until the three outstanding preconditions of docs/CALIBRATION.md, section 2 are met (pilot
shadow-mode data, an agreed alerts-per-nurse-per-shift ceiling and a site data agreement),
and that document is the protocol that replaces them. The weights precondition is already
met by the lane models of record, eyes 345c30a5 and joints a0a035fe. Do not quote a number
from here as a product number and do not read a firing count from any run here as a
false-alarm rate.
"""

from __future__ import annotations

from vividly.rules.base import (
    SECONDS_PER_MINUTE,
    BookRule,
    Rule,
    RuleConfig,
    RuleFiring,
    load_rule_config,
    window_positions,
    window_postures,
)
from vividly.rules.collapse import FallThenStillRule
from vividly.rules.convulsion import ConvulsiveMotionRule
from vividly.rules.egress import (
    EGRESS_RULE_ID,
    EGRESS_RULE_VERSION,
    Departure,
    EgressMonitor,
    EgressZones,
    ExitZone,
)
from vividly.rules.engine import DEFAULT_RULE_IDS, RuleEngine, build_default_rules
from vividly.rules.verify import (
    VERIFIER_VERDICTS,
    VERIFY_POLICIES,
    AbstainingVerifier,
    EventVerifier,
    VerifierVerdict,
    annotate_firing,
)

__all__ = [
    "DEFAULT_RULE_IDS",
    "EGRESS_RULE_ID",
    "EGRESS_RULE_VERSION",
    "SECONDS_PER_MINUTE",
    "VERIFIER_VERDICTS",
    "VERIFY_POLICIES",
    "AbstainingVerifier",
    "BookRule",
    "ConvulsiveMotionRule",
    "Departure",
    "EgressMonitor",
    "EgressZones",
    "EventVerifier",
    "ExitZone",
    "FallThenStillRule",
    "Rule",
    "RuleConfig",
    "RuleEngine",
    "RuleFiring",
    "VerifierVerdict",
    "annotate_firing",
    "build_default_rules",
    "load_rule_config",
    "window_positions",
    "window_postures",
]
