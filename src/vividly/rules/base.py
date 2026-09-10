"""Rule primitives: the firing record, the rule protocols, and every threshold.

Replay-pipeline structure, recorded 2026-08-07 (POLICY.md, section 1) and re-scoped on
2026-08-19: the rule layer detects acute events (a fall confirmed by a prolonged
down-and-still period; convulsive motion), never gradual deterioration. The prevention-era
detectors (stillness_v0, drift_v0, variance_collapse_v0, no_staff_contact_v0) were removed
at that re-scope and must not come back without an owner decision. A rule is a pure
function of one TrackState and the current source time. Rules hold no mutable state: the
only thing a rule instance carries is a frozen RuleConfig. Debounce, evaluation cadence,
verifier dispatch and firing history live in vividly.rules.engine, so a rule can be called
twice with the same inputs and must answer identically.

False-positive stance: the product priority is avoiding false alarms, at the cost of
detection latency and of missed marginal events. Structurally that means every rule here
is a multi-condition cascade that abstains on thin, stale, or ambiguous evidence, the
engine debounces per (rule, track), and the engine carries a verifier seam
(vividly.rules.verify) so a trained event verifier can veto a firing before it becomes an
alert. No rule may fire on a single-condition signal.

Threshold status: every number in RuleConfig is PROVISIONAL, and no threshold in
RuleConfig has been fitted to data. The fine-tuned weights of record do exist, the eyes
checkpoint 345c30a5 with ONNX 40262049 and the joints checkpoint a0a035fe with ONNX
6ba96285 (docs/CALIBRATION.md, section 2), but no detector or pose error distribution has
been fitted against shadow-mode data, because no shadow-mode data exists. The defaults
below are shapes with the right units, not clinically meaningful operating points. The
shipped values live in configs/pipeline/rules.yaml and each one carries the same warning.
See docs/CALIBRATION.md for the protocol that eventually replaces them, and note that it
cannot even start until its three outstanding preconditions are met: pilot shadow-mode
data, an agreed alerts-per-nurse-per-shift ceiling and a site data agreement.

Import rule: standard library plus pyyaml plus vividly.tracks only. No numpy, no cv2, no
torch, no rtmlib, not even lazily. Linux CI syncs the default fork, where numpy and cv2
are darwin-marked and therefore absent, and it collects this package.
tests/test_rules.py pins that with an AST check over every module in the package.

Clock: `now` is source time in seconds (frame index / fps for a replay source), never wall
clock. Every window in this package is anchored at `now`, not at a track's own
last_seen_t, so a track that has gone quiet inside the reap window cannot keep claiming a
full window of evidence. window_positions and window_postures below are what make that
true, and every rule goes through them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from vividly.tracks import BaselineStats, TrackBook, TrackState

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class RuleFiring:
    """One rule firing on one track.

    onset_t is when the condition began, detected_t is when the engine noticed. The two
    differ by at least the rule's own confirmation window, which is the point: the
    nurse-facing promise is the time the event started, not the time a threshold was
    crossed. `baseline` is the track's own baseline at firing time, carried along so a
    downstream record can show what the comparison was made against, and it is None when
    the track has no baseline yet (the acute-event rules do not require one: a person can
    collapse thirty seconds after entering the frame).
    """

    rule_id: str
    rule_version: str
    track_id: int
    onset_t: float
    detected_t: float
    baseline: BaselineStats | None
    details: dict[str, float | str]


@runtime_checkable
class Rule(Protocol):
    """A pure detector over one track.

    Implementations are stateless apart from their frozen config. `evaluate` returns a
    RuleFiring the moment the condition holds and keeps returning one for as long as it
    holds; suppressing the repeats is the engine's job, not the rule's.
    """

    rule_id: str
    rule_version: str

    def evaluate(self, track: TrackState, now: float) -> RuleFiring | None: ...


@runtime_checkable
class BookRule(Protocol):
    """A rule that also needs to see the other tracks in the room.

    No default rule needs this since no_staff_contact_v0 was removed, but the engine keeps
    the dispatch seam: a future room-context rule (e.g. corroborating a collapse against
    bystander motion) implements evaluate_book and the engine calls it without changes.
    The engine calls evaluate_book when a rule provides it and evaluate otherwise, so a
    BookRule is still a Rule.
    """

    rule_id: str
    rule_version: str

    def evaluate(self, track: TrackState, now: float) -> RuleFiring | None: ...

    def evaluate_book(
        self, book: TrackBook, track: TrackState, now: float
    ) -> RuleFiring | None: ...


@dataclass(frozen=True)
class RuleConfig:
    """Every threshold the two event rules, the egress head and the engine read.

    Every field below is PROVISIONAL under the standing threshold discipline: a placeholder
    chosen to be structurally sane, never a fitted value, and it stays provisional until the
    three outstanding preconditions of docs/CALIBRATION.md, section 2 are met: pilot
    shadow-mode data, an agreed alerts-per-nurse-per-shift ceiling and a site data
    agreement. The weights precondition is already met by the lane models of record, eyes
    345c30a5 and joints a0a035fe. The shipped values are configs/pipeline/rules.yaml; these
    defaults exist so a unit test can construct a config without a file, and the two are
    pinned equal by a test.
    """

    # -- engine ---------------------------------------------------------------
    # PROVISIONAL: a (rule_id, track_id) pair fires at most once per this many seconds.
    # 15 minutes is a guess at "long enough that one event does not fill the queue, short
    # enough that a nurse who dismissed one still gets told again". Note that
    # fall_then_still_v0 alerts on the event, not the state: once the fall evidence ages
    # out of collapse_search_s, a person still on the ground does not re-alert (see
    # vividly.rules.collapse), so this debounce is a repeat-guard, not the only guard.
    refire_after_s: float = 900.0

    # Off by default (0.0), and PROVISIONAL if ever turned on. A second, coarser debounce
    # scoped to the rule alone: while it is running, that rule alerts at most once per
    # this many seconds no matter which track fires it. The replay pipeline opens exactly
    # one source per process, so rule scope is camera scope; this is the live equivalent
    # of the camera-level episode merge the decision-layer instrument reports
    # (research/operating-points.md, section 5, item 2, per-camera episode merge), where
    # two tracks on one camera during one event count as one episode instead of two
    # alerts. 0.0 disables it entirely and leaves the per-(rule, track) debounce as the
    # only guard, which is what every run before 2026-09-05 did. Measured value at the
    # recommended operating points is real but modest: the seizure model of record, onnx
    # sha8 07c399be, at 0.8 goes from 54.4 to 32.6 eps/h on the 0.184 h of E2 negatives,
    # while the out-of-fold and fall surfaces save close to nothing. So it stays off until
    # pilot data says otherwise.
    camera_refire_after_s: float = 0.0

    # PROVISIONAL: minimum source-time gap between two evaluation passes. The pipeline
    # ticks once per frame (15 fps), and re-deciding 15 times a second costs a lot and
    # changes nothing, because every confirmation window is tens of seconds long. Onset
    # accuracy does not depend on this: onsets are recovered from stored history, not
    # from tick timing.
    eval_every_s: float = 1.0

    # -- fall_then_still_v0 -----------------------------------------------------
    # PROVISIONAL: how far back the evidence scan may look, and therefore the oldest a
    # fall may be and still produce an alert. Caps the cost of the backward scan. Must
    # cover fall_max_s + down_still_min_s + down_stale_s or the cascade can never
    # complete; RuleConfig raises if it does not.
    collapse_search_s: float = 120.0

    # PROVISIONAL: torso angle (degrees from vertical) at or below which a person counts
    # as upright before the fall. The cascade requires an upright sample inside the search
    # window: a person who entered the frame already on the ground, or who slid down over
    # minutes, produces no fall transition and the rule abstains (precision first).
    fall_upright_max_deg: float = 30.0

    # PROVISIONAL: torso angle at or above which a person counts as down. The down-phase
    # span is the contiguous run of samples at or above this angle.
    fall_down_min_deg: float = 60.0

    # PROVISIONAL: the upright-to-down transition must complete within this many seconds.
    # This is the gate that separates a fall from the slow slump the removed drift rule
    # used to chase: a transition longer than this is not an acute event and must not
    # alert here.
    fall_max_s: float = 3.0

    # PROVISIONAL: the box centre must move down the image by at least this many pixels
    # across the transition. Guards against pose-noise angle flips with no actual body
    # displacement. Pixels on the source frame: resolution and mount dependent, so this
    # is a per-site number pretending to be a constant.
    fall_drop_min_px: float = 40.0

    # PROVISIONAL: how long the person must stay down and still after the fall before the
    # cascade fires. This is the confirmation stage: a fall alone never alerts. Trading
    # detection latency for precision is the false-positive stance, in one number.
    down_still_min_s: float = 30.0

    # PROVISIONAL: mean per-step centre displacement, in pixels, at or below which the
    # down phase counts as still. Above it the rule abstains: a person moving on the
    # ground is not "down and still" (convulsive movement is seizure_candidate_v0's job).
    down_move_max_px: float = 6.0

    # PROVISIONAL: minimum posture samples inside the down span before it counts.
    down_min_samples: int = 15

    # PROVISIONAL: the newest posture sample may not be older than this, and the down
    # span may not bridge a posture blackout longer than this. Down-and-still is a claim
    # about right now, and stale evidence does not support it.
    down_stale_s: float = 5.0

    # -- seizure_candidate_v0 ---------------------------------------------------
    # PROVISIONAL: the window convulsive motion is measured over.
    convulse_window_s: float = 12.0

    # PROVISIONAL: the oscillation must span at least this many seconds inside the
    # window. Tonic-clonic convulsions run tens of seconds; a two-second shudder must
    # not alert.
    convulse_min_s: float = 8.0

    # PROVISIONAL: fraction of consecutive step pairs whose directions reverse (negative
    # dot product). Rhythmic convulsion reverses direction nearly every step; walking and
    # seated fidgeting almost never do. This is the signature, not the amplitude.
    convulse_reversal_frac: float = 0.6

    # PROVISIONAL: mean per-step centre displacement, in pixels, below which the motion
    # is jitter rather than convulsion and the rule abstains.
    convulse_amp_min_px: float = 3.0

    # PROVISIONAL: minimum position samples inside the window.
    convulse_min_samples: int = 20

    # PROVISIONAL: the newest position sample may not be older than this. Convulsion is
    # a claim about right now.
    convulse_stale_s: float = 2.0

    # -- egress_without_discharge_v0 (vividly.rules.egress) ----------------------
    # PROVISIONAL: how long a person who left through an exit zone must stay away, with
    # no discharge tag and no human-confirmed return, before the egress candidate fires.
    # The product statement is a person absent for more than N minutes; N is a placeholder
    # until shadow-mode data exists (docs/CALIBRATION.md, section 2, precondition 2). It
    # should exceed TrackConfig.reap_after_s by a wide margin (a departure is only noticed
    # at reap time), which every shipped config satisfies.
    egress_absent_min_s: float = 600.0

    # PROVISIONAL: minimum time the person must have been present before their exit
    # counts as a patient leaving. A walk-through or a visitor who looked in is not an
    # egress candidate; the presence floor is what says so.
    egress_min_presence_s: float = 120.0

    # PROVISIONAL: how far back along the track's own history (from its last sighting) an
    # exit-zone sample may lie for the departure to count as "through the exit". A couple
    # of frames of tracker noise at the door must not turn an exit into a mid-room loss.
    egress_exit_window_s: float = 5.0

    # -- not thresholds ------------------------------------------------------------
    # The rule allowlist (P2 of the Clip Library design, 2026-09-09): None means every rule
    # head this package ships, which is what every profile before the allowlist existed
    # ran. A profile that writes a Clip spool must state the list; run_pipeline enforces
    # that. Excluded from equality so a config with a list still equals the same thresholds.
    enabled: tuple[str, ...] | None = field(default=None, compare=False)
    # Which file each threshold came from (`load_rule_config` fills it), so a spool record
    # can name the fired rule's own file and hash it. Never a threshold; never compared.
    sources: Mapping[str, str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.enabled is not None:
            unknown = sorted(set(self.enabled) - set(ALL_RULE_IDS))
            if unknown:
                raise ValueError(
                    f"rules.enabled names unknown rule ids {unknown}; known ids are {ALL_RULE_IDS}"
                )
        positive = (
            "refire_after_s",
            "collapse_search_s",
            "fall_upright_max_deg",
            "fall_down_min_deg",
            "fall_max_s",
            "fall_drop_min_px",
            "down_still_min_s",
            "down_move_max_px",
            "down_stale_s",
            "convulse_window_s",
            "convulse_min_s",
            "convulse_amp_min_px",
            "convulse_stale_s",
            "egress_absent_min_s",
            "egress_min_presence_s",
            "egress_exit_window_s",
        )
        for name in positive:
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.eval_every_s < 0:
            raise ValueError(f"eval_every_s must be >= 0, got {self.eval_every_s}")
        # 0.0 is off, so this one is >= 0 rather than > 0 like refire_after_s.
        if self.camera_refire_after_s < 0:
            raise ValueError(
                f"camera_refire_after_s must be >= 0 (0 disables it), "
                f"got {self.camera_refire_after_s}"
            )
        if not 0.0 <= self.convulse_reversal_frac <= 1.0:
            raise ValueError(
                f"convulse_reversal_frac must be in [0, 1], got {self.convulse_reversal_frac}"
            )
        for name in ("down_min_samples", "convulse_min_samples"):
            value = getattr(self, name)
            if value < 2:
                raise ValueError(f"{name} must be >= 2, got {value}")
        if self.fall_down_min_deg <= self.fall_upright_max_deg:
            raise ValueError(
                "fall_down_min_deg must be > fall_upright_max_deg, otherwise upright and "
                f"down overlap: {self.fall_down_min_deg} <= {self.fall_upright_max_deg}"
            )
        needed = self.fall_max_s + self.down_still_min_s + self.down_stale_s
        if self.collapse_search_s < needed:
            raise ValueError(
                "collapse_search_s must be >= fall_max_s + down_still_min_s + down_stale_s, "
                "otherwise the scan cannot hold a completed cascade: "
                f"{self.collapse_search_s} < {needed}"
            )
        if self.convulse_window_s < self.convulse_min_s:
            raise ValueError(
                "convulse_window_s must be >= convulse_min_s, otherwise the window cannot "
                f"hold a qualifying span: {self.convulse_window_s} < {self.convulse_min_s}"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RuleConfig:
        """Build from a config block (the `rules:` mapping of a pipeline config).

        Unknown keys raise. A typo in a threshold name that silently kept a default is
        exactly the failure this project cannot afford: it would look like a calibrated
        run and be an uncalibrated one.
        """
        known = cls.threshold_names()
        unknown = sorted(set(data) - set(known) - _META_FIELDS)
        if unknown:
            raise ValueError(f"unknown RuleConfig keys: {unknown}; known keys are {sorted(known)}")
        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            if name == "enabled":
                kwargs[name] = _enabled_tuple(value)
            elif name == "sources":
                kwargs[name] = dict(value)
            elif name in _INT_FIELDS:
                kwargs[name] = int(value)
            else:
                kwargs[name] = float(value)
        return cls(**kwargs)

    @classmethod
    def threshold_names(cls) -> tuple[str, ...]:
        """Every threshold field, in declaration order; never the meta fields."""
        return tuple(f.name for f in fields(cls) if f.name not in _META_FIELDS)

    def rule_thresholds(self, rule_id: str) -> dict[str, float | int]:
        """The thresholds one rule head reads, and nothing from any other head."""
        return {name: getattr(self, name) for name in RULE_KEYS[rule_id]}

    def rule_file(self, rule_id: str) -> str | None:
        """The one file every threshold of `rule_id` was loaded from, or None when the
        thresholds came from more than one file or from no file at all."""
        files = {self.sources.get(name) for name in RULE_KEYS[rule_id]}
        if len(files) == 1 and None not in files:
            return next(iter(files))
        return None


_INT_FIELDS = frozenset({"down_min_samples", "convulse_min_samples"})
_META_FIELDS = frozenset({"enabled", "sources"})

#: Every rule head this package ships, in the order the engine evaluates them; the egress
#: head is an engine attachment rather than a Rule but has thresholds of its own.
ALL_RULE_IDS: tuple[str, ...] = (
    "fall_then_still_v0",
    "seizure_candidate_v0",
    "egress_without_discharge_v0",
)

#: Which threshold belongs to which head (the engine's own keys under "engine"). One file
#: per group under configs/pipeline/rules/; tests/test_rules.py pins that this partition
#: covers every threshold exactly once and that each file holds only its own keys.
RULE_KEYS: dict[str, tuple[str, ...]] = {
    "engine": ("refire_after_s", "camera_refire_after_s", "eval_every_s"),
    "fall_then_still_v0": (
        "collapse_search_s",
        "fall_upright_max_deg",
        "fall_down_min_deg",
        "fall_max_s",
        "fall_drop_min_px",
        "down_still_min_s",
        "down_move_max_px",
        "down_min_samples",
        "down_stale_s",
    ),
    "seizure_candidate_v0": (
        "convulse_window_s",
        "convulse_min_s",
        "convulse_reversal_frac",
        "convulse_amp_min_px",
        "convulse_min_samples",
        "convulse_stale_s",
    ),
    "egress_without_discharge_v0": (
        "egress_absent_min_s",
        "egress_min_presence_s",
        "egress_exit_window_s",
    ),
}


def _enabled_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ValueError(f"rules.enabled must be a list of rule ids, got {value!r}")
    out = tuple(str(v) for v in value)
    if len(set(out)) != len(out):
        raise ValueError(f"rules.enabled repeats a rule id: {list(out)}")
    return out


def load_rule_config(path: str | Path) -> RuleConfig:
    """Load configs/pipeline/rules.yaml (or the `rules:` block of any pipeline config).

    Accepts either a file whose top level is the threshold mapping, or one that nests it
    under a `rules:` key, so a pipeline config can carry its thresholds inline or point at
    the shared file. Every value still goes through RuleConfig validation.

    A file may carry `include: [path, ...]` (relative to the file, or absolute): the
    included files are loaded first, and a key that two included files both set is
    refused, so a threshold has exactly one home. Keys stated beside the `include` list
    override the included value and are recorded as coming from this file, which is how a
    demo profile shrinks a threshold and can still say where every number came from.
    `sources` on the returned config names the file behind every key.
    """
    config_path = Path(path)
    merged, sources = _load_rule_mapping(config_path, seen=())
    return RuleConfig.from_mapping({**merged, "sources": sources})


def _rule_block(config_path: Path) -> Mapping[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"{config_path} must contain a mapping, got {type(data).__name__}")
    block = data.get("rules", data)
    if not isinstance(block, Mapping):
        raise ValueError(f"{config_path}: `rules` must be a mapping, got {type(block).__name__}")
    return block


def merge_rule_mapping(
    block: Mapping[str, Any], *, base_dir: Path, origin: str, seen: tuple[Path, ...] = ()
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve `include` inside one rules mapping. Returns (thresholds, key -> file)."""
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    includes = block.get("include", ())
    if isinstance(includes, str) or not isinstance(includes, list | tuple):
        raise ValueError(f"{origin}: `include` must be a list of paths")
    for item in includes:
        target = Path(str(item)).expanduser()
        if not target.is_absolute():
            target = base_dir / target
        target = target.resolve()
        if target in seen:
            raise ValueError(f"{origin}: `include` cycle through {target}")
        part, part_sources = _load_rule_mapping(target, seen=(*seen, target))
        clash = sorted(set(part) & set(merged))
        if clash:
            raise ValueError(
                f"{origin}: thresholds {clash} are set by more than one included file "
                f"({sources[clash[0]]} and {part_sources[clash[0]]}); a threshold has one home"
            )
        merged.update(part)
        sources.update(part_sources)
    for key, value in block.items():
        if key == "include":
            continue
        merged[key] = value
        if key not in _META_FIELDS:
            sources[key] = origin
    return merged, sources


def _load_rule_mapping(
    config_path: Path, *, seen: tuple[Path, ...]
) -> tuple[dict[str, Any], dict[str, str]]:
    block = _rule_block(config_path)
    return merge_rule_mapping(
        block, base_dir=config_path.parent, origin=str(config_path), seen=seen or (config_path,)
    )


def window_positions(
    track: TrackState, now: float, window_s: float
) -> list[tuple[float, float, float]]:
    """(t, cx, cy) samples with now - window_s <= t <= now, oldest first.

    TrackState windows are anchored at the track's own last_seen_t, because the contract
    methods take no `now`. Rules must be anchored at `now` instead, at both ends:

      * A track that stopped producing detections 20 seconds ago is still in the book (it
        is inside reap_after_s) and must not be handed a full window that quietly ends 20
        seconds in the past. The widened lookback is what covers that.
      * A rule evaluated at a `now` earlier than the track's newest sample must not see
        the future. That happens when a caller re-evaluates a past instant, which replay
        seeking makes easy to do by accident, and a rule that answers using samples from
        after the timestamp it was asked about is not answering the question.
    """
    if not window_s > 0:
        raise ValueError(f"window_s must be > 0, got {window_s}")
    if track.n_observations == 0 or math.isnan(now):
        return []
    lookback = window_s + abs(now - track.last_seen_t)
    cutoff = now - window_s
    return [p for p in track.positions(lookback) if cutoff <= p[0] <= now]


def window_postures(track: TrackState, now: float, window_s: float) -> list[tuple[float, float]]:
    """(t, degrees-from-vertical) samples with now - window_s <= t <= now, oldest first.

    Same anchoring as window_positions. Shorter than window_positions over the same window
    whenever pose confidence dropped below TrackConfig.kp_conf_min, and empty for a track
    that never carried keypoints.
    """
    if not window_s > 0:
        raise ValueError(f"window_s must be > 0, got {window_s}")
    if track.n_observations == 0 or math.isnan(now):
        return []
    lookback = window_s + abs(now - track.last_seen_t)
    cutoff = now - window_s
    return [p for p in track.posture_angles(lookback) if cutoff <= p[0] <= now]
