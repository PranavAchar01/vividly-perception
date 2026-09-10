"""RuleEngine: the only stateful object in this package.

Rules are pure and keep answering yes for as long as their condition holds, which is
correct for a detector and unusable for a queue: one event would produce a firing per
tick until its evidence aged out. Everything that turns "the condition holds" into "a
human should be told" lives here, and there are exactly four pieces of it.

  * Debounce. A (rule_id, track_id) pair fires at most once per cfg.refire_after_s. Keyed
    on the pair, so two rules on one person, or one rule on two people, never suppress
    each other. Optionally, and off by default, a second coarser debounce keyed on the
    rule_id alone (cfg.camera_refire_after_s) merges a rule's alerts across every track
    in the process: the pipeline opens one source per process, so rule scope is camera
    scope, and this is the live form of the camera-level episode merge the decision-layer
    instrument reports (research/operating-points.md, section 5, item 2, per-camera
    episode merge). At 0.0, the shipped value, it is not consulted at all and the engine
    behaves exactly as it did before it existed. Its map is keyed by rule, not by track, so it deliberately survives the
    debounce cleanup below: a camera does not stop being one camera because a track was
    reaped.
  * Verification. When a verifier is attached (vividly.rules.verify), every firing
    that survives debounce is put to it before being returned: "rejected" drops the
    firing, "unavailable" drops it too under the "require" policy and passes it through
    stamped under "annotate". This is the seam the trained event verifier plugs into,
    and the reason a false positive can be stopped before it becomes an alert.
  * The egress head (vividly.rules.egress). Optional, attached at construction: after the
    per-track rules have run, the engine asks the EgressMonitor for the departures whose
    clock has run out, and those firings go through the same debounce and verifier path.
    It lives on the engine rather than in the rule list because it is not a pure function
    of a live track, since an egress candidate is a track that has already been reaped,
    and this engine is where the package keeps its state.
  * Evaluation cadence. cfg.eval_every_s skips evaluation passes that arrive too close
    together in source time. The pipeline ticks once per frame at 15 fps and every
    confirmation window is tens of seconds long, so re-deciding 15 times a second costs
    a lot and changes nothing. It does not affect onset accuracy: every onset is
    recovered from stored history by the rule, not from when the engine happened to look.
  * Debounce cleanup. When a track leaves the book (TrackBook reaps it after
    reap_after_s), its debounce entries go with it. Otherwise the table grows for the
    life of the process, and a recycled tracker id would inherit a stranger's suppression.
    A track that is merely absent for a few seconds is still in the book and keeps its
    entries, which is what makes leave-and-return quiet rather than noisy.

Debounce is recorded before the verifier is consulted, on purpose: a rejected firing
starts the same refire window a reported one does, so a persisting condition the verifier
keeps rejecting costs one verifier call per refire window, not one per evaluation pass.

Determinism is a requirement, not a nicety: tracks are evaluated in ascending track_id
order and rules in the order they were passed, so the returned list is byte-identical for
identical input (given a deterministic verifier). Sibling layers (the event store, the
dashboard) assume that.
"""

from __future__ import annotations

import math

from vividly.rules.base import ALL_RULE_IDS, Rule, RuleConfig, RuleFiring
from vividly.rules.collapse import FallThenStillRule
from vividly.rules.egress import EgressMonitor
from vividly.rules.verify import VERIFY_POLICIES, EventVerifier, annotate_firing
from vividly.tracks import TrackBook, TrackState

DEFAULT_RULE_IDS: tuple[str, ...] = (
    "fall_then_still_v0",
    "seizure_candidate_v0",
)


def build_default_rules(cfg: RuleConfig | None = None) -> list[Rule]:
    """The acute-event rules, in DEFAULT_RULE_IDS order, sharing one config.

    `cfg.enabled` (None means every rule) is the allowlist: a rule id absent from it is
    never constructed, and its module is never imported. The convulsion rule is imported
    here rather than at module level for exactly that reason: a wheel built for a Clip
    Library site ships without `vividly/rules/convulsion.py`, and this function still runs.
    """
    config = RuleConfig() if cfg is None else cfg
    wanted = set(ALL_RULE_IDS) if config.enabled is None else set(config.enabled)
    rules: list[Rule] = []
    if "fall_then_still_v0" in wanted:
        rules.append(FallThenStillRule(config))
    if "seizure_candidate_v0" in wanted:
        from vividly.rules.convulsion import ConvulsiveMotionRule

        rules.append(ConvulsiveMotionRule(config))
    return rules


class RuleEngine:
    """Runs rules over a TrackBook, debounces, and dispatches the verifier seam."""

    def __init__(
        self,
        rules: list[Rule],
        cfg: RuleConfig | None = None,
        verifier: EventVerifier | None = None,
        verify_policy: str = "annotate",
        egress: EgressMonitor | None = None,
    ) -> None:
        if verify_policy not in VERIFY_POLICIES:
            raise ValueError(
                f"verify_policy must be one of {VERIFY_POLICIES}, got {verify_policy!r}"
            )
        if verifier is not None and not callable(getattr(verifier, "verify", None)):
            raise TypeError(
                f"verifier must implement EventVerifier.verify, got {type(verifier).__name__}"
            )
        if egress is not None and not callable(getattr(egress, "tick", None)):
            raise TypeError(
                f"egress must be an EgressMonitor (it needs .tick), got {type(egress).__name__}"
            )
        self.rules = list(rules)
        self.cfg = RuleConfig() if cfg is None else cfg
        self.verifier = verifier
        self.verify_policy = verify_policy
        self.egress = egress
        self._last_fired: dict[tuple[str, int], float] = {}
        # Rule-scoped (= camera-scoped) debounce, consulted only when
        # cfg.camera_refire_after_s > 0. Keyed by rule_id alone and never pruned by track
        # reaping: see the module docstring.
        self._last_fired_camera: dict[str, float] = {}
        self._last_eval_t: float | None = None
        self.n_ticks: int = 0
        self.n_evaluations: int = 0
        self.n_suppressed: int = 0
        # The subset of n_suppressed that the camera-scoped debounce dropped; always 0
        # while that debounce is off.
        self.n_camera_suppressed: int = 0
        self.n_rejected: int = 0
        self.n_unconfirmed: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        ids = ", ".join(rule.rule_id for rule in self.rules)
        verifier = "none" if self.verifier is None else self.verifier.verifier_id
        egress = "off" if self.egress is None else self.egress.rule_id
        return (
            f"RuleEngine([{ids}], refire_after_s={self.cfg.refire_after_s}, "
            f"verifier={verifier}/{self.verify_policy}, egress={egress})"
        )

    def tick(self, book: TrackBook, now: float) -> list[RuleFiring]:
        """Evaluate every rule against every live track and return the firings to report.

        Firings suppressed by debounce are counted (n_suppressed) and dropped; firings
        the verifier rejects (n_rejected) or, under the "require" policy, cannot confirm
        (n_unconfirmed) are dropped too. `now` is source time; a nan clock (an empty book
        that has never seen an observation) is a no-op rather than an error, because a
        pipeline may tick before its first frame produces a detection.
        """
        self.n_ticks += 1
        if math.isnan(now):
            return []
        if self._last_eval_t is not None and now - self._last_eval_t < self.cfg.eval_every_s:
            return []
        self._last_eval_t = now
        self.n_evaluations += 1

        tracks = book.tracks()
        self._forget_missing(set(tracks))

        # Scene-aware verifier backends (heads trained with WindowSpec.scene_channels)
        # need the whole book, not just the firing track; same optional-hook pattern as
        # a rule's evaluate_book. Before the rules run, so this tick's own firings are
        # verified against the book the rules saw.
        if self.verifier is not None:
            observe = getattr(self.verifier, "observe_book", None)
            if observe is not None:
                observe(book, now)

        firings: list[RuleFiring] = []
        for track_id in sorted(tracks):
            track = tracks[track_id]
            for rule in self.rules:
                evaluate_book = getattr(rule, "evaluate_book", None)
                if evaluate_book is not None:
                    firing = evaluate_book(book, track, now)
                else:
                    firing = rule.evaluate(track, now)
                if firing is None:
                    continue
                admitted = self._admit(firing, track, now)
                if admitted is not None:
                    firings.append(admitted)
        if self.egress is not None:
            # After the live tracks, so the monitor's snapshot of this tick reflects the
            # book the rules just saw. A departed track has no TrackState to hand the
            # verifier; it gets None (see EventVerifier).
            for firing in self.egress.tick(book, now):
                admitted = self._admit(firing, None, now)
                if admitted is not None:
                    firings.append(admitted)
        return firings

    def _admit(self, firing: RuleFiring, track: TrackState | None, now: float) -> RuleFiring | None:
        """Debounce, then the verifier seam. None means the firing is dropped (and counted)."""
        key = (firing.rule_id, firing.track_id)
        last = self._last_fired.get(key)
        if last is not None and now - last < self.cfg.refire_after_s:
            self.n_suppressed += 1
            return None
        camera_window = self.cfg.camera_refire_after_s
        if camera_window > 0:
            # Checked after the per-track debounce and before the per-track timestamp is
            # written: a firing this suppresses did not happen, so it must not start the
            # track's own 15-minute refire window and mute that person on the strength of
            # somebody else's alert.
            camera_last = self._last_fired_camera.get(firing.rule_id)
            if camera_last is not None and now - camera_last < camera_window:
                self.n_suppressed += 1
                self.n_camera_suppressed += 1
                return None
            self._last_fired_camera[firing.rule_id] = now
        self._last_fired[key] = now
        if self.verifier is None:
            return firing
        verdict = self.verifier.verify(firing, track)
        if verdict.verdict == "rejected":
            self.n_rejected += 1
            return None
        if verdict.verdict == "unavailable" and self.verify_policy == "require":
            self.n_unconfirmed += 1
            return None
        return annotate_firing(firing, verdict)

    def last_fired_t(self, rule_id: str, track_id: int) -> float | None:
        """When this (rule_id, track_id) last produced a debounce-admitted firing, or None.

        "Admitted" includes firings the verifier then dropped: the refire window starts
        when the condition fired, not when a reviewer got told. See the module docstring.
        """
        return self._last_fired.get((rule_id, track_id))

    def last_fired_camera_t(self, rule_id: str) -> float | None:
        """When this rule last produced a camera-debounce-admitted firing, or None.

        Always None while cfg.camera_refire_after_s is 0: the map is only written when
        the camera-scoped debounce is running.
        """
        return self._last_fired_camera.get(rule_id)

    def forget(self, track_id: int) -> None:
        """Drop every per-(rule, track) debounce entry for one track id.

        The camera-scoped map is untouched on purpose: it is keyed by rule, its subject
        is the camera rather than the person, and clearing it when a track is reaped
        would let the next person through the door re-open an episode this one just
        opened.
        """
        for key in [k for k in self._last_fired if k[1] == track_id]:
            del self._last_fired[key]

    def _forget_missing(self, live_ids: set[int]) -> None:
        for key in [k for k in self._last_fired if k[1] not in live_ids]:
            del self._last_fired[key]
