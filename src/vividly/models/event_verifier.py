"""ONNX event-verifier backend: the trained implementation of the verify seam.

This is the model side of vividly.rules.verify, the first real backend of the
EventVerifier protocol. The engine hands it one RuleFiring plus the live TrackState it
fired on; this module rebuilds the exact row schema the training extraction recorded
(scripts/events/extract_track_sequences.py) from the track's rings and pushes it
through the same featurization the trainer consumed (vividly.verifier.features), so a
window scored here is distributed like a window the model was fitted on. Anything less
than that parity would make every threshold meaningless.

Weight class. Every model this backend loads is an interim prototype under the carve-out
in POLICY.md, rule 4 (vividly_weight_class 'prototype'): badged internal demonstration
only, never quoted as clinical detection performance, and replaced when clinical footage
exists. The two heads of record trained on different corpora: the fall head on the
public, licence-reviewed acted-fall sets le2i_fdd, gmdcsa24 and caucafall, and the
seizure head on the registered clinical train split wu_sahzu_emu_trainsplit, both read as
track and pose features and never as imagery (final_models/fall/model.json,
final_models/seizure/model.json). The training report's disclosure travels in report.json
next to the graph, but it is fixed boilerplate whose acted-fall and synthetic-motion
sentence describes the fall lane rather than both, so a head's corpus is read from its
sidecar. The demo overlay carries the badge.

Heads. One ONNX graph per rule id ("fall_then_still_v0", "seizure_candidate_v0" today).
A firing whose rule id has no head, egress firings included, gets "unavailable", never
a guess. Each head is built from a training run directory (outputs/verifier/<name>/):
the window geometry comes out of the embedded config in report.json rather than being
restated in the pipeline config, because a restated window that drifted from the one
the graph was trained with would silently mis-time every feature; and verifier.onnx is
refused unless its sha256 matches the report, which is the checkpoint provenance rule in
POLICY.md, rule 8. The sha8 rides in verifier_version, so every stamped alert names its
weights.

Scene channels. A head trained with WindowSpec.scene_channels (seizure final_v2
onward) needs what the training extraction had: every track's observed span in the
scene, not just the firing track. The engine supplies it by calling observe_book
(same getattr pattern as a rule's evaluate_book) with the TrackBook each tick; this
backend keeps a rolling span memory (SCENE_RETENTION_S) that outlives the book's
reap window, and refuses to score a scene-channels head if the hook never ran, because
a misconfigured pipeline must fail loudly rather than score on a half-empty scene.

Windowing. The window is anchored at the firing's onset_t, not at detection time: the
fall rule fires only after its down-and-still confirmation window, about 30 s after the
drop, so a trailing window at fire time would hold nothing but stillness. onset_t is
the rule's own statement of when the event began (last-upright time for the collapse
rule, sustained-motion start for the convulsion rule); the scored window is
[onset_t - pre_pad_s, onset_t - pre_pad_s + window_s], and the rings (default 300 s)
still hold it. featurize_window returning None, which means the track was absent for too
much of the window, becomes "unavailable", because thin evidence must abstain, not vote.

Threshold status: confirm_threshold defaults are PROVISIONAL (docs/CALIBRATION.md).
score >= threshold answers "confirmed", below answers "rejected"; the score itself is
always carried as the verdict confidence so the store shows the margin either way.

Persistence and hysteresis (optional, off by default; research/operating-points.md,
section 5, item 1, 2026-09-05). A head may be configured to require the decision-layer
study's persist:k@thr or hyst:hi|k@lo pattern instead of a single window:

    persist_k: 2      # score k windows at t_end + i*stride_s, confirm when all k >= thr
    hyst_hi: 0.9      # or confirm immediately if any of the k reaches this

persist_k defaults to 1 and hyst_hi to None, which is exactly one window at t_end and
the single-threshold decision this module has always made. The k == 1 path scores the
same array, calls the graph once, and returns the same verdict and confidence, so an
unconfigured head is byte-identical to the pre-2026-09-05 backend. The measurements are
what keep it off: persistence is negative for the fall head (UR ADL persist:2
catches 0 of 30) and negative for the seizure broad head at its elbow (persist:2@0.8
drops 2 convulsive-PG catches on the out-of-fold pool); the one recorded combination
that pays is chute24's hyst:0.9|2@0.65. This is the knob that expresses it, not a
recommendation to turn it on. Confidence reports the binding score: the maximum over the
k when the hysteresis leg confirmed or when the pattern failed, and the minimum over the
k when all k confirmed.

Abstention under persistence. If any of the k windows cannot be featurized, because the
track is absent for too much of it, the verdict is "unavailable", not "rejected": an
incomplete pattern is missing evidence, and thin evidence abstains rather than voting,
which is the same rule the single-window path follows.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vividly.models.checkpoints import is_lfs_pointer, sha256_of
from vividly.rules.base import RuleFiring
from vividly.rules.verify import VerifierVerdict
from vividly.tracks import TrackBook, TrackState
from vividly.verifier.features import (
    SceneContext,
    WindowSpec,
    featurize_window,
    tracks_from_rows,
)

#: Default seconds of pre-onset context inside the scored window. Training positives
#: contain the event at every offset the stride visited, so the model is tolerant of
#: where the event sits; 2 s of lead-in leaves a typical 1-3 s fall comfortably inside
#: an 8 s window. Override per head in the pipeline config's verify: block.
DEFAULT_PRE_PAD_S = 2.0

#: PROVISIONAL decision threshold (see module docstring).
DEFAULT_CONFIRM_THRESHOLD = 0.5

#: Windows scored per firing. 1 = the single anchor window at t_end, which is what every
#: head did before the persistence seam existed and what every head still does unless a
#: config asks for more (see the module docstring).
DEFAULT_PERSIST_K = 1

#: Config keys a `verify.heads.<rule_id>` block may carry. Unknown keys are refused
#: rather than ignored: a mistyped `confirm_treshold` that silently kept the 0.5 default
#: would make an uncalibrated run look like a calibrated one, which is the same failure
#: RuleConfig.from_mapping refuses for rule thresholds.
HEAD_CONFIG_KEYS: frozenset[str] = frozenset(
    {"run_dir", "pre_pad_s", "confirm_threshold", "persist_k", "hyst_hi"}
)

#: How long a track's observed span stays in the scene memory after it was last seen.
#: The scored window reaches at most window_s + pre_pad_s behind the firing's onset,
#: and onset itself trails `now` by the rule's confirmation window (about 30 s for the
#: fall rule); 120 s covers that whole reach with slack, so every track that
#: overlapped a scoreable window is still remembered even if the book reaped it
#: (reap_after_s 30) before the firing arrived.
SCENE_RETENTION_S = 120.0


class VerifierBackendError(RuntimeError):
    """A head that cannot be built: missing run dir, sha mismatch, bad report."""


def _as_float(value: Any, rule_id: str, key: str) -> float:
    """float(value) with the offending config location in the message, not a traceback."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise VerifierBackendError(f"verify.heads.{rule_id}.{key} must be a number, got {value!r}")
    try:
        return float(value)
    except ValueError as exc:
        raise VerifierBackendError(
            f"verify.heads.{rule_id}.{key} must be a number, got {value!r}"
        ) from exc


def _as_int(value: Any, rule_id: str, key: str) -> int:
    """int(value) that refuses a bool, a non-integral float, and anything non-numeric.

    int(2.5) would silently become 2, and a config that asked for two-and-a-half windows
    is a config that means something other than it says. Guessing on its behalf
    is how a run ends up not being the run its config describes.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerifierBackendError(
            f"verify.heads.{rule_id}.{key} must be a whole number, got {value!r}"
        )
    if isinstance(value, float) and not value.is_integer():
        raise VerifierBackendError(
            f"verify.heads.{rule_id}.{key} must be a whole number, got {value!r}"
        )
    return int(value)


@dataclass(frozen=True)
class VerifierHead:
    """One rule's trained graph plus everything needed to score with it.

    `session` is anything with onnxruntime's `run(None, {"windows": x}) -> [logits]`
    shape; tests inject fakes, from_config builds real CPU sessions.
    """

    session: Any
    spec: WindowSpec
    task: str
    weight_class: str
    sha8: str
    pre_pad_s: float = DEFAULT_PRE_PAD_S
    confirm_threshold: float = DEFAULT_CONFIRM_THRESHOLD
    # Calibration temperature from the training report (score = sigmoid(logit / T)).
    # 1.0 for uncalibrated heads. The eval harness applies the same division
    # (scripts/events/eval_event_verifier.py); a live path that skipped it would put
    # every calibrated model's thresholds on a different scale than every number
    # ever measured for it.
    temperature: float = 1.0
    # Consecutive stride-offset windows that must all clear confirm_threshold
    # (decision_layer_study's persist:k@thr). 1 = the single anchor window, off.
    persist_k: int = DEFAULT_PERSIST_K
    # The "hi" of hyst:hi|k@lo: any one of the k windows at or above this confirms
    # immediately, with confirm_threshold as the "lo". None = off. Meaningless with
    # persist_k == 1 (one window, hi >= lo, so both legs agree) and refused there.
    hyst_hi: float | None = None

    def __post_init__(self) -> None:
        validate_decision_policy(self.confirm_threshold, self.persist_k, self.hyst_hi)


def validate_decision_policy(
    confirm_threshold: float, persist_k: int, hyst_hi: float | None
) -> None:
    """Refuse a decision policy that cannot mean what it says.

    Shared by VerifierHead (so a hand-built head is checked too) and load_head, which
    calls it before opening the graph: a nonsense threshold should not cost an
    onnxruntime session first, the same way the calibration block is validated before
    the graph is opened.
    """
    if not 0.0 <= confirm_threshold <= 1.0:
        raise VerifierBackendError(f"confirm_threshold must be in [0, 1], got {confirm_threshold}")
    if isinstance(persist_k, bool) or not isinstance(persist_k, int):
        raise VerifierBackendError(f"persist_k must be an int, got {persist_k!r}")
    if persist_k < 1:
        raise VerifierBackendError(f"persist_k must be >= 1, got {persist_k}")
    if hyst_hi is None:
        return
    if not 0.0 <= hyst_hi <= 1.0:
        raise VerifierBackendError(f"hyst_hi must be in [0, 1] or None, got {hyst_hi}")
    if persist_k < 2:
        raise VerifierBackendError(
            "hyst_hi needs persist_k >= 2: with one window the immediate leg and the "
            "sustained leg are the same decision, so the knob would silently do nothing "
            "(hyst:hi|1@lo IS single:lo)"
        )
    if hyst_hi < confirm_threshold:
        raise VerifierBackendError(
            f"hyst_hi ({hyst_hi}) must be >= confirm_threshold ({confirm_threshold}): the "
            "hysteresis pair is hi|k@lo, and a hi below its own lo inverts the rule into "
            "single:hi"
        )


#: The committed name of a lane's graph under final_models/<lane>/; a training run names
#: its graph in report.json (verifier.onnx), and the lane directory keeps the same bytes as
#: model.onnx beside the copied report.
LANE_GRAPH_NAME = "model.onnx"


def load_head(
    run_dir: Path,
    *,
    pre_pad_s: float = DEFAULT_PRE_PAD_S,
    confirm_threshold: float = DEFAULT_CONFIRM_THRESHOLD,
    persist_k: int = DEFAULT_PERSIST_K,
    hyst_hi: float | None = None,
) -> VerifierHead:
    """Build a head from a run directory: report.json plus the graph it names.

    The directory is either a training run, where report.json sits beside the
    verifier.onnx it names, or a lane directory under final_models/, where the same graph
    is committed as model.onnx beside the report. Either way the graph is opened only if
    it hashes to the digest the report records.
    """
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        raise VerifierBackendError(
            f"no report.json under {run_dir}; a verifier head is built from a training "
            "run directory or a final_models lane directory, never from a bare .onnx"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        window = report["config"]["window"]
        task = report["task"]
        weight_class = report["vividly_weight_class"]
        expected_sha = report["onnx"]["sha256"]
        onnx_name = report["onnx"]["file"]
    except KeyError as exc:
        raise VerifierBackendError(f"{report_path} is missing key {exc}") from exc

    onnx_path = run_dir / onnx_name
    if not onnx_path.is_file() and (run_dir / LANE_GRAPH_NAME).is_file():
        onnx_path = run_dir / LANE_GRAPH_NAME
    if not onnx_path.is_file():
        raise VerifierBackendError(f"missing graph {onnx_path}")
    if is_lfs_pointer(onnx_path):
        raise VerifierBackendError(
            f"{onnx_path} is a Git LFS pointer, not the graph; run `git lfs pull` to fetch it"
        )
    actual_sha = sha256_of(onnx_path)
    if actual_sha != expected_sha:
        raise VerifierBackendError(
            f"{onnx_path} sha256 {actual_sha[:8]} does not match report "
            f"{expected_sha[:8]}; refusing to score with unaccounted-for weights"
        )

    temperature = float((report.get("calibration") or {}).get("temperature", 1.0))
    if not temperature > 0:
        raise VerifierBackendError(f"{report_path} calibration.temperature must be > 0")
    validate_decision_policy(float(confirm_threshold), persist_k, hyst_hi)

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return VerifierHead(
        session=session,
        spec=WindowSpec(**window),
        task=str(task),
        weight_class=str(weight_class),
        sha8=actual_sha[:8],
        pre_pad_s=float(pre_pad_s),
        confirm_threshold=float(confirm_threshold),
        temperature=temperature,
        persist_k=persist_k,
        hyst_hi=None if hyst_hi is None else float(hyst_hi),
    )


class OnnxEventVerifier:
    """EventVerifier over one ONNX head per rule id. See the module docstring."""

    verifier_id = "onnx_event_verifier_v1"

    def __init__(self, heads: Mapping[str, VerifierHead]) -> None:
        if not heads:
            raise VerifierBackendError("OnnxEventVerifier needs at least one head")
        self._heads = dict(heads)
        self.verifier_version = "+".join(
            f"{rule_id}:{head.sha8}" for rule_id, head in sorted(self._heads.items())
        )
        # Scene memory for SCENE_CHANNELS heads: track_id -> (first_seen_t,
        # last_seen_t), fed by observe_book each engine tick and retained past the
        # book's own reap so a window can still count a person who just left.
        self._scene_spans: dict[int, tuple[float, float]] = {}
        self._scene_observed = False

    def observe_book(self, book: TrackBook, now: float) -> None:
        """Engine hook: remember every live track's observed span.

        Called by RuleEngine.tick before rules run (the same getattr pattern as a
        rule's evaluate_book). The training extraction built SceneContext from all
        of a video's tracks; the live equivalent is the union of every span this
        verifier has ever seen in the book, pruned after SCENE_RETENTION_S, which
        also bridges the book's 30 s reap: a person who left the frame minutes
        before a firing still counts in the windows they overlapped.
        """
        for track in book.tracks().values():
            if math.isnan(track.first_seen_t):
                continue
            self._scene_spans[track.track_id] = (track.first_seen_t, track.last_seen_t)
        if not math.isnan(now):
            cutoff = now - SCENE_RETENTION_S
            self._scene_spans = {
                tid: span for tid, span in self._scene_spans.items() if span[1] >= cutoff
            }
        self._scene_observed = True

    @classmethod
    def from_config(
        cls, heads_cfg: Mapping[str, Mapping[str, Any]], repo_root: Path
    ) -> OnnxEventVerifier:
        """Build from a pipeline config's verify.heads block:

        verify:
          policy: annotate
          heads:
            fall_then_still_v0:
              run_dir: outputs/verifier/fall_v1
              pre_pad_s: 2.0            # optional
              confirm_threshold: 0.5    # optional, PROVISIONAL
              persist_k: 1              # optional, 1 = one window (the default)
              hyst_hi: null             # optional, needs persist_k >= 2

        Unknown keys raise (HEAD_CONFIG_KEYS), and so does a value that cannot mean what
        it says; the rule id is named in every message, because a pipeline config is
        edited by hand and a silent default here is an uncalibrated run wearing a
        calibrated run's clothes.
        """
        heads: dict[str, VerifierHead] = {}
        for rule_id, head_cfg in heads_cfg.items():
            unknown = sorted(set(head_cfg) - HEAD_CONFIG_KEYS)
            if unknown:
                raise VerifierBackendError(
                    f"verify.heads.{rule_id}: unknown keys {unknown}; known keys are "
                    f"{sorted(HEAD_CONFIG_KEYS)}"
                )
            if "run_dir" not in head_cfg:
                raise VerifierBackendError(f"verify.heads.{rule_id}: no run_dir")
            run_dir = Path(str(head_cfg["run_dir"]))
            if not run_dir.is_absolute():
                run_dir = repo_root / run_dir
            hyst_hi = head_cfg.get("hyst_hi")
            heads[rule_id] = load_head(
                run_dir,
                pre_pad_s=_as_float(
                    head_cfg.get("pre_pad_s", DEFAULT_PRE_PAD_S), rule_id, "pre_pad_s"
                ),
                confirm_threshold=_as_float(
                    head_cfg.get("confirm_threshold", DEFAULT_CONFIRM_THRESHOLD),
                    rule_id,
                    "confirm_threshold",
                ),
                persist_k=_as_int(
                    head_cfg.get("persist_k", DEFAULT_PERSIST_K), rule_id, "persist_k"
                ),
                hyst_hi=None if hyst_hi is None else _as_float(hyst_hi, rule_id, "hyst_hi"),
            )
        return cls(heads)

    # -- protocol -----------------------------------------------------------
    def verify(self, firing: RuleFiring, track: TrackState | None) -> VerifierVerdict:
        head = self._heads.get(firing.rule_id)
        if head is None or track is None:
            return self._verdict("unavailable", None)
        scores: list[float] = []
        for i in range(head.persist_k):
            # i == 0 is the anchor window at t_end, arithmetically untouched so an
            # unconfigured head (persist_k == 1) scores the exact float it always did.
            onset_t = firing.onset_t if i == 0 else firing.onset_t + i * head.spec.stride_s
            x = self._features(head, track, onset_t)
            if x is None:
                # An incomplete pattern is missing evidence, not evidence of absence.
                return self._verdict("unavailable", None)
            scores.append(self._score(head, x))
        if head.hyst_hi is not None and max(scores) >= head.hyst_hi:
            return self._verdict("confirmed", max(scores))
        if min(scores) >= head.confirm_threshold:
            return self._verdict("confirmed", min(scores))
        return self._verdict("rejected", max(scores))

    def _score(self, head: VerifierHead, x: np.ndarray) -> float:
        """One window through the graph: sigmoid(logit / T), clamped to [0, 1]."""
        logits = head.session.run(None, {"windows": x[None, :, :].astype(np.float32)})[0]
        logit = float(np.reshape(logits, (-1,))[0]) / head.temperature
        score = float(1.0 / (1.0 + math.exp(-logit)))
        return min(1.0, max(0.0, score))

    def _verdict(self, verdict: str, confidence: float | None) -> VerifierVerdict:
        return VerifierVerdict(
            verdict=verdict,
            verifier_id=self.verifier_id,
            verifier_version=self.verifier_version,
            confidence=confidence,
        )

    def _features(self, head: VerifierHead, track: TrackState, onset_t: float) -> np.ndarray | None:
        """Rebuild extraction-schema rows from the rings and featurize one window.

        Row values are rounded exactly as the extraction writer rounds them (t to 4
        places, geometry to 2, kpc to 4) so inference features match training features
        to the digit, not just in spirit.
        """
        win_start = onset_t - head.pre_pad_s
        t_end = win_start + head.spec.window_s
        span = track.last_seen_t - win_start + 1.0
        if not span > 0:
            return None
        positions = track.positions(span)
        boxes = track.boxes(span)
        if not positions or len(positions) != len(boxes):
            # The rings are appended in lockstep; a mismatch means this state predates
            # the box ring (or something is deeply wrong). Abstain, never guess.
            return None
        angles = dict(track.posture_angles(span))
        rows = []
        for (t, cx, cy), (_, w, h, kpc) in zip(positions, boxes, strict=True):
            angle = angles.get(t)
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
        tracks = tracks_from_rows(rows)
        if not tracks:
            return None
        scene = None
        if head.spec.scene_channels:
            if not self._scene_observed:
                raise VerifierBackendError(
                    "this head was trained with scene channels but the engine never "
                    "called observe_book; wire the verifier through RuleEngine (which "
                    "hands over the TrackBook each tick) instead of calling it bare"
                )
            spans = np.array(list(self._scene_spans.values()), dtype=np.float64).reshape(-1, 2)
            scene = SceneContext(spans=spans, starts=np.sort(spans[:, 0]))
        return featurize_window(tracks[0], t_end, head.spec, scene=scene)
