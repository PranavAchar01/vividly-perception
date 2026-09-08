"""Job engine: one uploaded video through the real pipeline with a chosen model pair.

The pipeline runner (scripts/run_pipeline.py) is loaded dynamically through
``importlib.util.spec_from_file_location``, then subclassed per job: ``_rules_step`` captures
firings for the flash/alert feed and ``_render`` draws the harness overlay, appends the
frame to the annotated H.264 writer, and publishes a throttled JPEG for the websocket
viewers. One worker thread processes jobs strictly serially, because the detector costs a
measured 76.3 ms per frame end to end at 640x640 on the processor (eyes graph 40262049,
research/eyes-instrument-grading.md), so two concurrent jobs would halve the throughput of
each and corrupt every latency number shown.

Honesty rails carried here: the device is hardcoded to cpu; the rules profile travels
into job.json unchanged; alerts are surfaced as candidate events with the rule ids
unchanged; and demo material lives only under the gitignored outputs/ trees, where
media.ensure_demo_dirs writes the do-not-train READMEs.
"""

from __future__ import annotations

import importlib.util
import json
import math
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from vividly.dashboard.app import _finite_tree, encode_jpeg_b64
from vividly.demo import catalog as demo_catalog
from vividly.demo import live as demo_live
from vividly.demo import media, overlay
from vividly.events import EventStore
from vividly.media.ffmpeg import probe_video, run_argv
from vividly.schemas.events import utc_now_iso

# Product naming for the three event models. This table is the naming of record: the display
# name leads in the interface, and the rule id itself stays in every alert record, rail and
# store.
EVENT_MODEL_NAMES = {
    "fall_then_still_v0": "Collapse to floor",
    "seizure_candidate_v0": "Rhythmic convulsion",
    "egress_without_discharge_v0": "Egress without discharge",
}

DISCLAIMER = (
    "Engineering demonstration, not a medical device. Rule firings are candidate events "
    "under provisional thresholds, not diagnoses."
)

# The two trained event-verifier heads, attached to every demo run from the lane
# directories under final_models/, which hold the models of record
# (final_models/fall/README.md and final_models/seizure/README.md). The backend
# (vividly.models.event_verifier) reads each lane's report.json for the window geometry
# and the calibration temperature and refuses model.onnx unless it hashes to the digest
# the report records. Both are prototype weights: internal demonstration only, never
# quoted as clinical detection performance. A clone without `git lfs pull` has pointer
# files in place of the graphs, and the heads are then left off rather than guessed.
VERIFIER_RUN_DIRS: dict[str, str] = {
    # fall_v3, checkpoint 3d2d17ee, graph 67954c04: all-cameras window AUC 0.79276 on the
    # multicam exam against fall_v2pose b66c3a0a at 0.78474.
    "fall_then_still_v0": "final_models/fall",
    # convulsion_v2_real, checkpoint 715d172c, graph 07c399be: window AUC 0.69841 on the
    # wusahzu testsplit against convulsion_v2pose 614b92fa at 0.53728.
    "seizure_candidate_v0": "final_models/seizure",
}

# Per-head confirm_threshold for the demo: the operating points the two lane pages record.
# The seizure page: 0.5 catches 12 of 13 events with 72.4 percent of quiet footage in
# alarm; 0.8 catches 8 of 13 at 33.2 percent; 0.9 leaves 8.1 percent in alarm. The fall
# page: at 0.95 the model of record keeps 111 of 200 all-camera catches with chute24 at
# 7.3 episodes per hour and 2.8 percent in alarm, and its quiet range on the UR activities
# of daily living recordings starts at 0.95, not 0.9. Every threshold in this repository
# is provisional until the calibration protocol in docs/CALIBRATION.md has its
# preconditions met; both neighbours are a one-line change here.
VERIFIER_CONFIRM_THRESHOLDS: dict[str, float] = {
    "seizure_candidate_v0": 0.8,
    "fall_then_still_v0": 0.95,
}

# Partial mappings for RuleConfig.from_mapping (unknown keys raise; unspecified keys keep
# the clinical defaults). "production" is the untouched clinical provisional set: on a
# short clip it usually fires nothing, by design. "demo" shortens the windows so a staged
# walkthrough can show the cascade at all. It is a demo calibration and never the clinical
# operating point, and the chosen profile is stored unchanged in job.json.
RULES_PROFILES: dict[str, dict[str, Any]] = {
    "production": {"rules_config": "configs/pipeline/rules.yaml"},
    "demo": {
        "rules": {
            "refire_after_s": 30.0,
            "eval_every_s": 0.5,
            "collapse_search_s": 60.0,  # >= fall_max_s(3) + down_still_min_s(5) + down_stale_s(5)
            "down_still_min_s": 5.0,
            "down_min_samples": 8,
            "convulse_window_s": 8.0,
            "convulse_min_s": 4.0,
            "convulse_min_samples": 12,
            # A demo calibration for egress_without_discharge_v0, never the clinical
            # operating point: the shipped clock (absent 600 s, presence 120 s,
            # configs/pipeline/rules.yaml) cannot complete inside a short staged clip.
            # At 20 s and 10 s a person who sits, walks out of the door and stays gone
            # fires while the clip is still rolling. Harmless when a job has no zones:
            # without an `egress:` block no monitor is built and these two settings are
            # read by nothing.
            "egress_absent_min_s": 20.0,
            "egress_min_presence_s": 10.0,
        }
    },
    # alerts recorded but not flashed/panelled; thresholds are the clinical set
    "hide": {"rules_config": "configs/pipeline/rules.yaml"},
}

# A demo calibration for jobs that carry exit zones, mirroring what
# configs/pipeline/egress_demo.yaml does to its fixture: the egress head only notices a
# departure when the TrackBook reaps the track, and the shipped 30 s reap placeholder
# would eat most of a short clip's tail before the 20 s absence clock even started. This
# is not a tracking recommendation: 8 s costs identity continuity across occlusions
# longer than 8 s, a trade a staged demo clip can afford and a clinical run cannot.
EGRESS_DEMO_REAP_AFTER_S = 8.0

# Exit-zone shape limits for the drawing tool. The name charset is strict on purpose:
# zone names travel into innerHTML on the job page, onto the MP4 overlay, and into
# alert records; [A-Za-z0-9 ._-] cannot break any of the three.
MAX_ZONES = 8
ZONE_NAME_RE = re.compile(r"^[A-Za-z0-9 ._-]{1,40}$")


def validate_zones(zones: Any, frame_w: int, frame_h: int) -> list[dict[str, Any]]:
    """Validate job-setup exit zones against the preview frame's dimensions.

    ``zones`` is the request payload: ``[{"name": str, "rect": [x1, y1, x2, y2]}, ...]``
    with rect corners in normalized-frame pixels (the space media.ensure_preview grabs
    its frame in). Returns the canonical list that is stored unchanged in job.json.
    Raises ValueError: inverted or out-of-range rectangles are refused here, before a
    job exists, because a zone the pipeline silently cannot contain a person in would
    look like egress armed and be a door nobody was watching.
    """
    if not isinstance(zones, list):
        raise ValueError("zones must be a list of {name, rect} entries")
    if len(zones) > MAX_ZONES:
        raise ValueError(f"at most {MAX_ZONES} exit zones, got {len(zones)}")
    if not frame_w > 0 or not frame_h > 0:
        raise ValueError(f"frame dimensions must be positive, got {frame_w}x{frame_h}")
    out: list[dict[str, Any]] = []
    names: set[str] = set()
    for i, entry in enumerate(zones):
        if not isinstance(entry, dict):
            raise ValueError(f"zones[{i}] must be a mapping with name and rect")
        extra = sorted(set(entry) - {"name", "rect"})
        if extra:
            raise ValueError(f"zones[{i}]: unknown keys {extra}")
        name = str(entry.get("name") or "").strip()
        if not ZONE_NAME_RE.fullmatch(name):
            raise ValueError(
                f"zones[{i}]: name must be 1-40 characters of letters, digits, spaces, "
                "dots, hyphens or underscores"
            )
        if name in names:
            raise ValueError(f"zones[{i}]: duplicate zone name {name!r}")
        names.add(name)
        rect = entry.get("rect")
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            raise ValueError(f"zones[{i}]: rect must be [x1, y1, x2, y2]")
        try:
            x1, y1, x2, y2 = (float(v) for v in rect)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"zones[{i}]: rect values must be numbers") from exc
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError(f"zones[{i}]: rect values must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"zones[{i}]: rectangle is inverted or empty ({rect})")
        if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
            raise ValueError(
                f"zones[{i}]: rectangle {rect} lies outside the {frame_w}x{frame_h} frame"
            )
        out.append({"name": name, "rect": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    return out


def zone_entries(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The stored rect zones as ``{name, polygon}`` entries.

    The one rect-to-polygon conversion, shared by the emitted ``egress:`` config block
    (EgressZones.from_mapping requires polygons) and the overlay's zone outlines, so the
    region the pipeline monitors and the region the viewer sees cannot diverge.
    """
    out = []
    for zone in zones:
        x1, y1, x2, y2 = (float(v) for v in zone["rect"])
        out.append(
            {
                "name": str(zone["name"]),
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            }
        )
    return out


_RUNNER_MODULE = "vividly_run_pipeline_script_demo"
STREAM_MIN_INTERVAL_S = 0.15  # ~6 Hz max to the browser while processing
JPEG_QUALITY = 80


def _load_runner(repo_root: Path) -> Any:
    """scripts/ is not a package; load run_pipeline.py by file path through importlib."""
    if _RUNNER_MODULE in sys.modules:
        return sys.modules[_RUNNER_MODULE]
    path = Path(repo_root) / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location(_RUNNER_MODULE, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RUNNER_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _entry_view(entry: dict[str, Any]) -> dict[str, Any]:
    """The subset of a catalog entry that travels in snapshots and job.json.

    `backend` is carried because the run config is built from this view: the eyes lane
    runs under rtdetr_onnx and the joints lane under rtmlib, and a job that recorded which
    weight it used without recording how it was run would not be reproducible.
    """
    return {
        "id": entry["id"],
        "display_name": entry["display_name"],
        "sha8": entry["sha8"],
        "backend": entry["backend"],
        "badge": entry["badge"],
        "weight_class": entry.get("weight_class"),
        "card": entry.get("card"),
        "input_size": list(entry["input_size"]),
    }


def build_run_cfg(
    *,
    job_id: str,
    runs_dir: Path,
    input_mp4: Path,
    detector: dict[str, Any],
    pose: dict[str, Any],
    score_thr: float,
    rules_profile: str,
    repo_root: Path,
    fps: float,
    zones: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The scripts/run_pipeline.py config for one demo job.

    Each model's backend comes from its
    catalog entry, never from a constant here: the eyes lane is an RT-DETRv4 graph that
    needs its own preprocessing and postprocessing (backend rtdetr_onnx, input 640x640,
    plain resize) and the joints lane is an RTMPose graph (backend rtmlib, input 192x256),
    and running one under the other's backend returns plausible numbers that are wrong.
    Both tracker gates sit one tenth above the detector score threshold, the convention
    every extraction configuration in this repository uses (score threshold 0.3 with both
    gates at 0.4, configs/events/extract_multicam_allcams_v2c.yaml). The tracker backend is
    the frozen baseline, bytetrack, at the library defaults for lost_track_buffer (30) and
    minimum_iou_threshold (0.1); with those gates that is the operating point every
    extraction configuration here runs, and so the one the verifier track trees were built
    under. It is not the shipped replay tracker: configs/pipeline/replay_demo.yaml runs
    botsort at lost_track_buffer 1800, adopted on the tracking defect
    src/vividly/models/tracker.py records. Nothing in this repository records why the demo
    keeps the extraction point, so the difference is stated here as a fact and not as a
    rationale. Device is hardcoded cpu: RtmlibPose has no selector for the Metal execution
    provider and the RT-DETRv4 backend refuses anything but cpu and cuda. ``zones``
    (validate_zones output) arms the egress head; empty or None leaves the config byte for
    byte what it was before exit zones existed.
    """
    gate = round(min(0.9, score_thr + 0.1), 2)
    cfg: dict[str, Any] = {
        "schema_version": 1,
        "run_name": job_id,
        "headless": True,
        "source": {"path": str(input_mp4), "rate": None},  # unpaced: process flat-out
        "detector": {
            "backend": str(detector["backend"]),
            "onnx": detector["onnx_path"],  # absolute (CWD-relative paths become URLs)
            "input_size": list(detector["input_size"]),
            "device": "cpu",
            "score_thr": float(score_thr),
            "name": f"{detector['display_name']} ({detector['sha8']})",
        },
        "pose": {
            "backend": str(pose["backend"]),
            "onnx": pose["onnx_path"],
            "input_size": list(pose["input_size"]),
            "device": "cpu",
            "name": f"{pose['display_name']} ({pose['sha8']})",
        },
        "tracker": {
            "backend": "bytetrack",
            "track_activation_threshold": gate,
            "high_conf_det_threshold": gate,
            "lost_track_buffer": 30,
            "minimum_iou_threshold": 0.1,
            "minimum_consecutive_frames": 2,
            "frame_rate": fps,
        },
        "tracks": {"fps": fps, "ring_window_s": 300.0},
        "output": {"dir": str(runs_dir)},
        "dashboard": {"jpeg_quality": JPEG_QUALITY},
    }
    profile = RULES_PROFILES[rules_profile]
    if "rules_config" in profile:
        cfg["rules_config"] = str(Path(repo_root) / profile["rules_config"])
    else:
        cfg["rules"] = dict(profile["rules"])
    if zones:
        # The egress head (egress_without_discharge_v0): zones drawn on the normalized
        # preview frame, in exactly the pixel space `source` above decodes to. The rect
        # becomes the 4-vertex polygon EgressZones.from_mapping requires.
        cfg["egress"] = {"enabled": True, "zones": zone_entries(zones)}
        if rules_profile == "demo":
            # Demo calibration (see EGRESS_DEMO_REAP_AFTER_S): a departure is only
            # noticed at reap time, so the demo shortens the reap the way
            # configs/pipeline/egress_demo.yaml does. Clinical profiles keep the
            # shipped TrackConfig default and simply never fire on short clips.
            cfg["tracks"]["reap_after_s"] = EGRESS_DEMO_REAP_AFTER_S
    # Verifier heads attach from the lane directories (see VERIFIER_RUN_DIRS). Policy
    # "annotate": verdicts are stamped onto alerts; a rejected candidate is suppressed by
    # the engine and counted, which is the false-positive stance made visible. A lane
    # directory keeps its graph as model.onnx; a training run names it verifier.onnx.
    heads: dict[str, dict[str, object]] = {}
    for rule_id, rel in VERIFIER_RUN_DIRS.items():
        run_dir = Path(repo_root) / rel
        graph_present = any((run_dir / name).is_file() for name in ("model.onnx", "verifier.onnx"))
        if (run_dir / "report.json").is_file() and graph_present:
            heads[rule_id] = {"run_dir": str(run_dir)}
            threshold = VERIFIER_CONFIRM_THRESHOLDS.get(rule_id)
            if threshold is not None:
                heads[rule_id]["confirm_threshold"] = threshold
    if heads:
        cfg["verify"] = {"policy": "annotate", "heads": heads}
    return cfg


class JobState:
    """One demo run. The worker thread writes and the event loop reads, all under _lock."""

    def __init__(
        self,
        job_id: str,
        job_dir: Path,
        *,
        source_label: str,
        detector: dict[str, Any],
        pose: dict[str, Any],
        score_thr: float,
        rules_profile: str,
        max_s: float | None,
        live: bool = False,
        zones: list[dict[str, Any]] | None = None,
    ) -> None:
        self.job_id = job_id
        self.job_dir = Path(job_dir)
        self._lock = threading.Lock()
        self.created_utc = utc_now_iso()
        self.status = "queued"  # queued|normalizing|running|done|failed|cancelled|interrupted
        self.error: str | None = None
        self.source_label = source_label
        self.detector = detector  # _entry_view shapes
        self.pose = pose
        self.score_thr = float(score_thr)
        self.rules_profile = rules_profile
        self.max_s = max_s
        self.zones = [dict(z) for z in zones or []]  # validate_zones shapes, unchanged
        self.live = live  # camera session: watch only, nothing recorded
        self.video: dict[str, Any] | None = None  # probe of the normalized input
        self.n_total: int | None = None
        self.frames_done = 0
        self.t_source = 0.0
        self.tracks_live = 0
        self.frame_b64: str | None = None
        self.alerts: list[dict[str, Any]] = []
        self.latency_p50: dict[str, Any] | None = None
        self.latency_p95: dict[str, Any] | None = None
        self.annotated_ready = False
        self.thumb_done = False
        self.cancel_requested = False
        self.compare_group: str | None = None
        self.compare_peer: str | None = None
        self.started_monotonic: float | None = None
        self.pipeline: Any = None  # set while running; request_stop() is thread-safe

    # -- worker-side updates ------------------------------------------------
    def set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            if error is not None:
                self.error = error

    def set_video(self, info: dict[str, Any], n_total: int | None) -> None:
        with self._lock:
            self.video = info
            self.n_total = n_total

    def mark_frame(self, frames_done: int, t_source: float, tracks_live: int) -> None:
        with self._lock:
            self.frames_done = frames_done
            self.t_source = t_source
            self.tracks_live = tracks_live

    def set_frame_b64(self, b64: str | None) -> None:
        with self._lock:
            self.frame_b64 = b64

    def add_alerts(self, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            self.alerts.extend(rows)

    def replace_alerts(self, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            self.alerts = list(rows)

    def set_latency(self, summary: dict[str, Any] | None) -> None:
        if not summary:
            return
        with self._lock:
            self.latency_p50 = summary.get("p50_ms")
            self.latency_p95 = summary.get("p95_ms")

    # -- shared views -------------------------------------------------------
    def snapshot(self, queue_position: int | None = None) -> dict[str, Any]:
        with self._lock:
            done = self.frames_done
            total = self.n_total
            progress = (done / total) if (total or 0) > 0 else None
            proc_fps = None
            eta_s = None
            if self.started_monotonic is not None and done > 0:
                elapsed = time.monotonic() - self.started_monotonic
                if elapsed > 0:
                    proc_fps = done / elapsed
                    if total and proc_fps > 0:
                        eta_s = max(0.0, (total - done) / proc_fps)
            snap = {
                "job_id": self.job_id,
                "status": self.status,
                "error": self.error,
                "created_utc": self.created_utc,
                "source_label": self.source_label,
                "video": self.video,
                "models": {"detector": self.detector, "pose": self.pose},
                "score_thr": self.score_thr,
                "rules_profile": self.rules_profile,
                "max_s": self.max_s,
                "zones": [dict(z) for z in self.zones],
                "live": self.live,
                "frame_b64": self.frame_b64 if self.status == "running" else None,
                "frame_index": done,
                "n_total": total,
                "t_source": self.t_source,
                "progress": progress,
                "proc_fps": proc_fps,
                "eta_s": eta_s,
                "queue_position": queue_position,
                "tracks_live": self.tracks_live,
                "alerts": list(self.alerts[-50:]),
                "latency_p50_ms": self.latency_p50,
                "latency_p95_ms": self.latency_p95,
                "annotated_ready": self.annotated_ready,
                "compare_group": self.compare_group,
                "compare_peer": self.compare_peer,
                "disclaimer": DISCLAIMER,
            }
        return _finite_tree(snap)

    def persist(self) -> None:
        snap = self.snapshot()
        snap.pop("frame_b64", None)
        snap.pop("queue_position", None)
        (self.job_dir / "job.json").write_text(json.dumps(snap, indent=2) + "\n")

    @classmethod
    def from_json(cls, job_dir: Path) -> JobState | None:
        try:
            row = json.loads((job_dir / "job.json").read_text())
        except (OSError, ValueError):
            return None
        models = row.get("models") or {}
        job = cls(
            row.get("job_id") or job_dir.name,
            job_dir,
            source_label=row.get("source_label") or "?",
            detector=models.get("detector") or {},
            pose=models.get("pose") or {},
            score_thr=float(row.get("score_thr") or 0.4),
            rules_profile=row.get("rules_profile") or "demo",
            max_s=row.get("max_s"),
            live=bool(row.get("live")),
            zones=row.get("zones") or [],
        )
        job.created_utc = row.get("created_utc") or job.created_utc
        status = row.get("status") or "interrupted"
        # a job.json frozen mid-flight means the process died under it
        job.status = "interrupted" if status in {"queued", "normalizing", "running"} else status
        job.error = row.get("error")
        job.video = row.get("video")
        job.n_total = row.get("n_total")
        job.frames_done = int(row.get("frame_index") or 0)
        job.alerts = list(row.get("alerts") or [])
        job.latency_p50 = row.get("latency_p50_ms")
        job.latency_p95 = row.get("latency_p95_ms")
        job.annotated_ready = (job_dir / "annotated.mp4").is_file()
        job.thumb_done = (job_dir / "thumb.jpg").is_file()
        job.compare_group = row.get("compare_group")
        job.compare_peer = row.get("compare_peer")
        return job


class DemoFrameHook:
    """Per-job glue between the pipeline tick and overlay/writer/stream. Worker thread
    only, so it needs no lock of its own; it pushes into JobState which has one."""

    def __init__(
        self, job: JobState, writer: media.AnnotatedWriter | None, rules_label: str
    ) -> None:
        """writer=None is the live-camera mode: frames are streamed, never written."""
        self.job = job
        self.writer = writer
        self.rules_label = rules_label
        self.show_alerts = job.rules_profile != "hide"
        self._flashes: dict[int, float] = {}  # track_id -> detected_t (source clock)
        self._banners: list[tuple[float, str]] = []  # (expires_t, text)
        self._last_pub = 0.0
        self._hud_left = self._models_line(job)
        # exit-zone outlines for the overlay: same polygons the egress: block carries
        self._zones = zone_entries(job.zones) if job.zones else []

    @staticmethod
    def _models_line(job: JobState) -> str:
        det, pose = job.detector, job.pose
        parts = [f"DET {det['display_name']} {det['sha8']}"]
        if det.get("weight_class") == "prototype":
            parts.append("[PROTOTYPE - INTERNAL DEMO]")
        parts.append(f"POSE {pose['display_name']} {pose['sha8']}")
        if pose.get("weight_class") == "prototype":
            parts.append("[PROTOTYPE]")
        return "  ".join(parts)

    def on_firings(self, firings: list[Any], t: float) -> None:
        rows = []
        for f in firings:
            if self.show_alerts:
                self._flashes[int(f.track_id)] = float(f.detected_t)
                self._banners.append(
                    (
                        float(f.detected_t) + overlay.FLASH_S,
                        f"CANDIDATE EVENT  {EVENT_MODEL_NAMES.get(f.rule_id, f.rule_id)}"
                        f" - track {f.track_id} - onset {float(f.onset_t):.1f} s",
                    )
                )
            rows.append(
                {
                    "rule_id": f.rule_id,
                    "rule_version": f.rule_version,
                    "track_id": int(f.track_id),
                    "onset_ts": float(f.onset_t),
                    "detected_ts": float(f.detected_t),
                    "details": {k: v for k, v in dict(f.details).items()},
                }
            )
        if rows:
            self.job.add_alerts(rows)

    def scene(self, pipeline: Any) -> dict[str, Any]:
        now = float(pipeline.book.now)
        tracks = []
        for tid, ts in sorted(pipeline.book.tracks().items()):
            obs = ts.last_observation
            if obs is None:
                continue
            flash = None
            fired = self._flashes.get(tid)
            if fired is not None:
                dt = now - fired
                if 0.0 <= dt <= overlay.FLASH_S:
                    flash = dt
                elif dt > overlay.FLASH_S:
                    del self._flashes[tid]
            half_w, half_h = obs.w / 2.0, obs.h / 2.0
            tracks.append(
                {
                    "track_id": tid,
                    "xyxy": (obs.cx - half_w, obs.cy - half_h, obs.cx + half_w, obs.cy + half_h),
                    "keypoints": obs.keypoints,
                    "is_staff": ts.is_staff,
                    "flash": flash,
                }
            )
        self._banners = [b for b in self._banners if b[0] >= now]
        return {
            "tracks": tracks,
            "banners": [text for _, text in self._banners],
            "zones": self._zones,
        }

    def on_frame(self, pipeline: Any, image: Any) -> None:
        job = self.job
        scene = self.scene(pipeline)
        done = (self.writer.frames_written if self.writer else job.frames_done) + 1
        total = job.n_total
        now = float(pipeline.book.now)
        proc = ""
        if job.started_monotonic is not None:
            elapsed = time.monotonic() - job.started_monotonic
            if elapsed > 0.5:
                proc = f" - {done / elapsed:.1f} fps"
        if job.live:
            right = f"LIVE {int(now) // 60}:{int(now) % 60:02d}{proc} - {self.rules_label}"
            sub = "engineering demo - not a medical device - live, not recorded"
        else:
            right = f"frame {done}/{total or '?'}{proc} - {self.rules_label}"
            sub = "engineering demo - not a medical device"
        hud = {"left": self._hud_left, "right": right, "sub": sub}
        overlay.draw_frame(image, scene, hud)
        if self.writer is not None:
            self.writer.write(image)
        job.mark_frame(done, now, len(scene["tracks"]))

        # a live session persists nothing at all from the camera, thumbnails included
        if (
            not job.live
            and not job.thumb_done
            and total
            and done >= max(1, math.ceil(total * 0.25))
        ):
            import cv2  # heavy: function-local by package rule

            cv2.imwrite(str(job.job_dir / "thumb.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            job.thumb_done = True

        now_mono = time.monotonic()
        if now_mono - self._last_pub >= STREAM_MIN_INTERVAL_S:
            self._last_pub = now_mono
            job.set_frame_b64(encode_jpeg_b64(image, JPEG_QUALITY))


def make_pipeline(
    runner: Any,
    run_cfg: dict[str, Any],
    repo_root: Path,
    hook: DemoFrameHook,
    live_source: Any = None,
):
    """Subclass the dynamically-loaded Pipeline with the demo seams attached.

    live_source swaps the footage source for a LiveSource; the cfg still names a real
    clip path so construction-time checks pass, but no frame is ever read from it.
    """

    class DemoPipeline(runner.Pipeline):  # type: ignore[misc,name-defined]
        def _rules_step(self, observations, t):
            firings = super()._rules_step(observations, t)
            if firings:
                hook.on_firings(firings, t)
            return firings

        def _render(self, image):
            hook.on_frame(self, image)
            return None  # the demo owns encoding; base _publish gets frame_b64=None

        def _footage_source(self):
            return live_source if live_source is not None else super()._footage_source()

    return DemoPipeline(run_cfg, repo_root)


class JobManager:
    """Owns the job registry, the single worker thread, and the demo directories."""

    def __init__(self, repo_root: Path, outputs_root: Path | None = None) -> None:
        """repo_root locates the runner, the configs and the model record; outputs_root
        (tests) moves just the demo_uploads and demo_runs trees somewhere disposable."""
        self.repo_root = Path(repo_root)
        self.dirs = media.ensure_demo_dirs(Path(outputs_root) if outputs_root else self.repo_root)
        self.sample_clip = self.repo_root / "tests" / "fixtures" / "sample_clip.mp4"
        self._jobs: dict[str, JobState] = {}
        self._order: list[str] = []
        self._pending: list[str] = []
        self._live_sources: dict[str, demo_live.LiveSource] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._load_persisted()
        self._worker = threading.Thread(
            target=self._work_loop, name="vividly-demo-worker", daemon=True
        )
        self._worker.start()

    def catalog(self) -> dict[str, Any]:
        """The picker catalog: the final_models/ lanes plus dropped-in .onnx uploads."""
        return demo_catalog.scan_catalog(self.repo_root, self.dirs["models"])

    # -- registry -----------------------------------------------------------
    def _load_persisted(self) -> None:
        for job_dir in sorted(self.dirs["runs"].iterdir() if self.dirs["runs"].is_dir() else []):
            if not job_dir.is_dir():
                continue
            job = JobState.from_json(job_dir)
            if job is not None:
                self._jobs[job.job_id] = job
                self._order.append(job.job_id)

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job_id: str) -> int | None:
        with self._lock:
            if job_id in self._pending:
                return self._pending.index(job_id)
        return None

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            ordered = [self._jobs[j] for j in reversed(self._order)]
            pending = list(self._pending)
        out = []
        for job in ordered:
            pos = pending.index(job.job_id) if job.job_id in pending else None
            snap = job.snapshot(queue_position=pos)
            snap.pop("frame_b64", None)  # the list view never needs frames
            out.append(snap)
        return out

    # -- job creation -------------------------------------------------------
    def create_job(
        self,
        *,
        source_path: Path,
        source_label: str,
        move_source: bool,
        detector_id: str,
        pose_id: str,
        score_thr: float = 0.4,
        rules_profile: str = "demo",
        max_s: float | None = None,
        zones: list[dict[str, Any]] | None = None,
    ) -> JobState:
        if rules_profile not in RULES_PROFILES:
            raise ValueError(f"unknown rules profile {rules_profile!r}")
        if not 0.05 <= float(score_thr) <= 0.9:
            raise ValueError("score_thr must be within [0.05, 0.9]")
        if max_s is not None and not 1.0 <= float(max_s) <= 3600.0:
            raise ValueError("max_s must be within [1, 3600] seconds")
        zones = [dict(z) for z in zones or []]
        # bounds were checked against the preview frame at the API layer (validate_zones);
        # this re-derives the polygons so a malformed direct call fails here, not in the
        # worker thread forty seconds into a run
        try:
            zone_entries(zones)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed zones: {exc}") from exc
        source_path = Path(source_path)
        if not source_path.is_file():
            raise ValueError(f"source video not found: {source_path}")

        cat = self.catalog()
        det = demo_catalog.find_entry(cat, detector_id)
        pose = demo_catalog.find_entry(cat, pose_id)
        if det is None or det["kind"] != "detector":
            raise ValueError(f"unknown detector {detector_id!r}")
        if pose is None or pose["kind"] != "pose":
            raise ValueError(f"unknown pose model {pose_id!r}")
        for entry in (det, pose):
            if entry.get("disabled"):
                raise ValueError(f"{entry['id']} is unavailable: {entry['disabled']}")

        job_id = uuid.uuid4().hex[:8]
        job_dir = self.dirs["runs"] / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        original = job_dir / f"original{source_path.suffix.lower()}"
        if move_source:
            shutil.move(str(source_path), original)
        else:
            shutil.copy2(source_path, original)

        job = JobState(
            job_id,
            job_dir,
            source_label=source_label,
            detector=_entry_view(det),
            pose=_entry_view(pose),
            score_thr=float(score_thr),
            rules_profile=rules_profile,
            max_s=float(max_s) if max_s is not None else None,
            zones=zones,
        )
        job.detector["onnx_path"] = det["onnx_path"]  # worker needs the abs path
        job.pose["onnx_path"] = pose["onnx_path"]
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._pending.append(job_id)
        job.persist()
        self._queue.put(job_id)
        return job

    def create_live_job(
        self,
        *,
        detector_id: str,
        pose_id: str,
        score_thr: float = 0.4,
        rules_profile: str = "demo",
        max_s: float | None = None,
    ) -> JobState:
        """A camera session: same models and rules, no input file, nothing recorded.

        The LiveSource is registered at creation so the frames websocket can start
        feeding while the job is still queued (its 1-deep mailbox just keeps the
        freshest frame until the worker picks the job up).
        """
        if rules_profile not in RULES_PROFILES:
            raise ValueError(f"unknown rules profile {rules_profile!r}")
        if not 0.05 <= float(score_thr) <= 0.9:
            raise ValueError("score_thr must be within [0.05, 0.9]")
        if max_s is not None and not 1.0 <= float(max_s) <= 3600.0:
            raise ValueError("max_s must be within [1, 3600] seconds")
        cat = self.catalog()
        det = demo_catalog.find_entry(cat, detector_id)
        pose = demo_catalog.find_entry(cat, pose_id)
        if det is None or det["kind"] != "detector":
            raise ValueError(f"unknown detector {detector_id!r}")
        if pose is None or pose["kind"] != "pose":
            raise ValueError(f"unknown pose model {pose_id!r}")
        for entry in (det, pose):
            if entry.get("disabled"):
                raise ValueError(f"{entry['id']} is unavailable: {entry['disabled']}")

        job_id = uuid.uuid4().hex[:8]
        job_dir = self.dirs["runs"] / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        cap_s = float(max_s) if max_s is not None else demo_live.LIVE_MAX_S_DEFAULT
        job = JobState(
            job_id,
            job_dir,
            source_label="live camera",
            detector=_entry_view(det),
            pose=_entry_view(pose),
            score_thr=float(score_thr),
            rules_profile=rules_profile,
            max_s=cap_s,
            live=True,
        )
        job.detector["onnx_path"] = det["onnx_path"]
        job.pose["onnx_path"] = pose["onnx_path"]
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._pending.append(job_id)
            self._live_sources[job_id] = demo_live.LiveSource(max_s=cap_s)
        job.persist()
        self._queue.put(job_id)
        return job

    def push_live_frame(self, job_id: str, data: bytes) -> bool:
        """Frames-websocket entry point. False tells the socket to close."""
        with self._lock:
            src = self._live_sources.get(job_id)
        if src is None:
            return False
        return src.push(data)

    def stop_live(self, job_id: str) -> bool:
        """End a live session cleanly: a stopped running session finalizes as done,
        because stopping is its normal ending, and a stopped queued one becomes
        cancelled."""
        job = self.get(job_id)
        if job is None or not job.live:
            return False
        with self._lock:
            src = self._live_sources.get(job_id)
        if src is not None:
            src.stop()
        with self._lock:
            if job_id in self._pending:
                self._pending.remove(job_id)
                job.set_status("cancelled")
                job.persist()
                return True
        return job.status in {"queued", "running"}

    def link_compare(self, a_id: str, b_id: str) -> None:
        """Mark two jobs as one side-by-side comparison (same input, two detectors)."""
        a, b = self.get(a_id), self.get(b_id)
        if a is None or b is None:
            return
        group = uuid.uuid4().hex[:8]
        a.compare_group = b.compare_group = group
        a.compare_peer, b.compare_peer = b_id, a_id
        a.persist()
        b.persist()

    # -- lifecycle ----------------------------------------------------------
    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        with self._lock:
            if job_id in self._pending:
                self._pending.remove(job_id)
                job.cancel_requested = True
                job.set_status("cancelled")
                job.persist()
                return True
        if job.status in {"normalizing", "running"}:
            job.cancel_requested = True
            if job.pipeline is not None:
                job.pipeline.request_stop()
            return True
        return False

    def delete(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in {"normalizing", "running"}:
            return False
        with self._lock:
            self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)
            if job_id in self._pending:
                self._pending.remove(job_id)
        shutil.rmtree(job.job_dir, ignore_errors=True)
        return True

    def shutdown(self) -> None:
        with self._lock:
            current = [j for j in self._jobs.values() if j.status in {"normalizing", "running"}]
            for src in self._live_sources.values():
                src.stop()
        for job in current:
            job.cancel_requested = True
            if job.pipeline is not None:
                job.pipeline.request_stop()
        self._queue.put(None)
        self._worker.join(timeout=10.0)

    # -- the worker ---------------------------------------------------------
    def _work_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            job = self.get(job_id)
            if job is None or job.status == "cancelled":
                continue
            with self._lock:
                if job_id in self._pending:
                    self._pending.remove(job_id)
            try:
                self._run_job(job)
            except Exception as exc:  # the job page is where failures surface
                detail = f"{exc}"
                if not isinstance(exc, (RuntimeError, ValueError)):
                    detail = f"{exc}\n{traceback.format_exc()[-1500:]}"
                job.set_status("failed", error=detail)
            finally:
                job.pipeline = None
                job.persist()

    def _run_job(self, job: JobState) -> None:
        if job.live:
            self._run_live(job)
            return
        job.set_status("normalizing")
        original = next(job.job_dir.glob("original.*"), None)
        if original is None:
            raise RuntimeError("original upload missing from the job directory")
        input_mp4 = job.job_dir / "input.mp4"
        run_argv(media.normalize_argv(original, input_mp4))
        info = probe_video(input_mp4)
        fps = float(info.get("fps") or media.NORMALIZE_FPS)
        n_frames = info.get("nb_frames")
        max_frames = None
        if job.max_s is not None:
            max_frames = math.ceil(job.max_s * fps)
        n_total = None
        if isinstance(n_frames, int) and n_frames > 0:
            n_total = min(n_frames, max_frames) if max_frames else n_frames
        elif max_frames:
            n_total = max_frames
        job.set_video(dict(info), n_total)
        if job.cancel_requested:
            job.set_status("cancelled")
            return
        if job.zones:
            # Zones were validated against the preview frame, which shares the normalize
            # scale filter, so this can only trip if the cached preview and the actual
            # normalize disagreed. Fail loudly rather than monitor a door that is not
            # where the user drew it.
            w, h = int(info["width"]), int(info["height"])
            for zone in job.zones:
                x1, y1, x2, y2 = (float(v) for v in zone["rect"])
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    raise RuntimeError(
                        f"exit zone {zone['name']!r} lies outside the processed frame "
                        f"({w}x{h}); re-upload the clip and redraw the zone"
                    )

        rules_label = f"{job.rules_profile} rules"
        run_cfg = build_run_cfg(
            job_id=job.job_id,
            runs_dir=self.dirs["runs"],
            input_mp4=input_mp4,
            detector=job.detector,
            pose=job.pose,
            score_thr=job.score_thr,
            rules_profile=job.rules_profile,
            repo_root=self.repo_root,
            fps=fps,
            zones=job.zones,
        )
        runner = _load_runner(self.repo_root)
        writer = media.AnnotatedWriter(
            job.job_dir / "annotated.mp4", int(info["width"]), int(info["height"]), fps
        )
        hook = DemoFrameHook(job, writer, rules_label)
        pipeline = make_pipeline(runner, run_cfg, self.repo_root, hook)
        job.pipeline = pipeline
        job.started_monotonic = time.monotonic()
        job.set_status("running")
        job.persist()
        try:
            pipeline.run(max_frames)
        finally:
            pipeline.close()
            writer.close()  # raises on encoder failure after the pipeline is down
        job.annotated_ready = (job.job_dir / "annotated.mp4").is_file()
        job.set_frame_b64(None)

        # authoritative alert list + latency summary come from the run's own artifacts
        store = EventStore(job.job_dir / "alerts.sqlite")
        try:
            alerts = [a.model_dump(mode="json") for a in store.alerts(limit=1000)]
        finally:
            store.close()
        alerts.sort(key=lambda r: float(r["onset_ts"]))
        job.replace_alerts(alerts)
        job.set_latency(_latency_summary(job.job_dir / "latency.jsonl"))
        if not job.thumb_done:
            _thumb_from_video(job.job_dir / "annotated.mp4", job.job_dir / "thumb.jpg")
            job.thumb_done = (job.job_dir / "thumb.jpg").is_file()
        job.set_status("cancelled" if job.cancel_requested else "done")

    def _run_live(self, job: JobState) -> None:
        """A camera session. No normalize, no writer, no thumbnail: the browser pushes
        JPEG frames into the registered LiveSource and the annotated stream going back
        over the job websocket is the only place a camera pixel ever appears."""
        with self._lock:
            src = self._live_sources.get(job.job_id)
        if src is None:
            raise RuntimeError("live session has no frame source (already stopped?)")
        run_cfg = build_run_cfg(
            job_id=job.job_id,
            runs_dir=self.dirs["runs"],
            # a real, existing clip keeps construction-time checks happy; the overridden
            # _footage_source() means not one frame is ever read from it
            input_mp4=self.sample_clip,
            detector=job.detector,
            pose=job.pose,
            score_thr=job.score_thr,
            rules_profile=job.rules_profile,
            repo_root=self.repo_root,
            fps=demo_live.LIVE_FPS,
        )
        runner = _load_runner(self.repo_root)
        hook = DemoFrameHook(job, None, f"{job.rules_profile} rules")
        pipeline = make_pipeline(runner, run_cfg, self.repo_root, hook, live_source=src)
        job.pipeline = pipeline
        job.started_monotonic = time.monotonic()
        job.set_status("running")
        job.persist()
        stopped_before_first_frame = False
        try:
            pipeline.run(None)
        except demo_live.LiveStopped:
            stopped_before_first_frame = True
        finally:
            src.stop()
            pipeline.close()
            with self._lock:
                self._live_sources.pop(job.job_id, None)
        job.set_frame_b64(None)
        if src.info is not None:
            job.set_video(
                {
                    "width": src.info.width,
                    "height": src.info.height,
                    "fps": demo_live.LIVE_FPS,
                    "frames_received": src.n_received,
                    "frames_dropped": src.n_dropped,
                },
                None,
            )

        store = EventStore(job.job_dir / "alerts.sqlite")
        try:
            alerts = [a.model_dump(mode="json") for a in store.alerts(limit=1000)]
        except Exception:
            alerts = []
        finally:
            store.close()
        alerts.sort(key=lambda r: float(r["onset_ts"]))
        job.replace_alerts(alerts)
        job.set_latency(_latency_summary(job.job_dir / "latency.jsonl"))
        if stopped_before_first_frame or job.cancel_requested:
            job.set_status("cancelled")
        else:
            job.set_status("done")


def _latency_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    summary = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("summary"):
                summary = row
    return summary


def _thumb_from_video(video: Path, dst: Path) -> None:
    """First-frame thumbnail fallback for runs too short to hit the 25% grab."""
    if not video.is_file():
        return
    import cv2  # heavy: function-local by package rule

    cap = cv2.VideoCapture(str(video))
    try:
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    finally:
        cap.release()
