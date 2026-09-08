"""Evidence packs and the per-rule false-alarm report: the untrained half of the verifier
harness, in which a human is the verifier.

An evidence pack is what a reviewer needs to judge one firing: the frames from a little
before the rule's onset to a little after its detection, the alert record itself (rule,
track, onset, the rule's own details, the baseline snapshot), the reviewer's label if one
exists, and enough provenance to find the run again. It is written as a directory of
JPEGs plus one manifest.json, a format a trained verifier can consume later without
anything here training anything today: whether packs may ever train a model is
undecided, POLICY.md, rule 4 stands, and the labels in a pack come only from the
dashboard's Agree / False-alarm route.

The false-alarm report turns those labels into the per-rule numbers the calibration
protocol needs (docs/CALIBRATION.md step 1): firings, labelled, agreed, false alarms, the
false-alarm fraction among the labelled, and firings per camera-hour when the caller knows
the camera-time. On footage with no events in it (the BU burn-in) every firing is a false
positive whether or not anyone has clicked, and the report says which it is counting.

Pure parts first (window arithmetic, manifest, report); the two extractors at the bottom
copy or decode frames and are the only functions that touch the filesystem. cv2 is
imported inside the clip extractor only, never at module level.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vividly.rules import DEFAULT_RULE_IDS
from vividly.schemas.events import FEEDBACK_LABELS

SCHEMA_VERSION = 1
DEFAULT_PRE_S = 10.0
DEFAULT_POST_S = 5.0
MANIFEST_NAME = "manifest.json"
FRAMES_DIR = "frames"
NOTE = (
    "Review material for a human verifier. The label, when present, is the "
    "dashboard's Agree / False-alarm click on this alert. Nothing in this pack trains any "
    "model until an owner decision says what may; POLICY.md, rule 4 stands."
)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class EvidenceError(ValueError):
    """An alert or window a human fixes."""


@dataclass(frozen=True)
class EvidenceWindow:
    """Clip-time bounds of one pack: [onset - pre, detected + post], clamped at 0."""

    onset_t: float
    detected_t: float
    start_t: float
    end_t: float

    def frame_range(self, fps: float, n_frames: int | None) -> range:
        """Source frame indices inside the window, first and last inclusive, clamped."""
        if not math.isfinite(fps) or fps <= 0.0:
            raise EvidenceError(f"fps must be a finite positive rate, got {fps!r}")
        first = max(0, math.floor(self.start_t * fps))
        last = math.ceil(self.end_t * fps)
        if n_frames is not None:
            last = min(last, n_frames - 1)
        if last < first:
            return range(first, first)
        return range(first, last + 1)

    @property
    def span_s(self) -> float:
        return self.end_t - self.start_t


def window_for(
    alert: Mapping[str, Any], *, pre_s: float = DEFAULT_PRE_S, post_s: float = DEFAULT_POST_S
) -> EvidenceWindow:
    """The review window for one alert record (a dict or an AlertRecord dump)."""
    for name, value in (("pre_s", pre_s), ("post_s", post_s)):
        if not math.isfinite(value) or value < 0.0:
            raise EvidenceError(f"{name} must be >= 0 and finite, got {value!r}")
    try:
        onset = float(alert["onset_ts"])
        detected = float(alert["detected_ts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"alert needs numeric onset_ts and detected_ts: {exc}") from exc
    if not math.isfinite(onset) or not math.isfinite(detected):
        raise EvidenceError("alert onset_ts / detected_ts must be finite")
    if detected < onset:
        raise EvidenceError(f"detected_ts {detected} precedes onset_ts {onset}")
    return EvidenceWindow(
        onset_t=onset,
        detected_t=detected,
        start_t=max(0.0, onset - pre_s),
        end_t=detected + post_s,
    )


def pack_id(alert: Mapping[str, Any]) -> str:
    """A filesystem-safe directory name derived from the alert id (which is itself derived)."""
    raw = str(alert.get("alert_id") or "")
    if not raw:
        raw = f"{alert.get('rule_id', 'rule')}-{alert.get('track_id', 'x')}-{alert.get('onset_ts', 0)}"
    return _SAFE.sub("_", raw).strip("_") or "alert"


def pack_manifest(
    alert: Mapping[str, Any],
    window: EvidenceWindow,
    frames: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    source: Mapping[str, Any],
    run: Mapping[str, Any],
    extracted_utc: str,
) -> dict[str, Any]:
    """The one JSON file beside the frames. Everything a reviewer or a loader needs."""
    label = alert.get("feedback")
    if label is not None and label not in FEEDBACK_LABELS:
        raise EvidenceError(f"unknown feedback label {label!r}; expected one of {FEEDBACK_LABELS}")
    return {
        "schema_version": SCHEMA_VERSION,
        "pack_id": pack_id(alert),
        "alert": dict(alert),
        "label": label,
        "window": asdict(window),
        "fps": float(fps),
        "n_frames": len(frames),
        "frames": [dict(f) for f in frames],
        "source": dict(source),
        "run": dict(run),
        "extracted_utc": extracted_utc,
        "note": NOTE,
    }


# ------------------------------------------------------------------ report
def false_alarm_report(
    alerts: Sequence[Mapping[str, Any]],
    *,
    camera_s: float | None = None,
    rule_ids: Sequence[str] = DEFAULT_RULE_IDS,
    all_false_positives: bool = False,
) -> dict[str, Any]:
    """Per-rule counts from the reviewer labels.

    `all_false_positives=True` is the negative-control stance (footage known to contain no
    events): every firing counts as a false positive regardless of labels, and the report
    says so. Otherwise only the dashboard's labels are counted and the unlabelled remainder
    is reported as exactly that, never as agreement and never as error.
    """
    rules = list(rule_ids)
    for a in alerts:
        if str(a.get("rule_id")) not in rules:
            rules.append(str(a.get("rule_id")))
    per_rule: dict[str, dict[str, Any]] = {}
    for rid in rules:
        rows = [a for a in alerts if str(a.get("rule_id")) == rid]
        agree = sum(1 for a in rows if a.get("feedback") == "agree")
        false_alarm = sum(1 for a in rows if a.get("feedback") == "false_alarm")
        labelled = agree + false_alarm
        n = len(rows)
        fa_count = n if all_false_positives else false_alarm
        per_rule[rid] = {
            "firings": n,
            "labelled": labelled,
            "agree": agree,
            "false_alarm": false_alarm,
            "unlabelled": n - labelled,
            "false_alarm_frac_of_labelled": (
                round(false_alarm / labelled, 3) if labelled else None
            ),
            "false_positives_counted": fa_count,
            "false_positives_per_camera_hour": (
                round(fa_count / (camera_s / 3600.0), 3) if camera_s else None
            ),
        }
    return {
        "rule_ids": rules,
        "n_alerts": len(alerts),
        "camera_s": camera_s,
        "camera_hours": round(camera_s / 3600.0, 4) if camera_s else None,
        "all_false_positives": bool(all_false_positives),
        "per_rule": per_rule,
        "status": (
            "negative control: every firing counted as a false positive (footage contains no events)"
            if all_false_positives
            else "reviewer labels only; unlabelled firings are neither agreement nor error"
        ),
    }


def render_false_alarm_report(report: Mapping[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"{report['n_alerts']} alert(s)"
        + (
            f" over {report['camera_hours']:.2f} camera-hours"
            if report.get("camera_hours") is not None
            else ""
        )
        + f". {report['status']}. Thresholds PROVISIONAL (POLICY.md, measurement discipline; "
        "docs/CALIBRATION.md): a count, not a rate.",
        "",
        "| rule | firings | labelled | agree | false alarm | unlabelled | false-alarm frac (labelled) | false positives counted | per camera-hour |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rid in report["rule_ids"]:
        r = report["per_rule"][rid]
        frac = r["false_alarm_frac_of_labelled"]
        per = r["false_positives_per_camera_hour"]
        lines.append(
            f"| `{rid}` | {r['firings']} | {r['labelled']} | {r['agree']} | {r['false_alarm']} | "
            f"{r['unlabelled']} | {'n/a' if frac is None else f'{frac:.2f}'} | "
            f"{r['false_positives_counted']} | {'n/a' if per is None else f'{per:.2f}'} |"
        )
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------ extraction
def frame_rows_for_files(
    files: Sequence[Path], fps: float, window: EvidenceWindow
) -> list[dict[str, Any]]:
    """Which of an ordered frame listing fall inside the window, with their clip times."""
    rows = []
    for index in window.frame_range(fps, len(files)):
        rows.append({"index": index, "t_source": index / fps, "src": str(files[index])})
    return rows


def extract_from_files(
    files: Sequence[Path], fps: float, window: EvidenceWindow, pack_dir: Path
) -> list[dict[str, Any]]:
    """Copy the window's frames out of an image sequence. Returns the manifest frame rows."""
    rows = frame_rows_for_files(files, fps, window)
    frames_dir = pack_dir / FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for row in rows:
        src = Path(row["src"])
        name = f"{row['index']:07d}{src.suffix.lower() or '.jpg'}"
        shutil.copy2(src, frames_dir / name)
        out.append(
            {"index": row["index"], "t_source": row["t_source"], "file": f"{FRAMES_DIR}/{name}"}
        )
    return out


def extract_from_clip(
    clip: Any, window: EvidenceWindow, pack_dir: Path, *, jpeg_quality: int = 85
) -> list[dict[str, Any]]:
    """Decode the window's frames out of a FrameSource (ReplaySource) and write JPEGs."""
    import cv2

    info = clip.open()
    rng = window.frame_range(info.fps, info.n_frames)
    frames_dir = pack_dir / FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    if len(rng) == 0:
        return out
    clip.seek(rng.start)
    for frame in clip.frames():
        if frame.index >= rng.stop:
            break
        name = f"{frame.index:07d}.jpg"
        ok = cv2.imwrite(
            str(frames_dir / name), frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        )
        if not ok:
            raise EvidenceError(f"could not write {frames_dir / name}")
        out.append(
            {"index": frame.index, "t_source": frame.t_source, "file": f"{FRAMES_DIR}/{name}"}
        )
    return out


def write_pack(pack_dir: Path, manifest: Mapping[str, Any]) -> Path:
    pack_dir.mkdir(parents=True, exist_ok=True)
    path = pack_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_pack(pack_dir: Path) -> dict[str, Any]:
    return json.loads((Path(pack_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_POST_S",
    "DEFAULT_PRE_S",
    "FRAMES_DIR",
    "MANIFEST_NAME",
    "NOTE",
    "SCHEMA_VERSION",
    "EvidenceError",
    "EvidenceWindow",
    "extract_from_clip",
    "extract_from_files",
    "false_alarm_report",
    "frame_rows_for_files",
    "pack_id",
    "pack_manifest",
    "read_pack",
    "render_false_alarm_report",
    "window_for",
    "write_pack",
]
