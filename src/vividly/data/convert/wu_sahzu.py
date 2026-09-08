"""WU-SAHZU-EMU-Video labels + clip cutting (registry entries wu_sahzu_emu and
wu_sahzu_emu_trainsplit).

Real clinical epilepsy-monitoring-unit video (Second Affiliated Hospital of Zhejiang
University; released with VSViG, ECCV 2024, arXiv:2311.14775) mirrored from
huggingface.co/datasets/xuyankun/WU-SAHZU-EMU-Video under Apache-2.0. 14 patients,
33 seizure events, wall/ceiling 1080p30 camera, faces masked upstream.

Both registry entries name this module as their converter, but build_index accepts the
wu_sahzu_emu layout only: it reads root/dataset/Label.xlsx and root/dataset/patNN/*.mp4,
while the trainsplit root holds patNN/*.mp4 with no workbook beside them.

wu_sahzu_emu is registry-enforced test_only (bucket EVAL_ONLY) and is the exam-side
patient pool. The train and test split it was held out for was decided on 2026-08-26 and
is recorded as the second, disjoint patient-level entry wu_sahzu_emu_trainsplit (bucket
TRAIN_PROTOTYPE, status available, 4 patients, 8 events). The seizure model of record,
convulsion_v2_real, checkpoint 715d172c, ONNX 07c399be, trains on those four patients and
is graded at window AUC 0.69841 on the 16-video testsplit, patients pat03, pat07, pat14,
pat04, pat09, pat11 (final_models/seizure/README.md,
configs/events/eval_wusahzu_testsplit_v2real.yaml).

Labels (dataset/Label.xlsx, parsed with the stdlib because the curate env is not a
given here): rows (PatID, Seizure Type, #Seizure, EEG onset, Clinical Onset). PatID
and Seizure Type live only on each patient's first row (merged cells) and forward-
fill. Onset cells are Excel day fractions, so multiply by 86400 for seconds. Seizure
Type is per patient: PG = focal-to-generalized tonic-clonic (convulsive, the class
the Rhythmic convulsion verifier targets), P = focal (motion often subtle; evals
report these separately rather than scoring the verifier on a class it does not
claim). EEG onset precedes clinical onset on every row.

Onset semantics: clinical onset, the point of obvious behavioral abnormalities, is when a
camera-only system could first see anything, so the event interval for labeling is
[clinical_onset, clinical_onset + ICTAL_S], the upstream paper's own heuristic (it
ships no end times, recording only that the event is under 2 minutes after clinical
onset). eeg_onset_t is carried in every
record for latency analysis; windows before clinical onset count as negatives for a
behaviour model by definition of clinical onset.

Cutting: recordings range from 80 s clips to 61-minute continuous recordings with
the seizure an hour in. Extraction replays every frame through the full perception
stack, so hour-long files are cut to [clinical_onset - CUT_PRE_S,
clinical_onset + CUT_POST_S] first. Cuts re-encode (libx264 veryfast crf 18):
stream-copy can only cut on keyframes, which would shift the clip's t=0 by an
unrecorded amount and silently misalign every onset. Files at or under
WHOLE_FILE_MAX_S are kept whole (no re-encode, no pixel change). Non-seizure files
(free*.mp4 / no-Sz*.mp4) are kept whole as the false-alarm probe. Every record
carries cut_start_s so clip time maps back to recording time.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

INDEX_FILENAME = "index.json"

SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

#: The upstream paper's ictal-period heuristic (no end times ship with the labels).
ICTAL_S = 120.0
#: Cut window around clinical onset for long recordings.
CUT_PRE_S = 180.0
CUT_POST_S = 240.0
#: Files at or under this length are used whole (no re-encode).
WHOLE_FILE_MAX_S = 600.0

CONVULSIVE_TYPE = "PG"  # focal-to-generalized tonic-clonic
FOCAL_TYPE = "P"


class WuSahzuParseError(ValueError):
    """Label workbook or tree state that cannot become an index."""


def parse_label_xlsx(path: Path) -> list[dict[str, Any]]:
    """The 33 event rows: patid, seizure_type, seizure, eeg_onset_s, clinical_onset_s."""
    with zipfile.ZipFile(path) as z:
        shared_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in si.iter(f"{SHEET_NS}t")) for si in shared_root]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[str, Any]] = []
    patid: str | None = None
    seizure_type: str | None = None
    for row in sheet.iter(f"{SHEET_NS}row"):
        cells: dict[str, str] = {}
        for c in row.iter(f"{SHEET_NS}c"):
            ref = c.get("r") or ""
            col = "".join(ch for ch in ref if ch.isalpha())
            v = c.find(f"{SHEET_NS}v")
            if v is None or v.text is None:
                continue
            cells[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
        if cells.get("A") == "PatID":
            continue  # header
        if "A" in cells:
            patid = cells["A"].strip()
            seizure_type = None
        if "B" in cells:
            seizure_type = cells["B"].strip()
        if "C" not in cells:
            continue
        if patid is None or seizure_type is None:
            raise WuSahzuParseError(f"event row {cells} before any PatID/type row")
        if seizure_type not in (CONVULSIVE_TYPE, FOCAL_TYPE):
            raise WuSahzuParseError(f"{patid}: unknown seizure type {seizure_type!r}")
        try:
            eeg = float(cells["D"]) * 86400.0
            clinical = float(cells["E"]) * 86400.0
        except (KeyError, ValueError) as exc:
            raise WuSahzuParseError(f"{patid} {cells.get('C')}: bad onset cells") from exc
        if clinical < eeg:
            raise WuSahzuParseError(
                f"{patid} {cells['C']}: clinical onset {clinical:.1f}s before EEG onset {eeg:.1f}s"
            )
        rows.append(
            {
                "patid": patid,
                "seizure_type": seizure_type,
                "seizure": cells["C"].strip(),
                "eeg_onset_s": round(eeg, 3),
                "clinical_onset_s": round(clinical, 3),
            }
        )
    if not rows:
        raise WuSahzuParseError(f"{path}: no event rows parsed")
    return rows


def event_video_rel(event: Mapping[str, Any]) -> Path:
    """dataset/patNN/SzK<type>.mp4 - the type suffix is the patient's type column."""
    pat = event["patid"].lower()  # Pat01 -> pat01
    return Path("dataset") / pat / f"{event['seizure']}{event['seizure_type']}.mp4"


def _probe_duration_cv2(path: Path) -> tuple[float, float, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise WuSahzuParseError(f"cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not fps > 0 or n <= 0:
        raise WuSahzuParseError(f"container reports fps={fps} frames={n} for {path}")
    return n / fps, fps, n


def plan_cut(duration_s: float, clinical_onset_s: float) -> tuple[float, float] | None:
    """(cut_start_s, cut_duration_s) for a long recording, None to use the whole file."""
    if duration_s <= WHOLE_FILE_MAX_S:
        return None
    start = max(0.0, clinical_onset_s - CUT_PRE_S)
    end = min(duration_s, clinical_onset_s + CUT_POST_S)
    if not end > start:
        raise WuSahzuParseError(
            f"cut window [{start:.1f}, {end:.1f}] empty for onset "
            f"{clinical_onset_s:.1f}s in a {duration_s:.1f}s file"
        )
    return round(start, 3), round(end - start, 3)


def _cut_argv(ffmpeg: str, src: Path, dst: Path, start_s: float, dur_s: float) -> list[str]:
    """Frame-accurate re-encode cut (input -ss with re-encode decodes forward from the
    prior keyframe internally, so the clip really starts at start_s)."""
    return [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{dur_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]


def negative_videos(root: Path) -> list[Path]:
    """Non-seizure recordings (free*.mp4 / no-Sz*.mp4), relative to root."""
    out = []
    for p in sorted((root / "dataset").glob("pat*/*.mp4")):
        if not p.name.startswith("Sz"):
            out.append(p.relative_to(root))
    return out


def build_index(
    root: Path,
    *,
    probe: Callable[[Path], tuple[float, float, int]] | None = None,
    cut: bool = True,
) -> dict[str, Any]:
    """One record per event video (cut clip when long) + one per non-seizure video.

    With cut=True, missing clips are produced under clips/ via ffmpeg re-encode;
    upstream bytes are never modified.
    """
    if probe is None:
        probe = _probe_duration_cv2
    events = parse_label_xlsx(root / "dataset" / "Label.xlsx")

    records: list[dict[str, Any]] = []
    for ev in events:
        src_rel = event_video_rel(ev)
        src = root / src_rel
        if not src.is_file():
            raise WuSahzuParseError(f"missing event video {src}")
        duration, fps, n_frames = probe(src)
        onset = ev["clinical_onset_s"]
        if onset >= duration:
            raise WuSahzuParseError(
                f"{src_rel}: clinical onset {onset:.1f}s beyond duration {duration:.1f}s"
            )
        window = plan_cut(duration, onset)
        if window is None:
            video_rel, cut_start = src_rel, 0.0
        else:
            cut_start, cut_dur = window
            video_rel = Path("clips") / src_rel.relative_to("dataset")
            dst = root / video_rel
            if cut:
                if not dst.is_file():
                    import subprocess

                    from vividly.media.ffmpeg import ffmpeg_bin

                    dst.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        _cut_argv(ffmpeg_bin(), src, dst, cut_start, cut_dur),
                        check=True,
                    )
                duration, fps, n_frames = probe(dst)
            elif dst.is_file():
                duration, fps, n_frames = probe(dst)
            else:
                raise WuSahzuParseError(f"cut disabled and clip missing: {dst}")
        onset_clip = onset - cut_start
        eeg_clip = max(0.0, ev["eeg_onset_s"] - cut_start)
        event_end = min(onset_clip + ICTAL_S, duration)
        records.append(
            {
                "location": ev["patid"].lower(),
                "video": str(video_rel),
                "video_upstream": str(src_rel),
                "fps": fps,
                "n_frames_container": n_frames,
                "annotation": "dataset/Label.xlsx (VSViG expert onsets; end = onset + "
                f"{ICTAL_S:g}s paper heuristic)",
                "has_fall": False,
                "has_seizure": True,
                "seizure_type": ev["seizure_type"],
                "convulsive": ev["seizure_type"] == CONVULSIVE_TYPE,
                "seizure_id": ev["seizure"],
                "cut_start_s": cut_start,
                "eeg_onset_t": round(eeg_clip, 3),
                "clinical_onset_t": round(onset_clip, 3),
                "events_t": [[round(onset_clip, 3), round(event_end, 3)]],
            }
        )

    for rel in negative_videos(root):
        duration, fps, n_frames = probe(root / rel)
        records.append(
            {
                "location": rel.parts[1],
                "video": str(rel),
                "video_upstream": str(rel),
                "fps": fps,
                "n_frames_container": n_frames,
                "annotation": "non-seizure recording (filename convention)",
                "has_fall": False,
                "has_seizure": False,
                "seizure_type": None,
                "convulsive": False,
                "seizure_id": None,
                "cut_start_s": 0.0,
                "eeg_onset_t": None,
                "clinical_onset_t": None,
                "events_t": [],
            }
        )

    summary = {
        "n_videos": len(records),
        "n_seizure": sum(1 for r in records if r["has_seizure"]),
        "n_convulsive": sum(1 for r in records if r["convulsive"]),
        "n_focal": sum(1 for r in records if r["has_seizure"] and not r["convulsive"]),
        "n_negative": sum(1 for r in records if not r["has_seizure"]),
        "n_patients": len({r["location"] for r in records}),
    }
    return {"dataset": "wu_sahzu_emu", "summary": summary, "videos": records}


def write_index(root: Path, index: Mapping[str, Any] | None = None) -> Path:
    if index is None:
        index = build_index(root)
    out = root / "annotations" / INDEX_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
