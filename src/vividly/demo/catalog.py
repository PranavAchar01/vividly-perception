"""Model catalog: what the demo platform's picker offers.

Truth source: the `final_models/<lane>/model.json` sidecars. Each lane directory carries
exactly one sidecar describing that lane's model of record: its kind, the inference
backend it must run under, the display name the picker shows, the file name and sha256 of
its ONNX graph, the sha256 of the checkpoint the graph was exported from, the input size,
the weight class, and one sentence each for the training data and the backbone pretraining
data.

The picker offers the detector and pose lanes only, because those two are the models the
replay pipeline builds. The event verifier lanes are attached separately by
`vividly.demo.jobs` from their own lane directories. The skip on a sidecar that records a
pending status is a standing guard that no sidecar in the record exercises: every lane
under `final_models/` is measured and none carries a `status` key.

The second source is a dropped-in `.onnx` upload under the gitignored demo uploads tree,
stored by digest, listed after the lanes and badged as unverified.

The backend is read from the sidecar and never guessed from a file name. The eyes lane is
an RT-DETRv4 graph with its own preprocessing and postprocessing and the joints lane is an
RTMPose graph; running either under the other's backend raises nothing and returns
plausible numbers that are wrong. Provenance is read from the sidecar for the same reason:
`rtmlib_backend` self-reporting inspects the file name alone, so every locally exported
graph would label itself identically no matter what produced it.

Rescanning on every call is what lets a sidecar written during a training day appear in the
picker on the next fetch, with no watcher and no restart.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vividly.models.checkpoints import is_lfs_pointer
from vividly.schemas.events import utc_now_iso

#: The model record: one directory per lane, at the repository root.
MODEL_RECORD_DIR = "final_models"

#: The per-lane sidecar file name.
SIDECAR_NAME = "model.json"

#: The kinds the picker offers. Event verifier and re-identification lanes carry sidecars
#: of their own and are reached through the model record, not through this picker.
PICKER_KINDS = frozenset({"detector", "pose"})

#: Fallback input size (w, h) per kind, used only when a sidecar omits `input_size`.
DETECTOR_DEFAULT_INPUT = (640, 640)
POSE_DEFAULT_INPUT = (192, 256)

#: The backend a dropped-in upload runs under. An upload carries no sidecar and no
#: provenance, so it is assumed to follow the rtmlib graph convention; a graph with
#: different preprocessing belongs in the model record, where its backend is declared.
UPLOAD_BACKEND = "rtmlib"


# (path, mtime_ns, size, recorded digest) -> the digest matched. Hashing the largest graph
# in the record, the joints lane at about 54 MB, costs a noticeable fraction of a second;
# the page refetches the catalog every few seconds, and this cache is what keeps those
# rescans instant.
_sha_ok_cache: dict[tuple[str, int, int, str], bool] = {}


def _read_json(path: Path) -> dict | None:
    try:
        row = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return row if isinstance(row, dict) else None


def _sha256_matches(onnx: Path, recorded: str) -> bool | None:
    """None = nothing to check against; True or False = verified result (cached).

    The recorded digest is part of the cache key, so editing a sidecar re-verifies the
    same file rather than returning the previous answer.
    """
    want = recorded.strip().lower()
    if not want:
        return None
    try:
        st = onnx.stat()
    except OSError:
        return None
    key = (str(onnx), st.st_mtime_ns, st.st_size, want)
    hit = _sha_ok_cache.get(key)
    if hit is None:
        h = hashlib.sha256()
        with onnx.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        hit = h.hexdigest() == want
        _sha_ok_cache[key] = hit
    return hit


def _input_size(sidecar: dict[str, Any], kind: str) -> tuple[int, int]:
    """The sidecar's `input_size` as (w, h), or the kind's default when it is absent."""
    size = sidecar.get("input_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            return (int(size[0]), int(size[1]))
        except (TypeError, ValueError):
            pass
    return DETECTOR_DEFAULT_INPUT if kind == "detector" else POSE_DEFAULT_INPUT


def _picker_order(entry: dict[str, Any]) -> tuple:
    rank = {"prototype": 0, "vividly": 1, "upload": 2}
    return (rank.get(entry["badge"], 9), entry["display_name"])


def _entry_from_lane(lane_dir: Path) -> dict[str, Any] | None:
    """One `final_models/<lane>/model.json` as a picker entry, or None to skip the lane.

    A lane is skipped when it has no sidecar, when its kind is not one the picker offers,
    when it records a pending status, or when the sidecar names no backend, no graph file
    or no ONNX digest. A lane whose graph is absent, is still a Git LFS pointer, or whose
    digest does not match is listed and disabled, because a picker that silently drops a
    model of record is how a demo ends up running something other than what it says it
    runs.
    """
    sidecar = _read_json(lane_dir / SIDECAR_NAME)
    if sidecar is None:
        return None
    kind = str(sidecar.get("kind") or "")
    if kind not in PICKER_KINDS:
        return None
    if str(sidecar.get("status") or "").lower() == "pending":
        return None
    backend = str(sidecar.get("backend") or "")
    onnx_name = str(sidecar.get("onnx") or "")
    digest = str(sidecar.get("onnx_sha256") or "").strip().lower()
    if not backend or not onnx_name or not digest:
        return None

    lane = str(sidecar.get("lane") or lane_dir.name)
    onnx = lane_dir / onnx_name
    present = onnx.is_file()
    pointer = present and is_lfs_pointer(onnx)
    sha_ok = _sha256_matches(onnx, digest) if present and not pointer else None
    weight_class = sidecar.get("weight_class")
    width, height = _input_size(sidecar, kind)
    disabled = None
    if not present:
        rel = f"{MODEL_RECORD_DIR}/{lane}/{onnx_name}"
        disabled = f"weight file missing: run git lfs pull to fetch {rel}"
    elif pointer:
        rel = f"{MODEL_RECORD_DIR}/{lane}/{onnx_name}"
        disabled = f"weight file is a Git LFS pointer: run git lfs pull to fetch {rel}"
    elif sha_ok is False:
        disabled = "digest mismatch, not the recorded artefact"
    return {
        "id": f"{lane}/{digest[:8]}",
        "lane": lane,
        "kind": kind,
        "backend": backend,
        "display_name": str(sidecar.get("display_name") or lane),
        "sha8": digest[:8],
        "onnx_path": str(onnx.resolve()),
        "onnx_present": present and not pointer,
        "sha256_verified": bool(sha_ok),
        "input_size": [width, height],
        "badge": "prototype" if weight_class == "prototype" else "vividly",
        "weight_class": weight_class,
        "card": None,
        "source": MODEL_RECORD_DIR,
        "disabled": disabled,
    }


def _entry_from_upload(upload_dir: Path) -> dict[str, Any] | None:
    """A dropped-in .onnx (outputs/demo_uploads/models/<sha8>/): meta.json plus the blob."""
    meta = _read_json(upload_dir / "meta.json")
    if not meta or meta.get("kind") not in PICKER_KINDS:
        return None
    onnx = next(iter(sorted(upload_dir.glob("*.onnx"))), None)
    size = meta.get("input_size") or [0, 0]
    return {
        "id": f"upload/{upload_dir.name}",
        "lane": None,
        "kind": meta["kind"],
        "backend": UPLOAD_BACKEND,
        "display_name": f"Unverified upload: {meta.get('display_name') or upload_dir.name}",
        "sha8": upload_dir.name,
        "onnx_path": str(onnx.resolve()) if onnx else "",
        "onnx_present": onnx is not None,
        "sha256_verified": onnx is not None,  # the directory is named by the digest read here
        "input_size": [int(size[0]), int(size[1])],
        "badge": "upload",
        "weight_class": None,
        "card": None,
        "source": "upload",
        "disabled": None if onnx else "uploaded file missing",
    }


def scan_catalog(repo_root: Path, uploads_models_dir: Path | None = None) -> dict[str, Any]:
    """Enumerate the model record lanes and the dropped-in uploads as picker entries,
    prototypes first and unverified uploads last."""
    repo_root = Path(repo_root)
    record = repo_root / MODEL_RECORD_DIR
    detectors: list[dict[str, Any]] = []
    poses: list[dict[str, Any]] = []
    if record.is_dir():
        for lane_dir in sorted(p for p in record.iterdir() if p.is_dir()):
            entry = _entry_from_lane(lane_dir)
            if entry is None:
                continue
            (detectors if entry["kind"] == "detector" else poses).append(entry)
    if uploads_models_dir is not None and Path(uploads_models_dir).is_dir():
        for upload_dir in sorted(p for p in Path(uploads_models_dir).iterdir() if p.is_dir()):
            entry = _entry_from_upload(upload_dir)
            if entry is None:
                continue
            (detectors if entry["kind"] == "detector" else poses).append(entry)
    detectors.sort(key=_picker_order)
    poses.sort(key=_picker_order)
    return {"generated_utc": utc_now_iso(), "detectors": detectors, "poses": poses}


def find_entry(catalog: dict[str, Any], entry_id: str) -> dict[str, Any] | None:
    for entry in [*catalog.get("detectors", []), *catalog.get("poses", [])]:
        if entry["id"] == entry_id:
            return entry
    return None


def infer_onnx_input_size(path: Path) -> tuple[int, int]:
    """(w, h) from the graph's first input, for dropped-in .onnx files. Static shapes
    only: a dynamic dimension means the trained size is unknowable here, and a wrong
    input size degrades quietly rather than raising, so refuse instead of guessing."""
    import onnxruntime as ort  # heavy: function-local by package rule

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape  # NCHW
    if len(shape) != 4 or not all(isinstance(d, int) and d > 0 for d in shape[2:]):
        raise ValueError(f"input shape {shape} is not static NCHW; specify input size manually")
    return (int(shape[3]), int(shape[2]))
