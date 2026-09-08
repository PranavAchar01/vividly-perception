"""AI City MTMC ground truth -> per-scene COCO person detection JSON + frames.

The PhysicalAI-SmartSpaces MTMC_Tracking_2025 layout (verified on disk 2026-08-31):
each scene directory holds `videos/Camera_NNNN.mp4` (30 fps, 9,000 frames = 300 s),
`calibration.json`, and one `ground_truth.json` keyed by global frame index
("0".."8999"), each value a list of objects:

    {"object type": "Person" | "Forklift" | "NovaCarter" | "Transporter" | ...,
     "object id": int,
     "2d bounding box visible": {"Camera_0000": [x1, y1, x2, y2], ...},
     ... 3d fields ignored here ...}

Only the configured object types (default: Person alone, because the robots are not
people and never enter a person-detection corpus) are emitted, and only their
visible-region boxes: that key is the box convention recorded 2026-08-31, so the
released boxes are used as-is with no amodal re-derivation.

Frame sampling is a uniform stride over the ground-truth frame indices (default 30 =
1 sampled frame per second): recording diversity beats frame density, and adjacent
30 fps frames are near-duplicates that would only fatten the corpus. Extraction shells
out to ffmpeg with subprocess argv (ffmpeg-python is banned repo-wide), selects
`not(mod(n, stride))`, and the extractor then renames the sequential outputs to their
source frame index so `frames/<Camera>/<frame:06d>.jpg` matches the ground-truth key
forever. Every plan is cross-checked after extraction: file count per camera must
equal the plan, and every COCO file_name must exist on disk, so a silent ffmpeg
undercount becomes a raised error, not a corrupted corpus.

Empty sampled frames (no visible person in that camera) are kept as negative images
by default: real training negatives, exactly what a detector needs to not hallucinate.

The person category id is 0, not the COCO-conventional 1. This is the v2 corpus rule:
the trainers index the class head by the raw ``category_id`` whenever
``remap_mscoco_category: False``, evidenced in the DEIM loader (DEIM @ 09d35d5,
engine/data/dataset/coco_dataset.py) and in the audited D-FINE repository whose model
code DEIM vendors (D-FINE @ 956d170, src/data/dataset/coco_dataset.py). So a 1-based id with
``num_classes: 1`` is an out-of-range label and a device-side assert on the first
batch (DEIM #12/#25/#109, D-FINE #10/#37). Emitting 0 here means the trainers consume
these files unpatched; pycocotools/faster-coco-eval accept id 0 without complaint.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# id 0 on purpose; see the module docstring (v2 corpus rule: trainers index by raw id).
PERSON_CATEGORY = {"id": 0, "name": "person", "supercategory": "person"}


class MTMCFormatError(ValueError):
    """The ground truth on disk does not match the documented MTMC schema."""


class ExtractionMismatch(RuntimeError):
    """ffmpeg produced different frames than the plan; the corpus must not be trusted."""


@dataclass(frozen=True)
class CameraPlan:
    camera: str
    video: Path
    width: int
    height: int
    n_video_frames: int


@dataclass
class SceneReport:
    scene: str
    cameras: int = 0
    sampled_frames: int = 0
    images: int = 0
    person_boxes: int = 0
    dropped_boxes: int = 0
    object_types_seen: dict[str, int] = field(default_factory=dict)


def load_ground_truth(path: Path) -> dict[int, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise MTMCFormatError(f"{path}: expected a non-empty frame-keyed object")
    try:
        gt = {int(k): v for k, v in raw.items()}
    except ValueError as exc:
        raise MTMCFormatError(f"{path}: non-integer frame key: {exc}") from None
    probe = next(iter(gt.values()))
    if not isinstance(probe, list):
        raise MTMCFormatError(f"{path}: frame values must be object lists")
    return gt


def sample_frames(frame_keys: list[int], stride: int) -> list[int]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return [k for k in sorted(frame_keys) if k % stride == 0]


def probe_video(video: Path) -> tuple[int, int, int]:
    """(width, height, n_frames) via ffprobe. Frame count from nb_frames when the
    container carries it, else counted the slow exact way (-count_frames)."""
    argv = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames",
        "-of",
        "json",
        str(video),
    ]
    out = subprocess.run(argv, check=True, capture_output=True, text=True).stdout
    stream = json.loads(out)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    nb = stream.get("nb_frames")
    if nb in (None, "N/A"):
        argv2 = [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(video),
        ]
        out2 = subprocess.run(argv2, check=True, capture_output=True, text=True).stdout
        nb = json.loads(out2)["streams"][0]["nb_read_frames"]
    return width, height, int(nb)


def extract_frames(
    video: Path, out_dir: Path, stride: int, frame_indices: list[int], jpeg_q: int = 2
) -> None:
    """Extract every stride-th frame and rename outputs to their source frame index.

    ffmpeg numbers selected frames sequentially (1, 2, 3, ...); selected frame k is
    source frame (k-1)*stride, which must equal frame_indices[k-1]. That is asserted, so a
    plan/extraction drift raises instead of mislabeling frames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_pattern = out_dir / "_tmp_%06d.jpg"
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select='not(mod(n,{stride}))'",
        "-fps_mode",
        "vfr",
        "-q:v",
        str(jpeg_q),
        "-start_number",
        "1",
        str(tmp_pattern),
    ]
    subprocess.run(argv, check=True, capture_output=True)
    produced = sorted(out_dir.glob("_tmp_*.jpg"))
    if len(produced) != len(frame_indices):
        for p in produced:
            p.unlink()
        raise ExtractionMismatch(
            f"{video.name}: ffmpeg produced {len(produced)} frames, plan expected "
            f"{len(frame_indices)} (stride {stride})"
        )
    for k, tmp in enumerate(produced):
        source_frame = (int(tmp.stem.split("_")[-1]) - 1) * stride
        if source_frame != frame_indices[k]:
            raise ExtractionMismatch(
                f"{video.name}: output {tmp.name} maps to source frame {source_frame}, "
                f"plan says {frame_indices[k]}"
            )
        tmp.rename(out_dir / f"{source_frame:06d}.jpg")


def _clamped_box(xyxy: list, width: int, height: int, min_side_px: float) -> list[float] | None:
    if len(xyxy) != 4:
        raise MTMCFormatError(f"box must be [x1,y1,x2,y2], got {xyxy!r}")
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
    w, h = x2 - x1, y2 - y1
    if w < min_side_px or h < min_side_px:
        return None
    return [x1, y1, w, h]


def build_scene_coco(
    gt: dict[int, list[dict]],
    cameras: list[CameraPlan],
    frame_indices: list[int],
    scene: str,
    object_types: tuple[str, ...] = ("Person",),
    min_side_px: float = 2.0,
    keep_empty: bool = True,
    report: SceneReport | None = None,
) -> dict:
    """One COCO dict for the whole scene; file_name is `<Camera>/<frame:06d>.jpg`."""
    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 0
    ann_id = 0
    wanted = set(object_types)
    for cam in cameras:
        for frame in frame_indices:
            boxes: list[list[float]] = []
            for obj in gt.get(frame, ()):
                otype = str(obj.get("object type"))
                if report is not None:
                    report.object_types_seen[otype] = report.object_types_seen.get(otype, 0) + 1
                if otype not in wanted:
                    continue
                visible = obj.get("2d bounding box visible") or {}
                xyxy = visible.get(cam.camera)
                if xyxy is None:
                    continue
                box = _clamped_box(xyxy, cam.width, cam.height, min_side_px)
                if box is None:
                    if report is not None:
                        report.dropped_boxes += 1
                    continue
                boxes.append(box)
            if not boxes and not keep_empty:
                continue
            image_id += 1
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{cam.camera}/{frame:06d}.jpg",
                    "width": cam.width,
                    "height": cam.height,
                    "scene": scene,
                    "camera": cam.camera,
                    "source_frame": frame,
                }
            )
            for box in boxes:
                ann_id += 1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": PERSON_CATEGORY["id"],
                        "bbox": [round(v, 2) for v in box],
                        "area": round(box[2] * box[3], 2),
                        "iscrowd": 0,
                    }
                )
    if report is not None:
        report.images = len(images)
        report.person_boxes = len(annotations)
    return {
        "info": {
            "description": f"PhysicalAI-SmartSpaces {scene} person boxes (visible-region, as released)",
            "source": "nvidia/PhysicalAI-SmartSpaces MTMC_Tracking_2025",
        },
        "images": images,
        "annotations": annotations,
        "categories": [PERSON_CATEGORY],
    }


def convert_scene(
    scene_dir: Path,
    frames_out: Path,
    ann_out: Path,
    stride: int = 30,
    object_types: tuple[str, ...] = ("Person",),
    min_side_px: float = 2.0,
    keep_empty: bool = True,
    jpeg_q: int = 2,
    extract: bool = True,
    frame_tolerance: float = 0.01,
) -> SceneReport:
    """Convert one scene end to end. With extract=False, plan + COCO only (dry run)."""
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg/ffprobe not on PATH; the extractor shells out to them")
    gt = load_ground_truth(scene_dir / "ground_truth.json")
    videos = sorted((scene_dir / "videos").glob("Camera_*.mp4"))
    if not videos:
        raise MTMCFormatError(f"{scene_dir}: no Camera_*.mp4 under videos/")
    report = SceneReport(scene=scene_dir.name, cameras=len(videos))

    gt_frames = sorted(gt.keys())
    cameras: list[CameraPlan] = []
    for video in videos:
        width, height, n_frames = probe_video(video)
        # The GT is global-frame-indexed; a video materially shorter than the GT would
        # mislabel every sampled frame past its end. Tolerate container off-by-a-few.
        if abs(n_frames - len(gt_frames)) > max(1, int(frame_tolerance * len(gt_frames))):
            raise MTMCFormatError(
                f"{video.name}: {n_frames} frames vs {len(gt_frames)} ground-truth frames "
                f"- outside the {frame_tolerance:.0%} tolerance, refusing to mislabel"
            )
        cameras.append(CameraPlan(video.stem, video, width, height, n_frames))

    usable = min(len(gt_frames), min(c.n_video_frames for c in cameras))
    frame_indices = sample_frames([f for f in gt_frames if f < usable], stride)
    report.sampled_frames = len(frame_indices)

    if extract:
        for cam in cameras:
            cam_dir = frames_out / cam.camera
            done = all((cam_dir / f"{f:06d}.jpg").is_file() for f in frame_indices)
            if not done:  # resumable: a fully-extracted camera is never re-decoded
                extract_frames(cam.video, cam_dir, stride, frame_indices, jpeg_q)

    coco = build_scene_coco(
        gt, cameras, frame_indices, scene_dir.name, object_types, min_side_px, keep_empty, report
    )
    if extract:
        missing = [
            img["file_name"]
            for img in coco["images"]
            if not (frames_out / img["file_name"]).is_file()
        ]
        if missing:
            raise ExtractionMismatch(
                f"{scene_dir.name}: {len(missing)} COCO images missing on disk "
                f"(first: {missing[0]})"
            )
        ann_out.parent.mkdir(parents=True, exist_ok=True)
        ann_out.write_text(json.dumps(coco), encoding="utf-8")
    return report
