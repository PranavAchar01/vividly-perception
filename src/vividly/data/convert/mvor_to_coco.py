"""camma_mvor_2018.json -> v2-convention person-detection COCO (local exam surface).

MVOR is EVAL_ONLY and clinical-adjacent: it never leaves this machine and never
reaches a pod. This converter produces the local processor-only probe surface for the
exported v2 person detector: visible-person boxes in the v2 corpus convention
(person = category id 0) so the same eval code reads every v2 surface.

Source quirks handled here, each verified on disk 2026-08-31:
- category_id is the string '1' in both the category table and every annotation;
- image ids are large structured ints (e.g. 10020000017 = day 1, cam 2, frame
  17); they are kept exactly as released so every eval row traces back to the
  released dataset without a mapping file;
- annotations carry keypoints/person_role/face_blurred extras, which are dropped so
  that only boxes remain; `annotations3D` and `multiview_images` are ignored entirely;
- bbox is COCO [x, y, w, h] pixels; boxes are clamped to the frame and
  sub-min_side slivers dropped, mirroring the SmartSpaces converter's rule;
- images with zero persons are kept as negatives (exam realism).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PERSON_CATEGORY = {"id": 0, "name": "person", "supercategory": "person"}


class MVORFormatError(ValueError):
    """The file on disk does not match the documented camma_mvor_2018 schema."""


@dataclass
class MVORReport:
    images: int = 0
    boxes: int = 0
    dropped_boxes: int = 0
    negatives: int = 0
    missing_files: int = 0


def _clamped_xywh(bbox: list, width: int, height: int, min_side_px: float) -> list[float] | None:
    if len(bbox) != 4:
        raise MVORFormatError(f"bbox must be [x,y,w,h], got {bbox!r}")
    x, y, w, h = (float(v) for v in bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 - x1 < min_side_px or y2 - y1 < min_side_px:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def convert_mvor(
    source_json: Path,
    image_root: Path | None = None,
    min_side_px: float = 2.0,
) -> tuple[dict, MVORReport]:
    """Convert the released MVOR file to a v2 person COCO dict.

    With image_root given, every file_name is checked on disk and a missing
    frame is a hard error - an exam whose images are absent must refuse to
    exist, not silently shrink.
    """
    raw = json.loads(source_json.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if key not in raw:
            raise MVORFormatError(f"{source_json}: missing {key!r}")
    cats = raw["categories"]
    if len(cats) != 1 or str(cats[0].get("id")) != "1" or cats[0].get("name") != "person":
        raise MVORFormatError(
            f"{source_json}: expected the single string-id person category, got {cats!r}"
        )
    report = MVORReport()
    dims: dict[int, tuple[int, int]] = {}
    images: list[dict] = []
    for img in raw["images"]:
        width, height = int(img["width"]), int(img["height"])
        dims[img["id"]] = (width, height)
        if image_root is not None and not (image_root / img["file_name"]).is_file():
            report.missing_files += 1
            continue
        images.append(
            {
                "id": img["id"],  # kept as released: day/cam/frame provenance
                "file_name": img["file_name"],
                "width": width,
                "height": height,
            }
        )
    if report.missing_files:
        raise MVORFormatError(
            f"{report.missing_files} of {len(raw['images'])} MVOR frames missing under "
            f"{image_root} - refusing to build a silently smaller exam"
        )
    kept_ids = {i["id"] for i in images}
    annotations: list[dict] = []
    ann_id = 0
    for ann in raw["annotations"]:
        if str(ann.get("category_id")) != "1":
            raise MVORFormatError(f"unexpected category_id {ann.get('category_id')!r}")
        if ann["image_id"] not in kept_ids:
            raise MVORFormatError(f"annotation for unknown image_id {ann['image_id']}")
        width, height = dims[ann["image_id"]]
        box = _clamped_xywh(ann["bbox"], width, height, min_side_px)
        if box is None:
            report.dropped_boxes += 1
            continue
        ann_id += 1
        annotations.append(
            {
                "id": ann_id,
                "image_id": ann["image_id"],
                "category_id": PERSON_CATEGORY["id"],
                "bbox": [round(v, 2) for v in box],
                "area": round(box[2] * box[3], 2),
                "iscrowd": 0,
            }
        )
    with_ann = {a["image_id"] for a in annotations}
    report.images = len(images)
    report.boxes = len(annotations)
    report.negatives = sum(1 for i in images if i["id"] not in with_ann)
    coco = {
        "info": {
            "description": "MVOR person boxes, v2 convention (local exam surface)",
            "source": "camma_mvor_2018.json (University of Strasbourg, CC BY-NC-SA 4.0)",
        },
        "images": images,
        "annotations": annotations,
        "categories": [PERSON_CATEGORY],
    }
    return coco, report
