"""Merge per-scene COCO person JSONs into training/dev-val manifests.

The SmartSpaces converter (mtmc_to_coco) writes one COCO JSON per scene with
file_name relative to that scene's frame root (`Camera_NNNN/<frame:06d>.jpg`).
The vendored trainers want one json and one img_folder, so this module prefixes
each scene's file_names with the scene directory, reassigns image/annotation ids
sequentially, and enforces that every input carries the identical category table
(the v2 corpus rule: person, id 0; a mixed-convention merge is refused, not
repaired).

Dev-val thinning rides the converter's `source_frame` field: `frame_stride: 150`
keeps only images whose source frame index is a multiple of 150. Because every
scene was extracted at stride 30 and 150 = 5 x 30, the thinned manifest is an
exact subset of the frames already on disk, so there is no re-extraction, and the
dev-val rule (Warehouse_005 held out by scene from training) stays a pure config fact.

Negative images (no annotations) survive merging and thinning: they are real
training signal, not noise to drop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class MergeError(ValueError):
    """Inputs disagree in a way that would corrupt the merged manifest."""


@dataclass(frozen=True)
class MergeSource:
    path: Path
    prefix: str  # prepended to every file_name, "" for none
    frame_stride: int | None = None  # keep images with source_frame % stride == 0


def _load(path: Path) -> dict:
    coco = json.loads(path.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if key not in coco or not isinstance(coco[key], list):
            raise MergeError(f"{path}: not a COCO detection file (missing {key})")
    return coco


def merge_coco(sources: list[MergeSource]) -> tuple[dict, list[str]]:
    """Merge sources into one COCO dict; returns (merged, per-source report lines)."""
    if not sources:
        raise MergeError("no sources to merge")
    categories: list[dict] | None = None
    images: list[dict] = []
    annotations: list[dict] = []
    lines: list[str] = []
    next_image_id = 1
    next_ann_id = 1
    for src in sources:
        coco = _load(src.path)
        if categories is None:
            categories = coco["categories"]
        elif coco["categories"] != categories:
            raise MergeError(
                f"{src.path}: category table {coco['categories']!r} differs from the "
                f"first source's {categories!r} - refusing a mixed-convention merge"
            )
        wanted_images = coco["images"]
        if src.frame_stride is not None:
            if src.frame_stride < 1:
                raise MergeError(f"{src.path}: frame_stride must be >= 1")
            missing = [i["id"] for i in wanted_images if "source_frame" not in i]
            if missing:
                raise MergeError(
                    f"{src.path}: frame_stride filtering needs source_frame on every "
                    f"image; {len(missing)} images lack it"
                )
            wanted_images = [i for i in wanted_images if i["source_frame"] % src.frame_stride == 0]
        id_map: dict[int, int] = {}
        for img in wanted_images:
            new = dict(img)
            id_map[img["id"]] = next_image_id
            new["id"] = next_image_id
            if src.prefix:
                new["file_name"] = f"{src.prefix}/{img['file_name']}"
            images.append(new)
            next_image_id += 1
        kept_anns = 0
        for ann in coco["annotations"]:
            if ann["image_id"] not in id_map:
                continue  # annotation of a stride-filtered image
            new = dict(ann)
            new["id"] = next_ann_id
            new["image_id"] = id_map[ann["image_id"]]
            annotations.append(new)
            next_ann_id += 1
            kept_anns += 1
        lines.append(
            f"{src.path.name}: {len(wanted_images)} images / {kept_anns} boxes"
            + (f" (source_frame % {src.frame_stride} == 0)" if src.frame_stride else "")
        )
    assert categories is not None
    merged = {
        "info": {"description": "vividly v2 merged person manifest"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return merged, lines
