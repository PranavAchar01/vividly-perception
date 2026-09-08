"""MVOR -> v2 COCO converter pins: the string-category quirk, id preservation,
clamping, negatives, and the missing-frame refusal."""

import json
from pathlib import Path

import pytest

from vividly.data.convert.mvor_to_coco import MVORFormatError, convert_mvor

CAT = [{"id": "1", "name": "person", "supercategory": "person"}]


def _write(tmp_path: Path, images, annotations, categories=CAT) -> Path:
    p = tmp_path / "camma.json"
    p.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    return p


def test_converts_string_category_and_keeps_structured_ids(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        images=[
            {
                "id": 10020000017,
                "file_name": "day1/cam2/color/000017.png",
                "width": 640,
                "height": 480,
            },
            {
                "id": 10020000018,
                "file_name": "day1/cam2/color/000018.png",
                "width": 640,
                "height": 480,
            },
        ],
        annotations=[
            # runs past the frame edge: clamped, kept
            {"id": 1, "image_id": 10020000017, "category_id": "1", "bbox": [600, 400, 80, 120]},
            # sliver: dropped
            {"id": 2, "image_id": 10020000017, "category_id": "1", "bbox": [10, 10, 1, 50]},
        ],
    )
    coco, report = convert_mvor(src)
    assert coco["categories"][0]["id"] == 0
    assert [i["id"] for i in coco["images"]] == [10020000017, 10020000018]
    (ann,) = coco["annotations"]
    assert ann["category_id"] == 0 and ann["image_id"] == 10020000017
    x, y, w, h = ann["bbox"]
    assert x + w <= 640 and y + h <= 480
    assert report.dropped_boxes == 1 and report.negatives == 1  # image 18 kept, empty


def test_unexpected_category_table_refused(tmp_path: Path) -> None:
    src = _write(tmp_path, images=[], annotations=[], categories=[{"id": "2", "name": "chair"}])
    with pytest.raises(MVORFormatError, match="person category"):
        convert_mvor(src)


def test_missing_frames_refuse_a_smaller_exam(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        images=[{"id": 1, "file_name": "day1/cam1/color/000001.png", "width": 64, "height": 48}],
        annotations=[],
    )
    with pytest.raises(MVORFormatError, match="refusing"):
        convert_mvor(src, image_root=tmp_path / "empty_root")
