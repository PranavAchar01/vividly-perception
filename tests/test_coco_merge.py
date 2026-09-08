"""Manifest-merge pins (v2 training corpus): id remapping, scene prefixing,
dev-val stride thinning, and the mixed-convention refusal."""

import json
from pathlib import Path

import pytest

from vividly.data.convert.coco_merge import MergeError, MergeSource, merge_coco

CAT0 = [{"id": 0, "name": "person", "supercategory": "person"}]


def _scene(tmp_path: Path, name: str, frames: list[int], boxes_per_frame: int = 1) -> Path:
    images, anns = [], []
    for i, f in enumerate(frames, start=1):
        images.append(
            {
                "id": i,
                "file_name": f"Camera_0000/{f:06d}.jpg",
                "width": 64,
                "height": 48,
                "source_frame": f,
            }
        )
        for _ in range(boxes_per_frame):
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": i,
                    "category_id": 0,
                    "bbox": [1, 1, 5, 5],
                    "area": 25,
                    "iscrowd": 0,
                }
            )
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"images": images, "annotations": anns, "categories": CAT0}))
    return p


def test_merge_remaps_ids_and_prefixes_scene(tmp_path: Path) -> None:
    a = _scene(tmp_path, "A", [0, 30])
    b = _scene(tmp_path, "B", [0])
    merged, lines = merge_coco([MergeSource(a, "Warehouse_000"), MergeSource(b, "Warehouse_001")])
    assert [i["id"] for i in merged["images"]] == [1, 2, 3]
    assert merged["images"][0]["file_name"] == "Warehouse_000/Camera_0000/000000.jpg"
    assert merged["images"][2]["file_name"] == "Warehouse_001/Camera_0000/000000.jpg"
    # the third image's annotation follows it to its new id
    assert merged["annotations"][2]["image_id"] == 3
    assert [a_["id"] for a_ in merged["annotations"]] == [1, 2, 3]
    assert len(lines) == 2


def test_frame_stride_thins_to_devval_subset(tmp_path: Path) -> None:
    p = _scene(tmp_path, "W5", [0, 30, 60, 90, 120, 150, 300])
    merged, _ = merge_coco([MergeSource(p, "Warehouse_005", frame_stride=150)])
    kept = [i["source_frame"] for i in merged["images"]]
    assert kept == [0, 150, 300]
    # annotations of dropped frames are gone; survivors point at remapped ids
    assert {a["image_id"] for a in merged["annotations"]} == {1, 2, 3}


def test_stride_needs_source_frame(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "c/0.jpg"}],
                "annotations": [],
                "categories": CAT0,
            }
        )
    )
    with pytest.raises(MergeError, match="source_frame"):
        merge_coco([MergeSource(p, "", frame_stride=150)])


def test_mixed_category_conventions_refused(tmp_path: Path) -> None:
    a = _scene(tmp_path, "A", [0])
    b = tmp_path / "b.json"
    b.write_text(
        json.dumps(
            {
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
            }
        )
    )
    with pytest.raises(MergeError, match="mixed-convention"):
        merge_coco([MergeSource(a, ""), MergeSource(b, "")])


def test_negatives_survive_merge(tmp_path: Path) -> None:
    p = _scene(tmp_path, "N", [0], boxes_per_frame=0)
    merged, _ = merge_coco([MergeSource(p, "S")])
    assert len(merged["images"]) == 1 and len(merged["annotations"]) == 0


def test_non_coco_refused(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{}")
    with pytest.raises(MergeError, match="COCO"):
        merge_coco([MergeSource(p, "")])
