"""MTMC -> COCO converter pins (v2 training corpus, research/v2-training-stack.md
section 7e).

Pure-logic tests run everywhere; the end-to-end test synthesizes a tiny mp4 with
ffmpeg (the same binary the converter shells out to) and is skipped where ffmpeg is
absent.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vividly.data.convert.mtmc_to_coco import (
    CameraPlan,
    ExtractionMismatch,
    MTMCFormatError,
    SceneReport,
    build_scene_coco,
    convert_scene,
    load_ground_truth,
    sample_frames,
)

W, H = 64, 48


def _gt(n_frames: int = 12) -> dict[int, list[dict]]:
    """Two persons and one forklift; person 2 exits the frame edge on odd frames."""
    gt: dict[int, list[dict]] = {}
    for f in range(n_frames):
        objs = [
            {
                "object type": "Person",
                "object id": 1,
                "2d bounding box visible": {"Camera_0000": [10, 10, 30, 40]},
            },
            {
                "object type": "Forklift",
                "object id": 9,
                "2d bounding box visible": {"Camera_0000": [0, 0, 20, 20]},
            },
        ]
        if f % 2 == 1:
            objs.append(
                {
                    "object type": "Person",
                    "object id": 2,
                    # runs past the right edge: must be clamped to width
                    "2d bounding box visible": {"Camera_0000": [50, 5, 80, 45]},
                }
            )
        gt[f] = objs
    return gt


def test_sample_frames_is_a_uniform_stride_from_zero() -> None:
    assert sample_frames(list(range(10)), 3) == [0, 3, 6, 9]
    assert sample_frames(list(range(10)), 1) == list(range(10))
    with pytest.raises(ValueError):
        sample_frames([0, 1], 0)


def test_build_coco_filters_robots_clamps_edges_and_keeps_negatives() -> None:
    gt = _gt()
    cam = CameraPlan("Camera_0000", Path("x.mp4"), W, H, 12)
    report = SceneReport(scene="s")
    coco = build_scene_coco(gt, [cam], [0, 1], "s", ("Person",), 2.0, True, report)
    assert [c["name"] for c in coco["categories"]] == ["person"]
    # v2 corpus law: person is category id 0 (both trainers index by raw category_id
    # when remap_mscoco_category is off; a 1-based id device-asserts at num_classes=1)
    assert coco["categories"][0]["id"] == 0
    assert {a["category_id"] for a in coco["annotations"]} == {0}
    # frame 0: one person; frame 1: two persons; the forklift never appears
    assert len(coco["annotations"]) == 3
    assert report.object_types_seen["Forklift"] == 2
    # the edge person's box is clamped to the frame, never past it
    clamped = coco["annotations"][-1]["bbox"]
    assert clamped[0] + clamped[2] <= W
    # negatives kept: an image row exists even for a frame with no boxes
    empty = build_scene_coco({0: []}, [cam], [0], "s", ("Person",), 2.0, True)
    assert len(empty["images"]) == 1 and len(empty["annotations"]) == 0
    # and dropped when keep_empty=False
    none = build_scene_coco({0: []}, [cam], [0], "s", ("Person",), 2.0, False)
    assert len(none["images"]) == 0


def test_file_name_encodes_camera_and_source_frame() -> None:
    cam = CameraPlan("Camera_0007", Path("x.mp4"), W, H, 12)
    coco = build_scene_coco(_gt(), [cam], [0, 3], "s", ("Person",), 2.0, True)
    names = [i["file_name"] for i in coco["images"]]
    assert names == ["Camera_0007/000000.jpg", "Camera_0007/000003.jpg"]
    assert [i["source_frame"] for i in coco["images"]] == [0, 3]


def test_ground_truth_loader_rejects_wrong_shapes(tmp_path: Path) -> None:
    p = tmp_path / "gt.json"
    p.write_text("[]")
    with pytest.raises(MTMCFormatError):
        load_ground_truth(p)
    p.write_text('{"not_an_int": []}')
    with pytest.raises(MTMCFormatError):
        load_ground_truth(p)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_convert_scene_end_to_end(tmp_path: Path) -> None:
    scene = tmp_path / "Lab_042"
    (scene / "videos").mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={W}x{H}:rate=6:duration=2",
            "-pix_fmt",
            "yuv420p",
            str(scene / "videos" / "Camera_0000.mp4"),
        ],
        check=True,
    )
    (scene / "ground_truth.json").write_text(json.dumps({str(k): v for k, v in _gt(12).items()}))

    frames_out = tmp_path / "frames"
    ann_out = tmp_path / "ann" / "lab42.json"
    report = convert_scene(
        scene, frames_out, ann_out, stride=4, object_types=("Person",), extract=True
    )
    assert report.sampled_frames == 3  # frames 0, 4, 8
    coco = json.loads(ann_out.read_text())
    assert len(coco["images"]) == 3
    for img in coco["images"]:
        f = frames_out / img["file_name"]
        assert f.is_file(), f"extracted frame missing: {f}"
    # frame files are named by source index, the property the extractor cross-checks
    assert sorted(p.name for p in (frames_out / "Camera_0000").glob("*.jpg")) == [
        "000000.jpg",
        "000004.jpg",
        "000008.jpg",
    ]
    # extraction is resumable: a second run must not re-decode (it would raise on
    # the tmp files if it did) and must produce identical output
    report2 = convert_scene(
        scene, frames_out, ann_out, stride=4, object_types=("Person",), extract=True
    )
    assert report2.images == report.images


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_convert_scene_refuses_video_shorter_than_ground_truth(tmp_path: Path) -> None:
    scene = tmp_path / "Short_000"
    (scene / "videos").mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={W}x{H}:rate=6:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(scene / "videos" / "Camera_0000.mp4"),
        ],
        check=True,
    )
    # 60 GT frames vs a 6-frame video: far outside tolerance, must refuse loudly
    (scene / "ground_truth.json").write_text(json.dumps({str(k): [] for k in range(60)}))
    with pytest.raises(MTMCFormatError, match="tolerance"):
        convert_scene(scene, tmp_path / "f", tmp_path / "a.json", stride=2)


def test_extraction_mismatch_is_a_named_error() -> None:
    # The class the cross-checks raise; pinned so a refactor cannot silently
    # downgrade a corpus-integrity failure to a generic exception.
    assert issubclass(ExtractionMismatch, RuntimeError)
