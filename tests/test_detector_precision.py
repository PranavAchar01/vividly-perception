"""I1 precision instrument: matching, per-camera split, markdown (no models, no GPU).

The fixture is hand-built so every TP/FP/FN below is countable by eye. Boxes are
COCO xywh. Two cameras under one scene, four images, deliberately including an
empty-detection image (FP/img denominator) and an `iscrowd` ignore region.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eval" / "detector_precision_at_thr.py"


def _load_script() -> ModuleType:
    """scripts/ is not a package, so load the file by path.

    The module must be in sys.modules before exec_module: @dataclass resolves
    annotations through sys.modules[cls.__module__] and raises otherwise.
    """
    spec = importlib.util.spec_from_file_location("detector_precision_at_thr", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


F = _load_script()

CAM_A = "Warehouse_005/Camera_0000"
CAM_B = "Warehouse_005/Camera_0001"

# image 1 (cam A): 2 GT. det at .90 hits GT1 exactly (TP), det at .45 overlaps GT2 below
#                  IoU 0.5 and is a false positive until the score cut drops it, det at
#                  .35 is nowhere (FP).
# image 2 (cam A): 1 GT, no detections -> 1 FN, and it still counts as an image.
# image 3 (cam B): 1 GT + 1 iscrowd ignore region. det .80 hits the GT (TP);
#                  det .60 lands inside the crowd region only -> ignored, not FP.
# image 4 (cam B): 0 GT, 1 det at .50 -> pure FP.
GT = {
    "images": [
        {"id": 1, "file_name": f"{CAM_A}/000001.jpg", "width": 200, "height": 200},
        {"id": 2, "file_name": f"{CAM_A}/000002.jpg", "width": 200, "height": 200},
        {"id": 3, "file_name": f"{CAM_B}/000003.jpg", "width": 200, "height": 200},
        {"id": 4, "file_name": f"{CAM_B}/000004.jpg", "width": 200, "height": 200},
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 10, 20, 20],
            "area": 400,
            "iscrowd": 0,
        },
        # GT2 is 20x20 at (60,60); the .45 detection below is 20x20 at (60,60) shifted
        # so that intersection/union is 196/604, about 0.32, below 0.5 and so a miss at
        # IoU 0.5.
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [60, 60, 20, 20],
            "area": 400,
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 1,
            "bbox": [10, 10, 20, 20],
            "area": 400,
            "iscrowd": 0,
        },
        {
            "id": 4,
            "image_id": 3,
            "category_id": 1,
            "bbox": [10, 10, 20, 20],
            "area": 400,
            "iscrowd": 0,
        },
        {
            "id": 5,
            "image_id": 3,
            "category_id": 1,
            "bbox": [100, 100, 60, 60],
            "area": 3600,
            "iscrowd": 1,
        },
    ],
    "categories": [{"id": 1, "name": "person"}],
}

DETS = [
    {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.90},
    {"image_id": 1, "category_id": 1, "bbox": [66, 66, 20, 20], "score": 0.45},
    {"image_id": 1, "category_id": 1, "bbox": [150, 150, 20, 20], "score": 0.35},
    {"image_id": 3, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.80},
    {"image_id": 3, "category_id": 1, "bbox": [110, 110, 20, 20], "score": 0.60},
    {"image_id": 4, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.50},
]


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    gt = tmp_path / "gt.json"
    det = tmp_path / "det.json"
    gt.write_text(json.dumps(GT))
    det.write_text(json.dumps(DETS))
    return gt, det


# ---------------------------------------------------------------- camera_key


@pytest.mark.parametrize(
    ("file_name", "depth", "expected"),
    [
        ("Warehouse_005/Camera_0000/000123.jpg", 2, "Warehouse_005/Camera_0000"),
        ("Warehouse_005/Camera_0000/000123.jpg", 1, "Warehouse_005"),
        ("Warehouse_005/Camera_0000/000123.jpg", 9, "Warehouse_005/Camera_0000"),
        ("Camera_0000/000123.jpg", 2, "Camera_0000"),
        ("000123.jpg", 2, "(unknown)"),
        ("./Warehouse_005/Camera_0000/f.jpg", 2, "Warehouse_005/Camera_0000"),
    ],
)
def test_camera_key(file_name: str, depth: int, expected: str) -> None:
    assert F.camera_key(file_name, depth) == expected


# ---------------------------------------------------------------- matching


def test_fixture_iou_is_what_the_comments_claim(pair) -> None:
    """Guard the fixture itself: the .45 box must sit below IoU 0.5, not at it."""
    gt_path, det_path = pair
    per_image, _ = F.load_pair(gt_path, det_path)
    ious = F.box_ious([[66, 66, 20, 20]], [[60, 60, 20, 20]], [0])
    assert 0.30 < float(ious[0, 0]) < 0.50
    assert len(per_image) == 4


def test_counts_at_each_threshold(pair) -> None:
    gt_path, det_path = pair
    per_image, file_names = F.load_pair(gt_path, det_path)
    tables = F.grade(per_image, file_names, thresholds=(0.3, 0.4, 0.5), iou_thr=0.5)

    # thr 0.3: dets .90 TP, .45 FP (IoU too low), .35 FP, .80 TP, .60 ignored, .50 FP
    t = tables[0.3]["OVERALL"]
    assert (t.tp, t.fp, t.fn) == (2, 3, 2)
    assert t.n_images == 4
    assert t.n_gt == 4  # the iscrowd region is not in the recall denominator
    assert t.n_det == 5  # the crowd-matched detection left the tally
    assert t.fp_per_image == pytest.approx(3 / 4)
    assert t.precision == pytest.approx(2 / 5)
    assert t.recall == pytest.approx(2 / 4)

    # thr 0.4 drops the .35 FP only
    t = tables[0.4]["OVERALL"]
    assert (t.tp, t.fp, t.fn) == (2, 2, 2)
    assert t.precision == pytest.approx(0.5)

    # thr 0.5 additionally drops the .45 FP; the .50 FP survives (>= is inclusive)
    t = tables[0.5]["OVERALL"]
    assert (t.tp, t.fp, t.fn) == (2, 1, 2)
    assert t.precision == pytest.approx(2 / 3)
    assert t.fp_per_image == pytest.approx(1 / 4)
    assert t.f1 == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))


def test_per_camera_split_sums_to_overall(pair) -> None:
    gt_path, det_path = pair
    per_image, file_names = F.load_pair(gt_path, det_path)
    table = F.grade(per_image, file_names, thresholds=(0.4,), iou_thr=0.5)[0.4]

    assert set(table) == {"OVERALL", CAM_A, CAM_B}
    a, b, o = table[CAM_A], table[CAM_B], table["OVERALL"]
    # cam A: images 1-2 hold 3 GT; only GT1 is matched -> 2 missed
    assert (a.tp, a.fp, a.fn, a.n_images, a.n_gt) == (1, 1, 2, 2, 3)
    # cam B: image 3's single non-crowd GT is matched, image 4 has none
    assert (b.tp, b.fp, b.fn, b.n_images, b.n_gt) == (1, 1, 0, 2, 1)
    for attr in ("tp", "fp", "fn", "n_images", "n_gt", "n_det"):
        assert getattr(a, attr) + getattr(b, attr) == getattr(o, attr)


def test_ignore_region_absorbs_a_detection_instead_of_scoring_it(pair) -> None:
    """Drop the iscrowd flag and the same detection becomes a plain FP."""
    gt_path, det_path = pair
    strict = json.loads(gt_path.read_text())
    for ann in strict["annotations"]:
        ann["iscrowd"] = 0
    strict_path = gt_path.with_name("gt_nocrowd.json")
    strict_path.write_text(json.dumps(strict))

    per_image, file_names = F.load_pair(strict_path, det_path)
    t = F.grade(per_image, file_names, thresholds=(0.3,), iou_thr=0.5)[0.3]["OVERALL"]
    assert (t.tp, t.fp) == (2, 4)  # was (2, 3)
    assert t.n_gt == 5  # the former ignore region now demands recall


def test_higher_iou_threshold_unmatches_the_loose_box(pair) -> None:
    gt_path, det_path = pair
    per_image, file_names = F.load_pair(gt_path, det_path)
    loose = F.grade(per_image, file_names, thresholds=(0.3,), iou_thr=0.3)[0.3]["OVERALL"]
    assert loose.tp == 3  # the .45 box now matches GT2
    tight = F.grade(per_image, file_names, thresholds=(0.3,), iou_thr=0.95)[0.3]["OVERALL"]
    assert tight.tp == 2  # exact-overlap boxes still match at 0.95


def test_max_dets_cap_hides_low_score_detections(pair) -> None:
    gt_path, det_path = pair
    per_image, file_names = F.load_pair(gt_path, det_path)
    capped = F.grade(per_image, file_names, thresholds=(0.3,), iou_thr=0.5, max_dets=1)[0.3]
    # image 1 keeps only the .90 TP; image 3 keeps only the .80 TP
    assert (capped["OVERALL"].tp, capped["OVERALL"].fp) == (2, 1)


# ---------------------------------------------------------------- loaders / IO


def test_load_pair_accepts_a_wrapped_results_dict(tmp_path: Path, pair) -> None:
    gt_path, _ = pair
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"annotations": DETS}))
    per_image, _ = F.load_pair(gt_path, wrapped)
    assert sum(len(v.boxes_dt) for v in per_image.values()) == len(DETS)


def test_load_pair_rejects_a_non_list_payload(tmp_path: Path, pair) -> None:
    gt_path, _ = pair
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"unexpected": 1}))
    with pytest.raises(ValueError, match="results LIST"):
        F.load_pair(gt_path, bad)


def test_detections_are_sorted_by_score_descending(pair) -> None:
    gt_path, det_path = pair
    per_image, _ = F.load_pair(gt_path, det_path)
    scores = per_image[1].scores_dt
    assert scores == sorted(scores, reverse=True)


def test_category_filter_drops_foreign_classes(tmp_path: Path, pair) -> None:
    gt_path, _ = pair
    mixed = tmp_path / "mixed.json"
    mixed.write_text(
        json.dumps([*DETS, {"image_id": 1, "category_id": 7, "bbox": [0, 0, 5, 5], "score": 0.99}])
    )
    per_image, _ = F.load_pair(gt_path, mixed, category_ids=[1])
    assert sum(len(v.boxes_dt) for v in per_image.values()) == len(DETS)


def test_markdown_labels_every_row_and_flags_the_demo_point(pair) -> None:
    gt_path, det_path = pair
    per_image, file_names = F.load_pair(gt_path, det_path)
    tables = F.grade(per_image, file_names, thresholds=(0.3, 0.4, 0.5), iou_thr=0.5)
    md = F.to_markdown(
        tables, {"gt": str(gt_path), "det": str(det_path), "iou": 0.5}, label="synthetic-devval"
    )

    assert "the demo point" in md
    assert "## score >= 0.4" in md
    body = [ln for ln in md.splitlines() if ln.startswith("| ") and "camera |" not in ln]
    assert body, "no data rows rendered"
    assert all("synthetic-devval" in ln for ln in body)
    assert md.count(CAM_A) == 3  # one row per threshold
    assert "no sidecar found" in md  # no sidecar next to the fixture


def test_cli_writes_both_artifacts_and_reads_the_sidecar(tmp_path: Path, pair, capsys) -> None:
    gt_path, det_path = pair
    det_path.with_suffix(".json.sidecar.json").write_text(
        json.dumps({"checkpoint_sha256": "deadbeef" * 8})
    )
    out_md, out_json = tmp_path / "r.md", tmp_path / "r.json"
    rc = F.main(
        [
            "--gt",
            str(gt_path),
            "--det",
            str(det_path),
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
            "--title",
            "fixture",
        ]
    )
    assert rc == F.EXIT_OK
    assert out_md.is_file() and out_json.is_file()
    payload = json.loads(out_json.read_text())
    assert payload["meta"]["checkpoint_sha256"] == "deadbeef" * 8
    assert set(payload["tables"]) == {"0.3", "0.4", "0.5"}
    assert payload["tables"]["0.5"]["OVERALL"]["tp"] == 2
    assert ("deadbeef" * 8) in out_md.read_text()


def test_cli_rejects_a_missing_file_and_a_bad_iou(tmp_path: Path, pair) -> None:
    gt_path, det_path = pair
    assert F.main(["--gt", str(tmp_path / "nope.json"), "--det", str(det_path)]) == F.EXIT_CONFIG
    assert F.main(["--gt", str(gt_path), "--det", str(det_path), "--iou", "0"]) == F.EXIT_CONFIG
