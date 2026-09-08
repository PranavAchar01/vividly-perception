"""scripts/reid/mevid_dataset.py: the four ways a MEVID loader is silently wrong.

Every assertion here is on a failure mode that produces a model, not a crash:

  1. the inclusive end offset, reading `names[start:end]` instead of `names[start:end+1]`
     drops exactly one frame from every tracklet in the corpus and nothing complains;
  2. pid relabelling: a classifier head sized to raw upstream ids, or ordered by file
     order rather than by id, trains fine and means something different on every run;
  3. the query/gallery partition: query_IDX.txt holds line offsets, so an off-by-one or a
     set/list confusion moves tracklets between the two halves of the benchmark;
  4. the outfit lookup: the whole reason MEVID was chosen is clothing change, and an
     outfit column read from the wrong position turns the outfit-aware evaluation into
     noise that still prints a plausible number.

The suite runs against a synthetic eleven-chip corpus built in tmp_path, six train and
five test, written as real 4x8 JPEGs, so it needs torchreid but no MEVID bytes.
`importorskip` at module scope is what lets the mac dev fork, torch-free by design,
collect and skip this file cleanly instead of erroring at import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torchreid")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "reid" / "mevid_dataset.py"


def _load_script() -> ModuleType:
    """scripts/ is not a package, so load the file by path (same as test_export_osnet)."""
    spec = importlib.util.spec_from_file_location("mevid_dataset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_script()

# --- the synthetic corpus -------------------------------------------------------------
# Six train chips in two tracklets, five test chips in two tracklets, one query line.
# The two test tracklets are the same person (pid 102) in two different outfits on two
# different cameras, which is the exact configuration the outfit-aware evaluation exists
# for: without the outfit column they are indistinguishable rows.
TRAIN_NAMES = [
    "0201O003C330T000F00000.jpg",
    "0201O003C330T000F00001.jpg",
    "0201O003C330T000F00002.jpg",
    "0405O001C336T001F00000.jpg",
    "0405O001C336T001F00001.jpg",
    "0405O001C336T001F00002.jpg",
]
# start end pid outfit camera. Both rows are 3 chips long only under an inclusive end.
TRAIN_INFO = "0 2 201 3 330\n3 5 405 1 336\n"

TEST_NAMES = [
    "0102O002C420T000F00000.jpg",
    "0102O002C420T000F00001.jpg",
    "0102O002C420T000F00002.jpg",
    "0102O005C424T001F00000.jpg",
    "0102O005C424T001F00001.jpg",
]
# Deliberately uneven: 3 chips then 2, so a fixed-length assumption cannot pass by luck.
TEST_INFO = "0 2 102 2 420\n3 4 102 5 424\n"
QUERY_IDX = "0\n"


def _write_chip(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 8), color=(11, 22, 33)).save(path, format="JPEG")


@pytest.fixture
def mevid_root(tmp_path: Path) -> Path:
    """A minimal but structurally faithful MEVID release, MARS-style `<pid>/` subdirs."""
    root = tmp_path / "mevid"
    for name in TRAIN_NAMES:
        _write_chip(root / "bbox_train" / name[:4] / name)
    for name in TEST_NAMES:
        _write_chip(root / "bbox_test" / name[:4] / name)
    (root / "train_name.txt").write_text("\n".join(TRAIN_NAMES) + "\n", encoding="utf-8")
    (root / "test_name.txt").write_text("\n".join(TEST_NAMES) + "\n", encoding="utf-8")
    (root / "track_train_info.txt").write_text(TRAIN_INFO, encoding="utf-8")
    (root / "track_test_info.txt").write_text(TEST_INFO, encoding="utf-8")
    (root / "query_IDX.txt").write_text(QUERY_IDX, encoding="utf-8")
    return root


@pytest.fixture
def dataset(mevid_root: Path):
    """The loader, constructed the way torchreid's VideoDataManager constructs it."""
    from torchvision import transforms

    return M.MEVID(
        root=str(mevid_root),
        transform=transforms.Compose([transforms.Resize((8, 4)), transforms.ToTensor()]),
        mode="train",
        seq_len=2,
        sample_method="evenly",
        verbose=False,
    )


# --- filename decoding ----------------------------------------------------------------
def test_chip_name_decodes_all_five_fields() -> None:
    """The README's own example chip, field by field. Indices start at 0 upstream."""
    chip = M.parse_chip_name("0201O003C330T004F00192.jpg")
    assert chip == M.ChipName(pid=201, outfit_id=3, camid=330, tracklet=4, frame=192)
    assert M.parse_chip_name("0201O003C330T004F00192") == chip  # suffix optional


@pytest.mark.parametrize(
    "bad", ["0201O003C330T004.jpg", "not-a-chip.jpg", "0201X003C330T004F00192.jpg", ""]
)
def test_malformed_chip_names_are_refused(bad: str) -> None:
    with pytest.raises(M.MevidFormatError):
        M.parse_chip_name(bad)


# --- the inclusive end offset ---------------------------------------------------------
def test_end_index_is_inclusive(dataset) -> None:
    """`0 2 ...` is a three-chip tracklet.

    The upstream README's worked example is "0 9 103 20 424", described as a tracklet that
    "starts at line #0 and ends at image #9", which is ten chips. An exclusive read gives
    nine, and every tracklet in the corpus quietly loses its last frame.
    """
    lengths = [len(row[0]) for row in dataset.train]
    assert lengths == [3, 3], "train rows 0-2 and 3-5 are three chips each under an inclusive end"
    assert [len(row[0]) for row in dataset.query] == [3]
    assert [len(row[0]) for row in dataset.gallery] == [2], "test row `3 4` is two chips"
    # And the chips are the right ones, in file order, not merely the right count.
    assert [Path(p).name for p in dataset.train[0][0]] == TRAIN_NAMES[0:3]
    assert [Path(p).name for p in dataset.train[1][0]] == TRAIN_NAMES[3:6]


def test_every_listed_chip_is_used_exactly_once(dataset) -> None:
    """The two train tracklets between them consume all six name-file lines."""
    used = [Path(p).name for row in dataset.train for p in row[0]]
    assert sorted(used) == sorted(TRAIN_NAMES)


# --- counts and the split ------------------------------------------------------------
def test_tracklet_counts(dataset) -> None:
    assert len(dataset.train) == 2
    assert len(dataset.query) == 1
    assert len(dataset.gallery) == 1
    # query + gallery partition the test table; nothing is duplicated or lost.
    assert len(dataset.query) + len(dataset.gallery) == len(TEST_INFO.strip().splitlines())


def test_query_gallery_split_follows_query_idx(dataset) -> None:
    """query_IDX.txt names line offsets into track_test_info.txt; the rest is gallery."""
    assert dataset.query_indices == [0]
    # Row 0 is camera 420, row 1 is camera 424: the split landed on the right rows.
    assert dataset.query[0][2] == 420
    assert dataset.gallery[0][2] == 424


def test_query_selection_is_positional_not_by_pid(tmp_path: Path, mevid_root: Path) -> None:
    """Selecting row 1 instead of row 0 must move the second tracklet to query.

    Both test rows share pid 102, so a loader that mistook query_IDX for an identity list
    would produce the same split either way and this is the only assertion that notices.
    """
    from torchvision import transforms

    (mevid_root / "query_IDX.txt").write_text("1\n", encoding="utf-8")
    ds = M.MEVID(
        root=str(mevid_root),
        transform=transforms.ToTensor(),
        mode="train",
        verbose=False,
    )
    assert ds.query[0][2] == 424
    assert ds.gallery[0][2] == 420


def test_out_of_range_query_index_is_refused(mevid_root: Path) -> None:
    from torchvision import transforms

    (mevid_root / "query_IDX.txt").write_text("7\n", encoding="utf-8")
    with pytest.raises(M.MevidFormatError, match="only 2 rows"):
        M.MEVID(root=str(mevid_root), transform=transforms.ToTensor(), verbose=False)


# --- pid relabelling ------------------------------------------------------------------
def test_train_pids_are_relabelled_to_a_contiguous_range(dataset) -> None:
    """0..N-1, ascending by upstream id, which is the classifier head's column order.

    Sorted rather than first-seen so two runs over the same corpus produce checkpoints
    whose logits mean the same thing.
    """
    assert dataset.pid_relabel == {201: 0, 405: 1}
    assert sorted(row[1] for row in dataset.train) == [0, 1]
    assert dataset.num_train_pids == 2
    assert dataset.train[0][1] == 0  # pid 201, the lower upstream id
    assert dataset.train[1][1] == 1  # pid 405


def test_test_pids_are_left_raw(dataset) -> None:
    """torchreid never relabels the test side, and neither does this loader: query and gallery pids
    are compared to each other, not to the training head."""
    assert dataset.query[0][1] == 102
    assert dataset.gallery[0][1] == 102


def test_camids_are_the_raw_camera_numbers(dataset) -> None:
    """424 is MEVA's camera G424, not a 0..K index. Remapping would break the per-camera
    debias pass, which groups on exactly this integer."""
    assert {row[2] for row in dataset.train} == {330, 336}
    assert {row[2] for row in dataset.query} | {row[2] for row in dataset.gallery} == {
        420,
        424,
    }


# --- the outfit lookup ----------------------------------------------------------------
def test_outfit_lookup_by_tracklet_index(dataset) -> None:
    """{line offset in track_*_info.txt -> outfit_id}, the key space query_IDX indexes."""
    assert dataset.train_outfit_by_track == {0: 3, 1: 1}
    assert dataset.test_outfit_by_track == {0: 2, 1: 5}


def test_split_metadata_is_positionally_aligned(dataset) -> None:
    """meta[i] describes split[i]: pid and camid must agree with the tuple they sit beside,
    which is what makes it safe for the trainer to zip them against extracted features."""
    for split, meta in (
        (dataset.train, dataset.train_meta),
        (dataset.query, dataset.query_meta),
        (dataset.gallery, dataset.gallery_meta),
    ):
        assert len(split) == len(meta)
        for entry, row in zip(split, meta, strict=True):
            assert (row.pid, row.camid) == (entry[1], entry[2])


def test_outfits_distinguish_two_tracklets_of_one_person(dataset) -> None:
    """The MEVID case that matters: same pid, different clothes, different camera."""
    assert dataset.query_meta[0].pid == dataset.gallery_meta[0].pid == 102
    assert dataset.query_meta[0].outfit_id == 2
    assert dataset.gallery_meta[0].outfit_id == 5
    assert dataset.query_meta[0].outfit_id != dataset.gallery_meta[0].outfit_id


def test_meta_for_names_the_three_splits(dataset) -> None:
    assert M.MEVID  # the public name is the constructible one
    assert dataset.meta_for("query") is dataset.query_meta
    assert dataset.meta_for("gallery") is dataset.gallery_meta
    assert dataset.meta_for("train") is dataset.train_meta
    with pytest.raises(ValueError, match="unknown mode"):
        dataset.meta_for("val")


# --- layout tolerance and root resolution ---------------------------------------------
def test_root_may_be_the_parent_directory(mevid_root: Path) -> None:
    """torchreid's VideoDataManager passes a parent root by convention; Vividly configs
    name the dataset directory. Both have to work, or one of the two callers is broken."""
    assert M.resolve_dataset_root(mevid_root) == mevid_root.resolve()
    assert M.resolve_dataset_root(mevid_root.parent) == mevid_root.resolve()


def test_missing_release_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(M.MevidFormatError, match="no MEVID release"):
        M.resolve_dataset_root(tmp_path / "nothing-here")


def test_flat_bbox_layout_still_resolves(tmp_path: Path) -> None:
    """Chips directly under bbox_train/ rather than in `<pid>/` subdirectories.

    The MARS-style subdirectory is a fast path, not a requirement: a release that unpacks
    flat must load rather than raise, because the failure would surface as a missing-file
    error hundreds of gigabytes into a download.
    """
    from torchvision import transforms

    root = tmp_path / "flat"
    for name in TRAIN_NAMES:
        _write_chip(root / "bbox_train" / name)
    for name in TEST_NAMES:
        _write_chip(root / "bbox_test" / name)
    (root / "train_name.txt").write_text("\n".join(TRAIN_NAMES) + "\n", encoding="utf-8")
    (root / "test_name.txt").write_text("\n".join(TEST_NAMES) + "\n", encoding="utf-8")
    (root / "track_train_info.txt").write_text(TRAIN_INFO, encoding="utf-8")
    (root / "track_test_info.txt").write_text(TEST_INFO, encoding="utf-8")
    (root / "query_IDX.txt").write_text(QUERY_IDX, encoding="utf-8")

    ds = M.MEVID(root=str(root), transform=transforms.ToTensor(), verbose=False)
    assert [len(row[0]) for row in ds.train] == [3, 3]


# --- malformed tables ------------------------------------------------------------------
def test_track_info_needs_five_columns(mevid_root: Path) -> None:
    (mevid_root / "track_train_info.txt").write_text("0 2 201 3\n", encoding="utf-8")
    with pytest.raises(M.MevidFormatError, match="5 integer columns"):
        M.read_track_info(mevid_root / "track_train_info.txt")


def test_reversed_offsets_are_refused(mevid_root: Path) -> None:
    (mevid_root / "track_train_info.txt").write_text("5 2 201 3 330\n", encoding="utf-8")
    with pytest.raises(M.MevidFormatError, match="precedes start_index"):
        M.read_track_info(mevid_root / "track_train_info.txt")


def test_release_float_notation_reads_as_the_same_rows(mevid_root: Path) -> None:
    """The v1 release writes every index table in numpy's default float notation.

    On disk `track_train_info.txt` opens with
    `0.000000000000000000e+00 9.000000000000000000e+00 1.030000000000000000e+02 ...`, not
    with the README's `0 9 103 20 424`. Both spellings must read as identical rows, and
    `query_IDX.txt` in the same notation must give the same offsets."""
    plain = M.read_track_info(mevid_root / "track_train_info.txt")
    as_floats = (
        "\n".join(
            " ".join(f"{float(v):.18e}" for v in line.split()) for line in TRAIN_INFO.splitlines()
        )
        + "\n"
    )
    assert as_floats.startswith("0.000000000000000000e+00 2.000000000000000000e+00 ")
    (mevid_root / "track_train_info.txt").write_text(as_floats, encoding="utf-8")
    assert M.read_track_info(mevid_root / "track_train_info.txt") == plain
    (mevid_root / "query_IDX.txt").write_text(
        "1.752000000000000000e+03\n0.000000000000000000e+00\n", encoding="utf-8"
    )
    assert M.read_query_indices(mevid_root / "query_IDX.txt") == [1752, 0]


def test_fractional_index_values_are_refused(mevid_root: Path) -> None:
    """A float that is not integral cannot be an offset, an id or a camera number."""
    (mevid_root / "track_train_info.txt").write_text("0 2.5 201 3 330\n", encoding="utf-8")
    with pytest.raises(M.MevidFormatError, match="non-integer column"):
        M.read_track_info(mevid_root / "track_train_info.txt")
    (mevid_root / "query_IDX.txt").write_text("0.5\n", encoding="utf-8")
    with pytest.raises(M.MevidFormatError, match="integers"):
        M.read_query_indices(mevid_root / "query_IDX.txt")


# start end pid outfit camera, with a zero-length row in the middle: end is start minus one.
TRAIN_INFO_WITH_EMPTY_ROW = "0 2 201 3 330\n3 2 999 9 999\n3 5 405 1 336\n"


def test_zero_length_row_is_skipped_but_keeps_its_offset(mevid_root: Path) -> None:
    """The release carries one zero-length row (train line 1889, end = start - 1).

    It must contribute no tracklet and no classifier column, while every row after it keeps
    the line offset that query_IDX.txt and the outfit lookup address it by."""
    from torchvision import transforms

    (mevid_root / "track_train_info.txt").write_text(TRAIN_INFO_WITH_EMPTY_ROW, encoding="utf-8")
    rows = M.read_track_info(mevid_root / "track_train_info.txt")
    assert [r.n_chips for r in rows] == [3, 0, 3]
    ds = M.MEVID(root=str(mevid_root), transform=transforms.ToTensor(), verbose=False)
    assert [len(t[0]) for t in ds.train] == [3, 3]
    assert ds.num_train_pids == 2
    assert 999 not in ds.pid_relabel
    assert ds.train_outfit_by_track == {0: 3, 1: 9, 2: 1}


def test_zero_length_test_row_keeps_query_offsets(mevid_root: Path) -> None:
    """An empty test row must not shift the offsets query_IDX.txt selects on."""
    from torchvision import transforms

    (mevid_root / "track_test_info.txt").write_text(
        "0 2 102 2 420\n3 2 102 7 999\n3 4 102 5 424\n", encoding="utf-8"
    )
    (mevid_root / "query_IDX.txt").write_text("2\n", encoding="utf-8")
    ds = M.MEVID(root=str(mevid_root), transform=transforms.ToTensor(), verbose=False)
    assert len(ds.query) == 1
    assert len(ds.gallery) == 1
    assert ds.query_meta[0].outfit_id == 5
    assert ds.gallery_meta[0].outfit_id == 2


def test_name_list_drops_the_trailing_blank_line(mevid_root: Path) -> None:
    """A trailing newline is not a chip. Keeping it as an empty name would shift every
    offset after it, which is the quietest possible way to corrupt the whole corpus."""
    assert M.read_name_list(mevid_root / "train_name.txt") == TRAIN_NAMES


# --- torchreid integration -------------------------------------------------------------
def test_dataset_is_a_torchreid_video_dataset(dataset) -> None:
    from torchreid.data.datasets import VideoDataset

    assert isinstance(dataset, VideoDataset)


def test_getitem_returns_a_seq_len_clip(dataset) -> None:
    """torchreid hands the engine (seq_len, C, H, W); the trainer's reshape depends on it."""
    item = dataset[0]
    assert item["pid"] == 0
    assert item["camid"] == 330
    assert tuple(item["img"].shape) == (2, 3, 8, 4)  # seq_len=2 from the fixture


def test_register_mevid_is_idempotent() -> None:
    """A resumed run, or a second manager in one process, must not crash on re-registration;
    torchreid's own register_video_dataset raises on a duplicate name."""
    from torchreid.data.datasets import __video_datasets as registry

    name = "mevid_test_registration"
    try:
        M.register_mevid(name)
        M.register_mevid(name)
        assert registry[name] is M.MEVID
        with pytest.raises(ValueError, match="already has a different"):
            registry[name] = object
            M.register_mevid(name)
    finally:
        registry.pop(name, None)
