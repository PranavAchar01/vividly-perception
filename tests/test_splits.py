"""Held-out splits by recording: the partition itself, proved on synthetic annotations.

A split is by recording, never by frame: two frames of the same recording are near duplicates,
so a frame-level partition puts the same scene on both sides and reports a number that measures
nothing. Every assertion here runs on a synthetic COCO document, so the file needs no dataset
bytes and covers the module wherever it is called from.
"""

from __future__ import annotations

import pytest

from vividly.data.splits import (
    SEQ_KEY,
    SplitError,
    SplitSpec,
    canonical_bytes,
    sequences_in,
    sha256_hex,
    split_coco,
    summarize,
)


# --------------------------------------------------------------- fixtures
def _toy_coco(seqs: dict[str, int], *, tracks: bool = True) -> dict:
    images, anns = [], []
    img_id = ann_id = 0
    for seq, n in seqs.items():
        for i in range(n):
            img_id += 1
            images.append(
                {
                    "id": img_id,
                    "file_name": f"{seq}/{i:06d}.jpg",
                    "width": 64,
                    "height": 64,
                    SEQ_KEY: seq,
                }
            )
            for k in range(2):
                ann_id += 1
                a = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [1, 1, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                    SEQ_KEY: seq,
                }
                if tracks:
                    a["vividly_track_id"] = k + 1
                anns.append(a)
    return {
        "info": {"vividly_dataset": "toy"},
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": [{"id": 1, "name": "person"}],
    }


SPEC = SplitSpec(name="fam", dataset="toy", train=("a", "b"), test=("c",))


# --------------------------------------------------------------- pure partition
def test_split_partitions_by_sequence_and_keeps_ids() -> None:
    coco = _toy_coco({"a": 3, "b": 2, "c": 4})
    train, test = split_coco(coco, SPEC, parent_sha256="ab" * 32)
    assert {im[SEQ_KEY] for im in train["images"]} == {"a", "b"}
    assert {im[SEQ_KEY] for im in test["images"]} == {"c"}
    assert len(train["images"]) == 5 and len(test["images"]) == 4
    assert len(train["annotations"]) == 10 and len(test["annotations"]) == 8
    train_ids = {im["id"] for im in train["images"]}
    assert all(a["image_id"] in train_ids for a in train["annotations"])
    assert train_ids.isdisjoint({im["id"] for im in test["images"]})
    # ids are the parent's ids, untouched
    assert sorted(train_ids | {im["id"] for im in test["images"]}) == list(range(1, 10))
    # provenance rides in info, categories are copied, the parent is not mutated
    assert train["info"]["vividly_split"]["side"] == "train"
    assert test["info"]["vividly_split"]["sequences"] == ["c"]
    assert train["info"]["vividly_split"]["parent_sha256"] == "ab" * 32
    assert train["info"]["vividly_split"]["spec"]["name"] == "fam"
    assert train["categories"] == coco["categories"]
    assert "vividly_split" not in coco["info"]


def test_summary_counts_images_annotations_and_sequence_scoped_tracks() -> None:
    coco = _toy_coco({"a": 3, "c": 1})
    s = summarize(coco)
    assert s["images"] == 4 and s["annotations"] == 8
    assert s["sequences"] == {"a": 3, "c": 1}
    assert s["tracks"] == 4  # 2 ids x 2 sequences: tracks are per recording
    assert summarize(_toy_coco({"a": 1}, tracks=False))["tracks"] == 0


def test_sequences_in_requires_the_converter_key() -> None:
    coco = _toy_coco({"a": 1})
    del coco["images"][0][SEQ_KEY]
    with pytest.raises(SplitError, match=SEQ_KEY):
        sequences_in(coco)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (SplitSpec("f", "toy", ("a",), ("a",)), "both train and test"),
        (SplitSpec("f", "toy", ("a", "b", "c"), ()), ">= 1 sequence"),
        (SplitSpec("f", "toy", ("a",), ("c",)), "not assigned"),
        (SplitSpec("f", "toy", ("a", "b"), ("c", "zz")), "does not contain"),
        (SplitSpec("f", "toy", ("a", "a"), ("b", "c")), "duplicate"),
        (SplitSpec("f", "toy", ("a",), ("b",), drop=("b", "c")), "both test and drop"),
    ],
)
def test_spec_validation_refuses_overlap_gaps_and_unknowns(spec: SplitSpec, match: str) -> None:
    coco = _toy_coco({"a": 1, "b": 1, "c": 1})
    with pytest.raises(SplitError, match=match):
        split_coco(coco, spec)


def test_drop_removes_a_sequence_from_both_sides() -> None:
    coco = _toy_coco({"a": 1, "b": 1, "c": 1})
    train, test = split_coco(coco, SplitSpec("f", "toy", ("a",), ("c",), drop=("b",)))
    assert [im[SEQ_KEY] for im in train["images"]] == ["a"]
    assert [im[SEQ_KEY] for im in test["images"]] == ["c"]


def test_canonical_bytes_are_deterministic_and_hashable() -> None:
    coco = _toy_coco({"a": 2, "c": 1})
    a = canonical_bytes(split_coco(coco, SplitSpec("f", "toy", ("a",), ("c",)))[0])
    b = canonical_bytes(split_coco(coco, SplitSpec("f", "toy", ("a",), ("c",)))[0])
    assert a == b and a.endswith(b"\n") and len(sha256_hex(a)) == 64
