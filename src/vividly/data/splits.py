"""Held-out splits by recording unit for TRAIN_PROTOTYPE sets.

A TRAIN_PROTOTYPE entry may train prototype-class weights, under POLICY.md, rule 6, and a
number measured from them is quoted externally only with counsel sign-off on the entry's
conflict note, under POLICY.md, section 3, prototype corpus admission. A number only means
something if the frames it was measured on were never trained on, and for video data
whether a frame was seen has to be decided per recording (video, or segment and camera),
not per frame: adjacent frames of one video are near-duplicates, so a random frame split
leaks the test set into training almost completely.

This module is the pure part: a `SplitSpec` names which `vividly_sequence` values go to
`train`, which to `test`, and which are dropped, and `split_coco()` partitions one COCO
document into two. `vividly_sequence` is a convention this module reads rather than a key
any converter under `src/vividly/data/convert/` writes today; a caller supplies it on
every image and annotation, or `sequences_in()` refuses the document. Every sequence
present in the document must be assigned to exactly one side, because a sequence the spec
forgot is an error rather than a silent default, and both sides must be non-empty.

Nothing here opens an image, and nothing here knows where the bytes live: the registry
resolves paths and holds them inside the licence bucket, exactly as for every other
dataset accessor.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

#: The per-image and per-annotation key a document must carry to be splittable here.
SEQ_KEY = "vividly_sequence"


class SplitError(ValueError):
    """A spec or document a human fixes; callers map it to their config exit code."""


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """Which sequences of one dataset are train, test, or dropped."""

    name: str  # split family, e.g. "wu_sahzu_exam_v1"
    dataset: str  # registry name
    train: tuple[str, ...]
    test: tuple[str, ...]
    drop: tuple[str, ...] = ()
    seq_key: str = SEQ_KEY

    def validate(self, present: Iterable[str]) -> None:
        """Both sides non-empty, the three lists disjoint, and exactly the present set covered."""
        train, test, drop = set(self.train), set(self.test), set(self.drop)
        if not self.train or not self.test:
            raise SplitError(f"{self.dataset}: train and test must both name >= 1 sequence")
        for a, b, la, lb in (
            (train, test, "train", "test"),
            (train, drop, "train", "drop"),
            (test, drop, "test", "drop"),
        ):
            if a & b:
                raise SplitError(f"{self.dataset}: {sorted(a & b)} listed in both {la} and {lb}")
        for label, seqs in (("train", self.train), ("test", self.test), ("drop", self.drop)):
            if len(set(seqs)) != len(seqs):
                raise SplitError(f"{self.dataset}: duplicate sequence in {label}: {list(seqs)}")
        present_set = set(present)
        assigned = train | test | drop
        missing = sorted(present_set - assigned)
        if missing:
            raise SplitError(
                f"{self.dataset}: sequences present but not assigned to train/test/drop: "
                f"{missing}. Every recording is decided explicitly."
            )
        unknown = sorted(assigned - present_set)
        if unknown:
            raise SplitError(
                f"{self.dataset}: spec names sequences the document does not contain: {unknown}"
            )


def sequences_in(coco: Mapping, seq_key: str = SEQ_KEY) -> dict[str, int]:
    """Image count per sequence, in first-seen order. Raises if an image lacks the key."""
    out: dict[str, int] = {}
    for im in coco["images"]:
        if seq_key not in im:
            raise SplitError(
                f"image id {im.get('id')!r} ({im.get('file_name')!r}) has no {seq_key!r}; this "
                "document was not written by a Vividly converter, or the key was renamed"
            )
        out[str(im[seq_key])] = out.get(str(im[seq_key]), 0) + 1
    return out


def split_coco(
    coco: Mapping, spec: SplitSpec, *, parent_sha256: str | None = None
) -> tuple[dict, dict]:
    """Partition one COCO document into (train, test) by `spec`.

    Ids are kept as they are (COCO eval does not care and the numbers stay traceable to
    the parent). `info` is copied and gains a `vividly_split` record naming the spec, the
    side, and the parent document's sha256 when the caller knows it.
    """
    present = sequences_in(coco, spec.seq_key)
    spec.validate(present)
    sides = {"train": set(spec.train), "test": set(spec.test)}
    out: dict[str, dict] = {}
    for side, seqs in sides.items():
        keep_images = [im for im in coco["images"] if str(im[spec.seq_key]) in seqs]
        keep_ids = {im["id"] for im in keep_images}
        keep_anns = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
        info = copy.deepcopy(dict(coco.get("info") or {}))
        info["vividly_split"] = {
            "spec": asdict(spec),
            "side": side,
            "sequences": sorted(seqs),
            "parent_sha256": parent_sha256,
        }
        out[side] = {
            "info": info,
            "licenses": copy.deepcopy(list(coco.get("licenses") or [])),
            "images": copy.deepcopy(keep_images),
            "annotations": copy.deepcopy(keep_anns),
            "categories": copy.deepcopy(list(coco.get("categories") or [])),
        }
        if not keep_images:
            raise SplitError(f"{spec.dataset}: {side} side would be empty")
    return out["train"], out["test"]


def summarize(doc: Mapping, seq_key: str = SEQ_KEY) -> dict:
    """Counts a human checks against the registry entry before quoting anything.

    `vividly_track_id`, like the sequence key, comes from the caller's own annotation
    writer; no converter under `src/vividly/data/convert/` emits it, and an annotation
    without it is simply not counted towards `tracks`.
    """
    per_seq = sequences_in(doc, seq_key)
    tracks = {
        (str(a.get(seq_key)), a["vividly_track_id"])
        for a in doc["annotations"]
        if a.get("vividly_track_id") is not None
    }
    return {
        "images": len(doc["images"]),
        "annotations": len(doc["annotations"]),
        "sequences": per_seq,
        "tracks": len(tracks),
    }


def canonical_bytes(doc: Mapping) -> bytes:
    """Canonical encoding of a split document: compact separators, ensure_ascii false,
    key order preserved, UTF-8, newline-terminated."""
    return (json.dumps(doc, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
