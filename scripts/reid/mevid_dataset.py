#!/usr/bin/env python3
"""MEVID as a torchreid video re-ID dataset (v2 re-ID Phase A).

MEVID is a tracklet-level person re-identification benchmark built on top of the MEVA
security-camera corpus: ordinary perspective/top-view outdoor and indoor cameras, which is
exactly the v2 target domain. Re-identification here is the session-scoped appearance
embedding and nothing else: appearance from body crops, no identity signal from pose,
body shape or anything face-adjacent (POLICY.md, section 4, and rule 2 on facial
recognition).

On-disk format, from the upstream README (github.com/Kitware/MEVID):

    <root>/
      bbox_train/                 jpeg chips in per-identity subdirectories
      bbox_test/                  jpeg chips in per-identity subdirectories
      track_train_info.txt        5 int columns per tracklet
      track_test_info.txt         5 int columns per tracklet
      train_name.txt              one chip filename per line
      test_name.txt               one chip filename per line
      query_IDX.txt               line offsets into track_test_info.txt

A chip filename encodes everything about the crop:

    0201O003C330T004F00192.jpg
    person 0201, outfit 003, camera 330, tracklet 004, frame 00192; every index starts at 0.
    The fields run together with no punctuation: the letters O, C, T and F delimit them.

`track_{train,test}_info.txt` rows are `start_index end_index person_id outfit_id camera_id`,
where start/end are offsets into the corresponding `{train,test}_name.txt`. **The end offset
is inclusive.** The README's worked example describes a tracklet that "starts at line #0 and
ends at image #9" and holds ten chips, so the slice is `names[start : end + 1]`. Getting
that wrong silently drops one frame from every tracklet in the corpus, which is why
tests/test_mevid_dataset.py pins it explicitly.

`query_IDX.txt` selects query tracklets by line offset into `track_test_info.txt`; every
other test tracklet is gallery. Both are drawn from the same `bbox_test/` tree.

The v1 annotation release writes all three index tables in numpy's default float notation
rather than as plain integers: a row of `track_train_info.txt` reads
`0.000000000000000000e+00 9.000000000000000000e+00 1.030000000000000000e+02 ...` and a line
of `query_IDX.txt` reads `1.752000000000000000e+03`, while the README's worked example shows
the same row as `0 9 103 20 424`. The parsers below accept both spellings. A value with a
fractional part is a format error, because a line offset, an id or a camera number cannot
have one.

The release also carries exactly one zero-length row: line 1889 of `track_train_info.txt`
has start 1128477 and end 1128476 (person 78, outfit 30, camera 341), a tracklet whose chips
were all filtered out upstream, and it is why the table has 6,339 rows where the README
counts 6,338 tracklets. The loader keeps that row so every later offset stays where
`query_IDX.txt` and the outfit lookup expect it, and skips it when building the split
lists. A row whose end precedes its start by more than one is still a format error.

Beyond torchreid's own (img_paths, pid, camid) triples this module also carries the outfit
id through, per split, so the trainer can run an outfit-aware evaluation. torchreid has no
concept of a fourth attribute, so the outfit lives in parallel metadata keyed by position
rather than being smuggled into the tuples.

torchreid is imported inside the class factory, never at module import time. The mac dev
fork is deliberately torch-free, so `import mevid_dataset` has to succeed there for
the guards, the linter and a no-torchreid pytest collection; only constructing
`MEVID(...)` requires the library.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

#: Upstream file names. All five sit at the dataset root next to bbox_train/ and bbox_test/.
TRACK_INFO = {"train": "track_train_info.txt", "test": "track_test_info.txt"}
NAME_LIST = {"train": "train_name.txt", "test": "test_name.txt"}
BBOX_DIR = {"train": "bbox_train", "test": "bbox_test"}
QUERY_IDX = "query_IDX.txt"

#: `0201O003C330T004F00192.jpg`. Widths are not fixed by the README beyond the worked
#: example, so each field is `\d+` rather than a pinned width: a 5-digit frame counter or a
#: 3-digit person id must parse, not crash. The anchors are what keep it strict.
CHIP_RE = re.compile(
    r"^(?P<pid>\d+)O(?P<outfit>\d+)C(?P<camid>\d+)T(?P<tracklet>\d+)F(?P<frame>\d+)$"
)

#: How many characters of the chip name name its bbox_* subdirectory. MEVID follows the
#: MARS layout, where a chip lives under a directory named by its zero-padded person id.
#: This is only a fast path: a miss falls back to indexing the whole bbox tree, so a
#: different nesting scheme costs one walk instead of failing.
PID_DIR_WIDTH = 4


class ChipName(NamedTuple):
    """The five fields encoded in a MEVID chip filename."""

    pid: int
    outfit_id: int
    camid: int
    tracklet: int
    frame: int


class TrackletRow(NamedTuple):
    """One row of `track_{train,test}_info.txt`, as written on disk.

    `end_index` is INCLUSIVE (see the module docstring). `index` is the row's own 0-based
    line offset in the file, which is what `query_IDX.txt` selects on and what the
    outfit lookup is keyed by.
    """

    index: int
    start_index: int
    end_index: int
    pid: int
    outfit_id: int
    camid: int

    @property
    def n_chips(self) -> int:
        """Chips in the tracklet under the inclusive end offset; 0 for a zero-length row."""
        return self.end_index - self.start_index + 1


class TrackletMeta(NamedTuple):
    """Per-tracklet identity metadata, aligned positionally with a torchreid split list.

    `pid` matches the pid in the split's own tuples: relabelled 0..N-1 for `train`,
    raw upstream ids for `query` and `gallery` (torchreid never relabels the test side).
    """

    pid: int
    outfit_id: int
    camid: int


class MevidFormatError(RuntimeError):
    """The bytes under `root` are not a MEVID release this loader recognises."""


# ------------------------------------------------------------------ parsing primitives
def _int_field(text: str) -> int:
    """One integer column as the index tables spell it: `424` or `4.240000000000000000e+02`.

    Raises ValueError for anything non-numeric and for a value with a fractional part; the
    callers turn that into a MevidFormatError naming the file and line. Every offset in the
    corpus is below 2**53, so the float round trip is exact.
    """
    value = float(text)
    if not value.is_integer():
        raise ValueError(f"not an integer: {text!r}")
    return int(value)


def parse_chip_name(name: str) -> ChipName:
    """Decode `0201O003C330T004F00192.jpg` into its five integer fields.

    The `.jpg` suffix is optional so the same function works on a bare stem.
    """
    stem = name[: -len(Path(name).suffix)] if Path(name).suffix else name
    match = CHIP_RE.match(stem)
    if match is None:
        raise MevidFormatError(
            f"chip name {name!r} does not match the MEVID pattern "
            "<pid>O<outfit>C<camera>T<tracklet>F<frame>[.jpg]"
        )
    return ChipName(
        pid=int(match["pid"]),
        outfit_id=int(match["outfit"]),
        camid=int(match["camid"]),
        tracklet=int(match["tracklet"]),
        frame=int(match["frame"]),
    )


def read_name_list(path: Path) -> list[str]:
    """`{train,test}_name.txt` as a list of chip filenames, one per line.

    Blank lines are dropped rather than kept as empty names: a trailing newline is not a
    chip, and keeping it would shift every offset after it.
    """
    if not path.is_file():
        raise MevidFormatError(f"missing {path.name}: expected it at {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_track_info(path: Path) -> list[TrackletRow]:
    """`track_{train,test}_info.txt` as typed rows, in file order.

    Five whitespace-separated integer columns per row: start, end, pid, outfit, camera, in
    either the README's plain spelling or the release's float notation (module docstring).
    Rows are returned with their own line offset attached, because that offset is the
    identity `query_IDX.txt` refers to.
    """
    if not path.is_file():
        raise MevidFormatError(f"missing {path.name}: expected it at {path}")
    rows: list[TrackletRow] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if not text:
            continue
        fields = text.split()
        if len(fields) != 5:
            raise MevidFormatError(
                f"{path.name} line {lineno}: expected 5 integer columns "
                f"(start end pid outfit camera), got {len(fields)}: {text!r}"
            )
        try:
            start, end, pid, outfit, camid = (_int_field(f) for f in fields)
        except ValueError as exc:
            raise MevidFormatError(
                f"{path.name} line {lineno}: non-integer column: {text!r}"
            ) from exc
        # end == start - 1 is a zero-length row (module docstring); anything earlier is
        # a reversed row and the table cannot be trusted.
        if end < start - 1:
            raise MevidFormatError(
                f"{path.name} line {lineno}: end_index {end} precedes start_index {start}"
            )
        rows.append(
            TrackletRow(
                index=len(rows),
                start_index=start,
                end_index=end,
                pid=pid,
                outfit_id=outfit,
                camid=camid,
            )
        )
    if not rows:
        raise MevidFormatError(f"{path} contains no tracklet rows")
    return rows


def read_query_indices(path: Path) -> list[int]:
    """`query_IDX.txt` as a list of 0-based line offsets into `track_test_info.txt`.

    Written whitespace-agnostically: upstream ships one index per line, but a single
    whitespace-separated block is the other shape these files come in and costs nothing
    to accept.
    """
    if not path.is_file():
        raise MevidFormatError(f"missing {QUERY_IDX}: expected it at {path}")
    try:
        indices = [_int_field(tok) for tok in path.read_text(encoding="utf-8").split()]
    except ValueError as exc:
        raise MevidFormatError(f"{path.name}: expected whitespace-separated integers") from exc
    if not indices:
        raise MevidFormatError(f"{path} selects no query tracklets")
    return indices


def resolve_dataset_root(root: str | Path) -> Path:
    """The directory that holds `track_train_info.txt`.

    Accepts either the dataset directory itself, `data/train_prototype/mevid`, the layout
    root configs/datasets/mevid.yaml declares, or a torchreid-style parent that contains it,
    `data/train_prototype` with `mevid/` inside, because
    `VideoDataManager(root=...)` is a *parent* by torchreid convention while every
    Vividly config names the dataset directory directly.
    """
    base = Path(root).expanduser()
    for candidate in (base, base / "mevid", base / "MEVID"):
        if (candidate / TRACK_INFO["train"]).is_file():
            return candidate.resolve()
    raise MevidFormatError(
        f"no MEVID release under {base}: expected {TRACK_INFO['train']} in {base}, "
        f"{base / 'mevid'} or {base / 'MEVID'}. Point `root` at the directory holding "
        "bbox_train/, bbox_test/ and the five index files."
    )


# ------------------------------------------------------------------ chip path resolution
def _index_bbox_tree(bbox_dir: Path) -> dict[str, Path]:
    """Every jpeg under `bbox_dir`, keyed by bare filename.

    The fallback for when the MARS-style `<pid>/<chip>.jpg` fast path misses. One walk,
    cached by the caller: configs/datasets/mevid.yaml records 3,609,156 chip files in the
    train split and 1,186,251 in the test split, so doing this per chip would be the
    difference between seconds and hours.
    """
    index: dict[str, Path] = {}
    for path in bbox_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            index.setdefault(path.name, path)
    return index


class _ChipResolver:
    """Chip filename -> absolute path, fast path first, whole-tree index as fallback."""

    def __init__(self, bbox_dir: Path) -> None:
        self.bbox_dir = bbox_dir
        self._index: dict[str, Path] | None = None

    def __call__(self, name: str) -> Path:
        direct = self.bbox_dir / name[:PID_DIR_WIDTH] / name
        if direct.is_file():
            return direct
        flat = self.bbox_dir / name
        if flat.is_file():
            return flat
        if self._index is None:
            self._index = _index_bbox_tree(self.bbox_dir)
        found = self._index.get(name)
        if found is None:
            raise MevidFormatError(
                f"chip {name!r} is listed in the name file but is not under {self.bbox_dir}"
            )
        return found


# ------------------------------------------------------------------ split construction
def build_split(
    *,
    dataset_root: Path,
    split: str,
    rows: Sequence[TrackletRow],
    names: Sequence[str],
    pid_map: Mapping[int, int] | None = None,
) -> tuple[list[tuple[tuple[str, ...], int, int]], list[TrackletMeta]]:
    """Turn tracklet rows into torchreid triples plus the parallel outfit metadata.

    Returns `(tracklets, meta)` where `tracklets[i]` is
    `((img_path, ...), pid, camid)` exactly as torchreid's `VideoDataset` wants it, and
    `meta[i]` is the `(pid, outfit_id, camid)` for the same tracklet. `pid_map` relabels
    ids to a contiguous 0..N-1 range for the train split; passing None keeps the raw
    upstream ids, which is what the test side needs.

    `camid` stays the integer from column 5. It is a real camera number (C330, C336, ...),
    not a 0..K index, and torchreid only ever compares camids for equality, so remapping
    them would throw away the one thing the per-camera debias pass needs to group on.
    """
    bbox_dir = dataset_root / BBOX_DIR[split]
    if not bbox_dir.is_dir():
        raise MevidFormatError(f"missing {BBOX_DIR[split]}/ under {dataset_root}")
    resolve = _ChipResolver(bbox_dir)

    tracklets: list[tuple[tuple[str, ...], int, int]] = []
    meta: list[TrackletMeta] = []
    for row in rows:
        if row.n_chips == 0:
            # A zero-length row (module docstring): kept in `rows` so later offsets stay
            # put, contributes no tracklet.
            continue
        # INCLUSIVE end offset - see the module docstring. `end + 1` is the whole reason
        # this function exists rather than being three lines inline.
        chip_names = names[row.start_index : row.end_index + 1]
        if not chip_names:
            raise MevidFormatError(
                f"{TRACK_INFO[split]} row {row.index} selects lines "
                f"{row.start_index}..{row.end_index} of {NAME_LIST[split]}, which holds "
                f"{len(names)} names - the slice is empty"
            )
        paths = tuple(str(resolve(name)) for name in chip_names)
        pid = pid_map[row.pid] if pid_map is not None else row.pid
        tracklets.append((paths, pid, row.camid))
        meta.append(TrackletMeta(pid=pid, outfit_id=row.outfit_id, camid=row.camid))
    return tracklets, meta


def relabel_map(rows: Iterable[TrackletRow]) -> dict[int, int]:
    """Upstream person id -> contiguous 0..N-1 label, in ascending id order.

    Sorted rather than first-seen so the mapping is a pure function of the id set: the
    classifier head's column order must not depend on file order, or two runs over the
    same corpus produce checkpoints whose logits mean different things.
    """
    return {pid: label for label, pid in enumerate(sorted({row.pid for row in rows}))}


# ------------------------------------------------------------------ the torchreid class
_IMPL: type | None = None


def _build_impl() -> type:
    """Create (once) the real `VideoDataset` subclass.

    Deferred so that importing this module never imports torchreid: the mac dev fork is
    torch-free by design and still has to lint, collect tests and run the guards over
    this file.
    """
    global _IMPL
    if _IMPL is not None:
        return _IMPL

    try:
        # torchreid is MIT (KaiyangZhou/deep-person-reid), sanctioned in the cuda and
        # curate forks; see docs/LICENCE_ALLOWLIST.md.
        from torchreid.data.datasets import VideoDataset
    except ImportError as exc:  # pragma: no cover - exercised only on a torchreid-free env
        raise ImportError(
            "MEVID needs torchreid, and the locked training environment does not carry it: "
            "`uv sync --extra cuda --no-group mac --locked` installs the environment, then "
            "the pinned clone is installed on top with `uv pip install --no-deps "
            "--no-build-isolation`. TRAINING.md gives the sequence in full under "
            "'Re-identification: phase A'. On macOS the `curate` extra carries the library."
        ) from exc

    class _MEVID(VideoDataset):  # type: ignore[misc, valid-type]
        """MEVID tracklet re-ID, in torchreid's VideoDataset shape.

        Extra attributes beyond the torchreid contract, all positionally aligned with
        `self.train` / `self.query` / `self.gallery`:

        * `train_meta`, `query_meta`, `gallery_meta` - lists of `TrackletMeta`
          `(pid, outfit_id, camid)`, so an outfit-aware evaluation can be run without
          re-parsing filenames.
        * `train_outfit_by_track`, `test_outfit_by_track` - `{tracklet line offset ->
          outfit_id}` keyed by the row's own offset in `track_{train,test}_info.txt`,
          which is the same key space `query_IDX.txt` indexes.
        * `query_indices` - the offsets read from `query_IDX.txt`.
        * `pid_relabel` - the raw-id -> 0..N-1 map applied to the train split.
        """

        dataset_dir = "mevid"

        def __init__(
            self,
            root: str = "",
            seq_len: int = 15,
            sample_method: str = "evenly",
            **kwargs: Any,
        ) -> None:
            self.root = str(root)
            self.dataset_root = resolve_dataset_root(root)

            train_rows = read_track_info(self.dataset_root / TRACK_INFO["train"])
            test_rows = read_track_info(self.dataset_root / TRACK_INFO["test"])
            train_names = read_name_list(self.dataset_root / NAME_LIST["train"])
            test_names = read_name_list(self.dataset_root / NAME_LIST["test"])
            self.query_indices = read_query_indices(self.dataset_root / QUERY_IDX)

            out_of_range = [i for i in self.query_indices if not 0 <= i < len(test_rows)]
            if out_of_range:
                raise MevidFormatError(
                    f"{QUERY_IDX} selects rows {out_of_range[:5]} but "
                    f"{TRACK_INFO['test']} has only {len(test_rows)} rows"
                )

            # A zero-length row contributes no tracklet, so its id must not get a
            # classifier column either.
            self.pid_relabel = relabel_map(row for row in train_rows if row.n_chips)
            train, self.train_meta = build_split(
                dataset_root=self.dataset_root,
                split="train",
                rows=train_rows,
                names=train_names,
                pid_map=self.pid_relabel,
            )

            # A tracklet is query if its line offset appears in query_IDX.txt; every other
            # test tracklet is gallery. Duplicate offsets would double-count a query, so
            # membership is a set and the split is a partition by construction.
            query_set = set(self.query_indices)
            query_rows = [row for row in test_rows if row.index in query_set]
            gallery_rows = [row for row in test_rows if row.index not in query_set]
            query, self.query_meta = build_split(
                dataset_root=self.dataset_root,
                split="test",
                rows=query_rows,
                names=test_names,
            )
            gallery, self.gallery_meta = build_split(
                dataset_root=self.dataset_root,
                split="test",
                rows=gallery_rows,
                names=test_names,
            )

            self.train_outfit_by_track = {r.index: r.outfit_id for r in train_rows}
            self.test_outfit_by_track = {r.index: r.outfit_id for r in test_rows}

            super().__init__(
                train,
                query,
                gallery,
                seq_len=seq_len,
                sample_method=sample_method,
                **kwargs,
            )

        def meta_for(self, mode: str) -> list[TrackletMeta]:
            """`(pid, outfit_id, camid)` for one split, aligned with its tuple list."""
            try:
                return {
                    "train": self.train_meta,
                    "query": self.query_meta,
                    "gallery": self.gallery_meta,
                }[mode]
            except KeyError:
                raise ValueError(
                    f"unknown mode {mode!r}; expected train, query or gallery"
                ) from None

    _IMPL = _MEVID
    return _IMPL


class MEVID:
    """MEVID video re-ID dataset for torchreid.

    Constructing this returns a real `torchreid.data.datasets.VideoDataset` subclass
    instance; the indirection exists only so that importing this module does not import
    torchreid (see the module docstring). `__new__` returning a foreign instance means
    Python skips `__init__`, which is exactly the behaviour wanted here.

    Register it with `register_mevid()` and then name it as a `VideoDataManager` source.
    """

    def __new__(cls, root: str = "", **kwargs: Any) -> Any:
        return _build_impl()(root=root, **kwargs)


def register_mevid(name: str = "mevid") -> None:
    """Register MEVID under `name` with torchreid, idempotently.

    torchreid's `register_video_dataset` raises on a duplicate name, which turns a second
    call in the same process (a resumed run, a test that builds two managers) into a
    crash. Re-registering the same class is a no-op here; a DIFFERENT class under the same
    name is still an error, because that would silently swap the corpus underneath a run.
    """
    from torchreid.data import register_video_dataset
    from torchreid.data.datasets import __video_datasets

    existing = __video_datasets.get(name)
    if existing is MEVID:
        return
    if existing is not None:
        raise ValueError(
            f"torchreid already has a different video dataset registered as {name!r}: "
            f"{existing!r}. Refusing to replace it."
        )
    register_video_dataset(name, MEVID)
