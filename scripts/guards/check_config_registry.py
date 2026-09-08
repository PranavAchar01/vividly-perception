#!/usr/bin/env python3
"""Registry guard: every corpus a training configuration reaches is a registered training set.

The sibling guard, scripts/guards/check_train_data_paths.py, proves that a path inside the
repository resolves into the training trees. That is necessary and not sufficient. A directory
dropped into either tree, a staging copy on a training machine, or a track tree extracted from
an evaluation-only corpus all have no licence record, no provenance and no counsel review, and
would enter a fine-tune unnoticed. This guard resolves each of them to a registry entry and
applies the bucket rule.

Three resolutions, over the same training surface the sibling guard defines.

1. Track trees. A training configuration addresses its inputs as outputs/events/<tree>. The
   configuration that produced a tree declares both the tree and the dataset behind it, so the
   producers under configs/events/ are read first and give a tree family for each registered
   corpus. A tree is matched to the longest producer family that is a prefix of its name, so a
   re-extraction of the same corpus resolves to the same entry as the extraction of record. A
   tree that matches no family is unresolvable and fails: nothing may train on a tree whose
   corpus cannot be named.
2. Repository data paths. A path under data/ is owned by the registry entry whose layout root
   contains it.
3. Staging paths. A path outside the repository is a working copy on a training machine. Its
   corpus is the first path segment that names a registry entry or one of the built corpora
   below, each of which is a corpus assembled from registered entries by a recipe under
   configs/convert/.

The bucket rule, applied to every resolved entry:

  TRAIN_OK        may train any weight.
  TRAIN_PROTOTYPE may train prototype weights only, and the configuration must declare the
                  prototype weight class. A re-identification configuration declares it in the
                  file; an event training configuration inherits the declaration the event
                  verifier trainer stamps into every checkpoint it writes.
  EVAL_ONLY       never trains. The one admissible form is a corpus with a registered
                  train-side slice, named <corpus>_trainsplit, read through a locations filter
                  that selects the training recordings. Without the slice entry or without the
                  filter the tree is refused.
  LEGAL_HOLD      never resolved at all.

Usage:
    uv run python scripts/guards/check_config_registry.py            # the training surface
    uv run python scripts/guards/check_config_registry.py FILE...    # pre-commit mode
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_train_data_paths import (  # noqa: E402
    EVAL_ONLY_SEGMENT,
    PROTOTYPE,
    WEIGHT_CLASS_KEY,
    data_path_candidates,
    declared_weight_class,
    event_sources,
    iter_targets,
    load_document,
)

from vividly.data.registry import REPO_ROOT as R_ROOT  # noqa: E402
from vividly.data.registry import Bucket, Registry  # noqa: E402

#: Where the event lane writes and reads its track trees.
TRACK_TREE_ROOT = "outputs/events"

#: Producer configurations live here: an extraction or diet configuration declares the tree it
#: writes and the corpus it reads.
PRODUCER_GLOB = "configs/events/*.yaml"

#: A corpus assembled from registered entries and staged on a training machine under a name of
#: its own, mapped to the entries its bytes come from. Each recipe is a configuration under
#: configs/convert/, and the entries are checked against the registry at startup so that a
#: renamed or rebucketed entry fails here rather than quietly widening the surface.
BUILT_CORPORA: dict[str, tuple[str, ...]] = {
    # configs/convert/person_v2_manifests.yaml merges the training scenes into one corpus.
    "person_v2": ("physicalai_smartspaces",),
    # The COCO 2017 release, staged under its upstream directory name.
    "coco2017": ("coco",),
}

#: Track trees that carry no imagery and no dataset, so no registry entry can exist for them.
#: Motion synthesis in pose and track space is a recorded carve-out of the event data rule;
#: generative patient imagery remains barred. The producing script is checked to exist, so a
#: name here can never grandfather a tree nothing in the tree builds.
SYNTHESISED_TREES: dict[str, str] = {
    "synth_convulsion": "scripts/events/make_synth_convulsion.py",
}

#: The suffix that marks a registered train-side slice of an evaluation-only corpus.
TRAINSPLIT_SUFFIX = "_trainsplit"


def _rel(path: Path) -> Path:
    try:
        return path.resolve().relative_to(R_ROOT)
    except ValueError:
        return path


def _family(tree_name: str) -> str:
    """The tree family a producer defines: its directory name without a version suffix.

    caucafall_tracks_v2c and caucafall_tracks_v2pose are two extractions of one corpus, so
    both must resolve to the same registry entry. Dropping a trailing version segment is what
    makes a re-extraction resolve without a second producer configuration having to survive
    in the repository.
    """
    head, sep, tail = tree_name.rpartition("_")
    if sep and len(tail) > 1 and tail[0] == "v" and tail[1].isdigit():
        return head
    return tree_name


def producer_families(reg: Registry) -> tuple[dict[str, str], list[str]]:
    """Map a tree family to the registry entry behind it, reading the producer configurations."""
    families: dict[str, str] = {}
    problems: list[str] = []
    roots = {
        (R_ROOT / reg[name].layout.root).resolve(): name
        for name in reg.names()
        if reg[name].layout.root
    }
    for path in sorted(R_ROOT.glob(PRODUCER_GLOB)):
        doc = load_document(path)
        out_dir = doc.get("out_dir")
        if not isinstance(out_dir, str) or not out_dir.startswith(TRACK_TREE_ROOT + "/"):
            continue
        family = _family(Path(out_dir).name)
        named = doc.get("dataset")
        owner: str | None = None
        if isinstance(named, str):
            if named in reg:
                owner = named
            else:
                problems.append(
                    f"Unregistered producer corpus {_rel(path)}: dataset {named!r} has no entry "
                    "under configs/datasets/, so every tree it writes is unattributable."
                )
                continue
        else:
            for raw in sorted(data_path_candidates(path)):
                resolved = (
                    Path(raw).resolve() if Path(raw).is_absolute() else (R_ROOT / raw).resolve()
                )
                owner = next(
                    (
                        n
                        for root, n in roots.items()
                        if resolved == root or root in resolved.parents
                    ),
                    None,
                )
                if owner:
                    break
        if owner:
            families.setdefault(family, owner)
    return families, problems


def _owner_of_path(raw: str, roots: dict[Path, str], reg: Registry) -> str | None:
    """The registry entry a data path or a staging path belongs to."""
    resolved = Path(raw).resolve() if Path(raw).is_absolute() else (R_ROOT / raw).resolve()
    owner = next(
        (n for root, n in roots.items() if resolved == root or root in resolved.parents), None
    )
    if owner:
        return owner
    for part in Path(raw).parts:
        if part in reg:
            return part
        if part in BUILT_CORPORA:
            return part
    return None


def _entries_for(owner: str) -> tuple[str, ...]:
    return BUILT_CORPORA.get(owner, (owner,))


def _bucket_errors(
    reg: Registry, where: Path, subject: str, owner: str, weight_class: str | None
) -> list[str]:
    """The bucket rule for one resolved corpus, ignoring the evaluation-only slice case."""
    errors: list[str] = []
    for name in _entries_for(owner):
        bucket = reg[name].bucket
        if bucket is Bucket.TRAIN_OK:
            continue
        if bucket is Bucket.TRAIN_PROTOTYPE:
            if weight_class != PROTOTYPE:
                errors.append(
                    f"Prototype corpus without declaration {where}: {subject} belongs to the "
                    f"{bucket.value} entry {name!r} and the configuration does not declare "
                    f"{WEIGHT_CLASS_KEY}: {PROTOTYPE}."
                )
            continue
        errors.append(
            f"Corpus in training from the {bucket.value} bucket {where}: {subject} belongs to the "
            f"{bucket.value} entry {name!r}, which never trains anything."
        )
    return errors


def check_tree(
    reg: Registry,
    families: dict[str, str],
    where: Path,
    tree: str,
    locations: list | None,
    weight_class: str | None,
) -> list[str]:
    name = Path(tree).name
    for token, producer in SYNTHESISED_TREES.items():
        if name == token or name.startswith(token + "_"):
            if not (R_ROOT / producer).exists():
                return [
                    f"Synthesis producer missing {where}: tree {tree!r} is recorded as "
                    f"synthesised motion produced by {producer}, which is not in the tree."
                ]
            return []
    matched = [family for family in families if name == family or name.startswith(family + "_")]
    if not matched:
        return [
            f"Unresolvable track tree {where}: {tree!r} matches no extraction configuration "
            f"under configs/events/, so the corpus behind it cannot be named. Known families: "
            f"{sorted(families)}."
        ]
    owner = families[max(matched, key=len)]
    bucket = reg[owner].bucket
    if bucket is not Bucket.EVAL_ONLY:
        return _bucket_errors(reg, where, f"tree {tree!r}", owner, weight_class)
    slice_name = owner + TRAINSPLIT_SUFFIX
    if slice_name not in reg:
        return [
            f"Corpus in training from the EVAL_ONLY bucket {where}: tree {tree!r} was extracted "
            f"from the EVAL_ONLY entry {owner!r}, which has no registered train-side slice "
            f"({slice_name!r}), so no part of it may train."
        ]
    if not isinstance(locations, list) or not locations:
        return [
            f"Corpus without a slice filter in the EVAL_ONLY bucket {where}: tree {tree!r} was "
            f"extracted from the EVAL_ONLY entry {owner!r}. Only the recordings of the slice "
            f"{slice_name!r} may train, and they are selected by a locations filter."
        ]
    return _bucket_errors(reg, where, f"tree {tree!r}", slice_name, weight_class)


def check(reg: Registry, families: dict[str, str], roots: dict[Path, str], path: Path) -> list[str]:
    where = _rel(path)
    weight_class = declared_weight_class(path)
    errors: list[str] = []
    for source in event_sources(path):
        tree = source.get("dir")
        if isinstance(tree, str) and tree.startswith(TRACK_TREE_ROOT + "/"):
            errors += check_tree(reg, families, where, tree, source.get("locations"), weight_class)
    for raw in sorted(data_path_candidates(path)):
        if EVAL_ONLY_SEGMENT in Path(raw).parts:
            continue  # already reported by the sibling guard, with the same verdict
        owner = _owner_of_path(raw, roots, reg)
        if owner is None:
            errors.append(
                f"Unregistered data {where}: {raw!r} belongs to no registry entry and to no "
                "built corpus. Register the corpus under configs/datasets/, or record how it "
                "is assembled from registered entries."
            )
            continue
        errors += _bucket_errors(reg, where, f"path {raw!r}", owner, weight_class)
    return errors


def main(argv: list[str]) -> int:
    reg = Registry.from_dir(repo_root=R_ROOT)
    errors: list[str] = []
    for corpus, names in sorted(BUILT_CORPORA.items()):
        for name in names:
            if name not in reg:
                errors.append(
                    f"Built corpus names an absent entry: {corpus!r} is recorded as built from "
                    f"{name!r}, which has no entry under configs/datasets/."
                )
    families, problems = producer_families(reg)
    # A broken producer is a property of the whole tree rather than of the files a commit
    # touched, so it is reported in full-scan mode only. The pre-commit hook and the workflow
    # both run the full scan, so nothing escapes; naming it in filename mode would only report
    # files the author did not touch.
    if not argv:
        errors += problems

    roots = {
        (R_ROOT / reg[name].layout.root).resolve(): name
        for name in reg.names()
        if reg[name].layout.root
    }
    targets, skipped = iter_targets([Path(a) for a in argv])
    for path in targets:
        errors += check(reg, families, roots, path)

    for error in errors:
        print(error, file=sys.stderr)
    print(
        f"check_config_registry: {len(targets)} training config(s), {len(families)} tree "
        f"family(ies), {skipped} evaluation config(s) skipped, {len(errors)} violation(s)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
