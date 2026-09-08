#!/usr/bin/env python3
"""Training-path guard: a training configuration never reads data outside the training trees.

Surface, the v2 training configurations:

  configs/events/train_*.yaml, configs/events/rot_*.yaml and configs/events/lopo_*.yaml that
  carry a top-level `train:` block. The remaining files under those names are evaluation and
  validation configurations, which are entitled to read evaluation-only recordings, and the
  guard reports how many it skipped for that reason so the surface is never silently empty.
  configs/reid/*.yaml.
  third_party/configs/*.yml and third_party/configs/*.py.

Four rules, all mechanical.

1. A repository path under data/ is resolved with Path.resolve(), which follows symbolic
   links, and must land inside data/train_ok/ or data/train_prototype/. A link from
   data/train_ok/x to data/eval_only/y is therefore caught, which a plain string test misses.
2. A path whose segments name the evaluation-only tree is refused whether it is
   repository-relative or absolute, so a staging path such as /scratch/eval_only/x cannot
   launder the rule.
3. A configuration that reads data/train_prototype/ must declare the prototype weight class.
   A re-identification configuration declares it in the file as `weight_class: prototype`.
   The event verifier trainer declares it in code instead, as the module constant this guard
   reads back from scripts/events/train_event_verifier.py, so every event verifier is
   prototype class by construction and the guard fails if that constant is removed or given
   another value.
4. Exam recordings never enter training. A training configuration that reads a clinical
   track tree must carry a locations filter, and that filter may not name a focal exam
   recording (pat06, pat08, pat10, pat12) or a convulsive exam recording (pat03, pat07).

Division of labour with the sibling guard. This guard owns paths that live inside the
repository tree and the locations filter. A path that resolves outside the repository, such
as the staging copy a training pod reads, is handed to scripts/guards/check_config_registry.py,
which resolves the corpus name against the dataset registry and applies the bucket rule. The
bucket behind a track tree under outputs/events/ belongs to that guard for the same reason.

Static reading only, and deliberately so. The pose fine-tune configuration inherits from a
base file that is installed into the upstream checkout at launch and does not exist in this
repository, so materialising a configuration here is not possible. The scan reads what this
repository carries, which is what a review can see.

Usage:
    uv run python scripts/guards/check_train_data_paths.py            # the whole training surface
    uv run python scripts/guards/check_train_data_paths.py FILE...    # pre-commit mode
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_OK = (REPO_ROOT / "data" / "train_ok").resolve()
TRAIN_PROTOTYPE = (REPO_ROOT / "data" / "train_prototype").resolve()

#: The training surface, by glob. Event configurations are filtered further by the presence
#: of a train block; see is_event_training_config.
EVENT_GLOBS = (
    "configs/events/train_*.yaml",
    "configs/events/rot_*.yaml",
    "configs/events/lopo_*.yaml",
)
REID_GLOBS = ("configs/reid/*.yaml",)
THIRD_PARTY_GLOBS = ("third_party/configs/*.yml", "third_party/configs/*.py")
SURFACE_GLOBS = EVENT_GLOBS + REID_GLOBS + THIRD_PARTY_GLOBS

#: The path segment that names the evaluation-only tree, wherever it appears.
EVAL_ONLY_SEGMENT = "eval_only"

#: The key a re-identification configuration assigns to opt into prototype data, and the only
#: value it may carry. A shippable recipe omits the key.
WEIGHT_CLASS_KEY = "weight_class"
PROTOTYPE = "prototype"

#: The event verifier trainer stamps the weight class into every checkpoint it writes, so an
#: event training configuration has no weight-class key of its own. The declaration is this
#: module constant, read back below so that removing it fails the guard rather than silently
#: turning every event verifier into an undeclared weight.
EVENT_TRAINER = Path("scripts/events/train_event_verifier.py")
EVENT_TRAINER_CONSTANT = "WEIGHT_CLASS"

#: Exam recordings of the clinical corpus. The focal recordings are excluded from convulsion
#: training as label-uncertain and hold the focal exam; the convulsive pair holds the
#: convulsive exam. Neither set may appear in a training configuration's locations filter.
EXAM_LOCATIONS = {
    "pat06": "focal exam",
    "pat08": "focal exam",
    "pat10": "focal exam",
    "pat12": "focal exam",
    "pat03": "convulsive exam",
    "pat07": "convulsive exam",
}

#: A track tree whose name begins with this token carries the clinical corpus, so a training
#: configuration reading it is subject to the exam rule above.
CLINICAL_TREE_TOKEN = "wu_sahzu"

#: Keys whose string values are data paths whatever they look like. A value such as
#: /scratch/corpus/ann.json has no data/ substring and is still a training input.
DATA_KEYS = frozenset(
    {
        "ann_dir",
        "ann_file",
        "annotations_dir",
        "bbox_file",
        "data_dir",
        "data_prefix",
        "data_root",
        "img_dir",
        "img_folder",
        "img_path",
        "img_prefix",
        "root",
        "seg_prefix",
        "sessions_dir",
        "video_root",
    }
)

#: Track trees are addressed under this prefix and are the sibling guard's subject.
TRACK_TREE_PREFIX = "outputs/"


def looks_like_data_path(value: str) -> bool:
    return value.startswith("data/") or "/data/" in value or value.startswith("data\\")


def is_enforced(value: str) -> bool:
    """A collected string this guard resolves, as opposed to a relative fragment."""
    if value.startswith(TRACK_TREE_PREFIX):
        return False
    return Path(value).is_absolute() or looks_like_data_path(value)


def _strings_under(obj: object, out: set[str]) -> None:
    """Every string leaf below a data key counts, whatever the shape around it."""
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _strings_under(value, out)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            _strings_under(value, out)


def _walk(obj: object, out: set[str]) -> None:
    if isinstance(obj, str):
        if looks_like_data_path(obj):
            out.add(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key in DATA_KEYS:
                _strings_under(value, out)
            else:
                _walk(value, out)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            _walk(value, out)


def load_document(path: Path) -> dict:
    """A configuration as a mapping, or an empty mapping when it is not one."""
    if path.suffix == ".py":
        return {}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: not loadable YAML ({exc.__class__.__name__})") from exc
    return doc if isinstance(doc, dict) else {}


def _collect_python(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and looks_like_data_path(node.value)
        ):
            out.add(node.value)
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (
                    kw.arg in DATA_KEYS
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    out.add(kw.value.value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in DATA_KEYS
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    out.add(node.value.value)
    return out


def data_path_candidates(path: Path) -> set[str]:
    """Every string in a configuration that names a dataset input.

    One owner for the collection logic: the registry guard imports this function so that both
    guards see exactly the same strings. A path one guard rejects can then never slip past the
    other because their collections drifted apart.
    """
    if path.suffix == ".py":
        raw = _collect_python(path)
    else:
        raw = set()
        _walk(load_document(path), raw)
    return {value for value in raw if is_enforced(value)}


def event_sources(path: Path) -> list[dict]:
    """The sources block of an event configuration, as a list of mappings."""
    doc = load_document(path)
    sources = doc.get("sources")
    if not isinstance(sources, list):
        return []
    return [item for item in sources if isinstance(item, dict)]


def is_event_training_config(path: Path) -> bool:
    """True when an event configuration trains, as opposed to scoring an exported verifier.

    A training configuration carries a top-level train block holding the epochs, the seed and
    the optimiser settings. An evaluation or validation configuration carries a model block
    naming an exported graph and no train block, and reads evaluation recordings by design.
    """
    return isinstance(load_document(path).get("train"), dict)


def is_event_config(path: Path) -> bool:
    return _rel(path).as_posix().startswith("configs/events/")


def declared_weight_class(path: Path) -> str | None:
    """The weight class a configuration declares, from the file or from the trainer."""
    if is_event_config(path):
        return event_trainer_weight_class()
    value = load_document(path).get(WEIGHT_CLASS_KEY)
    return value if isinstance(value, str) else None


def event_trainer_weight_class() -> str | None:
    """The weight class the event verifier trainer stamps into every checkpoint it writes."""
    source = REPO_ROOT / EVENT_TRAINER
    if not source.exists():
        return None
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == EVENT_TRAINER_CONSTANT
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    return node.value.value
    return None


def _under(resolved: Path, root: Path) -> bool:
    try:
        resolved.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve(raw: str) -> Path:
    return Path(raw).resolve() if Path(raw).is_absolute() else (REPO_ROOT / raw).resolve()


def _rel(path: Path) -> Path:
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


def check_paths(path: Path, weight_class: str | None) -> list[str]:
    rel = _rel(path)
    errors: list[str] = []
    if weight_class is not None and weight_class != PROTOTYPE:
        errors.append(
            f"Bad weight class {rel}: {WEIGHT_CLASS_KEY} is {weight_class!r}; the only "
            f"declarable value is {PROTOTYPE!r}, and a shippable recipe omits the key."
        )
    for raw in sorted(data_path_candidates(path)):
        if EVAL_ONLY_SEGMENT in Path(raw).parts:
            errors.append(
                f"Evaluation-only reference {rel}: {raw!r} names the evaluation-only tree. "
                "Evaluation-only data measures a model and never trains one."
            )
            continue
        resolved = _resolve(raw)
        if _under(resolved, TRAIN_OK):
            continue
        if _under(resolved, TRAIN_PROTOTYPE):
            if weight_class == PROTOTYPE:
                continue
            errors.append(
                f"Prototype data without declaration {rel}: {raw!r} resolves to {resolved}, "
                f"inside {TRAIN_PROTOTYPE}, and the configuration does not declare "
                f"{WEIGHT_CLASS_KEY}: {PROTOTYPE}. A weight trained from prototype data is "
                "tagged as a prototype weight; add the declaration or read the training tree."
            )
            continue
        if resolved.is_relative_to(REPO_ROOT):
            errors.append(
                f"Path outside the training trees {rel}: {raw!r} resolves to {resolved}, "
                f"which is inside the repository but outside {TRAIN_OK} and {TRAIN_PROTOTYPE}."
            )
        # A path outside the repository is a staging copy on a training machine. Its corpus is
        # resolved against the registry by scripts/guards/check_config_registry.py.
    return errors


def check_locations(path: Path) -> list[str]:
    rel = _rel(path)
    errors: list[str] = []
    for source in event_sources(path):
        tree = source.get("dir")
        if not isinstance(tree, str) or CLINICAL_TREE_TOKEN not in tree:
            continue
        locations = source.get("locations")
        if not isinstance(locations, list) or not locations:
            errors.append(
                f"Clinical tree without a locations filter {rel}: source {tree!r} carries no "
                "locations filter, so the whole corpus including the exam recordings would "
                "enter training. Name the training recordings explicitly."
            )
            continue
        for name in locations:
            if name in EXAM_LOCATIONS:
                errors.append(
                    f"Exam recording in training {rel}: source {tree!r} lists {name!r}, which "
                    f"holds the {EXAM_LOCATIONS[name]}. An exam recording never trains."
                )
    return errors


def check(path: Path) -> list[str]:
    errors = check_paths(path, declared_weight_class(path))
    if is_event_config(path):
        errors += check_locations(path)
    return errors


def in_surface(path: Path) -> bool:
    rel = _rel(path).as_posix()
    return any(Path(rel).match(pattern) for pattern in SURFACE_GLOBS)


def iter_targets(files: list[Path]) -> tuple[list[Path], int]:
    """The training configurations to check, and how many evaluation configurations were skipped."""
    if files:
        candidates = [p for p in files if in_surface(p) and p.is_file()]
    else:
        candidates = sorted({p for glob in SURFACE_GLOBS for p in REPO_ROOT.glob(glob)})
    targets: list[Path] = []
    skipped = 0
    for path in candidates:
        if is_event_config(path) and not is_event_training_config(path):
            skipped += 1
            continue
        targets.append(path)
    return targets, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    targets, skipped = iter_targets(args.files)

    errors: list[str] = []
    declared = event_trainer_weight_class()
    if declared != PROTOTYPE:
        errors.append(
            f"Event trainer declaration lost {EVENT_TRAINER}: {EVENT_TRAINER_CONSTANT} is "
            f"{declared!r}, not {PROTOTYPE!r}. Every event verifier is a prototype weight, "
            "and that declaration is what tags the checkpoints it writes."
        )
    for path in targets:
        try:
            errors += check(path)
        except ValueError as exc:
            errors.append(str(exc))

    for error in errors:
        print(error, file=sys.stderr)
    print(
        f"check_train_data_paths: {len(targets)} training config(s), "
        f"{skipped} evaluation config(s) skipped, {len(errors)} violation(s)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
