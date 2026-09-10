#!/usr/bin/env python3
"""Clip Library profile guard (CLIP-LIBRARY-DESIGN.md, P2 and D-C).

Any pipeline profile that writes a Clip spool (it sets `clips.spool_dir`) is a profile of
the non-device layer, whatever its file name, and must hold five lines: no verifier head,
no reference to the third rule head or the periodicity analyzer, no on-disk event store
outside the library's own pipeline directory, no persisted re-association, and an explicit
`rules.enabled` allowlist limited to the two Clip triggers. It also may not write under the
repository's outputs/ tree. scripts/run_pipeline.py refuses the same things at start; this
guard refuses them at commit time and in `make guards`, keyed on the field rather than on
the file name so a site profile saved under any name is held to the same rules.

Usage:
    uv run python scripts/guards/check_clip_library_config.py            # configs/pipeline/**
    uv run python scripts/guards/check_clip_library_config.py FILE...    # pre-commit mode
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_GLOB = "configs/pipeline/**/*.yaml"

#: The only rule heads a spool-writing profile may list.
ALLOWED_RULES = frozenset({"fall_then_still_v0", "egress_without_discharge_v0"})

#: Substrings (case-insensitive) that must not appear anywhere in such a profile, comments
#: included: the third head, its weights, and the periodicity analyzer's vocabulary.
FORBIDDEN_TEXT = (
    "seizure_candidate_v0",
    "final_models/seizure",
    "convuls",
    "band-pass",
    "hz_ratio",
    "hf_ratio",
)

#: Where the pipeline's own files may go: only the library's data directory, by variable.
DATA_PREFIXES = ("${VIVIDLY_CLIP_DATA}/", "$VIVIDLY_CLIP_DATA/")
MEMORY_STORE = ":memory:"
INDEX_FILE = "rules.yaml"


def _under_data(value: Any, subdir: str) -> bool:
    text = str(value)
    return any(text.startswith(prefix + subdir) for prefix in DATA_PREFIXES)


def check_profile(path: Path, text: str, doc: Any) -> list[str]:
    """Every finding for one profile; empty when it is not a spool-writing profile."""
    if not isinstance(doc, Mapping):
        return []
    clips = doc.get("clips")
    if not isinstance(clips, Mapping) or "spool_dir" not in clips:
        return []
    findings: list[str] = []
    rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path

    if "verify" in doc:
        findings.append(
            f"{rel}: carries a `verify` block; a spool-writing profile runs no verifier"
        )
    lowered = text.lower()
    for needle in FORBIDDEN_TEXT:
        if needle in lowered:
            findings.append(
                f"{rel}: mentions {needle!r}; that head and its vocabulary may not appear"
            )

    rules = doc.get("rules")
    enabled = rules.get("enabled") if isinstance(rules, Mapping) else None
    if not enabled:
        findings.append(
            f"{rel}: omits `rules.enabled`; the allowlist is mandatory with `clips.spool_dir`"
        )
    elif isinstance(enabled, str) or not isinstance(enabled, list):
        findings.append(f"{rel}: `rules.enabled` must be a list of rule ids")
    else:
        outside = sorted(set(map(str, enabled)) - ALLOWED_RULES)
        if outside:
            findings.append(
                f"{rel}: `rules.enabled` names {outside}; only {sorted(ALLOWED_RULES)} may run"
            )
    if isinstance(rules, Mapping):
        for item in rules.get("include") or []:
            if str(item).endswith(INDEX_FILE):
                findings.append(
                    f"{rel}: includes {item}; the index carries every head, include the per-rule files"
                )
    if "rules_config" in doc:
        findings.append(
            f"{rel}: uses `rules_config`; state `rules.include` with the per-rule files instead"
        )

    store = doc.get("store") or {}
    store_path = store.get("path") if isinstance(store, Mapping) else None
    if store_path is None:
        findings.append(
            f"{rel}: omits `store.path`; it must be `:memory:` or under ${{VIVIDLY_CLIP_DATA}}/pipeline"
        )
    elif str(store_path) != MEMORY_STORE and not _under_data(store_path, "pipeline"):
        findings.append(
            f"{rel}: `store.path` {store_path!r} is on disk outside ${{VIVIDLY_CLIP_DATA}}/pipeline"
        )

    reassoc = doc.get("reassoc") or {}
    if isinstance(reassoc, Mapping) and bool(reassoc.get("persist", False)):
        findings.append(f"{rel}: `reassoc.persist` is true; re-association may not persist")
    reid = doc.get("reid")
    if isinstance(reid, Mapping) and bool(reid.get("enabled", True)):
        findings.append(f"{rel}: enables `reid`; a spool-writing profile runs no re-identification")

    output = doc.get("output") or {}
    out_dir = output.get("dir") if isinstance(output, Mapping) else None
    if out_dir is None or not _under_data(out_dir, "pipeline"):
        findings.append(
            f"{rel}: `output.dir` {out_dir!r} must be ${{VIVIDLY_CLIP_DATA}}/pipeline; nothing may go under outputs/"
        )
    for where, value in _string_values(doc):
        if re.match(r"^(\./)?outputs(/|$)", value) or "/outputs/" in value:
            findings.append(
                f"{rel}: {where} names the outputs/ tree; a spool-writing profile writes nothing there"
            )
    if not _under_data(clips.get("spool_dir"), "spool"):
        findings.append(f"{rel}: `clips.spool_dir` must be ${{VIVIDLY_CLIP_DATA}}/spool")
    return findings


def _string_values(node: Any, where: str = "") -> list[tuple[str, str]]:
    """Every string value in the document with its dotted key path (comments excluded)."""
    out: list[tuple[str, str]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            out.extend(_string_values(value, f"{where}.{key}" if where else str(key)))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(_string_values(value, f"{where}[{i}]"))
    elif isinstance(node, str):
        out.append((where, node))
    return out


def _string_values(node: Any, where: str = "") -> list[tuple[str, str]]:
    """Every string value in the document with its dotted key path (comments excluded)."""
    out: list[tuple[str, str]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            out.extend(_string_values(value, f"{where}.{key}" if where else str(key)))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(_string_values(value, f"{where}[{i}]"))
    elif isinstance(node, str):
        out.append((where, node))
    return out


def check_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: unreadable ({exc})"]
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [f"{path}: not valid YAML ({exc})"]
    return check_profile(path, text, doc)


def main(argv: list[str]) -> int:
    targets = [Path(a).resolve() for a in argv] if argv else sorted(REPO_ROOT.glob(PROFILE_GLOB))
    findings: list[str] = []
    for path in targets:
        if path.suffix not in {".yaml", ".yml"}:
            continue
        findings.extend(check_file(path))
    if findings:
        print("clip library profile guard: FAIL", file=sys.stderr)
        for line in findings:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"clip library profile guard: ok ({len(targets)} file(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
