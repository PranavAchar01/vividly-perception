#!/usr/bin/env python3
"""Name guard: no new fisheye reference enters the live tree.

The no-fisheye rule retires the domain from the product. No fisheye dataset is fetched,
re-downloaded or licence-requested; no fisheye imagery, real or rendered, enters any training
or evaluation manifest in any lane; and the quarantined artefacts, the overhead-fisheye
corpora CEPDOF, HABBOF and FRIDA, the rendered fisheye corpus, and every fisheye-trained
weight, are history rather than inputs. The rule is stated in POLICY.md under rule 1.

This guard is the mechanical half of that rule, in the same shape the face-biometric rule is
enforced by check_face_deps.py. It cannot be a plain deny-list, because the tree still carries
records that have to be able to name what was retired: the licence provenance of a corpus, a
comparison whose arms are recorded as barred, the guards themselves. Failing on those would
fail the tree as checked in, which is a broken build rather than a guard.

So the rule enforced here is that no new reference appears, not that no reference exists:

  * scripts/guards/no_fisheye_allowlist.txt is the inventory of the files that already carried
    a reference. Those files are skipped in full.
  * Any other file under the scan surface that names a banned token, in its contents or in its
    own path, is a violation.

A new fisheye configuration, a new fisheye script, a new fisheye dataset fetch in a workflow,
or a reference newly added to a clean live-lane file therefore all fail, while the records keep
building. A file is exempted as a whole rather than line by line because a per-line baseline
rots the moment anything above the line moves, and a baseline that silently stops matching is
the one failure mode this rule must not have.

Extending the inventory is an owner decision and never the way to land a change. The fix for a
violation is to remove the reference, and the guard says so on every failure.

Scan surface: configs/, scripts/, src/, tests/ and .github/, which is code, configuration,
fixtures and continuous integration, everywhere a reference could go live. The document tree is
deliberately not scanned: the policy, the model record and the licence notes have to be able to
say the word in order to retire it, and no manifest is ever loaded from there. The output and
data trees are outside the roots because they are outside version control. third_party/ is in
version control but is also outside the roots, because it holds vendored upstream configuration
and patches rather than code this repository authors.

Matching is case-insensitive and word-shaped. The text is split into alphanumeric pieces at
every separator and at each camel-case transition, so `coco_fisheye_render`, `FisheyeModel`,
`CEPDOF`, `bu_neg_habbof_v2` and `crops_frida` all match, while `Friday` does not: `frida` is a
real collision risk and a bare substring test would fire on every weekday. A second pass
squashes separators out entirely for the tokens where that is unambiguous, which catches the
`fish-eye` and `Fish Eye` spellings; `frida` is excluded from that pass by name, because
squashing is exactly what would make `Friday` match.

Usage:
    uv run python scripts/guards/check_no_fisheye.py           # scan the whole surface
    uv run python scripts/guards/check_no_fisheye.py FILE...   # pre-commit mode
    uv run python scripts/guards/check_no_fisheye.py --report  # audit: every reference, exempted
                                                               # or not; read-only, always exits 0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path("scripts/guards/no_fisheye_allowlist.txt")
POLICY_SECTION = "POLICY.md, rule 1"

#: What each banned token is, for the failure message. Keys are lowercase and matching
#: lowercases the text. These are the quarantined corpora plus the name of the retired domain
#: itself, which is what the render harness and the camera model were built on.
BANNED_TOKENS: dict[str, str] = {
    "cepdof": "quarantined overhead-fisheye corpus",
    "habbof": "quarantined overhead-fisheye corpus",
    "frida": "quarantined overhead-fisheye corpus",
    "coco_fisheye_render": "quarantined rendered fisheye corpus, deleted 2026-09-05",
    "fisheye": "the retired domain itself: the corpora, the render harness and the camera model",
}

#: Tokens whose separator-squashed form cannot occur inside an unrelated word, so a squashed
#: second pass is safe for them and catches `fish-eye`, `Fish Eye`, `FishEye`. `frida` is
#: deliberately absent: squashed, it matches `Friday`.
SQUASH_SAFE = frozenset({"cepdof", "habbof", "coco_fisheye_render", "fisheye"})

SCAN_ROOTS = ("configs", "scripts", "src", "tests", ".github")

#: The two files that must name these tokens in order to ban them. Excluded here rather than
#: through the inventory, because the inventory cannot sanely list itself.
SELF_FILES = frozenset(
    {"scripts/guards/check_no_fisheye.py", "scripts/guards/no_fisheye_allowlist.txt"}
)

SKIP_DIR_PARTS = frozenset(
    {"__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache", "node_modules"}
)
#: Contents unreadable as text. The path of such a file is still checked, which is the point:
#: a fisheye frame dropped into a fixture tree is caught by its filename alone.
BINARY_SUFFIXES = frozenset(
    {
        ".avi",
        ".bin",
        ".db",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".mov",
        ".mp4",
        ".npy",
        ".npz",
        ".onnx",
        ".parquet",
        ".pdf",
        ".png",
        ".pt",
        ".pth",
        ".pyc",
        ".so",
        ".sqlite",
        ".sqlite3",
        ".tar",
        ".ttf",
        ".webm",
        ".woff",
        ".woff2",
        ".zip",
    }
)

#: A file usually violates on one idea, not on forty lines of it. Report the first few and
#: count the rest, so the explanation below stays visible instead of scrolling away.
MAX_REPORTED_LINES = 5

#: One alphanumeric piece: a capitals run such as CEPDOF, a capitalised word such as Fisheye,
#: a lowercase run, or a digit run. Splitting digits off letters is what makes `bu_neg_cepdof_v2`
#: and `frida1` match; keeping a capitals run whole is what keeps `FRIDAY` from matching.
_PIECE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")


class Hit(NamedTuple):
    """One banned reference. `lineno` 0 means the file's own path, not a line of content."""

    lineno: int
    tokens: tuple[str, ...]


class AllowlistError(Exception):
    """The inventory itself is malformed."""


def _rel(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def _pieces(text: str) -> str:
    """`text` as underscore-joined lowercase word pieces, wrapped in underscores.

    'coco_train_fisheye.yaml' becomes '_coco_train_fisheye_yaml_', so a token is found by
    searching for '_<token>_' and a longer word that merely contains a token, such as
    'Friday', never matches.
    """
    return "_" + "_".join(m.group(0).lower() for m in _PIECE.finditer(text)) + "_"


def _squash(text: str) -> str:
    """Every non-alphanumeric removed, lowercased: `fish-eye` and `Fish Eye` become `fisheye`."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def tokens_in(text: str) -> tuple[str, ...]:
    """Banned tokens present in `text`, sorted, deduplicated."""
    pieces = _pieces(text)
    found = {token for token in BANNED_TOKENS if f"_{token}_" in pieces}
    squashed = _squash(text)
    found |= {token for token in SQUASH_SAFE if _squash(token) in squashed}
    return tuple(sorted(found))


def load_allowlist() -> frozenset[str]:
    """The inventory: exact repository-relative posix paths, one per line, `#` comments.

    Exact paths only. A pattern would let one line exempt a whole tree, including files that do
    not exist yet, which is precisely the escape hatch this rule must not have, so `*`, `?` and
    `[` are refused outright, and so are absolute paths, parent traversal and directories.
    """
    path = REPO_ROOT / ALLOWLIST
    if not path.exists():
        raise AllowlistError(f"{ALLOWLIST} is missing; it is the reference inventory")
    entries: dict[str, int] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{ALLOWLIST}:{lineno}"
        if any(ch in line for ch in "*?["):
            raise AllowlistError(
                f"{where}: '{line}' contains a wildcard. Entries are exact file paths; a "
                "pattern would exempt files that do not exist yet."
            )
        if line.startswith("/") or ".." in Path(line).parts:
            raise AllowlistError(f"{where}: '{line}' must be a repo-relative path without '..'")
        if line in entries:
            raise AllowlistError(f"{where}: '{line}' is already listed on line {entries[line]}")
        if (REPO_ROOT / line).is_dir():
            raise AllowlistError(f"{where}: '{line}' is a directory; list the files themselves")
        entries[line] = lineno
    return frozenset(entries)


def _read_text(path: Path) -> str | None:
    """File contents as text, or None when there are none to read."""
    if path.suffix.lower() in BINARY_SUFFIXES:
        return None
    try:
        return path.read_bytes().decode("utf-8", errors="ignore")
    except OSError:
        return None


def scan_file(path: Path) -> list[Hit]:
    """Every banned reference in `path`: its own name first, then its lines."""
    rel = _rel(path).as_posix()
    hits: list[Hit] = []
    if found := tokens_in(rel):
        hits.append(Hit(0, found))
    text = _read_text(path)
    if text is None:
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        if found := tokens_in(line):
            hits.append(Hit(lineno, found))
    return hits


def in_scan_surface(path: Path) -> bool:
    rel = _rel(path)
    return bool(rel.parts) and rel.parts[0] in SCAN_ROOTS


def iter_targets(argv: list[str]) -> list[Path]:
    """Pre-commit passes filenames; with none, walk the whole scan surface."""
    if argv:
        return [p for p in map(Path, argv) if in_scan_surface(p) and p.is_file()]
    targets: list[Path] = []
    for root in SCAN_ROOTS:
        targets += sorted(p for p in (REPO_ROOT / root).rglob("*") if p.is_file())
    return [p for p in targets if not (SKIP_DIR_PARTS & set(_rel(p).parts))]


def _explain() -> None:
    print(
        "\n"
        f"  The no-fisheye rule, {POLICY_SECTION}. No fisheye dataset is fetched,\n"
        "  re-downloaded or licence-requested; no fisheye imagery, real or rendered, enters a\n"
        "  training or evaluation manifest in any lane; the fisheye camera model and the render\n"
        "  harness are retired and produce nothing new.\n"
        "\n"
        "  The fix is to remove the reference.\n"
        "\n"
        "  The files above are not in the inventory of what already carried a reference\n"
        f"  ({ALLOWLIST}). That inventory is a record, not a\n"
        "  configuration knob: adding a path to it is an owner decision and never the way to\n"
        "  land a change. If an entry is genuinely missing, stop and ask.",
        file=sys.stderr,
    )


def _report(allowed: frozenset[str]) -> int:
    """Audit mode: every fisheye reference in the tree, exempted or not. Read-only."""
    listed: list[str] = []
    unlisted: list[str] = []
    for path in iter_targets([]):
        rel = _rel(path).as_posix()
        if rel in SELF_FILES:
            continue
        hits = scan_file(path)
        if not hits:
            continue
        tokens = sorted({t for hit in hits for t in hit.tokens})
        line = f"{rel}  [{','.join(tokens)}]  {len(hits)} hit(s)"
        (listed if rel in allowed else unlisted).append(line)
    print(f"# exempted by {ALLOWLIST} ({len(listed)} file(s))")
    print("\n".join(listed))
    print(f"\n# not exempted ({len(unlisted)} file(s)), these fail the guard")
    print("\n".join(unlisted) if unlisted else "(none)")
    return 0


def _stale(allowed: frozenset[str]) -> list[str]:
    """Inventory entries that no longer need to be there.

    Advisory, never a failure: removing a reference is the outcome this rule wants and must not
    break the build. A stale entry does keep a now-clean file permanently unscanned, so it is
    reported for pruning."""
    notes: list[str] = []
    for entry in sorted(allowed):
        path = REPO_ROOT / entry
        if not path.exists():
            notes.append(f"{entry}: file no longer exists")
        elif not scan_file(path):
            notes.append(f"{entry}: no fisheye reference left")
    return notes


def main(argv: list[str]) -> int:
    report_only = "--report" in argv
    files = [a for a in argv if not a.startswith("-")]
    unknown = [a for a in argv if a.startswith("-") and a != "--report"]
    if unknown:
        print(f"check_no_fisheye: unknown option(s) {' '.join(unknown)}", file=sys.stderr)
        return 2

    try:
        allowed = load_allowlist()
    except AllowlistError as exc:
        print(f"No-fisheye inventory error {exc}", file=sys.stderr)
        return 2

    if report_only:
        return _report(allowed)

    failed = False
    for path in iter_targets(files):
        rel = _rel(path).as_posix()
        if rel in SELF_FILES or rel in allowed:
            continue
        hits = scan_file(path)
        for hit in hits[:MAX_REPORTED_LINES]:
            where = "its filename" if hit.lineno == 0 else "this line"
            why = "; ".join(f"'{t}' = {BANNED_TOKENS[t]}" for t in hit.tokens)
            print(f"No-fisheye violation {rel}:{hit.lineno}: {where} names {why}", file=sys.stderr)
        if len(hits) > MAX_REPORTED_LINES:
            print(
                f"No-fisheye violation {rel}: and "
                f"{len(hits) - MAX_REPORTED_LINES} more reference(s) in this file",
                file=sys.stderr,
            )
        failed |= bool(hits)

    if failed:
        _explain()
        return 1

    if not files:  # full-scan mode is the only mode that can judge the whole list
        for note in _stale(allowed):
            print(f"check_no_fisheye: stale inventory entry, {note}; prune it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
