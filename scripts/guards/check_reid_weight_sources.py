#!/usr/bin/env python3
"""Weight-provenance guard: a banned re-identification weight source never enters the tree.

Two weight families are permanently out of bounds. They are licence traps rather than quality
problems, so the enforcement sits upstream of any download: the moment a locator, a model-hub
repository identifier, a checkpoint filename or a fetcher target naming one of them lands in
the repository, this guard fails. The register of what is admitted and what is refused, with
the primary sources for each, is docs/LICENCE_ALLOWLIST.md under the heading
"Re-identification weight sources".

Scanned surface: pyproject.toml, uv.lock, the root Makefile, and every file under scripts/,
configs/, src/, tests/ and .github/, both file contents and the file's own path, because a
checkpoint can arrive as a filename alone. The workflow tree and the Makefile are in scope
because a workflow step or a make recipe fetching a banned checkpoint is a download like any
other. The document tree is deliberately not scanned: the allowlist names these sources on
purpose, in order to ban them.

The deny-list keys are a superset of the tokens the model layer refuses in
src/vividly/models/reid_backend.py (BANNED_LINEAGE_TOKENS + BANNED_SOURCE_TOKENS): the bare
owner "kadirnar" is a key in its own right, not only the joined "kadirnar/osnet", so a fetcher
that assembles the repo id from an owner variable, or a config that splits it across two keys,
cannot pass a guard the model layer would fail. tests/test_reid_weight_guard.py asserts that
superset relation so the two lists cannot drift apart.

The match is a case-insensitive substring test on the raw text, with %2F decoded first so a
percent-encoded Hugging Face path cannot slip past. Substring, not word-boundary, on
purpose: the `-NL` and `_nl` descendants of the research-only corpus must be caught too, and
a word-boundary match would let an underscored suffix through. A second, separator-free pass
squashes every non-alphanumeric out of both the line and the key, so separator games
(`TransReIDSSL`, `lu-person`, `kadirnar osnet`) cannot dodge a token either.

A locator split across two physical lines reassembles at load time but leaves every single
line clean, so line-by-line matching alone is not enough. Two extra passes close that gap and
run on every non-authoring file: a whole-file squashed pass (the file's prose-blanked text
with every separator, including the newlines, removed, so a YAML backslash continuation, a
folded scalar or a shell line-continuation rejoins before matching) reports the line the first
fragment sits on; and, for Python, adjacent string literals are joined the way implicit string
concatenation joins them at runtime (`"kadir" "nar/osnet"` is one string) and matched as one.

Two things are not provenance and are therefore not violations:

  1. Python prose. Nothing fetches a sentence. In a .py file comment text and string literals
     are blanked before matching unless the literal is fetch-shaped: it carries no internal
     whitespace, which is what a URL, an `owner/repo` id, a checkpoint filename or a bare owner
     about to be pasted into a repo id all look like and what a sentence never does. A docstring
     that names a family in passing, in an ordinary sentence, is prose;
     `"kadirnar/osnet"` is a repo id, and a bare `"kadirnar"` is an owner, and both fail with
     nothing else on the line. Whitespace, not a separator, is the test - demanding a separator
     was the hole a split owner walked through. The exemption is withdrawn outright for any
     comment line carrying a locator scheme or a checkpoint extension (RECIPE_HINTS), so a
     commented-out download recipe is still caught. An unparseable .py file gets the strict raw
     scan, never a free pass, and so does every non-Python file (YAML, TOML, uv.lock, Makefile,
     shell).
  2. Anything inside an authoring location, that is the trees scripts/guards/ and tests/, plus the
     two deny-list-defining files src/vividly/models/reid_backend.py and
     scripts/reid/export_osnet_onnx.py, when it is either a Python string literal or comment (a
     deny-list has to be able to write the exact token it expects to be refused:
     `("luperson", "solider")` as tuple data and `"weights/luperson_r50.pth"` as a
     parametrised test input are both the deny-list, not a fetch), or a line carrying the
     comment `reid-guard: allow`. That pragma is the escape for the non-Python files in the
     authoring trees, and it is honoured only inside an authoring location. Everywhere else it
     is inert and the line still fails, with a message saying so. That is the fail-closed
     half: a fetcher cannot annotate its way past this guard, and neither this file nor the
     deny-list it mirrors can flag itself.

Filenames are checked in every tree including the authoring ones, so a real banned checkpoint
parked under tests/fixtures/ is caught by its name whatever the code says.

Usage:
    uv run python scripts/guards/check_reid_weight_sources.py            # scan the whole surface
    uv run python scripts/guards/check_reid_weight_sources.py FILE...    # pre-commit mode
"""

from __future__ import annotations

import io
import sys
import tokenize
from pathlib import Path
from typing import NamedTuple

# Every key below is a literal this guard hunts for; every one of these lines therefore
# carries the pragma, because naming a banned string in order to ban it is the one use that
# is always legitimate. Keep keys lowercase - matching lowercases the haystack. The key set
# is a superset of reid_backend.BANNED_LINEAGE_TOKENS + BANNED_SOURCE_TOKENS: the bare owner
# "kadirnar" is a key on its own, not only the joined repo id, so an owner split from its repo
# (across variables, or across two YAML keys) cannot pass a guard the model layer would fail.
BANNED_SOURCES: dict[str, str] = {
    "kadirnar/osnet": "GPL-3.0-labelled HF repack of OSNet",  # reid-guard: allow
    "kadirnar": "owner of the GPL-3.0-labelled OSNet repack",  # reid-guard: allow
    "luperson": "research-only corpus, commercial use forbidden",  # reid-guard: allow
    "solider": "LUPerson-trained backbone, ship=never",  # reid-guard: allow
    # One canonical key: the squashed second pass makes it catch transreid_ssl,
    # TransReIDSSL, and every other separator variant too.
    "transreid-ssl": "LUPerson-trained backbone, ship=never",  # reid-guard: allow
    "pass-reid": "LUPerson-trained backbone, ship=never",  # reid-guard: allow
    "personvit": "LUPerson-trained backbone, ship=never",  # reid-guard: allow
    # Retracted 2019 over an IRB breach; every derivative and every weight whose
    # lineage passes through it is tainted (research/reid-data-licences.md).
    "dukemtmc": "retracted dataset, IRB breach; all derivatives tainted",  # reid-guard: allow
}

PRAGMA = "reid-guard: allow"
# The two trees whose job is to name the banned strings in order to ban them. Inside them
# the pragma works and Python literals are prose; outside them neither is true.
AUTHORING_TREES: tuple[Path, ...] = (Path("scripts/guards"), Path("tests"))
# Individual files outside those trees that define the deny-list by naming every banned token,
# exactly as scripts/guards/ and tests/ do: the model-layer refusal and the refusal in the ONNX
# exporter. They get the same latitude for the same reason, and nothing else outside the trees
# does. Keep this list in step with the deny-lists those files declare.
AUTHORING_FILES: tuple[Path, ...] = (
    Path("src/vividly/models/reid_backend.py"),
    Path("scripts/reid/export_osnet_onnx.py"),
)

SCAN_ROOTS = ("scripts", "configs", "src", "tests", ".github")
EXTRA_FILES = ("pyproject.toml", "uv.lock", "Makefile")
REPO_ROOT = Path(__file__).resolve().parents[2]

SKIP_DIR_PARTS = frozenset(
    {"__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache", "node_modules"}
)
# Contents unreadable as text. The path of such a file is still checked, which is the point:
# a banned checkpoint dropped into tests/fixtures/ is caught by its filename alone.
BINARY_SUFFIXES = frozenset(
    {
        ".avi",
        ".bin",
        ".gz",
        ".jpeg",
        ".jpg",
        ".mov",
        ".mp4",
        ".npy",
        ".npz",
        ".onnx",
        ".pdf",
        ".png",
        ".pt",
        ".pth",
        ".pyc",
        ".so",
        ".tar",
        ".webm",
        ".zip",
    }
)
# A Python comment or docstring line carrying one of these is not prose any more, it is a
# download recipe. Such a line keeps the strict treatment even though it cannot execute.
RECIPE_HINTS = (
    "://",
    ".pth",
    ".pt",
    ".onnx",
    ".ckpt",
    ".safetensors",
    ".pkl",
    ".tar",
    ".zip",
    ".bin",
)


class Hit(NamedTuple):
    """One banned reference. `lineno` 0 means the file's own path, not a line of content."""

    lineno: int
    literal: str
    stray_pragma: bool


def _rel(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def _normalise(text: str) -> str:
    """Lowercase, and decode the one encoding that hides a repo id inside a URL."""
    return text.lower().replace("%2f", "/")


def _squash(text: str) -> str:
    """Strip every non-alphanumeric, defeating separator variants of a banned key."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def banned_in_text(text: str) -> list[str]:
    """Banned source literals present in `text`, case-insensitively, sorted.

    Two passes: the literal key against the %2F-decoded lowercase line, then the
    separator-squashed key against the separator-squashed line.
    """
    haystack = _normalise(text)
    squashed = _squash(text)
    return sorted(
        literal for literal in BANNED_SOURCES if literal in haystack or _squash(literal) in squashed
    )


def is_authoring(path: Path) -> bool:
    """True inside the authoring trees (scripts/guards/, tests/) and for the two deny-list
    files: guard code, its negative tests, and the model/exporter refusals that must name
    every banned token in order to refuse it."""
    rel = _rel(path)
    if any(d == rel.parent or d in rel.parents for d in AUTHORING_TREES):
        return True
    return rel in AUTHORING_FILES


def in_scan_surface(path: Path) -> bool:
    """True for the EXTRA_FILES and anything under the scan roots."""
    rel = _rel(path)
    if rel.as_posix() in EXTRA_FILES:
        return True
    return bool(rel.parts) and rel.parts[0] in SCAN_ROOTS


def _read_text(path: Path) -> str | None:
    """File contents as text, or None when there are none to read. No size cap on purpose:
    skipping a file silently is the one failure mode a deny-list guard must not have.
    Undecodable bytes are dropped rather than raising, so a mislabelled binary still gets its
    ASCII strings read."""
    if path.suffix.lower() in BINARY_SUFFIXES:
        return None
    try:
        return path.read_bytes().decode("utf-8", errors="ignore")
    except OSError:
        return None


def _is_recipe(line: str) -> bool:
    """True when a line looks like a download recipe rather than prose about one."""
    low = _normalise(line)
    return any(hint in low for hint in RECIPE_HINTS)


def _string_inner(literal: str) -> str:
    """The text inside a Python string token: prefix letters and quotes stripped."""
    body = literal
    cut = 0
    while cut < len(body) and body[cut] in "rRbBuUfF":
        cut += 1
    if cut < len(body) and body[cut] in "\"'":
        body = body[cut:]
    for quote in ('"""', "'''", '"', "'"):
        if body.startswith(quote) and body.endswith(quote) and len(body) >= 2 * len(quote):
            return body[len(quote) : -len(quote)]
    return body


def _is_fetch_literal(text: str) -> bool:
    """True when a string is a fetch fragment rather than a sentence about one.

    The one thing every fetchable token shares - a URL, an `owner/repo` id, a checkpoint
    filename, and a bare owner like `kadirnar` about to be pasted into a repo id - is that it
    carries no internal whitespace. A sentence about a banned family always does. That single
    distinction is what lets a docstring or an error message name these families in an
    ordinary sentence without tripping the guard, while a bare owner literal still fails on a
    line with nothing else on it.

    The earlier form of this test also demanded a separator (`/`, `.`, `:`), which is exactly
    the hole a split owner walked through: `"kadirnar"` has no separator, so it read as prose.
    Whitespace alone is the honest signal; the deny-list-defining files that must write bare
    tokens as tuple data are handled by their authoring status, not by pretending those tokens
    are prose.
    """
    body = text.strip()
    return bool(body) and not any(ch.isspace() for ch in body)


def _squash_with_lines(lines: list[str]) -> tuple[str, list[int]]:
    """The whole text squashed to lowercase alphanumerics, plus a parallel list mapping each
    squashed character back to the 1-based line it came from.

    Squashing the whole file, not each line, is what rejoins a locator a loader would rejoin:
    the separators a continuation hides behind - a trailing backslash, a folded-scalar
    newline, the whitespace between bracketed fragments - are all non-alphanumeric and drop
    out, so the fragments become adjacent exactly as the loader sees them. The line map lets a
    whole-file-only hit still name a line: the one the first fragment sits on.
    """
    chars: list[str] = []
    line_map: list[int] = []
    for lineno, line in enumerate(lines, start=1):
        for ch in _normalise(line):
            if ch.isalnum():
                chars.append(ch)
                line_map.append(lineno)
    return "".join(chars), line_map


def _string_token_types() -> set[int]:
    types = {tokenize.STRING}
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)  # 3.12+ splits f-strings up
    if fstring_middle is not None:
        types.add(fstring_middle)
    return types


def _concat_hits(text: str) -> list[tuple[int, str]]:
    """Banned literals assembled by Python implicit string concatenation, with each run's line.

    Adjacent string literals separated only by whitespace or a comment are one string at
    runtime (`"kadir" "nar/osnet"` is `"kadirnar/osnet"`), so a locator can be split across
    tokens to keep every single fragment clean. Each maximal run of adjacent string tokens is
    joined and matched, and a hit is reported at the run's first line. Returns [] when the
    source does not tokenize; the whole-file squash still covers the same split. Non-authoring
    callers only - an authoring file writes deny-list literals on purpose.
    """
    string_types = _string_token_types()
    neutral = {tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}
    hits: list[tuple[int, str]] = []
    run: list[tokenize.TokenInfo] = []

    def flush() -> None:
        if len(run) >= 2:
            joined = "".join(_string_inner(tok.string) for tok in run)
            for literal in banned_in_text(joined):
                hits.append((run[0].start[0], literal))
        run.clear()

    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in string_types:
                run.append(tok)
            elif tok.type in neutral:
                continue
            else:
                flush()
        flush()
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    return hits


def python_prose_blanked(text: str, *, authoring: bool = False) -> list[str] | None:
    """`text` split into lines, with Python comments and non-fetch string literals blanked.

    Comments are blanked everywhere except on a download-recipe line. A string literal that
    carries a banned token is blanked (treated as prose) unless it is a fetch fragment, that is
    it has no internal whitespace - a URL, an `owner/repo` id, a filename or a bare owner - in
    which case it is kept for the scanner to fail on. That whitespace test is what protects a
    docstring or an error message naming these families in passing while still failing a bare
    `"kadirnar"`.

    With `authoring=True` (the trees scripts/guards/ and tests/, plus the two deny-list files)
    every banned literal counts as prose, fetch-shaped or not, because a deny-list has to name
    the exact token it refuses - `("luperson", "solider")` as tuple data,
    `"weights/luperson_r50.pth"` as a parametrised test input. Those locations are the only
    ones that get that latitude.

    Returns None when the source does not tokenize, so the caller falls back to the raw scan:
    an unparseable file is treated strictly, never waved through. tokenize is used rather
    than ast because it reports character columns for every literal, including each piece of
    an implicit concatenation and the interior of an f-string, where ast positions are either
    approximate or absent on 3.11.
    """
    lines = text.splitlines()
    string_types = _string_token_types()

    spans: list[tuple[int, int, int, int]] = []
    protected: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in string_types:
                if not banned_in_text(tok.string):
                    continue
                if not authoring and _is_fetch_literal(_string_inner(tok.string)):
                    protected.update(range(tok.start[0], tok.end[0] + 1))
                    continue
            elif tok.type != tokenize.COMMENT:
                continue
            spans.append((tok.start[0], tok.start[1], tok.end[0], tok.end[1]))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None

    for first, start_col, last, end_col in spans:
        for lineno in range(first, min(last, len(lines)) + 1):
            line = lines[lineno - 1]
            if lineno in protected or (not authoring and _is_recipe(line)):
                continue
            start = start_col if lineno == first else 0
            end = end_col if lineno == last else len(line)
            lines[lineno - 1] = line[:start] + " " * max(0, end - start) + line[end:]
    return lines


def scan_file(path: Path) -> list[Hit]:
    """Every banned reference in `path`: its name, its lines, then any cross-line reassembly."""
    authoring = is_authoring(path)
    hits: list[Hit] = [Hit(0, lit, False) for lit in banned_in_text(_rel(path).as_posix())]
    text = _read_text(path)
    if text is None:
        return hits
    raw = text.splitlines()
    is_py = path.suffix.lower() == ".py"
    blanked = python_prose_blanked(text, authoring=authoring) if is_py else None
    scanned = blanked if blanked is not None else raw

    seen: set[str] = set()
    for lineno, line in enumerate(scanned, start=1):
        literals = banned_in_text(line)
        if not literals:
            continue
        # The pragma lives in a comment, which the Python pass may just have blanked, so it
        # is always read off the original line.
        if PRAGMA in _normalise(raw[lineno - 1]):
            if authoring:
                continue
            hits += [Hit(lineno, literal, True) for literal in literals]
        else:
            hits += [Hit(lineno, literal, False) for literal in literals]
        seen.update(literals)

    # Authoring locations name these strings on purpose; the reassembly passes below would
    # only re-flag that deny-list data.
    if authoring:
        return hits

    # A locator split across adjacent Python string literals (implicit concatenation).
    if is_py and blanked is not None:
        for lineno, literal in _concat_hits(text):
            if literal not in seen:
                hits.append(Hit(lineno, literal, False))
                seen.add(literal)

    # A locator split across physical lines by any continuation a loader rejoins: squash the
    # whole prose-blanked file and match once, reporting the first fragment's line.
    squashed, line_map = _squash_with_lines(scanned)
    for literal in BANNED_SOURCES:
        if literal in seen:
            continue
        idx = squashed.find(_squash(literal))
        if idx != -1:
            hits.append(Hit(line_map[idx], literal, False))
            seen.add(literal)
    return hits


def iter_targets(argv: list[str]) -> list[Path]:
    """Pre-commit passes filenames; with none, walk the whole scan surface."""
    if argv:
        return [p for p in map(Path, argv) if in_scan_surface(p) and p.is_file()]
    targets = [REPO_ROOT / name for name in EXTRA_FILES]
    for root in SCAN_ROOTS:
        targets += sorted(p for p in (REPO_ROOT / root).rglob("*") if p.is_file())
    return [p for p in targets if p.exists() and not (SKIP_DIR_PARTS & set(_rel(p).parts))]


def main(argv: list[str]) -> int:
    failed = False
    for path in iter_targets(argv):
        for hit in scan_file(path):
            where = "its filename" if hit.lineno == 0 else "this line"
            note = (
                f" The '{PRAGMA}' pragma is honoured only under "
                f"{' and '.join(f'{d}/' for d in AUTHORING_TREES)}, and does nothing here."
                if hit.stray_pragma
                else ""
            )
            print(
                f"Re-identification weight-source violation {_rel(path)}:{hit.lineno}: "
                f"{where} names "
                f"'{hit.literal}' ({BANNED_SOURCES[hit.literal]}). This weight source is "
                f"banned outright; see docs/LICENCE_ALLOWLIST.md, "
                f"'Re-identification weight sources'.{note}",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
