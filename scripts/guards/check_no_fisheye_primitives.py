#!/usr/bin/env python3
"""Primitive guard: the OpenCV fisheye and warp primitives are banned outright.

The no-fisheye rule, POLICY.md, rule 1, bans the domain: no fisheye dataset is fetched or
requested, and no fisheye imagery, real or rendered, enters a training or evaluation manifest
in any lane. The sibling guard, scripts/guards/check_no_fisheye.py, enforces that on names, in
file contents and in file paths. This guard enforces it on the code that would produce the
imagery, which is a different surface and needs a different reading.

Banned anywhere under src/, scripts/ and tests/:

    cv2.fisheye.*                  the fisheye camera model and its calibration entry points
    cv2.remap                      the resampling call any distortion pipeline ends in
    cv2.initUndistortRectifyMap    the map builder that feeds remap

The map builder is banned with remap rather than separately: banning the resampling call while
leaving its map builder in place would retire half of one primitive. There is no allowlist and
no exempt directory. A rectification or distortion pipeline is not reintroduced under a new
name, and a change that needs one of these calls is a change to the rule, not to this file.

Reading is by syntax tree rather than by substring, so an attribute of a project configuration
object such as preset.fisheye.canvas is untouched while cv2.fisheye.initUndistortRectifyMap is
caught, an aliased import such as `import cv2 as cv` is followed, and `from cv2 import remap`
cannot launder the call.

Usage:
    uv run python scripts/guards/check_no_fisheye_primitives.py          # scan the whole surface
    uv run python scripts/guards/check_no_fisheye_primitives.py FILE...  # pre-commit mode
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

DENIED_MEMBERS = frozenset({"fisheye", "remap", "initUndistortRectifyMap"})
SCAN_ROOTS = ("src", "scripts", "tests")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _rel(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def _dotted(node: ast.Attribute) -> list[str] | None:
    """The parts of cv2.fisheye.initUndistortRectifyMap for that expression, else None."""
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    parts.reverse()
    return parts


def _cv2_aliases(tree: ast.AST) -> set[str]:
    """An aliased import makes a second root name for the same module."""
    aliases = {"cv2"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "cv2" and alias.asname:
                    aliases.add(alias.asname)
    return aliases


def violations(path: Path) -> list[tuple[int, str]]:
    """Every denied primitive in a file, as line number and expression."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # a file that does not parse is another check's failure
        print(f"{path}: skipped, syntax error at line {exc.lineno}", file=sys.stderr)
        return []

    aliases = _cv2_aliases(tree)
    found: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts = _dotted(node)
            if not parts or parts[0] not in aliases:
                continue
            for index, part in enumerate(parts[1:], start=1):
                if part in DENIED_MEMBERS:
                    found.add((node.lineno, ".".join(parts[: index + 1])))
                    break
        elif isinstance(node, ast.ImportFrom) and node.module:
            root, _, tail = node.module.partition(".")
            if root not in aliases:
                continue
            if tail.split(".")[0] in DENIED_MEMBERS:
                found.add((node.lineno, node.module))
                continue
            for alias in node.names:
                if alias.name in DENIED_MEMBERS:
                    found.add((node.lineno, f"{node.module}.{alias.name}"))
    return sorted(found)


def iter_targets(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a) for a in argv if a.endswith(".py")]
    return sorted(p for root in SCAN_ROOTS for p in (REPO_ROOT / root).rglob("*.py"))


def main(argv: list[str]) -> int:
    failed = False
    for path in iter_targets(argv):
        for lineno, expr in violations(path):
            print(
                f"Fisheye primitive {_rel(path)}:{lineno}: uses {expr!r}. The fisheye camera "
                "model and the render harness are retired, and the OpenCV fisheye and warp "
                "primitives are banned outright under POLICY.md, rule 1. The fix is to "
                "remove the call, never to add an exemption.",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
