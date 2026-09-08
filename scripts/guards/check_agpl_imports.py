#!/usr/bin/env python3
"""AGPL packages may be imported only from the quarantine module, under POLICY.md, rule 5.

Fails on `import ultralytics`, `from boxmot import ...`, or
`importlib.import_module("ultralytics")` anywhere except
src/vividly/eval/agpl_ref.py.

Usage:
    uv run python scripts/guards/check_agpl_imports.py          # scan src/, scripts/, tests/, configs/
    uv run python scripts/guards/check_agpl_imports.py FILE...  # pre-commit mode
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# torchreid is not in this set. It is MIT, it had only ever appeared here as a dependency of
# the AGPL tracker, and docs/LICENCE_ALLOWLIST.md admits it in the cuda and curate forks for
# the OSNet export and for work on a GPU. This guard covers AGPL code only. torchreid stays
# out of the shipped package through its own import-linter contract, and out of the macOS
# development fork through the uv extras.
AGPL_ROOTS = frozenset({"ultralytics", "boxmot", "yolov5", "yolov8"})
ALLOWED_FILES = frozenset({Path("src/vividly/eval/agpl_ref.py")})
SCAN_ROOTS = ("src", "scripts", "tests", "configs")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _rel(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def violations(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, module) for every AGPL import in `path`."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # a file that does not parse is another check's finding
        print(f"{path}: skipped (syntax error at line {exc.lineno})", file=sys.stderr)
        return []

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in AGPL_ROOTS:
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in AGPL_ROOTS:
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            # importlib.import_module("ultralytics") / __import__("boxmot")
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in {"import_module", "__import__"}:
                for arg in node.args:
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.split(".")[0] in AGPL_ROOTS
                    ):
                        found.append((node.lineno, arg.value))
    return found


def iter_targets(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a) for a in argv if a.endswith(".py")]
    return sorted(p for root in SCAN_ROOTS for p in (REPO_ROOT / root).rglob("*.py"))


def main(argv: list[str]) -> int:
    failed = False
    for path in iter_targets(argv):
        rel = _rel(path)
        if rel in ALLOWED_FILES:
            continue
        for lineno, module in violations(path):
            print(
                f"AGPL quarantine violation {rel}:{lineno}: imports '{module}'. "
                f"AGPL code is allowed only in {', '.join(sorted(str(p) for p in ALLOWED_FILES))}.",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
