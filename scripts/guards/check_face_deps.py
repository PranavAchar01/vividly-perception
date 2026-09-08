#!/usr/bin/env python3
"""No facial-recognition code or dependencies, anywhere, under POLICY.md, rule 2.

Five checks, all hard failures:
  1. no import of any denied module in src/, scripts/, tests/, configs/
  2. no access to a denied cv2 face attribute in those same files
  3. no string literal naming a face-model artefact in those same files
  4. no denied distribution name in pyproject.toml's dependency tables
  5. no denied distribution name anywhere in uv.lock, which is where a face-biometric
     package arriving transitively, as another package's dependency, would show up first

Usage:
    uv run python scripts/guards/check_face_deps.py          # scan the repo, pyproject, uv.lock
    uv run python scripts/guards/check_face_deps.py FILE...  # pre-commit mode
    uv run python scripts/guards/check_face_deps.py X.lock   # scan a specific lockfile
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Import roots (what appears in `import X`).
DENIED_MODULES = frozenset(
    {
        "face_recognition",
        "insightface",
        "deepface",
        "retinaface",
        "retina_face",
        "facenet",
        "facenet_pytorch",
        "mtcnn",
        "dlib",
        "mediapipe",
        "face_alignment",
        "facexlib",
        "arcface",
    }
)
# Distribution names (what appears in pyproject.toml).
DENIED_DISTS = frozenset(
    {
        "face-recognition",
        "face_recognition",
        "insightface",
        "deepface",
        "retina-face",
        "retinaface",
        "facenet-pytorch",
        "mtcnn",
        "dlib",
        "dlib-bin",
        "mediapipe",
        "face-alignment",
        "facexlib",
    }
)
SCAN_ROOTS = ("src", "scripts", "tests", "configs")
REPO_ROOT = Path(__file__).resolve().parents[2]
_DEP_LINE = re.compile(r'^\s*"([A-Za-z0-9._-]+)\s*[<>=!~\[;]?', re.MULTILINE)
# uv.lock package blocks start with an unindented `name = "..."`; the indented ones inside
# a [package.dependencies] list are edges to those blocks, so column 0 is the full set.
_LOCK_NAME = re.compile(r'^name = "([A-Za-z0-9._-]+)"', re.MULTILINE)


def _rel(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


# OpenCV ships face detection and recognition inside packages this project legitimately
# depends on, since opencv-contrib is a hard dependency of rtmlib. A module deny-list
# cannot see cv2.face, FaceDetectorYN or FaceRecognizerSF usage, so those attribute
# accesses are denied here by name.
# String-literal model names catch cv2.dnn loads of face networks.
DENIED_CV2_ATTRS = frozenset(
    {
        "face",  # cv2.face.* (contrib recognizer module)
        "FaceDetectorYN",
        "FaceRecognizerSF",
        "createFaceDetectionModel",
        "createFaceRecognitionModel",
    }
)
_DENIED_MODEL_NAME = re.compile(r"yunet|sface|retinaface|arcface|facenet", re.IGNORECASE)


def module_violations(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [
                (node.lineno, a.name) for a in node.names if a.name.split(".")[0] in DENIED_MODULES
            ]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] in DENIED_MODULES
        ):
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.Attribute) and node.attr in DENIED_CV2_ATTRS:
            # cv2.face.LBPHFaceRecognizer_create, cv2.FaceDetectorYN.create, ...
            found.append((node.lineno, f"cv2 attribute '{node.attr}'"))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.endswith((".onnx", ".caffemodel", ".pb", ".tflite"))
            and _DENIED_MODEL_NAME.search(node.value)
        ):
            found.append((node.lineno, f"face model artifact {node.value!r}"))
    return found


def pyproject_violations() -> list[str]:
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return []
    text = pyproject.read_text(encoding="utf-8")
    return sorted({m for m in _DEP_LINE.findall(text) if m.lower() in DENIED_DISTS})


def lock_violations(lock: Path | None = None) -> list[str]:
    """Denied distributions pinned anywhere in a uv lockfile, transitive ones included.

    pyproject.toml records only what was requested directly. A face-biometric package that
    arrives as another package's dependency is invisible there and fully installed anyway,
    so the lock is the surface that actually answers whether it is in the environment.
    """
    path = (REPO_ROOT / "uv.lock") if lock is None else lock
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return sorted({m for m in _LOCK_NAME.findall(text) if m.lower() in DENIED_DISTS})


def iter_targets(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a) for a in argv if a.endswith(".py")]
    return sorted(p for root in SCAN_ROOTS for p in (REPO_ROOT / root).rglob("*.py"))


def iter_locks(argv: list[str]) -> list[Path]:
    return [Path(a) for a in argv if a.endswith(".lock")] or [REPO_ROOT / "uv.lock"]


def main(argv: list[str]) -> int:
    failed = False
    for path in iter_targets(argv):
        for lineno, module in module_violations(path):
            print(
                f"Face-recognition violation {_rel(path)}:{lineno}: imports '{module}'. "
                "POLICY.md, rule 2 forbids face detection, embedding or matching anywhere in "
                "this repository.",
                file=sys.stderr,
            )
            failed = True
    for dist in pyproject_violations():
        print(
            f"Face-recognition violation pyproject.toml: declares dependency '{dist}'. "
            "POLICY.md, rule 2 forbids face-biometric dependencies anywhere in this "
            "repository.",
            file=sys.stderr,
        )
        failed = True
    for lock in iter_locks(argv):
        for dist in lock_violations(lock):
            print(
                f"Face-recognition violation {_rel(lock)}: locks distribution '{dist}'. "
                "POLICY.md, rule 2 forbids face-biometric dependencies, transitive ones "
                "included. Find the package that pulls it in and drop that package.",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
