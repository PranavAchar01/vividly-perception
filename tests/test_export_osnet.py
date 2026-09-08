"""scripts/reid/export_osnet_onnx.py: the refusals, and the promise that it never downloads.

Deliberately cheap and torch-free. Every case here is reachable before the script imports
torchreid, which is the point: the licence tripwire and the "this is a local path, not a
URL" check are argument validation, so they hold in the mac dev fork, in linux CI, and in
any environment where the export itself could not possibly run.

Nothing in this file builds a model, reads a checkpoint, or writes an ONNX file. The export
this script performed is the initialisation export: `osnet_x1_0_imagenet.pth`, sha256
`fe2d63f9157c28a4a8d8ca29bec12d5b2988ac0346d712025789ea9174968e79`, fetched 2026-08-08 from
the canonical torchreid distribution and exported 2026-08-09 at opset 13 to ONNX sha256
`73e76e2d0003f6b31eb1d47ecdbe991faf1327704599fc70c449079e3c80e6cf`. The lane's shipped
`final_models/reid/model.onnx`, sha256
`60164a9945445efbc5ea3d2abafb3fe0d78f1bcd2b4616ffd47634d87ef808a7` at opset 17 from
checkpoint `66623239`, was written by the trainer's own export function, not by this script.
The export command is in TRAINING.md, "Re-identification: phase A", under its "Commands"
block; the weight provenance is in final_models/reid/README.md, "Identity and provenance".
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "reid" / "export_osnet_onnx.py"


def _load_script() -> ModuleType:
    """scripts/ is not a package, so load the file by path (same as test_download_dataset)."""
    spec = importlib.util.spec_from_file_location("export_osnet_onnx", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


F = _load_script()


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )


# =============================================================== the CLI contract
def test_help_exits_zero_and_states_the_no_download_rule() -> None:
    proc = _run("--help")
    assert proc.returncode == 0, proc.stderr
    assert "--weights" in proc.stdout
    assert "Never downloads" in proc.stdout
    assert "docs/LICENCE_ALLOWLIST.md" in proc.stdout


def test_weights_is_required() -> None:
    proc = _run()
    assert proc.returncode == F.EXIT_USAGE  # argparse's own code
    assert "--weights" in proc.stderr


# =============================================================== licence refusals
# The paths below are the banned literals themselves, which is the only way to prove the
# refusal fires. scripts/guards/check_reid_weight_sources.py honours its `reid-guard: allow`
# pragma under tests/ for exactly this case; every such line carries it.
@pytest.mark.parametrize(
    ("path", "token"),
    [
        ("/data/luperson/osnet_x1_0.pth", "luperson"),  # reid-guard: allow
        ("/data/LUPerson/OSNet.pth", "luperson"),  # reid-guard: allow
        ("./solider_swin_base.pth", "solider"),  # reid-guard: allow
        ("./SOLIDER-swin.pth", "solider"),  # reid-guard: allow
        ("weights/transreid_ssl_vit.pth", "transreidssl"),  # reid-guard: allow
        ("weights/TransReID-SSL.vit.pth", "transreidssl"),  # reid-guard: allow
        ("weights/TransReIDSSL-vit.pth", "transreidssl"),  # reid-guard: allow
        ("weights/LU-Person_r50.pth", "luperson"),  # reid-guard: allow
        ("~/hf/kadirnar/osnet/osnet_x1_0.pth", "kadirnar"),  # reid-guard: allow
    ],
)
def test_banned_lineage_exits_six(path: str, token: str) -> None:
    proc = _run("--weights", path)
    assert proc.returncode == F.EXIT_LICENCE
    assert "LICENCE REFUSAL" in proc.stderr
    assert token in proc.stderr
    assert "docs/LICENCE_ALLOWLIST.md" in proc.stderr


def test_licence_check_runs_before_the_file_is_looked_for(tmp_path: Path) -> None:
    """A banned name that also does not exist is 6, not 4.

    Ordering is the whole safety property: if existence were checked first, the refusal
    would depend on whether the operator had already downloaded the file.
    """
    missing = tmp_path / "luperson_osnet.pth"  # reid-guard: allow
    assert not missing.exists()
    proc = _run("--weights", str(missing))
    assert proc.returncode == F.EXIT_LICENCE


def test_banned_lineage_in_the_output_path_is_refused_too(tmp_path: Path) -> None:
    weights = tmp_path / "osnet_x1_0.pth"
    weights.write_bytes(b"not a real checkpoint")
    out = tmp_path / "solider" / "osnet.onnx"  # reid-guard: allow
    proc = _run("--weights", str(weights), "--out", str(out))
    assert proc.returncode == F.EXIT_LICENCE
    assert not out.exists()
    assert not out.parent.exists()


def test_a_refusal_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "out" / "osnet.onnx"
    proc = _run("--weights", "/data/luperson/osnet.pth", "--out", str(out))  # reid-guard: allow
    assert proc.returncode == F.EXIT_LICENCE
    assert list(tmp_path.iterdir()) == []


# =============================================================== weights handling
def test_missing_weights_path_is_nonzero(tmp_path: Path) -> None:
    proc = _run("--weights", str(tmp_path / "nope.pth"))
    assert proc.returncode != 0
    assert proc.returncode == F.EXIT_WEIGHTS
    assert "does not exist" in proc.stderr


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/osnet_x1_0.pth",
        "http://example.invalid/osnet.pth",
        "s3://bucket/osnet.pth",
    ],
)
def test_a_url_is_refused_rather_than_fetched(url: str) -> None:
    proc = _run("--weights", url)
    assert proc.returncode == F.EXIT_WEIGHTS
    assert "never downloads" in proc.stderr


def test_a_directory_is_not_a_checkpoint(tmp_path: Path) -> None:
    proc = _run("--weights", str(tmp_path))
    assert proc.returncode == F.EXIT_WEIGHTS
    assert "not a file" in proc.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("torchreid") is not None,
    reason="torchreid is installed here, so the graceful-degradation path is unreachable",
)
def test_missing_torchreid_names_the_fork_that_has_it(tmp_path: Path) -> None:
    weights = tmp_path / "osnet_x1_0.pth"
    weights.write_bytes(b"not a real checkpoint")
    proc = _run("--weights", str(weights))
    assert proc.returncode == F.EXIT_NO_TORCHREID
    assert "curate" in proc.stderr
    assert F.TORCHREID_GIT_REV[:12] in proc.stderr


# =============================================================== pure helpers
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TransReID-SSL", ["transreidssl"]),  # reid-guard: allow
        ("transreid ssl", ["transreidssl"]),  # reid-guard: allow
        ("TransReID.SSL", ["transreidssl"]),  # reid-guard: allow
        ("TransReIDSSL", ["transreidssl"]),  # reid-guard: allow
        ("kadirnar/osnet", ["kadirnar"]),  # reid-guard: allow
        ("LUPerson-NL", ["luperson"]),  # reid-guard: allow
        ("LU-Person", ["luperson"]),  # reid-guard: allow
        ("luperson_solider.pth", ["luperson", "solider"]),  # reid-guard: allow
        ("osnet_x1_0_imagenet.pth", []),
        ("osnet_x1_0_market1501.pth", []),  # eval-only by posture, not blocked by name
    ],
)
def test_denied_lineage_tokens(text: str, expected: list[str]) -> None:
    assert F.denied_lineage_tokens(text) == expected


def test_check_lineage_raises_with_the_allowlist_citation() -> None:
    with pytest.raises(F.LicenceRefusal, match=re.escape("docs/LICENCE_ALLOWLIST.md")):
        F.check_lineage("--weights", "solider.pth")  # reid-guard: allow
    F.check_lineage("--weights", "osnet_x1_0_imagenet.pth")  # no raise


def test_export_geometry_is_the_osnet_one() -> None:
    assert F.INPUT_SHAPE == (1, 3, 256, 128)
    assert F.DEFAULT_VARIANT == "osnet_x1_0"
    # onnxruntime supports every opset from 7 up, so the pin is conservative by design.
    assert 7 <= F.DEFAULT_OPSET <= 17


def test_the_pinned_rev_is_a_full_sha_not_a_branch() -> None:
    assert len(F.TORCHREID_GIT_REV) == 40
    assert set(F.TORCHREID_GIT_REV) <= set("0123456789abcdef")
    assert F.TORCHREID_GIT_URL.endswith("KaiyangZhou/deep-person-reid.git")


# =============================================================== source-level guarantees
def test_module_has_no_top_level_heavy_imports() -> None:
    """torch, torchreid and onnx stay inside functions, or linux CI dies at collection."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    heavy = {"torch", "torchreid", "onnx", "numpy", "cv2"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not {a.name.split(".")[0] for a in node.names} & heavy, ast.dump(node)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in heavy, node.module


def _keyword_values(tree: ast.AST, name: str) -> list[object]:
    """Every literal passed as `name=<literal>` anywhere in the file."""
    return [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == name
        and isinstance(node.value, ast.Constant)
    ]


def test_the_script_has_no_way_to_fetch_anything() -> None:
    """No downloader, and torchreid's own model-zoo auto-fetch is never switched on."""
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    downloaders = {"urllib", "requests", "gdown", "httpx", "wget", "boto3"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not {a.name.split(".")[0] for a in node.names} & downloaders
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in downloaders
    assert _keyword_values(tree, "pretrained") == [False]
    assert "load_state_dict_from_url" not in source


def test_torch_load_is_never_trusted_with_arbitrary_pickles() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assert _keyword_values(tree, "weights_only") == [True]
