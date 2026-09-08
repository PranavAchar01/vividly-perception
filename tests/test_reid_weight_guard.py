"""The banned-weight-source guard (scripts/guards/check_reid_weight_sources.py).

These are the regression tests for three confirmed red-team bypasses of that guard, plus the
anti-drift check that keeps its deny-list a superset of the model layer's:

  * A1: the bare owner. The old key set carried only the joined "kadirnar/osnet", so an owner
    split from its repo (across two YAML keys, or an owner variable pasted into the id) passed
    a guard the model layer would already fail. The bare owner is now a key, and a bare owner
    literal is treated as a fetch fragment (no internal whitespace), not prose.
  * A2: the scanned surface. A CI step or a make recipe curling a banned checkpoint used to be
    invisible; .github/ and the root Makefile are now in scope.
  * A3: cross-line reassembly. A locator split across two physical lines (Python implicit
    concatenation, a YAML backslash continuation) rejoins at load time; the whole-file squashed
    pass and the adjacent-literal join catch it.

Everything here runs with numpy alone: the guard imports nothing heavy, and the model-layer
deny-lists it mirrors cost only numpy to import, so this file runs on the CI fork that has no
onnxruntime and no cv2. No weights are fetched and no host is contacted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from vividly.models.reid_backend import BANNED_LINEAGE_TOKENS, BANNED_SOURCE_TOKENS

REPO = Path(__file__).resolve().parent.parent
GUARD = REPO / "scripts" / "guards" / "check_reid_weight_sources.py"


def _load_guard() -> ModuleType:
    """scripts/ is not a package, so load the guard by path (same as test_export_osnet)."""
    spec = importlib.util.spec_from_file_location("check_reid_weight_sources", GUARD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = _load_guard()


def _literals(tmp_path: Path, name: str, text: str) -> set[str]:
    """Banned literals scan_file reports for a file `name` containing `text`, written under
    tmp_path so the file is outside every authoring location and gets the strict treatment."""
    path = tmp_path / name
    path.write_text(text)
    return {hit.literal for hit in G.scan_file(path)}


# --------------------------------------------------------------- anti-drift (A1)
def test_guard_covers_every_reid_backend_token() -> None:
    """The guard's deny-list is a superset of reid_backend's: every lineage and source token
    the model layer refuses is caught here too, so the two lists cannot silently drift apart.
    This is the check that would have failed while only the joined 'kadirnar/osnet' was a key
    and the bare owner was not."""
    tokens = tuple(BANNED_LINEAGE_TOKENS) + tuple(BANNED_SOURCE_TOKENS)
    assert tokens, "reid_backend declared no banned tokens; the fixture is wrong"
    for token in tokens:
        assert G.banned_in_text(token), (
            f"guard does not catch reid_backend token {token!r}: the deny-lists have drifted, "
            "and a fetcher naming this token would pass the guard while the model layer fails it"
        )


def test_bare_source_owner_is_its_own_key() -> None:
    """The specific drift that opened A1: the bare owner has to be a key in its own right, not
    only a prefix of the joined repo id, so an owner split from its repo is caught on the line
    that names it."""
    for token in BANNED_SOURCE_TOKENS:
        assert token in G.BANNED_SOURCES


# --------------------------------------------------------------- A1: bare owner
def test_bare_owner_split_across_two_yaml_keys(tmp_path: Path) -> None:
    literals = _literals(tmp_path, "embedder.yaml", "embedder:\n  owner: kadirnar\n  repo: osnet\n")
    assert "kadirnar" in literals


def test_owner_variable_pasted_into_repo_id(tmp_path: Path) -> None:
    literals = _literals(
        tmp_path,
        "fetch.py",
        '_OWNER = "kadirnar"\nURL = f"https://huggingface.co/{_OWNER}/osnet/resolve/main/x.onnx"\n',
    )
    assert "kadirnar" in literals


@pytest.mark.parametrize(
    ("text", "is_fetch"),
    [
        ("kadirnar", True),  # bare owner, no separator: the A1 gap
        ("kadirnar/osnet", True),  # repo id
        ("weights/luperson_r50.pth", True),  # filename
        ("https://hf.co/x", True),  # url
        ("Under the rules every LUPerson-derived weight is banned", False),  # a sentence
        ("kadirnar osnet", False),  # has whitespace: reads as prose
        ("   ", False),  # blank
    ],
)
def test_is_fetch_literal_uses_whitespace_not_a_separator(text: str, is_fetch: bool) -> None:
    """A fetch fragment has no internal whitespace; a sentence about a banned family always
    does. Requiring a separator on top of that was what let a bare owner read as prose."""
    assert G._is_fetch_literal(text) is is_fetch


def test_prose_docstring_naming_families_is_not_a_violation(tmp_path: Path) -> None:
    """The other side of the whitespace test: a docstring or comment may name these families
    to explain the ban without tripping the guard, as long as it is a sentence."""
    text = (
        '"""Under the rules every LUPerson-derived weight (SOLIDER, TransReID-SSL) is banned."""\n'
        "# see docs/LICENCE_ALLOWLIST.md for the kadirnar OSNet repack rationale\n"
        "X = 1\n"
    )
    assert _literals(tmp_path, "doc.py", text) == set()


# --------------------------------------------------------------- A2: scan surface
def test_github_and_makefile_are_in_the_scan_surface() -> None:
    assert G.in_scan_surface(REPO / ".github" / "workflows" / "ci.yml")
    assert G.in_scan_surface(REPO / "Makefile")


def test_iter_targets_walks_github_and_the_makefile() -> None:
    """The full-tree walk (no filenames) must actually visit the widened surface, not just
    admit it: a CI curl is only caught if the workflow file is one of the targets."""
    targets = {G._rel(p).as_posix() for p in G.iter_targets([])}
    assert "Makefile" in targets
    assert any(t.startswith(".github/") for t in targets)


def test_ci_step_curling_a_banned_checkpoint_is_flagged(tmp_path: Path) -> None:
    literals = _literals(
        tmp_path,
        "ci.yml",
        "steps:\n  - run: curl -L https://huggingface.co/kadirnar/osnet/resolve/main/x.onnx\n",
    )
    assert {"kadirnar", "kadirnar/osnet"} & literals


def test_make_recipe_curling_a_banned_checkpoint_is_flagged(tmp_path: Path) -> None:
    literals = _literals(
        tmp_path,
        "Makefile",
        "fetch:\n\tcurl -L https://huggingface.co/kadirnar/osnet/resolve/main/x.onnx\n",
    )
    assert {"kadirnar", "kadirnar/osnet"} & literals


# --------------------------------------------------------------- A3: cross-line
def test_python_implicit_concatenation_reassembles(tmp_path: Path) -> None:
    """`"kadir" "nar/osnet"` is one string at runtime. Broken mid-owner, no single fragment
    carries the token, but the adjacent-literal join and the whole-file squash reassemble it."""
    path = tmp_path / "concat.py"
    path.write_text(
        "WEIGHTS = (\n"
        '    "https://huggingface.co/kadir"\n'
        '    "nar/osnet/resolve/main/osnet_x1_0.onnx"\n'
        ")\n"
    )
    hits = G.scan_file(path)
    literals = {hit.literal for hit in hits}
    assert {"kadirnar", "kadirnar/osnet"} & literals
    # Reported at the fragment's line, not as a filename hit (lineno 0).
    assert any(hit.lineno >= 1 for hit in hits)


def test_yaml_backslash_continuation_reassembles(tmp_path: Path) -> None:
    """A YAML double-quoted scalar with a trailing backslash rejoins the next line at load
    time. Split mid-lineage, each physical line is clean; the whole-file squash catches it."""
    literals = _literals(tmp_path, "cont.yaml", 'weights: "transreid\\\nssl_vit_base.pth"\n')
    assert "transreid-ssl" in literals


def test_unsplit_locator_on_one_line_still_caught(tmp_path: Path) -> None:
    """Guard against a regression where the reassembly passes replace, rather than add to, the
    line-by-line scan: a plain single-line locator must still be caught, at its line."""
    path = tmp_path / "plain.py"
    path.write_text('REPO = "kadirnar/osnet"\n')
    hits = G.scan_file(path)
    assert any(hit.literal == "kadirnar/osnet" and hit.lineno == 1 for hit in hits)


# --------------------------------------------------------------- authoring exemptions
def test_deny_list_defining_files_are_authoring() -> None:
    """The two files outside the authoring trees that define the deny-list by naming every
    banned token get authoring latitude, or the guard would flag the very lists it mirrors."""
    assert G.is_authoring(REPO / "src" / "vividly" / "models" / "reid_backend.py")
    assert G.is_authoring(REPO / "scripts" / "reid" / "export_osnet_onnx.py")
    assert G.is_authoring(GUARD)


def test_authoring_deny_list_files_pass_clean() -> None:
    assert G.scan_file(REPO / "src" / "vividly" / "models" / "reid_backend.py") == []
    assert G.scan_file(REPO / "scripts" / "reid" / "export_osnet_onnx.py") == []


def test_pragma_is_inert_outside_authoring(tmp_path: Path) -> None:
    """A fetcher cannot annotate its way past the guard: the allow pragma is honoured only in
    an authoring location, and elsewhere the line still fails, flagged as a stray pragma."""
    path = tmp_path / "sneaky.py"
    path.write_text('REPO = "kadirnar/osnet"  # reid-guard: allow\n')
    hits = G.scan_file(path)
    assert "kadirnar/osnet" in {hit.literal for hit in hits}
    assert any(hit.stray_pragma for hit in hits)


# --------------------------------------------------------------- end to end
def test_shipped_tree_is_clean() -> None:
    """The tree as checked in passes the whole-surface walk. Runs in-process so a false
    positive introduced by the reassembly passes fails here, not only in the shell selftest."""
    assert G.main([]) == 0
