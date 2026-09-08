"""scripts/data/download_dataset.py: the acquisition gates, the plan, and the double-locked fetch.

Nothing here touches the network. The one test that reaches the fetch branch injects a
recording runner, so the branch is proven reachable without a single byte moving. Every
other test asserts the runner was never called at all.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "data" / "download_dataset.py"


def _load_script() -> ModuleType:
    """scripts/ is not a package, so load the file by path.

    The module must be in sys.modules before exec_module: @dataclass resolves annotations
    through sys.modules[cls.__module__] and raises AttributeError otherwise.
    """
    spec = importlib.util.spec_from_file_location("download_dataset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


F = _load_script()

# Fixed vectors: sha256 of the exact byte strings below. No RNG anywhere in this file.
# A fixed hash input, kept from an earlier name for this script: PAYLOAD_SHA is computed
# over these exact bytes, so the byte string itself never changes.
PAYLOAD = b"vividly fetch_dataset test vector\n"
PAYLOAD_SHA = "b8b5fabfa0aa2d31434c13e0a70cc0e1c5f441579cac1e095593fb07f599ee2f"
WRONG_SHA = "deadbeef" * 8  # 64 hex chars, and not all-digits: YAML would read that as an int


class Recorder:
    """Stand-in for subprocess. Records argv and never executes anything."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> None:
        self.calls.append(tuple(argv))


# --------------------------------------------------------------- toy registry
def _entry(**over: str) -> str:
    """Toy dataset YAML, same pattern as tests/test_registry.py::_entry."""
    base = {
        "name": "toy",
        "bucket": "TRAIN_OK",
        "status": "planned",
        "root": "data/train_ok/toy",
        "commercial_use": "permitted",
        "counsel": "[]",
        "cleared": "null",
        "conflict": "null",
        "method": "http",
        "human_required": "false",
        "urls": "[https://example.invalid/toy.zip]",
        "human_steps": "[]",
        "files": "[]",
        "notes": "toy notes",
    }
    base.update(over)
    return textwrap.dedent(f"""\
        schema_version: 1
        name: {base["name"]}
        display_name: toy
        task: [detection]
        bucket: {base["bucket"]}
        status: {base["status"]}
        licensing:
          license: toy licence
          license_url: https://example.invalid
          license_verbatim: toy
          license_conflict_note: {base["conflict"]}
          commercial_use: {base["commercial_use"]}
          attribution_required: false
          redistribute: false
          counsel_questions: {base["counsel"]}
          counsel_cleared_utc: {base["cleared"]}
        provenance:
          collector: test
          imagery_source: test
          contains_real_people: false
          simulated_medical_events: false
          citation: test
        acquisition:
          method: {base["method"]}
          human_required: {base["human_required"]}
          urls: {base["urls"]}
          mirrors: []
          human_steps: {base["human_steps"]}
        layout:
          root: {base["root"]}
          images: {{train: images, val: images}}
          annotations: {{train: annotations/train.json, val: annotations/train.json}}
        ann_format: coco_keypoints
        converter: null
        files: {base["files"]}
        hashes_unavailable_reason: built by the test
        counts: {{}}
        notes: {base["notes"]}
    """)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "configs" / "datasets").mkdir(parents=True)
    (tmp_path / "data" / "train_ok").mkdir(parents=True)
    (tmp_path / "data" / "eval_only").mkdir(parents=True)
    return tmp_path


def _write(repo: Path, filename: str, text: str) -> None:
    (repo / "configs" / "datasets" / filename).write_text(text)


def _cfg(repo: Path) -> str:
    return str(repo / "configs" / "datasets")


# --------------------------------------------------------------- legal hold
def test_legal_hold_is_refused_and_quotes_the_hold_reason(repo: Path, capsys) -> None:
    _write(
        repo,
        "held.yaml",
        _entry(
            name="held",
            bucket="LEGAL_HOLD",
            status="blocked",
            root="null",
            commercial_use="prohibited",
            counsel="[CQ-5-toy-provenance]",
            human_required="true",
            human_steps='["BLOCKED PENDING COUNSEL. Do not download."]',
            notes="footage of identifiable people that the collector cannot sublicense",
        ),
    )
    runner = Recorder()
    code = F.main(["--name", "held", "--config", _cfg(repo), "--dry-run"], runner=runner)
    out = capsys.readouterr().out
    assert code == F.EXIT_GATE_CLOSED == 4
    assert "LEGAL_HOLD" in out and "REFUSED" in out
    assert "cannot sublicense" in out  # the hold reason, quoted from notes
    assert "CQ-5-toy-provenance" in out
    assert "curl" not in out  # no plan is printed for a held entry
    assert runner.calls == []


def test_legal_hold_beats_the_human_steps_path(repo: Path, capsys) -> None:
    """A LEGAL_HOLD entry is also human_required. The hold must win, not the form."""
    _write(
        repo,
        "held.yaml",
        _entry(
            name="held",
            bucket="LEGAL_HOLD",
            status="blocked",
            root="null",
            commercial_use="prohibited",
            counsel="[CQ-5-toy-provenance]",
            human_required="true",
            human_steps='["fill the request form"]',
        ),
    )
    code = F.main(["--name", "held", "--config", _cfg(repo), "--dry-run"], runner=Recorder())
    assert code == F.EXIT_GATE_CLOSED
    assert "fill the request form" not in capsys.readouterr().out


def test_gate_check_raises_legal_hold_violation(repo: Path) -> None:
    _write(
        repo,
        "held.yaml",
        _entry(
            name="held",
            bucket="LEGAL_HOLD",
            status="blocked",
            root="null",
            commercial_use="prohibited",
            counsel="[CQ-5-toy-provenance]",
        ),
    )
    reg = F.load_registry(Path(_cfg(repo)))
    with pytest.raises(F.R.LegalHoldViolation):
        F.gate_check(reg["held"])


# --------------------------------------------------------------- counsel gate
def test_open_counsel_questions_refuse_acquisition(repo: Path, capsys) -> None:
    """The counsel gate (CounselGateOpen) binds acquisition too, not only training use."""
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
            counsel="[CQ-NC-EVAL]",
        ),
    )
    runner = Recorder()
    code = F.main(["--name", "restricted", "--config", _cfg(repo), "--dry-run"], runner=runner)
    out = capsys.readouterr().out
    assert code == F.EXIT_GATE_CLOSED == 4
    assert "CQ-NC-EVAL" in out
    assert "counsel gate" in out and "CounselGateOpen" in out
    assert "curl" not in out
    assert runner.calls == []


def test_gate_check_raises_counsel_gate_open(repo: Path) -> None:
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
            counsel="[CQ-NC-EVAL]",
        ),
    )
    reg = F.load_registry(Path(_cfg(repo)))
    with pytest.raises(F.R.CounselGateOpen, match="CQ-NC-EVAL"):
        F.gate_check(reg["restricted"])


def test_recorded_counsel_clearance_opens_the_gate(repo: Path, capsys) -> None:
    """Clearing is a deliberate YAML edit, and only then does planning proceed."""
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
            counsel="[CQ-NC-EVAL]",
            cleared='"2026-08-07T00:00:00Z"',
        ),
    )
    runner = Recorder()
    code = F.main(["--name", "restricted", "--config", _cfg(repo), "--dry-run"], runner=runner)
    out = capsys.readouterr().out
    assert code == F.EXIT_OK
    assert "https://example.invalid/toy.zip" in out
    assert "data/eval_only/restricted" in out
    assert runner.calls == []


# --------------------------------------------------------------- human steps
def test_human_required_prints_numbered_steps_and_exits_3(repo: Path, capsys) -> None:
    steps = '["Open the request page and fill the form.", "Solve the captcha and submit."]'
    _write(repo, "toy.yaml", _entry(human_required="true", human_steps=steps))
    runner = Recorder()
    code = F.main(["--name", "toy", "--config", _cfg(repo), "--dry-run"], runner=runner)
    out = capsys.readouterr().out
    assert code == F.EXIT_HUMAN_REQUIRED == 3
    assert "1. Open the request page and fill the form." in out
    assert "2. Solve the captcha and submit." in out
    assert "curl" not in out  # exits before the plan: no fetching, not even planned
    assert runner.calls == []


def test_human_required_with_empty_steps_is_flagged_as_a_registry_bug(repo: Path, capsys) -> None:
    _write(repo, "toy.yaml", _entry(human_required="true", human_steps="[]"))
    code = F.main(["--name", "toy", "--config", _cfg(repo), "--dry-run"], runner=Recorder())
    assert code == F.EXIT_HUMAN_REQUIRED
    assert "registry bug" in capsys.readouterr().out


# --------------------------------------------------------------- the dry-run plan
def test_dry_run_prints_the_real_coco_urls_and_verification_plan(capsys) -> None:
    """Against the shipped registry: COCO is TRAIN_OK, no counsel questions, no human step."""
    runner = Recorder()
    code = F.main(["--name", "coco", "--dry-run"], runner=runner)
    out = capsys.readouterr().out
    assert code == F.EXIT_OK
    for url in (
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    ):
        assert url in out
    assert "curl -fSL --retry 5 --retry-delay 10 -C -" in out
    assert "data/train_ok/coco" in out
    assert "academictorrents.com" in out  # the recorded mirror
    assert "verification plan" in out
    assert "hashes_unavailable_reason" in out  # COCO's zips were deleted; stated honestly
    assert "train_images: 118,287" in out  # counts to re-verify after extraction
    assert "--dry-run: stopping before any acquisition." in out
    assert runner.calls == []


def test_dry_run_never_shells_out(repo: Path, monkeypatch) -> None:
    """Belt and braces: subprocess.run is poisoned for the whole planning path."""

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("download_dataset planned a run of subprocess.run")

    monkeypatch.setattr(F.subprocess, "run", _boom)
    _write(repo, "toy.yaml", _entry())
    assert F.main(["--name", "toy", "--config", _cfg(repo), "--dry-run"]) == F.EXIT_OK


def test_plan_lines_are_pure(repo: Path) -> None:
    """plan_lines() builds text only: it must not create the destination directory."""
    _write(repo, "toy.yaml", _entry())
    reg = F.load_registry(Path(_cfg(repo)))
    ds = reg["toy"]
    dest = F.dest_dir(reg, ds)
    lines = F.plan_lines(ds, dest)
    assert any("https://example.invalid/toy.zip" in line for line in lines)
    assert not dest.exists()


def test_gdrive_and_manual_methods_get_honest_commands(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry(method="gdrive"))
    reg = F.load_registry(Path(_cfg(repo)))
    cmds = F.fetch_commands(reg["toy"], Path("/dest"))
    assert cmds[0].argv is not None and cmds[0].argv[0] == "gdown"

    _write(repo, "toy.yaml", _entry(method="manual"))
    reg = F.load_registry(Path(_cfg(repo)))
    cmds = F.fetch_commands(reg["toy"], Path("/dest"))
    assert cmds[0].argv is None
    assert "no command can be built from the registry" in cmds[0].display


# --------------------------------------------------------------- produced, not fetched
# No shipped entry is produced locally today: every registered dataset arrives by download.
# Both branches stay covered on scratch entries so the refusal is still proved for the day
# a produced entry is registered again.
def test_derived_dataset_names_its_producer_and_refuses(repo: Path, capsys) -> None:
    _write(repo, "parent.yaml", _entry(name="parent"))
    _write(
        repo,
        "derived.yaml",
        _entry(name="derived", method="derived", urls="[]") + "derived_from: [parent]\n",
    )
    code = F.main(["--name", "derived", "--config", _cfg(repo), "--dry-run"], runner=Recorder())
    out = capsys.readouterr().out
    assert code == F.EXIT_NOT_FETCHABLE == 6
    assert "NOT FETCHABLE" in out
    assert "a repository generator script, not a download" in out
    assert "derived_from : ['parent']" in out


def test_generator_dataset_refuses_but_still_shows_the_binary_urls(repo: Path, capsys) -> None:
    _write(
        repo,
        "made.yaml",
        _entry(
            name="made",
            method="generator",
            urls="[https://example.invalid/StandaloneOSX_toy.zip]",
        ),
    )
    code = F.main(["--name", "made", "--config", _cfg(repo), "--dry-run"], runner=Recorder())
    out = capsys.readouterr().out
    assert code == F.EXIT_NOT_FETCHABLE
    assert "StandaloneOSX_toy.zip" in out
    assert "NOT FETCHABLE" in out


# --------------------------------------------------------------- the fetch interlock
@pytest.mark.parametrize(
    ("yes", "env", "expected"),
    [
        (False, {}, False),
        (True, {}, False),
        (False, {F.ALLOW_FETCH_ENV: "1"}, False),
        (True, {F.ALLOW_FETCH_ENV: "0"}, False),
        (True, {F.ALLOW_FETCH_ENV: "true"}, False),
        (True, {F.ALLOW_FETCH_ENV: "1"}, True),
    ],
)
def test_fetch_is_authorised_needs_both_locks(yes: bool, env: dict, expected: bool) -> None:
    assert F.fetch_is_authorised(yes, env) is expected


def test_yes_alone_cannot_fetch(repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv(F.ALLOW_FETCH_ENV, raising=False)
    _write(repo, "toy.yaml", _entry())
    runner = Recorder()
    code = F.main(["--name", "toy", "--config", _cfg(repo), "--yes"], runner=runner)
    err = capsys.readouterr().err
    assert code == F.EXIT_FETCH_NOT_AUTHORISED == 5
    assert "Nothing was downloaded" in err
    assert runner.calls == []
    assert not (repo / "data" / "train_ok" / "toy").exists()


def test_env_var_alone_cannot_fetch(repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(F.ALLOW_FETCH_ENV, "1")
    _write(repo, "toy.yaml", _entry())
    runner = Recorder()
    code = F.main(["--name", "toy", "--config", _cfg(repo)], runner=runner)
    assert code == F.EXIT_FETCH_NOT_AUTHORISED
    assert "Nothing was downloaded" in capsys.readouterr().err
    assert runner.calls == []
    assert not (repo / "data" / "train_ok" / "toy").exists()


def test_fetch_branch_is_reachable_only_with_both_locks(repo: Path, monkeypatch) -> None:
    """Proves the branch exists and is correct, with a recording runner instead of curl."""
    monkeypatch.setenv(F.ALLOW_FETCH_ENV, "1")
    files = "[{name: toy.zip, size_bytes: 34, sha256: " + PAYLOAD_SHA + "}]"
    _write(repo, "toy.yaml", _entry(files=files))
    runner = Recorder()
    code = F.main(["--name", "toy", "--config", _cfg(repo), "--yes"], runner=runner)

    assert [c[0] for c in runner.calls] == ["curl"]
    argv = runner.calls[0]
    assert "https://example.invalid/toy.zip" in argv
    assert str(repo / "data" / "train_ok" / "toy" / "toy.zip") in argv
    # The runner downloaded nothing, so verification must fail rather than claim success.
    assert code == F.EXIT_VERIFY_FAILED == 1
    receipt = repo / "data" / "train_ok" / "toy" / F.RECEIPT_NAME
    assert receipt.is_file()
    assert '"verified": false' in receipt.read_text()


def test_fetch_verifies_bytes_that_are_actually_there(repo: Path, monkeypatch) -> None:
    """Same branch, but the runner writes the expected payload: verification passes."""
    monkeypatch.setenv(F.ALLOW_FETCH_ENV, "1")
    files = "[{name: toy.zip, size_bytes: 34, sha256: " + PAYLOAD_SHA + "}]"
    _write(repo, "toy.yaml", _entry(files=files))

    def runner(argv: Sequence[str]) -> None:
        Path(argv[argv.index("-o") + 1]).write_bytes(PAYLOAD)

    code = F.main(["--name", "toy", "--config", _cfg(repo), "--yes"], runner=runner)
    assert code == F.EXIT_OK
    receipt = repo / "data" / "train_ok" / "toy" / F.RECEIPT_NAME
    assert '"verified": true' in receipt.read_text()


def test_a_gated_dataset_cannot_be_fetched_even_with_both_locks(repo: Path, monkeypatch) -> None:
    """The interlock is the last line, not the first: the gate still refuses above it."""
    monkeypatch.setenv(F.ALLOW_FETCH_ENV, "1")
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
            counsel="[CQ-NC-EVAL]",
        ),
    )
    runner = Recorder()
    code = F.main(["--name", "restricted", "--config", _cfg(repo), "--yes"], runner=runner)
    assert code == F.EXIT_GATE_CLOSED
    assert runner.calls == []


# --------------------------------------------------------------- verification unit tests
def test_verify_files_accepts_matching_size_and_hash(repo: Path, tmp_path: Path) -> None:
    files = "[{name: toy.zip, size_bytes: 34, sha256: " + PAYLOAD_SHA + "}]"
    _write(repo, "toy.yaml", _entry(files=files))
    ds = F.load_registry(Path(_cfg(repo)))["toy"]
    (tmp_path / "toy.zip").write_bytes(PAYLOAD)
    assert F.verify_files(ds, tmp_path) == []


def test_verify_files_reports_missing_wrong_size_and_wrong_hash(repo: Path, tmp_path: Path) -> None:
    files = (
        "[{name: a.zip, size_bytes: 34, sha256: " + PAYLOAD_SHA + "}, "
        "{name: b.zip, size_bytes: 34, sha256: " + PAYLOAD_SHA + "}, "
        "{name: c.zip, size_bytes: 34, sha256: " + WRONG_SHA + "}]"
    )
    _write(repo, "toy.yaml", _entry(files=files))
    ds = F.load_registry(Path(_cfg(repo)))["toy"]
    (tmp_path / "b.zip").write_bytes(b"short")
    (tmp_path / "c.zip").write_bytes(PAYLOAD)
    problems = F.verify_files(ds, tmp_path)
    assert len(problems) == 3
    assert "a.zip: missing" in problems[0]
    assert "size 5 != expected 34" in problems[1]
    assert f"sha256 {PAYLOAD_SHA} != expected {WRONG_SHA}" in problems[2]


def test_verify_files_refuses_to_call_an_unhashable_entry_verified(repo: Path, tmp_path) -> None:
    text = _entry().replace("hashes_unavailable_reason: built by the test", "")
    _write(repo, "toy.yaml", text)
    ds = F.load_registry(Path(_cfg(repo)))["toy"]
    problems = F.verify_files(ds, tmp_path)
    assert len(problems) == 1
    assert "nothing can be verified" in problems[0]


def test_sha256_file_matches_the_fixed_vector(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(PAYLOAD)
    assert F.sha256_file(path) == PAYLOAD_SHA
    assert F.sha256_file(path, chunk_bytes=7) == PAYLOAD_SHA  # chunking is not a hash change


# --------------------------------------------------------------- shipped registry today
def test_unknown_dataset_exits_2(capsys) -> None:
    code = F.main(["--name", "not_a_dataset", "--dry-run"], runner=Recorder())
    assert code == F.EXIT_UNKNOWN_DATASET == 2
    assert "registry error" in capsys.readouterr().err


#: Where EVAL_ONLY bytes would land. Snapshotted around every shipped-registry test below:
#: the claim those tests make is "this run created nothing", and comparing a listing says
#: that whatever else is already sitting in the tree.
EVAL_ONLY_TREE = REPO / "data" / "eval_only"


def _listing(root: Path) -> set[str]:
    """Every path under `root` right now. An absent directory lists as nothing."""
    return {str(p) for p in root.rglob("*")} if root.is_dir() else set()


def _restricted_shipped_plan() -> list[tuple[str, int, str]]:
    """(name, exit code, banner) for every shipped entry that is not ordinary TRAIN_OK data.

    Read out of the registry fields the tool actually branches on, in the tool's own order:
    bucket LEGAL_HOLD, then the counsel gate, then acquisition.human_required, then the
    produced-locally methods, then a plan. Deriving it is the point: an entry leaves the
    counsel branch as soon as its own `counsel_cleared_utc` is recorded in its YAML, and a
    hardcoded list here would have to be edited each time that happens; a derived one just
    follows.
    """
    reg = F.R.default_registry()
    rows: list[tuple[str, int, str]] = []
    for name in reg.names():
        ds = reg[name]
        lic = ds.licensing
        if ds.bucket is F.R.Bucket.LEGAL_HOLD:
            rows.append((name, F.EXIT_GATE_CLOSED, "LEGAL_HOLD"))
        elif lic.counsel_questions and not lic.counsel_cleared_utc:
            rows.append((name, F.EXIT_GATE_CLOSED, "CounselGateOpen"))
        elif ds.bucket is not F.R.Bucket.EVAL_ONLY:
            continue  # ordinary TRAIN_OK data; the coco and producer tests cover that path
        elif ds.acquisition.human_required:
            rows.append((name, F.EXIT_HUMAN_REQUIRED, "HUMAN ACTION REQUIRED"))
        elif ds.acquisition.method in F.PRODUCER_HINT:
            rows.append((name, F.EXIT_NOT_FETCHABLE, "NOT FETCHABLE"))
        else:
            rows.append((name, F.EXIT_OK, "ACQUISITION PLAN"))
    return rows


RESTRICTED_SHIPPED = _restricted_shipped_plan()
LEGAL_HOLDS = [n for n, code, banner in RESTRICTED_SHIPPED if banner == "LEGAL_HOLD"]
# Non-vacuity guard for the parametrised group below, which would otherwise report green on
# an empty registry.
assert RESTRICTED_SHIPPED, "no shipped entry reaches a gate; the group below is moot"


def test_no_shipped_entry_sits_in_legal_hold_today() -> None:
    """Stated rather than left silent. The bucket itself is not retired: its refusal is
    proved on scratch entries just above, and on both fetch locks just below, so the day an
    entry lands in it the mechanism is already covered."""
    assert LEGAL_HOLDS == []


def _cleared_and_fetchable() -> str:
    """The sharpest cleared name: cleared, no registration form, and a runnable command.

    For this entry the fetch interlock is the last thing between the registry and someone
    else's bytes, so it is the one the interlock test should use. Falls back to any entry
    that reaches the plan, for the day a method stops resolving to an argv.
    """
    reg = F.R.default_registry()
    planned = [n for n, code, _ in RESTRICTED_SHIPPED if code == F.EXIT_OK]
    assert planned, "no shipped entry reaches the acquisition plan; the interlock test is moot"
    runnable = [
        n for n in planned if any(c.argv for c in F.fetch_commands(reg[n], Path("/dev/null")))
    ]
    return (runnable or planned)[0]


CLEARED_AND_FETCHABLE = _cleared_and_fetchable()


@pytest.mark.parametrize(("name", "expected", "banner"), RESTRICTED_SHIPPED)
def test_legal_holds_refuse_form_entries_exit_3_and_cleared_entries_only_plan(
    name: str, expected: int, banner: str, monkeypatch, capsys
) -> None:
    """The shipped registry as it stands: no entry carries an open counsel question.

    Every entry with counsel questions records its own `counsel_cleared_utc`, so the counsel
    gate no longer stands in front of any shipped entry. What is left is the split this test
    names. A LEGAL_HOLD entry, which no clearance covers, refuses permanently. An entry whose
    provider wants a registration form still exits 3 and prints the form steps, because a
    clearance covers use ahead of counsel and never the form: a human goes through it rather
    than around it. The rest print a plan and stop. Every group ends with a recording runner
    that was never called and a data tree that gained nothing: a plan is text.
    """
    monkeypatch.delenv(F.ALLOW_FETCH_ENV, raising=False)
    before = _listing(EVAL_ONLY_TREE)
    runner = Recorder()
    code = F.main(["--name", name, "--dry-run"], runner=runner)
    out = capsys.readouterr().out

    assert code == expected
    assert banner in out
    assert runner.calls == []
    if expected == F.EXIT_OK:
        assert "--dry-run: stopping before any acquisition." in out
    else:
        assert "curl" not in out  # a refused entry gets no plan, not even a printed one
    assert _listing(EVAL_ONLY_TREE) == before, "planning must not create a single path"


def test_a_legal_hold_refuses_even_with_both_fetch_locks_open(
    repo: Path, monkeypatch, capsys
) -> None:
    """Both locks are open here on purpose, and the hold still wins.

    This is the refusal a mechanical clearance cannot move: `counsel_cleared_utc` opens the
    counsel gate, LEGAL_HOLD is a bucket, and `gate_check` tests the bucket first. Clearing
    a hold means a reviewed edit that changes the bucket, which no flag and no memo can
    stand in for.
    """
    _write(
        repo,
        "held.yaml",
        _entry(
            name="held",
            bucket="LEGAL_HOLD",
            status="blocked",
            root="null",
            commercial_use="prohibited",
            cleared="'2026-08-08'",
            counsel="[CQ-5-toy-provenance]",
        ),
    )
    monkeypatch.setenv(F.ALLOW_FETCH_ENV, "1")
    before = _listing(repo / "data")
    runner = Recorder()
    code = F.main(["--name", "held", "--config", _cfg(repo), "--yes"], runner=runner)
    assert code == F.EXIT_GATE_CLOSED == 4
    assert "LEGAL_HOLD" in capsys.readouterr().out
    assert runner.calls == []
    assert _listing(repo / "data") == before


@pytest.mark.parametrize(
    ("yes", "env"),
    [(False, None), (False, "1"), (True, None), (True, "0"), (True, "true")],
    ids=["neither-lock", "env-only", "yes-only", "yes-plus-zero", "yes-plus-true"],
)
def test_a_cleared_entry_still_needs_both_fetch_locks(
    yes: bool, env: str | None, monkeypatch, capsys
) -> None:
    """A recorded counsel clearance opens the counsel gate; it does not open the fetch
    interlock.

    Nothing does except a human typing both halves, and this test is what stops a clearance
    being read as permission to auto-download: it covers use of a dataset ahead of counsel,
    not acquisition without a human. The name under test is the sharp case, since it is
    cleared and has no registration form to slow anyone down.
    """
    if env is None:
        monkeypatch.delenv(F.ALLOW_FETCH_ENV, raising=False)
    else:
        monkeypatch.setenv(F.ALLOW_FETCH_ENV, env)
    before = _listing(EVAL_ONLY_TREE)
    runner = Recorder()
    argv = ["--name", CLEARED_AND_FETCHABLE, *(["--yes"] if yes else [])]
    code = F.main(argv, runner=runner)
    assert code == F.EXIT_FETCH_NOT_AUTHORISED == 5
    assert "Nothing was downloaded" in capsys.readouterr().err
    assert runner.calls == []
    assert _listing(EVAL_ONLY_TREE) == before


# --------------------------------------------------------------- house style
def test_no_em_or_en_dashes_in_this_script_or_its_test() -> None:
    """House style over these two files only; no repo-wide dash guard exists to lean on.

    Built from code points so this test does not trip over itself.
    """
    em_dash, en_dash = chr(0x2014), chr(0x2013)
    for path in (SCRIPT, Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert em_dash not in text, f"{path.name} contains an em dash"
        assert en_dash not in text, f"{path.name} contains an en dash"


def test_urls_sharing_a_file_name_land_in_their_recorded_directories(repo: Path) -> None:
    """Two interval CSVs both named ADL.csv under different subject folders must not overwrite
    each other: the destination comes from the recorded files[] tail, not the URL basename.
    A URL that matches no recorded file still lands under its last path component."""
    urls = (
        "https://example.invalid/r/abc/Subject%201/ADL.csv",
        "https://example.invalid/r/abc/Subject%202/ADL.csv",
        "https://example.invalid/r/abc/toy.zip",
        "https://example.invalid/other/README",
    )
    files = ", ".join(
        f"{{name: '{name}', size_bytes: 1, sha256: '{WRONG_SHA}'}}"
        for name in (
            "annotations_upstream/Subject 1/ADL.csv",
            "annotations_upstream/Subject 2/ADL.csv",
            "upstream/toy.zip",
        )
    )
    _write(repo, "toy.yaml", _entry(urls=f"[{', '.join(urls)}]", files=f"[{files}]"))
    reg = F.load_registry(Path(_cfg(repo)))
    ds = reg["toy"]
    dest = F.dest_dir(reg, ds)
    outs = [Path(c.argv[-2]) for c in F.fetch_commands(ds, dest)]
    assert outs == [
        dest / "annotations_upstream/Subject 1/ADL.csv",
        dest / "annotations_upstream/Subject 2/ADL.csv",
        dest / "upstream/toy.zip",
        dest / "README",
    ]
    assert len(set(outs)) == len(outs)
