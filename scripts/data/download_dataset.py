#!/usr/bin/env python3
"""The registry-driven acquisition entry point.

    uv run python scripts/data/download_dataset.py --name coco --dry-run

Dataset acquisition goes through here so that the licence buckets, the counsel gate
and the human-only steps are enforced by code instead of by memory. Two registered sets
are fetched by their own scripts instead: `ur_fall` by scripts/data/fetch_ur_fall.sh and
`ajile12` by scripts/data/harvest_ajile12.py, both named in TRAINING.md. The tool resolves
the name through the registry and then, in this order:

  1. bucket LEGAL_HOLD                -> refuse loudly, quote the hold reason, exit 4
  2. open counsel questions           -> refuse, list the questions, exit 4
  3. acquisition.human_required       -> print the numbered human steps, exit 3
  4. method generator|derived         -> name the producer, exit 6 (nothing to download)
  5. otherwise                        -> print the exact commands, sizes and sha256 plan
  6. --dry-run                        -> stop after printing, exit 0. Without it the plan
                                         still prints and the run stops at the fetch
                                         interlock below, exit 5, so planning is what a
                                         plain invocation does.

Exit codes: 0 planned or fetched and verified; 1 post-fetch verification failed;
2 unknown dataset or unloadable registry; 3 a human must act first; 4 a licence gate is
closed (LEGAL_HOLD or an open counsel question); 5 the fetch interlock is not satisfied;
6 this dataset is produced rather than downloaded.

This tool downloads nothing by itself. The fetch branch is implemented but is
double-locked: it needs both --yes on the command line and VIVIDLY_ALLOW_FETCH=1 in the
environment. Either one alone refuses with exit 5.

Entries that were considered and are not used are listed in DATASETS.md, and the counsel
history behind them is in docs/COUNSEL_QUESTIONS.md.

It never rewrites a dataset YAML. Verification results are written to a fetch receipt
inside the dataset root and the YAML block is printed for a human to paste, because the
YAMLs carry the licence reasoning in comments that a round-trip dump would destroy, and
because licensing.counsel_questions / licensing.counsel_cleared_utc must only ever change
by a deliberate, reviewed human edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vividly.data import registry as R  # noqa: E402
from vividly.data.registry import Bucket, Dataset, Registry  # noqa: E402

EXIT_OK = 0
EXIT_VERIFY_FAILED = 1
EXIT_UNKNOWN_DATASET = 2
EXIT_HUMAN_REQUIRED = 3
EXIT_GATE_CLOSED = 4
EXIT_FETCH_NOT_AUTHORISED = 5
EXIT_NOT_FETCHABLE = 6

ALLOW_FETCH_ENV = "VIVIDLY_ALLOW_FETCH"
ALLOW_FETCH_VALUE = "1"
RECEIPT_NAME = "_fetch_receipt.json"

# Methods that produce bytes locally instead of downloading a dataset. The hint names what
# does produce them; the entry's own notes and converter fields carry the detail.
PRODUCER_HINT: dict[str, str] = {
    "derived": (
        "a repository generator script, not a download; the entry's own acquisition "
        "and converter fields name the script and the parent corpus it reads"
    ),
    "generator": (
        "the upstream generator binary listed in acquisition.urls, run locally, "
        "then acquisition -> converter"
    ),
}

Runner = Callable[[Sequence[str]], None]


@dataclass(frozen=True, slots=True)
class FetchCommand:
    """One acquisition step. argv is None when the registry cannot name an exact command."""

    display: str
    argv: tuple[str, ...] | None = None


# ------------------------------------------------------------------ registry access
def load_registry(config_dir: Path | None) -> Registry:
    """Resolve the registry. No argument, or the shipped configs/datasets, uses the cached
    default_registry(); any other directory is loaded fresh so tests can use a toy tree."""
    if config_dir is None:
        return R.default_registry()
    config_dir = Path(config_dir).resolve()
    if config_dir == R.CONFIG_DIR.resolve():
        return R.default_registry()
    # A toy registry lives at <root>/configs/datasets, so the repo root is two levels up.
    return Registry.from_dir(config_dir, repo_root=config_dir.parent.parent)


# ------------------------------------------------------------------ gates
def gate_check(ds: Dataset) -> None:
    """Raise if any licence gate is closed. Returns None only when acquisition may proceed.

    Deliberately raises instead of returning a flag: a caller cannot forget to check a
    raised exception. LegalHoldViolation is the stronger refusal and is tested first;
    CounselGateOpen carries the same semantics as Registry._manifest(), namely that open
    counsel questions with no licensing.counsel_cleared_utc block the dataset mechanically;
    tests/test_download_dataset.py pins that on the acquisition side. Acquisition is bound
    by the same gate as manifest building, because acquiring the bytes is the step that is
    hardest to undo.
    """
    if ds.bucket is Bucket.LEGAL_HOLD:
        raise R.LegalHoldViolation(
            f"'{ds.name}' is LEGAL_HOLD and must not be acquired "
            f"(open counsel items: {list(ds.licensing.counsel_questions)})"
        )
    if ds.licensing.counsel_questions and not ds.licensing.counsel_cleared_utc:
        raise R.CounselGateOpen(
            f"'{ds.name}' has open counsel questions "
            f"{list(ds.licensing.counsel_questions)} and no counsel_cleared_utc. "
            "No acquisition may proceed until counsel answers are recorded in its "
            "configs/datasets/ entry."
        )


def legal_hold_lines(ds: Dataset) -> list[str]:
    """The loud refusal, quoting the hold reason out of the YAML rather than paraphrasing."""
    lines = [
        "",
        "#" * 78,
        f"# REFUSED: dataset '{ds.name}' is LEGAL_HOLD. Do not download it.",
        "#" * 78,
        f"  display_name : {ds.display_name}",
        f"  status       : {ds.status.value}",
        f"  licence      : {ds.licensing.license}",
        f"  licence url  : {ds.licensing.license_url}",
        f"  layout.root  : {ds.layout.root!r}  (LEGAL_HOLD entries have no local root)",
        "",
        "  open counsel questions:",
    ]
    lines += [f"    - {q}" for q in ds.licensing.counsel_questions] or ["    - (none recorded)"]
    if ds.licensing.license_conflict_note:
        lines += ["", "  licence conflict note (copied from the YAML):"]
        lines += [f"    {line}" for line in _wrapped(ds.licensing.license_conflict_note)]
    lines += ["", "  hold reason (notes, copied from the YAML):"]
    lines += [f"    {line}" for line in _wrapped(ds.notes)]
    lines += [
        "",
        "  Clearing a hold is a human act: a reviewed edit to "
        f"{_rel(ds.source_yaml)} that changes the",
        "  bucket and records licensing.counsel_cleared_utc. No flag on this tool can do it.",
        "",
    ]
    return lines


def counsel_gate_lines(ds: Dataset, exc: R.CounselGateOpen) -> list[str]:
    lines = [
        "",
        "#" * 78,
        f"# REFUSED: dataset '{ds.name}' is behind the counsel gate (CounselGateOpen).",
        "#" * 78,
        f"  bucket       : {ds.bucket.value}",
        f"  licence      : {ds.licensing.license}",
        f"  licence url  : {ds.licensing.license_url}",
        f"  commercial   : {ds.licensing.commercial_use.value}",
        f"  cleared utc  : {ds.licensing.counsel_cleared_utc!r}",
        "",
        "  open counsel questions:",
    ]
    lines += [f"    - {q}" for q in ds.licensing.counsel_questions]
    if ds.licensing.license_conflict_note:
        lines += ["", "  licence conflict note (copied from the YAML):"]
        lines += [f"    {line}" for line in _wrapped(ds.licensing.license_conflict_note)]
    lines += [
        "",
        f"  {exc}",
        "",
        "  The gate is mechanical and applies to acquisition exactly as it applies to manifest",
        "  building. It opens when licensing.counsel_cleared_utc is set in "
        f"{_rel(ds.source_yaml)}.",
        "",
    ]
    return lines


def human_step_lines(ds: Dataset) -> list[str]:
    """The numbered human steps. This tool never completes forms, registrations or captchas."""
    lines = [
        "",
        "=" * 78,
        f"= HUMAN ACTION REQUIRED before '{ds.name}' can be acquired.",
        "=" * 78,
        f"  method       : {ds.acquisition.method}",
        f"  licence      : {ds.licensing.license}",
        f"  licence url  : {ds.licensing.license_url}",
        "",
        "  Steps (copied from acquisition.human_steps):",
    ]
    steps = ds.acquisition.human_steps
    if steps:
        for i, step in enumerate(steps, start=1):
            wrapped = _wrapped(step)
            lines.append(f"    {i}. {wrapped[0]}")
            lines += [f"       {line}" for line in wrapped[1:]]
    else:
        lines.append("    (acquisition.human_required is true but human_steps is empty:")
        lines.append("     that is a registry bug, fix the YAML before acquiring anything)")
    lines += [
        "",
        "  A human does the form, the registration and the captcha. This tool does not, and",
        "  no flag makes it. Re-run with --dry-run afterwards to see the download plan.",
        "",
    ]
    return lines


# ------------------------------------------------------------------ the plan
def dest_dir(reg: Registry, ds: Dataset) -> Path:
    """Where bytes would land: <bucket_root>/<name>, and nowhere else."""
    return reg.bucket_root(ds.bucket) / ds.name


def fetch_commands(ds: Dataset, dest: Path) -> list[FetchCommand]:
    """Exact per-URL commands, dispatched on acquisition.method."""
    method = ds.acquisition.method
    cmds: list[FetchCommand] = []
    for url in ds.acquisition.urls:
        if method == "http":
            out = dest / _dest_name(ds, url)
            argv = (
                "curl",
                "-fSL",
                "--retry",
                "5",
                "--retry-delay",
                "10",
                "-C",
                "-",
                "-o",
                str(out),
                url,
            )
            cmds.append(FetchCommand(display=shlex.join(argv), argv=argv))
        elif method == "gdrive":
            argv = ("gdown", url, "-O", str(dest / _dest_name(ds, url)))
            cmds.append(FetchCommand(display=shlex.join(argv), argv=argv))
        elif method == "huggingface":
            repo_id = url.rstrip("/").split("huggingface.co/datasets/")[-1]
            names = [f.name for f in ds.files] or ["<filename, from files[] once recorded>"]
            for name in names:
                cmds.append(
                    FetchCommand(
                        display=(
                            'python -c "from huggingface_hub import hf_hub_download; '
                            f"hf_hub_download(repo_id='{repo_id}', repo_type='dataset', "
                            f"filename='{name}', local_dir='{dest}')\""
                        )
                    )
                )
        else:
            cmds.append(
                FetchCommand(
                    display=(
                        f"(method '{method}': no command can be built from the registry; "
                        f"a signed-in human downloads {url})"
                    )
                )
            )
    return cmds


def verification_plan_lines(ds: Dataset) -> list[str]:
    """What would be checked after a fetch: byte sizes and sha256 from the files block."""
    lines = ["  verification plan (files block of the YAML):"]
    if ds.files:
        for f in ds.files:
            lines.append(f"    {f.name}")
            lines.append(f"      size   : {f.size_bytes:,} bytes")
            lines.append(f"      sha256 : {f.sha256}")
    else:
        reason = ds.hashes_unavailable_reason or "(files is empty and no reason is recorded)"
        lines.append("    no per-file hashes are recorded for this entry, so a fetch cannot be")
        lines.append("    byte-verified. hashes_unavailable_reason:")
        lines += [f"      {line}" for line in _wrapped(reason)]
    if ds.counts:
        lines.append("  post-extraction counts to re-verify:")
        lines += [f"    {k}: {v:,}" for k, v in ds.counts]
    else:
        lines.append("  post-extraction counts to re-verify: (none recorded)")
    return lines


def plan_lines(ds: Dataset, dest: Path) -> list[str]:
    """The dry-run report. Pure: builds text, touches no disk and no network."""
    lines = [
        "",
        "-" * 78,
        f"- ACQUISITION PLAN for '{ds.name}' ({ds.display_name})",
        "-" * 78,
        f"  bucket       : {ds.bucket.value}",
        f"  status       : {ds.status.value}",
        f"  method       : {ds.acquisition.method}",
        f"  licence      : {ds.licensing.license}",
        f"  commercial   : {ds.licensing.commercial_use.value}",
        f"  redistribute : {ds.licensing.redistribute}"
        f"{'' if ds.licensing.redistribute else '  (never mirror to project object storage)'}",
        f"  destination  : {dest}",
        f"  source yaml  : {_rel(ds.source_yaml)}",
        "",
    ]
    if ds.acquisition.urls:
        lines.append("  commands:")
        for cmd in fetch_commands(ds, dest):
            lines.append(f"    {cmd.display}")
    else:
        lines.append("  commands: (acquisition.urls is empty)")
    if ds.acquisition.mirrors:
        lines.append("  mirrors (fallback only):")
        lines += [f"    {m}" for m in ds.acquisition.mirrors]
    lines.append("")
    lines += verification_plan_lines(ds)
    lines.append("")
    return lines


def producer_lines(ds: Dataset) -> list[str]:
    """Refusal for generator/derived entries: this tool downloads, it does not produce."""
    hint = PRODUCER_HINT.get(ds.acquisition.method, "an unregistered producer")
    lines = [
        "",
        f"  NOT FETCHABLE: '{ds.name}' has method '{ds.acquisition.method}'.",
        f"  It is produced by {hint}.",
        f"  converter    : {ds.converter!r}",
        f"  derived_from : {list(ds.derived_from)}",
        "  notes (copied from the YAML):",
    ]
    lines += [f"    {line}" for line in _wrapped(ds.notes)]
    lines.append("")
    return lines


# ------------------------------------------------------------------ verification
def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk_bytes):
            h.update(block)
    return h.hexdigest()


def verify_files(ds: Dataset, dest: Path) -> list[str]:
    """Check every files[] record against what is on disk. Returns the problems found.

    An empty list means verified. Local filesystem only: never opens a socket.
    """
    problems: list[str] = []
    if not ds.files:
        reason = ds.hashes_unavailable_reason
        if not reason:
            problems.append(
                f"'{ds.name}': files is empty and hashes_unavailable_reason is null, so nothing "
                "can be verified; record hashes in the YAML before trusting these bytes"
            )
        return problems
    for f in ds.files:
        path = dest / f.name
        if not path.is_file():
            problems.append(f"{f.name}: missing at {path}")
            continue
        actual_size = path.stat().st_size
        if actual_size != f.size_bytes:
            problems.append(f"{f.name}: size {actual_size:,} != expected {f.size_bytes:,} bytes")
            continue
        actual_sha = sha256_file(path)
        if actual_sha != f.sha256:
            problems.append(f"{f.name}: sha256 {actual_sha} != expected {f.sha256}")
    return problems


def write_receipt(ds: Dataset, dest: Path, problems: Sequence[str]) -> Path:
    """Record the verification outcome next to the bytes. Never edits the dataset YAML."""
    receipt = {
        "dataset": ds.name,
        "bucket": ds.bucket.value,
        "source_yaml": _rel(ds.source_yaml),
        "verified_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [
            {
                "name": f.name,
                "size_bytes": f.size_bytes,
                "sha256": f.sha256,
                "present": (dest / f.name).is_file(),
            }
            for f in ds.files
        ],
        "expected_counts": dict(ds.counts),
        "problems": list(problems),
        "verified": not problems,
    }
    path = dest / RECEIPT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ the locked fetch branch
def _run(argv: Sequence[str]) -> None:
    # argv list, never shell=True: no string is ever handed to a shell.
    subprocess.run(list(argv), check=True)


def fetch(ds: Dataset, dest: Path, *, runner: Runner = _run) -> int:
    """Run the acquisition commands and verify. Only ever reached through main()'s interlock.

    runner is injected so tests can prove reachability without executing anything.
    """
    cmds = fetch_commands(ds, dest)
    runnable = [c for c in cmds if c.argv is not None]
    if not runnable:
        print(
            f"  no runnable command for method '{ds.acquisition.method}'; "
            "this dataset cannot be fetched by this tool",
            file=sys.stderr,
        )
        return EXIT_NOT_FETCHABLE
    dest.mkdir(parents=True, exist_ok=True)
    if ds.acquisition.method in ("http", "gdrive"):
        for url in ds.acquisition.urls:
            (dest / _dest_name(ds, url)).parent.mkdir(parents=True, exist_ok=True)
    for cmd in runnable:
        print(f"  $ {cmd.display}")
        runner(cmd.argv)  # type: ignore[arg-type]
    problems = verify_files(ds, dest)
    receipt = write_receipt(ds, dest, problems)
    print(f"  receipt: {receipt}")
    if problems:
        print(f"  VERIFICATION FAILED for '{ds.name}':", file=sys.stderr)
        for p in problems:
            print(f"    - {p}", file=sys.stderr)
        return EXIT_VERIFY_FAILED
    print(f"  verified: {len(ds.files)} file(s) match files[] size and sha256")
    for key, value in ds.counts:
        print(f"  expected after extraction: {key} = {value:,}")
    return EXIT_OK


def fetch_is_authorised(yes: bool, env: dict[str, str] | None = None) -> bool:
    """Both locks must be open: --yes on the command line and the env var in the process."""
    env = os.environ if env is None else env
    return bool(yes) and env.get(ALLOW_FETCH_ENV) == ALLOW_FETCH_VALUE


# ------------------------------------------------------------------ helpers
def _basename(url: str) -> str:
    return Path(url.split("?")[0].rstrip("/")).name or "download"


def _dest_name(ds: Dataset, url: str) -> Path:
    """Where one URL lands under the dataset directory.

    The URL's trailing path components, URL-decoded, are matched against the recorded
    files[] names, longest common tail first, so two URLs that end in the same file name
    under different parent directories (Subject 1/ADL.csv and Subject 2/ADL.csv) land in
    their own recorded places instead of overwriting each other. A URL that matches no
    recorded file, or that ties between several, lands under its last path component.
    """
    parts = [unquote(p) for p in url.split("?")[0].rstrip("/").split("/") if p]
    best_len = 0
    best: list[str] = []
    for f in ds.files:
        fparts = Path(f.name).parts
        n = 0
        while n < min(len(fparts), len(parts)) and fparts[-1 - n] == parts[-1 - n]:
            n += 1
        if n and n > best_len:
            best_len, best = n, [f.name]
        elif n and n == best_len:
            best.append(f.name)
    if len(best) == 1:
        return Path(best[0])
    return Path(_basename(url))


def _rel(path_str: str) -> str:
    try:
        return str(Path(path_str).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path_str


def _wrapped(text: str, width: int = 88) -> list[str]:
    """Fold a YAML block scalar without importing textwrap semantics this tool does not need."""
    words = " ".join(str(text).split())
    if not words:
        return ["(empty)"]
    out: list[str] = []
    line = ""
    for word in words.split(" "):
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="download_dataset.py",
        description="Registry-driven dataset acquisition. Plans by default, never downloads "
        "without both --yes and " + ALLOW_FETCH_ENV + "=1.",
    )
    ap.add_argument(
        "--name", required=True, help="registry entry name (configs/datasets/<name>.yaml)"
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="dataset registry directory; defaults to configs/datasets",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument(
        "--yes",
        action="store_true",
        help=f"half of the fetch interlock; needs {ALLOW_FETCH_ENV}=1 too",
    )
    return ap


def main(argv: Sequence[str] | None = None, *, runner: Runner = _run) -> int:
    args = build_parser().parse_args(argv)

    try:
        reg = load_registry(args.config)
        ds = reg[args.name]
    except R.RegistryError as exc:
        print(f"registry error: {exc}", file=sys.stderr)
        return EXIT_UNKNOWN_DATASET

    # Gates first, always. Caught here only to render the refusal and pick an exit code;
    # no branch below this point runs for a dataset whose gate raised.
    try:
        gate_check(ds)
    except R.LegalHoldViolation as exc:
        print("\n".join(legal_hold_lines(ds)))
        print(f"{exc}", file=sys.stderr)
        return EXIT_GATE_CLOSED
    except R.CounselGateOpen as exc:
        print("\n".join(counsel_gate_lines(ds, exc)))
        print(f"{exc}", file=sys.stderr)
        return EXIT_GATE_CLOSED

    if ds.acquisition.human_required:
        print("\n".join(human_step_lines(ds)))
        return EXIT_HUMAN_REQUIRED

    dest = dest_dir(reg, ds)
    print("\n".join(plan_lines(ds, dest)))

    if ds.acquisition.method in PRODUCER_HINT:
        print("\n".join(producer_lines(ds)))
        return EXIT_NOT_FETCHABLE

    if args.dry_run:
        print("  --dry-run: stopping before any acquisition.\n")
        return EXIT_OK

    if not fetch_is_authorised(args.yes):
        print(
            "  REFUSED: acquisition needs both --yes and "
            f"{ALLOW_FETCH_ENV}={ALLOW_FETCH_VALUE} in the environment.\n"
            f"  --yes={args.yes}, {ALLOW_FETCH_ENV}={os.environ.get(ALLOW_FETCH_ENV)!r}.\n"
            "  Nothing was downloaded. Re-run with --dry-run to see the plan.",
            file=sys.stderr,
        )
        return EXIT_FETCH_NOT_AUTHORISED

    return fetch(ds, dest, runner=runner)


if __name__ == "__main__":
    sys.exit(main())
