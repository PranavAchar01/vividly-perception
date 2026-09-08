#!/usr/bin/env python
r"""Weight soups for the v2 detector seeds (research/eyes-instrument-protocol.md, I3
item 2): uniform and greedy.

Both examples below are illustrative; the scores are placeholders, not recorded
results, and TRAINING.md carries the pod invocation with its real paths.

    # uniform-N soup of same-arch seeds
    uv run python scripts/detector/soup_v2.py uniform --family rtv4_s \
        --ckpt <seed0>.pth \
        --ckpt <seed1>.pth \
        --ckpt <seed2>.pth \
        --out $OUT_DIR/soup_uniform3.pth

    # greedy soup: one candidate per invocation, score fed back from the pod
    uv run python scripts/detector/soup_v2.py greedy --family rtv4_s --state soup.json \
        --out-dir $OUT_DIR/greedy \
        --ckpt <seed0>.pth --score 0.5000 \
        --ckpt <seed1>.pth --score 0.4900 \
        --ckpt <seed2>.pth --score 0.4800
    # ...score the emitted candidate with --test-only, then:
    uv run python scripts/detector/soup_v2.py greedy --state soup.json --report-score 0.5100

What it writes. Averaged tensors go into both `model` and `ema['module']`, and
`ema['updates']` is preserved from the first ingredient. Both keys are load-bearing:
`DetSolver.val()` scores `ema.module if ema else model` (rtv4
engine/solver/det_solver.py) while the checkpoint load in the RT-DETRv4 exporter
`tools/deployment/export_onnx.py` prefers `ema.module` and falls back to `model`, so a
soup written to only one of them silently scores or exports the wrong weights. `updates`
is preserved because `ModelEMA.load_state_dict` reads it (engine/optim/ema.py) and a
missing value would reset the counter. No optimizer or scaler state is carried: a soup has no
training trajectory, and `BaseSolver.load_state_dict` simply reports that the
optimizer state was not loaded for the absent keys.

What is averaged. `--source ema` (default) averages `ema['module']`, because that is
what the trainer scores and what `best_stg2.pth` was selected on. Only floating-point
tensors are averaged. Integer buffers, of which `num_batches_tracked` is the only one
in this architecture, are copied from the first ingredient; averaging them would be
meaningless and `torch.mean` on an int64 tensor raises anyway.

=============================== Architecture guard ===============================
Measured 2026-09-05 on the two on-disk checkpoints: DEIM-D-FINE-S `e98de401` and
RT-DETRv4-S `345c30a5` are **tensor-layout identical**, 794 keys each, zero
key-set difference, zero shape difference, zero dtype difference. Both are
HGNetv2-B0 + HybridEncoder + DFINETransformer at the same widths, so a structural
check cannot tell the two families apart and a cross-family soup would load,
average and save without a single warning while producing garbage.

So the structural check below (key sets / shapes / dtypes) is necessary, not
sufficient, and `--family` is required. Ingredients whose sha256 is in
KNOWN_CHECKPOINTS are checked against it and a mismatch is fatal; unknown sha256s
are allowed with a printed warning, because an ingredient whose full digest was never
recorded cannot be looked up and refusing it outright would block the soup entirely.
`--i-know-these-are-the-same-arch` is the only override and it prints a banner. Do not
add a quieter one.
==================================================================================

Exit codes: 0 ok, 3 config/prerequisite problem, 4 architecture mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

EXIT_OK = 0
EXIT_CONFIG = 3
EXIT_ARCH = 4

# sha256 -> family. Every checkpoint whose full digest is recorded in the repository.
# B1 seed 1 `17362dca`, the one other checkpoint this script has souped, is recorded with
# eight hex characters only and so cannot be listed here; final_models/eyes/README.md
# carries that open provenance gap. check_family prints "family unverifiable" for it.
KNOWN_CHECKPOINTS = {
    "345c30a539c2b6bb665302825f418322284b98f6cbcf55d58d84ffeb75b16e7e": (
        "rtv4_s",
        "B1 seed 0, model of record (research/eyes-rtdetrv4-training-record.md)",
    ),
    "e98de401670d34d61d03100974eb72449607b319611ff454e8a865cd07b81147": (
        "deim_dfine_s",
        "DEIM seed 1, retained fallback (research/eyes-noise-floor.md)",
    ),
    "238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee": (
        "rtv4_s",
        "rtv4_s_coco.pth COCO init (third_party/PINS.yaml)",
    ),
    "a089c9eb798f8a0322e05f757fe8e8188eee34aa578788a35ec253bcec8a6b38": (
        "deim_dfine_s",
        "deim_dfine_s_coco.pth COCO init (third_party/PINS.yaml)",
    ),
}


class ArchMismatch(RuntimeError):
    """Ingredients are not the same architecture (or not provably the same)."""


# --------------------------------------------------------------------------- util


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def extract_tensors(state: dict[str, Any], source: str, path: Path) -> dict[str, torch.Tensor]:
    """Pull the weight dict the soup should average out of a trainer checkpoint."""
    if source == "ema":
        ema = state.get("ema")
        if not isinstance(ema, dict) or "module" not in ema:
            raise ArchMismatch(
                f"{path}: --source ema but the checkpoint has no ema['module'] "
                f"(top-level keys: {sorted(state)}). Use --source model if this is a "
                "non-EMA checkpoint, and say so in the run note."
            )
        return ema["module"]
    if "model" not in state:
        raise ArchMismatch(f"{path}: no 'model' key (top-level keys: {sorted(state)})")
    return state["model"]


def check_same_arch(tensor_sets: list[dict[str, torch.Tensor]], paths: list[Path]) -> None:
    """Key sets, shapes and dtypes must agree exactly. Raises ArchMismatch with a diff."""
    ref, ref_path = tensor_sets[0], paths[0]
    ref_keys = set(ref)
    for sd, path in zip(tensor_sets[1:], paths[1:], strict=True):
        keys = set(sd)
        only_ref, only_new = sorted(ref_keys - keys), sorted(keys - ref_keys)
        if only_ref or only_new:
            raise ArchMismatch(
                f"key-set mismatch between {ref_path.name} and {path.name}: "
                f"{len(only_ref)} only in the first (e.g. {only_ref[:5]}), "
                f"{len(only_new)} only in the second (e.g. {only_new[:5]})"
            )
        bad_shape = [
            (k, tuple(ref[k].shape), tuple(sd[k].shape)) for k in ref if ref[k].shape != sd[k].shape
        ]
        if bad_shape:
            raise ArchMismatch(
                f"shape mismatch between {ref_path.name} and {path.name}: "
                f"{len(bad_shape)} tensors differ, e.g. {bad_shape[:5]}"
            )
        bad_dtype = [(k, ref[k].dtype, sd[k].dtype) for k in ref if ref[k].dtype != sd[k].dtype]
        if bad_dtype:
            raise ArchMismatch(
                f"dtype mismatch between {ref_path.name} and {path.name}: "
                f"{len(bad_dtype)} tensors differ, e.g. {bad_dtype[:5]}"
            )


def check_family(shas: list[str], paths: list[Path], family: str, override: bool) -> None:
    """The guard the structural check cannot be: known sha256 -> declared family."""
    problems = []
    unknown = []
    for sha, path in zip(shas, paths, strict=True):
        known = KNOWN_CHECKPOINTS.get(sha)
        if known is None:
            unknown.append((path, sha))
        elif known[0] != family:
            problems.append(f"  {path.name} ({sha[:8]}) is {known[0]}: {known[1]}")
    for path, sha in unknown:
        print(
            f"NOTE: {path.name} sha256 {sha[:8]} is not in KNOWN_CHECKPOINTS; family unverifiable"
        )
    if problems:
        msg = (
            f"family guard: --family {family} but these ingredients are recorded otherwise:\n"
            + "\n".join(problems)
            + "\n  DEIM and RT-DETRv4 checkpoints are tensor-layout identical, so the structural "
            "check will not catch this. Souping across families produces garbage."
        )
        if not override:
            raise ArchMismatch(msg)
        print("=" * 78)
        print("Override active: proceeding across recorded families anyway:")
        print(msg)
        print("=" * 78)


def average_tensors(
    tensor_sets: list[dict[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Uniform mean of float tensors; non-float copied from the first. -> (sd, n_avg, n_copied)."""
    ref = tensor_sets[0]
    out: dict[str, torch.Tensor] = {}
    n_avg = n_copied = 0
    n = len(tensor_sets)
    for key, ref_tensor in ref.items():
        if ref_tensor.is_floating_point():
            acc = ref_tensor.detach().to(torch.float64).clone()
            for sd in tensor_sets[1:]:
                acc += sd[key].detach().to(torch.float64)
            out[key] = (acc / n).to(ref_tensor.dtype)
            n_avg += 1
        else:
            out[key] = ref_tensor.detach().clone()
            n_copied += 1
    return out, n_avg, n_copied


def make_soup(
    paths: list[Path],
    out: Path,
    family: str,
    source: str = "ema",
    override: bool = False,
    note: str = "",
) -> dict[str, Any]:
    """Load, guard, average, write to both keys, hash. Returns the sidecar payload."""
    if len(paths) < 2:
        raise ValueError(f"a soup needs at least 2 ingredients, got {len(paths)}")

    shas = [sha256_file(p) for p in paths]
    check_family(shas, paths, family, override)

    states = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    tensor_sets = [extract_tensors(s, source, p) for s, p in zip(states, paths, strict=True)]
    check_same_arch(tensor_sets, paths)

    souped, n_avg, n_copied = average_tensors(tensor_sets)
    first = states[0]
    updates = first.get("ema", {}).get("updates") if isinstance(first.get("ema"), dict) else None

    payload: dict[str, Any] = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "last_epoch": first.get("last_epoch", -1),  # carried, meaningless for a soup
        "model": souped,
        "ema": {"module": souped, "updates": updates},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    out_sha = sha256_file(out)

    meta = {
        "kind": "weight_soup",
        "family": family,
        "source_key": source,
        "n_ingredients": len(paths),
        "ingredients": [{"path": str(p), "sha256": s} for p, s in zip(paths, shas, strict=True)],
        "tensors_averaged": n_avg,
        "tensors_copied_from_first": n_copied,
        "ema_updates_from": str(paths[0]),
        "ema_updates": updates,
        "written_to_keys": ["model", "ema.module"],
        "keys_share_storage": True,  # one tensor set saved once; both keys point at it
        "out": str(out),
        "out_sha256": out_sha,
        "family_override": bool(override),
        "note": note,
        "written_by": "scripts/detector/soup_v2.py",
        "date": payload["date"],
    }
    out.with_suffix(out.suffix + ".sidecar.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"souped {len(paths)} checkpoints ({n_avg} tensors averaged, {n_copied} copied)")
    for p, s in zip(paths, shas, strict=True):
        print(f"  ingredient {s[:8]}  {p}")
    print(f"wrote {out}")
    print(f"sha256  {out_sha}")
    return meta


# ------------------------------------------------------------------------- uniform


def cmd_uniform(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.ckpt]
    for p in paths:
        if not p.is_file():
            print(f"FATAL: no such checkpoint: {p}", file=sys.stderr)
            return EXIT_CONFIG
    try:
        make_soup(paths, args.out, args.family, args.source, args.override, args.note)
    except ArchMismatch as exc:
        print(f"FATAL (architecture): {exc}", file=sys.stderr)
        return EXIT_ARCH
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    return EXIT_OK


# -------------------------------------------------------------------------- greedy


def _load_state(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def _emit_candidate(state: dict[str, Any], out_dir: Path, override: bool) -> int:
    """Build the next candidate soup (accepted + cursor) and record it as pending."""
    ings = state["ingredients"]
    cursor = state["cursor"]
    if cursor >= len(ings):
        return _finalise(state, out_dir, override)

    indices = [*state["accepted"], cursor]
    paths = [Path(ings[i]["path"]) for i in indices]
    tag = "-".join(str(i) for i in indices)
    out = out_dir / f"greedy_cand_{tag}.pth"
    meta = make_soup(
        paths, out, state["family"], state["source"], override, note=f"greedy candidate {tag}"
    )
    state["pending"] = {"indices": indices, "out": str(out), "sha256": meta["out_sha256"]}
    print()
    print("Next: score this candidate on the pod, then feed the number back:")
    print(f"  cd $TRAINER && .venv/bin/python train.py -c $CONFIG --test-only -r {out}")
    print(f"  <this script> greedy --state {state['_state_path']} --report-score <AP>")
    print(f"  current best (accepted {state['accepted']}): {state['best_score']}")
    return EXIT_OK


def _finalise(state: dict[str, Any], out_dir: Path, override: bool) -> int:
    """No new file is written here on purpose.

    The winning soup is the exact candidate that was scored; re-souping the same
    ingredients would produce a different sha256 (the `date` field alone changes the
    bytes) and the recorded AP would no longer identify the file it came from, which
    is precisely what the rule that every accuracy number carries its checkpoint hash
    (POLICY.md, rule 10) exists to prevent.
    """
    del out_dir, override  # nothing new is built; kept for signature symmetry
    ings = state["ingredients"]
    accepted = state["accepted"]
    print()
    print("=" * 78)
    print(f"Greedy soup complete. accepted {accepted} of {len(ings)}; rejected {state['rejected']}")
    for i in accepted:
        print(f"  kept  #{i} score {ings[i]['score']}  {ings[i]['path']}")
    for i in state["rejected"]:
        print(f"  drop  #{i} score {ings[i]['score']}  {ings[i]['path']}")
    print(f"best measured score: {state['best_score']}")
    if len(accepted) == 1:
        print("Only the best single seed survived, so there is no soup to write.")
        print(f"The claim is that seed {ings[accepted[0]]['path']} stands alone.")
        print("=" * 78)
        state["final"] = {
            "single": ings[accepted[0]]["path"],
            "sha256": ings[accepted[0]]["sha256"],
        }
        return EXIT_OK
    best = state.get("best_out")
    print(f"Winner (already scored, do not rebuild): {best}")
    print(f"sha256  {state.get('best_sha')}")
    print("Bar for the claim: best single seed + max(0.0012, the measured seed spread).")
    print("=" * 78)
    state["final"] = {"out": best, "sha256": state.get("best_sha")}
    return EXIT_OK


def cmd_greedy(args: argparse.Namespace) -> int:
    state_path: Path = args.state
    out_dir: Path = args.out_dir

    if state_path.is_file():
        state = _load_state(state_path)
        if args.ckpt:  # re-supplied: they must match what the state recorded
            supplied = [sha256_file(Path(p)) for p in args.ckpt]
            recorded = [i["sha256"] for i in state["ingredients"]]
            if sorted(supplied) != sorted(recorded):
                print(
                    f"FATAL: {state_path} was started with a different ingredient set. "
                    "Delete it to start over, or drop --ckpt to continue this run.",
                    file=sys.stderr,
                )
                return EXIT_CONFIG
    else:
        if not args.ckpt or len(args.ckpt) != len(args.score):
            print(
                "FATAL: the first greedy call needs one --score per --ckpt (the seed's own "
                "val AP, used only to order the ingredients).",
                file=sys.stderr,
            )
            return EXIT_CONFIG
        if len(args.ckpt) < 2:
            print("FATAL: a greedy soup needs at least 2 ingredients", file=sys.stderr)
            return EXIT_CONFIG
        paths = [Path(p) for p in args.ckpt]
        for p in paths:
            if not p.is_file():
                print(f"FATAL: no such checkpoint: {p}", file=sys.stderr)
                return EXIT_CONFIG
        ings = [
            {"path": str(p), "sha256": sha256_file(p), "score": float(s)}
            for p, s in zip(paths, args.score, strict=True)
        ]
        ings.sort(key=lambda d: -d["score"])  # greedy soup ingests in val-score order
        state = {
            "family": args.family,
            "source": args.source,
            "ingredients": ings,
            "accepted": [0],  # the best single model always starts the soup
            "rejected": [],
            "cursor": 1,
            "best_score": ings[0]["score"],
            "pending": None,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        print(f"greedy soup initialised: {len(ings)} ingredients in val-score order")
        for i, ing in enumerate(ings):
            print(f"  #{i} {ing['score']:.4f}  {ing['sha256'][:8]}  {ing['path']}")
        print(f"seeded with #0 (score {state['best_score']}) as the standing best")

    state["_state_path"] = str(state_path)

    if args.report_score is not None:
        pending = state.get("pending")
        if not pending:
            print(
                f"FATAL: --report-score given but {state_path} has no pending candidate. "
                "Run without --report-score to emit one.",
                file=sys.stderr,
            )
            return EXIT_CONFIG
        idx = pending["indices"][-1]
        score = float(args.report_score)
        improved = score > state["best_score"]
        verb = "accept" if improved else "reject"
        print(
            f"{verb} #{idx}: candidate {pending['sha256'][:8]} scored {score:.5f} "
            f"vs standing best {state['best_score']:.5f}"
        )
        if improved:
            state["accepted"] = pending["indices"]
            state["best_score"] = score
            # the scored file IS the artefact; _finalise must never rebuild it
            state["best_out"] = pending["out"]
            state["best_sha"] = pending["sha256"]
        else:
            state["rejected"].append(idx)
        state.setdefault("history", []).append(
            {
                "indices": pending["indices"],
                "score": score,
                "accepted": improved,
                "sha256": pending["sha256"],
            }
        )
        state["pending"] = None
        state["cursor"] = idx + 1
    elif state.get("pending"):
        print(
            f"NOTE: {state_path} already has a pending candidate {state['pending']['out']}. "
            "Score it and come back with --report-score; re-emitting instead.",
        )
        state["pending"] = None

    rc = _emit_candidate(state, out_dir, args.override)
    state.pop("_state_path", None)
    _save_state(state_path, state)
    return rc


# ---------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--family", default=None, help="rtv4_s | deim_dfine_s (required to start)")
        p.add_argument("--source", choices=("ema", "model"), default="ema")
        p.add_argument(
            "--i-know-these-are-the-same-arch",
            dest="override",
            action="store_true",
            help="override the recorded-family guard; prints a banner",
        )
        p.add_argument("--note", default="")

    uni = sub.add_parser("uniform", help="uniform average of N same-arch checkpoints")
    uni.add_argument("--ckpt", action="append", required=True, help="repeat per ingredient")
    uni.add_argument("--out", required=True, type=Path)
    common(uni)
    uni.set_defaults(func=cmd_uniform)

    grd = sub.add_parser("greedy", help="greedy soup; one candidate per call, score fed back")
    grd.add_argument("--ckpt", action="append", default=[], help="first call only")
    grd.add_argument("--score", action="append", type=float, default=[], help="first call only")
    grd.add_argument("--state", required=True, type=Path)
    grd.add_argument("--out-dir", type=Path, default=Path())
    grd.add_argument(
        "--report-score", type=float, default=None, help="score of the pending candidate"
    )
    common(grd)
    grd.set_defaults(func=cmd_greedy)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    starting_fresh = args.cmd == "uniform" or not args.state.is_file()
    if starting_fresh and not args.family:
        ap.error("--family is required (rtv4_s | deim_dfine_s); see the architecture guard note")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
