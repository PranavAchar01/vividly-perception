#!/usr/bin/env python
r"""Uniform weight soup of the v2 RTMPose-m seed checkpoints: written and hashed, not scored.

    uv run python scripts/pose/soup_checkpoints.py \
        --ckpt outputs/pod_backup/p0_best_e38_a0a035fe.pth \
        --ckpt outputs/pod_backup/p1_best_e30_8d990435.pth \
        --ckpt outputs/pod_backup/p2_best_e32_5bcca5bd.pth \
        --out  outputs/pose_soup_v2/pose_soup_v2_3seed.pth

That is the command recorded in research/pose-weight-soup.md. Every path in it sits under
`outputs/`, which is gitignored, so neither the three ingredient checkpoints nor the soup
and its sidecar are in a clone; that page is the committed record of their digests.

This is the pose twin of `scripts/detector/soup_v2.py`, and it exists because the two
lanes' checkpoints are not the same kind of file. The detector seeds come out of the
rtv4/DEIM trainer, which pickles nothing but tensors and plain containers, so plain
`torch.load(..., weights_only=False)` reads them. The pose seeds come out of mmengine,
which pickles live framework objects into the checkpoint: the top-level `message_hub`
key holds `mmengine.logging.history_buffer.HistoryBuffer` instances (the scalar log
ring-buffers). This script imports torch at module scope and needs nothing else; away from
the training pod that torch comes from the `curate` environment, which carries no mmengine,
because the mmengine of record (onedl-mmengine 0.10.12, research/pose-stack.md) is installed
only in the vendored pose trainer environment. So the default unpickler dies on
`ModuleNotFoundError: mmengine` before it ever reaches the weights.

How it reads them. `_StubbingUnpickler` resolves every `mmengine.*` global to one inert
placeholder class, records which globals were stubbed, and lets the rest of the pickle
(torch tensors, numpy, OrderedDict) resolve normally. The placeholder needs a metaclass
`__getattr__` because the pickle stream contains `getattr(HistoryBuffer, 'min')`, that is
the statistics methods HistoryBuffer stores by reference, and a bare class raises
AttributeError there. Nothing stubbed is ever read: the stubs live only in `message_hub`,
which this script drops.

What is averaged. `state_dict` only, and `state_dict` is already the EMA-swapped weight
set: mmengine's `EMAHook._swap_ema_state_dict` swaps `checkpoint['state_dict']` with
`checkpoint['ema_state_dict']['module.*']` in `before_save_checkpoint`, precisely so a
deployment loader can take `state_dict` and get the averaged model. So after the swap
`state_dict` = EMA weights (what the trainer selected `best_coco_AP_*` on and what
`export_rtmpose_onnx.py` exports) and `ema_state_dict` holds the raw online weights.
Souping `ema_state_dict` would average the wrong half. `--source` can say so explicitly;
the default is the EMA-swapped `state_dict`.

Only floating-point tensors are averaged, in float64, then cast back to the ingredient
dtype. The 57 integer `num_batches_tracked` buffers are copied from the first ingredient:
averaging a step counter is meaningless and `torch.mean` on int64 raises anyway.

What it writes. `{'meta': <first ingredient's meta>, 'state_dict': <soup>}` and nothing
else. `message_hub` is dropped (it is the unpicklable part, it is per-run logging state,
and a soup has no run), `ema_state_dict` is dropped (the online weights of a single seed
are not a property of the soup), and no optimizer state is carried (a soup has no training
trajectory). `meta` is carried across unchanged from the first ingredient because it holds
the recipe text (`meta['cfg']`), `dataset_meta` (mmpose needs it to decode), and the seed,
epoch and experiment_name of that ingredient, so the sidecar records that it describes
ingredient 0 only.

Do not put the digest in --out. Measured 2026-09-05: `torch.save` writes a zip whose
internal archive prefix is the output file's own basename, so the same soup written to
`pose_soup_v2_3seed.pth` and to `pose_soup_v2_3seed_46e6a8e8.pth` hashes to `46e6a8e8...`
and `5dc02ce9...` respectively. Naming the file after its digest therefore invalidates the
digest, so the sha8-in-the-name convention the ONNX exports use cannot work here. The name
is fixed and digest-free; the digest lives in the sidecar, in the results document and in
the lane page. (Same bytes for the same name: re-running reproduces `46e6a8e8...` exactly.)

The number this does not produce. A soup is a claim about accuracy and this script cannot
score one. Scoring RTMPose needs mmpose, mmcv and mmengine on a CUDA box; none of that
exists in the environment described above and the v2 lane does not put it there. So the
artifact is written, hashed and recorded as unscored, and the pod-side single look stays
pre-declared: anything below the standing bar is no result, not a rival. The bar recorded
in-repo is 0.7742 (the model of record, checkpoint a0a035fe, at 0.774115 plus 2 sigma;
research/pose-seed-floor-and-champion.md). Do not quote a soup AP that no run produced, as
POLICY.md, rule 9 requires.

Exit codes: 0 ok, 3 config/prerequisite problem, 4 architecture mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

EXIT_OK = 0
EXIT_CONFIG = 3
EXIT_ARCH = 4

# sha256 -> (family, note). The pose twin of soup_v2.py's table. Same posture: a known
# digest whose family disagrees with --family is fatal; an unknown digest warns and
# proceeds, because the structural check below still has to pass.
KNOWN_CHECKPOINTS = {
    "a0a035fe65909980447560a70ae2f8d13400c1d12490a929e7cf3c5a7b26f102": (
        "rtmpose_m_person_v2",
        "P0 seed 0 e38, model of record, AP 0.774115 (research/pose-seed-floor-and-champion.md)",
    ),
    "8d99043597076a672e7f4bc1ce886111d06a62de287d4801e489080561fa40ae": (
        "rtmpose_m_person_v2",
        "P1 seed 1 e30, AP 0.774000 (research/pose-seed-floor-and-champion.md)",
    ),
    "5bcca5bd3a33c18fbba70db89a6d9d8f9c2703679b3c820dc267090546cf5f07": (
        "rtmpose_m_person_v2",
        "P2 seed 2 e32, AP 0.774057 (research/pose-seed-floor-and-champion.md)",
    ),
    "e48f03d0cfe1285ee8b6d3457ac3ce33a4594a92b080053ab1ec4a7e300975f2": (
        "rtmpose_m_body7",
        "body7 init / recipe gate 0.764499 (research/pose-seed-floor-and-champion.md)",
    ),
}

STUBBED_MODULE_PREFIXES = ("mmengine",)


class ArchMismatch(RuntimeError):
    """Ingredients are not the same architecture (or not provably the same)."""


# ------------------------------------------------------------------ stubbing unpickler


def _inert(*_args: Any, **_kwargs: Any) -> None:
    """Stand-in for any attribute looked up on a stubbed class. Returns None, always."""
    return None


class _StubMeta(type):
    """`getattr(HistoryBuffer, 'min')` appears in the pickle stream; answer everything."""

    def __getattr__(cls, name: str):  # metaclass hook: `cls` is the first argument here
        return _inert


class MMEngineStub(metaclass=_StubMeta):
    """Inert placeholder for an unresolvable mmengine global.

    Absorbs any constructor arguments and any `__setstate__` payload so the pickle
    machinery can finish building the object graph. Nothing ever reads one back: every
    stub in these checkpoints lives under `message_hub`, and `message_hub` is dropped
    before the soup is written.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__["_stub_args"] = (args, kwargs)

    def __setstate__(self, state: Any) -> None:
        self.__dict__["_stub_state"] = state

    def __getattr__(self, name: str):
        return _inert

    def __repr__(self) -> str:
        return "<MMEngineStub>"


def make_stub_pickle_module() -> tuple[types.ModuleType, set[str]]:
    """A `pickle_module` for torch.load that survives absent mmengine.

    Returns (module, stubbed) where `stubbed` fills in with the "module.Name" of every
    global that was replaced, so the run can print what it substituted rather than
    substitute it silently.
    """
    stubbed: set[str] = set()

    class _StubbingUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.split(".")[0] in STUBBED_MODULE_PREFIXES:
                stubbed.add(f"{module}.{name}")
                return MMEngineStub
            return super().find_class(module, name)

    mod = types.ModuleType("vividly_stub_pickle")
    mod.Unpickler = _StubbingUnpickler  # type: ignore[attr-defined]
    mod.Pickler = pickle.Pickler  # type: ignore[attr-defined]
    mod.load = pickle.load  # type: ignore[attr-defined]
    mod.dump = pickle.dump  # type: ignore[attr-defined]
    mod.loads = pickle.loads  # type: ignore[attr-defined]
    mod.dumps = pickle.dumps  # type: ignore[attr-defined]
    mod.UnpicklingError = pickle.UnpicklingError  # type: ignore[attr-defined]
    return mod, stubbed


def load_mm_checkpoint(path: Path) -> tuple[dict[str, Any], set[str]]:
    """torch.load an mmengine checkpoint without mmengine installed.

    `weights_only=False` is required and is not a loosening of anything: torch >= 2.6
    defaults to a restricted unpickler that ignores `pickle_module` entirely, so the
    stub would never be consulted and the load would fail on the mmengine global. These
    files are this project's own pod output, hash-checked against KNOWN_CHECKPOINTS
    above; the trust boundary is the digest, not the unpickler.
    """
    stub_module, stubbed = make_stub_pickle_module()
    state = torch.load(path, map_location="cpu", weights_only=False, pickle_module=stub_module)
    if not isinstance(state, dict):
        raise ArchMismatch(f"{path}: expected a dict checkpoint, got {type(state).__name__}")
    return state, stubbed


# --------------------------------------------------------------------------------- util


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def extract_state_dict(state: dict[str, Any], source: str, path: Path) -> dict[str, Any]:
    """Pull the weight dict to average. See the note on what is averaged in the docstring."""
    if source not in state:
        raise ArchMismatch(
            f"{path}: no {source!r} key (top-level keys: {sorted(state)}). "
            "mmengine checkpoints carry 'state_dict' (EMA-swapped) and, when EMAHook ran, "
            "'ema_state_dict' (the raw online weights, prefixed 'module.')."
        )
    sd = state[source]
    if source == "ema_state_dict":
        # strip the 'module.' prefix and the non-tensor 'steps' counter so the two
        # sources are structurally comparable
        sd = {k[len("module.") :]: v for k, v in sd.items() if k.startswith("module.")}
    bad = [k for k, v in sd.items() if not torch.is_tensor(v)]
    if bad:
        raise ArchMismatch(f"{path}: {len(bad)} non-tensor entries in {source} (e.g. {bad[:5]})")
    return sd


def check_same_arch(tensor_sets: list[dict[str, Any]], paths: list[Path]) -> None:
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
    """Known digest -> declared family. The guard the structural check cannot be."""
    problems, unknown = [], []
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
            + "\n  RTMPose-m variants are tensor-layout identical across training mixtures "
            "(body7 and person_v2 differ only in weight values), so the structural check "
            "will not catch a cross-lineage soup."
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
    source: str = "state_dict",
    override: bool = False,
    note: str = "",
) -> dict[str, Any]:
    """Load, guard, average, write, hash. Returns the sidecar payload."""
    if len(paths) < 2:
        raise ValueError(f"a soup needs at least 2 ingredients, got {len(paths)}")

    shas = [sha256_file(p) for p in paths]
    check_family(shas, paths, family, override)

    states: list[dict[str, Any]] = []
    stubbed: set[str] = set()
    for p in paths:
        st, sb = load_mm_checkpoint(p)
        states.append(st)
        stubbed |= sb
    if stubbed:
        print(f"stubbed {len(stubbed)} unresolvable global(s): {sorted(stubbed)}")

    tensor_sets = [extract_state_dict(s, source, p) for s, p in zip(states, paths, strict=True)]
    check_same_arch(tensor_sets, paths)

    souped, n_avg, n_copied = average_tensors(tensor_sets)

    first_meta = states[0].get("meta")
    meta = dict(first_meta) if isinstance(first_meta, dict) else {}
    # Belt and braces: `message_hub` is a TOP-LEVEL key in these files, not a meta key,
    # so this pop is normally a no-op. It stays because dropping the framework-state blob
    # is the invariant, not the key's current address.
    meta.pop("message_hub", None)

    payload = {"meta": meta, "state_dict": souped}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    out_sha = sha256_file(out)

    sidecar = {
        "kind": "weight_soup",
        "lane": "pose_v2",
        "family": family,
        "source_key": source,
        "source_is_ema_swapped": source == "state_dict",
        "n_ingredients": len(paths),
        "ingredients": [
            {
                "path": str(p),
                "sha256": s,
                "seed": states[i].get("meta", {}).get("seed"),
                "epoch": states[i].get("meta", {}).get("epoch"),
                "experiment_name": states[i].get("meta", {}).get("experiment_name"),
            }
            for i, (p, s) in enumerate(zip(paths, shas, strict=True))
        ],
        "state_dict_keys": len(souped),
        "tensors_averaged": n_avg,
        "tensors_copied_from_first": n_copied,
        "stubbed_globals": sorted(stubbed),
        "dropped_keys": sorted(set(states[0]) - {"meta", "state_dict"}),
        "meta_from": str(paths[0]),
        "out": str(out),
        "out_sha256": out_sha,
        "family_override": bool(override),
        "scored": False,
        "scored_note": (
            "Unscored where it was written: no mmpose, mmcv or mmengine environment. The "
            "pod-side single look is pre-declared; recorded bar 0.7742 = the model of "
            "record, checkpoint a0a035fe, at 0.774115 plus 2 sigma "
            "(research/pose-seed-floor-and-champion.md). Below the bar is no result."
        ),
        "note": note,
        "written_by": "scripts/pose/soup_checkpoints.py",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out.with_suffix(out.suffix + ".sidecar.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    print(
        f"souped {len(paths)} checkpoints from '{source}' "
        f"({len(souped)} state_dict keys: {n_avg} averaged, {n_copied} copied)"
    )
    for p, s in zip(paths, shas, strict=True):
        print(f"  ingredient {s}  {p}")
    print(f"wrote {out}")
    print(f"soup sha256 {out_sha}")
    print("Unscored here (no mmpose environment). Do not quote an AP for this file.")
    return sidecar


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", action="append", required=True, help="repeat once per ingredient")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--family", default="rtmpose_m_person_v2")
    ap.add_argument(
        "--source",
        choices=("state_dict", "ema_state_dict"),
        default="state_dict",
        help="state_dict is the EMA-swapped weight set and is the correct default",
    )
    ap.add_argument(
        "--i-know-these-are-the-same-arch",
        dest="override",
        action="store_true",
        help="override the recorded-family guard; prints a banner",
    )
    ap.add_argument("--note", default="")
    ap.set_defaults(func=cmd_uniform)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
