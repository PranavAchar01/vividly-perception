#!/usr/bin/env python3
"""OSNet to ONNX export, with the weight-provenance rules wired into the command line.

    UV_PROJECT_ENVIRONMENT=.venv-curate uv sync --extra curate --no-default-groups
    .venv-curate/bin/python scripts/reid/export_osnet_onnx.py \
        --weights checkpoints/osnet_x1_0_imagenet.pth \
        --out outputs/reid_mevid_a1/osnet_x1_0.onnx

This is the bridge between the two halves of the weight-provenance rule in POLICY.md,
rule 11: torchreid (MIT, the canonical KaiyangZhou/deep-person-reid distribution) lives
in the `cuda` and `curate` forks only, and the mac dev fork stays torch-free and runs
appearance embeddings through onnxruntime. Turning a .pth into a .onnx is the one step
that has to cross that boundary, and this script is that step and nothing else.

It never downloads. `--weights` is a required local path, a URL is refused outright, and
no torchreid model-zoo auto-fetch is reachable from here (`pretrained=False`, always). A
person fetches the weight file deliberately, records its sha256, and writes the model card
before first use; see final_models/reid/README.md, section "Identity and provenance".

It also refuses the banned lineages by name. A path naming any token in
DENIED_LINEAGE_TOKENS below, that is luperson, solider, transreidssl, passreid, personvit,
dukemtmc or kadirnar, exits 6 without reading a byte, because those are the ways a
non-shippable weight arrives in practice: LUPerson's terms forbid commercial use, SOLIDER,
TransReID-SSL, PASS-reID and PersonViT are LUPerson descendants, DukeMTMC was retracted in
2019 over an IRB breach and taints every derivative, and kadirnar/osnet is the
GPL-3.0-labelled Hugging Face repack. See docs/LICENCE_ALLOWLIST.md, section
"Re-identification weight sources", recorded 2026-08-07. That check is a tripwire on
the obvious cases, not a provenance system: the provenance record is the sidecar json this
script writes plus the model card a person writes. A weight file with an innocent name and
a LUPerson lineage passes this check and is still forbidden.

Exit codes:
    0  onnx + sidecar written
    2  usage error (argparse)
    3  torchreid (or torch) is not importable in this environment
    4  weights problem: a URL, a missing path, a directory, or an unreadable checkpoint
    5  export failed, or the loaded checkpoint does not match the requested variant
    6  licence refusal: the path names a banned re-ID lineage
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only; the runtime imports stay inside run()
    import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The only sanctioned torchreid distribution. The `torchreid` name on the public index
#: is a stale 0.2.5 sdist from 2022 and is not this project; it is never substituted here.
TORCHREID_GIT_URL = "https://github.com/KaiyangZhou/deep-person-reid.git"
TORCHREID_GIT_REV = "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"

DEFAULT_VARIANT = "osnet_x1_0"
#: ImageNet head width. The classifier is dropped on export either way, so this only has to
#: let the checkpoint's own classifier tensor match; a mismatch is skipped, never fatal.
DEFAULT_NUM_CLASSES = 1000
#: Conservative on purpose. onnxruntime supports every opset from 7 upward, so a low opset
#: is the compatible choice rather than the risky one, and OSNet is conv/bn/relu/pool/gemm
#: only, all of which have been stable since opset 13. Raise it only with a measured reason.
DEFAULT_OPSET = 13
#: OSNet's native input geometry. Batch is the only dynamic axis: a re-ID batch is N crops
#: of one fixed size, and letting H/W float would silently change the embedding.
INPUT_SHAPE = (1, 3, 256, 128)
INPUT_NAME = "images"
OUTPUT_NAME = "embeddings"
BATCH_AXIS = "batch"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NO_TORCHREID = 3
EXIT_WEIGHTS = 4
EXIT_EXPORT = 5
EXIT_LICENCE = 6

#: Substrings that identify a weight file this script may not export, matched against the whole
#: path after case-folding and collapsing `-`, ` ` and `.` to `_`.
DENIED_LINEAGE_TOKENS: tuple[str, ...] = (
    "luperson",
    "solider",
    "transreidssl",
    "passreid",
    "personvit",
    "dukemtmc",
    "kadirnar",
)

#: Copied into every sidecar so the rule travels with the artefact, not just with the repo.
SOURCE_RULE = (
    "OSNet weights come only from the canonical MIT torchreid distribution "
    "(KaiyangZhou/deep-person-reid), ImageNet-pretrained variant preferred. The "
    "kadirnar/osnet Hugging Face repack is labelled GPL-3.0 and is banned. LUPerson-derived "
    "weights (SOLIDER, TransReID-SSL and kin) are ship=never, benchmarking only. "
    "Market-1501-trained variants are eval-only until counsel says otherwise. "
    "Recorded 2026-08-07; docs/LICENCE_ALLOWLIST.md."
)

_URL_SCHEMES = ("http://", "https://", "ftp://", "s3://", "gs://", "hf://")


class LicenceRefusal(Exception):
    """The path names a lineage that the weight-provenance rule forbids."""


class WeightsError(Exception):
    """`--weights` is not a readable local checkpoint."""


class MissingTorchreid(Exception):
    """torchreid or torch is absent from this interpreter."""


class ExportError(Exception):
    """The model would not build, load, or export."""


# ------------------------------------------------------------------ licence tripwire
def normalise_for_lineage(text: str) -> str:
    """Case-fold and strip separators, so `TransReID-SSL`, `transreid_ssl`, and
    `TransReIDSSL` all reduce to the same separator-free string. The deny tokens are
    spelled separator-free to match."""
    lowered = text.lower()
    for sep in ("-", "_", " ", ".", "/", "\\"):
        lowered = lowered.replace(sep, "")
    return lowered


def denied_lineage_tokens(text: str) -> list[str]:
    """Banned lineage tokens present in `text`, in declaration order."""
    haystack = normalise_for_lineage(text)
    return [token for token in DENIED_LINEAGE_TOKENS if token in haystack]


def check_lineage(label: str, value: str) -> None:
    """Raise LicenceRefusal if `value` names a banned lineage."""
    hits = denied_lineage_tokens(value)
    if hits:
        raise LicenceRefusal(
            f"{label} '{value}' names banned re-ID lineage {hits}. "
            "LUPerson-derived weights (SOLIDER, TransReID-SSL and kin) are ship=never, and "
            "the kadirnar/osnet repack is GPL-3.0-labelled and banned outright. See "
            "docs/LICENCE_ALLOWLIST.md, 'Re-identification weight sources'. Nothing was "
            "read or written."
        )


# ------------------------------------------------------------------ weights resolution
def resolve_weights(raw: str) -> Path:
    """Turn `--weights` into an existing local file, refusing every remote form."""
    if raw.startswith(_URL_SCHEMES) or "://" in raw:
        raise WeightsError(
            f"--weights '{raw}' is a URL. This script never downloads anything. Fetch the "
            "checkpoint deliberately, record its sha256, write its provenance, then pass "
            "the local path. See final_models/reid/README.md."
        )
    path = Path(raw).expanduser()
    if not path.exists():
        raise WeightsError(
            f"--weights '{raw}' does not exist at {path}. No weight file is committed to "
            "this repo and nothing here fetches one."
        )
    if not path.is_file():
        raise WeightsError(f"--weights '{raw}' is not a file ({path}).")
    return path


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ model build + load
def import_torchreid() -> Any:
    """Import torchreid lazily, with a message that names the fork that has it."""
    try:
        # torchreid is MIT (KaiyangZhou/deep-person-reid) and sanctioned in the cuda and
        # curate forks; see docs/LICENCE_ALLOWLIST.md.
        import torchreid
    except ImportError as exc:
        raise MissingTorchreid(
            "torchreid is not importable in this interpreter.\n"
            "The mac dev fork is torch-free by design: embedding inference there goes "
            "through onnxruntime, which is what this script exists to feed.\n"
            "Run the export from the curate fork, which pins the canonical MIT distribution "
            f"at rev {TORCHREID_GIT_REV[:12]}:\n"
            "    UV_PROJECT_ENVIRONMENT=.venv-curate uv sync --extra curate "
            "--no-default-groups\n"
            "    .venv-curate/bin/python scripts/reid/export_osnet_onnx.py ...\n"
            "On the training pod the same library is installed from a clone at that pin "
            "with `uv pip install --no-deps --no-build-isolation`, which TRAINING.md "
            "records; the `cuda` extra does not carry it."
        ) from exc
    return torchreid


def build_model(variant: str, num_classes: int) -> torch.nn.Module:
    """Build an untrained torchreid model. `pretrained=False` is not negotiable.

    `pretrained=True` would make torchreid reach for its model zoo over the network, which
    both breaks the no-download rule and bypasses the provenance record entirely.
    """
    torchreid = import_torchreid()
    try:
        model = torchreid.models.build_model(
            name=variant, num_classes=num_classes, pretrained=False
        )
    except KeyError as exc:
        # torchreid's own KeyError already lists every registered model name.
        raise ExportError(f"unknown torchreid variant '{variant}': {exc}") from exc
    model.eval()
    return model


def load_weights(model: torch.nn.Module, weights: Path) -> dict[str, Any]:
    """Load a local checkpoint into `model`, and refuse a silent no-op.

    torchreid's own `load_pretrained_weights` prints a warning and carries on when nothing
    matched, which is exactly how a wrong-variant checkpoint turns into an ONNX file full
    of random numbers. Here, zero matched tensors is a hard error.

    `weights_only=True` is passed explicitly: a .pth is a pickle, and a checkpoint that
    needs arbitrary code execution to load is refused rather than trusted.
    """
    import torch

    try:
        raw = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several unrelated types from here
        raise WeightsError(
            f"could not read {weights} as a plain tensor checkpoint: {exc}. This script "
            "loads with weights_only=True and does not fall back; a checkpoint that "
            "requires unpickling arbitrary objects is not executed."
        ) from exc

    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise WeightsError(f"{weights} does not contain a state dict (got {type(state)}).")

    target = model.state_dict()
    matched: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in state.items():
        name = key[len("module.") :] if key.startswith("module.") else key
        if name in target and target[name].shape == value.shape:
            matched[name] = value
        else:
            skipped.append(name)
    if not matched:
        raise ExportError(
            f"{weights} shares no tensor with the requested model: 0 of {len(state)} keys "
            "matched by name and shape. Wrong variant, or not a torchreid checkpoint."
        )
    model.load_state_dict(matched, strict=False)
    return {
        "tensors_in_checkpoint": len(state),
        "tensors_loaded": len(matched),
        "tensors_skipped": sorted(skipped),
        "tensors_left_at_init": sorted(set(target) - set(matched)),
    }


# ------------------------------------------------------------------ export
def export_onnx(model: torch.nn.Module, out: Path, opset: int) -> dict[str, Any]:
    """Write `out` and return the graph facts worth recording."""
    import torch

    dummy = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        embedding = model(dummy)
    if embedding.ndim != 2:
        raise ExportError(
            f"expected a 2-D (batch, dim) embedding in eval mode, got shape "
            f"{tuple(embedding.shape)}. The model is probably still in training mode."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {INPUT_NAME: {0: BATCH_AXIS}, OUTPUT_NAME: {0: BATCH_AXIS}}
    # dynamo=False keeps the TorchScript exporter, which honours `dynamic_axes` directly and
    # needs no onnxscript. torch >= 2.9 defaults to the torch.export path; the old path is
    # pinned here so the emitted graph does not change under a torch bump. The kwarg does not exist
    # before torch 2.5, hence the introspection rather than a version comparison.
    kwargs: dict[str, Any] = {}
    exporter = "torchscript"
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    else:  # pragma: no cover - only reachable on torch < 2.5
        exporter = "torchscript (implicit)"
    try:
        torch.onnx.export(
            model,
            (dummy,),
            str(out),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            **kwargs,
        )
    except Exception as exc:  # the exporter raises many unrelated types
        raise ExportError(f"torch.onnx.export failed: {exc}") from exc
    if not out.is_file():
        raise ExportError(f"torch.onnx.export returned without writing {out}")

    graph: dict[str, Any] = {
        "exporter": exporter,
        "opset": opset,
        "input_name": INPUT_NAME,
        "input_shape": list(INPUT_SHAPE),
        "output_name": OUTPUT_NAME,
        "embedding_dim": int(embedding.shape[1]),
        "dynamic_axes": {INPUT_NAME: [0], OUTPUT_NAME: [0]},
        "onnx_checker": "not run (onnx package absent)",
    }
    try:
        import onnx
    except ImportError:
        return graph
    onnx.checker.check_model(str(out))
    graph["onnx_checker"] = "passed"
    graph["onnx_ir_version"] = int(onnx.load(str(out)).ir_version)
    return graph


def sidecar_document(
    *,
    variant: str,
    weights: Path,
    weights_sha256: str,
    onnx_path: Path,
    onnx_sha256: str,
    graph: dict[str, Any],
    load_report: dict[str, Any],
    versions: dict[str, str],
) -> dict[str, Any]:
    """The provenance record that ships next to the .onnx file."""
    return {
        "artefact": "osnet-onnx-export",
        "schema_version": 1,
        "exported_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variant": variant,
        "weights": {
            "path": str(weights),
            "filename": weights.name,
            "sha256": weights_sha256,
            "bytes": weights.stat().st_size,
            **load_report,
        },
        "onnx": {
            "path": str(onnx_path),
            "filename": onnx_path.name,
            "sha256": onnx_sha256,
            "bytes": onnx_path.stat().st_size,
            **graph,
        },
        "torchreid": {
            "source": TORCHREID_GIT_URL,
            "git_rev": TORCHREID_GIT_REV,
            "licence": "MIT",
            **versions,
        },
        "source_rule": SOURCE_RULE,
        # Never `yes` from a script. The provenance is written by a person against the
        # sha256 above, and ModelProvenance.assert_shippable() refuses anything that is
        # not `yes`.
        "shippable": "counsel",
        "model_card": (
            "final_models/reid/README.md (record the provenance before first use, POLICY.md, rule 8)"
        ),
    }


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_osnet_onnx.py",
        description=(
            "Export a torchreid OSNet checkpoint to ONNX. Never downloads: --weights is a "
            "required local path. Banned re-ID lineages are refused by name (exit 6)."
        ),
        epilog="Weight-provenance rules recorded 2026-08-07: docs/LICENCE_ALLOWLIST.md.",
    )
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="torchreid model name")
    parser.add_argument(
        "--weights",
        required=True,
        help="local path to the .pth checkpoint (REQUIRED; a URL is refused)",
    )
    parser.add_argument(
        "--out", default=None, help="output .onnx path (default: outputs/reid/<variant>.onnx)"
    )
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    return parser


def run(args: argparse.Namespace) -> int:
    check_lineage("--weights", args.weights)
    check_lineage("--variant", args.variant)
    out = Path(args.out) if args.out else REPO_ROOT / "outputs" / "reid" / f"{args.variant}.onnx"
    check_lineage("--out", str(out))

    weights = resolve_weights(args.weights)
    # Again on the resolved path: a clean-looking symlink into a luperson/ tree is exactly
    # the case a name check on the argument alone would wave through.
    check_lineage("resolved --weights", str(weights.resolve()))
    weights_sha = sha256_file(weights)

    model = build_model(args.variant, args.num_classes)
    load_report = load_weights(model, weights)
    graph = export_onnx(model, out, args.opset)

    import torch

    torchreid = import_torchreid()  # already imported by build_model; this is the cache
    versions = {"version": str(torchreid.__version__), "torch": str(torch.__version__)}
    document = sidecar_document(
        variant=args.variant,
        weights=weights,
        weights_sha256=weights_sha,
        onnx_path=out,
        onnx_sha256=sha256_file(out),
        graph=graph,
        load_report=load_report,
        versions=versions,
    )
    sidecar = out.with_name(out.name + ".json")
    sidecar.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"onnx written: {out}")
    print(f"sidecar written: {sidecar}")
    print(f"variant: {args.variant}  embedding_dim: {graph['embedding_dim']}")
    print(f"weights sha256: {weights_sha}")
    print(f"onnx    sha256: {document['onnx']['sha256']}")
    print(
        f"tensors loaded: {load_report['tensors_loaded']} of {load_report['tensors_in_checkpoint']}"
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except LicenceRefusal as exc:
        print(f"LICENCE REFUSAL: {exc}", file=sys.stderr)
        return EXIT_LICENCE
    except MissingTorchreid as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NO_TORCHREID
    except WeightsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_WEIGHTS
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_EXPORT


if __name__ == "__main__":
    raise SystemExit(main())
