"""Checkpoint provenance registry.

Mirrors the posture of the dataset registry: a frozen dataclass loaded from YAML, with two
role rules enforced at the resolvers. A `load_from` path is refused for any checkpoint that
does not carry FINETUNE_BASE, and an export path is refused for a checkpoint that carries
EVAL_REFERENCE without SHIPPABLE_BASE. The rule behind both is POLICY.md, rule 8, checkpoint
provenance is recorded rather than assumed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

Role = Literal["SHIPPABLE_BASE", "FINETUNE_BASE", "EVAL_REFERENCE"]

REGISTRY_PATH = Path("configs/checkpoints/registry.yaml")
_CHUNK = 1 << 20


@dataclass(frozen=True)
class CounselFlag:
    id: str
    status: str
    blocks: str = ""


@dataclass(frozen=True)
class Checkpoint:
    id: str
    local_path: Path
    source_url: str
    source_filename: str
    sha256: str
    size_bytes: int
    arch: str
    training_data: tuple[str, ...]
    backbone_pretrain_data: tuple[str, ...]
    roles: tuple[Role, ...]
    counsel_flags: tuple[CounselFlag, ...] = field(default_factory=tuple)
    reported_metric: dict | None = None
    filename_token: str | None = None

    def cite(self) -> str:
        """The only supported way to name this checkpoint in a metric table or a slide.

        Average precision does not identify a checkpoint, since RTMPose-m AIC+COCO and
        RTMPose-l COCO-only both read 0.758, so every accuracy number must travel with this
        string.
        """
        return f"{self.id}@{self.sha256[:12]} ({self.source_filename})"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Checkpoint]:
    raw = yaml.safe_load(path.read_text())
    out: dict[str, Checkpoint] = {}
    for e in raw["checkpoints"]:
        out[e["id"]] = Checkpoint(
            id=e["id"],
            local_path=Path(e["local_path"]),
            source_url=e["source_url"],
            source_filename=e["source_filename"],
            sha256=e["sha256"],
            size_bytes=e["size_bytes"],
            arch=e["arch"],
            training_data=tuple(e["training_data"]),
            backbone_pretrain_data=tuple(e["backbone_pretrain"]["data"]),
            roles=tuple(e["roles"]),
            counsel_flags=tuple(
                CounselFlag(id=f["id"], status=f.get("status", "OPEN"), blocks=f.get("blocks", ""))
                for f in e.get("counsel_flags", [])
            ),
            reported_metric=e.get("reported_metric"),
            filename_token=e.get("filename_token"),
        )
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


#: The first line of every Git LFS pointer file, as the LFS specification fixes it.
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def is_lfs_pointer(path: Path) -> bool:
    """True when `path` holds a Git LFS pointer rather than the object it stands for.

    A checkout made without `git lfs pull` carries a three-line text pointer (version,
    oid, size) in place of each graph under final_models/. Hashing such a file can never
    match the recorded digest, so every seam that opens a graph checks this first and
    names the fix, `git lfs pull`, instead of reporting a bare digest mismatch.
    """
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as fh:
            return fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def verify(ckpt: Checkpoint) -> None:
    """Raise unless the blob on disk is byte-identical to what the registry claims."""
    if not ckpt.local_path.is_file():
        raise FileNotFoundError(
            f"{ckpt.id}: {ckpt.local_path} missing; re-fetch it from the source_url "
            f"recorded in configs/checkpoints/registry.yaml"
        )
    size = ckpt.local_path.stat().st_size
    if size != ckpt.size_bytes:
        raise ValueError(f"{ckpt.id}: size {size} != registry {ckpt.size_bytes}")
    got = sha256_of(ckpt.local_path)
    if got != ckpt.sha256:
        raise ValueError(f"{ckpt.id}: sha256 {got} != registry {ckpt.sha256}")


def resolve_for_training(name: str, registry: dict[str, Checkpoint] | None = None) -> Path:
    """The only supported way to obtain a `load_from` path.

    Refuses EVAL_REFERENCE checkpoints. This is the checkpoint-side twin of
    build_train_manifest()'s EVAL_ONLY refusal in the dataset registry.
    """
    registry = registry or load_registry()
    ckpt = registry[name]
    if "FINETUNE_BASE" not in ckpt.roles:
        raise ValueError(
            f"{name} has roles {ckpt.roles}; it is not a FINETUNE_BASE and cannot appear in a "
            "`load_from`. A checkpoint carrying EVAL_REFERENCE alone is an eval reference "
            "and never a training source."
        )
    verify(ckpt)
    return ckpt.local_path


def resolve_for_export(name: str, registry: dict[str, Checkpoint] | None = None) -> Path:
    """Same refusal for the ONNX export path: an EVAL_REFERENCE must never become a shipped
    artefact."""
    registry = registry or load_registry()
    ckpt = registry[name]
    if "EVAL_REFERENCE" in ckpt.roles and "SHIPPABLE_BASE" not in ckpt.roles:
        raise ValueError(
            f"{name} is EVAL_REFERENCE; refusing to export it as a deployable artefact"
        )
    verify(ckpt)
    return ckpt.local_path
