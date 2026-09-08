"""The checkpoint provenance registry (POLICY.md, rule 8).

All of these run in the macOS development environment; no torch needed.
"""

from pathlib import Path

import pytest

from vividly.models.checkpoints import (
    is_lfs_pointer,
    load_registry,
    resolve_for_export,
    resolve_for_training,
)

REG = load_registry()


def test_is_lfs_pointer_recognises_a_pointer_and_nothing_else(tmp_path: Path) -> None:
    """The five graphs under final_models/ are Git LFS objects. Before `git lfs pull` a
    checkout holds a three-line text pointer in their place, and every seam that opens a
    graph asks this question first so its error names the fix."""
    pointer = tmp_path / "pointer.onnx"
    pointer.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 42\n"
    )
    blob = tmp_path / "blob.onnx"
    blob.write_bytes(b"\x08\x07\x12\x00not a pointer" + b"\x00" * 2048)
    small = tmp_path / "small.onnx"
    small.write_bytes(b"version 1 of something else")
    assert is_lfs_pointer(pointer) is True
    assert is_lfs_pointer(blob) is False
    assert is_lfs_pointer(small) is False
    assert is_lfs_pointer(tmp_path / "absent.onnx") is False


def test_every_entry_has_a_sha256_and_provenance():
    for c in REG.values():
        assert len(c.sha256) == 64
        assert c.training_data, f"{c.id} has no training_data"
        assert c.backbone_pretrain_data, f"{c.id} has no backbone pretrain lineage"


def test_evalref_cannot_be_a_finetune_base():
    with pytest.raises(ValueError, match="eval reference"):
        resolve_for_training("rtmpose_m_aic_coco_EVALREF")


def test_evalref_cannot_be_exported():
    with pytest.raises(ValueError, match="EVAL_REFERENCE"):
        resolve_for_export("rtmpose_m_aic_coco_EVALREF")


def test_rtmpose_filename_tokens_are_not_hashes():
    # Regression lock on the finding recorded at configs/checkpoints/registry.yaml lines 11
    # and 12: for RTMPose files the 8-hex token inside source_filename is not the sha256
    # prefix. If someone "fixes" the registry by assuming it is, this fails.
    for cid in ("rtmpose_m_coco_only", "rtmpose_m_aic_coco_EVALREF"):
        c = REG[cid]
        assert c.filename_token and not c.sha256.startswith(c.filename_token)


def test_cite_carries_a_hash():
    assert "929f00f2" in REG["rtmpose_m_coco_only"].cite()
