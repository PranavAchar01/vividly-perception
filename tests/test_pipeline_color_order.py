"""Coupling gate: `pose.color_order` may only be set by an extraction config.

The red-green-blue colour fix reaches the live path only through a full track-tree
re-extraction under configs/events/extract_*, never as a backend change. Every deployed
verifier was trained on features extracted in the current blue-green-red order, so the live
and demo paths must keep feeding that order until those verifiers are retrained on trees
extracted in the corrected one. run_pipeline.py defaults the key to "bgr" and pops it
runner-side; these tests pin the places that must never override that default. The rule and
the measured cost of the fix are in TRAINING.md, joints lane, "Colour order".
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def _files(*globs: str) -> list[Path]:
    out: list[Path] = []
    for g in globs:
        out.extend(sorted(REPO.glob(g)))
    return out


def test_live_pipeline_configs_never_set_color_order() -> None:
    """The demo/live path reads configs/pipeline/, and none of it may set the key."""
    offenders = [
        p
        for p in _files("configs/pipeline/**/*.yaml", "configs/pipeline/**/*.yml")
        if "color_order" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"live-path configs set pose.color_order: {offenders}"


def test_demo_code_never_sets_color_order() -> None:
    """DemoPipeline builds its own run config in code, and that code may not set the key."""
    offenders = [
        p
        for p in _files("src/vividly/demo/**/*.py")
        if "color_order" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"demo code sets pose.color_order: {offenders}"


def test_v2c_extract_configs_set_rgb() -> None:
    """Every v2-coupled extraction config must carry the rider explicitly."""
    v2c = _files("configs/events/extract_*_v2c.yaml")
    if not v2c:
        pytest.skip("no v2c extraction configs authored yet")
    for p in v2c:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert cfg["pipeline"]["pose"].get("color_order") == "rgb", (
            f"{p} is a v2c extraction config but does not set pose.color_order: rgb"
        )
