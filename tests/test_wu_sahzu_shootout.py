"""The clinical epilepsy-monitoring shootout: the patient split is disjoint and the
geometry is shared.

The convulsion-verifier arms are compared on one held-out patient set: the v1 variants
(synthetic / synthetic+real / real), then every later arm listed below. These tests pin
the properties the comparison's honesty rests on: train and test patients never overlap
anywhere, the real training configs use exactly the registry-granted train patients, and
every variant trains and tests on the same window geometry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vividly.data import registry as R

REPO = Path(__file__).resolve().parents[1]
EVENTS = REPO / "configs" / "events"

TRAIN_CONFIGS = [
    "train_convulsion_v1_real.yaml",
    "train_convulsion_v1_synthreal.yaml",
    "train_convulsion_v2_real.yaml",
    "train_convulsion_v3_real.yaml",
    "train_convulsion_v3_h128.yaml",
    "train_convulsion_v3_d4.yaml",
    "train_convulsion_v3_h128e90.yaml",
    "train_convulsion_v2pose.yaml",
    "train_convulsion_v6_real.yaml",
]
EVAL_CONFIGS = [
    "eval_wusahzu_testsplit_synthetic.yaml",
    "eval_wusahzu_testsplit_synthreal.yaml",
    "eval_wusahzu_testsplit_real.yaml",
    "eval_wusahzu_testsplit_v2real.yaml",
    "eval_wusahzu_testsplit_v3real.yaml",
    "eval_wusahzu_testsplit_v2pose.yaml",
    "eval_wusahzu_testsplit_v4real.yaml",
    "eval_wusahzu_testsplit_v6real.yaml",
]

# The geometry the whole comparison shares; scene_channels is a feature-set choice
# (part of what a v2 model is), not geometry, and may differ between v1 and v2,
# but each eval must match its model's training feature set.
GEOMETRY_KEYS = ("window_s", "stride_s", "n_steps", "policy", "overlap_min_s", "min_present")


def _cfg(name: str) -> dict:
    return yaml.safe_load((EVENTS / name).read_text(encoding="utf-8"))


def _wu_locations(cfg: dict) -> set[str]:
    out: set[str] = set()
    for source in cfg["sources"]:
        if "wu_sahzu" in source["dir"]:
            out.update(source.get("locations") or ())
    return out


def trainsplit_patients() -> set[str]:
    ds = R.default_registry()["wu_sahzu_emu_trainsplit"]
    return {f.name.split("/")[0] for f in ds.files}


def test_trainsplit_entry_is_a_train_interim_prototype_grant() -> None:
    ds = R.default_registry()["wu_sahzu_emu_trainsplit"]
    assert ds.bucket is R.Bucket.TRAIN_PROTOTYPE
    assert ds.licensing.event_grant is not None
    assert ds.licensing.event_grant.role == "train_interim"
    assert ds.provenance.real_medical_events
    # same upstream release as the holdout entry, sliced by patient (not derived_from
    # in the registry's contamination sense; the split is the contamination control)
    holdout = R.default_registry()["wu_sahzu_emu"]
    assert ds.licensing.license_url == holdout.licensing.license_url


def test_full_holdout_entry_still_never_trains() -> None:
    ds = R.default_registry()["wu_sahzu_emu"]
    assert ds.bucket is R.Bucket.EVAL_ONLY
    assert ds.licensing.event_grant is not None
    assert ds.licensing.event_grant.role == "test_only"


@pytest.mark.parametrize("train_name", TRAIN_CONFIGS)
@pytest.mark.parametrize("eval_name", EVAL_CONFIGS)
def test_train_and_test_patients_are_disjoint(train_name: str, eval_name: str) -> None:
    train_locs = _wu_locations(_cfg(train_name))
    eval_locs = _wu_locations(_cfg(eval_name))
    assert train_locs and eval_locs
    assert not train_locs & eval_locs, f"patient leak: {sorted(train_locs & eval_locs)}"


@pytest.mark.parametrize("train_name", TRAIN_CONFIGS)
def test_real_training_uses_exactly_the_granted_patients(train_name: str) -> None:
    assert _wu_locations(_cfg(train_name)) == trainsplit_patients()


def test_all_variants_share_one_window_geometry() -> None:
    reference = _cfg("train_convulsion_verifier_v1.yaml")["window"]
    for name in TRAIN_CONFIGS + EVAL_CONFIGS:
        window = _cfg(name)["window"]
        for key in GEOMETRY_KEYS:
            assert window.get(key) == reference.get(key), (
                f"{name} diverges from the shared window on {key}"
            )


def test_v2_eval_matches_v2_training_feature_set() -> None:
    for name in (
        "train_convulsion_v2_real.yaml",
        "eval_wusahzu_testsplit_v2real.yaml",
        "train_convulsion_v3_real.yaml",
        "eval_wusahzu_testsplit_v3real.yaml",
    ):
        assert _cfg(name)["window"]["scene_channels"] is True
    # and the v1 evals stay 8-channel (flag absent or false)
    for name in ("eval_wusahzu_testsplit_real.yaml", "train_convulsion_v1_real.yaml"):
        assert not _cfg(name)["window"].get("scene_channels")


def test_test_split_names_no_train_patient_anywhere() -> None:
    granted = trainsplit_patients()
    for name in EVAL_CONFIGS:
        assert not _wu_locations(_cfg(name)) & granted
