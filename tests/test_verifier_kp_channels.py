"""KP_CHANNELS tests (fall push v3 arm F2, fall_kpgeo_v1).

The load-bearing claims: (a) the block is off by default and byte-identical to the
pre-change featurizer when not requested, and the golden hashes below were captured
from the featurizer before the KP block existed (2026-09-05, same synthetic
window); (b) each validity flag goes to 0, with its value forced to 0 and never to
NaN, when the defining joints are missing or below spec.kp_conf_min; (c) no NaN
ever reaches the feature matrix, kp-less trees included; (d) channel_names(spec)
matches the emitted matrix width for every block combination.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from vividly.verifier.features import (
    CHANNELS,
    KP_CHANNELS,
    FeatureError,
    WindowSpec,
    channel_names,
    featurize_window,
    kp_geom_samples,
    tracks_from_rows,
)

BASE = dict(
    window_s=8.0,
    stride_s=1.5,
    n_steps=64,
    policy="contain",
    min_present=0.35,
    contain_margin_s=0.5,
)

# sha256 of featurize_window(...).tobytes() on _golden_rows(), captured on the
# pre-change featurizer (before KP_CHANNELS existed). If one of these ever
# changes, an existing model's input geometry changed underneath it.
GOLDEN = {
    "default": "7f6e1fffaec6a145ac8f1979b7ba1058143edf009659e3e32a798d161ad62523",
    "rhythm": "7b69f78b0aacc3e5a019292e76caaba520eecc646644739991c46b55daebc3e7",
    "limb": "8cbbcb481d925138a1081b6ee7d5982bb985ed21a4a5b9fc122ead392b85d4c1",
}


def _golden_rows() -> list[dict]:
    """Deterministic rows (no RNG): drifting box + full COCO-17 kp field.

    Copied exactly from the pre-change golden capture; do not edit.
    """
    t = np.arange(0.0, 8.0, 1.0 / 25.0)
    rows = []
    for i, ti in enumerate(t):
        cx = 160.0 + 3.0 * float(np.sin(0.7 * ti))
        cy = 200.0 + 5.0 * float(np.sin(0.31 * ti))
        h = 100.0 + 4.0 * float(np.cos(0.5 * ti))
        kp = [
            [
                cx + ((j * 7) % 5 - 2.0) * 8.0,
                cy + ((j * 13) % 9 - 4.0) * 9.0,
                0.35 + 0.03 * (j % 5),
            ]
            for j in range(17)
        ]
        rows.append(
            {
                "f": i,
                "t": round(float(ti), 4),
                "track": 1,
                "cx": round(cx, 4),
                "cy": round(cy, 4),
                "w": round(h * 0.45, 4),
                "h": round(h, 4),
                "angle": 12.0 + float(np.sin(ti)),
                "kpc": 0.6,
                "kp": kp,
            }
        )
    return rows


def _body_rows(
    t: np.ndarray, *, upright: bool = True, conf: float = 0.8, h: float = 100.0
) -> list[dict]:
    """A stick figure standing (or lying) still; every joint at `conf`."""
    rows = []
    for i, ti in enumerate(t):
        cx, cy = 160.0, 200.0
        kp = [[cx, cy, conf] for _ in range(17)]
        if upright:
            for j in (0, 1, 2, 3, 4):
                kp[j] = [cx, cy - 0.45 * h, conf]  # head
            kp[5], kp[6] = [cx - 0.15 * h, cy - 0.3 * h, conf], [cx + 0.15 * h, cy - 0.3 * h, conf]
            kp[11], kp[12] = [cx - 0.1 * h, cy + 0.1 * h, conf], [cx + 0.1 * h, cy + 0.1 * h, conf]
            kp[9], kp[10] = [cx - 0.3 * h, cy, conf], [cx + 0.3 * h, cy, conf]
            kp[15], kp[16] = [cx - 0.1 * h, cy + 0.5 * h, conf], [cx + 0.1 * h, cy + 0.5 * h, conf]
        else:  # lying along +x: shoulders left, hips right, all at one height
            for j in (0, 1, 2, 3, 4):
                kp[j] = [cx - 0.45 * h, cy, conf]
            kp[5], kp[6] = [cx - 0.3 * h, cy - 0.05 * h, conf], [cx - 0.3 * h, cy + 0.05 * h, conf]
            kp[11], kp[12] = (
                [cx + 0.1 * h, cy - 0.05 * h, conf],
                [cx + 0.1 * h, cy + 0.05 * h, conf],
            )
            kp[9], kp[10] = [cx, cy - 0.1 * h, conf], [cx, cy + 0.1 * h, conf]
            kp[15], kp[16] = (
                [cx + 0.5 * h, cy - 0.05 * h, conf],
                [cx + 0.5 * h, cy + 0.05 * h, conf],
            )
        rows.append(
            {
                "f": i,
                "t": round(float(ti), 4),
                "track": 1,
                "cx": cx,
                "cy": cy,
                "w": h * 0.5,
                "h": h,
                "angle": 10.0,
                "kpc": conf,
                "kp": kp,
            }
        )
    return rows


def _col(x: np.ndarray, name: str) -> np.ndarray:
    return x[:, len(CHANNELS) + KP_CHANNELS.index(name)]


# ------------------------------------------------------- (a) default-off byte-identity
def test_kp_channels_absent_is_byte_identical_to_pre_change_output() -> None:
    tr = tracks_from_rows(_golden_rows())[0]
    for name, spec in (
        ("default", WindowSpec(**BASE)),
        ("rhythm", WindowSpec(**BASE, rhythm_channels=True)),
        ("limb", WindowSpec(**BASE, limb_channels=True)),
    ):
        x = featurize_window(tr, 8.0, spec)
        assert x is not None
        assert hashlib.sha256(x.tobytes()).hexdigest() == GOLDEN[name], name


def test_kp_channels_on_leaves_base_columns_untouched_and_appends_last() -> None:
    tr = tracks_from_rows(_golden_rows())[0]
    x_off = featurize_window(tr, 8.0, WindowSpec(**BASE))
    x_on = featurize_window(tr, 8.0, WindowSpec(**BASE, kp_channels=True))
    assert x_off is not None and x_on is not None
    assert x_off.shape == (64, len(CHANNELS))
    assert x_on.shape == (64, len(CHANNELS) + len(KP_CHANNELS))
    np.testing.assert_array_equal(x_on[:, : len(CHANNELS)], x_off)


# ------------------------------------------------------- (b) validity gating
def test_low_confidence_joints_zero_the_flags_and_the_values() -> None:
    t = np.arange(0.0, 8.0, 1.0 / 25.0)
    clear = tracks_from_rows(_body_rows(t, conf=0.8))[0]
    unsure = tracks_from_rows(_body_rows(t, conf=0.29))[0]  # just under the 0.3 gate
    spec = WindowSpec(**BASE, kp_channels=True)
    x_clear = featurize_window(clear, 8.0, spec)
    x_unsure = featurize_window(unsure, 8.0, spec)
    assert x_clear is not None and x_unsure is not None
    for flag in (n for n in KP_CHANNELS if n.endswith("_valid")):
        assert _col(x_clear, flag).max() == 1.0, flag
    assert np.all(x_unsure[:, len(CHANNELS) :] == 0.0)  # every value and flag


def test_gate_boundary_equal_confidence_passes() -> None:
    # posture_angle_deg contract: strictly below fails, equal passes.
    vals = kp_geom_samples(
        np.array([0.0, 0.5, 1.0]),
        tracks_from_rows(_body_rows(np.array([0.0, 0.5, 1.0]), conf=0.3))[0].kpts,
        100.0,
        0.3,
    )
    assert vals[:, KP_CHANNELS.index("kp_torso_valid")].min() == 1.0


def test_per_channel_validity_only_the_starved_channel_dies() -> None:
    t = np.arange(0.0, 8.0, 1.0 / 25.0)
    rows = _body_rows(t, conf=0.8)
    for r in rows:  # ankles unresolved everywhere; the rest of the body is fine
        r["kp"][15][2] = 0.05
        r["kp"][16][2] = 0.05
    tr = tracks_from_rows(rows)[0]
    x = featurize_window(tr, 8.0, WindowSpec(**BASE, kp_channels=True))
    assert x is not None
    for name in (
        "kp_ankle_dx",
        "kp_ankle_dy",
        "kp_ankle_valid",
        "kp_extent_shankle",
        "kp_extent_shankle_valid",
    ):
        assert np.all(_col(x, name) == 0.0), name
    assert _col(x, "kp_torso_valid").max() == 1.0
    assert _col(x, "kp_wrist_valid").max() == 1.0
    assert _col(x, "kp_extent_headhip_valid").max() == 1.0


def test_spec_kp_conf_min_reaches_the_matrix() -> None:
    t = np.arange(0.0, 8.0, 1.0 / 25.0)
    tr = tracks_from_rows(_body_rows(t, conf=0.4))[0]
    loose = featurize_window(tr, 8.0, WindowSpec(**BASE, kp_channels=True, kp_conf_min=0.2))
    strict = featurize_window(tr, 8.0, WindowSpec(**BASE, kp_channels=True, kp_conf_min=0.5))
    assert loose is not None and strict is not None
    assert _col(loose, "kp_torso_valid").max() == 1.0
    assert np.all(strict[:, len(CHANNELS) :] == 0.0)


def test_bad_kp_conf_min_refused() -> None:
    with pytest.raises(FeatureError, match="kp_conf_min"):
        WindowSpec(kp_conf_min=1.5)


# ------------------------------------------------------- (c) no NaN, ever
def test_no_nan_for_kpless_tree_all_nan_rows_and_gaps() -> None:
    spec = WindowSpec(**BASE, kp_channels=True)
    t = np.arange(0.0, 8.0, 1.0 / 25.0)

    # A pre-v2kp tree (rows carry no kp at all): loads, kp block all-zero.
    bare = [{k: v for k, v in r.items() if k != "kp"} for r in _body_rows(t)]
    tr = tracks_from_rows(bare)[0]
    assert tr.kpts is None
    x = featurize_window(tr, 8.0, spec)
    assert x is not None and np.isfinite(x).all()
    assert np.all(x[:, len(CHANNELS) :] == 0.0)

    # kp present but the estimator found nothing (all-nan rows).
    rows = _body_rows(t)
    for r in rows:
        r["kp"] = [[float("nan")] * 3 for _ in range(17)]
    tr = tracks_from_rows(rows)[0]
    x = featurize_window(tr, 8.0, spec)
    assert x is not None and np.isfinite(x).all()
    assert np.all(x[:, len(CHANNELS) :] == 0.0)

    # A mid-window tracker gap: absent steps carry 0 in values and flags.
    keep = (t < 3.0) | (t > 5.5)
    tr = tracks_from_rows(_body_rows(t[keep]))[0]
    x = featurize_window(tr, 8.0, spec)
    assert x is not None and np.isfinite(x).all()
    present = x[:, CHANNELS.index("present")]
    gap = present == 0.0
    assert gap.any()
    assert np.all(x[gap][:, len(CHANNELS) :] == 0.0)


def test_kp_geom_samples_never_nan_on_degenerate_inputs() -> None:
    t = np.array([0.0, 0.1])
    # coincident shoulders and hips: torso vector has no direction
    rows = _body_rows(t)
    for r in rows:
        for j in (5, 6, 11, 12):
            r["kp"][j] = [160.0, 200.0, 0.9]
    v = kp_geom_samples(t, tracks_from_rows(rows)[0].kpts, 100.0, 0.3)
    assert np.isfinite(v).all()
    assert v[:, KP_CHANNELS.index("kp_torso_valid")].max() == 0.0
    assert v[:, KP_CHANNELS.index("kp_torso_cos")].max() == 0.0
    # no scale reference at all
    assert np.all(kp_geom_samples(t, None, 100.0, 0.3) == 0.0)
    assert np.all(kp_geom_samples(t, tracks_from_rows(_body_rows(t))[0].kpts, 0.0, 0.3) == 0.0)


# ------------------------------------------------------- (d) channel_names contract
def test_channel_names_matches_matrix_width_for_every_block_combo() -> None:
    t = np.arange(0.0, 8.0, 1.0 / 25.0)
    tr = tracks_from_rows(_body_rows(t))[0]
    for rhythm in (False, True):
        for limb in (False, True):
            for kp in (False, True):
                spec = WindowSpec(
                    **BASE, rhythm_channels=rhythm, limb_channels=limb, kp_channels=kp
                )
                x = featurize_window(tr, 8.0, spec)
                assert x is not None
                names = channel_names(spec)
                assert x.shape[1] == len(names), (rhythm, limb, kp)
                assert len(set(names)) == len(names)
                if kp:
                    assert names[-len(KP_CHANNELS) :] == KP_CHANNELS


# ------------------------------------------------------- physical sanity
def test_upright_and_lying_geometry_read_differently() -> None:
    t = np.arange(0.0, 8.0, 1.0 / 25.0)
    spec = WindowSpec(**BASE, kp_channels=True)
    up = featurize_window(tracks_from_rows(_body_rows(t, upright=True))[0], 8.0, spec)
    down = featurize_window(tracks_from_rows(_body_rows(t, upright=False))[0], 8.0, spec)
    assert up is not None and down is not None
    # torso axis: vertical when standing (cos ~ 1), horizontal when lying (|sin| ~ 1)
    assert _col(up, "kp_torso_cos").mean() > 0.9
    assert abs(_col(up, "kp_torso_sin").mean()) < 0.2
    assert abs(_col(down, "kp_torso_sin").mean()) > 0.9
    # spine extents compress when lying
    assert _col(up, "kp_extent_headhip").mean() > 2.0 * _col(down, "kp_extent_headhip").mean()
    assert _col(up, "kp_extent_shankle").mean() > 2.0 * _col(down, "kp_extent_shankle").mean()


def test_head_vertical_velocity_measures_a_drop_in_torso_lengths() -> None:
    t = np.arange(0.0, 8.0, 1.0 / 25.0)
    rows = _body_rows(t, upright=True)
    # head drops 80 px over 4..5 s; torso length is |(-0.3h) - (+0.1h)| = 40 px
    for r in rows:
        ti = r["t"]
        dy = 80.0 * min(max(ti - 4.0, 0.0), 1.0)
        for j in (0, 1, 2, 3, 4):
            r["kp"][j][1] += dy
    tr = tracks_from_rows(rows)[0]
    x = featurize_window(tr, 8.0, WindowSpec(**BASE, kp_channels=True))
    assert x is not None
    vy = _col(x, "kp_head_vy")
    valid = _col(x, "kp_head_vy_valid")
    assert valid.max() == 1.0
    # 80 px/s over a 40 px torso = 2 torso-lengths/s, positive (image y grows down)
    assert vy.max() == pytest.approx(2.0, rel=0.15)
    assert abs(vy[: len(vy) // 4]).max() < 0.2
