"""Track-space augmentation and push-2 hard-negative weights.

The camera-geometry family (fall push v3 arm F3, bottom half of this file) adds its
own claims: the transform is the exact identity at strength 0 / s_y = 1, the posture
angle is recomputed from the warped keypoints and provably equals the extractor's own
`vividly.tracks.posture_angle_deg`, foreshortening and roll move it the way geometry
says they must, no augmented row is NaN or out of the featurizer's range, and with no
`geo_augment:` key nothing changes at all.

The load-bearing claims, in the order the seizure push-2 protocol demands them
(research/seizure-push-2-protocol.md items 2-3). Augmentation is not synthesis: the
rule that admits it, and its train-rows-only scope, is in POLICY.md.

  * Off by default is not merely equivalent, it is the same bytes: with no `augment:` and no
    `negative_class_weights:` key nothing in the featurization or the training data
    changes, and `make_augmenter` hands back None so the caller's array is untouched.
  * Identity round-trip: zero draw probabilities, and a time-warp factor of exactly
    1.0, are the exact identity, not merely close.
  * Label invariance: the augmenter never sees a label, and row order and row count
    survive every draw, so the caller's y stays aligned with its X.
  * No NaN, ever, and no channel leaves the featurizer's declared value range.
  * The `present >= 0.5` floor the protocol fixed is never crossed, including on
    windows that already start below it (min_present is 0.35).
  * The negative classes are the OOF diagnosis's classes, and the AJILE12 diet
    (force_label) is never folded into `seizure_free`.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vividly.tracks.state import posture_angle_deg
from vividly.verifier.augment import (
    GEO_ROLL_MAX_DEG,
    GEO_SY_MIN,
    GEO_TRAPEZOID_MAX,
    PRESENT_FLOOR,
    WARP_MAX,
    WARP_MIN,
    Augmenter,
    AugmentError,
    AugmentSpec,
    GeoAugmenter,
    GeoDraw,
    GeoSpec,
    make_augmenter,
    make_geo_augmenter,
    mirror_vx,
    observation_dropout,
    time_warp,
    torso_angles_deg,
    warp_track_window,
)
from vividly.verifier.features import (
    CHANNELS,
    SCENE_CHANNELS,
    TrackArrays,
    WindowSource,
    WindowSpec,
    channel_names,
    featurize_window,
    windows_for_video,
)

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "scripts" / "events" / "train_event_verifier.py"

# The arm-B window geometry, copied exactly from configs/events/rot_train_conv_rotB_*.yaml.
ARM_B = WindowSpec(
    window_s=12.0,
    stride_s=3.0,
    n_steps=64,
    policy="overlap",
    overlap_min_s=4.0,
    min_present=0.35,
    scene_channels=True,
)
ARM_B_CHANNELS = channel_names(ARM_B)


def _load_trainer():
    spec = importlib.util.spec_from_file_location("vividly_train_event_verifier", TRAINER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture_track(seed: int = 0, n: int = 130) -> TrackArrays:
    """A deterministic 13 s track with a gap, so `present` is genuinely mixed."""
    rng = np.random.default_rng(seed)
    t = np.round(np.arange(n) * 0.1, 4)
    keep = ~((t > 4.0) & (t < 5.5))  # a tracker gap
    t = t[keep]
    cx = 160.0 + 12.0 * np.sin(t * 2.4) + rng.normal(0, 0.4, len(t))
    cy = 240.0 + 9.0 * np.cos(t * 3.1) + rng.normal(0, 0.4, len(t))
    h = 100.0 + rng.normal(0, 1.0, len(t))
    return TrackArrays(
        track_id=1,
        t=t,
        cx=cx,
        cy=cy,
        w=h * 0.55,
        h=h,
        angle=np.where(np.arange(len(t)) % 7 == 0, np.nan, 22.0),
        kpc=np.full(len(t), 0.72),
    )


def _fixture_window(seed: int = 0) -> np.ndarray:
    """One featurized (64, 8) window with no scene columns, so no SceneContext is needed."""
    spec = WindowSpec(window_s=12.0, stride_s=3.0, n_steps=64, policy="overlap", overlap_min_s=4.0)
    x = featurize_window(_fixture_track(seed), 12.9, spec)
    assert x is not None
    return x


def _cols(names: tuple[str, ...] = CHANNELS) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}


# ------------------------------------------------------- default off / byte identity
def test_make_augmenter_is_off_by_default() -> None:
    for cfg in (None, {}, False, {"enabled": False}):
        assert make_augmenter(cfg, ARM_B_CHANNELS) is None


def test_featurization_is_byte_identical_with_augment_module_present() -> None:
    """The golden hash of the featurizer's output on a fixed fixture.

    The augmentation levers add nothing to vividly.verifier.features, and the golden
    digest below is what holds that claim. The real proof at training scale is in
    research/seizure-push-2-arms.md: rerunning configs/events/rot_train_conv_rotB_f2_s1.yaml
    under the modified trainer reproduces onnx sha256 b9e04d03... bit for bit.
    """
    x = _fixture_window(0)
    assert x.shape == (64, len(CHANNELS))
    assert x.dtype == np.float32
    digest = hashlib.sha256(x.tobytes()).hexdigest()
    assert digest == "14dd1831e53ffb15dbd356da3e8581e32fc054f8e1f32236ea9e38d2330d1bae"


def test_augmenter_never_mutates_its_input() -> None:
    X = np.stack([_fixture_window(s) for s in range(4)])
    before = X.copy()
    aug = make_augmenter({"seed": 7}, CHANNELS)
    assert aug is not None
    aug.augment_epoch(X, 3)
    assert np.array_equal(X, before)


# ------------------------------------------------------------- identity round-trip
def test_zero_probabilities_are_the_exact_identity() -> None:
    X = np.stack([_fixture_window(s) for s in range(5)])
    aug = make_augmenter({"mirror_p": 0.0, "warp_p": 0.0, "dropout_p": 0.0, "seed": 1}, CHANNELS)
    assert aug is not None
    out = aug.augment_epoch(X, 11)
    assert out.dtype == X.dtype
    assert np.array_equal(out, X)  # byte-identical, not allclose


def test_time_warp_factor_one_is_the_exact_identity() -> None:
    x = _fixture_window(2)
    vel = ((0, 10.0), (1, 10.0))
    assert np.array_equal(time_warp(x, 1.0, vel), x)


def test_mirror_is_involutive_and_touches_vx_only() -> None:
    x = _fixture_window(3)
    c = _cols()
    once = mirror_vx(x, (c["vx"],))
    assert np.array_equal(mirror_vx(once, (c["vx"],)), x)
    assert np.array_equal(once[:, c["vx"]], -x[:, c["vx"]])
    others = [i for i in range(x.shape[1]) if i != c["vx"]]
    assert np.array_equal(once[:, others], x[:, others])


def test_same_seed_and_epoch_reproduce_and_epochs_differ() -> None:
    X = np.stack([_fixture_window(s) for s in range(6)])
    a = make_augmenter({"seed": 20260826}, CHANNELS)
    b = make_augmenter({"seed": 20260826}, CHANNELS)
    assert a is not None and b is not None
    assert np.array_equal(a.augment_epoch(X, 4), b.augment_epoch(X, 4))
    assert not np.array_equal(a.augment_epoch(X, 4), a.augment_epoch(X, 5))


# ------------------------------------------------------------------ label invariance
def test_row_order_and_count_are_preserved() -> None:
    """Row i in == row i out, so the caller's labels stay attached to their window.

    The marker lives in a scene column: scene features are video-level, not per-step
    observations, so no transform is allowed to touch them, which makes them an
    honest row fingerprint.
    """
    n_rows = 12
    X = np.zeros((n_rows, 64, len(CHANNELS) + len(SCENE_CHANNELS)), dtype=np.float32)
    for i in range(n_rows):
        X[i, :, : len(CHANNELS)] = _fixture_window(i)
        X[i, :, len(CHANNELS)] = np.float32(0.001 * (i + 1))  # scene_n_tracks marker
    aug = make_augmenter({"seed": 5}, ARM_B_CHANNELS)
    assert aug is not None
    out = aug.augment_epoch(X, 2)
    assert out.shape == X.shape
    for i in range(n_rows):
        assert np.allclose(out[i, :, len(CHANNELS)], np.float32(0.001 * (i + 1)))


def test_augmenter_api_takes_no_labels() -> None:
    import inspect

    for fn in (Augmenter.augment_epoch, Augmenter.augment_window):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"y", "label", "labels"})


# ------------------------------------------------------------------------- no NaN
def test_no_nan_and_channels_stay_in_range_over_a_battery() -> None:
    X = np.stack([_fixture_window(s) for s in range(16)])
    aug = make_augmenter({"mirror_p": 1.0, "warp_p": 1.0, "dropout_p": 1.0, "seed": 99}, CHANNELS)
    assert aug is not None
    c = _cols()
    for epoch in range(1, 31):
        out = aug.augment_epoch(X, epoch)
        assert np.isfinite(out).all()
        assert np.abs(out[:, :, c["vy"]]).max() <= 10.0
        assert np.abs(out[:, :, c["vx"]]).max() <= 10.0
        for flag in ("present", "angle_valid"):
            vals = np.unique(out[:, :, c[flag]])
            assert set(vals.tolist()) <= {0.0, 1.0}


def test_present_floor_is_never_crossed() -> None:
    X = np.stack([_fixture_window(s) for s in range(16)])
    c = _cols()
    start = X[:, :, c["present"]].mean(axis=1)
    aug = make_augmenter({"mirror_p": 0.0, "warp_p": 0.0, "dropout_p": 1.0, "seed": 3}, CHANNELS)
    assert aug is not None
    for epoch in range(1, 21):
        out = aug.augment_epoch(X, epoch)
        after = out[:, :, c["present"]].mean(axis=1)
        # never below the floor, and never below where the window already was
        assert (after >= np.minimum(start, PRESENT_FLOOR) - 1e-6).all()


def test_dropout_is_a_noop_when_present_already_below_the_floor() -> None:
    x = _fixture_window(1).copy()
    c = _cols()
    x[:, c["present"]] = 0.0
    x[:40, c["present"]] = 1.0  # 0.625 present
    x[:25, c["present"]] = 1.0
    x[25:, c["present"]] = 0.0  # 0.39 present: below the 0.5 floor already
    aug = make_augmenter({"mirror_p": 0.0, "warp_p": 0.0, "dropout_p": 1.0, "seed": 8}, CHANNELS)
    assert aug is not None
    out = aug.augment_epoch(x[None], 1)
    assert np.array_equal(out[0], x)


def test_dropout_writes_the_featurizers_own_absent_fill() -> None:
    x = _fixture_window(4)
    c = _cols()
    out = observation_dropout(
        x, 10, 8, c["present"], ((c["h_rel"], 1.0), (c["aspect"], 1.0), (c["vx"], 0.0))
    )
    assert (out[10:18, c["present"]] == 0.0).all()
    assert (out[10:18, c["h_rel"]] == 1.0).all()
    assert (out[10:18, c["aspect"]] == 1.0).all()
    assert (out[10:18, c["vx"]] == 0.0).all()
    assert np.array_equal(out[18:], x[18:])


# ------------------------------------------------------------------ protocol bounds
def test_warp_range_outside_the_protocol_is_refused() -> None:
    with pytest.raises(AugmentError):
        AugmentSpec(warp_min=0.5)
    with pytest.raises(AugmentError):
        AugmentSpec(warp_max=1.5)
    AugmentSpec(warp_min=WARP_MIN, warp_max=WARP_MAX)  # the declared bounds are legal


def test_unknown_augment_key_is_refused() -> None:
    with pytest.raises(AugmentError):
        make_augmenter({"shatter_p": 0.5}, CHANNELS)


def test_augmenter_needs_the_base_channels() -> None:
    with pytest.raises(AugmentError):
        make_augmenter({"seed": 1}, ("scene_n_tracks", "scene_births"))


def test_time_warp_rescales_velocity_and_keeps_flags_binary() -> None:
    x = _fixture_window(5)
    c = _cols()
    out = time_warp(x, 1.18, ((c["vy"], 10.0), (c["vx"], 10.0)))
    assert set(np.unique(out[:, c["present"]]).tolist()) <= {0.0, 1.0}
    assert set(np.unique(out[:, c["angle_valid"]]).tolist()) <= {0.0, 1.0}
    # a slowed window carries smaller per-second velocities
    assert np.abs(out[:, c["vy"]]).max() <= np.abs(x[:, c["vy"]]).max() + 1e-6


# ------------------------------------------------- lever 1: negative-class weights
def test_negative_weights_default_to_one_and_name_the_oof_classes() -> None:
    tr = _load_trainer()
    assert set(tr.NEG_CLASSES) == {
        "seizure_free",
        "owner_offevent",
        "nonowner_track",
        "forced_negative",
    }
    classes = np.array(
        ["positive", "seizure_free", "nonowner_track", "forced_negative", "seizure_free"]
    )
    w = tr.negative_weights(classes, {"seizure_free": 4.0})
    assert w.tolist() == [1.0, 4.0, 1.0, 1.0, 4.0]
    assert w.dtype == np.float32


def test_negative_weights_reject_typos_and_non_positive_alpha() -> None:
    tr = _load_trainer()
    classes = np.array(["seizure_free"])
    with pytest.raises(SystemExit):
        tr.negative_weights(classes, {"seizure-free": 2.0})
    with pytest.raises(SystemExit):
        tr.negative_weights(classes, {"seizure_free": 0.0})


def test_forced_negative_sources_are_never_called_seizure_free(tmp_path: Path) -> None:
    """The AJILE12 diet's metas carry empty events_t too; class (d) must stay (d)."""
    tr = _load_trainer()
    for name in ("evented", "diet"):
        (tmp_path / name / "meta").mkdir(parents=True)
        (tmp_path / name / "sequences").mkdir(parents=True)
    import json

    rows = []
    t = np.round(np.arange(160) * 0.1, 4)
    for i, ti in enumerate(t):
        rows.append(
            {
                "f": i,
                "t": float(ti),
                "track": 1,
                "cx": 160.0 + float(np.sin(ti)),
                "cy": 240.0 + float(np.cos(ti)),
                "w": 55.0,
                "h": 100.0,
                "angle": 20.0,
                "kpc": 0.7,
            }
        )
    body = "\n".join(json.dumps(r) for r in rows) + "\n"
    for name, meta in (
        ("evented", {"key": "k1", "location": "patX", "events_t": [[4.0, 9.0]]}),
        ("diet", {"key": "d1", "location": "subX", "events_t": []}),
    ):
        (tmp_path / name / "meta" / "m.json").write_text(json.dumps(meta), encoding="utf-8")
        (tmp_path / name / "sequences" / f"{meta['key']}.jsonl").write_text(body, encoding="utf-8")
    cfg = {
        "task": "convulsion",
        "window": {
            "window_s": 12.0,
            "stride_s": 3.0,
            "n_steps": 64,
            "policy": "overlap",
            "overlap_min_s": 4.0,
            "min_present": 0.35,
        },
        "sources": [
            {"dir": str(tmp_path / "evented")},
            {"dir": str(tmp_path / "diet"), "force_label": 0},
        ],
        "train": {},
        "out_dir": str(tmp_path / "out"),
    }
    _X, y, groups, neg, srcs = tr.build_dataset(cfg, classify_negatives=True)
    assert srcs is None  # sources are opt-in; classify_negatives does not turn them on
    assert neg is not None and len(neg) == len(y)
    assert "seizure_free" not in set(neg.tolist())  # nothing here is a free recording
    assert "forced_negative" in set(neg.tolist())
    # every diet window is forced_negative, and no diet window is labelled positive
    diet = np.array([g.endswith("::subX") for g in groups])
    assert (neg[diet] == "forced_negative").all()
    assert (y[diet] == 0).all()
    # ... and the default path still hands back None
    assert tr.build_dataset(cfg)[3] is None


# ============================================ camera geometry (fall push v3 arm F3)
# The fall push v3 window geometry, copied exactly from configs/events/train_fall_F0_*.yaml.
FALL = WindowSpec(
    window_s=8.0,
    stride_s=1.5,
    n_steps=64,
    policy="contain",
    min_present=0.35,
    contain_margin_s=0.5,
)
FALL_COLS = _cols(channel_names(FALL))


def _kp_body(cx: float, cy: float, tilt_deg: float, torso_px: float = 60.0, conf: float = 0.9):
    """COCO-17 keypoints for a body whose torso axis is `tilt_deg` off vertical.

    Shoulder-mid -> hip-mid is exactly the torso axis, so the posture angle of this
    body is `tilt_deg` by construction and any deviation is the transform's doing.
    """
    th = np.radians(tilt_deg)
    axis = np.array([np.sin(th), np.cos(th)])  # image y grows down: upright = +y
    perp = np.array([axis[1], -axis[0]])
    sh_mid = np.array([cx, cy]) - axis * torso_px / 2.0
    hip_mid = np.array([cx, cy]) + axis * torso_px / 2.0
    kp = np.zeros((17, 3), dtype=np.float64)
    kp[:, 2] = conf
    head = sh_mid - axis * 20.0
    for j in range(5):  # nose, eyes, ears
        kp[j, :2] = head + perp * (j - 2) * 2.0
    kp[5, :2], kp[6, :2] = sh_mid + perp * 14.0, sh_mid - perp * 14.0
    kp[7, :2], kp[8, :2] = sh_mid + perp * 18.0 + axis * 18.0, sh_mid - perp * 18.0 + axis * 18.0
    kp[9, :2], kp[10, :2] = sh_mid + perp * 20.0 + axis * 34.0, sh_mid - perp * 20.0 + axis * 34.0
    kp[11, :2], kp[12, :2] = hip_mid + perp * 10.0, hip_mid - perp * 10.0
    kp[13, :2], kp[14, :2] = hip_mid + perp * 9.0 + axis * 28.0, hip_mid - perp * 9.0 + axis * 28.0
    kp[15, :2], kp[16, :2] = hip_mid + perp * 8.0 + axis * 55.0, hip_mid - perp * 8.0 + axis * 55.0
    return kp


def _kp_track(tilts: list[float], *, stored_angle: float | None = 5.0) -> TrackArrays:
    """An 8 s track at 10 Hz whose torso tilt walks through `tilts` in equal phases.

    `stored_angle` is deliberately wrong (a constant that matches no phase): a warp
    that copied the recorded angle instead of recomputing it would hand that constant
    straight back, and the tests below would see it.
    """
    n = 80
    t = np.round(np.arange(n) * 0.1, 4)
    per = n // len(tilts)
    seq = [tilts[min(i // per, len(tilts) - 1)] for i in range(n)]
    kpts = np.stack([_kp_body(160.0, 240.0, a) for a in seq])
    x0, y0 = kpts[:, :, 0].min(axis=1), kpts[:, :, 1].min(axis=1)
    x1, y1 = kpts[:, :, 0].max(axis=1), kpts[:, :, 1].max(axis=1)
    return TrackArrays(
        track_id=1,
        t=t,
        cx=(x0 + x1) / 2.0,
        cy=(y0 + y1) / 2.0,
        w=x1 - x0,
        h=y1 - y0,
        angle=np.full(n, np.nan if stored_angle is None else stored_angle),
        kpc=np.full(n, 0.9),
        kpts=kpts,
    )


def _geo_fixture(tilts: list[float] | None = None) -> tuple[list[WindowSource], np.ndarray]:
    tr = _kp_track(tilts if tilts is not None else [0.0, 45.0, 90.0])
    t_end = float(tr.t[-1])
    x = featurize_window(tr, t_end, FALL)
    assert x is not None
    return [WindowSource(track=tr, t_end=t_end)], x[None]


# --------------------------------------------------------------- angle is recomputed
def test_geo_angle_recompute_equals_the_extractors_own_function() -> None:
    """`torso_angles_deg` is a vectorized copy of vividly.tracks.posture_angle_deg.

    Including the gate semantics: a torso joint below kp_conf_min kills the angle, and
    the extraction wrote exactly that as `angle: null`.
    """
    rng = np.random.default_rng(4)
    kpts = rng.uniform(-300, 300, size=(200, 17, 3))
    kpts[:, :, 2] = rng.uniform(0.0, 1.0, size=(200, 17))
    got = torso_angles_deg(kpts, 0.3)
    for i in range(len(kpts)):
        ref = posture_angle_deg([tuple(k) for k in kpts[i]], 0.3)
        if ref is None:
            assert np.isnan(got[i])
        else:
            assert got[i] == pytest.approx(ref, abs=1e-12)
    assert np.isnan(got).any() and np.isfinite(got).any()  # both branches exercised


def test_geo_angle_is_recomputed_from_the_warped_keypoints_never_copied() -> None:
    """Upright -> mid-fall -> fallen, foreshortened: the angle follows the geometry.

    Vertical foreshortening by s_y turns a torso (dx, dy) into (dx, s_y*dy), so the
    tilt goes atan(|dx|/|dy|) -> atan(|dx|/(s_y|dy|)). 0 and 90 degrees are its fixed
    points and everything between tips away from vertical, which is the whole reason
    the arm exists (a high camera makes a standing body look like a falling one) and
    the reason a copied angle would be a lie.
    """
    tr = _kp_track([0.0, 45.0, 90.0], stored_angle=5.0)
    assert np.allclose(tr.angle, 5.0)  # the recorded angle is deliberately wrong
    base = torso_angles_deg(tr.kpts, 0.3)
    assert np.allclose(np.unique(np.round(base, 6)), [0.0, 45.0, 90.0])

    sy = 0.55
    out = warp_track_window(tr, GeoDraw(sy=sy), 0.3)
    expected = np.degrees(np.arctan2(np.sin(np.radians(base)), sy * np.cos(np.radians(base))))
    # KP_ROUND: warped keypoints are written on the extraction's own 0.1 px grid, which
    # is worth at most atan(0.1/60) = 0.1 deg of angle, far below the pose estimator's
    # own noise, and the price of augmented rows being numerically indistinguishable
    # from recorded ones.
    assert np.allclose(out.angle, expected, atol=0.15)
    assert np.allclose(np.unique(np.round(out.angle, 1)), [0.0, 61.2, 90.0], atol=0.15)
    assert not np.allclose(out.angle, 5.0)  # nothing was copied


def test_geo_roll_tips_an_upright_body_by_the_roll_angle() -> None:
    tr = _kp_track([0.0], stored_angle=None)
    for roll in (5.0, 8.0, 12.0, -12.0):
        out = warp_track_window(tr, GeoDraw(roll_rad=float(np.radians(roll))), 0.3)
        assert np.allclose(out.angle, abs(roll), atol=0.15)  # 0.1 px kp grid, see above


def test_geo_foreshortening_never_reduces_apparent_tilt_and_mirror_never_changes_it() -> None:
    for tilt in (0.0, 10.0, 30.0, 60.0, 89.0, 90.0):
        tr = _kp_track([tilt], stored_angle=None)
        before = torso_angles_deg(tr.kpts, 0.3)
        for sy in (0.55, 0.7, 0.85):
            after = warp_track_window(tr, GeoDraw(sy=sy), 0.3).angle
            assert (after >= before - 0.15).all()
        # the posture angle is a tilt magnitude, so a mirrored body has the same one
        mirrored = warp_track_window(tr, GeoDraw(mirror=True), 0.3).angle
        assert np.allclose(mirrored, before, atol=0.15)


def test_geo_warp_keeps_boxes_valid_and_leaves_time_and_confidence_alone() -> None:
    tr = _kp_track([0.0, 45.0, 90.0])
    for draw in (
        GeoDraw(sy=0.55),
        GeoDraw(roll_rad=float(np.radians(12))),
        GeoDraw(gain=GEO_TRAPEZOID_MAX),
        GeoDraw(gain=-GEO_TRAPEZOID_MAX),
        GeoDraw(sy=0.6, roll_rad=float(np.radians(-11)), gain=0.09, mirror=True),
    ):
        out = warp_track_window(tr, draw, 0.3)
        assert (out.w > 0).all() and (out.h > 0).all()
        assert np.isfinite(out.cx).all() and np.isfinite(out.cy).all()
        assert np.isfinite(out.kpts).all()
        assert np.array_equal(out.t, tr.t)  # no time transform in this family
        assert np.array_equal(out.kpc, tr.kpc)  # confidence is not geometry


# ----------------------------------------------------------------- identity / off
def test_geo_identity_draw_short_circuits_and_identity_spec_is_byte_identical() -> None:
    sources, base = _geo_fixture()
    tr = sources[0].track
    assert warp_track_window(tr, GeoDraw(), 0.3) is tr  # strength 0 touches nothing
    aug = GeoAugmenter(GeoSpec(k=3, seed=11), FALL, sources, base)
    out = aug.augment_epoch(7)
    assert out.shape == (3, FALL.n_steps, len(CHANNELS))
    for j in range(3):
        assert np.array_equal(out[j], base[0])  # byte-identical, not allclose


def test_geo_sy_one_no_roll_no_trapezoid_is_the_identity() -> None:
    sources, base = _geo_fixture()
    aug = GeoAugmenter(
        GeoSpec(sy_min=1.0, sy_max=1.0, roll_deg=0.0, trapezoid=0.0, mirror_p=0.0, k=4, seed=2),
        FALL,
        sources,
        base,
    )
    assert np.array_equal(aug.augment_epoch(1), np.repeat(base, 4, axis=0))


def test_make_geo_augmenter_is_off_by_default() -> None:
    sources, base = _geo_fixture()
    for cfg in (None, {}, False, {"enabled": False}):
        assert make_geo_augmenter(cfg, FALL, sources, base) is None


def test_with_sources_does_not_change_the_windows_it_builds() -> None:
    """Carrying the raw rows is additive: the windows are the same windows."""
    tr = _kp_track([0.0, 45.0, 90.0])
    plain = windows_for_video("k", [tr], (2.0, 4.0), FALL)
    withs = windows_for_video("k", [tr], (2.0, 4.0), FALL, with_sources=True)
    assert plain and len(plain) == len(withs)
    for a, b in zip(plain, withs, strict=True):
        assert a.label == b.label and a.t_end == b.t_end and a.track_id == b.track_id
        assert np.array_equal(a.x, b.x)
        assert a.source is None and b.source is not None
        # and the carried rows re-featurize to the window, to the bit
        assert np.array_equal(featurize_window(b.source.track, b.source.t_end, FALL), b.x)


# ------------------------------------------------------------------ no NaN / ranges
def test_geo_augment_epoch_is_finite_in_range_and_source_major() -> None:
    sources, base = _geo_fixture()
    sources = sources * 3  # three source windows, so ordering is observable
    base = np.repeat(base, 3, axis=0)
    aug = GeoAugmenter(
        GeoSpec(
            sy_min=GEO_SY_MIN,
            sy_max=1.0,
            roll_deg=GEO_ROLL_MAX_DEG,
            trapezoid=GEO_TRAPEZOID_MAX,
            mirror_p=0.5,
            k=4,
            seed=99,
        ),
        FALL,
        sources,
        base,
    )
    c = FALL_COLS
    for epoch in range(1, 11):
        out = aug.augment_epoch(epoch)
        assert out.shape == (12, FALL.n_steps, len(CHANNELS))
        assert out.dtype == base.dtype
        assert np.isfinite(out).all()
        assert np.abs(out[:, :, c["vy"]]).max() <= 10.0
        assert np.abs(out[:, :, c["vx"]]).max() <= 10.0
        assert out[:, :, c["h_rel"]].min() >= 0.0 and out[:, :, c["h_rel"]].max() <= 4.0
        assert out[:, :, c["aspect"]].min() >= 0.0 and out[:, :, c["aspect"]].max() <= 6.0
        assert out[:, :, c["angle_norm"]].min() >= 0.0 and out[:, :, c["angle_norm"]].max() <= 1.0
        for flag in ("present", "angle_valid"):
            assert set(np.unique(out[:, :, c[flag]]).tolist()) <= {0.0, 1.0}
        # no time transform: coverage is a property of the tracker, not of the camera
        assert np.array_equal(out[:, :, c["present"]], np.repeat(base[:, :, c["present"]], 4, 0))
    assert aug.n_fallback == 0


def test_geo_same_seed_and_epoch_reproduce_and_epochs_differ() -> None:
    sources, base = _geo_fixture()
    spec = GeoSpec(sy_min=0.7, sy_max=1.0, roll_deg=8.0, trapezoid=0.06, mirror_p=0.5, k=4, seed=3)
    a = GeoAugmenter(spec, FALL, sources, base)
    b = GeoAugmenter(spec, FALL, sources, base)
    assert np.array_equal(a.augment_epoch(5), b.augment_epoch(5))
    assert not np.array_equal(a.augment_epoch(5), a.augment_epoch(6))


def test_geo_verify_identity_is_zero_and_the_api_takes_no_labels() -> None:
    import inspect

    sources, base = _geo_fixture()
    aug = GeoAugmenter(GeoSpec(sy_min=0.7, sy_max=1.0, k=2, seed=1), FALL, sources, base)
    assert aug.verify_identity() == 0
    for fn in (GeoAugmenter.augment_epoch, GeoAugmenter.augment_one, GeoAugmenter.draw):
        assert not (set(inspect.signature(fn).parameters) & {"y", "label", "labels"})


def test_geo_verify_identity_catches_a_mismatched_base() -> None:
    sources, base = _geo_fixture()
    tampered = base.copy()
    tampered[0, 0, 0] += np.float32(0.5)
    aug = GeoAugmenter(GeoSpec(k=1), FALL, sources, tampered)
    assert aug.verify_identity() == 1


# ------------------------------------------------------------------ protocol bounds
def test_geo_bounds_outside_the_protocol_are_refused() -> None:
    with pytest.raises(AugmentError):
        GeoSpec(sy_min=0.5)  # below the protocol floor 0.55
    with pytest.raises(AugmentError):
        GeoSpec(sy_min=0.9, sy_max=0.8)  # inverted range
    with pytest.raises(AugmentError):
        GeoSpec(roll_deg=GEO_ROLL_MAX_DEG + 0.1)
    with pytest.raises(AugmentError):
        GeoSpec(trapezoid=GEO_TRAPEZOID_MAX + 0.01)
    with pytest.raises(AugmentError):
        GeoSpec(k=0)
    with pytest.raises(AugmentError):
        GeoSpec(mirror_p=1.5)
    # the declared extremes are legal
    GeoSpec(
        sy_min=GEO_SY_MIN, sy_max=1.0, roll_deg=GEO_ROLL_MAX_DEG, trapezoid=GEO_TRAPEZOID_MAX, k=8
    )


def test_geo_unknown_key_is_refused() -> None:
    sources, base = _geo_fixture()
    with pytest.raises(AugmentError):
        make_geo_augmenter({"shear_deg": 3.0}, FALL, sources, base)


def test_geo_refuses_tracks_without_keypoints() -> None:
    """No `kp` rows means the angle could only be copied, which is the one thing
    this transform must never do."""
    tr = _kp_track([0.0])
    kpless = TrackArrays(
        track_id=tr.track_id,
        t=tr.t,
        cx=tr.cx,
        cy=tr.cy,
        w=tr.w,
        h=tr.h,
        angle=tr.angle,
        kpc=tr.kpc,
        kpts=None,
    )
    with pytest.raises(AugmentError):
        warp_track_window(kpless, GeoDraw(sy=0.7), 0.3)
    x = featurize_window(kpless, float(tr.t[-1]), FALL)
    assert x is not None
    with pytest.raises(AugmentError):
        GeoAugmenter(
            GeoSpec(k=1), FALL, [WindowSource(track=kpless, t_end=float(tr.t[-1]))], x[None]
        )


def test_geo_refuses_a_scene_channels_spec() -> None:
    sources, base = _geo_fixture()
    scene_spec = WindowSpec(
        window_s=8.0, stride_s=1.5, n_steps=64, policy="contain", scene_channels=True
    )
    with pytest.raises(AugmentError):
        GeoAugmenter(GeoSpec(k=1), scene_spec, sources, base)
