"""Demo-platform tests: media argv builders, catalog, and package import hygiene.

Style mirrors tests/test_dashboard.py: no models, no sockets, no network; cv2/numpy only
behind importorskip where pixels are actually touched; AST + subprocess import checks so
Linux CI (no numpy, no cv2 installed) can collect and pass this file.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from vividly.demo import catalog as demo_catalog
from vividly.demo import jobs as demo_jobs
from vividly.demo import live as demo_live
from vividly.demo import media as demo_media

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_SRC = REPO_ROOT / "src" / "vividly" / "demo"
MODEL_RECORD = REPO_ROOT / "final_models"

HEAVY_MODULES = {"cv2", "numpy", "torch", "rtmlib", "trackeval", "supervision", "onnxruntime"}


# -------------------------------------------------------------- import hygiene
def _module_level_statements(tree: ast.Module) -> list[ast.stmt]:
    """Top-level statements, skipping the bodies of `if TYPE_CHECKING:` blocks."""
    out: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            continue
        out.append(node)
    return out


def test_demo_package_has_no_top_level_heavy_imports() -> None:
    """Same contract as the dashboard's: CI with no numpy/cv2 must collect this package."""
    offenders: list[str] = []
    for py in sorted(DEMO_SRC.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in _module_level_statements(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            offenders.extend(
                f"{py.name}:{node.lineno}: {name}" for name in names if name in HEAVY_MODULES
            )
    assert not offenders, f"module-level heavy imports: {offenders}"


def test_importing_demo_pulls_in_no_model_stack() -> None:
    """Belt and braces over the AST check: import it for real and look at sys.modules."""
    # The probe reads the same set the AST check uses, so the two cannot drift apart.
    code = (
        "import sys; import vividly.demo; "
        f"heavy = set({sorted(HEAVY_MODULES)!r}) & "
        "{m.split('.')[0] for m in sys.modules}; "
        "sys.exit(1 if heavy else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# -------------------------------------------------------------- media: filenames
def test_sanitize_filename_tames_stem_and_lowercases_suffix() -> None:
    assert demo_media.sanitize_filename("IMG 2041 (1).MOV") == "IMG_2041_1.mov"
    assert demo_media.sanitize_filename("../..//evil path.mp4") == "evil_path.mp4"
    assert demo_media.sanitize_filename("!!!.mp4") == "upload.mp4"
    long = "a" * 200 + ".webm"
    assert demo_media.sanitize_filename(long) == "a" * 80 + ".webm"


def test_sanitize_filename_rejects_non_video_suffixes() -> None:
    for bad in ("model.onnx", "notes.txt", "archive.zip", "noext"):
        with pytest.raises(ValueError):
            demo_media.sanitize_filename(bad)


# -------------------------------------------------------------- media: argv goldens
def test_normalize_argv_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIVIDLY_FFMPEG", "/golden/ffmpeg")
    argv = demo_media.normalize_argv(Path("in.MOV"), Path("out.mp4"))
    assert argv == [
        "/golden/ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        "in.MOV",
        "-vf",
        "scale='min(1280,iw)':'min(1280,ih)':force_original_aspect_ratio=decrease"
        ":force_divisible_by=2,fps=15",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        "out.mp4",
    ]


def test_annotated_argv_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIVIDLY_FFMPEG", "/golden/ffmpeg")
    argv = demo_media.annotated_argv(Path("annotated.mp4"), width=640, height=480, fps=15.0)
    # the input side is the rawvideo pipe shape: raw BGR frames on stdin at size and rate
    assert argv[:6] == ["/golden/ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error"]
    assert argv[6:16] == [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "640x480",
        "-r",
        "15",
        "-i",
        "pipe:0",
    ]
    assert argv[16:] == [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        "annotated.mp4",
    ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_annotated_writer_roundtrip(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    dst = tmp_path / "annotated.mp4"
    writer = demo_media.AnnotatedWriter(dst, width=64, height=48, fps=15.0)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for _ in range(5):
        writer.write(frame)
    with pytest.raises(ValueError):  # size mismatch must fail loudly, not corrupt
        writer.write(np.zeros((48, 63, 3), dtype=np.uint8))
    writer.close()
    writer.close()  # idempotent
    assert dst.is_file() and dst.stat().st_size > 0
    assert not dst.with_name(dst.name + ".ffmpeg.log").exists()

    from vividly.media.ffmpeg import probe_video

    info = probe_video(dst)
    assert info["codec_name"] == "h264"
    assert info["nb_frames"] == 5


# -------------------------------------------------------------- media: demo dirs
def test_ensure_demo_dirs_creates_trees_with_do_not_train_readmes(tmp_path: Path) -> None:
    paths = demo_media.ensure_demo_dirs(tmp_path)
    for key in ("uploads", "incoming", "models", "runs"):
        assert paths[key].is_dir(), key
    for root in (paths["uploads"], paths["runs"]):
        text = (root / "README.md").read_text()
        assert "training" in text and "never" in text
    marker = paths["uploads"] / "README.md"
    marker.write_text("edited")
    demo_media.ensure_demo_dirs(tmp_path)  # idempotent: does not clobber
    assert marker.read_text() == "edited"


# -------------------------------------------------------------- catalog: fixture tree
def _write_lane(
    root: Path,
    lane: str,
    *,
    kind: str,
    backend: str,
    display_name: str,
    input_size: tuple[int, int] | None = (640, 640),
    weight_class: str | None = "prototype",
    blob: bytes | None = b"onnx-bytes",
    digest_of: bytes | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """One final_models/<lane>/ directory: the model.json sidecar, plus the graph itself
    unless `blob` is None, which stages a lane whose weights were never pulled.

    The sidecar carries the fields the catalog contract names. `digest_of` records the
    digest of bytes other than the ones written, which is how a tampered graph is staged.
    """
    d = root / "final_models" / lane
    d.mkdir(parents=True)
    source = digest_of if digest_of is not None else (blob if blob is not None else lane.encode())
    sidecar: dict[str, Any] = {
        "lane": lane,
        "kind": kind,
        "backend": backend,
        "display_name": display_name,
        "onnx": "model.onnx",
        "onnx_sha256": hashlib.sha256(source).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(b"checkpoint-" + lane.encode()).hexdigest(),
        "weight_class": weight_class,
        "training_data": "fixture data, no corpus",
        "backbone_pretrain": "fixture data, no corpus",
    }
    if input_size is not None:
        sidecar["input_size"] = [int(input_size[0]), int(input_size[1])]
    if extra:
        sidecar.update(extra)
    (d / "model.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    if blob is not None:
        (d / "model.onnx").write_bytes(blob)
    return d


def _sha8(lane_dir: Path) -> str:
    return json.loads((lane_dir / "model.json").read_text())["onnx_sha256"][:8]


def test_catalog_lists_the_detector_and_pose_lanes_only(tmp_path: Path) -> None:
    """The picker offers what the replay pipeline can build. The event verifier lanes are
    attached from their own run directories by the job engine, a kind the picker does not
    offer is skipped, and a sidecar that records a pending status is skipped rather than
    offered as a model nobody can select."""
    eyes = _write_lane(
        tmp_path,
        "eyes",
        kind="detector",
        backend="rtdetr_onnx",
        display_name="Vividly Eyes",
        input_size=(640, 640),
        blob=b"eyes-graph",
    )
    joints = _write_lane(
        tmp_path,
        "joints",
        kind="pose",
        backend="rtmlib",
        display_name="Vividly Joints",
        input_size=(192, 256),
        blob=b"joints-graph",
    )
    _write_lane(
        tmp_path,
        "fall",
        kind="event_verifier",
        backend="event_verifier",
        display_name="Vividly Fall",
        input_size=None,
    )
    _write_lane(
        tmp_path,
        "reid",
        kind="reid",
        backend="osnet_onnx",
        display_name="Vividly Re-identification",
        input_size=None,
        blob=None,
    )
    # A picker kind whose sidecar records a pending status: skipped on the status check,
    # which is the branch the reid lane above does not reach.
    _write_lane(
        tmp_path,
        "staged",
        kind="detector",
        backend="rtdetr_onnx",
        display_name="Vividly Staged",
        input_size=(640, 640),
        blob=b"staged-graph",
        extra={"status": "pending"},
    )
    (tmp_path / "final_models" / "notalane").mkdir()  # no sidecar: skipped

    cat = demo_catalog.scan_catalog(tmp_path)
    by_id = {e["id"]: e for e in [*cat["detectors"], *cat["poses"]]}
    assert set(by_id) == {f"eyes/{_sha8(eyes)}", f"joints/{_sha8(joints)}"}

    det = by_id[f"eyes/{_sha8(eyes)}"]
    assert det["kind"] == "detector"
    assert det["backend"] == "rtdetr_onnx"
    assert det["display_name"] == "Vividly Eyes"
    assert det["input_size"] == [640, 640]
    assert det["badge"] == "prototype"
    assert det["weight_class"] == "prototype"
    assert det["sha256_verified"] is True
    assert det["disabled"] is None
    assert det["source"] == "final_models"
    assert det["card"] is None  # provenance lives on the lane page, not in a card file
    assert Path(det["onnx_path"]) == (eyes / "model.onnx").resolve()

    pose = by_id[f"joints/{_sha8(joints)}"]
    assert pose["kind"] == "pose"
    assert pose["backend"] == "rtmlib"
    assert pose["input_size"] == [192, 256]


def test_catalog_disables_a_lane_whose_graph_is_still_a_git_lfs_pointer(tmp_path: Path) -> None:
    """A checkout made without `git lfs pull` holds a text pointer where the graph should
    be. The lane is listed and disabled with the fix in the reason, and the pointer is
    never hashed against the recorded digest, so no digest mismatch is reported for it."""
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"0" * 64 + b"\n"
        b"size 42000000\n"
    )
    eyes = _write_lane(
        tmp_path,
        "eyes",
        kind="detector",
        backend="rtdetr_onnx",
        display_name="Vividly Eyes",
        blob=pointer,
        digest_of=b"the real graph bytes",
    )
    cat = demo_catalog.scan_catalog(tmp_path)
    (det,) = cat["detectors"]
    assert det["id"] == f"eyes/{_sha8(eyes)}"
    assert det["onnx_present"] is False
    assert det["sha256_verified"] is False
    assert det["disabled"] == (
        "weight file is a Git LFS pointer: run git lfs pull to fetch final_models/eyes/model.onnx"
    )


def test_catalog_disables_a_lane_whose_graph_does_not_match_its_digest(tmp_path: Path) -> None:
    lane = _write_lane(
        tmp_path,
        "eyes",
        kind="detector",
        backend="rtdetr_onnx",
        display_name="Vividly Eyes",
        blob=b"x",
        digest_of=b"tampered",
    )
    entry = demo_catalog.scan_catalog(tmp_path)["detectors"][0]
    assert entry["id"] == f"eyes/{_sha8(lane)}"
    assert entry["onnx_present"] is True
    assert entry["sha256_verified"] is False
    assert "digest mismatch" in entry["disabled"]


def test_catalog_disables_a_lane_whose_graph_is_absent(tmp_path: Path) -> None:
    """The weights are stored with Git LFS, so a fresh clone without a pull has the
    sidecars and no graphs. That lane is listed and disabled, never silently dropped."""
    _write_lane(
        tmp_path,
        "joints",
        kind="pose",
        backend="rtmlib",
        display_name="Vividly Joints",
        input_size=None,
        blob=None,
    )
    entry = demo_catalog.scan_catalog(tmp_path)["poses"][0]
    assert entry["onnx_present"] is False
    assert entry["sha256_verified"] is False
    assert "git lfs pull" in entry["disabled"]
    assert entry["input_size"] == [192, 256]  # pose default when the sidecar omits it


@pytest.mark.parametrize("missing", ["backend", "onnx", "onnx_sha256"])
def test_catalog_skips_a_lane_whose_sidecar_is_incomplete(tmp_path: Path, missing: str) -> None:
    """A half-written sidecar names no runnable model, and offering one would put a
    model in the picker that the pipeline cannot build."""
    lane = _write_lane(
        tmp_path, "eyes", kind="detector", backend="rtdetr_onnx", display_name="Vividly Eyes"
    )
    sidecar = json.loads((lane / "model.json").read_text())
    del sidecar[missing]
    (lane / "model.json").write_text(json.dumps(sidecar))
    cat = demo_catalog.scan_catalog(tmp_path)
    assert cat["detectors"] == [] and cat["poses"] == []


def test_catalog_on_a_repo_without_a_model_record(tmp_path: Path) -> None:
    cat = demo_catalog.scan_catalog(tmp_path)
    assert cat["detectors"] == [] and cat["poses"] == []


def test_find_entry() -> None:
    cat = {"detectors": [{"id": "a/1"}], "poses": [{"id": "b/2"}]}
    assert demo_catalog.find_entry(cat, "b/2") == {"id": "b/2"}
    assert demo_catalog.find_entry(cat, "nope/0") is None


# -------------------------------------------------------------- overlay
def _synthetic_scene() -> tuple[dict, dict]:
    kps = [(40.0 + 10 * (i % 4), 60.0 + 8 * (i // 4), 0.9 if i != 3 else 0.1) for i in range(17)]
    scene = {
        "tracks": [
            {
                "track_id": 7,
                "xyxy": (30, 40, 120, 200),
                "keypoints": kps,
                "is_staff": False,
                "flash": 0.6,
            },
            {
                "track_id": 12,
                "xyxy": (180, 60, 280, 210),
                "keypoints": None,
                "is_staff": True,
                "flash": None,
            },
        ],
        "banners": ["CANDIDATE EVENT  Rhythmic convulsion - track 7 - onset 12.4 s"],
    }
    hud = {
        "left": "DET Vividly Eyes 40262049 [PROTOTYPE - INTERNAL DEMO]",
        "right": "frame 123/452 - e2e p50 104 ms - demo rules",
        "sub": "engineering demo - not a medical device",
    }
    return scene, hud


def test_overlay_draws_boxes_banner_and_hud(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    from vividly.demo import overlay

    scene, hud = _synthetic_scene()
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    out = overlay.draw_frame(img, scene, hud)
    assert out is img  # draws in place
    assert img[39:42, 30:120].any()  # flashed box top edge (track 7)
    assert img[60:210, 179:182].any()  # plain box left edge (track 12)
    assert img[0:26].any()  # banner strip (accent bar + text)
    assert img[-20:].any()  # HUD text rows
    cv2.imwrite(str(tmp_path / "overlay_eyeball.png"), img)  # for human eyeballing


def test_overlay_is_deterministic() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")  # numpy is a core dep; cv2 arrives only with the mac group
    from vividly.demo import overlay

    scene, hud = _synthetic_scene()
    a = overlay.draw_frame(np.zeros((240, 320, 3), np.uint8), scene, hud)
    b = overlay.draw_frame(np.zeros((240, 320, 3), np.uint8), scene, hud)
    assert np.array_equal(a, b)


def test_overlay_survives_degenerate_tracks() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from vividly.demo import overlay

    img = np.zeros((100, 100, 3), np.uint8)
    scene = {
        "tracks": [
            {"track_id": 1, "xyxy": (-50, -50, -10, -10), "keypoints": None},  # fully outside
            {"track_id": 2, "xyxy": (90, 90, 300, 300), "keypoints": None},  # clipped
            {"track_id": 3, "xyxy": (10, 10, 10, 10), "keypoints": None},  # zero area
        ],
        "banners": [],
    }
    overlay.draw_frame(img, scene, {})  # must not raise
    assert img[90:100, 90:100].any()  # the clipped box still drew its visible part


def test_track_color_is_stable_and_distinct() -> None:
    from vividly.demo import overlay

    assert overlay.track_color(7) == overlay.track_color(7)
    assert len({overlay.track_color(i) for i in range(12)}) >= 10


# --------------------------------------------------------- catalog: the model record
needs_model_record = pytest.mark.skipif(
    not (MODEL_RECORD / "eyes" / "model.json").is_file(),
    reason="final_models/ sidecars not present",
)


# -------------------------------------------------------------- jobs: cfg + profiles
def _fake_entry(
    kind: str, name: str, sha8: str, size: tuple[int, int], backend: str = "rtmlib"
) -> dict:
    return {
        "id": f"{name}/{sha8}",
        "lane": name,
        "kind": kind,
        "backend": backend,
        "display_name": name,
        "sha8": sha8,
        "badge": "vividly",
        "weight_class": None,
        "card": None,
        "input_size": list(size),
        "onnx_path": f"/abs/{name}.onnx",
        "disabled": None,
    }


def test_build_run_cfg_golden(tmp_path: Path) -> None:
    det = _fake_entry("detector", "eyes", "40262049", (640, 640), backend="rtdetr_onnx")
    pose = _fake_entry("pose", "joints", "6ba96285", (192, 256), backend="rtmlib")
    common = {
        "job_id": "j1",
        "runs_dir": tmp_path,
        "input_mp4": tmp_path / "input.mp4",
        "detector": det,
        "pose": pose,
        "score_thr": 0.4,
        "repo_root": REPO_ROOT,
        "fps": 15.0,
    }
    cfg = demo_jobs.build_run_cfg(rules_profile="demo", **common)
    assert cfg["run_name"] == "j1" and cfg["headless"] is True
    assert cfg["source"] == {"path": str(tmp_path / "input.mp4"), "rate": None}
    assert cfg["detector"]["device"] == "cpu" and cfg["pose"]["device"] == "cpu"
    assert cfg["detector"]["onnx"] == "/abs/eyes.onnx"
    assert cfg["detector"]["input_size"] == [640, 640]
    assert cfg["tracker"]["track_activation_threshold"] == 0.5  # det thr + 0.1 convention
    assert cfg["tracker"]["high_conf_det_threshold"] == 0.5
    assert cfg["tracker"]["frame_rate"] == 15.0 and cfg["tracks"]["fps"] == 15.0
    assert "rules" in cfg and "rules_config" not in cfg

    cfg2 = demo_jobs.build_run_cfg(rules_profile="production", **common)
    assert "rules" not in cfg2
    assert Path(cfg2["rules_config"]).is_absolute()
    assert cfg2["rules_config"].endswith("configs/pipeline/rules.yaml")


def test_build_run_cfg_takes_each_backend_from_its_catalog_entry(tmp_path: Path) -> None:
    """The eyes lane is an RT-DETRv4 graph and the joints lane an RTMPose graph, and each
    needs its own preprocessing and postprocessing. The backend therefore travels from the
    lane sidecar through the catalog entry into the run config; it is never a constant in
    the job engine, because running one graph under the other's backend raises nothing and
    returns plausible numbers that are wrong."""
    det = _fake_entry("detector", "eyes", "40262049", (640, 640), backend="rtdetr_onnx")
    pose = _fake_entry("pose", "joints", "6ba96285", (192, 256), backend="rtmlib")
    cfg = demo_jobs.build_run_cfg(
        job_id="j1",
        runs_dir=tmp_path,
        input_mp4=tmp_path / "input.mp4",
        detector=det,
        pose=pose,
        score_thr=0.4,
        rules_profile="demo",
        repo_root=REPO_ROOT,
        fps=15.0,
    )
    assert cfg["detector"]["backend"] == "rtdetr_onnx"
    assert cfg["pose"]["backend"] == "rtmlib"

    # an upload, or any other lane, is run under whatever backend its entry declares
    other = _fake_entry("detector", "upload", "aaaa1111", (640, 640), backend="rtmlib")
    cfg2 = demo_jobs.build_run_cfg(
        job_id="j2",
        runs_dir=tmp_path,
        input_mp4=tmp_path / "input.mp4",
        detector=other,
        pose=pose,
        score_thr=0.4,
        rules_profile="demo",
        repo_root=REPO_ROOT,
        fps=15.0,
    )
    assert cfg2["detector"]["backend"] == "rtmlib"

    # both keys must exist in the closed registry, or the pipeline refuses the job
    pytest.importorskip("numpy")  # vividly.models.base is numpy-typed
    from vividly.models.factory import DETECTORS, POSE

    assert cfg["detector"]["backend"] in DETECTORS
    assert cfg["pose"]["backend"] in POSE


def test_build_run_cfg_attaches_verifier_heads_when_trained(tmp_path: Path) -> None:
    """The live-refresh contract: a verifier training run landing on disk joins the
    next job's config; an untrained rule id attaches nothing."""
    det = _fake_entry("detector", "eyes", "40262049", (640, 640), backend="rtdetr_onnx")
    pose = _fake_entry("pose", "joints", "6ba96285", (192, 256), backend="rtmlib")
    common = {
        "job_id": "j1",
        "runs_dir": tmp_path,
        "input_mp4": tmp_path / "input.mp4",
        "detector": det,
        "pose": pose,
        "score_thr": 0.4,
        "repo_root": tmp_path,
        "fps": 15.0,
    }
    assert "verify" not in demo_jobs.build_run_cfg(rules_profile="demo", **common)

    # Follow VERIFIER_RUN_DIRS: the demo attaches the committed fall lane directory
    # final_models/fall (fall_v3, checkpoint 3d2d17ee, graph 67954c04), so the fixture must
    # too.
    run_dir = tmp_path / demo_jobs.VERIFIER_RUN_DIRS["fall_then_still_v0"]
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text("{}")
    (run_dir / "verifier.onnx").write_bytes(b"x")
    cfg = demo_jobs.build_run_cfg(rules_profile="demo", **common)
    assert cfg["verify"]["policy"] == "annotate"
    assert set(cfg["verify"]["heads"]) == {"fall_then_still_v0"}
    assert cfg["verify"]["heads"]["fall_then_still_v0"]["run_dir"] == str(run_dir)


def test_demo_rules_profile_is_a_valid_rule_config() -> None:
    """The demo profile must load through the real RuleConfig (typos/validators)."""
    from vividly.rules.base import RuleConfig

    rc = RuleConfig.from_mapping(demo_jobs.RULES_PROFILES["demo"]["rules"])
    assert rc.down_still_min_s == 5.0
    assert rc.refire_after_s == 30.0
    assert rc.fall_upright_max_deg == 30.0  # untouched keys keep the clinical defaults
    # egress demo calibration: shrunk clock and presence floor, untouched exit window
    assert rc.egress_absent_min_s == 20.0
    assert rc.egress_min_presence_s == 10.0
    assert rc.egress_exit_window_s == 5.0
    # the egress head only notices a departure when the engine evaluates between last
    # sighting and reap (vividly.rules.egress docstring): eval must beat the demo reap
    assert rc.eval_every_s < demo_jobs.EGRESS_DEMO_REAP_AFTER_S
    assert rc.egress_absent_min_s > demo_jobs.EGRESS_DEMO_REAP_AFTER_S


# -------------------------------------------------------------- jobs: hook + state
def _mk_job(tmp_path: Path, rules_profile: str = "demo") -> demo_jobs.JobState:
    det = {
        "id": "eyes/40262049",
        "display_name": "Vividly Eyes",
        "sha8": "40262049",
        "backend": "rtdetr_onnx",
        "badge": "prototype",
        "weight_class": "prototype",
        "card": None,
        "input_size": [640, 640],
    }
    pose = {
        "id": "joints/6ba96285",
        "display_name": "Vividly Joints",
        "sha8": "6ba96285",
        "backend": "rtmlib",
        "badge": "vividly",
        "weight_class": None,
        "card": None,
        "input_size": [192, 256],
    }
    return demo_jobs.JobState(
        "job1234x",
        tmp_path,
        source_label="clip.mp4",
        detector=det,
        pose=pose,
        score_thr=0.4,
        rules_profile=rules_profile,
        max_s=None,
    )


class _FakeWriter:
    def __init__(self) -> None:
        self.frames_written = 0

    def write(self, frame) -> None:
        self.frames_written += 1


def _fake_firing() -> NS:
    return NS(
        rule_id="seizure_candidate_v0",
        rule_version="0.1.0",
        track_id=7,
        onset_t=10.0,
        detected_t=12.0,
        details={"mean_step_px": 3.1},
    )


def _fake_pipeline(now: float) -> NS:
    obs = NS(cx=100.0, cy=80.0, w=40.0, h=60.0, keypoints=None)
    ts = NS(last_observation=obs, is_staff=False)
    return NS(book=NS(now=now, tracks=lambda: {7: ts}))


def test_frame_hook_firings_flash_and_banner_expiry(tmp_path: Path) -> None:
    job = _mk_job(tmp_path)
    hook = demo_jobs.DemoFrameHook(job, _FakeWriter(), "demo rules")
    hook.on_firings([_fake_firing()], 12.0)
    assert job.alerts and job.alerts[0]["rule_id"] == "seizure_candidate_v0"
    assert job.alerts[0]["onset_ts"] == 10.0

    scene = hook.scene(_fake_pipeline(now=13.0))
    assert scene["tracks"][0]["flash"] == pytest.approx(1.0)
    assert scene["banners"] and "CANDIDATE EVENT" in scene["banners"][0]
    # display name from EVENT_MODEL_NAMES in src/vividly/demo/jobs.py; the unmodified rule
    # id travels in the record
    assert "Rhythmic convulsion" in scene["banners"][0]

    late = hook.scene(_fake_pipeline(now=16.1))  # 4.1 s after detected_t: window over
    assert late["tracks"][0]["flash"] is None
    assert late["banners"] == []


def test_frame_hook_handles_an_egress_firing_for_a_reaped_track(tmp_path: Path) -> None:
    """An egress firing names a track the book has already reaped (see
    vividly.rules.egress): the banner and the alert row must still appear even though
    there is no live box left to flash."""
    job = _mk_job(tmp_path)
    hook = demo_jobs.DemoFrameHook(job, _FakeWriter(), "demo rules")
    firing = NS(
        rule_id="egress_without_discharge_v0",
        rule_version="0.1.0",
        track_id=42,  # not in the book: reaped before the clock ran out
        onset_t=30.0,
        detected_t=50.0,
        details={"exit_zone": "exit", "absent_s": 20.0},
    )
    hook.on_firings([firing], 50.0)
    assert job.alerts[0]["rule_id"] == "egress_without_discharge_v0"
    assert job.alerts[0]["onset_ts"] == 30.0  # when the person was last seen
    assert job.alerts[0]["details"]["exit_zone"] == "exit"
    scene = hook.scene(_fake_pipeline(now=50.5))  # book only holds track 7
    assert scene["banners"] and "Egress without discharge" in scene["banners"][0]
    assert all(tr["track_id"] != 42 for tr in scene["tracks"])  # nothing to flash, no crash


def test_frame_hook_hide_profile_records_but_never_shows(tmp_path: Path) -> None:
    job = _mk_job(tmp_path, rules_profile="hide")
    hook = demo_jobs.DemoFrameHook(job, _FakeWriter(), "hide")
    hook.on_firings([_fake_firing()], 12.0)
    assert job.alerts  # recorded
    scene = hook.scene(_fake_pipeline(now=13.0))
    assert scene["tracks"][0]["flash"] is None and scene["banners"] == []


def test_models_line_carries_shas_and_prototype_badge(tmp_path: Path) -> None:
    line = demo_jobs.DemoFrameHook._models_line(_mk_job(tmp_path))
    assert "40262049" in line and "6ba96285" in line
    assert "PROTOTYPE - INTERNAL DEMO" in line


JOB_CONTRACT_KEYS = frozenset(
    {
        "job_id",
        "status",
        "error",
        "created_utc",
        "video",
        "models",
        "score_thr",
        "rules_profile",
        "frame_b64",
        "frame_index",
        "n_total",
        "progress",
        "proc_fps",
        "eta_s",
        "queue_position",
        "tracks_live",
        "alerts",
        "latency_p50_ms",
        "latency_p95_ms",
        "annotated_ready",
        "compare_group",
        "compare_peer",
        "live",
        "zones",
        "disclaimer",
    }
)


def test_job_snapshot_contract_and_persist_roundtrip(tmp_path: Path) -> None:
    job = _mk_job(tmp_path)
    snap = job.snapshot()
    assert set(snap) >= JOB_CONTRACT_KEYS
    assert "not a medical device" in snap["disclaimer"]

    job.set_status("running")
    job.persist()
    restored = demo_jobs.JobState.from_json(tmp_path)
    assert restored is not None
    assert restored.status == "interrupted"  # frozen mid-run means the process died

    job.set_status("done")
    job.persist()
    restored2 = demo_jobs.JobState.from_json(tmp_path)
    assert restored2 is not None
    assert restored2.status == "done"
    assert restored2.detector["sha8"] == "40262049"


# -------------------------------------------------------------- app: asgi client
class _Resp(NS):
    def json(self) -> Any:
        return json.loads(self.body)


def _http_scope(method: str, path: str, body_len: int) -> dict[str, Any]:
    raw_path, _, query = path.partition("?")
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": raw_path,
        "raw_path": raw_path.encode(),
        "query_string": query.encode(),
        "headers": [(b"content-length", str(body_len).encode())],
        "client": ("127.0.0.1", 4000),
        "server": ("127.0.0.1", 8700),
    }


def _ws_scope(path: str) -> dict[str, Any]:
    return {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "ws",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 4001),
        "server": ("127.0.0.1", 8700),
        "subprotocols": [],
    }


class AsgiClient:
    """Drive the FastAPI app in process (tests/test_dashboard.py pattern: no httpx)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def get(self, path: str) -> _Resp:
        import asyncio

        return asyncio.run(self._request("GET", path, b""))

    def post(self, path: str, payload: Any = None, raw: bytes | None = None) -> _Resp:
        import asyncio

        body = raw if raw is not None else json.dumps({} if payload is None else payload).encode()
        return asyncio.run(self._request("POST", path, body))

    async def _request(self, method: str, path: str, body: bytes) -> _Resp:
        import asyncio

        pending = [{"type": "http.request", "body": body, "more_body": False}]
        received: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return pending.pop(0) if pending else {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            received.append(message)

        await asyncio.wait_for(self.app(_http_scope(method, path, len(body)), receive, send), 15)
        status = next(m["status"] for m in received if m["type"] == "http.response.start")
        payload = b"".join(
            m.get("body", b"") for m in received if m["type"] == "http.response.body"
        )
        return _Resp(status=status, body=payload)

    def ws_messages(self, path: str, max_messages: int = 6) -> list[dict[str, Any]]:
        """Connect, collect messages until close (or max), return the raw ASGI messages."""
        import asyncio

        return asyncio.run(self._ws(path, max_messages))

    def ws_push_bytes(self, path: str, payloads: list[bytes]) -> list[dict[str, Any]]:
        """Connect, send binary frames, disconnect; return what the app sent back."""
        import asyncio

        return asyncio.run(self._ws_push(path, payloads))

    async def _ws_push(self, path: str, payloads: list[bytes]) -> list[dict[str, Any]]:
        import asyncio
        import contextlib

        to_app: asyncio.Queue = asyncio.Queue()
        from_app: asyncio.Queue = asyncio.Queue()
        await to_app.put({"type": "websocket.connect"})
        for data in payloads:
            await to_app.put({"type": "websocket.receive", "bytes": data})
        await to_app.put({"type": "websocket.disconnect", "code": 1000})

        async def receive() -> dict[str, Any]:
            return await to_app.get()

        async def send(message: dict[str, Any]) -> None:
            await from_app.put(message)

        task = asyncio.create_task(self.app(_ws_scope(path), receive, send))
        with contextlib.suppress(BaseException):  # teardown is not under test
            await asyncio.wait_for(task, timeout=10)
        out: list[dict[str, Any]] = []
        while not from_app.empty():
            out.append(from_app.get_nowait())
        return out

    async def _ws(self, path: str, max_messages: int) -> list[dict[str, Any]]:
        import asyncio

        to_app: asyncio.Queue = asyncio.Queue()
        from_app: asyncio.Queue = asyncio.Queue()
        await to_app.put({"type": "websocket.connect"})

        async def receive() -> dict[str, Any]:
            return await to_app.get()

        async def send(message: dict[str, Any]) -> None:
            await from_app.put(message)

        task = asyncio.create_task(self.app(_ws_scope(path), receive, send))
        out: list[dict[str, Any]] = []
        try:
            for _ in range(max_messages):
                message = await asyncio.wait_for(from_app.get(), timeout=10)
                out.append(message)
                if message["type"] == "websocket.close":
                    break
        finally:
            import contextlib

            await to_app.put({"type": "websocket.disconnect", "code": 1000})
            with contextlib.suppress(BaseException):  # app task teardown is not under test
                await asyncio.wait_for(task, timeout=10)
        return out


@pytest.fixture()
def demo_app(tmp_path: Path):
    from vividly.demo.app import create_demo_app

    mgr = demo_jobs.JobManager(REPO_ROOT, outputs_root=tmp_path)
    try:
        yield AsgiClient(create_demo_app(mgr, max_upload_mb=1)), mgr
    finally:
        mgr.shutdown()


def test_page_serves_and_is_self_contained(demo_app) -> None:
    client, _ = demo_app
    r = client.get("/")
    assert r.status == 200
    html = r.body.decode()
    assert "Vividly - Model Demo" in html
    # The "not a medical device" line is burned into every recorded and live overlay by
    # the DemoFrameHook head-up display (test_overlay_draws_boxes_banner_and_hud); the
    # page itself carries no footer, decided on 2026-08-25.
    # The inlined logo's base64 can contain any letter run ("cdn" included), so scrub
    # data URIs before scanning for external assets.
    scrubbed = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", html)
    assert "https://" not in scrubbed and "cdn" not in scrubbed.lower()


def test_models_route_returns_catalog(demo_app) -> None:
    client, _ = demo_app
    body = client.get("/api/models").json()
    assert "detectors" in body and "poses" in body


def test_the_platform_serves_no_model_card_route(demo_app) -> None:
    """A lane's provenance lives on its page in the model record, not in a card file the
    platform serves, so there is no card route and no catalog entry links one."""
    client, _ = demo_app
    assert client.get("/api/models/card/nope.md").status == 404
    for entry in [*client.get("/api/models").json()["detectors"]]:
        assert entry["card"] is None


def test_upload_validates_suffix_cap_and_emptiness(demo_app) -> None:
    client, mgr = demo_app
    assert client.post("/api/upload?filename=evil.exe", raw=b"x").status == 400
    assert client.post("/api/upload?filename=a.mp4", raw=b"").status == 400
    big = b"0" * (1024 * 1024 + 100)  # over the 1 MB fixture cap
    assert client.post("/api/upload?filename=a.mp4", raw=big).status == 413
    assert not any((mgr.dirs["incoming"]).iterdir())  # every rejection cleaned up

    ok = client.post("/api/upload?filename=My Clip.MOV", raw=b"notavideo")
    assert ok.status == 200
    body = ok.json()
    assert body["filename"] == "My_Clip.mov" and body["bytes"] == 9
    assert (mgr.dirs["incoming"] / body["upload_id"] / "My_Clip.mov").is_file()


def test_job_creation_validates_inputs(demo_app) -> None:
    client, _ = demo_app
    r = client.post("/api/jobs", {"upload_id": "zzzz", "detector_id": "x", "pose_id": "y"})
    assert r.status == 400 and "unknown upload" in r.json()["error"]
    r2 = client.post("/api/jobs", {"sample": True, "detector_id": "no/pe", "pose_id": "no/pe"})
    assert r2.status == 400 and "unknown detector" in r2.json()["error"]


def test_job_routes_404_and_409(demo_app) -> None:
    client, _ = demo_app
    assert client.get("/api/jobs/deadbeef").status == 404
    assert client.get("/api/jobs/deadbeef/annotated.mp4").status == 404
    assert client.get("/api/jobs/deadbeef/thumb.jpg").status == 404
    assert client.post("/api/jobs/deadbeef/cancel").status == 409
    assert client.post("/api/jobs/deadbeef/delete").status == 409


def test_ws_unknown_job_closes_4404(demo_app) -> None:
    client, _ = demo_app
    messages = client.ws_messages("/ws/jobs/deadbeef")
    assert messages[0]["type"] == "websocket.accept"
    assert messages[-1]["type"] == "websocket.close" and messages[-1].get("code") == 4404


def test_ws_snapshot_contract_and_terminal_close(demo_app, tmp_path: Path) -> None:
    client, mgr = demo_app
    job = _mk_job(tmp_path)
    job.set_status("done")
    with mgr._lock:
        mgr._jobs[job.job_id] = job
        mgr._order.append(job.job_id)
    messages = client.ws_messages(f"/ws/jobs/{job.job_id}")
    sends = [m for m in messages if m["type"] == "websocket.send"]
    assert sends, messages
    snap = json.loads(sends[0]["text"])
    assert set(snap) >= JOB_CONTRACT_KEYS
    assert messages[-1]["type"] == "websocket.close"  # terminal status: one push, close

    listing = client.get("/api/jobs").json()
    assert any(j["job_id"] == job.job_id for j in listing["jobs"])
    assert all("frame_b64" not in j for j in listing["jobs"])


# -------------------------------------------------------------- phase 6: .onnx drop-in
TINY_ONNX = Path(__file__).parent / "fixtures" / "reid_synth" / "tiny_embedder.onnx"


def test_catalog_lists_dropped_in_uploads(tmp_path: Path) -> None:
    good = tmp_path / "models" / "aaaa1111"
    good.mkdir(parents=True)
    (good / "meta.json").write_text(
        json.dumps({"kind": "detector", "display_name": "mystery", "input_size": [320, 320]})
    )
    (good / "mystery.onnx").write_bytes(b"onnx")
    gone = tmp_path / "models" / "bbbb2222"  # meta without blob: listed but disabled
    gone.mkdir(parents=True)
    (gone / "meta.json").write_text(json.dumps({"kind": "pose", "input_size": [192, 256]}))
    junk = tmp_path / "models" / "cccc3333"  # bad kind: skipped entirely
    junk.mkdir(parents=True)
    (junk / "meta.json").write_text(json.dumps({"kind": "wat"}))

    cat = demo_catalog.scan_catalog(tmp_path, uploads_models_dir=tmp_path / "models")
    by_id = {e["id"]: e for e in [*cat["detectors"], *cat["poses"]]}
    assert set(by_id) == {"upload/aaaa1111", "upload/bbbb2222"}

    det = by_id["upload/aaaa1111"]
    assert det["kind"] == "detector"
    assert det["backend"] == demo_catalog.UPLOAD_BACKEND
    assert det["display_name"] == "Unverified upload: mystery"
    assert det["badge"] == "upload"
    assert det["input_size"] == [320, 320]
    assert det["disabled"] is None
    assert by_id["upload/bbbb2222"]["disabled"] == "uploaded file missing"


def test_uploads_sort_after_the_model_record(tmp_path: Path) -> None:
    _write_lane(
        tmp_path, "eyes", kind="detector", backend="rtdetr_onnx", display_name="Vividly Eyes"
    )
    up = tmp_path / "models" / "aaaa1111"
    up.mkdir(parents=True)
    (up / "meta.json").write_text(json.dumps({"kind": "detector", "input_size": [320, 320]}))
    (up / "m.onnx").write_bytes(b"onnx")
    cat = demo_catalog.scan_catalog(tmp_path, uploads_models_dir=tmp_path / "models")
    assert [e["badge"] for e in cat["detectors"]] == ["prototype", "upload"]
    # an upload declares no backend of its own, so it runs under the rtmlib convention
    assert cat["detectors"][1]["backend"] == demo_catalog.UPLOAD_BACKEND


@pytest.mark.skipif(not TINY_ONNX.is_file(), reason="reid_synth fixture absent")
def test_infer_onnx_input_size_reads_static_hw() -> None:
    pytest.importorskip("onnxruntime")  # ships with the mac group only
    # tiny_embedder input is (batch, 3, 256, 128): dynamic batch is fine, H/W must be static
    assert demo_catalog.infer_onnx_input_size(TINY_ONNX) == (128, 256)


def test_model_upload_route_validation(demo_app) -> None:
    client, mgr = demo_app
    assert client.post("/api/models/upload?kind=wat&filename=m.onnx", raw=b"x").status == 400
    assert client.post("/api/models/upload?kind=detector&filename=m.txt", raw=b"x").status == 400
    assert client.post("/api/models/upload?kind=detector&filename=m.onnx", raw=b"").status == 400
    big = b"0" * (1024 * 1024 + 100)  # over the 1 MB fixture cap
    assert client.post("/api/models/upload?kind=detector&filename=m.onnx", raw=big).status == 413

    r = client.post("/api/models/upload?kind=detector&filename=garbage.onnx", raw=b"not-onnx")
    assert r.status == 400 and "cannot use this model" in r.json()["error"]
    # every rejection cleaned up: no .part files, no half-made sha dirs
    assert list(mgr.dirs["models"].iterdir()) == []


@pytest.mark.skipif(not TINY_ONNX.is_file(), reason="reid_synth fixture absent")
def test_model_upload_route_happy_path_and_duplicate(demo_app) -> None:
    # The route infers the graph's input size, which needs onnxruntime; without it the
    # upload is correctly refused with 400, so the happy path is unobservable here.
    pytest.importorskip("onnxruntime")
    client, mgr = demo_app
    blob = TINY_ONNX.read_bytes()
    r = client.post("/api/models/upload?kind=pose&filename=Tiny Embedder.onnx", raw=blob)
    assert r.status == 200
    body = r.json()
    sha8 = hashlib.sha256(blob).hexdigest()[:8]
    assert body["id"] == f"upload/{sha8}"
    assert body["input_size"] == [128, 256]
    meta = json.loads((mgr.dirs["models"] / sha8 / "meta.json").read_text())
    assert meta["kind"] == "pose" and meta["sha256"] == hashlib.sha256(blob).hexdigest()

    again = client.post("/api/models/upload?kind=pose&filename=other-name.onnx", raw=blob)
    assert again.json() == {"id": f"upload/{sha8}", "duplicate": True}
    assert len(list((mgr.dirs["models"] / sha8).glob("*.onnx"))) == 1  # same bytes stored once

    listed = client.get("/api/models").json()
    entry = next(e for e in listed["poses"] if e["id"] == f"upload/{sha8}")
    assert entry["badge"] == "upload" and entry["input_size"] == [128, 256]


# -------------------------------------------------------------- phase 7: compare mode
def _enabled(entries: list[dict]) -> list[dict]:
    return [e for e in entries if not e["disabled"]]


def _runnable_pair(mgr: demo_jobs.JobManager) -> tuple[str, str]:
    """The first selectable detector and pose ids, or a skip when the graphs are absent.

    The model record ships its graphs through Git LFS, so a clone without a pull has the
    sidecars and no weights; there is nothing to run end to end on such a machine.
    """
    cat = mgr.catalog()
    dets, poses = _enabled(cat["detectors"]), _enabled(cat["poses"])
    if not dets or not poses:
        pytest.skip("no model record lane with its graph on disk")
    return dets[0]["id"], poses[0]["id"]


def test_compare_creates_linked_job_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vividly.demo.app import create_demo_app

    # stub the worker: this test is about creation + linking, not inference
    monkeypatch.setattr(demo_jobs.JobManager, "_run_job", lambda self, job: job.set_status("done"))
    mgr = demo_jobs.JobManager(REPO_ROOT, outputs_root=tmp_path)
    try:
        cat = mgr.catalog()
        dets, poses = _enabled(cat["detectors"]), _enabled(cat["poses"])
        if not dets or not poses:
            pytest.skip("model record graphs not on disk")
        client = AsgiClient(create_demo_app(mgr, max_upload_mb=1))

        up = client.post("/api/upload?filename=clip.mp4", raw=b"fake-video-bytes").json()
        body = client.post(
            "/api/jobs",
            {
                "upload_id": up["upload_id"],
                "detector_id": dets[0]["id"],
                "pose_id": poses[0]["id"],
                "compare_detector_id": dets[-1]["id"],
            },
        ).json()
        a_id, b_id = body["job_id"], body["compare_job_id"]
        assert b_id and b_id != a_id

        a, b = mgr.get(a_id), mgr.get(b_id)
        assert a.compare_group == b.compare_group is not None
        assert (a.compare_peer, b.compare_peer) == (b_id, a_id)
        assert b.detector["id"] == dets[-1]["id"]
        # one upload feeds both sides, then the incoming shell is gone
        assert (a.job_dir / "original.mp4").is_file()
        assert (b.job_dir / "original.mp4").is_file()
        assert list(mgr.dirs["incoming"].iterdir()) == []
        # linking persisted: survives a from_json reload
        restored = demo_jobs.JobState.from_json(a.job_dir)
        assert (restored.compare_group, restored.compare_peer) == (a.compare_group, b_id)

        snap = client.get(f"/api/jobs/{a_id}").json()
        assert snap["compare_peer"] == b_id and snap["compare_group"] == a.compare_group
    finally:
        mgr.shutdown()


def test_job_without_compare_reports_null_fields(demo_app, tmp_path: Path) -> None:
    client, mgr = demo_app
    job = _mk_job(tmp_path)
    job.set_status("done")
    with mgr._lock:
        mgr._jobs[job.job_id] = job
        mgr._order.append(job.job_id)
    snap = client.get(f"/api/jobs/{job.job_id}").json()
    assert snap["compare_group"] is None and snap["compare_peer"] is None


def test_page_carries_phase_6_and_7_controls(demo_app) -> None:
    client, _ = demo_app
    html = client.get("/").body.decode()
    for marker in ("addonnx", "onnxfile", "Compare against", "view-cmp", "Play both", "#cmp/"):
        assert marker in html, marker


# -------------------------------------------------------------- live camera lane
def test_live_source_mailbox_keeps_newest_frame() -> None:
    src = demo_live.LiveSource()
    assert src.push(b"a") and src.push(b"b") and src.push(b"c")
    assert (src.n_received, src.n_dropped) == (3, 2)
    assert src._box.get_nowait() == b"c"
    src.push(b"")  # empty and oversized frames are dropped, never buffered
    src.push(b"x" * (demo_live.MAX_FRAME_BYTES + 1))
    assert src.n_dropped == 4 and src._box.empty()


def test_live_source_stop_semantics() -> None:
    src = demo_live.LiveSource(first_frame_timeout_s=5.0)
    src.stop()
    assert src.push(b"a") is False
    with pytest.raises(demo_live.LiveStopped):
        src.open()
    with pytest.raises(RuntimeError, match="cannot seek"):
        src.seek(0)


def test_live_source_open_times_out_without_frames() -> None:
    src = demo_live.LiveSource(first_frame_timeout_s=0.05)
    with pytest.raises(RuntimeError, match="no camera frames"):
        src.open()


def test_live_source_frames_yield_then_end_on_idle() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    ok, jpeg = cv2.imencode(".jpg", np.zeros((32, 48, 3), dtype=np.uint8))
    assert ok
    data = jpeg.tobytes()
    src = demo_live.LiveSource(first_frame_timeout_s=2.0, idle_timeout_s=0.2)
    assert src.push(data)
    info = src.open()
    assert (info.width, info.height, info.n_frames) == (48, 32, None)
    src.push(data)
    frames = list(src.frames())  # drains both frames, then the idle timeout ends it
    assert [f.index for f in frames] == [0, 1]
    assert frames[0].t_source == 0.0 and frames[1].t_source >= 0.0
    assert frames[0].image.shape == (32, 48, 3)
    corrupt = demo_live.LiveSource(first_frame_timeout_s=2.0, idle_timeout_s=0.2)
    corrupt.push(b"not a jpeg")  # undecodable bytes are skipped, not fatal
    with pytest.raises(RuntimeError, match="no camera frames"):
        corrupt.first_frame_timeout_s = 0.3
        corrupt.open()


def test_live_flag_survives_persist_roundtrip(tmp_path: Path) -> None:
    job = _mk_job(tmp_path)
    job.live = True
    job.set_status("done")
    job.persist()
    restored = demo_jobs.JobState.from_json(tmp_path)
    assert restored is not None and restored.live is True


def test_live_job_creation_frames_ws_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vividly.demo.app import create_demo_app

    gate = threading.Event()

    def _stub(self, job):  # hold the worker so the live job stays observable
        gate.wait(timeout=10)
        job.set_status("done")

    monkeypatch.setattr(demo_jobs.JobManager, "_run_job", _stub)
    mgr = demo_jobs.JobManager(REPO_ROOT, outputs_root=tmp_path)
    try:
        cat = mgr.catalog()
        dets, poses = _enabled(cat["detectors"]), _enabled(cat["poses"])
        if not dets or not poses:
            pytest.skip("model record graphs not on disk")
        client = AsgiClient(create_demo_app(mgr, max_upload_mb=1))

        bad = client.post(
            "/api/jobs",
            {
                "live": True,
                "detector_id": dets[0]["id"],
                "pose_id": poses[0]["id"],
                "compare_detector_id": dets[0]["id"],
            },
        )
        assert bad.status == 400 and "comparison" in bad.json()["error"]

        r = client.post(
            "/api/jobs", {"live": True, "detector_id": dets[0]["id"], "pose_id": poses[0]["id"]}
        )
        assert r.status == 200 and r.json()["live"] is True
        job_id = r.json()["job_id"]
        snap = client.get(f"/api/jobs/{job_id}").json()
        assert snap["live"] is True
        assert snap["source_label"] == "live camera"
        assert snap["n_total"] is None
        assert snap["max_s"] == demo_live.LIVE_MAX_S_DEFAULT
        assert set(snap) >= JOB_CONTRACT_KEYS

        # the frames websocket feeds the registered source even while the job is queued
        client.ws_push_bytes(f"/ws/jobs/{job_id}/frames", [b"jpegbytes"])
        assert mgr._live_sources[job_id].n_received == 1

        # unknown/non-live jobs: the frames socket closes 4404
        messages = client.ws_push_bytes("/ws/jobs/deadbeef/frames", [b"x"])
        assert messages[-1]["type"] == "websocket.close"
        assert messages[-1].get("code") == 4404

        assert client.post(f"/api/jobs/{job_id}/stop").status == 200
        assert mgr._live_sources == {} or mgr._live_sources[job_id].stopped

        plain = client.post(
            "/api/jobs", {"sample": True, "detector_id": dets[0]["id"], "pose_id": poses[0]["id"]}
        )
        assert client.post(f"/api/jobs/{plain.json()['job_id']}/stop").status == 409
        assert client.post("/api/jobs/deadbeef/stop").status == 409
    finally:
        gate.set()
        mgr.shutdown()


def test_page_carries_live_camera_controls(demo_app) -> None:
    client, _ = demo_app
    html = client.get("/").body.decode()
    markers = (
        "Live camera",
        "getUserMedia",
        "/frames",
        "Stop session",
    )
    for marker in markers:
        assert marker in html, marker


def test_page_is_self_contained_and_branded(demo_app) -> None:
    """The brand mark is inlined (zero-network-assets rail) and the template
    placeholder never leaks to the browser."""
    client, _ = demo_app
    html = client.get("/").body.decode()
    assert "__LOGO__" not in html
    assert 'src="data:image/png;base64,' in html
    assert "http://" not in html and "https://" not in html  # no external asset URLs


# -------------------------------------------------------------- live: slow e2e
@pytest.mark.slow
def test_live_session_end_to_end(tmp_path: Path) -> None:
    """Real models, synthetic camera: sample-clip frames pushed the way a browser pushes
    them. It asserts the watch-only property directly, that a finished live session leaves
    no imagery behind at all."""
    cv2 = pytest.importorskip("cv2")
    mgr = demo_jobs.JobManager(REPO_ROOT, outputs_root=tmp_path)
    try:
        detector_id, pose_id = _runnable_pair(mgr)
        job = mgr.create_live_job(
            detector_id=detector_id,
            pose_id=pose_id,
            rules_profile="demo",
        )
        src = mgr._live_sources[job.job_id]
        cap = cv2.VideoCapture(str(REPO_ROOT / "tests" / "fixtures" / "sample_clip.mp4"))
        payloads = []
        try:
            while len(payloads) < 12:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                ok2, jpeg = cv2.imencode(".jpg", frame)
                assert ok2
                payloads.append(jpeg.tobytes())
        finally:
            cap.release()

        deadline = time.time() + 90  # model load + warmup happen inside this window
        i = 0
        while time.time() < deadline and job.frames_done < 5:
            src.push(payloads[i % len(payloads)])
            i += 1
            time.sleep(0.1)
        assert job.frames_done >= 1, f"status={job.status} error={job.error}"

        assert mgr.stop_live(job.job_id)
        deadline = time.time() + 30
        while job.status == "running" and time.time() < deadline:
            time.sleep(0.2)
        assert job.status == "done", f"status={job.status} error={job.error}"
        assert not (job.job_dir / "annotated.mp4").exists()
        assert not (job.job_dir / "thumb.jpg").exists()
        assert not list(job.job_dir.glob("*.jpg")) and not list(job.job_dir.glob("*.mp4"))
        snap = job.snapshot()
        assert snap["live"] is True and snap["annotated_ready"] is False
        assert (snap["video"] or {}).get("frames_received", 0) >= 1
    finally:
        mgr.shutdown()


# -------------------------------------------------------------- jobs: slow e2e
@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_job_end_to_end_on_sample_clip(tmp_path: Path) -> None:
    mgr = demo_jobs.JobManager(REPO_ROOT, outputs_root=tmp_path)
    try:
        detector_id, pose_id = _runnable_pair(mgr)
        job = mgr.create_job(
            source_path=REPO_ROOT / "tests" / "fixtures" / "sample_clip.mp4",
            source_label="sample_clip.mp4",
            move_source=False,
            detector_id=detector_id,
            pose_id=pose_id,
            score_thr=0.4,
            rules_profile="demo",
        )
        deadline = time.time() + 300
        while time.time() < deadline and job.status not in {"done", "failed", "cancelled"}:
            time.sleep(0.5)
        assert job.status == "done", f"status={job.status} error={job.error}"
        annotated = job.job_dir / "annotated.mp4"
        assert annotated.is_file()

        from vividly.media.ffmpeg import probe_video

        info = probe_video(annotated)
        assert info["codec_name"] == "h264"
        assert info["nb_frames"] == 20
        assert (job.job_dir / "job.json").is_file()
        assert (job.job_dir / "thumb.jpg").is_file()
        snap = job.snapshot()
        assert snap["annotated_ready"] is True
        assert snap["latency_p50_ms"], "latency summary missing"
    finally:
        mgr.shutdown()


# -------------------------------------------------------------- egress zones: preview frame
def test_preview_frame_argv_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIVIDLY_FFMPEG", "/golden/ffmpeg")
    argv = demo_media.preview_frame_argv(Path("in.mov"), Path("preview.jpg"), seek_s=1.0)
    assert argv == [
        "/golden/ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        "1",
        "-i",
        "in.mov",
        "-vf",
        "scale='min(1280,iw)':'min(1280,ih)':force_original_aspect_ratio=decrease"
        ":force_divisible_by=2",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        "preview.jpg",
    ]
    argv0 = demo_media.preview_frame_argv(Path("in.mov"), Path("p.jpg"), seek_s=0.0)
    assert "-ss" not in argv0  # a clip shorter than the seek falls back to frame 0

    # the preview's scale filter is normalize's scale filter (minus the fps resample):
    # the zone-drawing pixel space cannot drift from the pipeline's processing space
    preview_vf = argv[argv.index("-vf") + 1]
    norm = demo_media.normalize_argv(Path("a.mp4"), Path("b.mp4"))
    assert norm[norm.index("-vf") + 1] == f"{preview_vf},fps=15"


def _sof_jpeg(width: int, height: int) -> bytes:
    """A minimal JPEG header: SOI, one APP0, one SOF0 carrying the dimensions."""
    return (
        b"\xff\xd8"
        + b"\xff\xe0"
        + (16).to_bytes(2, "big")
        + b"\x00" * 14
        + b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03"
        + b"\x00" * 9
    )


def test_jpeg_dimensions_parses_sof_header() -> None:
    assert demo_media.jpeg_dimensions(_sof_jpeg(640, 360)) == (640, 360)
    with pytest.raises(ValueError, match="SOI"):
        demo_media.jpeg_dimensions(b"PNG not jpeg")
    with pytest.raises(ValueError, match="SOF"):
        demo_media.jpeg_dimensions(b"\xff\xd8\xff\xe0" + (16).to_bytes(2, "big") + b"\x00" * 14)


def test_jpeg_dimensions_agrees_with_cv2() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    ok, jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    assert demo_media.jpeg_dimensions(jpeg.tobytes()) == (64, 48)


def test_ensure_preview_serves_cache_without_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is the contract: once written, dims come from the server's own record
    and no ffmpeg run happens (a broken binary here proves it)."""
    monkeypatch.setenv("VIVIDLY_FFMPEG", str(tmp_path / "no-such-ffmpeg"))
    (tmp_path / "preview.jpg").write_bytes(_sof_jpeg(320, 240))
    (tmp_path / "preview.json").write_text(json.dumps({"width": 320, "height": 240}))
    info = demo_media.ensure_preview(tmp_path / "missing.mp4", tmp_path)
    assert info == {"width": 320, "height": 240}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ensure_preview_matches_normalized_geometry(tmp_path: Path) -> None:
    """The whole point of the preview: its dimensions equal the normalized video's, so
    zone coordinates drawn on it are valid pipeline coordinates."""
    from vividly.media.ffmpeg import probe_video, run_argv

    clip = REPO_ROOT / "tests" / "fixtures" / "sample_clip.mp4"
    info = demo_media.ensure_preview(clip, tmp_path)
    run_argv(demo_media.normalize_argv(clip, tmp_path / "norm.mp4"))
    probed = probe_video(tmp_path / "norm.mp4")
    assert (info["width"], info["height"]) == (probed["width"], probed["height"])
    assert json.loads((tmp_path / "preview.json").read_text()) == info

    # a corrupt cache is rebuilt, not trusted
    (tmp_path / "preview.json").write_text("not json")
    assert demo_media.ensure_preview(clip, tmp_path) == info


# -------------------------------------------------------------- egress zones: validation
def test_validate_zones_happy_path_is_canonical() -> None:
    zones = demo_jobs.validate_zones([{"name": "  exit ", "rect": [10, 20.55, 300, 200]}], 640, 360)
    assert zones == [{"name": "exit", "rect": [10.0, 20.6, 300.0, 200.0]}]
    # zero zones is a valid answer and means the feature is off
    assert demo_jobs.validate_zones([], 640, 360) == []
    # edge-touching is inside: the frame boundary is a legal zone edge
    full = demo_jobs.validate_zones([{"name": "exit", "rect": [0, 0, 640, 360]}], 640, 360)
    assert full[0]["rect"] == [0.0, 0.0, 640.0, 360.0]


def test_validate_zones_rejections() -> None:
    ok_rect = [10, 10, 100, 100]
    cases = [
        ("must be a list", {"name": "exit", "rect": ok_rect}),
        ("inverted", [{"name": "exit", "rect": [100, 10, 10, 100]}]),
        ("inverted", [{"name": "exit", "rect": [10, 100, 100, 100]}]),  # zero height
        ("outside", [{"name": "exit", "rect": [10, 10, 700, 100]}]),
        ("outside", [{"name": "exit", "rect": [-5, 10, 100, 100]}]),
        ("finite", [{"name": "exit", "rect": [10, 10, float("nan"), 100]}]),
        ("numbers", [{"name": "exit", "rect": [10, 10, "wat", 100]}]),
        ("rect must be", [{"name": "exit", "rect": [10, 10, 100]}]),
        ("name must be", [{"name": "", "rect": ok_rect}]),
        ("name must be", [{"name": "<img onerror=x>", "rect": ok_rect}]),  # innerHTML-safe
        ("name must be", [{"name": "x" * 41, "rect": ok_rect}]),
        ("duplicate", [{"name": "exit", "rect": ok_rect}, {"name": "exit", "rect": ok_rect}]),
        ("unknown keys", [{"name": "exit", "rect": ok_rect, "polygon": []}]),
        ("mapping", ["exit"]),
        (
            "at most",
            [{"name": f"z{i}", "rect": ok_rect} for i in range(demo_jobs.MAX_ZONES + 1)],
        ),
    ]
    for match, zones in cases:
        with pytest.raises(ValueError, match=match):
            demo_jobs.validate_zones(zones, 640, 360)
    with pytest.raises(ValueError, match="positive"):
        demo_jobs.validate_zones([{"name": "exit", "rect": ok_rect}], 0, 360)


def test_zone_entries_rect_to_polygon() -> None:
    entries = demo_jobs.zone_entries([{"name": "exit", "rect": [10.0, 20.0, 30.0, 40.0]}])
    assert entries == [
        {"name": "exit", "polygon": [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]}
    ]


def test_build_run_cfg_with_zones_golden(tmp_path: Path) -> None:
    """The exact egress block a zoned demo job hands scripts/run_pipeline.py, plus the
    demo-calibration reap shrink. No zones must stay byte-identical to before."""
    det = _fake_entry("detector", "eyes", "40262049", (640, 640), backend="rtdetr_onnx")
    pose = _fake_entry("pose", "joints", "6ba96285", (192, 256), backend="rtmlib")
    common = {
        "job_id": "j1",
        "runs_dir": tmp_path,
        "input_mp4": tmp_path / "input.mp4",
        "detector": det,
        "pose": pose,
        "score_thr": 0.4,
        "repo_root": REPO_ROOT,
        "fps": 15.0,
    }
    zones = [{"name": "main door", "rect": [1000.0, 100.0, 1280.0, 500.0]}]
    cfg = demo_jobs.build_run_cfg(rules_profile="demo", zones=zones, **common)
    assert cfg["egress"] == {
        "enabled": True,
        "zones": [
            {
                "name": "main door",
                "polygon": [
                    [1000.0, 100.0],
                    [1280.0, 100.0],
                    [1280.0, 500.0],
                    [1000.0, 500.0],
                ],
            }
        ],
    }
    # the emitted block parses through the pipeline's real schema
    from vividly.rules.egress import EgressZones

    parsed = EgressZones.from_mapping(cfg["egress"])
    assert parsed.enabled and [z.name for z in parsed.zones] == ["main door"]
    # demo calibration shrinks the reap (a departure is only noticed at reap time)
    assert cfg["tracks"]["reap_after_s"] == demo_jobs.EGRESS_DEMO_REAP_AFTER_S
    assert cfg["rules"]["egress_absent_min_s"] == 20.0

    # clinical profiles carry the zones but keep every clinical number untouched
    cfg_prod = demo_jobs.build_run_cfg(rules_profile="production", zones=zones, **common)
    assert cfg_prod["egress"]["enabled"] is True
    assert "reap_after_s" not in cfg_prod["tracks"]
    assert cfg_prod["rules_config"].endswith("configs/pipeline/rules.yaml")

    # zero zones = exactly today's behavior: no egress key, no tracks override
    cfg_none = demo_jobs.build_run_cfg(rules_profile="demo", **common)
    assert "egress" not in cfg_none
    assert "reap_after_s" not in cfg_none["tracks"]
    assert cfg_none == demo_jobs.build_run_cfg(rules_profile="demo", zones=[], **common)


def test_zones_persist_in_job_json_roundtrip(tmp_path: Path) -> None:
    zones = [{"name": "exit", "rect": [10.0, 20.0, 300.0, 200.0]}]
    job = demo_jobs.JobState(
        "job1234x",
        tmp_path,
        source_label="clip.mp4",
        detector={"display_name": "d", "sha8": "aaaa1111"},
        pose={"display_name": "p", "sha8": "bbbb2222"},
        score_thr=0.4,
        rules_profile="demo",
        max_s=None,
        zones=zones,
    )
    assert job.snapshot()["zones"] == zones
    job.set_status("done")
    job.persist()
    assert json.loads((tmp_path / "job.json").read_text())["zones"] == zones
    restored = demo_jobs.JobState.from_json(tmp_path)
    assert restored is not None and restored.zones == zones
    # legacy job.json without the key restores to no zones
    row = json.loads((tmp_path / "job.json").read_text())
    del row["zones"]
    (tmp_path / "job.json").write_text(json.dumps(row))
    legacy = demo_jobs.JobState.from_json(tmp_path)
    assert legacy is not None and legacy.zones == []


def test_frame_hook_scene_carries_zone_polygons(tmp_path: Path) -> None:
    job = _mk_job(tmp_path)
    job.zones = [{"name": "exit", "rect": [10.0, 20.0, 30.0, 40.0]}]
    hook = demo_jobs.DemoFrameHook(job, _FakeWriter(), "demo rules")
    scene = hook.scene(_fake_pipeline(now=1.0))
    assert scene["zones"] == [
        {"name": "exit", "polygon": [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]}
    ]
    # zoneless jobs draw nothing new
    bare = demo_jobs.DemoFrameHook(_mk_job(tmp_path / "b"), _FakeWriter(), "demo rules")
    assert bare.scene(_fake_pipeline(now=1.0))["zones"] == []


# -------------------------------------------------------------- egress zones: overlay
def _zone_scene() -> dict:
    return {
        "tracks": [],
        "banners": [],
        "zones": [
            {
                "name": "exit",
                "polygon": [[40.0, 60.0], [200.0, 60.0], [200.0, 180.0], [40.0, 180.0]],
            }
        ],
    }


def test_overlay_zone_outline_is_strictly_monochrome() -> None:
    """The zone overlay is white and black strokes only. Drawn on mid-grey, every touched
    pixel must be pure 0 or 255 and grey, with no antialiasing and no colour."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from vividly.demo import overlay

    img = np.full((240, 320, 3), 127, np.uint8)
    overlay.draw_frame(img, _zone_scene(), {})
    touched = img[(img != 127).any(axis=2)]
    assert touched.size, "zone drew nothing"
    assert set(np.unique(touched).tolist()) <= {0, 127, 255}
    assert (touched[:, 0] == touched[:, 1]).all() and (touched[:, 1] == touched[:, 2]).all()
    # outline present along the top edge, label chip above it
    assert img[58:63, 40:200].any()
    assert (img[40:60, 40:150] == 255).any(), "label text missing"


def test_overlay_zone_drawing_is_deterministic() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from vividly.demo import overlay

    a = overlay.draw_frame(np.full((240, 320, 3), 90, np.uint8), _zone_scene(), {})
    b = overlay.draw_frame(np.full((240, 320, 3), 90, np.uint8), _zone_scene(), {})
    assert np.array_equal(a, b)


def test_overlay_zone_degenerate_and_absent_zones() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from vividly.demo import overlay

    img = np.zeros((100, 100, 3), np.uint8)
    scene = {
        "tracks": [],
        "banners": [],
        "zones": [
            {"name": "thin", "polygon": [[10.0, 10.0], [20.0, 10.0]]},  # <3 vertices: skipped
            {"name": "off", "polygon": [[300.0, 300.0], [400.0, 300.0], [400.0, 400.0]]},
        ],
    }
    overlay.draw_frame(img, scene, {})  # must not raise; cv2 clips off-frame edges
    # a scene without the zones key (every pre-zone caller) is untouched behavior
    before = np.zeros((50, 50, 3), np.uint8)
    overlay.draw_frame(before, {"tracks": [], "banners": []}, {})
    assert not before.any()


# -------------------------------------------------------------- egress zones: app routes
def test_upload_preview_route_404s_and_serves_cache(demo_app) -> None:
    client, mgr = demo_app
    assert client.get("/api/uploads/nothexid/preview.jpg").status == 404
    assert client.get("/api/uploads/00000000/preview.jpg").status == 404

    up = client.post("/api/upload?filename=a.mp4", raw=b"notavideo").json()
    updir = mgr.dirs["incoming"] / up["upload_id"]
    # a cached preview serves as-is; no ffmpeg run happens on the cache path
    (updir / "preview.jpg").write_bytes(_sof_jpeg(64, 48))
    (updir / "preview.json").write_text(json.dumps({"width": 64, "height": 48}))
    r = client.get(f"/api/uploads/{up['upload_id']}/preview.jpg")
    assert r.status == 200 and r.body == _sof_jpeg(64, 48)

    # an upload ffmpeg cannot decode reports 422, not a traceback
    up2 = client.post("/api/upload?filename=b.mp4", raw=b"still not a video").json()
    r2 = client.get(f"/api/uploads/{up2['upload_id']}/preview.jpg")
    assert r2.status == 422 and "preview" in r2.json()["error"]


def test_job_zone_validation_at_the_api(demo_app) -> None:
    """Every rejection here fires before model validation, so no graphs are needed."""
    client, mgr = demo_app
    zones = [{"name": "exit", "rect": [10, 10, 100, 100]}]

    r = client.post("/api/jobs", {"live": True, "zones": zones, "detector_id": "x", "pose_id": "y"})
    assert r.status == 400 and "exit zones" in r.json()["error"]
    r = client.post(
        "/api/jobs", {"sample": True, "zones": zones, "detector_id": "x", "pose_id": "y"}
    )
    assert r.status == 400 and "uploaded clip" in r.json()["error"]

    up = client.post("/api/upload?filename=clip.mp4", raw=b"fake-bytes").json()
    payload = {"upload_id": up["upload_id"], "detector_id": "x", "pose_id": "y"}
    # zones with no preview cache: bounds cannot be checked, so the job is refused
    r = client.post("/api/jobs", {**payload, "zones": zones})
    assert r.status == 400 and "no preview" in r.json()["error"]

    updir = mgr.dirs["incoming"] / up["upload_id"]
    (updir / "preview.jpg").write_bytes(_sof_jpeg(640, 360))
    (updir / "preview.json").write_text(json.dumps({"width": 640, "height": 360}))
    bad = [{"name": "exit", "rect": [10, 10, 700, 100]}]
    r = client.post("/api/jobs", {**payload, "zones": bad})
    assert r.status == 400 and "outside" in r.json()["error"]
    inverted = [{"name": "exit", "rect": [100, 100, 10, 200]}]
    r = client.post("/api/jobs", {**payload, "zones": inverted})
    assert r.status == 400 and "inverted" in r.json()["error"]
    # in-range zones pass zone validation: the next failure is the unknown detector
    r = client.post("/api/jobs", {**payload, "zones": zones})
    assert r.status == 400 and "unknown detector" in r.json()["error"]


def test_job_with_zones_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vividly.demo.app import create_demo_app

    monkeypatch.setattr(demo_jobs.JobManager, "_run_job", lambda self, job: job.set_status("done"))
    mgr = demo_jobs.JobManager(REPO_ROOT, outputs_root=tmp_path)
    try:
        cat = mgr.catalog()
        dets, poses = _enabled(cat["detectors"]), _enabled(cat["poses"])
        if not dets or not poses:
            pytest.skip("model record graphs not on disk")
        client = AsgiClient(create_demo_app(mgr, max_upload_mb=1))

        up = client.post("/api/upload?filename=clip.mp4", raw=b"fake-video-bytes").json()
        updir = mgr.dirs["incoming"] / up["upload_id"]
        (updir / "preview.jpg").write_bytes(_sof_jpeg(640, 360))
        (updir / "preview.json").write_text(json.dumps({"width": 640, "height": 360}))
        zones = [{"name": "main door", "rect": [400, 40, 640, 300]}]
        body = client.post(
            "/api/jobs",
            {
                "upload_id": up["upload_id"],
                "detector_id": dets[0]["id"],
                "pose_id": poses[0]["id"],
                "zones": zones,
            },
        ).json()
        job = mgr.get(body["job_id"])
        canonical = [{"name": "main door", "rect": [400.0, 40.0, 640.0, 300.0]}]
        assert job.zones == canonical
        # the preview cache never leaks into the job: the moved source is the video
        assert (job.job_dir / "original.mp4").read_bytes() == b"fake-video-bytes"
        # unchanged in job.json and in the snapshot the page polls
        assert json.loads((job.job_dir / "job.json").read_text())["zones"] == canonical
        snap = client.get(f"/api/jobs/{body['job_id']}").json()
        assert snap["zones"] == canonical
    finally:
        mgr.shutdown()


def test_page_carries_zone_controls(demo_app) -> None:
    client, _ = demo_app
    html = client.get("/").body.decode()
    markers = (
        "zonecanvas",
        "Exit zone",
        "egress_without_discharge_v0",
        "Egress without discharge",
        "Clear zone",
        "preview.jpg",
        "zonename",
    )
    for marker in markers:
        assert marker in html, marker
    assert "detects seizures" not in html


@needs_model_record
def test_catalog_against_the_model_record() -> None:
    """The shipped record read through the picker's own contract: the eyes lane is an
    RT-DETRv4 graph at 640x640, the joints lane an RTMPose graph at 192x256, both
    prototype weights, and any graph present on this machine matches its recorded
    digest."""
    cat = demo_catalog.scan_catalog(REPO_ROOT)
    lanes = {e["lane"]: e for e in [*cat["detectors"], *cat["poses"]]}
    assert "eyes" in lanes and "joints" in lanes, sorted(lanes)

    eyes = lanes["eyes"]
    assert eyes["kind"] == "detector"
    assert eyes["backend"] == "rtdetr_onnx"
    assert eyes["input_size"] == [640, 640]
    assert eyes["badge"] == "prototype"
    assert eyes["id"] == f"eyes/{eyes['sha8']}"

    joints = lanes["joints"]
    assert joints["kind"] == "pose"
    assert joints["backend"] == "rtmlib"
    assert joints["input_size"] == [192, 256]
    assert joints["badge"] == "prototype"

    for entry in lanes.values():
        if entry["onnx_present"]:
            assert entry["sha256_verified"], entry["id"]
            assert entry["disabled"] is None

    assert cat["detectors"][0]["badge"] == "prototype"  # picker order
