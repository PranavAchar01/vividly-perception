"""Source-layer tests.

Two import rules hold throughout, because breaking either one turns a run with no numpy and
no cv2 installed red while the Mac stays green:

  * numpy and cv2 are darwin-marked in this repo's default `mac` group. Every test that
    needs one calls pytest.importorskip first.
  * every module under src/vividly/sources must import with neither present, which
    test_source_modules_have_no_top_level_numpy_or_cv2 pins at the AST level.

No test here loads a model, touches a network, or reads a fixture off disk that is not
written by the test itself. The replay clip is generated with cv2.VideoWriter and its
pixels encode the frame index, so seek exactness is verifiable without a checked-in
video. Anything needing real detector weights belongs in the acceptance run, not here.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from vividly.sources import (
    RTSP_NOT_IMPLEMENTED,
    Frame,
    FrameSource,
    RTSPSource,
    SourceInfo,
    SyntheticDetectionSource,
    pacing_delay,
    write_observations_jsonl,
)
from vividly.sources.replay import ReplaySource
from vividly.tracks.state import TrackObservation

REPO = Path(__file__).resolve().parent.parent
SOURCES_SRC = REPO / "src" / "vividly" / "sources"

CLIP_FPS = 10.0
CLIP_N = 12
CLIP_W = 64
CLIP_H = 48


# --------------------------------------------------------------- clip generation
def _gray_for_index(index: int) -> int:
    """Pixel level that encodes a frame index. Levels are 10 apart so lossy encoding
    cannot move a frame onto its neighbour's level."""
    return 10 * index + 5


def _index_from_frame(image: Any) -> int:
    """Recover the frame index from a decoded frame's central patch."""
    patch = image[CLIP_H // 4 : 3 * CLIP_H // 4, CLIP_W // 4 : 3 * CLIP_W // 4]
    return round((float(patch.mean()) - 5.0) / 10.0)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny synthetic clip whose frame N is a solid grey at level 10N + 5."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path_factory.mktemp("replay") / "counted.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (CLIP_W, CLIP_H))
    assert writer.isOpened(), "cv2.VideoWriter could not open an mp4v writer"
    for index in range(CLIP_N):
        writer.write(np.full((CLIP_H, CLIP_W, 3), _gray_for_index(index), dtype=np.uint8))
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


class FakeClock:
    """Deterministic stand-in for time.perf_counter plus time.sleep.

    Sleeping advances the clock by exactly the requested amount, so a paced loop can be
    exercised end to end without a wall-clock assertion.
    """

    def __init__(self, step_per_read: float = 0.0) -> None:
        self.now = 0.0
        self.step_per_read = step_per_read
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        value = self.now
        self.now += self.step_per_read
        return value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# --------------------------------------------------------------- contract shape
def test_source_info_fields_match_contract() -> None:
    names = [f.name for f in dataclasses.fields(SourceInfo)]
    assert names == ["fps", "width", "height", "n_frames"]
    info = SourceInfo(fps=15.0, width=1920, height=1080, n_frames=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.fps = 30.0  # type: ignore[misc]


def test_frame_fields_match_contract() -> None:
    names = [f.name for f in dataclasses.fields(Frame)]
    assert names == ["index", "t_source", "image", "wall_utc"]
    frame = Frame(index=3, t_source=0.2, image=None)  # type: ignore[arg-type]
    frame.index = 4  # Frame is mutable by contract; this must not raise
    assert frame.index == 4
    assert frame.wall_utc is None  # file sources carry no wall clock (design P4)


def test_frame_source_protocol_exposes_the_four_lifecycle_methods() -> None:
    for name in ("open", "frames", "seek", "close"):
        assert hasattr(FrameSource, name), name


def test_replay_source_satisfies_the_frame_source_protocol() -> None:
    source = ReplaySource("does-not-need-to-exist.mp4")
    assert isinstance(source, FrameSource)


def test_synthetic_detection_source_is_not_a_frame_source() -> None:
    """It yields observations, never images. Anything that treats it as a FrameSource
    is a wiring bug, so the negative is worth pinning."""
    source = SyntheticDetectionSource("fixture.jsonl")
    assert not isinstance(source, FrameSource)
    assert not hasattr(source, "frames")


# --------------------------------------------------------------- RTSP stub
def test_rtsp_source_stores_its_url_and_opens_nothing() -> None:
    source = RTSPSource("rtsp://camera.invalid/stream1")
    assert source.url == "rtsp://camera.invalid/stream1"


@pytest.mark.parametrize(
    ("method", "args"),
    [("open", ()), ("frames", ()), ("seek", (0,)), ("close", ())],
)
def test_rtsp_every_method_raises_the_contract_message(method: str, args: tuple) -> None:
    source = RTSPSource("rtsp://camera.invalid/stream1")
    with pytest.raises(NotImplementedError) as excinfo:
        getattr(source, method)(*args)
    assert str(excinfo.value) == RTSP_NOT_IMPLEMENTED
    assert "vividly.sources.rtsp.LiveRTSPSource" in str(excinfo.value)


# --------------------------------------------------------------- CI portability
def test_source_modules_have_no_top_level_numpy_or_cv2() -> None:
    """A Linux environment synced without the harness group has neither cv2 nor numpy, and
    it still collects this package.

    Imports inside a function body are fine. Imports inside `if TYPE_CHECKING:` are fine.
    A module-level `import cv2` breaks that collection, so it is an AST-level failure here.
    """
    offenders: list[str] = []
    for py in sorted(SOURCES_SRC.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in _module_level_statements(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name in {"cv2", "numpy", "torch", "rtmlib", "trackeval"}:
                    offenders.append(f"{py.name}:{node.lineno}: {name}")
    assert not offenders, f"module-level heavy imports: {offenders}"


def _module_level_statements(tree: ast.Module) -> list[ast.stmt]:
    """Top-level statements, skipping the bodies of `if TYPE_CHECKING:` blocks."""
    out: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            continue
        out.append(node)
    return out


# --------------------------------------------------------------- pacing arithmetic
def test_pacing_delay_is_zero_when_rate_is_none() -> None:
    assert pacing_delay(None, CLIP_FPS, 100, 0.0) == 0.0


def test_pacing_delay_at_realtime_targets_the_frame_interval() -> None:
    assert pacing_delay(1.0, 10.0, 1, 0.0) == pytest.approx(0.1)
    assert pacing_delay(1.0, 10.0, 5, 0.0) == pytest.approx(0.5)
    assert pacing_delay(1.0, 10.0, 5, 0.2) == pytest.approx(0.3)


def test_pacing_delay_never_goes_negative_when_playback_is_behind() -> None:
    assert pacing_delay(1.0, 10.0, 5, 9.0) == 0.0


def test_pacing_delay_scales_inversely_with_rate() -> None:
    assert pacing_delay(2.0, 10.0, 4, 0.0) == pytest.approx(0.2)
    assert pacing_delay(0.5, 10.0, 4, 0.0) == pytest.approx(0.8)


def test_pacing_delay_is_zero_for_a_source_without_a_frame_rate() -> None:
    assert pacing_delay(1.0, 0.0, 4, 0.0) == 0.0


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_replay_rejects_a_non_positive_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="rate must be positive or None"):
        ReplaySource("clip.mp4", rate)


# --------------------------------------------------------------- replay reading
def test_open_reports_the_clip_geometry(clip: Path) -> None:
    source = ReplaySource(clip)
    info = source.open()
    assert info.fps == pytest.approx(CLIP_FPS)
    assert (info.width, info.height) == (CLIP_W, CLIP_H)
    assert info.n_frames == CLIP_N
    assert source.open() is info  # idempotent while open
    source.close()


def test_missing_clip_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="replay clip not found"):
        ReplaySource(tmp_path / "nope.mp4").open()


def test_frames_yields_every_index_in_order_with_exact_t_source(clip: Path) -> None:
    source = ReplaySource(clip)
    got = [(f.index, f.t_source) for f in source.frames()]
    source.close()
    assert [index for index, _ in got] == list(range(CLIP_N))
    assert all(t == index / CLIP_FPS for index, t in got), "t_source must be index / fps exactly"


def test_frame_pixels_encode_their_own_index(clip: Path) -> None:
    """Proves the clip is a usable ruler before any seek test relies on it."""
    source = ReplaySource(clip)
    decoded = [_index_from_frame(f.image) for f in source.frames()]
    source.close()
    assert decoded == list(range(CLIP_N))


@pytest.mark.parametrize("target", [0, 1, 7, CLIP_N - 1])
def test_seek_is_frame_exact(clip: Path, target: int) -> None:
    source = ReplaySource(clip)
    source.open()
    source.seek(target)
    frame = next(iter(source.frames()))
    source.close()
    assert frame.index == target
    assert _index_from_frame(frame.image) == target
    assert frame.t_source == target / CLIP_FPS


def test_seek_backwards_replays_the_same_frames(clip: Path) -> None:
    source = ReplaySource(clip)
    source.open()
    source.seek(9)
    first = next(iter(source.frames()))
    source.seek(2)
    rewound = [(f.index, _index_from_frame(f.image)) for f in source.frames()]
    source.close()
    assert (first.index, _index_from_frame(first.image)) == (9, 9)
    assert rewound == [(i, i) for i in range(2, CLIP_N)]


def test_decode_fallback_seek_is_also_frame_exact(clip: Path) -> None:
    """Backends whose CAP_PROP_POS_FRAMES set is advisory take this path."""
    source = ReplaySource(clip)
    source.open()
    source._seek_by_decode(source._cap, 6)
    source._next_index = 6
    frame = next(iter(source.frames()))
    source.close()
    assert (frame.index, _index_from_frame(frame.image)) == (6, 6)


def test_seek_past_the_end_ends_iteration_rather_than_raising(clip: Path) -> None:
    source = ReplaySource(clip)
    source.open()
    source.seek(CLIP_N + 50)
    assert list(source.frames()) == []
    source.close()


def test_negative_seek_raises(clip: Path) -> None:
    source = ReplaySource(clip)
    with pytest.raises(ValueError, match="seek index must be >= 0"):
        source.seek(-1)
    source.close()


def test_frames_opens_the_clip_by_itself(clip: Path) -> None:
    source = ReplaySource(clip)
    assert source.info is None
    first = next(iter(source.frames()))
    assert source.info is not None
    assert first.index == 0
    source.close()


def test_close_is_idempotent_and_the_source_reopens(clip: Path) -> None:
    source = ReplaySource(clip)
    source.open()
    source.close()
    source.close()
    assert source.info is None
    assert source.open().n_frames == CLIP_N
    source.close()


def test_close_mid_iteration_stops_the_generator(clip: Path) -> None:
    source = ReplaySource(clip)
    stream = source.frames()
    assert next(stream).index == 0
    source.close()
    assert list(stream) == []


def test_context_manager_closes_the_capture(clip: Path) -> None:
    with ReplaySource(clip) as source:
        assert source.info is not None
    assert source.info is None


# --------------------------------------------------------------- replay pacing
def test_rate_none_never_sleeps(clip: Path) -> None:
    fake = FakeClock()
    source = ReplaySource(clip, None, clock=fake, sleep=fake.sleep)
    assert len(list(source.frames())) == CLIP_N
    source.close()
    assert fake.sleeps == []


def test_realtime_pacing_sleeps_one_frame_interval_per_frame(clip: Path) -> None:
    fake = FakeClock()
    source = ReplaySource(clip, 1.0, clock=fake, sleep=fake.sleep)
    assert len(list(source.frames())) == CLIP_N
    source.close()
    # The first frame is the pacing epoch and waits for nothing; every later frame waits
    # exactly one frame interval, because sleeping is the only thing that advances the
    # fake clock. No wall-clock assertion anywhere.
    assert fake.sleeps == pytest.approx([1.0 / CLIP_FPS] * (CLIP_N - 1))


def test_pacing_stops_waiting_when_decoding_is_slower_than_realtime(clip: Path) -> None:
    """A clock that jumps a full second per read is always behind, so nothing sleeps."""
    fake = FakeClock(step_per_read=1.0)
    source = ReplaySource(clip, 1.0, clock=fake, sleep=fake.sleep)
    assert len(list(source.frames())) == CLIP_N
    source.close()
    assert fake.sleeps == []


def test_seek_restarts_the_pacing_epoch(clip: Path) -> None:
    """After a scrub, playback must not sprint to catch up with the old epoch."""
    fake = FakeClock()
    source = ReplaySource(clip, 1.0, clock=fake, sleep=fake.sleep)
    stream = source.frames()
    assert [next(stream).index for _ in range(3)] == [0, 1, 2]
    assert len(fake.sleeps) == 2
    source.seek(8)
    frame = next(stream)
    source.close()
    assert frame.index == 8
    assert len(fake.sleeps) == 2, "the frame that starts a new epoch waits for nothing"


# --------------------------------------------------------------- synthetic source
def _kp(value: float = 1.0) -> list[list[float]]:
    return [[value, value + 1.0, 0.9] for _ in range(17)]


def _write_lines(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_synthetic_yields_track_observations_per_line(tmp_path: Path) -> None:
    fixture = _write_lines(
        tmp_path / "f.jsonl",
        [
            {
                "t": 0.0,
                "observations": [
                    {
                        "track_id": 1,
                        "cx": 10.0,
                        "cy": 20.0,
                        "w": 30.0,
                        "h": 40.0,
                        "keypoints": None,
                    },
                    {
                        "track_id": 2,
                        "cx": 11.0,
                        "cy": 21.0,
                        "w": 31.0,
                        "h": 41.0,
                        "keypoints": None,
                    },
                ],
            },
            {
                "t": 0.5,
                "observations": [
                    {"track_id": 1, "cx": 12.0, "cy": 22.0, "w": 30.0, "h": 40.0, "keypoints": None}
                ],
            },
        ],
    )
    batches = list(SyntheticDetectionSource(fixture).observations())
    assert [len(b) for b in batches] == [2, 1]
    assert all(isinstance(obs, TrackObservation) for batch in batches for obs in batch)
    first = batches[0][0]
    assert (first.track_id, first.t, first.cx, first.cy, first.w, first.h) == (
        1,
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
    )
    assert batches[1][0].t == 0.5


def test_synthetic_line_t_becomes_every_observation_t(tmp_path: Path) -> None:
    fixture = _write_lines(
        tmp_path / "f.jsonl",
        [
            {
                "t": 7.25,
                "observations": [
                    {"track_id": i, "cx": 1.0, "cy": 2.0, "w": 3.0, "h": 4.0, "keypoints": None}
                    for i in range(3)
                ],
            }
        ],
    )
    (batch,) = list(SyntheticDetectionSource(fixture).observations())
    assert {obs.t for obs in batch} == {7.25}


def test_synthetic_keypoints_null_becomes_none(tmp_path: Path) -> None:
    fixture = _write_lines(
        tmp_path / "f.jsonl",
        [
            {
                "t": 0.0,
                "observations": [
                    {"track_id": 1, "cx": 1.0, "cy": 2.0, "w": 3.0, "h": 4.0, "keypoints": None}
                ],
            }
        ],
    )
    (batch,) = list(SyntheticDetectionSource(fixture).observations())
    assert batch[0].keypoints is None


def test_synthetic_keypoints_become_seventeen_triples(tmp_path: Path) -> None:
    fixture = _write_lines(
        tmp_path / "f.jsonl",
        [
            {
                "t": 0.0,
                "observations": [
                    {"track_id": 1, "cx": 1.0, "cy": 2.0, "w": 3.0, "h": 4.0, "keypoints": _kp(5.0)}
                ],
            }
        ],
    )
    (batch,) = list(SyntheticDetectionSource(fixture).observations())
    kps = batch[0].keypoints
    assert kps is not None
    assert len(kps) == 17
    assert all(isinstance(kp, tuple) and len(kp) == 3 for kp in kps)
    assert kps[0] == (5.0, 6.0, 0.9)


def test_synthetic_empty_observation_list_yields_an_empty_batch(tmp_path: Path) -> None:
    """An empty waiting room is a real state, not a skipped frame."""
    fixture = _write_lines(tmp_path / "f.jsonl", [{"t": 0.0, "observations": []}])
    assert list(SyntheticDetectionSource(fixture).observations()) == [[]]


def test_synthetic_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    path.write_text(
        '{"t": 0.0, "observations": []}\n\n   \n{"t": 1.0, "observations": []}\n',
        encoding="utf-8",
    )
    assert list(SyntheticDetectionSource(path).observations()) == [[], []]


def test_synthetic_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="synthetic detection fixture not found"):
        list(SyntheticDetectionSource(tmp_path / "nope.jsonl").observations())


def test_synthetic_bad_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    path.write_text('{"t": 0.0, "observations": []}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"f\.jsonl:2: not valid JSON"):
        list(SyntheticDetectionSource(path).observations())


@pytest.mark.parametrize("missing", ["t", "observations"])
def test_synthetic_missing_required_key_raises(tmp_path: Path, missing: str) -> None:
    record: dict[str, Any] = {"t": 0.0, "observations": []}
    record.pop(missing)
    fixture = _write_lines(tmp_path / "f.jsonl", [record])
    with pytest.raises(ValueError, match=f"missing required key '?{missing}'?"):
        list(SyntheticDetectionSource(fixture).observations())


def test_synthetic_observation_missing_a_box_field_raises(tmp_path: Path) -> None:
    fixture = _write_lines(
        tmp_path / "f.jsonl",
        [{"t": 0.0, "observations": [{"track_id": 1, "cx": 1.0, "cy": 2.0, "w": 3.0}]}],
    )
    with pytest.raises(ValueError, match="missing required key 'h'"):
        list(SyntheticDetectionSource(fixture).observations())


def test_synthetic_rejects_a_boolean_where_a_number_belongs(tmp_path: Path) -> None:
    fixture = _write_lines(
        tmp_path / "f.jsonl",
        [{"t": 0.0, "observations": [{"track_id": True, "cx": 1, "cy": 2, "w": 3, "h": 4}]}],
    )
    with pytest.raises(ValueError, match="'track_id' must be an integer"):
        list(SyntheticDetectionSource(fixture).observations())


@pytest.mark.parametrize("n_kp", [0, 16, 18])
def test_synthetic_rejects_a_wrong_keypoint_count(tmp_path: Path, n_kp: int) -> None:
    fixture = _write_lines(
        tmp_path / "f.jsonl",
        [
            {
                "t": 0.0,
                "observations": [
                    {
                        "track_id": 1,
                        "cx": 1.0,
                        "cy": 2.0,
                        "w": 3.0,
                        "h": 4.0,
                        "keypoints": _kp()[:n_kp] if n_kp <= 17 else [*_kp(), [0.0, 0.0, 0.0]],
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="17 COCO-17 triples"):
        list(SyntheticDetectionSource(fixture).observations())


def test_synthetic_rejects_a_malformed_keypoint_triple(tmp_path: Path) -> None:
    bad = [[1.0, 2.0, 0.5] for _ in range(16)] + [[1.0, 2.0]]
    fixture = _write_lines(
        tmp_path / "f.jsonl",
        [
            {
                "t": 0.0,
                "observations": [
                    {"track_id": 1, "cx": 1.0, "cy": 2.0, "w": 3.0, "h": 4.0, "keypoints": bad}
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match=r"keypoint 16 must be \[x, y, conf\]"):
        list(SyntheticDetectionSource(fixture).observations())


def test_synthetic_rejects_time_going_backwards(tmp_path: Path) -> None:
    fixture = _write_lines(
        tmp_path / "f.jsonl",
        [{"t": 1.0, "observations": []}, {"t": 0.5, "observations": []}],
    )
    with pytest.raises(ValueError, match="t went backwards"):
        list(SyntheticDetectionSource(fixture).observations())


def test_synthetic_allows_repeated_timestamps(tmp_path: Path) -> None:
    fixture = _write_lines(
        tmp_path / "f.jsonl",
        [{"t": 1.0, "observations": []}, {"t": 1.0, "observations": []}],
    )
    assert list(SyntheticDetectionSource(fixture).observations()) == [[], []]


def test_synthetic_observations_is_re_iterable(tmp_path: Path) -> None:
    """The pipeline reads a fixture once, but the dashboard demo loops it."""
    fixture = _write_lines(tmp_path / "f.jsonl", [{"t": 0.0, "observations": []}])
    source = SyntheticDetectionSource(fixture)
    assert list(source.observations()) == list(source.observations())


def test_write_observations_jsonl_round_trips(tmp_path: Path) -> None:
    batches = [
        [
            TrackObservation(track_id=1, t=0.0, cx=1.0, cy=2.0, w=3.0, h=4.0, keypoints=None),
            TrackObservation(
                track_id=2,
                t=0.0,
                cx=5.0,
                cy=6.0,
                w=7.0,
                h=8.0,
                keypoints=tuple((float(i), float(i) + 0.5, 0.8) for i in range(17)),
            ),
        ],
        [TrackObservation(track_id=1, t=0.5, cx=1.5, cy=2.5, w=3.0, h=4.0, keypoints=None)],
    ]
    path = write_observations_jsonl(tmp_path / "out" / "f.jsonl", batches)
    assert path.exists()
    assert list(SyntheticDetectionSource(path).observations()) == batches
