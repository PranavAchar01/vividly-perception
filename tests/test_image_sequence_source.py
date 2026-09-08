"""ImageSequenceSource: a directory of numbered frames played back as a FrameSource.

Replay only: nothing here opens a camera. Several registered event datasets ship their
imagery as a directory of numbered stills rather than as a video file, and a directory of
stills records no frame rate, so the rate is a required argument that the caller reads from
the dataset's registry entry. The replay pipeline (scripts/run_pipeline.py) is the live
caller of the source class, and the evidence extractor (scripts/eval/extract_evidence.py)
uses `list_frames` from the same module.

Same import rules as tests/test_sources.py: cv2 and numpy are darwin-marked, so every test
that needs one calls pytest.importorskip, and the module itself must import with neither
(pinned at the AST level there). Frames are written by the test, pixels encode the index.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vividly.sources import (
    Frame,
    FrameSource,
    ImageSequenceSource,
    SourceInfo,
    frame_count,
    list_frames,
)
from vividly.sources.image_sequence import natural_key

FPS = 2.5
N = 7
W, H = 40, 24


def _gray(index: int) -> int:
    return 10 * index + 5


@pytest.fixture(scope="module")
def frames_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """N zero-padded frames with per-frame sidecars, plus the junk a frame directory carries."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    root = tmp_path_factory.mktemp("frames")
    for index in range(N):
        image = np.full((H, W, 3), _gray(index), dtype=np.uint8)
        assert cv2.imwrite(str(root / f"Seq_{index + 1:06d}.jpg"), image)
        (root / f"Seq_{index + 1:06d}.txt").write_text("person 1 2 3 4 0\n")
    (root / "notes.md").write_text("not a frame\n")
    (root / "sub").mkdir()
    assert cv2.imwrite(str(root / "sub" / "Seq_999999.jpg"), np.zeros((H, W, 3), dtype=np.uint8))
    return root


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# ------------------------------------------------------------- listing
def test_natural_key_orders_digit_runs_numerically() -> None:
    names = ["f_10.jpg", "f_2.jpg", "f_1.jpg", "f_01.jpg", "a_5.jpg"]
    assert sorted(names, key=natural_key) == [
        "a_5.jpg",
        "f_1.jpg",
        "f_01.jpg",
        "f_2.jpg",
        "f_10.jpg",
    ]


def test_list_frames_is_ordered_pattern_filtered_and_not_recursive(frames_dir: Path) -> None:
    files = list_frames(frames_dir, "*.jpg")
    assert [f.name for f in files] == [f"Seq_{i + 1:06d}.jpg" for i in range(N)]
    assert frame_count(frames_dir) == N
    assert frame_count(frames_dir, "*.txt") == N
    with pytest.raises(ValueError, match="no frames match"):
        list_frames(frames_dir, "*.png")
    with pytest.raises(FileNotFoundError):
        list_frames(frames_dir / "missing")
    for bad in ("", "sub/*.jpg", "../*.jpg"):
        with pytest.raises(ValueError, match="plain file glob"):
            list_frames(frames_dir, bad)


# ------------------------------------------------------------- contract
def test_image_sequence_source_is_a_frame_source_with_clip_time(frames_dir: Path) -> None:
    np = pytest.importorskip("numpy")
    src = ImageSequenceSource(frames_dir, FPS)
    assert isinstance(src, FrameSource)
    assert src.info is None and src.files == ()
    info = src.open()
    assert info == SourceInfo(fps=FPS, width=W, height=H, n_frames=N)
    assert src.open() is info  # idempotent while open
    frames = list(src.frames())
    assert [f.index for f in frames] == list(range(N))
    assert [f.t_source for f in frames] == [i / FPS for i in range(N)]
    for f in frames:
        assert isinstance(f, Frame)
        assert f.image.shape == (H, W, 3) and f.image.dtype == np.uint8
        assert int(f.image[0, 0, 0]) == _gray(f.index)  # JPEG of a flat grey is exact
    assert list(src.frames()) == []  # exhausted
    src.close()
    assert src.info is None and src.next_index == 0
    with src as again:
        assert again.info is not None and again.next_index == 0


def test_seek_is_frame_exact_and_past_the_end_is_end(frames_dir: Path) -> None:
    src = ImageSequenceSource(frames_dir, FPS)
    src.seek(4)
    frames = list(src.frames())
    assert [f.index for f in frames] == [4, 5, 6]
    assert int(frames[0].image[0, 0, 0]) == _gray(4)
    src.seek(N + 10)
    assert list(src.frames()) == []
    with pytest.raises(ValueError, match=">= 0"):
        src.seek(-1)
    # A seek between yields is honoured on the next iteration.
    src.seek(0)
    it = src.frames()
    assert next(it).index == 0
    src.seek(5)
    assert [f.index for f in it] == [5, 6]


def test_a_seek_mid_iteration_and_a_close_under_the_iterator_are_safe(frames_dir: Path) -> None:
    src = ImageSequenceSource(frames_dir, FPS)
    it = src.frames()
    assert next(it).index == 0
    src.close()
    assert list(it) == []


def test_pacing_honours_rate_and_never_catches_up_by_dropping(frames_dir: Path) -> None:
    clock = FakeClock()
    src = ImageSequenceSource(frames_dir, FPS, rate=1.0, clock=clock, sleep=clock.sleep)
    frames = list(src.frames())
    assert len(frames) == N
    # First frame at the epoch, then one frame period (0.4 s) before each of the rest.
    assert clock.sleeps == pytest.approx([1.0 / FPS] * (N - 1))
    unpaced = ImageSequenceSource(frames_dir, FPS, rate=None, clock=clock, sleep=clock.sleep)
    clock.sleeps.clear()
    assert len(list(unpaced.frames())) == N and clock.sleeps == []


def test_constructor_refuses_a_missing_or_bad_rate(frames_dir: Path) -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="fps must be a finite positive"):
            ImageSequenceSource(frames_dir, bad)
    with pytest.raises(ValueError, match="rate must be positive"):
        ImageSequenceSource(frames_dir, FPS, rate=0.0)
    with pytest.raises(TypeError):
        ImageSequenceSource(frames_dir)  # type: ignore[call-arg]


def test_open_reports_the_real_problems(frames_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ImageSequenceSource(tmp_path / "nope", FPS).open()
    with pytest.raises(ValueError, match="no frames match"):
        ImageSequenceSource(frames_dir, FPS, pattern="*.png").open()
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "f_000001.jpg").write_bytes(b"not a jpeg")
    with pytest.raises(RuntimeError, match="could not decode frame"):
        ImageSequenceSource(broken, FPS).open()
    # A hole in the middle raises on the frame, never skips it.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    holed = tmp_path / "holed"
    holed.mkdir()
    assert cv2.imwrite(str(holed / "f_000001.jpg"), np.zeros((H, W, 3), dtype=np.uint8))
    (holed / "f_000002.jpg").write_bytes(b"")
    src = ImageSequenceSource(holed, FPS)
    it = src.frames()
    assert next(it).index == 0
    with pytest.raises(RuntimeError, match=r"f_000002\.jpg"):
        next(it)
