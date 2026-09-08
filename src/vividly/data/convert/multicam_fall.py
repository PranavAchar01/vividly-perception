"""Montreal Multi-Camera Fall Dataset labels (registry entry multicam_fall, test set).

The second registry-enforced test_only fall set (event_grant role test_only; never
trains). 24 scenarios (chute01..chute24), each recorded in a furnished room by 8
synchronized IP cameras (configs/datasets/multicam_fall.yaml), geometrically the closest
public fall footage to the overhead deployment domain. This module carries the scenario
annotation transcribed from the upstream technical report (Auvinet et al., TR 1350,
tables pp. 17-22) with the 2025-11-28 Erratum applied, and builds the same
annotations/index.json schema the other fall converters write.

What the erratum changed (Feng & Meunier, Erratum-MCF.pdf):
  * camera references "11" -> camera 8 and "10" -> camera 7 (only cam1..cam8 exist);
  * full corrected period lists for scenarios 2, 7, 8, 14 and 18.

Frame numbers are relative to each scenario's reference camera. The 8 cameras started
at different instants (per-scenario frame delays, report p. 16), so build_index()
covers only the reference camera of each scenario: zero synchronization arithmetic,
zero misalignment risk.

All-cameras surface (2026-08-28, under the full-testing rule): the report's sync
model (section 4.2) is end-aligned, that is the recordings were stopped together, and
Delay_{c,s} = Nframe_{c,s} - min_c Nframe_{c,s} (the printed formula dropped its
leading term; the prose is unambiguous). Mapping reference-camera frames to camera c
is therefore frame_c = frame_ref + delay_c - delay_ref, at equal nominal rates.
CAMERA_DELAYS below transcribes Figure 3 (p. 16) and is validated against the
containers themselves: delays re-derived from the 192 actual frame counts agree
cell-for-cell for scenarios 1-23 (184/184 checkable cells; the table's smudged
scenario-5/camera-5 cell prints as "18 7" and resolves to 18 by probe). Scenario 24
is the known anomaly: its cam8 container is ~30 frames shorter than the table
implies (probe-derived pairwise offsets for cams 1-7 match the table to within one
frame, cam7 carrying a +1 residual; cam8 is off by 30), that is, cam8's file lost
roughly a second at one end. chute24 stages no falls (it is the confounder
scenario), so the table is kept as source of record and
chute24__cam8's period annotations are flagged approximate (+-30 frames) in its
record. build_index_all_cameras() emits all 192 views with shifted periods; falls
are required to survive shifting unclipped (delays max 109 against earliest fall frame
288; a clipped fall raises rather than silently shrinking an event).

Position codes (report p. 17): 1 walking/standing, 2 falling, 3 lying on the ground,
4 crouching, 5 moving down, 6 moving up, 7 sitting, 8 lying on a sofa, 9 moving
horizontally. Code 2 periods are the fall events; chute04 and chute23 each stage two
falls, and chute24 stages none (it is the confounder scenario: codes 4/5 crouch and
moving-down periods only). Codes 4, 5, 7 and 8 periods are kept per scenario so evals
can break out the confounder classes.

Frame rate: the upstream AVI headers claim 120 fps, which is wrong. The report never
states a capture rate (pp. 1-16 checked), the Gadspot GS-4600 IP cameras (report p. 4,
2010, MPEG4 over IP) are spec-capped at 30 fps, 120 is exactly 4x30, and gravity rules
120 out: chute10's annotated fall spans 19 frames, which at 120 fps is 0.16 s, faster
than free fall from standing height (1 m of free fall takes 0.45 s), while at 30 fps
it is a normal 0.63 s fall. Every lying and walking duration is likewise plausible at
30 and absurd at 120. There is no audio stream and no burned-in on-screen clock to measure
against, so 30.0 is recorded as an inference, not a measurement; even if the true rate
were NTSC 29.97 the residual error is 0.1% against the header's 300%.
ReplaySource derives its clock from the container header and has no override, so
remux_reference_videos() stream-copies each reference AVI into remuxed/ with the
header corrected to REFERENCE_FPS, leaving pixels bit-identical and changing only the
container rate, and build_index() indexes the remuxed files, refusing any container that
still reports an implausible rate.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

INDEX_FILENAME = "index.json"

FALL_CODE = 2
CONFOUNDER_CODES = (4, 5, 7, 8)

# The upstream AVI headers claim 120 fps; the true rate is 30 (see module docstring).
REFERENCE_FPS = 30.0
UPSTREAM_HEADER_FPS = 120.0
MAX_PLAUSIBLE_FPS = 60.0
REMUX_DIRNAME = "remuxed"

# (scenario, reference_camera, ((start, end, code), ...)) - report pp. 17-22 with the
# erratum's corrected lists for scenarios 2, 7, 8, 14, 18 and camera refs 11->8, 10->7.
SCENARIOS: tuple[tuple[int, int, tuple[tuple[int, int, int], ...]], ...] = (
    (1, 8, ((874, 1011, 1), (1012, 1079, 6), (1080, 1108, 2), (1109, 1285, 3))),
    (2, 4, ((308, 394, 1), (395, 425, 2), (426, 600, 3))),
    (3, 8, ((380, 590, 1), (591, 625, 2), (626, 784, 3))),
    (
        4,
        6,
        (
            (230, 287, 1),
            (288, 314, 2),
            (315, 380, 3),
            (381, 600, 6),
            (601, 638, 2),
            (639, 780, 3),
        ),
    ),
    (5, 8, ((288, 310, 1), (311, 336, 2), (337, 450, 3))),
    (6, 1, ((325, 582, 1), (583, 629, 2), (630, 750, 3))),
    (7, 6, ((330, 490, 1), (491, 520, 2), (521, 680, 3))),
    (8, 4, ((144, 350, 1), (351, 380, 2), (381, 520, 3))),
    (
        9,
        8,
        (
            (310, 472, 1),
            (473, 505, 5),
            (506, 576, 7),
            (577, 627, 6),
            (628, 651, 2),
            (652, 760, 3),
        ),
    ),
    (10, 8, ((315, 461, 1), (462, 511, 5), (512, 530, 2), (531, 680, 3))),
    (11, 7, ((378, 463, 1), (464, 489, 2), (490, 600, 3))),
    (12, 8, ((355, 604, 1), (605, 653, 2), (654, 770, 3))),
    (
        13,
        4,
        (
            (301, 430, 1),
            (431, 476, 5),
            (477, 525, 7),
            (526, 636, 5),
            (637, 717, 8),
            (718, 780, 6),
            (781, 822, 6),
            (823, 863, 2),
            (864, 960, 3),
        ),
    ),
    (
        14,
        6,
        (
            (372, 590, 1),
            (591, 620, 5),
            (621, 929, 8),
            (930, 969, 6),
            (970, 1019, 6),
            (1020, 1060, 2),
            (1061, 1115, 3),
        ),
    ),
    (
        15,
        8,
        (
            (363, 486, 1),
            (487, 530, 5),
            (531, 630, 7),
            (631, 754, 6),
            (755, 787, 2),
            (788, 870, 3),
        ),
    ),
    (
        16,
        4,
        (
            (380, 455, 1),
            (456, 488, 5),
            (489, 530, 4),
            (531, 568, 6),
            (569, 629, 5),
            (630, 645, 4),
            (646, 670, 6),
            (671, 731, 5),
            (732, 817, 7),
            (818, 890, 6),
            (891, 940, 2),
            (941, 1000, 3),
        ),
    ),
    (
        17,
        6,
        (
            (251, 315, 1),
            (316, 340, 5),
            (341, 361, 4),
            (362, 388, 6),
            (389, 410, 5),
            (411, 430, 4),
            (431, 460, 6),
            (461, 531, 5),
            (532, 620, 7),
            (621, 729, 6),
            (730, 770, 2),
            (771, 860, 3),
        ),
    ),
    (
        18,
        6,
        (
            (301, 480, 1),
            (481, 520, 5),
            (521, 655, 7),
            (656, 684, 6),
            (685, 710, 2),
            (711, 740, 3),
        ),
    ),
    (19, 7, ((255, 498, 1), (499, 600, 2), (601, 770, 3))),
    (20, 8, ((301, 544, 1), (545, 672, 2), (673, 800, 3))),
    (
        21,
        8,
        (
            (408, 537, 1),
            (538, 608, 5),
            (609, 794, 7),
            (795, 863, 6),
            (864, 901, 2),
            (902, 1040, 3),
        ),
    ),
    (
        22,
        1,
        (
            (317, 586, 1),
            (587, 685, 5),
            (686, 737, 7),
            (738, 766, 6),
            (767, 808, 2),
            (809, 930, 3),
        ),
    ),
    (
        23,
        8,
        (
            (393, 662, 1),
            (663, 688, 5),
            (689, 710, 4),
            (711, 744, 6),
            (745, 1519, 1),
            (1520, 1595, 2),
            (1596, 1661, 6),
            (1662, 1730, 1),
            (1731, 1769, 5),
            (1770, 1839, 4),
            (1840, 1886, 6),
            (1887, 2645, 1),
            (2646, 2698, 5),
            (2699, 2958, 8),
            (2959, 3035, 6),
            (3036, 3156, 1),
            (3157, 3237, 5),
            (3238, 3416, 8),
            (3417, 3573, 6),
            (3574, 3614, 2),
            (3615, 3745, 6),
            (3746, 3795, 5),
            (3796, 4042, 4),
            (4043, 4105, 6),
            (4106, 4204, 1),
            (4205, 4264, 5),
            (4265, 4440, 7),
            (4441, 4527, 6),
            (4528, 5200, 1),
        ),
    ),
    (
        24,
        6,
        (
            (350, 974, 1),
            (975, 1315, 1),
            (1316, 1351, 5),
            (1352, 1414, 4),
            (1415, 1450, 6),
            (1451, 1750, 1),
            (1751, 1805, 5),
            (1806, 1844, 4),
            (1845, 1884, 6),
            (1885, 2490, 1),
            (2491, 2514, 5),
            (2515, 2563, 4),
            (2564, 2587, 6),
            (2588, 3040, 1),
            (3041, 3077, 5),
            (3078, 3125, 6),
            (3126, 3243, 1),
            (3244, 3353, 1),
            (3354, 3401, 5),
            (3402, 3500, 4),
        ),
    ),
)


# Report Figure 3 (p. 16): per-scenario start-offset (in frames) of each camera,
# end-aligned (see module docstring). Index = camera number 1..8. Probe-validated
# against container frame counts 2026-08-28; scenario 5 camera 5 transcribed from the
# probe (the print reads "18 7"); scenario 24 carries the cam8 container anomaly.
CAMERA_DELAYS: dict[int, tuple[int, int, int, int, int, int, int, int]] = {
    1: (3, 3, 8, 4, 23, 6, 6, 0),
    2: (25, 40, 0, 16, 18, 33, 33, 6),
    3: (12, 16, 8, 16, 35, 20, 20, 0),
    4: (72, 79, 78, 0, 68, 82, 83, 56),
    5: (17, 24, 5, 11, 18, 26, 28, 0),
    6: (0, 100, 106, 90, 89, 103, 104, 89),
    7: (28, 14, 16, 0, 1, 17, 18, 20),
    8: (92, 79, 0, 81, 64, 81, 82, 56),
    9: (18, 9, 1, 19, 13, 11, 12, 0),
    10: (14, 15, 19, 33, 12, 17, 19, 0),
    11: (23, 4, 20, 14, 0, 6, 7, 12),
    12: (21, 6, 13, 8, 0, 3, 7, 0),
    13: (16, 33, 0, 7, 27, 27, 36, 13),
    14: (49, 36, 38, 0, 29, 29, 7, 14),
    15: (15, 19, 19, 15, 34, 40, 23, 0),
    16: (23, 29, 0, 2, 12, 9, 3, 3),
    17: (21, 26, 15, 0, 10, 0, 29, 18),
    18: (99, 105, 86, 0, 84, 108, 109, 77),
    19: (19, 27, 16, 19, 5, 29, 0, 20),
    20: (25, 9, 3, 10, 10, 4, 5, 0),
    21: (20, 30, 22, 3, 8, 33, 32, 0),
    22: (0, 46, 51, 41, 53, 46, 47, 34),
    23: (31, 52, 52, 45, 54, 60, 50, 0),
    24: (3, 36, 7, 0, 37, 10, 33, 1),
}

# chute24's cam8 container is ~30 frames short of the delay table's implication
# (probe 2026-08-28: derived pairwise offsets match the table for cams 1-7, cam8 is
# off by this many frames). No falls in chute24; its cam8 periods are approximate.
CHUTE24_CAM8_ANOMALY_FRAMES = 30
ALLCAMS_INDEX_FILENAME = "index_allcams.json"


class MulticamParseError(ValueError):
    """A tree or table state that cannot become an index."""


def validate_table(
    scenarios: tuple[tuple[int, int, tuple[tuple[int, int, int], ...]], ...] = SCENARIOS,
) -> None:
    """Mechanical checks on the transcription: 24 scenarios, valid camera numbers and
    codes, and strictly contiguous periods (each period starts one frame after the
    previous ends, which is true of every scenario in the report, so a typo, a swapped
    row or a dropped digit breaks contiguity and fails here)."""
    if [s for s, _, _ in scenarios] != list(range(1, 25)):
        raise MulticamParseError("scenario numbers must be exactly 1..24")
    for scenario, cam, periods in scenarios:
        if not 1 <= cam <= 8:
            raise MulticamParseError(f"chute{scenario:02d}: camera {cam} outside 1..8")
        prev_end: int | None = None
        for start, end, code in periods:
            if code not in range(1, 10):
                raise MulticamParseError(f"chute{scenario:02d}: position code {code}")
            if not 0 < start <= end:
                raise MulticamParseError(f"chute{scenario:02d}: bad period {start}-{end}")
            if prev_end is not None and start != prev_end + 1:
                raise MulticamParseError(
                    f"chute{scenario:02d}: period {start}-{end} not contiguous with "
                    f"previous end {prev_end}"
                )
            prev_end = end


def fall_periods(periods: tuple[tuple[int, int, int], ...]) -> list[tuple[int, int]]:
    return [(s, e) for s, e, code in periods if code == FALL_CODE]


def _upstream_rel(scenario: int, cam: int) -> Path:
    return Path("extracted") / "dataset" / f"chute{scenario:02d}" / f"cam{cam}.avi"


def _remuxed_rel(scenario: int, cam: int) -> Path:
    return Path(REMUX_DIRNAME) / f"chute{scenario:02d}" / f"cam{cam}.avi"


def _remux_argv(ffmpeg: str, src: Path, dst: Path, fps: float) -> list[str]:
    """Input-side -r re-stamps the (CFR) AVI at the corrected rate; -c:v copy keeps the
    video packets bit-identical. Only the container header changes."""
    return [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-r",
        f"{fps:g}",
        "-i",
        str(src),
        "-c:v",
        "copy",
        "-an",
        str(dst),
    ]


def remux_reference_videos(
    root: Path, *, fps: float = REFERENCE_FPS, force: bool = False
) -> list[Path]:
    """Stream-copy each scenario's reference-camera AVI into remuxed/ with the container
    rate corrected UPSTREAM_HEADER_FPS -> fps. Upstream bytes are never modified."""
    import subprocess

    from vividly.media.ffmpeg import ffmpeg_bin

    ffmpeg = ffmpeg_bin()
    out: list[Path] = []
    for scenario, cam, _ in SCENARIOS:
        src = root / _upstream_rel(scenario, cam)
        dst = root / _remuxed_rel(scenario, cam)
        if not src.is_file():
            raise MulticamParseError(f"missing upstream reference-camera video {src}")
        if force or not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(_remux_argv(ffmpeg, src, dst, fps), check=True)
        out.append(dst)
    return out


def _probe_video_cv2(path: Path) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise MulticamParseError(f"cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not fps > 0 or n <= 0:
        raise MulticamParseError(f"container reports fps={fps} frames={n} for {path}")
    return fps, n


def build_index(
    root: Path, *, probe: Callable[[Path], tuple[float, int]] | None = None
) -> dict[str, Any]:
    """One record per scenario, pointing at its reference camera's rate-corrected AVI
    under remuxed/chuteNN/ (run remux_reference_videos first). Multi-fall scenarios
    carry event_frames (a list); the single-interval fields stay for schema
    compatibility (null when >1 fall)."""
    if probe is None:
        probe = _probe_video_cv2
    validate_table()
    records: list[dict[str, Any]] = []
    for scenario, cam, periods in SCENARIOS:
        rel = _remuxed_rel(scenario, cam)
        video = root / rel
        if not video.is_file():
            raise MulticamParseError(
                f"missing reference-camera video {video}; run "
                "multicam_fall.remux_reference_videos(root) to build the "
                "rate-corrected tree"
            )
        fps, n_frames = probe(video)
        if fps > MAX_PLAUSIBLE_FPS:
            raise MulticamParseError(
                f"{rel}: container reports {fps:g} fps - the upstream 120 fps header "
                "bug (see module docstring); this file was not rate-corrected"
            )
        falls = fall_periods(periods)
        confounders = sorted({code for _, _, code in periods if code in CONFOUNDER_CODES})
        rec: dict[str, Any] = {
            "location": f"chute{scenario:02d}",
            "video": str(rel),
            "video_upstream": str(_upstream_rel(scenario, cam)),
            "fps": fps,
            "n_frames_container": n_frames,
            "annotation": "technical report tables pp.17-22 + Erratum-MCF (transcribed)",
            "has_fall": bool(falls),
            "fall_start_frame": falls[0][0] if len(falls) == 1 else None,
            "fall_end_frame": falls[0][1] if len(falls) == 1 else None,
            "event_frames": [[s, e] for s, e in falls],
            "confounder_codes": confounders,
            "n_periods": len(periods),
        }
        records.append(rec)
    summary = {
        "n_videos": len(records),
        "n_fall_videos": sum(1 for r in records if r["has_fall"]),
        "n_fall_events": sum(len(r["event_frames"]) for r in records),
        "n_no_fall": sum(1 for r in records if not r["has_fall"]),
    }
    return {"dataset": "multicam_fall", "summary": summary, "videos": records}


def write_index(root: Path, index: Mapping[str, Any] | None = None) -> Path:
    if index is None:
        index = build_index(root)
    out = root / "annotations" / INDEX_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out


def probe_delays(
    root: Path, *, probe: Callable[[Path], tuple[float, int]] | None = None
) -> dict[int, tuple[int, ...]]:
    """Re-derive the Figure-3 delays from the containers themselves:
    Delay_{c,s} = Nframe_{c,s} - min_c Nframe_{c,s} (report section 4.2)."""
    if probe is None:
        probe = _probe_video_cv2
    out: dict[int, tuple[int, ...]] = {}
    for scenario in range(1, 25):
        counts = [probe(root / _upstream_rel(scenario, cam))[1] for cam in range(1, 9)]
        m = min(counts)
        out[scenario] = tuple(n - m for n in counts)
    return out


def validate_delays(
    root: Path, *, probe: Callable[[Path], tuple[float, int]] | None = None
) -> None:
    """The transcribed CAMERA_DELAYS must agree with the container-derived delays for
    scenarios 1-23 cell-for-cell. Scenario 24 must show exactly the known anomaly:
    pairwise offsets among cams 1-7 matching the table, cam8 short by
    CHUTE24_CAM8_ANOMALY_FRAMES. Anything else is a new tree defect and refuses."""
    derived = probe_delays(root, probe=probe)
    for scenario in range(1, 24):
        if derived[scenario] != CAMERA_DELAYS[scenario]:
            raise MulticamParseError(
                f"chute{scenario:02d}: container-derived delays {derived[scenario]} "
                f"disagree with the report table {CAMERA_DELAYS[scenario]}"
            )
    # chute24 (no falls staged): the containers disagree with the table by a ~30-frame
    # cam8 tail loss plus a +-1-frame cam7 residual (measured 2026-08-28: derived
    # pairwise offsets vs table = +0,+0,+0,+0,+0,+1,-30 for cams 2..8 rel cam1).
    # Period annotations on this scenario's non-reference cameras are approximate
    # anyway; tolerate +-1 here, exactness stays required for 1-23.
    tab, der = CAMERA_DELAYS[24], derived[24]
    for c in range(1, 7):
        if abs((der[c] - der[0]) - (tab[c] - tab[0])) > 1:
            raise MulticamParseError(
                f"chute24: cam{c + 1} pairwise offset {der[c] - der[0]} disagrees "
                f"with table {tab[c] - tab[0]} beyond the recorded +-1"
            )
    anomaly = (der[7] - der[0]) - (tab[7] - tab[0])
    if abs(anomaly + CHUTE24_CAM8_ANOMALY_FRAMES) > 1:
        raise MulticamParseError(
            f"chute24 cam8 anomaly measures {anomaly}, expected about "
            f"{-CHUTE24_CAM8_ANOMALY_FRAMES} - the tree changed"
        )


def shift_periods(
    periods: tuple[tuple[int, int, int], ...],
    delay_ref: int,
    delay_cam: int,
    n_frames_cam: int,
) -> list[tuple[int, int, int]]:
    """Map reference-camera periods into camera c's frame numbering
    (frame_c = frame_ref + delay_cam - delay_ref), clipping to [1, n_frames_cam].
    A period entirely outside the camera's container is dropped; a fall period that
    would be clipped at all raises, because an event exam must never silently shrink
    an event window."""
    shift = delay_cam - delay_ref
    out: list[tuple[int, int, int]] = []
    for start, end, code in periods:
        s2, e2 = start + shift, end + shift
        cs, ce = max(1, s2), min(n_frames_cam, e2)
        if cs > ce:
            if code == FALL_CODE:
                raise MulticamParseError(
                    f"fall period {start}-{end} shifted by {shift} leaves the "
                    f"container entirely (n={n_frames_cam})"
                )
            continue
        if code == FALL_CODE and (cs != s2 or ce != e2):
            raise MulticamParseError(
                f"fall period {start}-{end} shifted by {shift} would be clipped to "
                f"{cs}-{ce} (n={n_frames_cam}); refusing to shrink an event"
            )
        out.append((cs, ce, code))
    return out


def remux_all_videos(root: Path, *, fps: float = REFERENCE_FPS, force: bool = False) -> list[Path]:
    """remux_reference_videos for every camera: all 192 AVIs stream-copied into
    remuxed/ with the container rate corrected. Upstream bytes never modified."""
    import subprocess

    from vividly.media.ffmpeg import ffmpeg_bin

    ffmpeg = ffmpeg_bin()
    out: list[Path] = []
    for scenario in range(1, 25):
        for cam in range(1, 9):
            src = root / _upstream_rel(scenario, cam)
            dst = root / _remuxed_rel(scenario, cam)
            if not src.is_file():
                raise MulticamParseError(f"missing upstream video {src}")
            if force or not dst.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(_remux_argv(ffmpeg, src, dst, fps), check=True)
            out.append(dst)
    return out


def build_index_all_cameras(
    root: Path, *, probe: Callable[[Path], tuple[float, int]] | None = None
) -> dict[str, Any]:
    """All 192 camera views with periods shifted into each camera's own frame
    numbering via CAMERA_DELAYS (run remux_all_videos and validate_delays first).
    Reference-camera records reproduce build_index() exactly (shift 0). The frozen
    24-video index.json is not touched; this writes a separate surface."""
    if probe is None:
        probe = _probe_video_cv2
    validate_table()
    records: list[dict[str, Any]] = []
    for scenario, ref_cam, periods in SCENARIOS:
        delays = CAMERA_DELAYS[scenario]
        for cam in range(1, 9):
            rel = _remuxed_rel(scenario, cam)
            video = root / rel
            if not video.is_file():
                raise MulticamParseError(
                    f"missing camera video {video}; run multicam_fall.remux_all_videos(root)"
                )
            fps, n_frames = probe(video)
            if fps > MAX_PLAUSIBLE_FPS:
                raise MulticamParseError(
                    f"{rel}: container reports {fps:g} fps - not rate-corrected"
                )
            shifted = shift_periods(periods, delays[ref_cam - 1], delays[cam - 1], n_frames)
            falls = [(s, e) for s, e, code in shifted if code == FALL_CODE]
            confounders = sorted({code for _, _, code in shifted if code in CONFOUNDER_CODES})
            alignment = "end-sync delays, report fig.3 (probe-validated)"
            if scenario == 24 and cam == 8:
                alignment = (
                    "APPROXIMATE +-"
                    f"{CHUTE24_CAM8_ANOMALY_FRAMES}: cam8 container-tail anomaly "
                    "(see converter docstring); no falls staged in chute24"
                )
            records.append(
                {
                    "location": f"chute{scenario:02d}",
                    "camera": cam,
                    "is_reference_camera": cam == ref_cam,
                    "delay_frames": delays[cam - 1],
                    "video": str(rel),
                    "video_upstream": str(_upstream_rel(scenario, cam)),
                    "fps": fps,
                    "n_frames_container": n_frames,
                    "annotation": "technical report tables pp.17-22 + Erratum-MCF "
                    "(transcribed), shifted per-camera",
                    "annotation_alignment": alignment,
                    "has_fall": bool(falls),
                    "fall_start_frame": falls[0][0] if len(falls) == 1 else None,
                    "fall_end_frame": falls[0][1] if len(falls) == 1 else None,
                    "event_frames": [[s, e] for s, e in falls],
                    "confounder_codes": confounders,
                    "n_periods": len(shifted),
                }
            )
    summary = {
        "n_videos": len(records),
        "n_fall_videos": sum(1 for r in records if r["has_fall"]),
        "n_fall_events": sum(len(r["event_frames"]) for r in records),
        "n_no_fall": sum(1 for r in records if not r["has_fall"]),
    }
    return {"dataset": "multicam_fall", "summary": summary, "videos": records}


def write_index_allcams(root: Path, index: Mapping[str, Any] | None = None) -> Path:
    """Write the all-cameras surface to annotations/index_allcams.json under `root`.

    This is how that file is produced. Like the other converters' write_index, it is run
    once over the dataset root by whoever prepares the corpus, not from inside the
    pipeline: `configs/events/extract_multicam_allcams_v2c.yaml` reads the file it writes.
    """
    if index is None:
        index = build_index_all_cameras(root)
    out = root / "annotations" / ALLCAMS_INDEX_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
