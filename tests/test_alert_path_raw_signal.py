"""Tripwire: no temporal smoothing may enter the alert path
(research/v2-training-stack.md section 7c).

The finding this enforces: a 5-frame moving average at 15 fps places an exact
spectral null on 3.0 Hz, the center of the clinically validated 2-6 Hz convulsive
band, so any smoother between the pose model and the event verifiers silently
erases the seizure signature the system exists to catch. Today the alert path
carries no smoothing at all; this test keeps it that way the same way the guard
scripts keep AGPL out: by scanning the source, so the violation is a red test at
commit time instead of a dead seizure lane in production.

Band-pass is not smoothing: a future verifier feature extractor may band-pass
raw history (~1-7 Hz keeps the band; scipy butter/sosfiltfilt in band-pass form
is legal). If one lands, add its file to BANDPASS_ALLOWED below with a comment
naming the filter design; the allowlist entry is the review trigger.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Everything between the model outputs and the alert store, plus the pipeline glue.
ALERT_PATH: tuple[str, ...] = (
    "src/vividly/tracks",
    "src/vividly/rules",
    "src/vividly/sources",
    "src/vividly/events",
    "scripts/run_pipeline.py",
)

# Unambiguous temporal-smoothing primitives and names. Deliberately not here:
# butter/sosfilt/filtfilt (the legal band-pass building blocks, gated by the
# allowlist instead) and plain "mean" (rules legitimately aggregate windows for
# geometry; the ban is on smoothing series that later feed spectral features).
BANNED_TOKENS: tuple[str, ...] = (
    "savgol",
    "savitzky",
    "SmoothNet",
    "OneEuro",
    "one_euro",
    "oneeuro",
    "medfilt",
    "median_filter",
    "uniform_filter",
    "gaussian_filter",
    "moving_average",
    "rolling_mean",
    "ewma",
    "exponential_moving",
    "np.convolve",
    "lfilter",
)

# Files allowed to carry an explicit band-pass (none yet). Path -> reason.
BANDPASS_ALLOWED: dict[str, str] = {}


def _alert_path_files() -> list[Path]:
    files: list[Path] = []
    for entry in ALERT_PATH:
        p = REPO / entry
        if p.is_file():
            files.append(p)
        else:
            files.extend(sorted(p.rglob("*.py")))
    assert files, "alert-path scan found no files - the layout moved, fix ALERT_PATH"
    return files


def test_no_smoothing_primitive_in_the_alert_path() -> None:
    hits: list[str] = []
    for f in _alert_path_files():
        rel = str(f.relative_to(REPO))
        if rel in BANDPASS_ALLOWED:
            continue
        text = f.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            for token in BANNED_TOKENS:
                if token.lower() in lowered:
                    hits.append(f"{rel}:{i}: {line.strip()!r} (token {token!r})")
    assert not hits, (
        "temporal smoothing entered the alert path - it nulls the 2-6 Hz seizure "
        "band (research/v2-training-stack.md section 7c). Raw signals feed the verifiers; "
        "smooth on the dashboard branch only.\n" + "\n".join(hits)
    )


def test_the_law_is_written_where_implementers_will_read_it() -> None:
    # The tripwire catches code; these docstrings catch the human before the code
    # exists. If either paragraph is deleted, the next verifier author never learns
    # why their input is raw.
    verify = (REPO / "src/vividly/rules/verify.py").read_text(encoding="utf-8")
    state = (REPO / "src/vividly/tracks/state.py").read_text(encoding="utf-8")
    # Matched case-insensitively: the rule is that the named paragraph exists, not how it
    # is capitalised.
    assert "raw-signal" in verify.lower()
    assert "raw-signal" in state.lower()
