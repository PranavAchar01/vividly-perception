# red_substance_on_floor: the contract

A prototype colour detector for the live appliance, in the shape of the two analyzers
beside it (`seizure.py`, `fall.py`): a config dataclass, a state dataclass, and an analyzer
with `update()` and `on_absent()`. It must cost near nothing per frame and add no delay.

This produces the `red_substance_on_floor` descriptor that the Clip Library vocabulary
already reserves. Keep that key; do not invent a new word for it.

## What makes this hard, and what actually separates a pool from the room

The room has red in it already: a red step ladder on the left wall, red handles, red on the
shelving. A plain hue threshold lights all of it up on frame one. Three things separate a
spill from the furniture, and the detector must use all three.

1. **Novelty.** Furniture has been red since the process started. Keep a slowly adapting
   per-pixel baseline of the red mask and score only red that is *new* against it. Prime the
   baseline for the first `prime_frames` updates and report nothing during priming.
2. **Place.** A pool is on the floor. Ignore everything above `floor_y_frac` of frame height.
3. **Persistence.** A pool stays. A red shirt crossing the floor does not hold still in one
   place for `min_hold_s`, and a compression artefact lasts one frame.

## Interface (fixed; every other part of this build codes against it)

```python
@dataclass(frozen=True)
class SubstanceConfig:
    proc_width: int = 320        # analysis runs at this width, not the display width
    sat_min: int = 90            # HSV, OpenCV ranges (H 0..179, S 0..255, V 0..255)
    val_min: int = 60
    hue_band_a: tuple[int, int] = (0, 10)     # red wraps the hue circle, so two bands
    hue_band_b: tuple[int, int] = (170, 179)
    floor_y_frac: float = 0.45   # rows above this fraction of height are never considered
    min_area_frac: float = 0.0008  # of the processed frame, below this is noise
    open_kernel: int = 3         # morphological open, kills speckle
    bg_alpha: float = 0.02       # baseline adaptation per update
    prime_frames: int = 30       # learn the room's own red before reporting anything
    min_hold_s: float = 1.5      # a candidate must persist this long to confirm
    release_s: float = 1.0       # gone this long before the state clears
    max_pools: int = 4           # largest N reported

@dataclass(frozen=True)
class SubstanceState:
    state: str = "idle"                      # "idle" | "candidate" | "confirmed"
    area_frac: float = 0.0                   # newly-red floor area, fraction of full frame
    n_pools: int = 0
    held_s: float = 0.0                      # how long the candidate has held
    growth_frac_per_s: float = 0.0           # positive while spreading
    polygons: tuple = ()                     # contours in DISPLAY pixel coords, each (N,1,2) int32
    priming: bool = False                    # True until the baseline has been learned

class SubstanceAnalyzer:
    def __init__(self, cfg: SubstanceConfig | None = None) -> None: ...
    def update(self, frame_bgr, now: float, exclude_boxes=()) -> SubstanceState: ...
    def on_absent(self, now: float) -> None: ...
    def reset(self) -> None: ...
```

`update` takes the display-resolution BGR frame, scales it down itself, and returns polygons
already scaled back to display coordinates so the compositor can draw them directly.
`exclude_boxes` is an iterable of xyxy arrays in display coordinates that are masked out
before scoring; the caller may pass tracked person boxes.

## Budget

Under 3 ms per update at `proc_width` 320 on this machine, measured, and it runs at a
capped rate on its own thread so it never competes with pose. No model file, no ONNX, no
allocation per frame beyond the mask buffers.
