"""Red substance on the floor: novelty against a learned room, on the floor, held over time.

The problem this solves is not "find red". The garage the appliance watches has a red step
ladder on the left wall and red handles on the shelving, so a hue threshold alone paints the
room on frame one and the descriptor is worthless. Three independent facts separate a pool of
liquid from the furniture, and all three are required before anything is reported:

  1. NOVELTY. The furniture has been red since the process started. A slowly adapting
     per-pixel baseline records how often each pixel is red; only red that is new against that
     baseline scores. The baseline is primed over the first `prime_frames` updates and nothing
     at all is reported while it is priming, so the room's own red is learned, never alerted.
  2. PLACE. A pool is on the floor. Everything above `floor_y_frac` of frame height is cut
     before any pixel is touched, which is also why the analysis is cheap: the resize itself
     only ever reads the floor band.
  3. PERSISTENCE. A pool stays put. A red shirt crossing the floor is novel and is on the
     floor, but it does not hold the SAME footprint for `min_hold_s`, and a compression
     artefact lasts one frame. The hold is measured against the footprint the candidate
     started with, not merely against "some red is present", so a moving red object restarts
     the clock instead of accumulating it.

Two failure modes get explicit machinery, because each one silently destroys the detector:

  * A baseline that keeps adapting LEARNS THE SPILL. At bg_alpha 0.02 a stationary pool is
    half-absorbed into the background in under two seconds of video and gone within a minute,
    so the one real event this exists for would erase itself while it was still on the floor.
    Adaptation is therefore frozen on exactly the pixels that are currently scoring as novel.
  * EXPOSURE AND WHITE BALANCE SHIFT when a person walks into frame; auto-gain lifts
    saturation across the whole image within one frame, and every marginally-red floor pixel
    crosses `sat_min` at once. Liquid does not teleport: a jump in novel area that appears in
    a single update AND is spread across the whole floor band is a camera response, not a
    pool. Such a frame is untrusted (it neither confirms nor releases) and the baseline
    re-learns fast for a moment, everywhere except a pool that was already being reported.

No model file, no per-frame allocation beyond the mask buffers, which are allocated once on
the first update and reused.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# --- constants that are mechanism, not tuning knobs; they stay out of the fixed config -----

# The baseline holds "how red this pixel usually is" on the 0..255 scale of the mask itself
# (accumulateWeighted takes the uint8 mask directly, so no per-frame float conversion). A
# pixel is NEW red only if it has been red in well under half of the recent frames: room
# fixtures sit pinned at 255, a fresh pool sits at 0.
BASELINE_RED_MAX = 110.0

# One update may add at most this fraction of the floor band before the frame is suspected of
# being a camera response rather than a spill. Measured on this camera: a spreading pool adds
# about 0.001 of the band per update, and a warm white-balance swing large enough to push the
# grey floor over sat_min added 0.0196 in a single update. 0.015 sits between them, fifteen
# times a pool's growth rate. The earlier 0.10 was reasoned rather than measured and was five
# times too coarse to see a real swing at all.
LIGHTING_JUMP_FRAC = 0.015
# ...and only if that novelty is spread over the band rather than pooled: the bounding box of
# the novel pixels must cover this fraction of the floor band. This is the gate that protects
# an instantaneous spill, which jumps as hard as a camera response but is compact: measured
# bounding boxes are 0.37 to 0.42 of the band for synthesised pools and 0.82 for the warm
# swing, so 0.60 clears both by about 1.4x. A real pool is compact even when it is large, and
# both gates together are what "everywhere at once" means.
LIGHTING_SPREAD_FRAC = 0.60
# After a lighting event the lifted pixels stay lifted, so the guard cannot be a single-frame
# veto: the baseline is given this long at a fast rate to absorb the new exposure, and the
# frames in that window stay untrusted.
LIGHTING_RELEARN_S = 1.0
LIGHTING_RELEARN_ALPHA = 0.25

# A candidate keeps its clock only while it still overlaps the footprint it started with.
# Liquid stays where it is; a person in a red shirt walks off their own footprint in well
# under a second and starts again from zero.
SEED_OVERLAP_MIN = 0.30

# Window over which spreading is measured, long enough to be stable at any capture rate and
# short enough that growth reported alongside a live event is current.
GROWTH_WINDOW_S = 2.0

# Fraction of the updates since a candidate started its clock in which the pool must actually
# have been there before it may confirm. Liquid does not blink: measured over three
# synthesised spills the pool was present in 100% of updates, while the two false candidates
# this detector produces (a floor pixel sitting on the sat_min boundary after a white-balance
# swing, and a person's forearm) were present in 39% and 17%. release_s alone cannot separate
# these, because it bridges every gap shorter than a second and so lets a blob that is absent
# half the time accumulate an unbroken hold.
DUTY_MIN = 0.80


@dataclass(frozen=True)
class SubstanceConfig:
    proc_width: int = 320  # analysis runs at this width, not the display width
    sat_min: int = 90  # HSV, OpenCV ranges (H 0..179, S 0..255, V 0..255)
    val_min: int = 60
    hue_band_a: tuple[int, int] = (0, 10)  # red wraps the hue circle, so two bands
    hue_band_b: tuple[int, int] = (170, 179)
    floor_y_frac: float = 0.45  # rows above this fraction of height are never considered
    min_area_frac: float = 0.0008  # of the processed frame, below this is noise
    open_kernel: int = 3  # morphological open, kills speckle
    bg_alpha: float = 0.02  # baseline adaptation per update
    prime_frames: int = 30  # learn the room's own red before reporting anything
    min_hold_s: float = 1.5  # a candidate must persist this long to confirm
    release_s: float = 1.0  # gone this long before the state clears
    max_pools: int = 4  # largest N reported


@dataclass(frozen=True)
class SubstanceState:
    state: str = "idle"  # "idle" | "candidate" | "confirmed"
    area_frac: float = 0.0  # newly-red floor area, fraction of full frame
    n_pools: int = 0
    held_s: float = 0.0  # how long the candidate has held
    growth_frac_per_s: float = 0.0  # positive while spreading
    polygons: tuple = ()  # contours in DISPLAY pixel coords, each (N,1,2) int32
    priming: bool = False  # True until the baseline has been learned


class SubstanceAnalyzer:
    """Streaming, whole-frame. Feed it (display BGR frame, t) per capture."""

    def __init__(self, cfg: SubstanceConfig | None = None) -> None:
        self.cfg = cfg or SubstanceConfig()
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.cfg.open_kernel, self.cfg.open_kernel)
        )
        self.reset()

    # -- lifecycle -------------------------------------------------------------------------

    def reset(self) -> None:
        self._shape: tuple | None = None  # (src_h, src_w, band_h, proc_w) the buffers fit
        self._levels: list[tuple[int, int, np.ndarray]] = []
        self._primed = 0
        self._bg: np.ndarray | None = None
        self._seed: np.ndarray | None = None  # footprint the current candidate started with
        self._prev_novel: np.ndarray | None = None
        self._prev_novel_frac = 0.0
        self._relearn_until = -1e18
        self._active_since: float | None = None
        self._below_since: float | None = None
        self._state = "idle"
        self._held = 0.0
        self._seen = 0  # updates in this episode where a pool was actually present
        self._updates = 0  # updates in this episode, present or not
        self._area_hist: deque[tuple[float, float]] = deque()

    def on_absent(self, now: float) -> None:
        """No trackable subject in frame. Deliberately a no-op on the pool state: a spill
        outlives the person who made it, and the moment everyone has walked away is exactly
        when the descriptor matters most. The only thing a subject's absence changes is that
        no exclusion boxes arrive, which `update` already handles."""

    # -- buffers ---------------------------------------------------------------------------

    def _alloc(self, src_h: int, src_w: int, y0_src: int, band_h: int, proc_w: int) -> None:
        """Allocate every mask buffer once. update() after this point does no allocation
        except the contour arrays it actually returns."""
        self._shape = (src_h, src_w, band_h, proc_w)

        # Downscale plan: exact halvings, then one step onto the target width. Going straight
        # from a 2304-wide frame to 320 with INTER_AREA measures 8.3 ms on this machine, fifty
        # times the cost of everything else in the update put together, because a non-integer
        # scale falls into OpenCV's generic weighted path. An exact halving hits the SIMD fast
        # path at 0.06 ms, and successive halvings are the same box average.
        self._levels = []
        h, w = src_h - y0_src, src_w
        cover_h, cover_w = float(h), float(w)  # source pixels the chain actually covers
        while w // 2 >= proc_w and h // 2 >= band_h:
            crop_h, crop_w = (h // 2) * 2, (w // 2) * 2  # an odd edge row/column is dropped
            cover_h *= crop_h / h
            cover_w *= crop_w / w
            h, w = h // 2, w // 2
            self._levels.append((crop_h, crop_w, np.empty((h, w, 3), np.uint8)))
        # Polygons are scaled back with these, so they must describe the region the chain
        # covered, not the region it was handed: an odd row dropped three levels up would
        # otherwise smear every contour a couple of display pixels down the frame.
        self._sx = cover_w / proc_w
        self._sy = cover_h / band_h

        z8 = (band_h, proc_w)
        self._small = np.empty((band_h, proc_w, 3), np.uint8)
        self._hsv = np.empty((band_h, proc_w, 3), np.uint8)
        self._mask_a = np.empty(z8, np.uint8)
        self._mask_b = np.empty(z8, np.uint8)
        self._red = np.empty(z8, np.uint8)
        self._fresh = np.empty(z8, np.uint8)
        self._novel = np.empty(z8, np.uint8)
        self._adapt = np.empty(z8, np.uint8)
        self._tmp = np.empty(z8, np.uint8)
        self._bg = np.zeros(z8, np.float32)
        self._seed = np.zeros(z8, np.uint8)
        # The pools with their interiors filled in. Persistence is measured on this and not
        # on the raw novelty mask: liquid deep enough to pool goes DARKER than val_min at its
        # centre, so a spreading pool's novelty is an annulus whose hole grows outward and
        # eventually swallows the footprint the clock started on. Measured on the raw mask a
        # motionless pool loses its own seed and restarts, which was observed at 12 s into a
        # spreading pool. Filled, the footprint strictly contains its origin, while a person
        # walking still carries their whole silhouette off the seed.
        self._filled = np.zeros(z8, np.uint8)
        self._prev_novel = np.zeros(z8, np.uint8)
        # The novelty that existed just BEFORE a lighting event, held for the whole relearn
        # window: it is the one region the fast relearn must not touch.
        self._preevent_novel = np.zeros(z8, np.uint8)
        self._primed = 0

    # -- main ------------------------------------------------------------------------------

    def update(self, frame_bgr, now: float, exclude_boxes=()) -> SubstanceState:
        cfg = self.cfg
        src_h, src_w = frame_bgr.shape[:2]
        proc_w = min(cfg.proc_width, src_w)
        proc_h = max(1, round(proc_w * src_h / src_w))
        y0_proc = int(cfg.floor_y_frac * proc_h)
        band_h = max(1, proc_h - y0_proc)
        y0_src = int(cfg.floor_y_frac * src_h)

        if self._shape != (src_h, src_w, band_h, proc_w):
            self._alloc(src_h, src_w, y0_src, band_h, proc_w)

        # PLACE, done at the cheapest possible point: only the floor band is ever resized or
        # converted, so the rows above floor_y_frac cost nothing rather than being masked
        # away after the work has already been paid for.
        src = frame_bgr[y0_src:]
        for crop_h, crop_w, buf in self._levels:
            cv2.resize(
                src[:crop_h, :crop_w],
                (buf.shape[1], buf.shape[0]),
                dst=buf,
                interpolation=cv2.INTER_AREA,
            )
            src = buf
        cv2.resize(src, (proc_w, band_h), dst=self._small, interpolation=cv2.INTER_AREA)
        cv2.cvtColor(self._small, cv2.COLOR_BGR2HSV, dst=self._hsv)

        # Red wraps the hue circle, so it takes two bands; sat_min/val_min keep grey floor and
        # dark shadow out, both of which drift through the low-hue band at low saturation.
        cv2.inRange(
            self._hsv,
            (cfg.hue_band_a[0], cfg.sat_min, cfg.val_min),
            (cfg.hue_band_a[1], 255, 255),
            dst=self._mask_a,
        )
        cv2.inRange(
            self._hsv,
            (cfg.hue_band_b[0], cfg.sat_min, cfg.val_min),
            (cfg.hue_band_b[1], 255, 255),
            dst=self._mask_b,
        )
        cv2.bitwise_or(self._mask_a, self._mask_b, dst=self._red)
        # Open kills single-pixel speckle from chroma subsampling, which otherwise flickers
        # a few "pools" per frame at the min_area boundary.
        cv2.morphologyEx(self._red, cv2.MORPH_OPEN, self._kernel, dst=self._red)

        sx, sy = self._sx, self._sy
        boxes = self._band_boxes(exclude_boxes, sx, sy, y0_src, band_h, proc_w)
        for bx1, by1, bx2, by2 in boxes:
            self._red[by1:by2, bx1:bx2] = 0

        # --- priming: learn the room, report nothing ---------------------------------------
        # A running mean (alpha 1/n) rather than bg_alpha, so the baseline actually reaches
        # the room's true red occupancy within prime_frames instead of being 45% of the way
        # there when reporting switches on.
        if self._primed < cfg.prime_frames:
            self._primed += 1
            cv2.accumulateWeighted(self._red, self._bg, 1.0 / self._primed)
            # Score the novelty anyway and carry it forward. The lighting guard works on the
            # CHANGE in novelty between updates, so if priming left prev at zero the first
            # scored frame would read as a jump from nothing and every clip would open on a
            # fake lighting event. Against a settled baseline this is near zero, which is the
            # honest starting point.
            cv2.compare(self._bg, BASELINE_RED_MAX, cv2.CMP_LT, dst=self._fresh)
            cv2.bitwise_and(self._red, self._fresh, dst=self._prev_novel)
            self._prev_novel_frac = cv2.countNonZero(self._prev_novel) / (band_h * proc_w)
            return SubstanceState(state="idle", priming=True)

        # --- NOVELTY -----------------------------------------------------------------------
        cv2.compare(self._bg, BASELINE_RED_MAX, cv2.CMP_LT, dst=self._fresh)
        cv2.bitwise_and(self._red, self._fresh, dst=self._novel)

        band_px = band_h * proc_w
        novel_px = cv2.countNonZero(self._novel)
        novel_frac = novel_px / band_px

        lighting = self._lighting_event(now, novel_frac, novel_px, band_px)
        self._adapt_baseline(now, boxes, lighting)
        self._prev_novel_frac = novel_frac
        np.copyto(self._prev_novel, self._novel)

        if lighting or now < self._relearn_until:
            # An untrusted frame must not confirm and must not release: freezing the state is
            # the only honest thing to do with a frame whose colours cannot be believed.
            return SubstanceState(
                state=self._state, held_s=self._held, priming=False, n_pools=0, polygons=()
            )

        # --- pools -------------------------------------------------------------------------
        min_area_px = cfg.min_area_frac * proc_w * proc_h
        contours, _ = cv2.findContours(self._novel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pools = [(cv2.contourArea(c), c) for c in contours]
        pools = [p for p in pools if p[0] >= min_area_px]
        pools.sort(key=lambda p: p[0], reverse=True)
        pools = pools[: cfg.max_pools]

        detected = bool(pools)
        area_frac = sum(a for a, _ in pools) / (proc_w * proc_h)

        # The footprint, interiors filled: what the pools OCCUPY rather than which of their
        # pixels happen to read as red this frame. Only pools that passed min_area go in, so
        # speckle can never seed a hold.
        self._filled[:] = 0
        if detected:
            cv2.drawContours(self._filled, [c for _, c in pools], -1, 255, -1)

        # --- PERSISTENCE, measured against the footprint the candidate started with ---------
        if detected and self._active_since is not None:
            cv2.bitwise_and(self._filled, self._seed, dst=self._tmp)
            inter = cv2.countNonZero(self._tmp)
            seed_px = cv2.countNonZero(self._seed)
            ref = min(cv2.countNonZero(self._filled), seed_px)
            if ref == 0 or inter < SEED_OVERLAP_MIN * ref:
                # The red that is here now is not the red that started the clock: something
                # moved. Re-seed rather than inherit a hold this footprint never earned.
                self._active_since = now
                self._below_since = None
                self._seen = 0
                self._updates = 0
                np.copyto(self._seed, self._filled)
        elif detected:
            np.copyto(self._seed, self._filled)

        return self._advance(now, detected, area_frac, pools, sx, sy, y0_src)

    # -- helpers ---------------------------------------------------------------------------

    @staticmethod
    def _band_boxes(exclude_boxes, sx, sy, y0_src, band_h, proc_w):
        """Display-coordinate xyxy boxes clipped into floor-band processing coordinates."""
        out = []
        for b in exclude_boxes:
            x1, y1, x2, y2 = (float(v) for v in b)
            bx1 = max(0, min(proc_w, int(x1 / sx)))
            bx2 = max(0, min(proc_w, round(x2 / sx)))
            by1 = max(0, min(band_h, int((y1 - y0_src) / sy)))
            by2 = max(0, min(band_h, round((y2 - y0_src) / sy)))
            if bx2 > bx1 and by2 > by1:
                out.append((bx1, by1, bx2, by2))
        return out

    def _lighting_event(self, now: float, novel_frac: float, novel_px: int, band_px: int) -> bool:
        """True when this frame's novelty is a camera response, not liquid: a large jump that
        happened in ONE update and is spread across the band rather than pooled anywhere."""
        if now < self._relearn_until:
            return False  # already inside a declared event; do not re-trigger on itself
        if novel_frac - self._prev_novel_frac < LIGHTING_JUMP_FRAC:
            return False
        if novel_px == 0:
            return False
        _bx, _by, bw, bh = cv2.boundingRect(self._novel)
        if (bw * bh) / band_px < LIGHTING_SPREAD_FRAC:
            return False
        # Snapshot the novelty from before the event ONCE. Using the live mask instead
        # would exclude the whole lifted band from relearning, so the baseline would
        # never catch up and every later frame would read as one enormous pool.
        np.copyto(self._preevent_novel, self._prev_novel)
        self._relearn_until = now + LIGHTING_RELEARN_S
        return True

    def _adapt_baseline(self, now: float, boxes, lighting: bool) -> None:
        """Move the baseline toward the current mask, but never on pixels whose novelty is
        the thing we are trying to report, and never behind an excluded person."""
        if lighting or now < self._relearn_until:
            # The exposure genuinely changed, so the room's baseline is stale and must catch
            # up fast. Everything except a pool that was ALREADY novel before the change:
            # relearning those pixels is precisely how a real spill would be erased by
            # someone walking past it.
            cv2.bitwise_not(self._preevent_novel, dst=self._adapt)
            alpha = LIGHTING_RELEARN_ALPHA
        else:
            # FREEZE ON NOVEL. Without this the baseline absorbs a stationary pool at
            # bg_alpha per frame and the pool stops being novel within seconds, which is the
            # one failure that would look like a working detector right up until it matters.
            cv2.bitwise_not(self._novel, dst=self._adapt)
            alpha = self.cfg.bg_alpha
        # A person's own clothing must not be written into the room's baseline, and the red
        # fixture they are standing in front of must not be UNLEARNED while they occlude it
        # (the mask reads 0 there, which would decay the ladder's baseline to zero and make
        # it novel the moment they step away).
        for bx1, by1, bx2, by2 in boxes:
            self._adapt[by1:by2, bx1:bx2] = 0
        cv2.accumulateWeighted(self._red, self._bg, alpha, mask=self._adapt)

    def _advance(self, now, detected, area_frac, pools, sx, sy, y0_src) -> SubstanceState:
        cfg = self.cfg
        if detected:
            self._below_since = None
            if self._active_since is None:
                self._active_since = now
                self._seen = 0
                self._updates = 0
            self._held = now - self._active_since
            self._seen += 1
            self._updates += 1
            if self._state != "confirmed":
                # Held long enough AND actually there for most of it. The second half is what
                # stops a blob flickering across the min_area boundary from riding release_s
                # to a confirmation it was never present for.
                duty = self._seen / self._updates
                ready = self._held >= cfg.min_hold_s and duty >= DUTY_MIN
                self._state = "confirmed" if ready else "candidate"
        elif self._active_since is not None:
            # A miss inside a live episode still counts against the duty cycle; only a full
            # release_s clears the episode and with it the count.
            self._updates += 1
            if self._below_since is None:
                self._below_since = now
            # A pool can vanish for a frame behind someone walking over it, so release is a
            # timer, not an instant.
            if now - self._below_since >= cfg.release_s:
                self._active_since = None
                self._below_since = None
                self._state = "idle"
                self._held = 0.0
                self._seen = 0
                self._updates = 0
                self._area_hist.clear()
        else:
            self._state = "idle"
            self._held = 0.0

        growth = self._growth(now, area_frac if detected else 0.0)
        if self._state == "idle":
            return SubstanceState(state="idle")
        return SubstanceState(
            state=self._state,
            area_frac=float(area_frac),
            n_pools=len(pools),
            held_s=float(self._held),
            growth_frac_per_s=growth,
            polygons=tuple(self._to_display(c, sx, sy, y0_src) for _, c in pools),
            priming=False,
        )

    def _growth(self, now: float, area_frac: float) -> float:
        self._area_hist.append((now, area_frac))
        cut = now - GROWTH_WINDOW_S
        while len(self._area_hist) > 2 and self._area_hist[0][0] < cut:
            self._area_hist.popleft()
        t0, a0 = self._area_hist[0]
        dt = now - t0
        if dt <= 1e-6:
            return 0.0
        return float((area_frac - a0) / dt)

    @staticmethod
    def _to_display(contour: np.ndarray, sx: float, sy: float, y0_src: int) -> np.ndarray:
        """Band processing coordinates back to display pixels, for the compositor to draw."""
        poly = contour.astype(np.float32)
        poly[:, :, 0] *= sx
        poly[:, :, 1] = poly[:, :, 1] * sy + y0_src
        return poly.astype(np.int32)

    def state_name(self) -> str:
        return self._state
