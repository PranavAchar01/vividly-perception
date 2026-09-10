"""Red substance on a tracked person: a colour cone, novelty against the person, held over time.

Produces the `red_substance_on_person` descriptor. The floor detector beside this one
(`substance.py`) learns the room per pixel because a floor does not move. A person does, so
nothing about the PERSON is learned per pixel in image space. Five independent facts separate
a substance on a person from the person and from the room, and all five are required before
anything is reported:

  1. COLOUR. Skin is reddish; what skin never has is chroma that is large relative to its
     lightness. The gate is a cone in CIE Lab rather than a hue band: a* at least `a_min`,
     a* at least `a_per_l` times L*, a hue angle inside the red sector, and a value ceiling
     `l_max` because the thing this exists for is dark. Calibrated on 438,598 skin pixels
     sampled at pose locations (nose, eyes, ears, hands; every third frame of both takes of
     this camera, at proc scale, clear of the step ladder's columns). Skin a* on this camera:
     p50 0, p99 16, p99.9 20, max 28, against a floor of 22 that rises with lightness (34 at
     L* 40). Zero nose, eye or ear pixels pass; 17 hand pixels do (0.004%, scattered, never
     a region). Per pixel, the a* shortfall of in-sector skin is 2.5 units at p0.1 and 6 at
     p1. Measured as regions, the margin is wider: with the cone loosened to a* >= 18,
     a*/L* >= 0.6, L* <= 65 the whole pipeline still finds nothing on either take.
  2. PLACE. Only pixels inside the person's own body mask count: capsules drawn between their
     keypoints and sized from their own skeleton, labelled by body part. The box alone would
     carry the room into the person.
  3. NOT THE ROOM. This camera looks at a red step ladder that, in this light, is as dark and
     as saturated as the substance (measured L* 13, a* 30, hue 31 deg): no colour threshold
     can separate them, and it is the ONLY thing that ever put a region on a person on these
     takes. So the room's own red is learned the way the floor detector learns it, a slow
     per-pixel occupancy baseline at quarter scale, updated only OUTSIDE people, and red on a
     person that sits on the room's static red is not scored. Without it take1 confirms a
     "head" hit at 57 s on the ladder behind someone's head; with it, nothing.
  4. NOVELTY. Red that passes all of the above is scored only if it is new for that track.
     For the first `prime_s` of each track, per body part, a coarse Lab histogram of the
     person is learned; a red garment or a red cap fills bins that later suppress it. Per
     part on purpose: a red shirt must not teach the head that red is normal. A new tracker
     id on the same person inherits the model instead of priming (see _donor), because
     priming at the moment of the event would learn the event.
  5. COHERENCE AND PERSISTENCE. A connected region of at least `min_area_frac` of the frame,
     on the same body part for `min_hold_s`, present in at least DUTY_MIN of the updates since
     its clock started. A compression fringe lasts a frame; skin that brushes the cone under
     a passing highlight comes and goes.

A camera colour response (auto white balance, exposure) is detected from the room, outside
people, and makes an update untrusted: it neither confirms nor releases, and both the room
and any person without a live episode re-learn what the camera now shows.

Vocabulary is observational by design: this module reports red on a person and never says
what the red is.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# --- mechanism constants (not tuning knobs, so they stay out of the fixed config) -----------

# Body part labels drawn into the per-person label map. Later draws override earlier ones,
# so the drawing order is also the priority: a raised hand in front of the face is a hand.
_BG, _HEAD, _HAND, _ARM, _TORSO, _LEG = 0, 1, 2, 3, 4, 5
_LABEL_NAME = {_HEAD: "head", _HAND: "hand", _ARM: "arm", _TORSO: "torso", _LEG: "leg"}
_N_LABELS = 6

# Parts that are one PLACE for persistence. Red that starts on the scalp and runs down the
# forehead changes the nearest keypoint from ear to eye without having moved anywhere, and a
# hand is the end of an arm; neither may restart a clock or drop a confirmation.
_PLACE = {
    "head": "head",
    "face": "head",
    "hand": "arm",
    "arm": "arm",
    "torso": "torso",
    "leg": "leg",
}

# COCO-17 indices.
_NOSE, _LEYE, _REYE, _LEAR, _REAR = 0, 1, 2, 3, 4
_LSH, _RSH, _LEL, _REL, _LWR, _RWR = 5, 6, 7, 8, 9, 10
_LHIP, _RHIP, _LKN, _RKN, _LAN, _RAN = 11, 12, 13, 14, 15, 16
_HEAD_KP = (_NOSE, _LEYE, _REYE, _LEAR, _REAR)

# Capsule geometry in units of the person's head width `u` (see _unit). Generous on purpose:
# a stray pixel at the edge of a sleeve costs nothing, a missed patch on the scalp costs the
# event. The room that leaks in through the margin is handled by the static-red map.
_HEAD_R = 0.95  # head circle radius: scalp and face with a hair margin
_HEAD_UP = 0.25  # the circle's centre sits this far above the eye/ear centre, along "up"
_ARM_W = 0.75
_LEG_W = 0.95
_TORSO_PAD = 0.30
_HAND_R = 0.55
_HAND_REACH = 0.40  # hand centre: this fraction of the forearm beyond the wrist
# Above the eye line by this much (along the person's own up direction) is scalp or
# forehead, reported as "head" whichever keypoint happens to be nearest.
_SCALP_ABOVE_EYES = 0.15

# Per-part appearance model: a coarse Lab histogram, 8 lightness x 32 x 32 chroma bins.
_L_SHIFT, _AB_SHIFT = 5, 3
_L_BINS, _AB_BINS = 256 >> _L_SHIFT, 256 >> _AB_SHIFT
_N_BINS = _L_BINS * _AB_BINS * _AB_BINS
# A bin belongs to the person once its 3 x 3 x 3 neighbourhood held this fraction of that
# part's pixels during priming. 1% is garment scale: a red shirt fills far more of a torso
# and a pair of lips far less of a head. The neighbourhood is what keeps a learned colour
# learned when the light on it shifts by a bin.
_LEARN_FRAC = 0.01
# A part is primed only after it has been seen in this many updates with at least this many
# pixels. A head that was turned away for the whole priming window is not "a head with no
# red on it"; it is unknown, and an unknown part reports nothing until it has been learned.
_PRIME_MIN_UPDATES = 5
_PRIME_MIN_PX = 150

# The room's static red, learned at half of proc_scale (quarter scale at the defaults).
# Running mean for the first _SCENE_PRIME updates, then a slow alpha: the step ladder does not
# move, and a slow room is what keeps a person who stood still and unposed for a moment from
# being written into it. People, and the last place a recently lost person was, are never
# learned at all.
_SCENE_PRIME = 20
_SCENE_ALPHA = 0.01
_SCENE_STATIC = 0.30 * 255.0  # red in at least 30% of the frames it was visible in
_SCENE_EXCLUDE_S = 3.0  # a lost track's last box stays out of the room this long
# An update in which red appears OUTSIDE people over more than this fraction of the frame at
# once is a camera response (auto white balance, exposure), not a substance: the whole room
# cannot turn red in one frame. Such a frame is untrusted: it neither confirms nor releases,
# and the room re-learns fast for _RELEARN_S. A head-sized region on this camera is 0.1% to
# 0.3% of the frame, and it is on a person, which this measure excludes by construction.
_LIGHTING_JUMP_FRAC = 0.005
# ...or in which the room's mean chroma (outside people, in a*/b* units) moves this far from
# its slow average: white balance measured directly, which catches a swing that turns skin
# red without pushing the room itself into the cone. Measured over both takes the room mean
# never strayed more than 0.16 a* / 0.29 b* from its average, so 2.0 is about 7x clear of
# the quietest room this camera has shown, and a swing large enough to put skin into the
# cone moves it by 10 or more.
_SWING_AB = 2.0
_AB_ALPHA = 0.02
_RELEARN_S = 1.0
_SWING_LAG_S = 0.5  # a candidate this young when a swing is detected is the swing's
_RELEARN_ALPHA = 0.25

# Fraction of the updates since a candidate's clock started in which the region was actually
# there (see substance.DUTY_MIN for the measurement behind the value).
_DUTY_MIN = 0.80
# Duty is measured over a recent window, not the whole episode. Counting from the first frame
# meant a candidate that began sparse (a thin trickle that flickers in and out of the hand's
# shadow) could never reach _DUTY_MIN however steady it later became, because its early
# misses stayed in the denominator forever. The window is at least as long as the hold, so
# confirmation still needs the red present on DUTY_MIN of the updates across the hold itself.
_DUTY_WINDOW_MIN_S = 1.0
_GROWTH_WINDOW_S = 2.0

# Identity hand-over between tracker ids (see _donor): how recently the old id must have been
# seen, how much the boxes must overlap, and how close the heads must be in head widths.
_INHERIT_GAP_S = 3.0
_INHERIT_IOU = 0.30
_INHERIT_HEAD_U = 0.8

# A track not seen for this long is dropped even if the caller never calls forget(), so a
# long session cannot accumulate dead tracks. Longer than a typical gap, so a person who
# steps out of view for a moment keeps what was learned about them.
_AUTO_FORGET_S = 60.0


@dataclass(frozen=True)
class PersonRedConfig:
    proc_scale: float = 0.5  # analysis scale relative to the display frame (1280 -> 640)
    # Colour cone, CIE Lab in real units (L* 0..100, a*/b* signed). Margins in the docstring.
    a_min: float = 22.0  # a* floor; skin p99.9 on this camera is 20
    a_per_l: float = 0.85  # a* >= a_per_l * L*: dark AND saturated
    hue_min_deg: float = 8.0  # atan2(b*, a*) floor: excludes magenta and pink
    hue_max_deg: float = 50.0  # ceiling: excludes orange and yellow
    l_max: float = 55.0  # value ceiling, L*
    open_kernel: int = 3  # morphological open at proc scale, kills chroma speckle
    min_area_frac: float = 0.00005  # of the full frame; 46 display px at 1280 x 720
    prime_s: float = 2.0  # learn the person before reporting anything on them
    min_hold_s: float = 1.5
    release_s: float = 1.0
    duty_min: float = _DUTY_MIN  # share of recent updates the region must be present in
    box_pad: float = 0.10  # crop margin around the tracked box, fraction of its size
    kp_thr: float = 0.30  # keypoint confidence below which a joint is not used
    max_regions: int = 3  # largest N regions per person reported


@dataclass(frozen=True)
class PersonRedHit:
    tid: int
    state: str  # "candidate" | "confirmed"
    part: str  # "head" | "face" | "hand" | "arm" | "torso" | "leg"
    area_frac: float  # of the full frame
    held_s: float
    growth_frac_per_s: float
    polygons: tuple = ()  # DISPLAY coords, each (N,1,2) int32


@dataclass(frozen=True)
class PersonRedState:
    hits: tuple = ()
    priming: bool = False


@dataclass
class _Track:
    first_t: float
    last_seen: float
    last_box: np.ndarray | None = None
    hist: np.ndarray = field(default_factory=lambda: np.zeros((_N_LABELS, _N_BINS), np.float32))
    prime_px: np.ndarray = field(default_factory=lambda: np.zeros(_N_LABELS, np.int64))
    prime_updates: np.ndarray = field(default_factory=lambda: np.zeros(_N_LABELS, np.int32))
    primed: np.ndarray = field(default_factory=lambda: np.zeros(_N_LABELS, bool))
    learned: np.ndarray = field(default_factory=lambda: np.zeros((_N_LABELS, _N_BINS), bool))
    state: str = "idle"
    active_since: float | None = None
    below_since: float | None = None
    seed_part: str = ""
    seen: int = 0
    updates: int = 0
    area_hist: deque = field(default_factory=deque)
    duty_hist: deque = field(default_factory=deque)  # (t, detected) over the duty window
    last_part: str = ""
    last_area: float = 0.0
    last_head: np.ndarray | None = None  # head keypoint centre, display px
    last_u: float = 0.0  # head width, display px

    def inherit(self, donor: _Track) -> None:
        """Take over a donor's appearance model and any live episode (see _donor)."""
        self.first_t = donor.first_t
        self.hist = donor.hist.copy()
        self.prime_px = donor.prime_px.copy()
        self.prime_updates = donor.prime_updates.copy()
        self.primed = donor.primed.copy()
        self.learned = donor.learned.copy()
        if donor.state != "idle":
            self.take_episode(donor)

    def take_episode(self, other: _Track) -> None:
        self.state = other.state
        self.active_since = other.active_since
        self.below_since = other.below_since
        self.seed_part = other.seed_part
        self.seen, self.updates = other.seen, other.updates
        self.area_hist = deque(other.area_hist)
        self.duty_hist = deque(other.duty_hist)
        self.last_part, self.last_area = other.last_part, other.last_area

    def rank(self) -> tuple:
        """Episode seniority: confirmed over candidate over idle, then the older clock."""
        lvl = {"confirmed": 2, "candidate": 1}.get(self.state, 0)
        return (lvl, -(self.active_since if self.active_since is not None else 1e18))


class PersonRedAnalyzer:
    """Streaming, per track. Feed it the display frame and the posed people once per update."""

    def __init__(self, cfg: PersonRedConfig | None = None) -> None:
        self.cfg = cfg or PersonRedConfig()
        k = max(1, int(self.cfg.open_kernel))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        c = self.cfg
        self._tan_lo = float(np.tan(np.radians(c.hue_min_deg)))
        self._tan_hi = float(np.tan(np.radians(c.hue_max_deg)))
        self._l8_max = int(np.floor(c.l_max * 255.0 / 100.0))
        self._a8_min = int(np.ceil(128.0 + c.a_min))
        self._apl8 = c.a_per_l * 100.0 / 255.0  # a* per 8-bit L unit
        # proc_scale = 2 x scene scale, so a proc pixel maps onto the room map by halving.
        self._scene_scale = c.proc_scale / 2.0
        self.reset()

    # -- lifecycle -------------------------------------------------------------------------

    def reset(self) -> None:
        self._tracks: dict[int, _Track] = {}
        self._scene_shape: tuple | None = None
        self._scene_n = 0
        self._scene_bg: np.ndarray | None = None
        self._scene_static: np.ndarray | None = None
        self._prev_room_new = 0.0
        self._room_ab = np.zeros(2)
        self._relearn_until = -1e18
        self._relearn_started = -1e18

    def forget(self, tid: int) -> None:
        self._tracks.pop(int(tid), None)

    # -- main ------------------------------------------------------------------------------

    def update(self, frame_bgr, now: float, people) -> PersonRedState:
        people = [
            (
                int(t),
                np.asarray(b, np.float32).reshape(4),
                np.asarray(k, np.float32).reshape(17, 2),
                np.asarray(s, np.float32).reshape(17),
            )
            for t, b, k, s in people
        ]
        trusted = self._room(frame_bgr, now, people)
        if self._relearn_started == now:
            # The room can see a swing an update or two late (its mean moves over a couple of
            # frames), and red that a swing painted onto skin in those frames must not keep
            # the hold it seeded. A candidate born that close before the swing is dropped and
            # re-learned with everything else; a confirmed episode is never touched.
            for tr in self._tracks.values():
                if (
                    tr.state == "candidate"
                    and tr.active_since is not None
                    and now - tr.active_since <= _SWING_LAG_S
                ):
                    self._clear_episode(tr)
        scene_ready = self._scene_n >= _SCENE_PRIME
        frame_px = float(frame_bgr.shape[0] * frame_bgr.shape[1])

        hits: list[PersonRedHit] = []
        priming = not scene_ready
        seen: set[int] = set()
        work = []
        for tid, box, kp, sc in people:
            seen.add(tid)
            tr = self._tracks.get(tid)
            ok = sc >= self.cfg.kp_thr
            head = [i for i in _HEAD_KP if ok[i]]
            head_c = kp[head].mean(axis=0) if head else None
            if tr is None:
                tr = _Track(first_t=now, last_seen=now)
                donor = self._donor(tid, box, head_c, now)
                if donor is not None:
                    tr.inherit(donor)
                self._tracks[tid] = tr
            tr.last_seen = now
            tr.last_box = box
            tr.last_head, tr.last_u = head_c, _unit(kp, ok, box)
            regions, is_priming = self._person(
                tr, frame_bgr, now, box, kp, sc, frame_px, scene_ready, not trusted
            )
            priming = priming or is_priming
            work.append((tid, tr, regions))

        twins = self._sync_twins(work, now)
        for tid, tr, regions in work:
            hit = self._advance(tr, tid, now, regions, trusted)
            if hit is not None and tid not in twins:
                hits.append(hit)
            elif hit is not None:
                twins[tid].append(hit)
        # One hit per person: of a set of twin ids, the hit that measured something this
        # update wins, the larger region breaking a tie.
        done: set[int] = set()
        for group in twins.values():
            if id(group) in done or not group:
                continue
            done.add(id(group))
            hits.append(max(group, key=lambda h: (bool(h.polygons), h.area_frac)))

        # Episodes already reported this update through a present id, so a twin that missed
        # this frame does not report the same episode a second time.
        live = {tr.active_since for _t, tr, _r in work if tr.state != "idle"}
        for tid in list(self._tracks):
            if tid in seen:
                continue
            tr = self._tracks[tid]
            gone = now - tr.last_seen
            if gone > _AUTO_FORGET_S:
                del self._tracks[tid]
                continue
            # Absent from this update: no evidence either way, so the state is frozen, not
            # advanced. A confirmed person stays on the banner (with nothing drawn: there is
            # no fresh position to draw at) for release_s, so a missed pose frame does not
            # blink the alert; after that the episode clears.
            if tr.state != "idle" and gone > self.cfg.release_s:
                self._clear_episode(tr)
            elif tr.state == "confirmed" and tr.active_since not in live:
                hits.append(
                    PersonRedHit(
                        tid=tid,
                        state="confirmed",
                        part=tr.last_part or tr.seed_part,
                        area_frac=tr.last_area,
                        held_s=float(now - (tr.active_since or now)),
                        growth_frac_per_s=0.0,
                        polygons=(),
                    )
                )
        return PersonRedState(hits=tuple(hits), priming=priming)

    def _sync_twins(self, work, now: float) -> dict[int, list]:
        """Ids that are the same person share one episode.

        The tracker can hold two ids on one person for seconds (on the 205301 take a second,
        lower box appears as the person goes down, and both are posed), and it can hand a
        person back to an id it lost a while ago. Each id would run its own clock on the same
        red, and which id happens to be posed on a given frame would then decide when the
        alert confirms and whether it blinks. So a posed id takes the senior episode of any
        id that is the same person, posed now or seen within _INHERIT_GAP_S, before advancing
        on its own evidence. Returns {tid: shared hit list} for every POSED id in a twin set,
        for the caller to report once.
        """
        twins: dict[int, list] = {}
        posed = {tid for tid, _tr, _r in work}
        recent = [
            (tid, tr)
            for tid, tr in self._tracks.items()
            if tid not in posed and tr.state != "idle" and now - tr.last_seen <= _INHERIT_GAP_S
        ]
        for _ta, tra, _ra in work:
            for _tb, trb in recent:
                if _same_person(tra, trb) and tra.rank() < trb.rank():
                    tra.take_episode(trb)
        for a in range(len(work)):
            for b in range(a + 1, len(work)):
                (ta, tra, _ra), (tb, trb, _rb) = work[a], work[b]
                if not _same_person(tra, trb):
                    continue
                lead, follow = (tra, trb) if tra.rank() >= trb.rank() else (trb, tra)
                if lead.state != "idle" and follow.rank() < lead.rank():
                    follow.take_episode(lead)
                shared = twins[ta] if ta in twins else twins.get(tb, [])
                twins[ta] = twins[tb] = shared
        return twins

    def _donor(self, tid: int, box, head_c, now: float) -> _Track | None:
        """An existing track that a NEW track id is the same person as, or None.

        The tracker does not keep one id per person. On the 205301 take a person going down to
        the floor gets a second, lower box and a new id while the first id is still alive,
        and the first id dies two seconds later; that is an ordinary tracker event, and it
        lands exactly at the moment this detector matters. A new id that primed from scratch
        would LEARN whatever is on the person at that moment as their normal appearance and
        could never report it. So a new id that sits on top of a recently seen, primed track
        (heads within _INHERIT_HEAD_U head widths and boxes at IoU _INHERIT_IOU or more)
        takes over that track's appearance model and its live episode instead of priming.
        Both conditions,
        because two people crossing overlap in box but not in head position, and a person
        bending over another overlaps in head position but not in box.
        """
        best, best_iou = None, 0.0
        for otid, o in self._tracks.items():
            if otid == tid or not o.primed.any() or o.last_box is None:
                continue
            if now - o.last_seen > _INHERIT_GAP_S:
                continue
            iou = _iou(box, o.last_box)
            if iou < _INHERIT_IOU:
                continue
            if head_c is None or o.last_head is None:
                continue
            if float(np.hypot(*(head_c - o.last_head))) > _INHERIT_HEAD_U * max(o.last_u, 1.0):
                continue
            if iou > best_iou:
                best, best_iou = o, iou
        return best

    # -- the room --------------------------------------------------------------------------

    def _room(self, frame, now, people) -> bool:
        """Maintain the room's static-red map. Returns False when this update is untrusted."""
        h, w = frame.shape[:2]
        ss = self._scene_scale
        sw, sh = max(1, round(w * ss)), max(1, round(h * ss))
        if self._scene_shape != (h, w):
            self._scene_shape = (h, w)
            self._scene_bg = np.zeros((sh, sw), np.float32)
            self._scene_static = np.zeros((sh, sw), np.uint8)
            self._scene_n = 0
            self._prev_room_new = 0.0
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)
        red = self._gate(lab)

        # Never learn where a person is, or where one was a moment ago.
        adapt = np.full((sh, sw), 255, np.uint8)
        boxes = [b for _t, b, _k, _s in people]
        boxes += [
            tr.last_box
            for tr in self._tracks.values()
            if tr.last_box is not None and 0.0 < now - tr.last_seen <= _SCENE_EXCLUDE_S
        ]
        for b in boxes:
            x1, y1, x2, y2 = (float(v) for v in b)
            pw, ph = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            bx1, by1 = max(0, int((x1 - pw) * ss)), max(0, int((y1 - ph) * ss))
            bx2, by2 = min(sw, int(np.ceil((x2 + pw) * ss))), min(sh, int(np.ceil((y2 + ph) * ss)))
            if bx2 > bx1 and by2 > by1:
                adapt[by1:by2, bx1:bx2] = 0

        # The room's mean chroma, outside people: the camera's white balance, measured.
        m = cv2.mean(lab, mask=adapt) if cv2.countNonZero(adapt) else (0.0, 128.0, 128.0)
        ab = np.array([m[1] - 128.0, m[2] - 128.0])

        trusted = True
        if self._scene_n < _SCENE_PRIME:
            self._scene_n += 1
            cv2.accumulateWeighted(red, self._scene_bg, 1.0 / self._scene_n, mask=adapt)
            self._room_ab = (
                ab if self._scene_n == 1 else self._room_ab + (ab - self._room_ab) / self._scene_n
            )
        else:
            # Red outside people that the room does not already hold.
            new = cv2.bitwise_and(red, adapt)
            new[self._scene_bg >= _SCENE_STATIC] = 0
            room_new = cv2.countNonZero(new) / float(sh * sw)
            jump = room_new - self._prev_room_new >= _LIGHTING_JUMP_FRAC
            shift = float(np.hypot(*(ab - self._room_ab))) >= _SWING_AB
            if now >= self._relearn_until and (jump or shift):
                self._relearn_until = now + _RELEARN_S
                self._relearn_started = now
            self._prev_room_new = room_new
            relearn = now < self._relearn_until
            trusted = not relearn
            alpha = _RELEARN_ALPHA if relearn else _SCENE_ALPHA
            cv2.accumulateWeighted(red, self._scene_bg, alpha, mask=adapt)
            self._room_ab = self._room_ab + (_RELEARN_ALPHA if relearn else _AB_ALPHA) * (
                ab - self._room_ab
            )
        static = (self._scene_bg >= _SCENE_STATIC).astype(np.uint8) * 255
        # One quarter-scale pixel of margin: the fringe a codec smears around a red edge, and
        # a pixel of camera shake.
        cv2.dilate(static, self._kernel, dst=self._scene_static)
        return trusted

    # -- per person ------------------------------------------------------------------------

    def _person(self, tr, frame, now, box, kp, sc, frame_px, scene_ready, relearn=False):
        """Returns (regions, priming). Regions are (area_frac, part, display polygons)."""
        cfg = self.cfg
        s = cfg.proc_scale
        ok = sc >= cfg.kp_thr
        if int(ok.sum()) < 4:
            return [], False
        crop = self._crop_rect(frame.shape, box, kp, ok)
        if crop is None:
            return [], False
        px0, py0, px1, py1 = crop
        inv = 1.0 / s
        src = frame[round(py0 * inv) : round(py1 * inv), round(px0 * inv) : round(px1 * inv)]
        cw, ch = px1 - px0, py1 - py0
        small = cv2.resize(src, (cw, ch), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)
        kpp = kp * s - np.array([px0, py0], np.float32)
        labels, geo = self._label_map((ch, cw), kpp, ok, box * s)

        self._learn(tr, lab, labels, now)
        if not scene_ready or not tr.primed.any():
            return [], True

        # COLOUR, inside the body only.
        gate = self._gate(lab)
        gate[labels == _BG] = 0
        # NOT THE ROOM: drop red that sits on the room's static red.
        sx0, sy0 = px0 // 2, py0 // 2
        roi = self._scene_static[sy0 : (py1 + 1) // 2 + 1, sx0 : (px1 + 1) // 2 + 1]
        up = np.repeat(np.repeat(roi, 2, axis=0), 2, axis=1)[py0 - 2 * sy0 :, px0 - 2 * sx0 :]
        up = up[:ch, :cw]
        gate[: up.shape[0], : up.shape[1]][up > 0] = 0
        cv2.morphologyEx(gate, cv2.MORPH_OPEN, self._kernel, dst=gate)

        ys, xs = np.nonzero(gate)
        if len(ys) == 0:
            return [], False
        # NOVELTY, looked up only for the pixels that are still in.
        lb = labels[ys, xs]
        idx = _bin_index(lab[ys, xs].astype(np.int32))
        if relearn and tr.state == "idle":
            # The camera's colour response just changed (see _room), and whatever it now
            # shows on this person is their new normal: the room is re-learning the same
            # way. A track with a live episode is left alone, so a swing cannot erase the
            # event it arrived during.
            for part in np.unique(lb):
                if tr.primed[part]:
                    _mark_bins(tr.learned[part], idx[lb == part])
        keep = tr.primed[lb] & ~tr.learned[lb, idx]
        gate[ys[~keep], xs[~keep]] = 0

        # COHERENCE.
        proc_frame_px = frame_px * s * s
        min_px = max(4.0, cfg.min_area_frac * proc_frame_px)
        n, cc, stats, cent = cv2.connectedComponentsWithStats(gate, connectivity=8)
        comps = sorted(
            (
                (int(stats[i, cv2.CC_STAT_AREA]), i)
                for i in range(1, n)
                if stats[i, cv2.CC_STAT_AREA] >= min_px
            ),
            reverse=True,
        )
        regions = []
        off = np.array([px0, py0], np.float32)
        for area, i in comps[: cfg.max_regions]:
            m = (cc == i).astype(np.uint8)
            part = _part_of(m, labels, cent[i], kpp, ok, geo)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = tuple(((c.astype(np.float32) + off) * inv).astype(np.int32) for c in cs)
            regions.append((area / proc_frame_px, part, polys))
        return regions, False

    def _crop_rect(self, shape, box, kp, ok):
        """The processing crop in PROC pixels: the tracked box padded and grown to include the
        head circle (red on the scalp sits above a box that may stop at the eyes)."""
        cfg = self.cfg
        h, w = shape[:2]
        s = cfg.proc_scale
        x1, y1, x2, y2 = (float(v) for v in box)
        pw, ph = cfg.box_pad * (x2 - x1), cfg.box_pad * (y2 - y1)
        x1, y1, x2, y2 = x1 - pw, y1 - ph, x2 + pw, y2 + ph
        pts = kp[ok]
        u = _unit(kp, ok, box)
        r = (_HEAD_R + _HEAD_UP + 0.2) * u
        x1, y1 = min(x1, float(pts[:, 0].min()) - r), min(y1, float(pts[:, 1].min()) - r)
        x2, y2 = max(x2, float(pts[:, 0].max()) + r), max(y2, float(pts[:, 1].max()) + r)
        px0 = max(0, int(np.floor(x1 * s)))
        py0 = max(0, int(np.floor(y1 * s)))
        px1 = min(int(w * s), int(np.ceil(x2 * s)))
        py1 = min(int(h * s), int(np.ceil(y2 * s)))
        if px1 - px0 < 4 or py1 - py0 < 4:
            return None
        return px0, py0, px1, py1

    def _label_map(self, hw, kpp, ok, box_p):
        """Body mask with part labels, at proc scale in crop coordinates."""
        labels = np.zeros(hw, np.uint8)
        u = _unit(kpp, ok, box_p)

        def pt(i):
            return (round(float(kpp[i, 0])), round(float(kpp[i, 1])))

        def seg(i, j, width, lab):
            if ok[i] and ok[j]:
                cv2.line(labels, pt(i), pt(j), lab, max(1, round(width * u)), cv2.LINE_8)

        quad = [i for i in (_LSH, _RSH, _RHIP, _LHIP) if ok[i]]
        if len(quad) >= 3:
            hull = cv2.convexHull(np.array([pt(i) for i in quad], np.int32))
            cv2.fillConvexPoly(labels, hull, _TORSO)
            cv2.polylines(labels, [hull], True, _TORSO, max(1, round(2 * _TORSO_PAD * u)))
        seg(_LSH, _RSH, 2 * _TORSO_PAD, _TORSO)
        seg(_LSH, _LHIP, 0.8, _TORSO)
        seg(_RSH, _RHIP, 0.8, _TORSO)
        for a, b, c in ((_LHIP, _LKN, _LAN), (_RHIP, _RKN, _RAN)):
            seg(a, b, _LEG_W, _LEG)
            seg(b, c, _LEG_W, _LEG)
        for sh_, el in ((_LSH, _LEL), (_RSH, _REL)):
            seg(sh_, el, _ARM_W, _ARM)
        for el, wr in ((_LEL, _LWR), (_REL, _RWR)):
            seg(el, wr, _ARM_W, _ARM)
            if ok[wr]:
                c = kpp[wr] + (_HAND_REACH * (kpp[wr] - kpp[el]) if ok[el] else 0.0)
                cv2.circle(
                    labels,
                    (round(float(c[0])), round(float(c[1]))),
                    max(1, round(_HAND_R * u)),
                    _HAND,
                    -1,
                )

        geo = {"u": u, "up": np.array([0.0, -1.0], np.float32), "eye_line": None}
        head = [i for i in _HEAD_KP if ok[i]]
        if head:
            hc = kpp[head].mean(axis=0)
            if ok[_LSH] and ok[_RSH]:
                v = hc - 0.5 * (kpp[_LSH] + kpp[_RSH])
                n = float(np.hypot(v[0], v[1]))
                if n > 1e-3:
                    geo["up"] = (v / n).astype(np.float32)
            c = hc + geo["up"] * (_HEAD_UP * u)
            cv2.circle(
                labels,
                (round(float(c[0])), round(float(c[1]))),
                max(2, round(_HEAD_R * u)),
                _HEAD,
                -1,
            )
            eyes = [i for i in (_LEYE, _REYE) if ok[i]]
            if eyes:
                geo["eye_line"] = kpp[eyes].mean(axis=0)
        return labels, geo

    # -- colour ----------------------------------------------------------------------------

    def _gate(self, lab: np.ndarray) -> np.ndarray:
        """The colour cone as a 0/255 uint8 mask, evaluated in OpenCV's 8-bit Lab.

        The box part of the cone (a* floor, L* ceiling) is one inRange over the whole image;
        the ratio and hue tests run only on the few pixels that survive it, which on this
        camera is well under 1% of a frame.
        """
        m = cv2.inRange(lab, (0, self._a8_min, 0), (self._l8_max, 255, 255))
        ys, xs = np.nonzero(m)
        if len(ys):
            v = lab[ys, xs].astype(np.float32)
            a = v[:, 1] - 128.0
            b = v[:, 2] - 128.0
            ok = (a >= self._apl8 * v[:, 0]) & (b >= self._tan_lo * a) & (b <= self._tan_hi * a)
            m[ys[~ok], xs[~ok]] = 0
        return m

    def _learn(self, tr: _Track, lab, labels, now: float) -> None:
        """Accumulate each part's appearance histogram until that part is primed."""
        if tr.primed[1:].all():
            return
        body = labels != _BG
        if not body.any():
            return
        lb = labels[body]
        idx = _bin_index(lab[body].astype(np.int32))
        for part in range(1, _N_LABELS):
            if tr.primed[part]:
                continue
            sel = lb == part
            n = int(sel.sum())
            if n == 0:
                continue
            tr.hist[part] += np.bincount(idx[sel], minlength=_N_BINS).astype(np.float32)
            tr.prime_px[part] += n
            tr.prime_updates[part] += 1
            if (
                now - tr.first_t >= self.cfg.prime_s
                and tr.prime_updates[part] >= _PRIME_MIN_UPDATES
                and tr.prime_px[part] >= _PRIME_MIN_PX
            ):
                tr.primed[part] = True
                sm = _box3(tr.hist[part].reshape(_L_BINS, _AB_BINS, _AB_BINS))
                tr.learned[part] = (sm / float(tr.prime_px[part]) >= _LEARN_FRAC).reshape(-1)
                tr.hist[part] = 0.0  # no longer needed; the learned bins are the model

    # -- state machine ---------------------------------------------------------------------

    def _note_duty(self, tr: _Track, now: float, detected: bool) -> None:
        tr.duty_hist.append((now, detected))
        horizon = now - max(self.cfg.min_hold_s, _DUTY_WINDOW_MIN_S)
        while len(tr.duty_hist) > 1 and tr.duty_hist[0][0] < horizon:
            tr.duty_hist.popleft()

    @staticmethod
    def _clear_episode(tr: _Track) -> None:
        tr.state = "idle"
        tr.active_since = None
        tr.below_since = None
        tr.seed_part = ""
        tr.seen = 0
        tr.updates = 0
        tr.area_hist.clear()
        tr.duty_hist.clear()
        tr.last_part = ""
        tr.last_area = 0.0

    def _advance(self, tr, tid, now, regions, trusted):
        cfg = self.cfg
        if not trusted:
            # A frame whose colours cannot be believed must not confirm and must not
            # release. Report the episode as it stood.
            if tr.state == "idle":
                return None
            return PersonRedHit(
                tid=tid,
                state=tr.state,
                part=tr.last_part or tr.seed_part,
                area_frac=tr.last_area,
                held_s=float(now - (tr.active_since or now)),
                growth_frac_per_s=0.0,
                polygons=(),
            )
        detected = bool(regions)
        if detected:
            parts = [r[1] for r in regions]
            if tr.active_since is None:
                tr.active_since = now
                tr.seen = 0
                tr.updates = 0
                tr.duty_hist.clear()
                tr.seed_part = parts[0]
            elif _PLACE[tr.seed_part] not in {_PLACE[p] for p in parts}:
                if tr.state == "confirmed":
                    # Confirmed red that is now somewhere else on the same person is still
                    # red on that person: the alert stays up and names where it is now.
                    tr.seed_part = parts[0]
                else:
                    # A candidate whose red left the place that started its clock: re-seed
                    # rather than inherit a hold this place never earned.
                    tr.active_since = now
                    tr.seen = 0
                    tr.updates = 0
                    tr.duty_hist.clear()
                    tr.seed_part = parts[0]
            tr.below_since = None
            tr.seen += 1
            tr.updates += 1
            self._note_duty(tr, now, True)
            held = now - tr.active_since
            if tr.state != "confirmed":
                duty = sum(d for _, d in tr.duty_hist) / len(tr.duty_hist)
                ready = held >= cfg.min_hold_s and duty >= cfg.duty_min
                tr.state = "confirmed" if ready else "candidate"
        elif tr.active_since is not None:
            tr.updates += 1
            self._note_duty(tr, now, False)
            if tr.below_since is None:
                tr.below_since = now
            if now - tr.below_since >= cfg.release_s:
                self._clear_episode(tr)
                return None
        else:
            return None

        area = float(sum(r[0] for r in regions)) if detected else 0.0
        tr.area_hist.append((now, area))
        while len(tr.area_hist) > 2 and tr.area_hist[0][0] < now - _GROWTH_WINDOW_S:
            tr.area_hist.popleft()
        t0, a0 = tr.area_hist[0]
        growth = float((area - a0) / (now - t0)) if now - t0 > 1e-6 else 0.0

        if detected:
            tr.last_part = tr.seed_part
            tr.last_area = area
        polys = tuple(p for r in regions for p in r[2])
        return PersonRedHit(
            tid=tid,
            state=tr.state,
            part=tr.seed_part,
            area_frac=area if detected else tr.last_area,
            held_s=float(now - tr.active_since),
            growth_frac_per_s=growth,
            polygons=polys,
        )


# --- helpers -----------------------------------------------------------------------------


def _bin_index(pix: np.ndarray) -> np.ndarray:
    return ((pix[..., 0] >> _L_SHIFT) * _AB_BINS + (pix[..., 1] >> _AB_SHIFT)) * _AB_BINS + (
        pix[..., 2] >> _AB_SHIFT
    )


def _mark_bins(row: np.ndarray, idx: np.ndarray) -> None:
    """Mark these histogram bins and their 3 x 3 x 3 neighbourhoods as learned, in place."""
    u = np.unique(idx)
    li, ai, bi = u // (_AB_BINS * _AB_BINS), (u // _AB_BINS) % _AB_BINS, u % _AB_BINS
    for dl in (-1, 0, 1):
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                l2, a2, b2 = li + dl, ai + da, bi + db
                ok = (
                    (l2 >= 0)
                    & (l2 < _L_BINS)
                    & (a2 >= 0)
                    & (a2 < _AB_BINS)
                    & (b2 >= 0)
                    & (b2 < _AB_BINS)
                )
                row[((l2 * _AB_BINS + a2) * _AB_BINS + b2)[ok]] = True


def _box3(h: np.ndarray) -> np.ndarray:
    """3 x 3 x 3 neighbourhood sum with zero padding."""
    out = h
    for ax in range(3):
        p = np.pad(out, [(1, 1) if i == ax else (0, 0) for i in range(3)])
        sl = [slice(None)] * 3
        acc = np.zeros_like(out)
        for k in range(3):
            sl[ax] = slice(k, k + out.shape[ax])
            acc += p[tuple(sl)]
        out = acc
    return out


def _unit(kp: np.ndarray, ok: np.ndarray, box) -> float:
    """The person's head width in the coordinates of `kp`, from whatever the skeleton offers.

    Every estimate shrinks under foreshortening (a profile hides the shoulder width, a lying
    person hides the torso length), so the largest available one is the least foreshortened.
    Clamped to the box so a wild keypoint cannot inflate the mask to the whole frame.
    """
    est = []
    if ok[_LSH] and ok[_RSH]:
        est.append(0.5 * float(np.hypot(*(kp[_LSH] - kp[_RSH]))))
    if ok[_LEAR] and ok[_REAR]:
        est.append(float(np.hypot(*(kp[_LEAR] - kp[_REAR]))))
    if ok[_LSH] and ok[_RSH] and ok[_LHIP] and ok[_RHIP]:
        v = 0.5 * (kp[_LSH] + kp[_RSH]) - 0.5 * (kp[_LHIP] + kp[_RHIP])
        est.append(0.33 * float(np.hypot(v[0], v[1])))
    x1, y1, x2, y2 = (float(v) for v in box)
    longest = max(1.0, x2 - x1, y2 - y1)
    est.append(0.07 * longest)
    return float(min(0.25 * longest, max(est)))


def _same_person(a: _Track, b: _Track) -> bool:
    if a.last_box is None or b.last_box is None or a.last_head is None or b.last_head is None:
        return False
    if _iou(a.last_box, b.last_box) < _INHERIT_IOU:
        return False
    u = max(a.last_u, b.last_u, 1.0)
    return float(np.hypot(*(a.last_head - b.last_head))) <= _INHERIT_HEAD_U * u


def _iou(a, b) -> float:
    ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = ix * iy
    ua = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    ub = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    return inter / max(1e-6, ua + ub - inter)


def _part_of(mask, labels, centroid, kpp, ok, geo) -> str:
    """The body part a region is on: the majority label under it, with the head split into
    "face" and "head" by the nearest COCO head keypoint, and anything above the eye line
    (scalp, forehead) reported as "head"."""
    lab = labels[mask > 0]
    lab = lab[lab != _BG]
    if lab.size == 0:
        return "torso"
    major = int(np.bincount(lab, minlength=_N_LABELS).argmax())
    if major != _HEAD:
        return _LABEL_NAME[major]
    c = np.asarray(centroid, np.float32)
    if geo["eye_line"] is not None and float(np.dot(c - geo["eye_line"], geo["up"])) > (
        _SCALP_ABOVE_EYES * geo["u"]
    ):
        return "head"
    best, best_d = "head", np.inf
    for i, name in (
        (_NOSE, "face"),
        (_LEYE, "face"),
        (_REYE, "face"),
        (_LEAR, "head"),
        (_REAR, "head"),
    ):
        if ok[i]:
            d = float(np.hypot(*(kpp[i] - c)))
            if d < best_d:
                best, best_d = name, d
    return best
