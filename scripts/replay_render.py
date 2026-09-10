"""Offline replay: the live appliance's whole inference stack over a video FILE.

This is `scripts/live_stream.py` with the camera and the HTTP server removed and a file
put in their place. It produces two things: an annotated video that is pixel-identical in
style to what a viewer sees on the live feed, and a machine-readable list of the events
that CONFIRMED, so a montage can be cut from real timestamps instead of eyeballed ones.

    uv run python scripts/replay_render.py --in take1.mp4 --out annot.mp4 --json ev.json \\
        --config configs/eval/live_final.yaml --start 71 --end 83

Nothing here re-implements the appliance. `draw_tracks`, `draw_banner`, `draw_hud`,
`TrackView`, `box_from_keypoints`, `body_confidence` and the person-gate constants are
loaded out of live_stream.py itself with importlib, and `draw_substance` /
`substance_banner` come from vividly.hud, exactly as the compositor uses them. The
inference order is the appliance's: detect + ByteTrack, pose, the per-track Seizure and
Fall analyzers, the SubstanceAnalyzer, then draw_tracks, draw_substance, draw_banner,
draw_hud.

THE CLOCK. Every analyzer in this system is driven by wall time, and the seizure analyzer
resamples its 4 s window onto a 20 Hz grid, so it believes whatever sampling rate the
timestamps imply. Live, `time.time()` IS the frame's time and that is honest. Offline it
is not: this process runs slower than real time (about 3 s of CPU per second of video on
this machine), so feeding time.time() would tell the analyzer that a 4 s window of motion
took 12 s, dividing every measured frequency by three and putting a 4 Hz convulsion at
1.3 Hz, below the 3 Hz band the detector is looking for. Every analyzer here is therefore
driven by the frame's own presentation time, `frame_index / fps`, and so is the overlay
animation phase passed to draw_substance. `time.time()` appears nowhere on the analysis
path.

Presentation time is `frame_index / fps` and not the container PTS on purpose. These takes
were recorded off an RTSP capture and their PTS arrive in bursts (three frames inside 20 ms,
then a 110 ms gap) which is a muxing artefact, not the camera seeing three frames in 20 ms;
handing those to the resampler would inject frequency content the room never contained.

Two differences from the live rig, both because there is no latency budget offline:
  * The detector runs on EVERY frame (live it runs on its own thread at --det-hz and the
    pose stage carries boxes forward between refreshes). This removes box staleness as an
    explanation for anything the analyzers do here.
  * The substance analyzer runs on every frame too (live: its own thread at --substance-hz).
    Its gates are wall-clock and duty-cycle based, so the rate changes almost nothing except
    how long the 30-update priming takes.
Everything else follows the appliance: the person gate, the pose cap and its priority order,
the per-track analyzer lifecycle, and the shipped thresholds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vividly.analysis import (  # noqa: E402
    FallAnalyzer,
    FallConfig,
    FallState,
    SeizureAnalyzer,
    SeizureConfig,
    SeizureState,
    SubstanceAnalyzer,
    SubstanceConfig,
)
from vividly.hud import draw_substance, substance_banner  # noqa: E402
from vividly.models.factory import build_detector, build_pose  # noqa: E402
from vividly.models.tracker import BoTSORT, ByteTrack  # noqa: E402


def _load_appliance():
    """Load scripts/live_stream.py as a module and take its drawing + gating code.

    Importing the file rather than copying it is the whole point: the rendered video has to
    be the same picture the live viewer gets, and a copied _bracket() drifts the first time
    someone changes the real one.
    """
    path = REPO / "scripts" / "live_stream.py"
    spec = importlib.util.spec_from_file_location("vividly_live_stream_ref", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the appliance module at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


LIVE = _load_appliance()

TrackView = LIVE.TrackView
draw_tracks = LIVE.draw_tracks
draw_banner = LIVE.draw_banner
draw_hud = LIVE.draw_hud
box_from_keypoints = LIVE.box_from_keypoints
body_confidence = LIVE.body_confidence
instant_fall = LIVE.instant_fall  # the same relabel the appliance applies, not a copy of it
merge_alerts = LIVE.merge_alerts
_PRUNE_S = LIVE._PRUNE_S
_CONF_ALPHA = LIVE._CONF_ALPHA
_PERSON_HOLD = LIVE._PERSON_HOLD


# --- event bookkeeping ---------------------------------------------------------------------


class Episode:
    """One CONFIRMED event, from the frame it confirmed to the frame its state cleared.

    `peak` fields are the extremes the analyzer reported across the WHOLE episode, onset
    included, which is what a cut wants: the strongest frame, not the first one.

    `banner_t` is when the alert first reached the SCREEN, which is not always when it
    confirmed. The compositor draws the tracks it posed this frame, so an event latched on a
    track the detector has lost confirms with nothing drawn; `banner_t` is null for exactly
    those, and it is the timestamp a montage should cut on when it is not.
    """

    def __init__(self, kind: str, track_id: int | None, t: float, onset_t: float | None) -> None:
        self.kind = kind
        self.track_id = track_id
        self.confirm_t = t
        self.onset_t = onset_t
        self.last_t = t
        self.held_s = 0.0
        self.peak: dict[str, float] = {}
        self.first_drawn_t: float | None = None
        self.last_drawn_t: float | None = None

    def _bump(self, key: str, value: float) -> None:
        if key not in self.peak or value > self.peak[key]:
            self.peak[key] = float(value)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "track_id": self.track_id,
            "confirm_t": round(self.confirm_t, 3),
            "onset_t": None if self.onset_t is None else round(self.onset_t, 3),
            "last_t": round(self.last_t, 3),
            "held_s": round(self.held_s, 3),
            "banner_t": None if self.first_drawn_t is None else round(self.first_drawn_t, 3),
            "banner_last_t": None if self.last_drawn_t is None else round(self.last_drawn_t, 3),
            "peak": {k: round(v, 4) for k, v in sorted(self.peak.items())},
        }


class EventLog:
    """Turns per-frame analyzer states into a list of confirmed events.

    An event opens the first frame a state reads CONFIRMED and stays open while it does; a
    return to any other state closes it. Re-confirmation after a clear is a new event, which
    is the honest reading: the analyzers latch and release on their own terms.

    Peaks are folded from the ONSET of the episode, not from the confirming frame. A seizure
    confirms after the jerking has been going for min_hold_s, so the frames that make the
    strongest picture are behind the confirmation, not ahead of it, and an event whose peak
    was measured only over its confirmed frames would point a cut at the tail.

    `visible` records whether the confirming frame was a POSED track, which is the only kind
    the compositor draws. A latched fall carried through an absence confirms in the analyzer
    and puts nothing at all on screen, and a montage needs to know the difference.
    """

    def __init__(self) -> None:
        self.events: list[Episode] = []
        self._open: dict[tuple[str, int | None], Episode] = {}
        self._pending: dict[tuple[str, int | None], dict[str, float]] = {}
        # Best near-miss per (kind, track): the longest the analyzer held a building state
        # without reaching the confirm bar. This is what says "it would have confirmed at
        # --seizure-hold 2.4" when it did not confirm at 5.0.
        self.best_candidate: dict[tuple[str, int | None], float] = {}

    def _note(
        self,
        key: tuple[str, int | None],
        t: float,
        state: str,
        confirmed: str,
        building: str,
        held: float,
        peaks: dict[str, float],
        visible: bool,
    ) -> None:
        if state == "idle":
            self._open.pop(key, None)
            self._pending.pop(key, None)
            return
        pend = self._pending.setdefault(key, {})
        for k, v in peaks.items():
            if k not in pend or v > pend[k]:
                pend[k] = float(v)
        if state == building:
            self.best_candidate[key] = max(self.best_candidate.get(key, 0.0), float(held))
        if state != confirmed:
            self._open.pop(key, None)
            return
        ep = self._open.get(key)
        if ep is None:
            ep = Episode(key[0], key[1], t, t - float(held))
            ep.peak.update(pend)
            self._open[key] = ep
            self.events.append(ep)
        ep.last_t = t
        ep.held_s = float(held)
        for k, v in pend.items():
            ep._bump(k, v)
        if visible:
            if ep.first_drawn_t is None:
                ep.first_drawn_t = t
            ep.last_drawn_t = t

    def seizure(self, tid: int, t: float, st: SeizureState, visible: bool = True) -> None:
        self._note(
            ("seizure", tid), t, st.state, "seizure", "candidate", st.held_s,
            {
                "n_limbs": st.n_limbs,
                "band_ratio": st.band_ratio,
                "dominant_hz": st.dominant_hz,
                "amplitude": st.amplitude,
            },
            visible,
        )

    def fall(self, tid: int, t: float, st: FallState, visible: bool = True) -> None:
        # No `spread` here: it is a stillness measure where LOW is the evidence, so folding it
        # with max would report the noisiest frame as the peak. It is in the per-frame trace.
        peaks = {"down_s": st.down_s, "drop": st.drop}
        self._note(
            ("fall", tid), t, st.state, "fall", "falling", st.down_s, peaks, visible
        )

    def substance(self, t: float, st) -> None:
        state = getattr(st, "state", "idle") if st is not None else "idle"
        if state == "idle" or st is None:
            self._note(("substance", None), t, "idle", "confirmed", "candidate", 0.0, {}, True)
            return
        self._note(
            ("substance", None), t, state, "confirmed", "candidate", st.held_s,
            {
                "area_frac": st.area_frac,
                "n_pools": st.n_pools,
                "growth_frac_per_s": st.growth_frac_per_s,
            },
            True,
        )


# --- the run -------------------------------------------------------------------------------


def _writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """An H.264 writer where the build has one, MPEG-4 otherwise. A silent all-zero file is
    the classic OpenCV failure here, so a writer that will not open is a hard error."""
    for tag in ("avc1", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag), fps, size)
        if w.isOpened():
            return w
        w.release()
    raise RuntimeError(f"no usable video writer for {path}")


def run(a: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(a.inp)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {a.inp}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not (1.0 < fps < 240.0):
        fps = 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = a.display_width / max(1, src_w)
    disp_w = a.display_width
    disp_h = int(round(src_h * scale / 2) * 2)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    f0 = max(0, round(a.start * fps)) if a.start is not None else 0
    f1 = min(n_total, round(a.end * fps)) if a.end is not None else n_total
    if f1 <= f0:
        raise RuntimeError(f"empty range: frames {f0}..{f1}")
    if f0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    cfg = yaml.safe_load((REPO / a.config).read_text())
    print(f"building detector + pose (ONNX/CPU) from {a.config} ...", flush=True)
    det_m = build_detector(dict(cfg["detector"]))
    pose_m = build_pose(dict(cfg["pose"]))
    # frame_rate is the appliance's value verbatim. It sets ByteTrack's lost-track window,
    # so changing it would change which identities survive a gap, which is not a replay
    # question.
    #
    # --tracker exists because "which identities survive a gap" turned out to BE the
    # question for the fall case: the detector goes blind for 1.65 s while a person is
    # prone on the floor, ByteTrack mints a new id on recovery, and the fall latch is stranded on
    # a track the compositor no longer draws. botsort is what this repo's own 2026-08-31
    # review adopted for exactly that failure (configs/pipeline/replay_demo.yaml, tracker
    # block). bytetrack stays the default so a plain run still mirrors the live appliance.
    if a.tracker == "botsort":
        tracker = BoTSORT(
            track_activation_threshold=0.25,
            lost_track_buffer=a.lost_buffer,
            velocity_damping=0.01,
            frame_rate=15,
        )
    else:
        tracker = ByteTrack(frame_rate=15.0)

    fall_cfg = FallConfig(hold_s=float(a.fall_hold))
    substance = SubstanceAnalyzer(SubstanceConfig()) if a.substance else None

    out = _writer(a.out, fps, (disp_w, disp_h)) if a.out else None
    log = EventLog()
    trace = Path(a.trace).open("w") if a.trace else None  # noqa: SIM115 (closed below)

    per_track: dict[int, list] = {}
    last_confirm: dict[int, tuple] = {}  # tid -> (t, sz, fl, box) of the last CONFIRMED frame
    live_box: dict[int, np.ndarray] = {}
    conf: dict[int, float] = {}
    good: dict[int, int] = {}

    shipped = {
        "seizure_hold": (a.seizure_hold, 5.0),
        "seizure_limbs": (a.seizure_limbs, 2),
        "seizure_release": (a.seizure_release, 1.5),
        "seizure_amp": (a.seizure_amp, 0.09),
        "seizure_hf": (a.seizure_hf, 0.35),
        "fall_hold": (a.fall_hold, 4.0),
        "fall_instant": (a.fall_instant, False),
        "tracker": (a.tracker, "bytetrack"),
        "alarm_release": (a.alarm_release, 0.0),
    }
    moved = {k: v for k, (v, d) in shipped.items() if v != d}
    if moved:
        print(
            "DEMONSTRATION SETTINGS, not the shipped detector: "
            + ", ".join(f"{k}={v}" for k, v in moved.items())
            + "\n  Every event below was produced with these, not with the values the"
            " appliance runs. Do not quote a number from this run as detector performance.",
            flush=True,
        )
    print(
        f"replay: {src_w}x{src_h} @ {fps:.3f} -> {disp_w}x{disp_h}, "
        f"frames {f0}..{f1} ({(f1 - f0) / fps:.2f}s of video)",
        flush=True,
    )

    idx = f0
    while idx < f1:
        ok, frame = cap.read()
        if not ok:
            break
        # THE CLOCK: the frame's own presentation time, in the source video's timebase, so a
        # reported confirm_t is a timestamp someone can seek to in the original file.
        now = idx / fps
        if abs(scale - 1.0) > 1e-3:
            frame = cv2.resize(frame, (disp_w, disp_h), interpolation=interp)
        h_img, w_img = frame.shape[:2]

        # --- detect + track, every frame ---------------------------------------------------
        d = det_m.detect_one(frame).filter(a.score)
        tracked = tracker.update(d, idx)
        # A fresh detection is authoritative on every frame here, so no box is ever carried
        # forward; box_from_keypoints still runs because the appliance's next-frame crop is
        # part of the pose path it feeds.
        live_box = {
            int(tracked.track_ids[i]): tracked.xyxy[i].astype(np.float32)
            for i in range(len(tracked))
        }
        all_tids = {int(t) for t in tracked.track_ids} if len(tracked) else set()

        # --- pose, capped and prioritized exactly as the appliance does ---------------------
        prio = []
        for i in range(len(tracked)):
            ent = per_track.get(int(tracked.track_ids[i]))
            active = 1 if (ent is not None and ent[3]) else 0
            prio.append((active, float(tracked.scores[i]), i))
        order = [i for *_, i in sorted(prio, reverse=True)]
        idxs = order[: a.max_pose] if a.max_pose else order

        views: list = []
        seen: set[int] = set()
        rows: list[dict] = []
        if idxs:
            boxes = np.stack(
                [
                    live_box.get(int(tracked.track_ids[i]), tracked.xyxy[i].astype(np.float32))
                    for i in idxs
                ]
            )
            poses = pose_m.estimate(frame, boxes)
            for k, i in enumerate(idxs):
                tid = int(tracked.track_ids[i])
                seen.add(tid)
                ent = per_track.get(tid)
                if ent is None:
                    ent = [
                        SeizureAnalyzer(_seizure_cfg(a))
                        if a.seizure
                        else None,
                        FallAnalyzer(fall_cfg) if a.fall else None,
                        now,
                        False,
                        None,  # last SeizureState
                        None,  # last FallState
                        None,  # last box
                    ]
                    per_track[tid] = ent
                kp, sc = poses.keypoints[k], poses.scores[k]
                box = boxes[k].astype(np.float32)
                live_box[tid] = box_from_keypoints(kp, sc, box, w_img, h_img)

                c = body_confidence(sc)
                conf[tid] = (
                    c if tid not in conf else (1 - _CONF_ALPHA) * conf[tid] + _CONF_ALPHA * c
                )
                longest = max(float(box[2] - box[0]), float(box[3] - box[1]))
                big_enough = longest >= a.min_person * h_img
                if conf[tid] >= a.body_conf and big_enough:
                    good[tid] = good.get(tid, 0) + 1
                else:
                    good[tid] = 0
                if good[tid] < _PERSON_HOLD:
                    if trace is not None:
                        rows.append(
                            {
                                "tid": tid,
                                "gated": True,
                                "body_conf": round(conf[tid], 3),
                                "big_enough": bool(big_enough),
                            }
                        )
                    continue
                sz = ent[0].update(kp, sc, now) if ent[0] else SeizureState()
                fl = instant_fall(ent[1].update(kp, sc, box, now), a.fall_instant) if ent[1] else FallState()
                ent[2] = now
                ent[3] = sz.state != "idle" or fl.state != "idle"
                ent[4], ent[5], ent[6] = sz, fl, box
                views.append(TrackView(tid, box, kp, sc, sz, fl))
                log.seizure(tid, now, sz, visible=True)
                log.fall(tid, now, fl, visible=True)
                if trace is not None:
                    rows.append(
                        {
                            "tid": tid,
                            "gated": False,
                            "body_conf": round(conf[tid], 3),
                            "box": [round(float(v), 1) for v in box],
                            "seizure": {
                                "state": sz.state,
                                "n_limbs": sz.n_limbs,
                                "band_ratio": round(sz.band_ratio, 4),
                                "dominant_hz": round(sz.dominant_hz, 3),
                                "amplitude": round(sz.amplitude, 4),
                                "held_s": round(sz.held_s, 3),
                                "limb_hz": {
                                    str(k2): round(float(v), 2)
                                    for k2, v in (sz.limb_hz or {}).items()
                                },
                            },
                            "fall": {
                                "state": fl.state,
                                "torso_deg": None
                                if not np.isfinite(fl.torso_deg)
                                else round(float(fl.torso_deg), 2),
                                "down_s": round(fl.down_s, 3),
                                "drop": round(fl.drop, 4),
                                "held_s": round(fl.held_s, 3),
                                "spread": round(float(fl.meta.get("spread", 0.0)), 4)
                                if fl.meta
                                else None,
                            },
                        }
                    )

        # --- tracks present but not posed, and tracks the tracker dropped -------------------
        # `alerts` carries CONFIRMED events whose track was not posed on this frame. They are
        # handed to draw_banner but never to draw_tracks: there are no fresh keypoints to
        # draw, and inventing a skeleton would be a lie. Without this the top alert blinks
        # off every frame the detector misses, which for a person lying motionless on the
        # floor is most of them, so the one alarm that matters most is the one that flickers.
        alerts: list = []
        for tid in list(per_track):
            if tid in seen:
                continue
            ent = per_track[tid]
            if tid in all_tids:
                ent[2] = now
                if _confirmed(ent[4], ent[5]) and ent[6] is not None:
                    alerts.append(TrackView(tid, ent[6], None, None, ent[4], ent[5]))
                continue
            sz_a = ent[0].on_absent(now) if ent[0] else None
            fl_a = instant_fall(ent[1].on_absent(now), a.fall_instant) if ent[1] else None
            # visible=False: an absent track is not in `views`, so nothing it says is drawn.
            if sz_a is not None:
                log.seizure(tid, now, sz_a, visible=False)
            if fl_a is not None:
                log.fall(tid, now, fl_a, visible=False)
            pruned = now - ent[2] > _PRUNE_S
            if trace is not None:
                # An absent track is where a latched fall lives, and it is also the one state
                # the compositor never draws (it renders `views`, which only holds posed
                # tracks). Tracing it is the difference between "the analyzer confirmed" and
                # "a viewer saw an alert".
                rows.append(
                    {
                        "tid": tid,
                        "absent": True,
                        "pruned": bool(pruned),
                        "gone_s": round(now - ent[2], 3),
                        "seizure": None
                        if sz_a is None
                        else {"state": sz_a.state, "held_s": round(sz_a.held_s, 3)},
                        "fall": None
                        if fl_a is None
                        else {
                            "state": fl_a.state,
                            "down_s": round(fl_a.down_s, 3),
                            "held_s": round(fl_a.held_s, 3),
                        },
                    }
                )
            if not pruned and _confirmed(sz_a, fl_a) and ent[6] is not None:
                # Both slots must be real states: draw_banner's _severity reads
                # .state on each, and an analyzer turned off by a flag returns None.
                alerts.append(
                    TrackView(
                        tid, ent[6], None, None, sz_a or SeizureState(), fl_a or FallState()
                    )
                )
            if pruned:
                del per_track[tid]

        # An alarm should de-escalate on a timer, not on a single frame. Without this a
        # 2-frame dropout in the analyzer (the person settles, `move_max` is exceeded for one
        # sample) blinks the whole red treatment off and on again, which on a screen a nurse
        # is watching reads as a fault in the system rather than a change in the patient.
        # --alarm-release 0 is the shipped behaviour.
        if a.alarm_release > 0.0:
            for v in list(views) + list(alerts):
                if _confirmed(v.seizure, v.fall):
                    last_confirm[v.tid] = (now, v.seizure, v.fall, v.box)
            live_conf = {v.tid for v in list(views) + list(alerts) if _confirmed(v.seizure, v.fall)}
            for tid, (t_c, sz_c, fl_c, box_c) in list(last_confirm.items()):
                if now - t_c > a.alarm_release:
                    del last_confirm[tid]
                elif tid not in live_conf:
                    alerts.append(TrackView(tid, box_c, None, None, sz_c, fl_c))

        # --- substance ---------------------------------------------------------------------
        sub = None
        if substance is not None:
            boxes_ex = tuple(v.box for v in views) if a.substance_exclude_people else ()
            sub = substance.update(frame, now, exclude_boxes=boxes_ex)
            log.substance(now, sub)

        # --- draw, in the compositor's order ------------------------------------------------
        if out is not None:
            canvas = frame  # nothing else reads the frame after this point
            if a.hud != "clean":
                shown = merge_alerts(views, alerts)
                draw_tracks(canvas, shown, a.kp_thr)
                # `now` and not time.time(): the pool overlay's pulse must animate on the
                # video's clock or it strobes at whatever rate this process happens to run.
                draw_substance(canvas, sub, now)
                draw_banner(canvas, shown, substance_banner(sub))
                draw_hud(canvas, len(shown), fps, fps, fps, 0.0, a.hud)
            out.write(canvas)

        if trace is not None:
            trace.write(
                json.dumps(
                    {
                        "frame": idx,
                        "t": round(now, 4),
                        "n_tracked": len(tracked),
                        "tracks": rows,
                        "substance": None
                        if sub is None
                        else {
                            "state": sub.state,
                            "area_frac": round(sub.area_frac, 6),
                            "n_pools": sub.n_pools,
                            "held_s": round(sub.held_s, 3),
                            "growth": round(sub.growth_frac_per_s, 6),
                            "priming": bool(sub.priming),
                        },
                    }
                )
                + "\n"
            )

        idx += 1
        if (idx - f0) % 100 == 0:
            print(f"  {idx - f0}/{f1 - f0} frames  t={now:.2f}s", flush=True)

    cap.release()
    if out is not None:
        out.release()
    if trace is not None:
        trace.close()

    events = [e.as_dict() for e in log.events]
    if a.json:
        Path(a.json).write_text(json.dumps(events, indent=1) + "\n")

    # --- readable summary -------------------------------------------------------------------
    print("")
    print(f"=== replay {Path(a.inp).name}  {f0 / fps:.2f}s .. {f1 / fps:.2f}s ===")
    if not events:
        print("NO CONFIRMED EVENTS.")
    for e in events:
        onset = "n/a" if e["onset_t"] is None else f"{e['onset_t']:.2f}s"
        who = "scene" if e["track_id"] is None else f"person #{e['track_id']}"
        peak = "  ".join(f"{k}={v}" for k, v in e["peak"].items())
        screen = (
            "NEVER DRAWN (track absent, compositor draws posed tracks only)"
            if e["banner_t"] is None
            else f"on screen {e['banner_t']:.2f}s..{e['banner_last_t']:.2f}s"
        )
        print(
            f"{e['kind'].upper():10s} {who:12s} confirm {e['confirm_t']:.2f}s  "
            f"onset {onset}  held {e['held_s']:.2f}s  last {e['last_t']:.2f}s"
        )
        print(f"{'':10s} {screen}")
        if peak:
            print(f"{'':10s} peak: {peak}")
    if log.best_candidate:
        print("")
        print("longest BUILDING state that never confirmed (the bar it would have needed):")
        for (kind, tid), held in sorted(log.best_candidate.items(), key=lambda kv: -kv[1]):
            who = "scene" if tid is None else f"person #{tid}"
            print(f"  {kind:10s} {who:12s} {held:.2f}s")
    if a.json:
        print(f"\nevents -> {a.json}")
    if a.trace:
        print(f"trace  -> {a.trace}")
    if a.out:
        print(f"video  -> {a.out}")
    return 0


def _seizure_cfg(a) -> SeizureConfig:
    """The shipped SeizureConfig with only the knobs the caller actually moved.

    Defaults here are the shipped values, so a plain run is the appliance. Every override is
    a DEMONSTRATION setting and is printed in the run header so no number from a tuned run
    can be mistaken for a number from the shipped detector.
    """
    return SeizureConfig(
        min_hold_s=a.seizure_hold,
        min_limbs=a.seizure_limbs,
        release_s=a.seizure_release,
        amp_min=a.seizure_amp,
        hf_ratio_thr=a.seizure_hf,
    )


def _confirmed(sz, fl) -> bool:
    """Is this track in a state a viewer must keep seeing? Only the two CONFIRMED states
    count. A `candidate` or `falling` track is an event still building, and the appliance's
    rule is that the screen stays quiet until something is certain."""
    return (sz is not None and sz.state == "seizure") or (fl is not None and fl.state == "fall")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="inp", required=True, help="input video file")
    ap.add_argument("--out", default="", help="annotated output video (omit to skip rendering)")
    ap.add_argument("--json", default="", help="write the confirmed-event list here")
    ap.add_argument(
        "--alarm-release", type=float, default=0.0,
        help="seconds a CONFIRMED alarm stays on screen after the analyzer clears (0 = shipped)",
    )
    ap.add_argument("--seizure-limbs", type=int, default=2, help="limbs jerking at once (shipped 2)")
    ap.add_argument(
        "--seizure-release", type=float, default=1.5, help="seconds below threshold before the candidate clears (shipped 1.5)"
    )
    ap.add_argument("--seizure-amp", type=float, default=0.09, help="min limb motion (shipped 0.09)")
    ap.add_argument("--seizure-hf", type=float, default=0.35, help="min clonic band fraction (shipped 0.35)")
    ap.add_argument(
        "--tracker",
        choices=("bytetrack", "botsort"),
        default="bytetrack",
        help="bytetrack mirrors the live appliance; botsort is the repo's shipped tracker",
    )
    ap.add_argument(
        "--lost-buffer",
        type=int,
        default=1800,
        help="botsort only: lost-track memory, counted in 30 fps frames (1800 = 60 s)",
    )
    ap.add_argument("--trace", default="", help="optional per-frame analyzer JSONL, for diagnosis")
    ap.add_argument("--config", default="configs/eval/live_final.yaml")
    ap.add_argument("--score", type=float, default=0.5, help="detector score cut (appliance: 0.5)")
    ap.add_argument("--body-conf", dest="body_conf", type=float, default=0.30)
    ap.add_argument("--min-person", dest="min_person", type=float, default=0.06)
    ap.add_argument("--seizure-hold", dest="seizure_hold", type=float, default=5.0)
    ap.add_argument("--fall-hold", dest="fall_hold", type=float, default=4.0)
    ap.add_argument(
        "--fall-instant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="demo: alarm at the moment of collapse instead of after the stillness hold",
    )
    ap.add_argument("--substance", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--substance-exclude-people",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="the appliance's default is off; a pool lying against a person is where one would be",
    )
    ap.add_argument("--seizure", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fall", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hud", choices=("demo", "debug", "clean"), default="demo")
    ap.add_argument("--kp-thr", dest="kp_thr", type=float, default=0.3)
    ap.add_argument("--max-pose", dest="max_pose", type=int, default=8)
    ap.add_argument("--display-width", dest="display_width", type=int, default=1280)
    ap.add_argument("--start", type=float, default=None, help="seconds into the source file")
    ap.add_argument("--end", type=float, default=None, help="seconds into the source file")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
