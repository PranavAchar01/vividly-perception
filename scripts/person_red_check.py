"""Run the person-red analyzer over a video file, with the appliance's own people, and report.

    uv run python scripts/person_red_check.py --in take.mp4 [--render out.mp4] [--json out.json]

The people come from the same stack replay_render.py runs: the detector of record on every
frame, BoT-SORT built exactly as replay_render builds it for --tracker botsort, RTMPose on
every tracked box, and the appliance's person gate (body confidence, minimum size, a hold of
_PERSON_HOLD frames), loaded out of scripts/live_stream.py so the gate cannot drift. Only
gated people reach the analyzer, which is what the compositor will hand it.

What it prints, per video:
  * every CONFIRMED episode: person, body part, confirm time, peak area and how long it held;
  * per person, the longest CANDIDATE hold that never confirmed and when it happened, which
    is the honest distance to a false positive on a clean take;
  * the cost of update() at p50/p95/max, in total and per person, measured around update()
    alone (never decode, detection, pose or drawing). The budget is 3 ms per person.

Detection and pose dominate the run time (about 1 s per frame on this machine with two
people), so --people-cache stores the gated people per frame the first time and replays them
after, which makes iterating on the analyzer a matter of seconds.

--dev-positive paints a crude synthetic positive into the frames before the analyzer sees
them: a dark-red blob with a feathered edge, tracked to the head of one person (the largest
posed person at the start time unless --dev-tid names one), growing over 2 s from --dev-t0.
It exists so the confirm path can be exercised before the compositor's real positive does;
nothing about it is a measurement of the detector on real footage.

Clock: frame_index / fps, exactly as replay_render.py, for the same reasons given there.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vividly.analysis.person_red import PersonRedAnalyzer, PersonRedConfig  # noqa: E402
from vividly.hud.person_red import draw_person_red, person_red_banner  # noqa: E402


def _load_appliance():
    """scripts/live_stream.py as a module, for the person gate and draw_banner (see
    replay_render.py for why the file is loaded rather than copied)."""
    path = REPO / "scripts" / "live_stream.py"
    spec = importlib.util.spec_from_file_location("vividly_live_stream_ref", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the appliance module at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _people_stream(a, fps: float, n_frames: int, size: tuple[int, int]):
    """Per frame: [(tid, box, kp, sc, gated)] from detector + BoT-SORT + pose + person gate.

    Loaded from --people-cache when it exists and matches; computed (and saved) otherwise.
    """
    cache = Path(a.people_cache) if a.people_cache else None
    st = Path(a.inp).stat()
    # The cache belongs to one file: a composited copy of a take has the same size and frame
    # count as the take, and its people must still be detected on its own pixels.
    ident = [str(Path(a.inp).resolve()), st.st_size, st.st_mtime_ns, list(size), a.config]
    if cache is not None and cache.exists():
        d = pickle.loads(cache.read_bytes())
        if d.get("ident") == ident and len(d["frames"]) >= n_frames:
            print(f"people   from cache {cache}", flush=True)
            return d["frames"]
        print(f"people   cache {cache} is for another file, recomputing", flush=True)

    from vividly.models.factory import build_detector, build_pose
    from vividly.models.tracker import BoTSORT

    live = _load_appliance()
    cfg = yaml.safe_load((REPO / a.config).read_text())
    print(f"people   building detector + pose (ONNX/CPU) from {a.config} ...", flush=True)
    det = build_detector(dict(cfg["detector"]))
    pose = build_pose(dict(cfg["pose"]))
    # The construction replay_render.py uses for --tracker botsort, verbatim.
    tracker = BoTSORT(
        track_activation_threshold=0.25,
        lost_track_buffer=a.lost_buffer,
        velocity_damping=0.01,
        frame_rate=15,
    )
    cap = cv2.VideoCapture(a.inp)
    w, h = size
    conf: dict[int, float] = {}
    good: dict[int, int] = {}
    frames = []
    t0 = time.time()
    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        d = det.detect_one(frame).filter(a.score)
        tr = tracker.update(d, idx)
        order = sorted(range(len(tr)), key=lambda i: -float(tr.scores[i]))[: a.max_pose]
        people = []
        if order:
            boxes = np.stack([tr.xyxy[i].astype(np.float32) for i in order])
            ps = pose.estimate(frame, boxes)
            for k, i in enumerate(order):
                tid = int(tr.track_ids[i])
                kp = ps.keypoints[k].astype(np.float32)
                sc = ps.scores[k].astype(np.float32)
                box = boxes[k]
                c = live.body_confidence(sc)
                conf[tid] = (
                    c
                    if tid not in conf
                    else (1 - live._CONF_ALPHA) * conf[tid] + live._CONF_ALPHA * c
                )
                big = max(float(box[2] - box[0]), float(box[3] - box[1])) >= a.min_person * h
                good[tid] = good.get(tid, 0) + 1 if (conf[tid] >= a.body_conf and big) else 0
                people.append((tid, box, kp, sc, good[tid] >= live._PERSON_HOLD))
        frames.append(people)
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{n_frames} frames  {time.time() - t0:.0f}s", flush=True)
    cap.release()
    if cache is not None:
        cache.write_bytes(
            pickle.dumps({"fps": fps, "size": size, "ident": ident, "frames": frames})
        )
        print(f"people   cached -> {cache}", flush=True)
    return frames


# --- the crude dev positive ---------------------------------------------------------------


def _head(kp, sc):
    ok = sc[:5] >= 0.3
    return kp[:5][ok].mean(axis=0) if ok.any() else None


_DEV_BGR = np.array([22.0, 18.0, 105.0], np.float32)  # sRGB (105, 18, 22): L* 22, a* 40
# How far a head may move between frames and still be the same person's head. On the 205301
# take a second person walks past the seated one at 69 s; with a looser bound the blob jumps
# onto the passer-by the one frame the seated person is not posed.
_DEV_MAX_JUMP_PX = 45.0


def _paint_dev_positive(frame, kp, sc, t, t0, grow_s, rng) -> None:
    """A dark-red blob with a feathered edge on the upper head of this pose, in place."""
    ok = sc >= 0.3
    head = [i for i in range(5) if ok[i]]
    if not head:
        return
    hc = kp[head].mean(axis=0)
    up = np.array([0.0, -1.0], np.float32)
    if ok[5] and ok[6]:
        v = hc - 0.5 * (kp[5] + kp[6])
        n = float(np.hypot(*v))
        if n > 1e-3:
            up = v / n
    shoulder = float(np.hypot(*(kp[5] - kp[6]))) if ok[5] and ok[6] else 0.0
    ears = float(np.hypot(*(kp[3] - kp[4]))) if ok[3] and ok[4] else 0.0
    u = max(0.5 * shoulder, ears, 12.0)  # head width, display px
    r_final = 0.45 * u
    r = r_final * (0.4 + 0.6 * min(1.0, max(0.0, (t - t0) / grow_s)))
    c = hc + up * (0.45 * u)
    x0, y0 = int(c[0] - r - 2), int(c[1] - r - 2)
    x1, y1 = int(c[0] + r + 3), int(c[1] + r + 3)
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    # A slightly irregular edge, so the blob is not a perfect disc.
    ang = np.arctan2(yy - c[1], xx - c[0])
    rr = r * (1.0 + 0.12 * np.sin(3.0 * ang + 1.3) + 0.06 * np.sin(5.0 * ang))
    d = np.hypot(xx - c[0], yy - c[1])
    feather = max(1.5, 0.35 * r)
    alpha = np.clip((rr - d) / feather, 0.0, 1.0) * 0.92
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    col = _DEV_BGR[None, None, :] + rng.normal(0.0, 4.0, roi.shape).astype(np.float32)
    out = roi * (1.0 - alpha[..., None]) + col * alpha[..., None]
    frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)


# --- the run -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--render", default="", help="annotated output video")
    ap.add_argument("--json", default="", help="write the report here")
    ap.add_argument("--people-cache", default="", help="pickle of per-frame gated people")
    ap.add_argument("--config", default="configs/eval/live_final.yaml")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--body-conf", type=float, default=0.30)
    ap.add_argument("--min-person", type=float, default=0.06)
    ap.add_argument("--max-pose", type=int, default=8)
    ap.add_argument("--lost-buffer", type=int, default=1800)
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--start", type=float, default=None, help="seconds; analysis starts here")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override one PersonRedConfig field for this run (printed in the header)",
    )
    ap.add_argument("--dev-positive", action="store_true", help="paint the crude dev positive")
    ap.add_argument("--dev-t0", type=float, default=34.0)
    ap.add_argument("--dev-grow", type=float, default=2.0)
    ap.add_argument("--dev-tid", type=int, default=None)
    ap.add_argument(
        "--hold-until",
        type=float,
        default=None,
        help="report whether the first "
        "confirmed episode stays confirmed from its confirm time through this time",
    )
    a = ap.parse_args(argv)

    cap = cv2.VideoCapture(a.inp)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.inp}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not (1.0 < fps < 240.0):
        fps = 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_w = a.display_width
    disp_h = int(round(src_h * disp_w / src_w / 2) * 2)
    cap.release()

    # People are computed from frame 0 regardless of --start (identities and the person gate
    # depend on history) and up to --end; a cache holding more frames serves a shorter run.
    f_end = min(n_total, round(a.end * fps)) if a.end is not None else n_total
    people_frames = _people_stream(a, fps, f_end, (disp_w, disp_h))
    f0 = max(0, round(a.start * fps)) if a.start is not None else 0
    f1 = min(len(people_frames), f_end)

    overrides = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in PersonRedConfig.__dataclass_fields__:
            raise SystemExit(f"--set: PersonRedConfig has no field {k!r}")
        overrides[k] = type(getattr(PersonRedConfig(), k))(v)
    if overrides:
        print(f"config   NOT the shipped PersonRedConfig: {overrides}", flush=True)
    analyzer = PersonRedAnalyzer(PersonRedConfig(**overrides))
    live = _load_appliance() if a.render else None
    cap = cv2.VideoCapture(a.inp)
    if f0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    writer = None
    if a.render:
        writer = cv2.VideoWriter(a.render, cv2.VideoWriter_fourcc(*"mp4v"), fps, (disp_w, disp_h))

    rng = np.random.default_rng(7)
    dev_tid = a.dev_tid
    dev_head = None  # where the dev positive's head was last frame, display px
    costs: list[float] = []
    per_person: list[float] = []
    episodes: list[dict] = []
    open_ep: dict[float, dict] = {}  # onset -> the confirmed episode currently open
    peak_cand: dict[int, dict] = {}  # tid -> {"held_s", "t", "part"} longest non-confirming
    any_confirmed: set[int] = set()  # frame indices on which some hit was confirmed
    n_people_seen: set[int] = set()

    for idx in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        now = idx / fps
        frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        people = [(t, b, k, s) for (t, b, k, s, g) in people_frames[idx] if g]
        n_people_seen.update(t for t, *_ in people)

        if a.dev_positive and now >= a.dev_t0:
            # The blob follows the PERSON, not a track id: the tracker may hand this person to
            # a new id mid-event, and a real substance does not come off when that happens.
            # Each frame it goes on the posed person whose head is nearest where it was.
            target = None
            if dev_head is None:
                pick = [p for p in people if dev_tid is None or p[0] == dev_tid]
                if pick:
                    target = max(
                        pick, key=lambda p: float((p[1][3] - p[1][1]) * (p[1][2] - p[1][0]))
                    )
                    dev_tid = target[0]
                    print(
                        f"dev      positive starts on person #{dev_tid} at {now:.2f}s", flush=True
                    )
            else:
                near = [
                    (float(np.hypot(*(_head(p[2], p[3]) - dev_head))), p)
                    for p in people
                    if _head(p[2], p[3]) is not None
                ]
                near = [q for q in near if q[0] <= _DEV_MAX_JUMP_PX]
                if near:
                    target = min(near, key=lambda q: q[0])[1]
            if target is not None and _head(target[2], target[3]) is not None:
                dev_head = _head(target[2], target[3])
                _paint_dev_positive(frame, target[2], target[3], now, a.dev_t0, a.dev_grow, rng)

        t0 = time.perf_counter()
        st = analyzer.update(frame, now, people)
        dt = (time.perf_counter() - t0) * 1e3
        costs.append(dt)
        if people:
            per_person.append(dt / len(people))

        # An episode is keyed by its onset, not by a track id: the analyzer hands a live
        # episode across ids of the same person, and that is one event, not several.
        live_eps = set()
        for hit in st.hits:
            if hit.state == "confirmed":
                any_confirmed.add(idx)
                key = round(now - hit.held_s, 2)
                live_eps.add(key)
                ep = open_ep.get(key)
                if ep is None:
                    ep = {
                        "tids": [],
                        "part": hit.part,
                        "confirm_t": round(now, 3),
                        "onset_t": round(now - hit.held_s, 3),
                        "peak_area_frac": 0.0,
                    }
                    open_ep[key] = ep
                    episodes.append(ep)
                if hit.tid not in ep["tids"]:
                    ep["tids"].append(hit.tid)
                ep["last_t"] = round(now, 3)
                ep["held_s"] = round(hit.held_s, 3)
                ep["peak_area_frac"] = round(max(ep["peak_area_frac"], hit.area_frac), 6)
            else:
                best = peak_cand.get(hit.tid)
                if best is None or hit.held_s > best["held_s"]:
                    peak_cand[hit.tid] = {
                        "held_s": round(hit.held_s, 3),
                        "t": round(now, 3),
                        "part": hit.part,
                        "area_frac": round(hit.area_frac, 6),
                    }
        for key in list(open_ep):
            if key not in live_eps:
                del open_ep[key]

        if writer is not None:
            canvas = frame
            for t, b, _k, _s in people:
                x1, y1, x2, y2 = (int(v) for v in b)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (130, 235, 150), 1)
                cv2.putText(
                    canvas,
                    f"#{t}",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (130, 235, 150),
                    1,
                    cv2.LINE_AA,
                )
            draw_person_red(canvas, st, now)
            live.draw_banner(canvas, [], person_red_banner(st))
            cand = [h for h in st.hits if h.state == "candidate"]
            dbg = f"{now:6.2f}s  people {len(people)}  priming {st.priming}  " + "  ".join(
                f"#{h.tid}:{h.state[:4]}:{h.part}:{h.held_s:.1f}s" for h in st.hits
            )
            cv2.putText(canvas, dbg, (10, disp_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(
                canvas,
                dbg,
                (10, disp_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 190, 255) if cand else (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    cap.release()
    if writer is not None:
        writer.release()

    c = np.asarray(costs) if costs else np.zeros(1)
    pp = np.asarray(per_person) if per_person else np.zeros(1)
    timing = {
        "n_updates": len(costs),
        "update_p50_ms": round(float(np.percentile(c, 50)), 3),
        "update_p95_ms": round(float(np.percentile(c, 95)), 3),
        "update_max_ms": round(float(c.max()), 3),
        "per_person_p50_ms": round(float(np.percentile(pp, 50)), 3),
        "per_person_p95_ms": round(float(np.percentile(pp, 95)), 3),
        "per_person_max_ms": round(float(pp.max()), 3),
    }
    # Whether SOME person carries a confirmed hit on every frame from the first confirmation
    # through --hold-until. Deliberately not tied to one track id: the tracker hands a person
    # to a new id mid-event (see PersonRedAnalyzer._donor) and the alert must survive that.
    hold = None
    if a.hold_until is not None and episodes:
        t_first = min(e["confirm_t"] for e in episodes)
        want = [i for i in range(f0, f1) if t_first <= i / fps + 1e-6 and i / fps <= a.hold_until]
        missing = [round(i / fps, 3) for i in want if i not in any_confirmed]
        hold = {
            "from": t_first,
            "through": a.hold_until,
            "stays_confirmed": not missing,
            "n_missing": len(missing),
            "missing_frames": missing[:20],
        }

    report = {
        "video": a.inp,
        "range_s": [round(f0 / fps, 3), round(f1 / fps, 3)],
        "fps": fps,
        "dev_positive": {"t0": a.dev_t0, "grow_s": a.dev_grow, "tid": dev_tid}
        if a.dev_positive
        else None,
        "people": sorted(n_people_seen),
        "confirmed": episodes,
        "peak_candidate": {str(k): v for k, v in sorted(peak_cand.items())},
        "hold_check": hold,
        "timing_ms": timing,
    }

    print(f"video    {a.inp}  {f0 / fps:.2f}s .. {f1 / fps:.2f}s  ({f1 - f0} frames @ {fps:.2f})")
    print(f"people   {len(n_people_seen)} tracks reached the analyzer: {sorted(n_people_seen)}")
    if episodes:
        for e in episodes:
            print(
                f"CONFIRMED person #{'/#'.join(str(t) for t in e['tids'])}  part {e['part']}  confirm {e['confirm_t']:.2f}s  "
                f"onset {e['onset_t']:.2f}s  held {e['held_s']:.2f}s  last {e['last_t']:.2f}s  "
                f"peak area {e['peak_area_frac'] * 100:.3f}%"
            )
    else:
        print("CONFIRMED none")
    if peak_cand:
        for tid, v in sorted(peak_cand.items()):
            print(
                f"candidate person #{tid}  longest non-confirming hold {v['held_s']:.2f}s at "
                f"{v['t']:.2f}s  part {v['part']}  area {v['area_frac'] * 100:.4f}%"
            )
    for tid in sorted(n_people_seen - set(peak_cand)):
        print(f"candidate person #{tid}  never a candidate")
    if hold is not None:
        print(
            f"hold     confirmed {hold['from']:.2f}s through {hold['through']:.2f}s: "
            f"{'yes' if hold['stays_confirmed'] else 'NO, ' + str(hold['n_missing']) + ' frames not confirmed'}"
        )
    print(
        f"cost     update p50 {timing['update_p50_ms']:.2f} ms  p95 {timing['update_p95_ms']:.2f} ms  "
        f"max {timing['update_max_ms']:.2f} ms;  per person p50 {timing['per_person_p50_ms']:.2f} ms  "
        f"p95 {timing['per_person_p95_ms']:.2f} ms  max {timing['per_person_max_ms']:.2f} ms"
    )
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1) + "\n")
        print(f"report   {a.json}")
    if a.render:
        print(f"video    {a.render}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
