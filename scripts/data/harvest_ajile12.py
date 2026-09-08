#!/usr/bin/env python3
"""Harvest AJILE12 (DANDI:000055) pose + interval labels, streaming, no bulk download.

The dandiset is 845.9 GB of NWB, almost all of it intracranial ecephys that no lane
here uses. This script range-reads only `intervals/epochs` (the activity and blocklist
interval labels) and `processing/behavior/Position/*` (9 upper-body keypoints,
30 Hz, camera-plane px) from each of the 55 session files over HTTPS, and lands
them under the registry root (configs/datasets/ajile12.yaml):

    data/train_prototype/ajile12/sessions/<sub>_<ses>.pose.npz
    data/train_prototype/ajile12/sessions/<sub>_<ses>.epochs.json
    data/train_prototype/ajile12/manifest.jsonl   (sha256 + size per landed file)

Writes are atomic (.tmp then rename) and sessions already landed are skipped, so
the harvest is resumable. Nothing here selects a training diet: diet assembly
(labelled-normal spans only, per the diet rule in the registry entry) is the separate
step that scripts/data/assemble_ajile12_diet.py performs with
configs/events/ajile12_diet_v1.yaml. WU-SAHZU remains the only seizure exam surface.

The harvest reads remote HDF5, so it needs h5py and remfile in the interpreter that runs
it. Neither is a declared dependency of any environment in TRAINING.md, which directs both
to be installed into the environment used for this step.

    uv run python scripts/data/harvest_ajile12.py --config configs/datasets/ajile12.yaml
        [--max-sessions N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

API = "https://api.dandiarchive.org/api/dandisets/000055/versions/draft/assets/"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def list_assets() -> list[dict]:
    out: list[dict] = []
    url = API + "?page_size=60"
    while url:
        page = _get_json(url)
        out.extend(page["results"])
        url = page.get("next")
    return out


def s3_url(asset_id: str) -> str:
    info = _get_json(API + asset_id + "/")
    urls = [u for u in info["contentUrl"] if "s3" in u]
    if not urls:
        raise RuntimeError(f"no s3 contentUrl for asset {asset_id}")
    return urls[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def harvest_one(asset: dict, sessions_dir: Path, manifest: Path) -> bool:
    """Land one session's pose npz + epochs json. Returns True if work was done."""
    import h5py
    import numpy as np
    import remfile

    stem = Path(asset["path"]).stem.replace("_behavior+ecephys", "")
    pose_out = sessions_dir / f"{stem}.pose.npz"
    ep_out = sessions_dir / f"{stem}.epochs.json"
    if pose_out.exists() and ep_out.exists():
        print(f"[skip] {stem} already landed", flush=True)
        return False

    url = s3_url(asset["asset_id"])
    f = h5py.File(remfile.File(url), "r")

    ep = f["intervals/epochs"]
    labels = [v.decode() if isinstance(v, bytes) else str(v) for v in ep["labels"][:]]
    epochs = {
        "session": stem,
        "asset_path": asset["path"],
        "labels": labels,
        "start_time_s": [float(x) for x in ep["start_time"][:]],
        "stop_time_s": [float(x) for x in ep["stop_time"][:]],
    }

    pos = f["processing/behavior/Position"]
    joints = sorted(pos.keys())
    arrays: dict[str, object] = {}
    rates: dict[str, float] = {}
    starts: dict[str, float] = {}
    for j in joints:
        g = pos[j]
        arrays[j] = np.asarray(g["data"][:], dtype=np.float32)
        st = g["starting_time"]
        starts[j] = float(st[()])
        rates[j] = float(st.attrs.get("rate", 0.0))
    n = {j: arrays[j].shape for j in joints}
    print(
        f"[pose] {stem}: joints={joints} shapes={sorted(set(n.values()))} "
        f"rate={sorted(set(rates.values()))}",
        flush=True,
    )

    # np.savez appends ".npz" to any filename that lacks it, so the tmp name must
    # already end in .npz for the atomic rename to find the file it wrote.
    tmp = pose_out.with_name(pose_out.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        joint_names=np.array(joints),
        starting_time_s=np.array([starts[j] for j in joints], dtype=np.float64),
        rate_hz=np.array([rates[j] for j in joints], dtype=np.float64),
        **{f"pose__{j}": arrays[j] for j in joints},
    )
    tmp.rename(pose_out)
    ep_tmp = ep_out.with_suffix(".tmp")
    ep_tmp.write_text(json.dumps(epochs))
    ep_tmp.rename(ep_out)

    with manifest.open("a") as mf:
        for p in (pose_out, ep_out):
            mf.write(
                json.dumps(
                    {
                        "name": f"sessions/{p.name}",
                        "size_bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                        "asset_path": asset["path"],
                    }
                )
                + "\n"
            )
    print(
        f"[done] {stem}: {pose_out.stat().st_size / 1e6:.1f} MB pose, {len(labels)} epochs",
        flush=True,
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--max-sessions", type=int, default=None)
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    if cfg["name"] != "ajile12":
        print("this harvester is ajile12-specific", file=sys.stderr)
        return 2
    root = Path(cfg["layout"]["root"])
    sessions_dir = root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"

    assets = sorted(list_assets(), key=lambda a: a["path"])
    if args.max_sessions:
        assets = assets[: args.max_sessions]
    print(f"{len(assets)} sessions to consider", flush=True)
    done = 0
    failed: list[str] = []
    for a in assets:
        try:
            if harvest_one(a, sessions_dir, manifest):
                done += 1
        except Exception as e:
            failed.append(a["path"])
            print(f"[FAIL] {a['path']}: {type(e).__name__}: {e}", flush=True)
    print(
        f"harvest pass complete: {done} landed, {len(failed)} failed, "
        f"{len(assets) - done - len(failed)} already present",
        flush=True,
    )
    if failed:
        print("failed sessions (re-run to retry):", *failed, sep="\n  ", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
