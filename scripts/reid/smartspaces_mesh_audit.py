#!/usr/bin/env python3
"""SmartSpaces mesh-diversity audit: contact sheets of every identity's crops.

The instrument behind research/reid-smartspaces-mesh-audit.md. The 2026-09-03 run
on the training pod covered 231 identities over Warehouse_000..004, produced 20
sheets, and the corpus was rejected: about 35 to 45 distinct avatar builds are
recycled across the labels.

For every identity it samples up to --per-id crops round-robin across that
identity's cameras (tallest-first within a camera), tiles them one row per
identity, and sorts rows by a mean-HSV torso signature so look-alike identities
land on adjacent rows, which makes avatar reuse across labels readable directly off
the sheets. It also writes stats.json with per-scene identity and crop counts at the
>=48x96 and >=64x128 gates.

Inputs:
  --gt-dir     directory of per-scene ground truth named <Scene>.json, either the
               raw AI City format ({frame: [{"object type", "object id",
               "2d bounding box visible"}, ...]}) or the slimmed upload format
               ({frame: [{"object id", "boxes"}, ...]}; produced by filtering to
               Person entries on 1 Hz frames).
  --frames-dir 1 Hz frame tree <Scene>/<Camera_XXXX>/<frame:06d>.jpg (the person_v2
               extraction; frame index is the source-video frame number).
  --out        output directory for sheet_NN.jpg + stats.json.
  --scenes     scene names to audit (default Warehouse_000..004).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

CROP_W, CROP_H = 96, 192
IDS_PER_SHEET = 12
LABEL_W = 110


def iter_person_boxes(obj):
    """Yield (camera, [x1, y1, x2, y2]) for one GT object in either format."""
    if "boxes" in obj:  # slimmed format, already Person-only
        yield from obj["boxes"].items()
    elif obj.get("object type") == "Person":
        yield from obj.get("2d bounding box visible", {}).items()


def sample(entries, per_id):
    by_cam = defaultdict(list)
    for e in entries:
        by_cam[e[0]].append(e)
    for cam in by_cam:
        by_cam[cam].sort(key=lambda e: e[5] - e[3], reverse=True)
    cams = sorted(by_cam, key=lambda c: by_cam[c][0][5] - by_cam[c][0][3], reverse=True)
    picked, ring = [], 0
    while len(picked) < per_id and any(by_cam.values()):
        cam = cams[ring % len(cams)]
        if by_cam[cam]:
            picked.append(by_cam[cam].pop(0))
        ring += 1
        if ring > 10 * per_id:
            break
    return picked


def load_crop(frames_dir, scene, cam, fi, x1, y1, x2, y2):
    path = Path(frames_dir) / scene / cam / f"{fi:06d}.jpg"
    try:
        im = Image.open(path)
    except OSError:
        return None
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(im.width, x2), min(im.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    c = im.crop((x1, y1, x2, y2))
    c.thumbnail((CROP_W, CROP_H))
    canvas = Image.new("RGB", (CROP_W, CROP_H), (24, 24, 24))
    canvas.paste(c, ((CROP_W - c.width) // 2, (CROP_H - c.height) // 2))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--gt-dir",
        required=True,
        help="directory of per-scene ground truth named <Scene>.json, in the raw AI City "
        "format or the slimmed person-only upload format",
    )
    ap.add_argument(
        "--frames-dir",
        required=True,
        help="1 Hz frame tree <Scene>/<Camera_XXXX>/<frame:06d>.jpg, the person_v2 extraction",
    )
    ap.add_argument("--out", required=True, help="output directory for sheet_NN.jpg + stats.json")
    ap.add_argument(
        "--scenes",
        nargs="+",
        default=[f"Warehouse_{i:03d}" for i in range(5)],
        help="scene names to audit (default: Warehouse_000 to Warehouse_004)",
    )
    ap.add_argument(
        "--per-id",
        type=int,
        default=10,
        help="crops sampled per identity, round-robin across that identity's cameras",
    )
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    boxes = defaultdict(list)  # (scene, oid) -> [(cam, fi, x1, y1, x2, y2)]
    stats = {}
    for scene in args.scenes:
        with (Path(args.gt_dir) / f"{scene}.json").open() as f:
            gt = json.load(f)
        ids_48 = set()
        n48 = n64 = 0
        for fk, objs in gt.items():
            fi = int(fk)
            if fi % 30:
                continue  # only the extracted 1 Hz frames exist as jpgs
            for o in objs:
                for cam, (x1, y1, x2, y2) in iter_person_boxes(o):
                    w, h = x2 - x1, y2 - y1
                    if w >= 48 and h >= 96:
                        n48 += 1
                        ids_48.add(o["object id"])
                        boxes[(scene, o["object id"])].append((cam, fi, x1, y1, x2, y2))
                    if w >= 64 and h >= 128:
                        n64 += 1
        stats[scene] = {
            "person_ids_with_crops_ge48x96": len(ids_48),
            "crops_ge48x96": n48,
            "crops_ge64x128": n64,
        }
        del gt
        print(f"{scene}: {len(ids_48)} IDs, {n48} crops>=48x96, {n64} crops>=64x128", flush=True)

    stats["_total"] = {
        "identities": len(boxes),
        "crops_ge48x96": sum(s["crops_ge48x96"] for s in stats.values() if "crops_ge48x96" in s),
        "crops_ge64x128": sum(s["crops_ge64x128"] for s in stats.values() if "crops_ge64x128" in s),
    }
    with (out_dir / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Total identities: {len(boxes)}", flush=True)

    rows = []  # ((label_top, label_bottom), signature, crops)
    for (scene, oid), entries in sorted(boxes.items()):
        crops = []
        for cam, fi, x1, y1, x2, y2 in sample(entries, args.per_id):
            c = load_crop(args.frames_dir, scene, cam, fi, x1, y1, x2, y2)
            if c is not None:
                crops.append(c)
        if not crops:
            continue
        hs, ss, vs = [], [], []
        for c in crops:
            torso = c.crop((CROP_W // 4, CROP_H // 5, 3 * CROP_W // 4, 3 * CROP_H // 5)).convert(
                "HSV"
            )
            px = list(torso.getdata())
            hs.append(sum(p[0] for p in px) / len(px))
            ss.append(sum(p[1] for p in px) / len(px))
            vs.append(sum(p[2] for p in px) / len(px))
        sig = (round(sum(hs) / len(hs) / 16), round(sum(vs) / len(vs) / 16), sum(ss) / len(ss))
        ncams = len({e[0] for e in entries})
        rows.append(((f"{scene[-3:]}:{oid}", f"{len(entries)}cr {ncams}cam"), sig, crops))

    rows.sort(key=lambda r: r[1])  # look-alike identities become adjacent rows
    sheet_w = LABEL_W + args.per_id * CROP_W
    for si in range(0, len(rows), IDS_PER_SHEET):
        chunk = rows[si : si + IDS_PER_SHEET]
        sheet = Image.new("RGB", (sheet_w, len(chunk) * CROP_H), (10, 10, 10))
        d = ImageDraw.Draw(sheet)
        for ri, (label, _sig, crops) in enumerate(chunk):
            y = ri * CROP_H
            d.text((6, y + 6), label[0], fill=(255, 255, 120))
            d.text((6, y + 24), label[1], fill=(200, 200, 200))
            for ci, c in enumerate(crops):
                sheet.paste(c, (LABEL_W + ci * CROP_W, y))
            d.line([(0, y), (sheet_w, y)], fill=(60, 60, 60))
        sheet.save(out_dir / f"sheet_{si // IDS_PER_SHEET:02d}.jpg", quality=88)
    print(
        f"sheets written: {(len(rows) + IDS_PER_SHEET - 1) // IDS_PER_SHEET} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
