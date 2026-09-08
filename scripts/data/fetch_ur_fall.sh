#!/usr/bin/env bash
# Fetch the UR Fall Detection Dataset (Kwolek & Kepski 2014) RGB videos + per-frame
# sync labels and per-frame event ground truth. License: CC BY-NC-SA 4.0, non-commercial
# academic use; whether that grant covers this use is the open counsel question
# CQ-22-acted-fall-dataset-licenses recorded in configs/datasets/ur_fall.yaml, and
# commercial licensing contact is mkepski@ur.edu.pl. POLICY.md, rule 4 carve-out (a)
# admits an acted-event set as an engineering-only test set; POLICY.md, rule 6 is the
# never-trains obligation. Destination is data/eval_only/: UR Fall is a held-out test set
# for the fall verifier, and a dataset that trains loses its test role permanently, so this
# one never trains.
#
#   bash scripts/data/fetch_ur_fall.sh [dest_root]
#
# A file that does not land is reported as MISS and the script exits 1, because a partial
# tree cannot be converted. The names, sizes and sha256 values of the complete tree are in
# the files[] block of configs/datasets/ur_fall.yaml.
set -euo pipefail
DEST="${1:-data/eval_only/ur_fall}"
BASE="http://fenix.ur.edu.pl/~mkepski/ds/data"
MISSES=0
mkdir -p "$DEST/videos" "$DEST/labels"
fetch() { # url -> path; resume, skip if complete
  local url="$1" out="$2"
  if [ -s "$out" ]; then return 0; fi
  if ! curl -fsSL --retry 3 -C - -o "$out" "$url"; then
    echo "MISS $url"
    MISSES=$((MISSES + 1))
  fi
}
for i in $(seq -w 1 30); do
  fetch "$BASE/fall-$i-cam0.mp4" "$DEST/videos/fall-$i-cam0.mp4"
  fetch "$BASE/fall-$i-cam1.mp4" "$DEST/videos/fall-$i-cam1.mp4"
  fetch "$BASE/fall-$i-data.csv" "$DEST/labels/fall-$i-data.csv"
done
for i in $(seq -w 1 40); do
  fetch "$BASE/adl-$i-cam0.mp4" "$DEST/videos/adl-$i-cam0.mp4"
  fetch "$BASE/adl-$i-data.csv" "$DEST/labels/adl-$i-data.csv"
done
# The two per-frame event ground-truth files. src/vividly/data/convert/ur_fall.py requires
# both by name, so the tree is not convertible without them.
fetch "$BASE/urfall-cam0-falls.csv" "$DEST/labels/urfall-cam0-falls.csv"
fetch "$BASE/urfall-cam0-adls.csv" "$DEST/labels/urfall-cam0-adls.csv"
for required in urfall-cam0-falls.csv urfall-cam0-adls.csv; do
  if [ ! -s "$DEST/labels/$required" ]; then
    echo "MISSING $DEST/labels/$required: place it from the dataset page before converting"
    MISSES=$((MISSES + 1))
  fi
done
echo "UR_FALL_FETCH: $(ls "$DEST/videos" | wc -l) videos, $(ls "$DEST/labels" | wc -l) label files"
if [ "$MISSES" -gt 0 ]; then
  echo "UR_FALL_FETCH_INCOMPLETE: $MISSES file(s) did not land; compare against configs/datasets/ur_fall.yaml"
  exit 1
fi
echo "UR_FALL_FETCH_DONE"
