#!/usr/bin/env bash
# Negative test for the guard hooks: every guard must reject a real violation.
# A guard that never fails is decoration, and this is what makes the rules testable.
# Run from anywhere:  bash scripts/guards/selftest.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

PY="${VIVIDLY_PY:-uv run --no-sync python}"
fail=0
cases=0
accepts=0
MIN_CASES=41

expect_reject() {  # expect_reject <label> <guard-cmd...>
  local label="$1"; shift
  cases=$((cases + 1))
  if "$@" >/dev/null 2>&1; then
    echo "FAIL: ${label}, guard exited 0 on a violation"; fail=1
  else
    echo "ok:   ${label}, guard rejected the violation"
  fi
}

# The other half of a credible guard: it must also stay quiet on a clean tree. A guard that
# rejects everything is as useless as one that rejects nothing, and a substring deny-list is
# exactly the kind that drifts into false positives.
expect_accept() {  # expect_accept <label> <guard-cmd...>
  local label="$1"; shift
  accepts=$((accepts + 1))
  if "$@" >/dev/null 2>&1; then
    echo "ok:   ${label}, guard accepted the clean tree"
  else
    echo "FAIL: ${label}, guard rejected a clean tree"; fail=1
  fi
}

# Two probes have to mutate a tracked file in place, because the guards read pyproject.toml
# and the Makefile from fixed paths, so the originals are parked here and restored
# unconditionally.
SCRATCH="$(mktemp -d)"

cleanup() {
  rm -f src/vividly/media/_guard_selftest.py \
        src/vividly/media/_guard_selftest_primitives.py \
        src/vividly/eval/_guard_selftest.py \
        configs/events/train__guard_selftest.yaml \
        configs/reid/_guard_selftest.yaml \
        configs/_guard_selftest.yaml \
        scripts/_guard_selftest_fetch.py \
        src/vividly/_guard_selftest_reid.py \
        .github/workflows/_guard_selftest.yml \
        configs/_guard_selftest_dataset.yaml \
        configs/_guard_selftest_fisheye_dome.yaml \
        configs/_guard_selftest_weekly.yaml \
        scripts/_guard_selftest_render.py \
        data/train_ok/_guard_selftest_link \
        data/train_ok/_guard_selftest_proto_link \
        configs/pipeline/_guard_selftest_clip_*.yaml
  if [ -f "${SCRATCH}/pyproject.toml.orig" ]; then
    cp -f "${SCRATCH}/pyproject.toml.orig" pyproject.toml
  fi
  if [ -f "${SCRATCH}/Makefile.orig" ]; then
    cp -f "${SCRATCH}/Makefile.orig" Makefile
  fi
  if [ -f "${SCRATCH}/no_fisheye_allowlist.orig" ]; then
    cp -f "${SCRATCH}/no_fisheye_allowlist.orig" scripts/guards/no_fisheye_allowlist.txt
  fi
  rm -rf "${SCRATCH}"
}
trap cleanup EXIT

# A fresh checkout has no data/ tree, which is gitignored by design. The probes need it to
# exist, and creating it is harmless.
mkdir -p data/train_ok data/eval_only data/train_prototype

# 1. AGPL quarantine
echo 'import ultralytics' > src/vividly/media/_guard_selftest.py
expect_reject "agpl/import ultralytics in a package module" $PY scripts/guards/check_agpl_imports.py
echo 'from boxmot import BotSort' > src/vividly/media/_guard_selftest.py
expect_reject "agpl/from boxmot in a package module" $PY scripts/guards/check_agpl_imports.py
rm -f src/vividly/media/_guard_selftest.py

# 1b. The quarantine is scoped to a file, not to the eval tree: agpl_ref.py is the only
# admitted path, so the same import one file over must still be rejected.
echo 'from boxmot import BotSort' > src/vividly/eval/_guard_selftest.py
expect_reject "agpl/from boxmot in eval, a legal tree and an illegal file" \
  $PY scripts/guards/check_agpl_imports.py src/vividly/eval/_guard_selftest.py
rm -f src/vividly/eval/_guard_selftest.py

# 1c. The clean tree passes. This catches over-approximation drift, such as a package that is
# not under AGPL creeping into the root set and refusing a legitimate import.
expect_accept "agpl/clean tree" $PY scripts/guards/check_agpl_imports.py

# 2. Face-biometric deny-list: a module import
echo 'import insightface' > src/vividly/media/_guard_selftest.py
expect_reject "face/import insightface" $PY scripts/guards/check_face_deps.py
rm -f src/vividly/media/_guard_selftest.py

# 2b. Face-biometric deny-list: a declared dependency
cp pyproject.toml "${SCRATCH}/pyproject.toml.orig"
awk '/^dependencies = \[$/ && !ins { print; print "    \"insightface>=0.7\","; ins = 1; next } { print }' \
  "${SCRATCH}/pyproject.toml.orig" > pyproject.toml
expect_reject "face/insightface in the pyproject dependencies" $PY scripts/guards/check_face_deps.py
cp -f "${SCRATCH}/pyproject.toml.orig" pyproject.toml
rm -f "${SCRATCH}/pyproject.toml.orig"

# 2c. Face-biometric deny-list: transitive arrival, visible only in the lockfile
printf 'name = "insightface"\nversion = "0.7.3"\n' > "${SCRATCH}/uv.lock"
expect_reject "face/insightface pinned in the lockfile" \
  $PY scripts/guards/check_face_deps.py "${SCRATCH}/uv.lock"

# ---- 3. Evaluation-only data never trains ----------------------------------
# Every case below names its probe file explicitly, so the verdict is attributable to the
# probe and not to anything else on the training surface.

# 3a. A direct reference to the evaluation-only tree from an event training configuration.
write_event_probe() {  # write_event_probe <sources-block-line...>
  {
    printf 'schema_version: 1\ntask: fall\nwindow: {window_s: 8.0, stride_s: 1.5, n_steps: 64, policy: contain, min_present: 0.35}\nsources:\n'
    cat
    printf 'model: {hidden: 64, blocks: 3, dropout: 0.15}\ntrain: {epochs: 1, batch: 8, lr: 0.001, seed: 1, device: cpu}\nout_dir: outputs/verifier/_guard_selftest\n'
  } > configs/events/train__guard_selftest.yaml
}

write_event_probe <<'PROBE'
  - dir: data/eval_only/multicam_fall
    event_keys: [fall_start_t, fall_end_t]
PROBE
expect_reject "paths/direct reference to the evaluation-only tree" \
  $PY scripts/guards/check_train_data_paths.py configs/events/train__guard_selftest.yaml

# 3b. The same leak laundered through a symbolic link out of the training tree.
ln -sfn ../eval_only/multicam_fall data/train_ok/_guard_selftest_link
write_event_probe <<'PROBE'
  - dir: data/train_ok/_guard_selftest_link
    event_keys: [fall_start_t, fall_end_t]
PROBE
expect_reject "paths/symbolic link from the training tree into the evaluation-only tree" \
  $PY scripts/guards/check_train_data_paths.py configs/events/train__guard_selftest.yaml
rm -f data/train_ok/_guard_selftest_link

# 3c. An exam recording named in a training configuration's locations filter. This is the
# rule that keeps the exam honest: a model is never trained on the recordings it is scored on.
write_event_probe <<'PROBE'
  - dir: outputs/events/wu_sahzu_tracks_v2c
    locations: [pat01, pat03]
PROBE
expect_reject "paths/exam recording in a training configuration's locations filter" \
  $PY scripts/guards/check_train_data_paths.py configs/events/train__guard_selftest.yaml

# 3d. ...and the clinical tree with no filter at all is the same leak, wholesale.
write_event_probe <<'PROBE'
  - dir: outputs/events/wu_sahzu_tracks_v2c
PROBE
expect_reject "paths/clinical track tree with no locations filter" \
  $PY scripts/guards/check_train_data_paths.py configs/events/train__guard_selftest.yaml

# 3e. An unresolvable track tree: no extraction configuration produces it, so the corpus
# behind it cannot be named and the licence position is unknown.
write_event_probe <<'PROBE'
  - dir: outputs/events/_guard_selftest_unknown_v1
    event_keys: [fall_start_t, fall_end_t]
PROBE
expect_reject "registry/track tree no extraction configuration produces" \
  $PY scripts/guards/check_config_registry.py configs/events/train__guard_selftest.yaml

# 3f. An evaluation-only track tree. The corpus is registered and its bucket forbids training,
# and it has no registered train-side slice, so no filter can rescue it.
write_event_probe <<'PROBE'
  - dir: outputs/events/ur_fall_tracks_v2c
    event_keys: [fall_start_t, fall_end_t]
PROBE
expect_reject "registry/evaluation-only track tree in a training configuration" \
  $PY scripts/guards/check_config_registry.py configs/events/train__guard_selftest.yaml
rm -f configs/events/train__guard_selftest.yaml

# 3g. The prototype tree is readable only by a configuration that declares the prototype
# weight class. Without the declaration it is a leak, because the resulting checkpoint would
# carry no record that it came from prototype data.
printf 'schema_version: 1\nroot: data/train_prototype/mevid\ndataset_name: mevid\nout_dir: outputs/_guard_selftest\n' \
  > configs/reid/_guard_selftest.yaml
expect_reject "paths/prototype tree without the prototype declaration" \
  $PY scripts/guards/check_train_data_paths.py configs/reid/_guard_selftest.yaml
expect_reject "registry/prototype corpus without the prototype declaration" \
  $PY scripts/guards/check_config_registry.py configs/reid/_guard_selftest.yaml

# 3h. ...and a symbolic link out of the training tree into the prototype tree is the same leak.
ln -sfn ../train_prototype/mevid data/train_ok/_guard_selftest_proto_link
printf 'schema_version: 1\nroot: data/train_ok/_guard_selftest_proto_link\ndataset_name: mevid\nout_dir: outputs/_guard_selftest\n' \
  > configs/reid/_guard_selftest.yaml
expect_reject "paths/symbolic link from the training tree into the prototype tree" \
  $PY scripts/guards/check_train_data_paths.py configs/reid/_guard_selftest.yaml
rm -f data/train_ok/_guard_selftest_proto_link

# 3i. The declaration has exactly one admissible value, so a typo does not open the tree.
printf 'schema_version: 1\nweight_class: shippable\nroot: data/train_prototype/mevid\ndataset_name: mevid\nout_dir: outputs/_guard_selftest\n' \
  > configs/reid/_guard_selftest.yaml
expect_reject "paths/prototype tree under a weight class that is not prototype" \
  $PY scripts/guards/check_train_data_paths.py configs/reid/_guard_selftest.yaml

# 3j. A directory inside a legal tree with no registry entry: the location is right and the
# licence record, the provenance and the counsel review are all missing. The path guard passes
# this one by design, which is exactly why the registry layer exists.
printf 'schema_version: 1\nweight_class: prototype\nroot: data/train_ok/_guard_selftest_unregistered\nout_dir: outputs/_guard_selftest\n' \
  > configs/reid/_guard_selftest.yaml
expect_reject "registry/unregistered directory inside the training tree" \
  $PY scripts/guards/check_config_registry.py configs/reid/_guard_selftest.yaml

# 3k. ...and the same inside the prototype tree, where the declaration alone is not enough.
printf 'schema_version: 1\nweight_class: prototype\nroot: data/train_prototype/_guard_selftest_unregistered\nout_dir: outputs/_guard_selftest\n' \
  > configs/reid/_guard_selftest.yaml
expect_reject "registry/unregistered directory inside the prototype tree" \
  $PY scripts/guards/check_config_registry.py configs/reid/_guard_selftest.yaml

# 3l. The positive half: a declared prototype recipe reading a registered prototype root
# passes both layers. Checked on the probe alone, so the rest of the surface cannot mask it.
printf 'schema_version: 1\nweight_class: prototype\nroot: data/train_prototype/mevid\ndataset_name: mevid\nout_dir: outputs/_guard_selftest\n' \
  > configs/reid/_guard_selftest.yaml
expect_accept "paths/declared prototype recipe on a registered prototype root" \
  $PY scripts/guards/check_train_data_paths.py configs/reid/_guard_selftest.yaml
expect_accept "registry/declared prototype recipe on a registered prototype root" \
  $PY scripts/guards/check_config_registry.py configs/reid/_guard_selftest.yaml
rm -f configs/reid/_guard_selftest.yaml

# ---- 4. The fisheye primitives, banned outright -----------------------------
cat > src/vividly/media/_guard_selftest_primitives.py <<'PROBE'
import cv2

maps = cv2.fisheye.initUndistortRectifyMap(None, None, None, None, (640, 480), cv2.CV_16SC2)
PROBE
expect_reject "primitives/cv2 fisheye map builder anywhere in the tree" \
  $PY scripts/guards/check_no_fisheye_primitives.py src/vividly/media/_guard_selftest_primitives.py

# The resampling call on its own, through an aliased import, with no fisheye name in sight.
cat > src/vividly/media/_guard_selftest_primitives.py <<'PROBE'
import cv2 as cv

warped = cv.remap(None, None, None, cv.INTER_LINEAR)
PROBE
expect_reject "primitives/cv2 remap through an aliased import" \
  $PY scripts/guards/check_no_fisheye_primitives.py src/vividly/media/_guard_selftest_primitives.py
rm -f src/vividly/media/_guard_selftest_primitives.py

expect_accept "primitives/clean tree" $PY scripts/guards/check_no_fisheye_primitives.py

# ---- 5. Banned re-identification weight provenance --------------------------
# Both banned literals live on the two assignment lines below, each carrying the guard's own
# allow pragma, which this file may use because it sits in scripts/guards/. The probe files are
# written into configs/ and scripts/, where that pragma is inert, so what the guard sees there
# is an ordinary violation with no annotation to hide behind.
OSNET_REPACK='kadirnar/osnet'   # reid-guard: allow
OSNET_OWNER='kadirnar'          # reid-guard: allow
BANNED_BACKBONE='solider'       # reid-guard: allow

# 5a. A weight locator in a configuration, caught by the full-tree walk with no filenames.
printf 'embedder:\n  weights_url: https://huggingface.co/%s/resolve/main/osnet_x1_0.onnx\n' \
  "${OSNET_REPACK}" > configs/_guard_selftest.yaml
expect_reject "reid/repacked osnet locator in a configuration" \
  $PY scripts/guards/check_reid_weight_sources.py
rm -f configs/_guard_selftest.yaml

# 5b. A fetcher target in a script, caught in pre-commit mode with the filename passed.
printf 'WEIGHTS_URL = "https://huggingface.co/user/%s/resolve/main/%s_swin_tiny.pth"\n' \
  "${BANNED_BACKBONE}" "${BANNED_BACKBONE}" > scripts/_guard_selftest_fetch.py
expect_reject "reid/research-only backbone fetcher target in a script" \
  $PY scripts/guards/check_reid_weight_sources.py scripts/_guard_selftest_fetch.py
rm -f scripts/_guard_selftest_fetch.py

# 5c. The prose exemption is not a laundering hole. Outside the two authoring trees a
# locator-shaped literal is a fetch target even with no scheme and no file extension anywhere
# on the line, which is the shape that would walk straight past a guard looking for a scheme.
printf 'REPO_ID = "%s"\n' "${OSNET_REPACK}" > src/vividly/_guard_selftest_reid.py
expect_reject "reid/bare repository identifier in the package, no scheme, no extension" \
  $PY scripts/guards/check_reid_weight_sources.py src/vividly/_guard_selftest_reid.py
rm -f src/vividly/_guard_selftest_reid.py

# 5d. Separator games do not dodge the deny-list: the squashed second pass catches a
# camel-cased spelling no literal key contains.
printf 'weights: TransReIDSSL_vit_base.pth\n' > configs/_guard_selftest.yaml  # reid-guard: allow
expect_reject "reid/camel-cased variant in a configuration" \
  $PY scripts/guards/check_reid_weight_sources.py
rm -f configs/_guard_selftest.yaml

# 5e. The bare owner, split from its repository across two keys, reassembles at load time but
# leaves neither line carrying the joined identifier. The bare-owner key catches it.
printf 'embedder:\n  owner: %s\n  repo: osnet\n  file: osnet_x1_0.onnx\n' "${OSNET_OWNER}" \
  > configs/_guard_selftest.yaml
expect_reject "reid/bare owner split across two configuration keys" \
  $PY scripts/guards/check_reid_weight_sources.py
rm -f configs/_guard_selftest.yaml

# 5f. The same split in Python: an owner variable pasted into the identifier through a format
# string. A bare owner literal is a fetch fragment rather than prose, because it carries no
# internal whitespace, so it fails with nothing else on the line.
printf '_OWNER = "%s"\nURL = f"https://huggingface.co/{_OWNER}/osnet/resolve/main/x.onnx"\n' \
  "${OSNET_OWNER}" > src/vividly/_guard_selftest_reid.py
expect_reject "reid/owner variable pasted into an identifier in the package" \
  $PY scripts/guards/check_reid_weight_sources.py src/vividly/_guard_selftest_reid.py
rm -f src/vividly/_guard_selftest_reid.py

# 5g. A workflow step fetching a banned checkpoint, caught by the full-tree walk.
mkdir -p .github/workflows
printf 'jobs:\n  fetch:\n    steps:\n      - run: curl -L https://huggingface.co/%s/resolve/main/osnet_x1_0.onnx -o w.onnx\n' \
  "${OSNET_REPACK}" > .github/workflows/_guard_selftest.yml
expect_reject "reid/banned checkpoint fetched in a workflow step" \
  $PY scripts/guards/check_reid_weight_sources.py
rm -f .github/workflows/_guard_selftest.yml

# 5h. The same recipe in the root Makefile, which is also in scope. Mutated in place like
# pyproject.toml above, and restored unconditionally by the trap.
cp Makefile "${SCRATCH}/Makefile.orig"
printf '\n_guard_selftest_fetch:\n\tcurl -L https://huggingface.co/%s/resolve/main/osnet_x1_0.onnx -o w.onnx\n' \
  "${OSNET_REPACK}" >> Makefile
expect_reject "reid/banned checkpoint fetched in a make recipe" \
  $PY scripts/guards/check_reid_weight_sources.py
cp -f "${SCRATCH}/Makefile.orig" Makefile
rm -f "${SCRATCH}/Makefile.orig"

# 5i. A locator split across two physical lines by implicit string concatenation, broken
# mid-owner so no single line carries the whole owner. The adjacent-literal join and the
# whole-file squashed pass are what reassemble it.
printf 'WEIGHTS = (\n    "https://huggingface.co/kadir"\n    "nar/osnet/resolve/main/osnet_x1_0.onnx"\n)\n' \
  > src/vividly/_guard_selftest_reid.py
expect_reject "reid/locator split across adjacent Python string literals" \
  $PY scripts/guards/check_reid_weight_sources.py src/vividly/_guard_selftest_reid.py
rm -f src/vividly/_guard_selftest_reid.py

# 5j. A locator split across physical lines by a backslash continuation inside a quoted scalar,
# broken mid-lineage. Each line is clean; the whole-file squashed pass rejoins them.
printf 'weights: "transreid\\\nssl_vit_base.pth"\n' > configs/_guard_selftest.yaml
expect_reject "reid/locator split by a backslash continuation in a quoted scalar" \
  $PY scripts/guards/check_reid_weight_sources.py
rm -f configs/_guard_selftest.yaml

# 5k. ...and the tree as checked in must pass, probes removed. This runs last on purpose:
# every probe above is gone by now, so a failure here is a real false positive, most likely a
# new file naming a banned source outside the document tree.
expect_accept "reid/clean tree" $PY scripts/guards/check_reid_weight_sources.py

# ---- 6. No new fisheye reference --------------------------------------------
# The tree still carries records that name what was retired, so the guard exempts the files in
# scripts/guards/no_fisheye_allowlist.txt and fails on a reference in any file that is not on
# it. Every probe below is a new file, which is exactly the shape the rule bans.

# 6a. A new configuration naming a quarantined corpus, caught by the full-tree walk.
printf 'dataset: cepdof\nsplit: train\nimages: data/train_prototype/cepdof/\n' \
  > configs/_guard_selftest_dataset.yaml
expect_reject "nofisheye/new configuration naming a quarantined corpus" \
  $PY scripts/guards/check_no_fisheye.py
rm -f configs/_guard_selftest_dataset.yaml

# 6b. A new script reaching for the retired camera model, caught in pre-commit mode with the
# filename passed. The camel case is on purpose: the piece splitter has to read FisheyeModel as
# two pieces, which a plain word-boundary expression would not.
printf 'PRESET_NAME = "dome_143"\nMODEL = FisheyeModel.from_fov(PRESET_NAME, 960, 960, 143.0)\n' \
  > scripts/_guard_selftest_render.py
expect_reject "nofisheye/new script instantiating the retired camera model" \
  $PY scripts/guards/check_no_fisheye.py scripts/_guard_selftest_render.py
rm -f scripts/_guard_selftest_render.py

# 6c. The filename alone is a reference: a new file named for the retired domain whose contents
# say nothing. This is how a dropped-in fixture or preset is caught with no readable text.
printf 'preset: dome_143\nk: [-0.04, 0.005, 0, 0]\n' > configs/_guard_selftest_fisheye_dome.yaml
expect_reject "nofisheye/new file named for the retired domain, clean contents" \
  $PY scripts/guards/check_no_fisheye.py configs/_guard_selftest_fisheye_dome.yaml
rm -f configs/_guard_selftest_fisheye_dome.yaml

# 6d. The false-positive control that shapes the whole matcher: one quarantined corpus name is
# a piece of an ordinary weekday, so a substring deny-list would reject every weekly schedule
# in the repository. Word pieces, not substrings, and this case is what keeps it that way.
printf 'schedule:\n  day: Friday\n  note: fridays are the burn-in slot\n' \
  > configs/_guard_selftest_weekly.yaml
expect_accept "nofisheye/an ordinary weekday is not a corpus name" \
  $PY scripts/guards/check_no_fisheye.py configs/_guard_selftest_weekly.yaml
rm -f configs/_guard_selftest_weekly.yaml

# 6e. The inventory is a list of exact paths and never a pattern. One wildcard line would
# exempt a whole tree, including files that do not exist yet, so it is refused outright.
# Mutated in place like pyproject.toml above, and restored unconditionally by the trap.
cp scripts/guards/no_fisheye_allowlist.txt "${SCRATCH}/no_fisheye_allowlist.orig"
printf 'configs/*\n' >> scripts/guards/no_fisheye_allowlist.txt
expect_reject "nofisheye/wildcard entry in the reference inventory" \
  $PY scripts/guards/check_no_fisheye.py
cp -f "${SCRATCH}/no_fisheye_allowlist.orig" scripts/guards/no_fisheye_allowlist.txt
rm -f "${SCRATCH}/no_fisheye_allowlist.orig"

# 6f. ...and the tree as checked in must pass, probes removed. A failure here means either a
# real new reference or an inventory entry whose file has been renamed out from under it.
expect_accept "nofisheye/clean tree" $PY scripts/guards/check_no_fisheye.py

# ---- 8. Clip Library profiles (CLIP-LIBRARY-DESIGN.md P2) ----------------------
# A profile that writes a Clip spool is held to the non-device rules whatever its name. Each
# probe below is the committed synthetic profile with exactly one rule broken.
clip_probe() {  # clip_probe <suffix> <python-transform-expression>
  $PY - "$1" "$2" <<'PYEOF'
import sys, re
from pathlib import Path
suffix, transform = sys.argv[1], sys.argv[2]
text = Path("configs/pipeline/clip_library_synth.yaml").read_text(encoding="utf-8")
text = eval(transform)
Path(f"configs/pipeline/_guard_selftest_clip_{suffix}.yaml").write_text(text, encoding="utf-8")
PYEOF
}
clip_probe verify 'text + "\nverify:\n  policy: annotate\n  heads: {}\n"'
expect_reject "clip/verify block in a spool-writing profile" \
  $PY scripts/guards/check_clip_library_config.py configs/pipeline/_guard_selftest_clip_verify.yaml
clip_probe head 'text.replace("    - egress_without_discharge_v0\n", "    - egress_without_discharge_v0\n    - seizure_candidate_v0\n", 1)'
expect_reject "clip/third head listed in rules.enabled" \
  $PY scripts/guards/check_clip_library_config.py configs/pipeline/_guard_selftest_clip_head.yaml
clip_probe store 'text.replace("path: \":memory:\"", "path: outputs/pipeline/alerts.sqlite")'
expect_reject "clip/on-disk store outside the data directory" \
  $PY scripts/guards/check_clip_library_config.py configs/pipeline/_guard_selftest_clip_store.yaml
clip_probe reassoc 'text.replace("persist: false", "persist: true")'
expect_reject "clip/persisted re-association" \
  $PY scripts/guards/check_clip_library_config.py configs/pipeline/_guard_selftest_clip_reassoc.yaml
clip_probe enabled 're.sub(r"  enabled:\n(    - .*\n)+", "", text)'
expect_reject "clip/rules.enabled omitted" \
  $PY scripts/guards/check_clip_library_config.py configs/pipeline/_guard_selftest_clip_enabled.yaml
clip_probe outputs 'text.replace("dir: ${VIVIDLY_CLIP_DATA}/pipeline", "dir: outputs/pipeline")'
expect_reject "clip/output under outputs/" \
  $PY scripts/guards/check_clip_library_config.py configs/pipeline/_guard_selftest_clip_outputs.yaml
clip_probe index 'text.replace("    - rules/engine.yaml\n", "    - rules.yaml\n", 1)'
expect_reject "clip/the rules index (every head) included" \
  $PY scripts/guards/check_clip_library_config.py configs/pipeline/_guard_selftest_clip_index.yaml
rm -f configs/pipeline/_guard_selftest_clip_*.yaml
expect_accept "clip/clean tree" $PY scripts/guards/check_clip_library_config.py

# ---- 7. The training surface as checked in ----------------------------------
# Both training guards must be quiet on the tree as it stands. A failure here is a real
# finding: a training configuration reaching data the registry does not admit.
expect_accept "paths/clean tree" $PY scripts/guards/check_train_data_paths.py
expect_accept "registry/clean tree" $PY scripts/guards/check_config_registry.py

if [ "$cases" -lt "$MIN_CASES" ]; then
  echo "FAIL: only ${cases} rejection cases ran, expected at least ${MIN_CASES}"
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "guard selftest: PASS (${cases} rejection cases, ${accepts} clean-tree case(s))"
else
  echo "guard selftest: FAIL (${cases} rejection cases, ${accepts} clean-tree case(s))"
fi
exit "$fail"
