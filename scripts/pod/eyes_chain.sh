#!/usr/bin/env bash
# Eyes v2 pod chain: B0, then I1, I2, B1 seed 1, B1 seed 2, soups and B2-DINOv2T.
#
#   bash scripts/pod/eyes_chain.sh [--dry-run] [--from STAGE] [--only STAGE] [--list]
#
# Run on the pod, detached, from the repository root:
#   nohup bash scripts/pod/eyes_chain.sh \
#       > /workspace/xlogs/eyes_chain.out 2>&1 &
#   tail -f /workspace/xlogs/eyes_chain.status
#
# ~28 h of GPU work. It is written to survive a disconnection: every
# stage appends a timestamped mark to the status file, and a re-run skips every
# stage already marked _OK. Killing it mid-stage and re-launching resumes at that
# stage. Fail-fast: the first _FAIL aborts the chain (except the B2 VRAM gate,
# which is a designed skip, not a failure).
#
# Queue order and rules: research/eyes-instrument-protocol.md (I1, I2, I3)
# and TRAINING.md, section "Eyes: person detector". Grading happens on the Mac:
# this chain only produces artefacts (detections json, onnx, checkpoints, logs).
#
# Verified from the Mac (a read of both trainers at their PINS.yaml pins, 2026-09-05):
#  - `--test-only` is `train.py -c CFG --test-only -r CKPT`  (the --test-only branch
#    of train.py and its argument definition)
#  - `--output-dir` overrides the config's output_dir       (the --output-dir branch
#    of train.py and its argument definition)
#  - `--test-only` writes only eval.pth (accumulate arrays), never detections, so
#    dump_detections_v2.py is run, and its header carries the full evidence
#    (the val() path in det_solver.py in both repos)
#  - export_onnx.py writes beside the checkpoint: resume.replace('.pth', '.onnx')
#    (the output-path line in tools/deployment/export_onnx.py, patched to opset 17
#    and batch-1)
#  - ulimit is load-bearing: the container's 1024-fd soft limit kills the first
#    epoch's eval (the ulimit note in setup_pod.sh's summary)

set -u
set -o pipefail

# --------------------------------------------------------------------- config
WORKSPACE="${WORKSPACE:-/workspace}"
# Defaults to the checkout this script lives in, derived the way setup_pod.sh derives
# its own root; set REPO= to point the chain at a different checkout.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TP="${TP:-$WORKSPACE/third_party}"
CKPT="${CKPT:-$WORKSPACE/ckpt}"
XLOGS="${XLOGS:-$WORKSPACE/xlogs}"
STATUS="${STATUS:-$XLOGS/eyes_chain.status}"
OUTROOT="${OUTROOT:-$WORKSPACE/outputs}"
# Where the operator scp's the three uploads to, before launching.
STAGING="${STAGING:-$WORKSPACE/staging}"
# DINOv2 teacher assets (repo clone + original state dict) live together here.
# These two paths are hardcoded in third_party/configs/rtv4_s_person_v2_dinov2t.yml
# (dinov2_repo_path and weights_path); changing one requires changing the other.
TEACHERS="${TEACHERS:-$WORKSPACE/teachers}"
DINOV2_REPO="$TEACHERS/dinov2"
DINOV2_WEIGHTS="$TEACHERS/dinov2_vitb14_pretrain.pth"
# setup_pod.sh's stage_smartspaces symlinks this dir at the repo's SmartSpaces bucket.
ANN_DIR="${ANN_DIR:-$WORKSPACE/data/person_v2/annotations}"

RTV4="$TP/RT-DETRv4"
DEIM="$TP/DEIM"
RTV4_CFG="${RTV4_CFG:-configs/rtv4/rtv4_s_person_v2_noteacher.yml}"
# Verified: this exact string is echoed by the run that produced e98de401 in
# outputs/pod_backup/train_seed1.log line 1, "'config': 'configs/deim_dfine/deim_s_person_v2.yml'".
# Source file third_party/configs/deim_s_person_v2.yml, installed here by
# setup_pod.sh's stage_trainers.
DEIM_CFG="${DEIM_CFG:-configs/deim_dfine/deim_s_person_v2.yml}"
# This file exists at $REPO/third_party/configs/$B2_CFG_NAME and is installed into the
# trainer tree by B2_FETCH_TEACHER.
B2_CFG_NAME="${B2_CFG_NAME:-rtv4_s_person_v2_dinov2t.yml}"
B2_CFG="configs/rtv4/$B2_CFG_NAME"
B2_CFG_SRC="$REPO/third_party/configs/$B2_CFG_NAME"

PINS="$REPO/third_party/PINS.yaml"
DINOV2_URL_FALLBACK="https://github.com/facebookresearch/dinov2"

# The uploads, as staged by the operator.
UP_B1="$STAGING/b1_best_stg2_345c30a5.pth"
UP_DEIM="$STAGING/best_stg2_seed1_e98de401.pth"
UP_DEVVAL="$STAGING/person_v2_devval.json"

B1_SEED0_CKPT="$CKPT/b1_best_stg2_345c30a5.pth"
DEIM_S1_CKPT="$CKPT/best_stg2_seed1_e98de401.pth"
RTV4_INIT="$CKPT/rtv4_s_coco.pth"

B1_SEED0_AP="${B1_SEED0_AP:-0.8356}"   # research/eyes-rtdetrv4-training-record.md

DUMPS="$XLOGS/dumps"
SOUPS="$CKPT/soups"

VRAM_GATE_MIB="${VRAM_GATE_MIB:-30500}"   # research/eyes-instrument-protocol.md, I3
VRAM_WINDOW="${VRAM_WINDOW:-900}"         # 15 min
VRAM_POLL="${VRAM_POLL:-30}"

# Set to 1 only as a deliberate act: it lets B2_FETCH_TEACHER clone dinov2 at whatever
# the default branch happens to be. The case below is the guard that refuses an
# unpinned teacher tree; third_party/PINS.yaml records repos.dinov2.pin today, so the
# guard covers a future gap rather than the current state.
ALLOW_UNPINNED_TEACHER="${ALLOW_UNPINNED_TEACHER:-0}"

DRY_RUN=0
FROM_STAGE=""
ONLY_STAGE=""

STAGES=(
  B0_TOOLS_ENV B0_SETUP B0_SMARTSPACES B0_STAGING
  I1_DUMP_RTV4 I1_DUMP_DEIM I1_DUMP_RTV4_HFLIP I1_DUMP_DEIM_HFLIP
  I2_EXPORT
  B1_SEED1 B1_SEED2
  SOUP_UNIFORM3 SOUP_GREEDY
  B2_FETCH_TEACHER B2_DINOV2T
)

# ---------------------------------------------------------------------- utils
now() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say() { printf '\n== [%s] %s\n' "$(now)" "$*"; }
mark() { mkdir -p "$XLOGS"; printf '%s %s\n' "$(now)" "$1" | tee -a "$STATUS"; }
# Substring match on " TOKEN" (marks are "<iso8601> <TOKEN>"). Deliberately a
# prefix match so has_mark "X_SKIPPED" catches X_SKIPPED_VRAM_GATE and similar; no _OK
# mark is a prefix of any other, so _OK lookups stay exact in practice.
has_mark() { [ -f "$STATUS" ] && grep -qF -- " $1" "$STATUS"; }

die() {
  printf '\nFATAL [%s]: %s\n' "$(now)" "$*" >&2
  mark "CHAIN_ABORTED"
  exit 1
}

# Run one stage. Skips when already _OK. Marks _OK / _FAIL. Fail-fast.
#   stage NAME "human description" -- cmd args...
stage() {
  local name="$1" desc="$2"; shift 3   # shift past name, desc and the literal --
  if [ -n "$ONLY_STAGE" ] && [ "$ONLY_STAGE" != "$name" ]; then return 0; fi
  if has_mark "${name}_OK"; then
    say "SKIP $name (already _OK in $STATUS)"
    return 0
  fi
  if has_mark "${name}_SKIPPED"; then
    say "SKIP $name (previously skipped by design)"
    return 0
  fi
  local log="$XLOGS/${name}.log"
  say "$name: $desc"
  printf '   log: %s\n   cmd: %s\n' "$log" "$*"
  if [ "$DRY_RUN" = 1 ]; then
    echo "   (dry run, not executed)"
    return 0
  fi
  mkdir -p "$XLOGS"
  mark "${name}_START"
  if "$@" >>"$log" 2>&1; then
    # a stage that marked itself _SKIPPED (the B2 gate) must not also read _OK
    if has_mark "${name}_SKIPPED"; then
      say "$name skipped by design, see $STATUS"
    else
      mark "${name}_OK"
    fi
  else
    local rc=$?
    mark "${name}_FAIL"
    die "$name exited $rc. Read $log. Nothing after this stage has run."
  fi
}

raise_ulimit() {
  # The ulimit note in setup_pod.sh's summary: the 1024-fd soft limit kills the first
  # epoch's eval
  # ("received 0 items of ancdata"). It must be raised in the same shell as the
  # launch, so every training helper below calls this itself.
  local hard; hard="$(ulimit -Hn 2>/dev/null || echo 1048576)"
  [ "$hard" = "unlimited" ] && hard=1048576
  ulimit -n "$hard" 2>/dev/null || ulimit -n 1048576 2>/dev/null || true
  echo "   ulimit -n = $(ulimit -n)"
}

# Parse the COCO AP out of a --test-only log. Fails loudly, never guesses.
parse_ap() {
  local log="$1" ap
  ap="$(grep -aE 'Average Precision.*IoU=0\.50:0\.95.*area=[[:space:]]*all' "$log" \
        | tail -1 | grep -oE '[0-9]*\.[0-9]+[[:space:]]*$' | tr -d '[:space:]')"
  if [ -z "$ap" ]; then
    ap="$(grep -a 'best_stat' "$log" | tail -1 | grep -oE '0\.[0-9]{4,}' | head -1)"
  fi
  [ -n "$ap" ] || die "no COCO AP line and no best_stat line in $log"
  printf '%s' "$ap"
}

sha() { sha256sum "$1" | cut -d' ' -f1; }

# --------------------------------------------------------------------- checks
preflight() {
  say "preflight"
  command -v nvidia-smi >/dev/null || die "nvidia-smi missing; is this a GPU pod?"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  [ -d "$REPO" ] || die "repo not at $REPO (set REPO=)"
  [ -x "$REPO/scripts/pod/setup_pod.sh" ] || die "$REPO/scripts/pod/setup_pod.sh missing"
  mkdir -p "$XLOGS" "$DUMPS" "$SOUPS" "$CKPT" "$TEACHERS"
  export PATH="$HOME/.local/bin:$PATH"

  # B0_TOOLS_ENV installs uv. Only complain if that stage is being
  # skipped past (--from or a stale _OK mark) and uv still is not there, because
  # then stage_smartspaces' `uv run` (setup_pod.sh's stage_smartspaces) would fail mid-chain.
  if ! command -v uv >/dev/null 2>&1; then
    if has_mark "B0_TOOLS_ENV_OK"; then
      die "uv not on PATH but B0_TOOLS_ENV is marked _OK in $STATUS.
     Either the mark is stale or ~/.local/bin is not on PATH. Fix with:
       bash $REPO/scripts/pod/setup_pod.sh tools env"
    fi
    echo "   uv absent; B0_TOOLS_ENV will install it (stage 1)"
  fi

  # The operator scp's the uploads into $STAGING; B0_STAGING installs
  # them into $CKPT. Accept either location so a re-run after install still passes.
  local missing=0 pair name staged installed
  for pair in "b1_best_stg2_345c30a5.pth" "best_stg2_seed1_e98de401.pth"; do
    staged="$STAGING/$pair"; installed="$CKPT/$pair"
    if [ -f "$installed" ]; then
      echo "   installed: $installed"
    elif [ -f "$staged" ]; then
      echo "   staged:    $staged"
    else
      echo "   missing:   $staged  (or $installed)"
      missing=1
    fi
  done
  if [ "$missing" != 0 ]; then
    die "upload the checkpoints above before launching. From the Mac:
       scp outputs/pod_backup/b1_best_stg2_345c30a5.pth   POD:$STAGING/
       scp outputs/pod_backup/best_stg2_seed1_e98de401.pth POD:$STAGING/
       scp <person_v2_devval.json>                         POD:$STAGING/
     (use scp, not rsync --info=progress2)"
  fi
  [ -f "$UP_DEVVAL" ] || [ -f "$ANN_DIR/person_v2_devval.json" ] || \
    echo "   NOTE: no $UP_DEVVAL and no $ANN_DIR/person_v2_devval.json yet;
     B0_SMARTSPACES is expected to produce it. B0_STAGING re-checks and fails if not."
  mark "PREFLIGHT_OK"
}

# ------------------------------------------------------------------ B0 setup
# Tools and env first. stage_smartspaces shells out to `uv run` (setup_pod.sh's stage_smartspaces),
# so `tools` (installs uv) and `env` (uv sync --extra cuda) are its prerequisites.
# The declared list in TRAINING.md, section "Eyes: person detector", is "trainers ckpts"
# plus "smartspaces"; this stage is that list's unstated prerequisite made explicit, not a
# widening of it.
b0_tools_env()   { bash "$REPO/scripts/pod/setup_pod.sh" tools env; }
b0_setup()       { bash "$REPO/scripts/pod/setup_pod.sh" trainers ckpts; }
b0_smartspaces() { bash "$REPO/scripts/pod/setup_pod.sh" smartspaces; }

# Install the three uploads. Runs after smartspaces so it can see whether the
# annotations the trainers point at already exist. Never overwrites.
b0_staging() {
  mkdir -p "$CKPT" "$ANN_DIR"
  local pair src dst
  for pair in "b1_best_stg2_345c30a5.pth" "best_stg2_seed1_e98de401.pth"; do
    src="$STAGING/$pair"; dst="$CKPT/$pair"
    if [ -f "$dst" ]; then
      echo "already installed: $dst"
    elif [ -f "$src" ]; then
      cp "$src" "$dst" || return 1
      echo "installed: $src -> $dst"
    else
      echo "missing both $src and $dst"; return 1
    fi
    echo "  sha256 $(sha "$dst")"
  done

  # The dev-val annotations. `smartspaces` regenerates these from the repo bucket
  # (setup_pod.sh's stage_smartspaces), so the upload is the fallback for a run where that stage
  # was skipped. Whatever smartspaces produced takes precedence; this never overwrites it.
  dst="$ANN_DIR/person_v2_devval.json"
  if [ -f "$dst" ]; then
    echo "dev-val annotations already present (smartspaces): $dst"
    [ -f "$UP_DEVVAL" ] && echo "  NOTE: $UP_DEVVAL also staged and was not used; the pipeline's own output takes precedence"
  elif [ -f "$UP_DEVVAL" ]; then
    cp "$UP_DEVVAL" "$dst" || return 1
    echo "installed dev-val annotations: $UP_DEVVAL -> $dst"
  else
    echo "no dev-val annotations at $dst and none staged at $UP_DEVVAL"
    echo "I1 dumps and every --test-only score read this file. Upload it or re-run B0_SMARTSPACES."
    return 1
  fi
  echo "  sha256 $(sha "$dst")"
  # Frames are large and are not staged through this path; they come from
  # smartspaces. Fail early and loudly rather than at the first dataloader batch.
  [ -d "$WORKSPACE/data/person_v2/frames/train" ] || {
    echo "no $WORKSPACE/data/person_v2/frames/train; B0_SMARTSPACES did not finish"; return 1; }
}

# --------------------------------------------------------------------- I1 dumps
# --test-only emits no detections (verified; see dump_detections_v2.py header), so
# the dumps come from this repository's own script running inside each trainer's venv.
dump() {  # trainer_dir config ckpt out_json [--hflip]
  local trainer="$1" cfg="$2" ckpt="$3" out="$4"; shift 4
  # return, not die: a missing trainer venv is this stage failing (so it gets a
  # _FAIL mark and a named log), not a chain-level invariant violation
  [ -x "$trainer/.venv/bin/python" ] || {
    echo "$trainer/.venv/bin/python missing; B0_SETUP did not finish"; return 1; }
  # the ulimit law applies to every dataloader spawn, not only to training: the DEIM
  # val loader died with 'received 0 items of ancdata' here (I1_DUMP_DEIM_FAIL,
  # 2026-09-05) exactly as the B0 setup notes warned for epoch-1 eval
  raise_ulimit
  ( cd "$trainer" && "$trainer/.venv/bin/python" \
      "$REPO/scripts/detector/dump_detections_v2.py" run \
      -c "$cfg" -r "$ckpt" --out "$out" "$@" )
}

i1_rtv4()       { dump "$RTV4" "$RTV4_CFG" "$B1_SEED0_CKPT" "$DUMPS/dets_b1_rtv4s_seed0_identity.json"; }
i1_deim()       { dump "$DEIM" "$DEIM_CFG" "$DEIM_S1_CKPT"  "$DUMPS/dets_deim_s_seed1_identity.json"; }
i1_rtv4_hflip() { dump "$RTV4" "$RTV4_CFG" "$B1_SEED0_CKPT" "$DUMPS/dets_b1_rtv4s_seed0_hflip.json" --hflip; }
i1_deim_hflip() { dump "$DEIM" "$DEIM_CFG" "$DEIM_S1_CKPT"  "$DUMPS/dets_deim_s_seed1_hflip.json" --hflip; }

# ------------------------------------------------------------------- I2 export
i2_export() {
  local onnx="${B1_SEED0_CKPT%.pth}.onnx"
  raise_ulimit
  ( cd "$RTV4" && "$RTV4/.venv/bin/python" tools/deployment/export_onnx.py \
      --check -c "$RTV4_CFG" -r "$B1_SEED0_CKPT" ) || return 1
  [ -f "$onnx" ] || { echo "export produced no $onnx"; return 1; }
  echo "onnx     $onnx"
  echo "sha256   $(sha "$onnx")"
  sha "$onnx" > "$onnx.sha256"
  echo "Pull to the Mac: scp this .onnx and .sha256, then run the ORT grader and CPU bench."
}

# -------------------------------------------------------------------- B1 seeds
train_seed() {  # seed
  # One `local` per line when a later assignment references an earlier one: bash
  # expands `$seed` in the second word before `local` assigns the first, and under
  # set -u that aborts the whole chain with no mark written
  # (B1_SEED1, pod, 2026-09-05, 3.7 h of silence before it was caught).
  local seed="$1"
  local out="$OUTROOT/rtv4_s_person_v2_seed$seed"
  raise_ulimit
  ( cd "$RTV4" && "$RTV4/.venv/bin/python" train.py -c "$RTV4_CFG" \
      --use-amp --seed="$seed" -t "$RTV4_INIT" --output-dir "$out" ) || return 1
  [ -f "$out/best_stg2.pth" ] || { echo "no best_stg2.pth in $out"; return 1; }
  # Watch for the NaN trap (PINS.yaml contingencies.nan_amp): both trainers
  # keep going after a NaN and consume pod hours silently.
  if [ -f "$RTV4/NaN.pth" ]; then
    echo "WARNING: $RTV4/NaN.pth exists; this seed may be a NaN run. Inspect it before using it."
  fi
  cp "$out/best_stg2.pth" "$CKPT/b1_best_stg2_seed$seed.pth"
  echo "seed $seed best_stg2 sha256 $(sha "$CKPT/b1_best_stg2_seed$seed.pth")"
}

b1_seed1() { train_seed 1; }
b1_seed2() { train_seed 2; }

test_only() {  # ckpt tag -> prints AP on stdout, leaves $XLOGS/testonly_<tag>.log
  # Note: callers capture this with $(...), so a `die` in here only kills the
  # subshell. Every call site must check that the captured value is non-empty;
  # `require_ap` below exists so that is one line and never forgotten.
  local ckpt="$1" tag="$2"
  local log="$XLOGS/testonly_$tag.log"  # split: same set -u trap as train_seed
  raise_ulimit >&2   # stdout here is the AP channel callers capture; nothing else may print to it
  ( cd "$RTV4" && "$RTV4/.venv/bin/python" train.py -c "$RTV4_CFG" \
      --test-only -r "$ckpt" --output-dir "$OUTROOT/testonly_$tag" ) >"$log" 2>&1 \
    || die "--test-only failed for $ckpt; read $log"
  parse_ap "$log"
}

require_ap() {  # takes value and label; returns 1, failing the stage, rather than scoring a blank
  if [ -z "${1:-}" ]; then
    echo "FATAL: no AP recovered for $2; refusing to record a blank score"
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------- soups
soup_uniform3() {
  # Scope cut (2026-09-06, the reduced path): under the status mark
  # B1_SEED2_SCOPE_CUT this builds a uniform-2 soup over seeds 0 and 1,
  # measures seed-1's own AP here (greedy, which normally measures it, is skipped
  # under the cut), and the pre-declared bar becomes
  #     best single seed + max(0.0012, |s0 - s1|).
  # Declared before any soup or seed-1 score existed. The stage token stays
  # SOUP_UNIFORM3 so resume marks stay stable; the artifacts say uniform2.
  local a="$B1_SEED0_CKPT" b="$CKPT/b1_best_stg2_seed1.pth" c="$CKPT/b1_best_stg2_seed2.pth"
  local n=3
  if has_mark "B1_SEED2_SCOPE_CUT"; then n=2; fi
  local out="$SOUPS/soup_uniform${n}.pth"
  local ingredients=("$a" "$b")
  [ "$n" = 3 ] && ingredients+=("$c")
  for f in "${ingredients[@]}"; do [ -f "$f" ] || { echo "missing ingredient $f"; return 1; }; done
  local ckpt_args=()
  for f in "${ingredients[@]}"; do ckpt_args+=(--ckpt "$f"); done
  # the trainer venv, not $REPO/.venv: soup_v2.py needs only stdlib + torch, and
  # B0 guarantees this venv exists whereas the repo env comes from B0_TOOLS_ENV
  "$RTV4/.venv/bin/python" "$REPO/scripts/detector/soup_v2.py" uniform --family rtv4_s \
      "${ckpt_args[@]}" --out "$out" \
      --note "uniform-$n of B1 seeds" || return 1
  if [ "$n" = 2 ]; then
    local ap1
    ap1="$(test_only "$b" "seed1")"
    require_ap "$ap1" "seed 1" || return 1
    echo "seed AP: s0=$B1_SEED0_AP s1=$ap1"
    printf '%s %s\n' "$B1_SEED0_AP" "$ap1" > "$SOUPS/seed_aps.txt"
  fi
  local ap; ap="$(test_only "$out" "soup_uniform${n}")"
  require_ap "$ap" "uniform-$n soup" || return 1
  echo "Uniform-$n soup AP = $ap   sha256 $(sha "$out")"
  echo "$ap" > "$SOUPS/soup_uniform${n}.ap"
  echo "Bar: best single seed plus max(0.0012, measured seed spread), computed on the Mac."
}

soup_greedy() {
  local a="$B1_SEED0_CKPT" b="$CKPT/b1_best_stg2_seed1.pth" c="$CKPT/b1_best_stg2_seed2.pth"
  local st="$SOUPS/greedy_state.json" ap1 ap2 py="$RTV4/.venv/bin/python"
  for f in "$a" "$b" "$c"; do [ -f "$f" ] || { echo "missing ingredient $f"; return 1; }; done
  ap1="$(test_only "$b" "seed1")"; require_ap "$ap1" "seed 1" || return 1
  ap2="$(test_only "$c" "seed2")"; require_ap "$ap2" "seed 2" || return 1
  echo "seed AP: s0=$B1_SEED0_AP s1=$ap1 s2=$ap2"
  printf '%s %s %s\n' "$B1_SEED0_AP" "$ap1" "$ap2" > "$SOUPS/seed_aps.txt"

  # soup_v2.py greedy emits one candidate per call and takes the score back on the
  # next call, so scoring stays a real --test-only rather than a guess.
  rm -f "$st"
  "$py" "$REPO/scripts/detector/soup_v2.py" greedy --family rtv4_s --state "$st" \
      --out-dir "$SOUPS" \
      --ckpt "$a" --score "$B1_SEED0_AP" --ckpt "$b" --score "$ap1" --ckpt "$c" --score "$ap2" \
      || return 1
  local i cand score
  for i in 1 2; do   # 3 ingredients give at most 2 candidates
    cand="$("$py" -c "import json,sys;s=json.load(open(sys.argv[1]));p=s.get('pending');print(p['out'] if p else '')" "$st")"
    [ -n "$cand" ] || break
    score="$(test_only "$cand" "greedy_$i")"
    require_ap "$score" "greedy candidate $i" || return 1
    echo "greedy candidate $i: $cand -> $score"
    "$py" "$REPO/scripts/detector/soup_v2.py" greedy --state "$st" --out-dir "$SOUPS" \
        --report-score "$score" || return 1
  done
  echo "--- greedy state ---"; cat "$st"
}

# ---------------------------------------------------------- B2 teacher assets
# Read one dotted key out of PINS.yaml. Uses the trainer venv's python (PyYAML is
# in RT-DETRv4/requirements.txt), not the repo env, for the same reason soup_v2.py
# does: B0_SETUP guarantees the trainer venv exists.
pins_get() {  # takes a dotted key, prints the value on stdout, empty if absent
  "$RTV4/.venv/bin/python" - "$PINS" "$1" <<'PY' 2>/dev/null
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
node = cfg
for part in sys.argv[2].split('.'):
    if not isinstance(node, dict) or part not in node:
        sys.exit(0)
    node = node[part]
print(node if node is not None else '')
PY
}

# The assertion that research/eyes-dinov2-teacher-arm.md section 5 calls the only guard.
# The load_state_dict call in dinov2_teacher.py passes strict=False: a state dict whose
# key namespace does not match, for example the Hugging Face `transformers` port
# facebook/dinov2-base, whose keys are embeddings.* and encoder.layer.N.* rather than
# the original patch_embed.* and blocks.N.*, loads zero tensors, logs the miss at INFO
# where nothing surfaces it, and raises nothing. The teacher would then train against
# random weights for about 13 h. This runs the exact same two calls on CPU, in seconds,
# before any GPU time is spent.
assert_teacher_loads_clean() {
  "$RTV4/.venv/bin/python" - "$DINOV2_REPO" "$DINOV2_WEIGHTS" <<'PY'
import sys, torch
repo, weights = sys.argv[1], sys.argv[2]
# Same three calls as the loader in engine/rtv4/dinov2_teacher.py, same order.
model = torch.hub.load(repo, 'dinov2_vitb14', source='local')
state = torch.load(weights, map_location='cpu')
msg = model.load_state_dict(state, strict=False)
missing, unexpected = list(msg.missing_keys), list(msg.unexpected_keys)
print(f"embed_dim        = {model.embed_dim}")
print(f"state dict keys  = {len(state)}")
print(f"missing_keys     = {len(missing)}")
print(f"unexpected_keys  = {len(unexpected)}")
bad = False
if missing:
    print("FATAL: missing_keys is not empty. First 10:", missing[:10])
    print("  The weights file does not match the hub architecture. If its keys look")
    print("  like embeddings.* or encoder.layer.N. this is the Hugging Face `transformers`")
    print("  port; fetch the original facebookresearch dinov2_vitb14_pretrain.pth.")
    bad = True
if unexpected:
    print("WARNING: unexpected_keys present (first 10):", unexpected[:10])
if model.embed_dim != 768:
    print(f"FATAL: embed_dim {model.embed_dim} != 768, the wrong DINOv2 size. The config's")
    print("  HybridEncoder.distill_teacher_dim is 768 and the channel check in")
    print("  rtv4_criterion.py raises on a mismatch. model_name must be dinov2_vitb14,")
    print("  not the vitl14 default.")
    bad = True
sys.exit(1 if bad else 0)
PY
}

b2_fetch_teacher() {
  # ---- 1. the B2 config itself ----
  [ -f "$B2_CFG_SRC" ] || { echo "B2 config source missing: $B2_CFG_SRC"; return 1; }
  install -D "$B2_CFG_SRC" "$RTV4/$B2_CFG" || return 1
  echo "B2 config installed: $RTV4/$B2_CFG"
  echo "  sha256 $(sha "$RTV4/$B2_CFG")"

  mkdir -p "$TEACHERS"
  local pin url want_sha
  pin="$(pins_get repos.dinov2.pin)"
  url="$(pins_get checkpoints.dinov2_vitb14.url)"
  want_sha="$(pins_get checkpoints.dinov2_vitb14.sha256)"
  [ -n "$url" ] || { echo "no checkpoints.dinov2_vitb14.url in $PINS"; return 1; }
  echo "PINS: repo pin '$pin'"
  echo "PINS: weights url $url"
  echo "PINS: weights sha256 '$want_sha'"

  # ---- 2. clone facebookresearch/dinov2 (source='local' needs its hubconf.py) ----
  local repo_url; repo_url="$(pins_get repos.dinov2.url)"; : "${repo_url:=$DINOV2_URL_FALLBACK}"
  if [ ! -d "$DINOV2_REPO/.git" ]; then
    git clone "$repo_url" "$DINOV2_REPO" || return 1
  fi
  case "$pin" in
    ''|TODO_PIN_AT_FIRST_CLONE|TODO_AT_CLONE)
      if [ "$ALLOW_UNPINNED_TEACHER" != 1 ]; then
        echo "PINS.yaml repos.dinov2.pin is still '$pin'."
        echo "House rule: third-party trees are pinned before they train anything."
        echo "The default-branch HEAD in this clone is:"
        echo "    $(git -C "$DINOV2_REPO" rev-parse HEAD)"
        echo "Record that SHA in $PINS as repos.dinov2.pin, then re-run this stage."
        echo "To proceed unpinned regardless (a deliberate act, recorded in the log):"
        echo "    ALLOW_UNPINNED_TEACHER=1 bash scripts/pod/eyes_chain.sh --only B2_FETCH_TEACHER"
        return 1
      fi
      echo "WARNING: proceeding unpinned (ALLOW_UNPINNED_TEACHER=1). HEAD is:"
      echo "    $(git -C "$DINOV2_REPO" rev-parse HEAD)"
      mark "B2_TEACHER_REPO_UNPINNED_$(git -C "$DINOV2_REPO" rev-parse --short HEAD)"
      ;;
    *)
      git -C "$DINOV2_REPO" fetch --quiet origin || true
      git -C "$DINOV2_REPO" checkout --quiet "$pin" || return 1
      [ "$(git -C "$DINOV2_REPO" rev-parse HEAD)" = "$pin" ] || {
        echo "dinov2 HEAD is not the pin $pin"; return 1; }
      echo "dinov2 checked out at pin $pin"
      ;;
  esac

  # ---- 3. licence gate. The pre-2023-06 dinov2 tree was CC-BY-NC-4.0; the pin must
  # be at or after the Apache relicense (PINS.yaml repos.dinov2.license_notes).
  local lic="$DINOV2_REPO/LICENSE"
  [ -f "$lic" ] || { echo "no LICENSE in $DINOV2_REPO; refusing"; return 1; }
  if grep -qi 'Apache License' "$lic"; then
    echo "licence OK: $lic reads Apache"
    grep -m1 -i 'Version 2\.0' "$lic" || true
  else
    echo "FATAL: $DINOV2_REPO/LICENSE does not read 'Apache License'. First 5 lines:"
    head -5 "$lic"
    echo "The pre-2023-06 dinov2 release was CC-BY-NC-4.0 and is banned here."
    return 1
  fi

  # ---- 4. weights: the original state dict, never the Hugging Face transformers port ----
  if [ ! -f "$DINOV2_WEIGHTS" ]; then
    echo "fetching $url"
    curl -fL --retry 3 -o "$DINOV2_WEIGHTS.part" "$url" || {
      echo "download failed. Do not substitute the Hugging Face port facebook/dinov2-base:"
      echo "it loads zero tensors under strict=False (research/eyes-dinov2-teacher-arm.md section 5)."
      echo "Take the correct URL from $DINOV2_REPO/README.md if this one has moved,"
      echo "and correct checkpoints.dinov2_vitb14.url in $PINS."
      rm -f "$DINOV2_WEIGHTS.part"; return 1; }
    mv "$DINOV2_WEIGHTS.part" "$DINOV2_WEIGHTS"
  fi
  local got_sha; got_sha="$(sha "$DINOV2_WEIGHTS")"
  echo "dinov2_vitb14_pretrain.pth sha256 $got_sha"
  case "$want_sha" in
    ''|TODO_AT_FIRST_DOWNLOAD|recorded-at-first-download)
      echo "Action required: record this sha256 in $PINS as"
      echo "    checkpoints.dinov2_vitb14.sha256: $got_sha"
      echo "(PINS' sha256-at-first-download law; not a blocker for this run, which is"
      echo " the download that establishes it, but it must be committed.)"
      mark "B2_TEACHER_SHA256_${got_sha:0:12}"
      ;;
    "$got_sha") echo "sha256 matches PINS" ;;
    *)
      echo "FATAL: sha256 mismatch."
      echo "  PINS says $want_sha"
      echo "  file is  $got_sha"
      echo "Investigate before deleting the PINS line."
      return 1
      ;;
  esac

  # ---- 5. The assertion. A wrong key namespace means a silent random teacher. ----
  say "asserting the teacher loads with empty missing_keys (CPU, seconds)"
  if ! assert_teacher_loads_clean; then
    echo "Teacher load assertion failed; refusing to start a 13 h run against a"
    echo "possibly random teacher. See research/eyes-dinov2-teacher-arm.md section 5."
    echo "If torch.hub.load itself failed on a missing dependency, install that"
    echo "dependency into $RTV4/.venv and re-run this stage. Do not skip the assertion."
    return 1
  fi
  echo "teacher assets ready: $DINOV2_REPO + $DINOV2_WEIGHTS"
}

# ------------------------------------------------------------------ B2 gated
b2_dinov2t() {
  # B2_FETCH_TEACHER installs the config and both assets, so
  # reaching here without them means that stage was skipped past with --from/--only.
  # Skip cleanly rather than fail, because an unprepared arm is not a chain failure.
  if [ ! -f "$RTV4/$B2_CFG" ]; then
    say "B2: $RTV4/$B2_CFG absent; run the B2_FETCH_TEACHER stage first"
    mark "B2_DINOV2T_SKIPPED_NO_CONFIG"
    return 0
  fi
  if [ ! -d "$DINOV2_REPO" ] || [ ! -f "$DINOV2_WEIGHTS" ]; then
    say "B2: DINOv2 clone ($DINOV2_REPO) or weights ($DINOV2_WEIGHTS) missing;
    run the B2_FETCH_TEACHER stage first"
    mark "B2_DINOV2T_SKIPPED_NO_TEACHER"
    return 0
  fi

  local out="$OUTROOT/rtv4_s_person_v2_b2_dinov2t" log="$XLOGS/B2_DINOV2T.log"
  mkdir -p "$out"
  raise_ulimit
  say "B2 dry-run VRAM gate: launching, sampling nvidia-smi for ${VRAM_WINDOW}s"
  ( cd "$RTV4" && "$RTV4/.venv/bin/python" train.py -c "$B2_CFG" \
      --use-amp --seed=1 -t "$RTV4_INIT" --output-dir "$out" ) >"$log" 2>&1 &
  local pid=$!

  # Poll and keep the maximum; one sample at t=15min misses the peak.
  local peak=0 used elapsed=0
  while [ "$elapsed" -lt "$VRAM_WINDOW" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; local rc=$?
      mark "B2_DINOV2T_FAIL"
      die "B2 died during the ${VRAM_WINDOW}s gate window (rc $rc, peak ${peak} MiB). Read $log."
    fi
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
    case "$used" in ''|*[!0-9]*) used=0 ;; esac
    [ "$used" -gt "$peak" ] && peak="$used"
    sleep "$VRAM_POLL"
    elapsed=$((elapsed + VRAM_POLL))
  done

  echo "B2 VRAM peak over ${VRAM_WINDOW}s: ${peak} MiB (gate ${VRAM_GATE_MIB} MiB)"
  mark "B2_VRAM_PEAK_${peak}_MIB"
  if [ "$peak" -gt "$VRAM_GATE_MIB" ]; then
    kill "$pid" 2>/dev/null; sleep 10; kill -9 "$pid" 2>/dev/null
    mark "B2_DINOV2T_SKIPPED_VRAM_GATE"
    say "B2 skipped: ${peak} MiB > ${VRAM_GATE_MIB} MiB. This is the declared outcome,
    not a failure, and the chain ends clean. Levers before retrying, least damage first
    (research/eyes-dinov2-teacher-arm.md section 8): target_downsample_factor 16 to 32;
    autocast or .half() the frozen teacher; micro-batch the teacher call. Do not
    reduce total_batch_size, because that breaks comparability with the B1 seeds."
    return 0
  fi

  say "B2 gate passed (${peak} MiB). Training continues (~13 h); waiting."
  if wait "$pid"; then
    [ -f "$out/best_stg2.pth" ] || { echo "no best_stg2.pth in $out"; return 1; }
    cp "$out/best_stg2.pth" "$CKPT/b2_dinov2t_best_stg2.pth"
    echo "B2 best_stg2 sha256 $(sha "$CKPT/b2_dinov2t_best_stg2.pth")"
    # B2's checkpoint carries the two extra encoder.feature_projector.0.* tensors,
    # so it must be scored through the B2 config, not $RTV4_CFG (see section 7 of the
    # DINOv2 teacher-arm note).
    local log="$XLOGS/testonly_b2_dinov2t.log" ap
    raise_ulimit
    ( cd "$RTV4" && "$RTV4/.venv/bin/python" train.py -c "$B2_CFG" \
        --test-only -r "$CKPT/b2_dinov2t_best_stg2.pth" \
        --output-dir "$OUTROOT/testonly_b2_dinov2t" ) >"$log" 2>&1 || return 1
    ap="$(parse_ap "$log")"
    require_ap "$ap" "B2-DINOv2T" || return 1
    echo "B2-DINOv2T AP = $ap"
  else
    return 1
  fi
}

# ------------------------------------------------------------------------ main
usage() {
  cat <<EOF
usage: bash scripts/pod/eyes_chain.sh [--dry-run] [--from STAGE] [--only STAGE] [--list]
stages: ${STAGES[*]}
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --from) shift; FROM_STAGE="${1:-}" ;;
    --only) shift; ONLY_STAGE="${1:-}" ;;
    --list) usage; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

# --from: pre-mark everything before STAGE as _OK so the normal skip logic handles it.
if [ -n "$FROM_STAGE" ]; then
  found=0
  for s in "${STAGES[@]}"; do
    [ "$s" = "$FROM_STAGE" ] && { found=1; break; }
  done
  [ "$found" = 1 ] || die "--from: unknown stage '$FROM_STAGE' (see --list)"
  for s in "${STAGES[@]}"; do
    [ "$s" = "$FROM_STAGE" ] && break
    has_mark "${s}_OK" || mark "${s}_OK"
  done
fi

mark "CHAIN_START"
preflight

stage B0_TOOLS_ENV    "setup_pod.sh tools env (prereq of smartspaces)" -- b0_tools_env
stage B0_SETUP        "setup_pod.sh trainers ckpts"                 -- b0_setup
stage B0_SMARTSPACES  "setup_pod.sh smartspaces (~8.1 GB)"          -- b0_smartspaces
stage B0_STAGING      "install uploads from $STAGING"               -- b0_staging
stage I1_DUMP_RTV4    "I1 dump: RTv4-S 345c30a5, dev-val"           -- i1_rtv4
stage I1_DUMP_DEIM    "I1 dump: DEIM-S e98de401, dev-val"           -- i1_deim
stage I1_DUMP_RTV4_HFLIP "I1 TTA dump: RTv4-S hflip"                -- i1_rtv4_hflip
stage I1_DUMP_DEIM_HFLIP "I1 TTA dump: DEIM-S hflip"                -- i1_deim_hflip
stage I2_EXPORT       "I2 ONNX export 345c30a5 (--check, opset 17)" -- i2_export
stage B1_SEED1        "B1 RTv4-S seed 1 (~13 h)"                    -- b1_seed1
if has_mark "B1_SEED2_SCOPE_CUT"; then
  say "== SKIP B1_SEED2 (scope cut 2026-09-06: 2-seed path; seed 2 can be queued later)"
  has_mark "B1_SEED2_SKIPPED_SCOPE_CUT" || mark "B1_SEED2_SKIPPED_SCOPE_CUT"
else
  stage B1_SEED2        "B1 RTv4-S seed 2 (~13 h)"                    -- b1_seed2
fi
stage SOUP_UNIFORM3   "uniform soup and --test-only (3-seed, or 2-seed under the scope cut)" -- soup_uniform3
if has_mark "B1_SEED2_SCOPE_CUT"; then
  say "== SKIP SOUP_GREEDY (scope cut: greedy over 2 ingredients adds nothing beyond uniform-2 and singles)"
  has_mark "SOUP_GREEDY_SKIPPED_SCOPE_CUT" || mark "SOUP_GREEDY_SKIPPED_SCOPE_CUT"
else
  stage SOUP_GREEDY     "greedy-3 soup, scored per candidate"         -- soup_greedy
fi
stage B2_FETCH_TEACHER "B2 config + dinov2 clone/weights + load assertion" -- b2_fetch_teacher
stage B2_DINOV2T      "B2-DINOv2T, VRAM-gated dry run then train"   -- b2_dinov2t

mark "ALL_DONE"
say "chain complete. Artefacts:
  detections  $DUMPS/*.json (plus .sidecar.json): grade on the Mac with
              scripts/eval/detector_precision_at_thr.py
  onnx        ${B1_SEED0_CKPT%.pth}.onnx (plus .sha256): Mac ORT grader and CPU bench
  seeds       $CKPT/b1_best_stg2_seed{1,2}.pth
  soups       $SOUPS/
  status      $STATUS"
