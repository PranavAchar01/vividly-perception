#!/usr/bin/env bash
# v2 pod bootstrap: bring one 5090 Blackwell pod up to the state the eyes-lane
# training runs need, one idempotent stage at a time.
#
#   bash scripts/pod/setup_pod.sh [all|tools|env|trainers|ckpts|smartspaces]...
#
# Run on the pod from the repo root (clone the repo first; use a deploy key or a
# token entered through the pod provider's env panel, and never record keys in any
# transcript or commit them). Idempotent: every stage checks before it acts, so re-running
# after a disconnect is safe and cheap.
#
# What it builds (defaults; override WORKSPACE to relocate):
#   /workspace/third_party/DEIM        Model A trainer @ pinned SHA, patched, own venv
#   /workspace/third_party/RT-DETRv4   Model B trainer @ pinned SHA, patched, own venv
#   /workspace/ckpt/                   COCO-lineage checkpoints + MANIFEST.sha256
#   <repo>/data/train_ok/physicalai_smartspaces/   train scenes fetched + converted
#   /workspace/data/person_v2/         symlinks the trainers' configs point at
#
# Data laws enforced here: only SmartSpaces training scenes are fetched (validation
# scenes are exam-side and never reach a pod); MVOR, the fall sets and WU-SAHZU are
# clinical-adjacent and are never fetched to any cloud machine, and no stage below
# knows how to.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TP="$WORKSPACE/third_party"
CKPT="$WORKSPACE/ckpt"
PINS="$REPO_ROOT/third_party/PINS.yaml"

DEIM_URL="https://github.com/Intellindust-AI-Lab/DEIM"
DEIM_PIN="09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
RTV4_URL="https://github.com/RT-DETRs/RT-DETRv4"
RTV4_PIN="55fefaaed7efe2a5f72d0a18fd4e05965e35c292"
# Google Drive file ids from PINS.yaml (Drive is not content-addressed: first
# download records sha256 into the manifest, and later runs refuse a mismatch).
DEIM_S_GDRIVE="1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC"
RTV4_S_GDRIVE="1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
SMARTSPACES_SCENES=(Warehouse_000 Warehouse_001 Warehouse_002 Warehouse_003 Warehouse_004 Warehouse_005)

say() { printf '\n== %s\n' "$*"; }
die() { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

stage_tools() {
  say "tools: ffmpeg/git/curl + uv"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq ffmpeg git curl >/dev/null
  fi
  command -v ffmpeg >/dev/null || die "ffmpeg missing and apt unavailable"
  # pods ship stale uv (seen: 0.9.0 against the required >=0.11), so check the
  # version, not only presence; the astral installer lands in ~/.local/bin, which wins
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1 || \
     [ "$(printf '%s\n' 0.11.0 "$(uv --version | awk '{print $2}')" | sort -V | head -1)" != "0.11.0" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  uv --version
  command -v nvidia-smi >/dev/null || die "nvidia-smi missing; is this a GPU pod?"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

stage_env() {
  say "repo env: uv sync --extra cuda (torch cu128)"
  cd "$REPO_ROOT"
  uv sync --extra cuda --no-group mac --locked
  uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
}

clone_pin_patch() {  # name url pin patches_dir config_src config_dst
  local name="$1" url="$2" pin="$3" patches="$4" cfg_src="$5" cfg_dst="$6"
  local dst="$TP/$name"
  mkdir -p "$TP"
  if [ ! -d "$dst/.git" ]; then
    git clone --no-checkout "$url" "$dst"
  fi
  # non-fatal: checkout of an already-present pin needs no network, and GitHub's
  # anonymous-clone throttle intermittently kills fetch on fresh pods
  git -C "$dst" fetch --quiet origin || true
  git -C "$dst" checkout --quiet "$pin"
  [ "$(git -C "$dst" rev-parse HEAD)" = "$pin" ] || die "$name: HEAD is not the pin $pin"
  local p
  for p in "$REPO_ROOT"/third_party/patches/"$patches"/*.patch; do
    if git -C "$dst" apply --reverse --check "$p" 2>/dev/null; then
      echo "  already applied: $(basename "$p")"
    else
      git -C "$dst" apply "$p"
      echo "  applied: $(basename "$p")"
    fi
  done
  install -D "$REPO_ROOT/third_party/configs/$cfg_src" "$dst/$cfg_dst"
  echo "  config installed: $cfg_dst"
}

make_trainer_venv() {  # repo_dir
  local dst="$1"
  if [ ! -x "$dst/.venv/bin/python" ]; then
    uv venv --python 3.11 "$dst/.venv"
  fi
  uv pip install --python "$dst/.venv/bin/python" --index-url "$TORCH_INDEX" torch torchvision
  # their requirements are unpinned; torch and torchvision come from cu128 above
  grep -viE '^(torch|torchvision)([><=~ ]|$)' "$dst/requirements.txt" > "$dst/.reqs-notorch.txt"
  uv pip install --python "$dst/.venv/bin/python" -r "$dst/.reqs-notorch.txt" onnx onnxsim
  "$dst/.venv/bin/python" -m pip freeze > "$dst/env-freeze.txt" 2>/dev/null || \
    uv pip freeze --python "$dst/.venv/bin/python" > "$dst/env-freeze.txt"
  echo "  venv ready; freeze written to $dst/env-freeze.txt"
}

stage_trainers() {
  say "Model A: DEIM @ ${DEIM_PIN:0:7}"
  clone_pin_patch DEIM "$DEIM_URL" "$DEIM_PIN" deim \
    deim_s_person_v2.yml configs/deim_dfine/deim_s_person_v2.yml
  make_trainer_venv "$TP/DEIM"
  say "Model B: RT-DETRv4 @ ${RTV4_PIN:0:7} (DINOv3 teacher disabled by config)"
  clone_pin_patch RT-DETRv4 "$RTV4_URL" "$RTV4_PIN" rtdetrv4 \
    rtv4_s_person_v2_noteacher.yml configs/rtv4/rtv4_s_person_v2_noteacher.yml
  make_trainer_venv "$TP/RT-DETRv4"
}

fetch_ckpt() {  # gdrive_id out_name
  local id="$1" out="$CKPT/$2"
  mkdir -p "$CKPT"
  if [ ! -f "$out" ]; then
    uvx gdown "https://drive.google.com/uc?id=$id" -O "$out"
  fi
  local sha; sha="$(sha256sum "$out" | cut -d' ' -f1)"
  local line="$sha  $(basename "$out")"
  touch "$CKPT/MANIFEST.sha256"
  if grep -q "  $(basename "$out")\$" "$CKPT/MANIFEST.sha256"; then
    grep -q "^$line\$" "$CKPT/MANIFEST.sha256" || die "$out sha256 changed against the manifest; refusing (Drive is not content-addressed; investigate before deleting the manifest line)"
    echo "  verified: $line"
  else
    echo "$line" >> "$CKPT/MANIFEST.sha256"
    echo "  recorded: $line"
  fi
}

stage_ckpts() {
  say "checkpoints (COCO lineage only; obj365 artifacts are banned)"
  fetch_ckpt "$DEIM_S_GDRIVE" deim_dfine_s_coco.pth
  fetch_ckpt "$RTV4_S_GDRIVE" rtv4_s_coco.pth
}

stage_smartspaces() {
  say "SmartSpaces training scenes: fetch, convert, then write manifests"
  cd "$REPO_ROOT"
  local root="data/train_ok/physicalai_smartspaces"
  # include-only patterns per the recorded Hugging Face trap (--exclude is silently
  # ignored when --include is present; mid-path wildcards match nothing)
  local args=()
  local s
  for s in "${SMARTSPACES_SCENES[@]}"; do
    args+=("--include" "MTMC_Tracking_2025/train/$s/videos/*")
    args+=("--include" "MTMC_Tracking_2025/train/$s/ground_truth.json")
    args+=("--include" "MTMC_Tracking_2025/train/$s/calibration.json")
  done
  uvx --from 'huggingface_hub[cli]' hf download nvidia/PhysicalAI-SmartSpaces \
    --repo-type dataset --local-dir "$root" "${args[@]}"
  uv run python scripts/data/convert_smartspaces.py --config configs/convert/smartspaces_train_v1.yaml
  uv run python scripts/data/merge_coco.py --config configs/convert/person_v2_manifests.yaml
  mkdir -p "$WORKSPACE/data/person_v2"
  ln -sfn "$REPO_ROOT/$root/frames" "$WORKSPACE/data/person_v2/frames"
  ln -sfn "$REPO_ROOT/$root/annotations" "$WORKSPACE/data/person_v2/annotations"
  echo "  /workspace/data/person_v2 points at the repo bucket (symlinks)"
}

summary() {
  cat <<EOF

== Pod ready. Next commands (in order):

# 1. Model A fine-tune (DEIM-D-FINE-S; plain python on purpose, see the config header).
#    ulimit is load-bearing: the container's 1024-fd soft limit kills the first
#    epoch eval ('received 0 items of ancdata', because validation workers stack on
#    training workers; seen live 2026-08-31, seed 0 attempt 1). Raise it in the same
#    shell that launches training:
cd $TP/DEIM && ulimit -n 1048576 && .venv/bin/python train.py -c configs/deim_dfine/deim_s_person_v2.yml \\
    --use-amp --seed=0 -t $CKPT/deim_dfine_s_coco.pth

# 2. Model B fine-tune (RT-DETRv4-S, teacher off; same ulimit law):
cd $TP/RT-DETRv4 && ulimit -n 1048576 && .venv/bin/python train.py -c configs/rtv4/rtv4_s_person_v2_noteacher.yml \\
    --use-amp --seed=0 -t $CKPT/rtv4_s_coco.pth

# 3. Export (opset 17, batch-1 dummy, already patched):
#    .venv/bin/python tools/deployment/export_onnx.py --check -c <same config> -r <best_stg2.pth>

Watch for: ./NaN.pth (drop --use-amp and restart from last checkpoint if it appears);
the 'Load model.state_dict {missed/unmatched}' line on the first -t run must list
only class-head tensors.
EOF
}

STAGES=("${@:-all}")
run() { local s="$1"; case "$s" in
  tools) stage_tools ;;
  env) stage_env ;;
  trainers) stage_trainers ;;
  ckpts) stage_ckpts ;;
  smartspaces) stage_smartspaces ;;
  all) stage_tools; stage_env; stage_trainers; stage_ckpts; stage_smartspaces ;;
  *) die "unknown stage '$s' (tools|env|trainers|ckpts|smartspaces|all)" ;;
esac; }
for s in "${STAGES[@]}"; do run "$s"; done
summary
