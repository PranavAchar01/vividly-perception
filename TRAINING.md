# Training

This document is the reproduction record for the five model lanes: eyes (person detector),
joints (pose), fall and seizure (event verifiers), and re-identification. Each section
below describes what the lane trains on, the gates a candidate must clear, the exact
commands that reproduce it, and the artefacts those commands write.

The standing rules that constrain every lane are in `POLICY.md`. The licence status and
acquisition record of every corpus named here are in `DATASETS.md` and, word for word, in
`configs/datasets/*.yaml`. The model of record for each lane, its hashes and its headline
number are in `final_models/README.md` and the five lane pages beside it. The evidence
documents behind every number are in `research/`.

## Environments

The environments this document uses are declared in `pyproject.toml`, and the conflicting
ones are mutually exclusive by declaration, so combining them is a resolver error rather
than a runtime surprise.

    uv sync                                          # macOS harness: onnxruntime, rtmlib, no torch
    uv sync --extra cuda --no-group mac --locked     # Linux training pod: torch on the cu128 index
    uv sync --group linux-harness --no-default-groups        # Linux extraction: onnxruntime only
    UV_PROJECT_ENVIRONMENT=.venv-curate uv sync --extra curate --no-default-groups

The two detector trainers are third-party repositories and get their own virtual
environments under `/workspace/third_party/<repo>/.venv`, built by
`scripts/pod/setup_pod.sh`; they are never mixed with the repository environment.
On macOS, ONNX inference runs on the CPU execution provider only.

`uv sync --extra cuda` on macOS installs nothing and exits zero. It is not a working
training environment, and a script that needs the training stack must be run on the pod.

The rented training GPUs are Blackwell parts at compute capability sm_120, which no torch
below 2.7 can drive, so the training extra resolves torch from the cu128 index and an
older framework stack is not an option on this hardware.

## Measurement discipline

These rules apply to every lane and are the reason the numbers in `final_models/` are
comparable to each other.

1. A model is compared only on an exam it never trained on. A dataset that enters a
   training manifest loses its exam role permanently.
2. Variant and epoch selection happens on the train-side validation split, never on an
   exam. Selection that touches an exam invalidates the exam.
3. Each candidate version gets one pre-declared look at each exam. Every look is written
   into the ledger of the protocol document in `research/` before the next candidate runs.
4. An exam is never edited after it is frozen. A change to the exam surface makes a new
   exam, and old numbers on the old surface are historical context, not comparisons.
5. A noise floor is measured before any comparison. The first adopted recipe in a lane
   trains at least three seeds, and a difference smaller than the measured seed spread is
   not a result.
6. Every accuracy number is quoted with the checkpoint or ONNX sha256 that produced it.
   One value can identify two different checkpoints, so a number without its hash is not
   quotable.
7. A steering surface is labelled as such. Synthetic development validation splits and
   the COCO smoke ladders steer choices; they are never quoted as deployment evidence.
8. A configuration that is declared as a fixed number of changes from a previous
   configuration states those changes in its own header before the run, so the run can be
   read afterwards as the single-variable experiment it claimed to be.
9. Every run that costs rented compute passes a dry-run gate first: memory and throughput
   are measured over a fixed window and compared against a declared ceiling, and the
   projected cost is written down before the money is spent.

Two metric names recur below. AP is box or keypoint detection quality against human
labels on a zero to one scale. AUC is the probability that a model ranks a random true
event window above a random quiet one, where 0.5 is a coin flip.

### Pose capacity ablation and the onnxruntime thread pool

A worked example of rules 8 and 9 above, and the reason the pose lane carries no
larger-backbone arm.

A pose-capacity ablation was declared on 2026-08-28 with one variable per point. The first
point was rtmpose-m at 256x192, the production setting, whose keypoint quality on the
WU-SAHZU extraction is recorded by `research/pose-noise-profile.md`: on the
`wu_sahzu_tracks_v2pose` tree, pose graph `6ba96285`, a median wrist confidence of 0.312
and a median ankle confidence of 0.310, with 51.9 percent of wrist and 52.6 percent of
ankle observations at or above the 0.3 keypoint gate. The second point was rtmpose-l at
256x192, which isolates model capacity. The third was rtmpose-l at 384x288, which adds
input resolution on top of that.

The two rtmpose-l runs were abandoned on 2026-08-29 on throughput observed while they ran,
12 seconds per frame and 16.5 seconds per frame on the 256-core pod, about 25 days for the
36 sequences, because that path paid for detection, tracking, rules and rendering per frame
rather than for pose alone. The cost did not stay inside those runs: onnxruntime sized its
intra-op thread pool to the full core count, 387 threads per process, regardless of taskset
and `OMP_NUM_THREADS`, and the concurrent trainings on the same host slowed from 1.07 to
6.5 seconds per step. Those throughput figures are run observations written down at the
time, not instrument output, and no artefact in this repository carries them. The question
was answered instead by reading the existing keypoint extraction.

Any relaunch of this ablation caps the onnxruntime thread pool through
`intra_op_num_threads` before the run starts, and strips the render and rules stages so that
the run pays for pose only.

## Eyes: person detector

The lane fine-tunes a small real-time detection transformer on the person class only, from
a COCO-pretrained initialisation, and grades it against a competing architecture trained
on the same data with the same schedule.

**Data.** `person_v2`, built from the PhysicalAI SmartSpaces training scenes: 55,200
training images with 828,230 boxes, and a development validation split of 1,800 images
with 33,755 boxes formed by holding out the scene `Warehouse_005`. The development
validation split is synthetic and steers selection only. Fetch, conversion and manifest
merge are stages of `scripts/pod/setup_pod.sh`, which calls
`scripts/data/convert_smartspaces.py` with `configs/convert/smartspaces_train_v1.yaml` and
`scripts/data/merge_coco.py` with `configs/convert/person_v2_manifests.yaml`. Only training
scenes are fetched to a pod; validation scenes are exam-side and never leave the local
machine.

**Pins and patches.** `third_party/PINS.yaml` pins both trainer repositories by commit,
records their licences, and records the sha256 of every checkpoint that is not
content-addressed at its source, refusing a later download whose digest differs. The pins
are `09d35d53d39ee3145a1e61e3a989b28b9468d1dd` for DEIM and
`55fefaaed7efe2a5f72d0a18fd4e05965e35c292` for RT-DETRv4, both Apache-2.0 and neither
carrying a tag. `setup_pod.sh` clones each at its pin, then applies the patch series under
`third_party/patches/deim/` (three patches: transform hooks for the installed torchvision,
batch-1 opset 17 export, and a solver load fix) and `third_party/patches/rtdetrv4/`
(five patches: batch-1 opset 17 export, the same solver load fix, profiler statistics on
the CPU, an evaluation buffer fix, and the legacy export path), then installs the training
configuration from `third_party/configs/` into the trainer tree. The teacher-distillation
arm additionally pins the feature backbone repository at
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` and its weights at sha256
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`; the fetch stage
asserts the weight load leaves no missing keys, because a silently empty load would train
a random teacher for the arm's full 16.8 hours, the duration that arm ran to completion on
`final_models/eyes/README.md`.

**Configurations.** `third_party/configs/rtv4_s_person_v2_noteacher.yml` is the recipe of
record. `third_party/configs/deim_s_person_v2.yml` is the competing architecture, and
`third_party/configs/deim_s_person_v2_zeroshot_eval.yml` scores its initialisation before
any fine-tuning. `third_party/configs/rtv4_s_person_v2_dinov2t.yml` is the
teacher-distillation arm and differs from the recipe of record in exactly three
distillation keys plus its own output directory.

**The chain.** `scripts/pod/eyes_chain.sh` runs the whole queue detached and is written to
survive a disconnection: each stage appends a timestamped mark to a status file, a re-run
skips every stage already marked complete, and the first failure aborts the chain. Its
stages are

    B0_TOOLS_ENV B0_SETUP B0_SMARTSPACES B0_STAGING
    I1_DUMP_RTV4 I1_DUMP_DEIM I1_DUMP_RTV4_HFLIP I1_DUMP_DEIM_HFLIP
    I2_EXPORT
    B1_SEED1 B1_SEED2
    SOUP_UNIFORM3 SOUP_GREEDY
    B2_FETCH_TEACHER B2_DINOV2T

Two mechanical facts are load-bearing and are recorded in the chain: the container's
default file-descriptor limit is exhausted during the first epoch's validation pass, so
`ulimit -n` is raised in the same shell that launches training; and `--test-only` writes an
accumulator file rather than detections, so detection dumps come from
`scripts/detector/dump_detections_v2.py` running inside the trainer's own environment.

**Gates.** The precision instrument grades both detectors on the development validation
split at score 0.3, 0.4 and 0.5 at intersection over union 0.5, reporting precision,
recall and false positives per image overall and per camera; 0.4 is the operating point
the demo draws at. The export instrument checks that the exported ONNX reproduces the
checkpoint's own detections and measures CPU milliseconds per frame. Before any track-tree
extraction may use the detector through the pipeline factory,
`scripts/eval/backend_parity_rtdetr.py` must pass: the `rtdetr_onnx` backend built through
the factory and fed a frame the way the replay source hands frames over must agree box for
box and score for score with the grader's own preprocessing on the parity sample. A
backend that reaches for a different resize moves scores by roughly 0.15 and fails the
gate, which is the intended behaviour; the fix is the backend, never a looser bar.

**Result of record.** The recipe of record at seed 0, checkpoint
`345c30a539c2b6bb665302825f418322284b98f6cbcf55d58d84ffeb75b16e7e`, reached development
validation AP 0.8356 at epoch 47 over 48 epochs at batch 32 with mixed precision, about
13.0 hours on one rented GPU. The pre-registered bar was 0.8353, which is the competing
architecture's best seed `e98de401` at 0.8341 plus its measured three-seed noise floor of
0.0012, that being the full spread from 0.8329 to 0.8341. Seed 1 of the same recipe
reached 0.8348 (checkpoint `17362dca`), giving a two-seed spread of 0.00075. The
displacement bar is 0.8368: the 0.8356 of record plus the larger of the 0.0012 noise floor
and that 0.00075 seed spread. A uniform two-seed weight soup scored 0.830 (`0d80cb4c`) and
was not promoted. The teacher-distillation arm reached 0.8345 (checkpoint `9ecb4627`)
against that 0.8368 bar and was not promoted, so the model of record is final for this
data era. The GPU memory gate for the teacher arm was 30,500 MiB, polled over a
fifteen-minute window and judged on the maximum rather than a single sample; the measured
peak was 23,758 MiB. Full record:
`research/eyes-rtdetrv4-training-record.md`, `research/eyes-seed-spread-and-soup.md`,
`research/eyes-noise-floor.md` and `research/eyes-dinov2-teacher-arm.md`.

### Commands

    # On the pod, from the repository root. Stages are idempotent.
    bash scripts/pod/setup_pod.sh tools env
    bash scripts/pod/setup_pod.sh trainers ckpts
    bash scripts/pod/setup_pod.sh smartspaces

    # Copy b1_best_stg2_345c30a5.pth, best_stg2_seed1_e98de401.pth and
    # person_v2_devval.json into /workspace/staging, then run the whole queue detached.
    nohup bash scripts/pod/eyes_chain.sh > /workspace/xlogs/eyes_chain.out 2>&1 &
    tail -f /workspace/xlogs/eyes_chain.status

    # Individual stages, and resuming from one.
    bash scripts/pod/eyes_chain.sh --list
    bash scripts/pod/eyes_chain.sh --only I2_EXPORT
    bash scripts/pod/eyes_chain.sh --from B1_SEED1

    # The two training runs the chain wraps, if run by hand.
    cd /workspace/third_party/RT-DETRv4 && ulimit -n 1048576 && \
      .venv/bin/python train.py -c configs/rtv4/rtv4_s_person_v2_noteacher.yml \
        --use-amp --seed=0 -t /workspace/ckpt/rtv4_s_coco.pth
    cd /workspace/third_party/DEIM && ulimit -n 1048576 && \
      .venv/bin/python train.py -c configs/deim_dfine/deim_s_person_v2.yml \
        --use-amp --seed=0 -t /workspace/ckpt/deim_dfine_s_coco.pth

    # Export, opset 17, batch 1. Writes the ONNX beside the checkpoint.
    cd /workspace/third_party/RT-DETRv4 && \
      .venv/bin/python tools/deployment/export_onnx.py --check \
        -c configs/rtv4/rtv4_s_person_v2_noteacher.yml \
        -r /workspace/ckpt/b1_best_stg2_345c30a5.pth

    # Weight soup over seed checkpoints, and the hflip fusion row.
    /workspace/third_party/RT-DETRv4/.venv/bin/python scripts/detector/soup_v2.py uniform \
        --family rtv4_s \
        --ckpt /workspace/ckpt/b1_best_stg2_345c30a5.pth \
        --ckpt /workspace/ckpt/b1_best_stg2_seed1.pth \
        --out  /workspace/ckpt/soups/soup_uniform2.pth
    uv run python scripts/detector/fuse_tta_dumps.py \
        --gt  data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json \
        --det identity=outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_identity.json \
        --det hflip=outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_hflip.json \
        --out outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_tta_wbf.json

    # Grading, on the local machine, CPU only.
    uv run python scripts/eval/onnx_parity_v2.py \
        --onnx   outputs/eyes_b1/b1_best_stg2_345c30a5.onnx \
        --gt     data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json \
        --images outputs/eyes_b1/parity_sample \
        --det    outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_identity.json \
        --out-json outputs/eyes_b1/i2/parity_345c30a5.json
    uv run python scripts/eval/detector_precision_at_thr.py \
        --gt  data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json \
        --det outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_identity.json \
        --out-md   outputs/eyes_b1/i1/precision_345c30a5.md \
        --out-json outputs/eyes_b1/i1/precision_345c30a5.json
    uv run python scripts/eval/eval_detector_v2_onnx.py \
        --onnx   outputs/eyes_b1/b1_best_stg2_345c30a5.onnx \
        --gt     data/eval_only/mvor/annotations/mvor_person_v2_coco.json \
        --images data/eval_only/mvor/camma_mvor_dataset \
        --out-det  outputs/eyes_b1/i2/dets_mvor_345c30a5.json \
        --out-json outputs/eyes_b1/i2/mvor_345c30a5.json \
        --cluster-depth 2 --bootstrap 200 --label mvor-internal
    uv run python scripts/eval/backend_parity_rtdetr.py

The clinical-adjacent exam set is converted locally by `scripts/data/convert_mvor.py` with
`configs/convert/mvor_person_v2.yaml`, is graded on the local machine only, is never
uploaded to a rented machine, and its licence is non-commercial, so its numbers are
internal.

### Artefacts

- `/workspace/outputs/rtv4_s_person_v2_seed<seed>/best_stg2.pth`, copied to
  `/workspace/ckpt/b1_best_stg2_seed<seed>.pth` with its sha256 printed.
- `b1_best_stg2_345c30a5.onnx` and `b1_best_stg2_345c30a5.onnx.sha256` beside the
  checkpoint.
- Detection dumps and their checkpoint sidecars under `/workspace/xlogs/dumps/`.
- Per-stage logs and the append-only mark file under `/workspace/xlogs/`.
- Per-epoch curves for the four detector runs, distilled by
  `scripts/analysis/extract_training_curves.py` and committed as
  `final_models/eyes/curves/detector_b1_rtv4s_seed0.csv`,
  `final_models/eyes/curves/detector_deim_seed0.csv`,
  `final_models/eyes/curves/detector_deim_seed1.csv` and
  `final_models/eyes/curves/detector_deim_seed2.csv`.
- The shipped lane directory: `final_models/eyes/model.onnx`, `model.onnx.sha256`,
  `model.json` and the figures rendered by `scripts/analysis/plot_final_curves.py`.

## Joints: pose

The lane fine-tunes a top-down pose estimator on COCO person keypoints from a released
multi-dataset initialisation, then exports the graph the rest of the stack consumes.

**Configuration.** `third_party/configs/rtmpose_m_person_v2_ft.py` is the recipe. It is
installed into the vendored pose repository at pin `759b39c1`, under
`configs/body_2d_keypoint/rtmpose/coco/`, so its base configuration resolves; its header
carries the provenance card: every delta from the published recipe is listed and everything
not listed is inherited unchanged. The deltas are the schedule (420 epochs to 60, the
second-stage window 30 to 5, cosine start moved by the recipe's own ratio, validation every
2 epochs instead of every 10), the base learning rate (4e-3 to 1e-4, which is a fine-tuning
reduction composed with linear batch scaling from the recipe's 1024 reference batch to the
256 this runs at), warmup (1000 iterations to 500), and the initialisation, which is the
released body7 checkpoint `e48f03d0`. Training data is COCO 2017 person keypoints, 118,287
training images and 262,465 keypoint annotations. Validation uses ground-truth boxes.

**Seeds and the floor.** Three runs differ only in `randomness.seed` and, for two of them,
in dataloader worker count, which affects throughput and nothing else. Best keypoint AP on
COCO val2017 with ground-truth boxes was 0.774115 at epoch 38 (checkpoint `a0a035fe`),
0.774000 at epoch 30 (`8d990435`) and 0.774057 at epoch 32 (`5bcca5bd`), a floor of
0.77406 with a spread of 0.000115. The measured baseline for the initialisation on the
identical protocol is 0.764499, so every seed clears it by about 0.0095. The bar for any
future rival recipe is 0.7742. Record: `research/pose-seed-floor-and-champion.md` and
`research/pose-p0-training-record.md`.

**Export.** `scripts/pose/export_rtmpose_onnx.py` takes the backbone, optional neck and
head, and emits the two classification vectors, leaving the decode outside the graph, which
is the contract the runtime pose wrapper expects. The export of record is 54.4 MB at opset
17 with outputs shaped (1, 17, 384) and (1, 17, 512), sha256
`6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527`. The script refuses to
declare success unless the maximum absolute difference between torch and onnxruntime is
below 1e-4; the export of record measured 2.876e-06 and 5.230e-06.

**Two different measurements.** The headline 0.774115 is a trainer-side number: the pose
framework's own evaluation loop, ground-truth boxes, flip test on, torch weights on GPU.
The deployed-path number is a different quantity: the exported ONNX driven through the
runtime pose wrapper on CPU onnxruntime, with boxes from a detector and no flip test. On
the frozen 500-image COCO subset (`configs/eval/subset_coco_val_person500.json`,
fingerprint `2db41bbad264b7dc`, written by `scripts/data/make_eval_subset.py`) with a fixed
detector supplying identical boxes in every
row, the initialisation scored keypoint AP 0.7285, the model of record without flip 0.7196
and the model of record with flip 0.7262. **These two numbers are not a before-and-after
of the same thing and must never be subtracted from each other.** The comparisons the
ladder supports are the ones inside the table: same subset, same detector, same decode, one
variable at a time.

**Colour order.** The runtime pose wrapper does not convert colour, and the pose model was
trained with a red-green-blue conversion in its preprocessor, so the deployed path has been
feeding channel-swapped crops. Two diagnostic rows measure the cost with a runner-side
`pose.color_order: rgb` key as the only difference: the model of record gains 0.0277
keypoint AP, interval 0.0215 to 0.0342, and the initialisation gains 0.0108. With the
colour order corrected the deployed path reproduces the trainer-side ordering, the model of
record beating its initialisation by 0.0081, interval 0.0023 to 0.0137. The free conversion
is worth about four times what flip test-time augmentation is worth, and flip costs 1.63
times on the pose stage, 20.21 to 32.86 milliseconds per crop. **The colour-order fix
reaches the live path only through a full track-tree re-extraction, never as a backend
hotfix**, because every event verifier was trained on features extracted in the current
colour order; changing the extraction underneath a trained verifier is a different model,
not a free improvement. The key is default off and is set only in the extraction
configurations. Record:
`research/pose-deployed-path-cpu-proof.md`.

### Commands

    # On the pod. $POSE_REPO is the vendored pose repository at pin 759b39c1. Install the
    # recipe into it so its base configuration resolves, then train from inside it.
    install -D third_party/configs/rtmpose_m_person_v2_ft.py \
        "$POSE_REPO/configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_person_v2_ft.py"
    # The recipe sets no randomness key, so the seed of the run of record is passed
    # explicitly; without it the launch is not the run this lane reproduces.
    cd "$POSE_REPO" && .venv/bin/python tools/train.py \
        configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_person_v2_ft.py \
        --cfg-options randomness.seed=0

    # Export the selected checkpoint and check parity against the torch forward. Still
    # inside $POSE_REPO, so the recipe resolves its base configuration; $REPO is this
    # repository's checkout.
    .venv/bin/python "$REPO/scripts/pose/export_rtmpose_onnx.py" \
        --config configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_person_v2_ft.py \
        --ckpt   /workspace/outputs/p0/best_coco_AP_epoch_38.pth \
        --out    pose_v2_champion_p0e38.onnx --device cpu

    # Uniform weight soup over the seed checkpoints. The script imports torch at module
    # scope, so it runs under a training environment on the pod, not the macOS harness.
    .venv/bin/python scripts/pose/soup_checkpoints.py \
        --ckpt outputs/pod_backup/p0_best_e38_a0a035fe.pth \
        --ckpt outputs/pod_backup/p1_best_e30_8d990435.pth \
        --ckpt outputs/pod_backup/p2_best_e32_5bcca5bd.pth \
        --out  outputs/pose_soup_v2/pose_soup_v2_3seed.pth

    # Deployed-path ladder with a per-image bootstrap interval. The first row is a
    # reproduction of a recorded number and its gate stops the ladder if it misses.
    uv run python scripts/pose/eval_pose_ladder.py \
        --config configs/eval/coco500_pose_body7_e48f03d0.yaml \
        --config configs/eval/coco500_pose_v2_6ba96285_noflip.yaml \
        --config configs/eval/coco500_pose_v2_6ba96285_flip.yaml \
        --out-md outputs/pose/coco500_v2pose.table.md

    # The two colour-order diagnostic rows.
    uv run python scripts/pose/eval_pose_ladder.py \
        --config configs/eval/coco500_pose_body7_e48f03d0_rgb.yaml \
        --config configs/eval/coco500_pose_v2_6ba96285_noflip_rgb.yaml \
        --out-md outputs/pose/coco500_v2pose_rgb.table.md

    # Per-joint confidence and jitter over track trees that carry keypoints.
    uv run python scripts/analysis/pose_noise_profile.py

### Artefacts

- `best_coco_AP_epoch_<n>.pth` per seed, with its sha256 recorded.
- `pose_v2_champion_p0e38.onnx` with the parity line and the ONNX sha256 printed by the
  exporter.
- The ladder table written by `--out-md`, carrying the pose ONNX sha256, the boxes source,
  the subset fingerprint, the host and the onnxruntime version on every row.
- Per-epoch curves for the three pose runs, committed as
  `final_models/joints/curves/pose_rtmpose_m_p0_seed0.csv`,
  `final_models/joints/curves/pose_rtmpose_m_p1_seed1.csv` and
  `final_models/joints/curves/pose_rtmpose_m_p2_seed2.csv`. Validation ran every second
  epoch, so the validation column is empty on odd epochs.
- The shipped lane directory: `final_models/joints/model.onnx`, `model.onnx.sha256`,
  `model.json` and the figures.

## Fall and seizure: event verifiers

Both event lanes share one machinery. A dataset is replayed through the same detector,
pose and tracker stack the live pipeline runs; the per-track observation stream is written
to disk as a track tree; windows are cut from that tree and a small temporal
convolutional network is trained on them; the exported graph is scored on held-out
datasets. The verifier therefore trains on exactly what the rule layer hands it at
inference, never on ground-truth boxes the live system does not have.

Both lanes produce prototype-class weights: interim, internal, never quoted as clinical
detection performance, never deployed commercially without counsel sign-off. The rules
that permit them to exist at all, and the boundary between augmentation and synthesis, are
in `POLICY.md`.

**Data.** The fall training corpora are `le2i_fdd`, `gmdcsa24` and `caucafall`. The fall
exams are `ur_fall` and `multicam_fall`, which never train. `multicam_fall` is scored on
two surfaces: the frozen reference-camera set and the all-camera expansion of 192 views,
184 fall videos carrying 200 fall events plus eight views of the crouch scenario
`chute24`, addressed by `index_allcams.json`. The seizure training corpus is
`wu_sahzu_emu_trainsplit`, patients pat01, pat02, pat05 and pat13, which is what the
seizure model of record trains on; the rotation recipes widen it to the eight-patient
training pool, bounded by the exam set `scripts/guards/check_train_data_paths.py`
enforces. The exam side is `wu_sahzu_emu`, which never trains. A matched-context negative
diet harvested from `ajile12` sits beside them, which is pose and interval data with no
imagery. Acquisition goes through
`scripts/data/download_dataset.py`, which refuses a legally held entry, refuses an entry
with an open counsel question, and prints the human steps for an entry that requires them;
the exam-side fall set has its own fetch script. The negative diet is harvested by
`scripts/data/harvest_ajile12.py` and assembled by `scripts/data/assemble_ajile12_diet.py`
with `configs/events/ajile12_diet_v1.yaml`, which takes labelled-normal spans only, caps
each session at 180 seconds and the whole diet at 7200 seconds, and is seeded.

**Track-tree extraction.** The current generation is the `_v2c` set, seven configurations
under `configs/events/`:

    extract_le2i_v2c.yaml               -> outputs/events/le2i_tracks_v2c
    extract_gmdcsa24_v2c.yaml           -> outputs/events/gmdcsa24_tracks_v2c
    extract_caucafall_v2c.yaml          -> outputs/events/caucafall_tracks_v2c
    extract_ur_fall_v2c.yaml            -> outputs/events/ur_fall_tracks_v2c
    extract_multicam_v2c.yaml           -> outputs/events/multicam_tracks_v2c
    extract_multicam_allcams_v2c.yaml   -> outputs/events/multicam_allcams_tracks_v2c
    extract_wu_sahzu_v2c.yaml           -> outputs/events/wu_sahzu_tracks_v2c

Every one of them declares two changes from the previous generation and nothing else: the
detector becomes the eyes model of record through the `rtdetr_onnx` backend at 640 by 640,
and the pose stage gains `color_order: rgb`. The tracker, the track state, the rule
configuration, keypoint emission and the dataset scope are unchanged. The pipeline
operating point is stated inline in each configuration: detector score threshold 0.3,
byte-track activation 0.4, high-confidence detection threshold 0.4, lost-track buffer 30
frames, minimum intersection over union 0.1, keypoint confidence floor 0.3, baseline window
300 seconds with a minimum of 30 samples, reap after 30 seconds, and rules from
`configs/pipeline/rules.yaml`. Keypoint emission is on, so one extraction serves the fall
lane, which ignores keypoints, and the seizure lane, which needs them.

Extraction is resumable per video: a video whose sequence file already exists is skipped.
The fall corpora and the fall exams are extracted locally; the clinical seizure corpus may
be extracted on a rented machine, and its raw video is deleted afterwards.

Every earlier track-tree generation is historical: the `_v1` trees the fall model of record
trained on, the `wu_sahzu_tracks_v2` tree the seizure model of record trained on, the
`_v2pose` trees that carry the recorded exam bars and the reslices onto them, and the
`_v2kp` trees the leave-one-patient-out seizure configurations name. Many recorded training
and evaluation configurations under `configs/events/` still point at them, but no extraction
configuration in this repository rebuilds any of them, so those runs cannot be reproduced
from here. The chain that does reproduce end to end is the `_v2c` extraction above, the
`_v2c` training recipes and the `_v2c` evaluation configurations beside them.

**Window geometry.** Fall windows are 8.0 seconds at stride 1.5 with 64 steps, labelled by
containment with a trailing margin of 0.5 seconds and a minimum presence of 0.35. Seizure
windows are 12.0 seconds at stride 3.0 with 64 steps, labelled by overlap with a minimum
overlap of 4.0 seconds, minimum presence 0.35, and scene channels on. The window geometry
of an evaluation configuration must equal the geometry of the trainer that produced the
graph, because the graph's input shape pins the step count and the time scale pins the
window length.

**Fall holdout, seeds and gates.** Checkpoint selection and every train-side comparison use
a pinned holdout of five recording groups: `Home_02`, `S4_ADL`, `S4_Fall`, `S9` and `S10`.
It is pinned because a seed-drawn split had put 126 of 190 positives into validation, which
confounded the comparison it was used for. Each arm runs at least the seeds 20260826,
20260901 and 20260905; the F0 control and the F2 arm were extended to five with 20260910
and 20260915, which is where the five-seed spread below comes from. Under containment
labelling the corpus yields 881 windows, 190 positive and 691 negative, of which the
pinned holdout is 18 positive and 151 negative. The control arm's
measured spread is 0.00184 over three seeds and 0.00442 over five, and the five-seed figure
is the promotion margin because it was fixed before the arms that it judged. Promotion
requires a holdout-mean improvement larger than that margin; only then does the median
validation seed spend its one look per exam. A promoted candidate must also pass the
specificity gate on the same look, measured by `scripts/events/metrics_deep.py`: at
threshold 0.8, the lie-transition false-positive count on the exam set must not exceed the
control's 15 of 46 windows across 9 of 16 videos, and the crouch-scenario in-alarm
fraction must not exceed the control's 0 of 74 on the reference camera and 11 of 654, that
is 0.0168, across all cameras. Three arms in the recorded push cleared or nearly cleared
the accuracy bar and were dropped on this gate or on the margin. The bar to beat on the
all-camera surface is window area under the curve 0.78474, held by `b66c3a0a`. The lane
headline for the model of record `67954c04` on that surface is 0.79276, recorded on
`final_models/fall/README.md`; the same weights re-sliced onto the `_v2pose` trees, which
gives both lineages one shared negative pool, scored 0.79283. Record:
`research/fall-push-v3-protocol.md`, `research/fall-noise-floor.md`,
`research/fall-geometric-augmentation-arm.md`.

**Seizure holdout, exams and bars.** The patient pool rotates over eight patients in four
pre-declared leave-two-out folds, three seeds per fold. The recipe that the refit repeats
holds out patients `pat02` and `pat09` for checkpoint selection only and runs at seed
20260826. Two
exams are permanent and never train: the focal exam is patients `pat06`, `pat08`, `pat10`
and `pat12`, with a bar of window area under the curve 0.87422 held by `b9e04d03`; the
convulsive exam is patients `pat03` and `pat07`, with a bar of 0.72916 held by the model of
record `07c399be`. The budget is one promoted candidate and one look at each exam,
recorded in the protocol ledger. Selection never touches either exam. Record:
`research/seizure-push-1-protocol.md`, `research/seizure-push-1-verdict.md`,
`research/seizure-push-2-protocol.md`, `research/seizure-push-3-protocol.md`,
`research/seizure-v2pose-comparison.md`.

**Recipes.** `configs/events/train_fall_verifier_v3.yaml` and
`configs/events/train_convulsion_v2_real.yaml` are the recipes that produced the two
current holders; the track trees they were trained on are the earlier generation and are
recorded on `final_models/fall/README.md` and `final_models/seizure/README.md`, with the
detector hash, the pose hash, the tracker settings and the declared deltas of the
extraction that built them. `configs/events/train_fall_F0_s20260826.yaml` and its two seed
siblings are the pinned fall control recipe, and `configs/events/rot_train_conv_finalB.yaml`
is the seizure recipe of the current focal-exam bar holder. The refit configurations that
re-point those two recipes at the `_v2c` trees carry the same two file names with a `_v2c`
marker added: `configs/events/train_fall_F0_v2c_s20260826.yaml` and its two seed siblings
for the fall control, and `configs/events/rot_train_conv_finalB_v2c.yaml` for the seizure
recipe. Their pre-declared looks are the `eval_fall_*_v2c.yaml` and `rot_eval_E*_*_v2c.yaml`
files beside them, one per exam and head. Both refits were measured on 2026-09-07 and neither
was promoted; the records are on `final_models/fall/README.md` and
`final_models/seizure/README.md`, and the detector probe that explains both movements is
`research/eyes-coupling-probe.md`.

**Study instruments.** Four further tools under `scripts/events/` read run outputs and
report; none of them trains or promotes anything.
`scripts/events/wu_sahzu_breakout.py` splits a seizure evaluation report by seizure type,
because scoring every video identically hides which class a verifier caught.
`scripts/events/decision_layer_study.py` replays an evaluation's full-precision window
stream through candidate alerting rules and reports catches per class, detection latency
and distinct false-alarm episodes per negative recording, so the decision rule can be
chosen separately from the model.
`scripts/events/ensemble_study.py` time-aligns two verifiers' window streams per video,
which is the only alignment available when the two were evaluated through different
extractions, and measures the combination families.
`scripts/events/make_synth_convulsion.py` materialises motion-level synthetic convulsion
sequences in the extractor's own layout, deterministic in its seed, under the recorded
carve-out that keeps synthesis at motion level and bars generative patient imagery.

### Commands

    # 1. Acquire. Refuses a held or counsel-blocked entry; prints human steps where needed.
    uv run python scripts/data/download_dataset.py --name le2i_fdd --dry-run
    uv run python scripts/data/download_dataset.py --name gmdcsa24 --dry-run
    uv run python scripts/data/download_dataset.py --name caucafall --dry-run
    uv run python scripts/data/download_dataset.py --name multicam_fall --dry-run
    bash scripts/data/fetch_ur_fall.sh data/eval_only/ur_fall
    # The harvest reads remote HDF5, so it needs h5py and remfile in the interpreter that
    # runs it. Neither is a declared runtime dependency of any environment above; install
    # both into the environment used for this step.
    uv run python scripts/data/harvest_ajile12.py --config configs/datasets/ajile12.yaml
    uv run python scripts/data/assemble_ajile12_diet.py \
        --config configs/events/ajile12_diet_v1.yaml

    # 2. Extract track trees. Resumable; --dry-run lists, --only and --max-frames smoke.
    uv run python scripts/events/extract_track_sequences.py \
        --config configs/events/extract_le2i_v2c.yaml
    uv run python scripts/events/extract_track_sequences.py \
        --config configs/events/extract_gmdcsa24_v2c.yaml
    uv run python scripts/events/extract_track_sequences.py \
        --config configs/events/extract_caucafall_v2c.yaml
    uv run python scripts/events/extract_track_sequences.py \
        --config configs/events/extract_ur_fall_v2c.yaml
    uv run python scripts/events/extract_track_sequences.py \
        --config configs/events/extract_multicam_v2c.yaml
    uv run python scripts/events/extract_track_sequences.py \
        --config configs/events/extract_multicam_allcams_v2c.yaml
    uv run python scripts/events/extract_track_sequences.py \
        --config configs/events/extract_wu_sahzu_v2c.yaml

    # 3. Train on the trees steps 1 and 2 wrote: the _v2c trees for both lanes, plus the
    #    ajile12 negative diet for the seizure recipe. One command per seed for fall, one
    #    for the seizure recipe. The seed siblings without the _v2c marker are the
    #    recorded runs and read the historical trees, which step 2 does not rebuild.
    uv run python scripts/events/train_event_verifier.py \
        --config configs/events/train_fall_F0_v2c_s20260826.yaml
    uv run python scripts/events/train_event_verifier.py \
        --config configs/events/train_fall_F0_v2c_s20260901.yaml
    uv run python scripts/events/train_event_verifier.py \
        --config configs/events/train_fall_F0_v2c_s20260905.yaml
    uv run python scripts/events/train_event_verifier.py \
        --config configs/events/rot_train_conv_finalB_v2c.yaml

    # 4. Score what step 3 trained. One look per promoted candidate per exam, then the
    #    deep metrics that the specificity gate reads. The three fall configurations all
    #    read seed 20260901, the median-validation seed that took the fall looks; the two
    #    seizure configurations read the finalB refit. Same rule as step 3: these read the
    #    _v2c exam trees, and the configurations without the marker read the historical
    #    ones.
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/eval_fall_urfall_F0_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/eval_fall_multicam_F0_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/eval_fall_multicam_allcams_F0_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/rot_eval_E1_finalB_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/rot_eval_E2_finalB_v2c.yaml
    uv run python scripts/events/metrics_deep.py \
        --eval-dir outputs/verifier/fall_F0_v2c_s20260901/eval_ur_fall \
                   outputs/verifier/fall_F0_v2c_s20260901/eval_multicam \
                   outputs/verifier/fall_F0_v2c_s20260901/eval_multicam_allcams \
                   outputs/verifier/conv_finalB_v2c/eval_E1

    # 5. Re-slice the frozen heads onto the same _v2c exam trees. These score heads that
    #    were trained earlier, not the step-3 candidates, so no step-3 output feeds them:
    #    the `*_champion_v2c.yaml` configurations read the committed lane graphs under
    #    final_models/, which need only a git lfs pull, and the v2pose configurations read
    #    the bar holders' own run artefacts from the earlier generation, which no step above
    #    rebuilds. They are what the candidates above are read against on one surface.
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/eval_fall_urfall_champion_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/eval_fall_multicam_champion_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/eval_fall_multicam_allcams_champion_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/rot_eval_E1_champion_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/rot_eval_E2_champion_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/eval_fall_urfall_v2pose_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/eval_fall_multicam_v2pose_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/eval_fall_multicam_allcams_v2pose_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/rot_eval_E1_finalBv2pose_v2c.yaml
    uv run python scripts/events/eval_event_verifier.py \
        --config configs/events/rot_eval_E2_finalBv2pose_v2c.yaml

    # 6. Regenerate the whole-lane statistics table from the run outputs.
    uv run python scripts/events/stats_compendium.py

### Artefacts

- Per dataset, under the extraction `out_dir`: `sequences/<Location>__<video>.jsonl`, one
  row per frame and track; `meta/<Location>__<video>.json` with the ground-truth interval,
  frames per second and counts; `runs/<key>/` with the underlying pipeline run; and
  `extraction.json` carrying the model provenance, the configuration digest and the totals.
- Per training run, under the trainer `out_dir`: `verifier.pt`, `verifier.onnx` and
  `report.json`. The report carries the best validation area under the curve, operating
  points at 0.5, 0.8 and 0.9, the checkpoint sha256, the ONNX sha256 and the full per-epoch
  history. Only the ONNX ships; the torch checkpoint stays a run artefact.
- Per evaluation run, under the evaluation `out_dir`: `report.json` and `windows.jsonl`,
  both carrying the model sha256 and the prototype disclosure.
- Per-epoch curves, committed as `final_models/fall/curves/*.csv` and
  `final_models/seizure/curves/*.csv`. The third column is named `val_ap` in all of them
  but carries a validation area under the curve for the event runs, so the axis label is
  set per lane rather than from the column name.
- The shipped lane directories: `final_models/fall/` and `final_models/seizure/`, each with
  `model.onnx`, `model.onnx.sha256`, `model.json`, `report.json` and the figures.

## Re-identification: phase A

Re-identification here is a session-scoped appearance embedding over person crops whose
long-gap re-associations are proposed to a human and never bound automatically. Appearance
only: no pose, gait, body-shape or face-adjacent signal. Every weight the lane produces is
prototype class.

**Corpus.** MEVID, a tracklet-level benchmark built on ordinary perspective and top-view
security-camera footage, which is the target domain. The registry entry
`configs/datasets/mevid.yaml` carries the recorded licence history and bans the actor
check-in photograph package by name, because it is face imagery. The loader,
`scripts/reid/mevid_dataset.py`, owns the on-disk format: chip filenames encode person,
outfit, camera, tracklet and frame; the tracklet tables index into the chip name lists with
an inclusive end offset, which is pinned by a test because getting it wrong silently drops
one frame from every tracklet. The loader carries the outfit identifier through as parallel
metadata so an outfit-aware evaluation is possible.

A second corpus was audited and rejected for this lane: the SmartSpaces identities were
found to recycle roughly 35 to 45 distinct avatar builds across 231 labels, which makes it
unusable as a re-identification training set. The instrument is
`scripts/reid/smartspaces_mesh_audit.py` and the record is
`research/reid-smartspaces-mesh-audit.md`.

**Recipe.** `configs/reid/train_mevid_a1.yaml` declares the weight class as prototype and
the trainer refuses to start if it says anything else, so a checkpoint can never end up
unlabelled. The model is OSNet-x1_0 from an ImageNet-only initialisation; no
re-identification-pretrained weight is used anywhere, the banned lineages are refused by
name at load and export time, and the allowlist is `docs/LICENCE_ALLOWLIST.md`, enforced by
`scripts/guards/check_reid_weight_sources.py`. The library is pinned at commit
`f8cd150fdf77e8d9e1ed143b7f308c2c609ded50` and is installed without dependencies and
without build isolation, because the same-named package on the public index is a different
project. Fixed settings: seed 20260907, 30 epochs, 8 frames sampled at random per tracklet
per epoch for training and a fixed span-covering draw for evaluation, 32 tracklets per
batch with 4 instances per identity under an identity sampler, eight identities per step,
input 256 by 128, random flip and random erasing as the only augmentations, the amsgrad
optimiser at learning rate 1.5e-3 and weight decay 5e-4, a single step schedule dropping by
a factor of ten at epoch 20, cross-entropy over the 104 training identities with label
smoothing and no metric term (`loss: softmax`, the library's own OSNet recipe and its
default learning rate for it), average pooling over frame embeddings, and evaluation every
10 epochs plus the unconditional final pass. The first five epochs train only the classifier
with the ImageNet trunk frozen in eval mode, which is torchreid's two-stepped transfer
learning. The first two runs of this recipe used the library's triplet engine instead, the
same cross-entropy plus a batch-hard triplet term at equal weight with margin 0.3 and the
library's triplet learning rate of 3e-4, and both collapsed to a constant feature, the
second even after the classifier-only phase. Both are recorded on the lane page; the triplet
keys stay in the configuration and are read only under `loss: triplet`.

**Three evaluation protocols**, all reported from one feature extraction so the deltas are
attributable to the protocol rather than to a re-run.

1. General: the library's standard protocol, euclidean distance on raw features. This is
   the number comparable to published results on this benchmark.
2. Camera-debiased: the per-camera mean embedding over the pooled query and gallery
   features is subtracted, features are renormalised, and cosine distance is used. It is
   reported next to a plain normalised cosine baseline so the debiasing effect is separated
   from the metric change.
3. Same-clothes approximation: an in-house definition, not the benchmark paper's protocol,
   and labelled as in-house everywhere it is printed. Positives require the same person and
   the same outfit; same-person different-outfit gallery tracklets are removed from the
   ranking rather than counted as negatives.

The in-house ranker is self-checked against the library's own ranker on the general
protocol, because the same-clothes approximation has no external result to check it against.

**Parity gate.** The export is written only after the evaluation, so a failed parity check
never also loses the numbers. The trainer refuses to declare success unless the minimum
cosine similarity between the torch forward and onnxruntime on random input is above 0.999.
The graph is opset 17, input named `images` shaped (1, 3, 256, 128) with batch as the only
free axis, output named `embeddings`.

**Status.** Measured on 2026-09-07. The three archive sha256 values are recorded in
`configs/datasets/mevid.yaml` and `DATASETS.md`, which is the precondition for quoting any
number from this corpus; the entry carries no placeholder digests, so flipping its status
without the hashes raises rather than passing quietly. The model of record, its result
under the three protocols and the bar it is measured against are on
`final_models/reid/README.md`, every number beside the embedder sha256. The
general-protocol mAP of 0.204782 sits below the external same-clothes floor of 0.39, and
that page says so. A rerun of the recipe on a 60-epoch schedule with the step at epoch 40,
declared in `research/reid-rerun-a2-protocol.md` before it ran, scored 0.188552 and was not
promoted; the lane page records it.

### Commands

    # On the pod: training environment, then the pinned library, then the run. The
    # locked sync needs uv 0.11 or later and removes every package the lock does not
    # name, so the library's own dependencies are installed after it. The library is
    # installed from a clone at its pin, without dependencies and without build
    # isolation, because the same-named package on the public index is a different
    # project; its build imports cv2, its data module imports gdown and the engine
    # imports tensorboard, none of which the cuda lock carries.
    uv sync --extra cuda --no-group mac --locked
    git clone https://github.com/KaiyangZhou/deep-person-reid.git /workspace/torchreid
    git -C /workspace/torchreid checkout f8cd150fdf77e8d9e1ed143b7f308c2c609ded50
    uv pip install cython tensorboard gdown opencv-python-headless onnx onnxruntime pillow
    uv pip install --no-deps --no-build-isolation /workspace/torchreid
    .venv/bin/python -u scripts/reid/train_reid_mevid_a1.py \
        --config configs/reid/train_mevid_a1.yaml

    # Standalone export of a locally fetched weight file. Never downloads; refuses a URL
    # and refuses a path naming a banned lineage.
    UV_PROJECT_ENVIRONMENT=.venv-curate uv sync --extra curate --no-default-groups
    .venv-curate/bin/python scripts/reid/export_osnet_onnx.py \
        --weights checkpoints/osnet_x1_0_imagenet.pth \
        --out outputs/reid_mevid_a1/osnet_x1_0.onnx

    # The corpus audit that rejected the alternative training set. Writes contact
    # sheets and per-scene identity and crop counts.
    uv run python scripts/reid/smartspaces_mesh_audit.py \
        --gt-dir     <one Scene.json per scene, raw or slimmed to person entries> \
        --frames-dir data/train_ok/physicalai_smartspaces/frames \
        --out        outputs/reid/smartspaces_mesh_audit

### Artefacts

Under `outputs/reid_mevid_a1/`:

- per-epoch checkpoints, pruned to the newest two, plus `model-best.pth.tar`, which is
  never pruned;
- `curves.csv` with columns `epoch, loss_t, loss_x, acc, lr, test_map, test_rank1`;
- `eval_summary.json` carrying all three protocols and the normalised cosine baseline,
  with the camera counts and any singleton cameras recorded;
- `osnet_x1_0_mevid_a1.onnx` and its `.sha256`, plus the sidecar recording the initialisation
  path and its digest.

The committed lane directory is `final_models/reid/`, which holds `model.json`,
`eval_summary.json`, the curve file `final_models/reid/curves/reid_mevid_a1.csv`, the
exported graph `model.onnx` with its `model.onnx.sha256`, and the two figures.
