<div align="center">

<img src="assets/brand/vividly-square-black-rounded-1024.png" alt="Vividly" width="120" />

# Vividly Machine Vision Engine

**Patient awareness for hospital waiting rooms, built on the cameras already on the ceiling.**

</div>

Vividly watches a waiting room for two acute events, a collapse to the floor and a rhythmic
convulsion, and for a patient who leaves through an exit without being discharged. A geometric
rule proposes a candidate from track state, a trained verifier confirms it, and only a
confirmed candidate becomes an alert. Nothing leaves the building except the alert metadata.
The runtime path loads committed graphs and assumes no cloud connectivity at inference time;
the only network use in this repository is fetching datasets and upstream checkpoints for
training.

This repository holds the five model lanes, the replay pipeline they plug into, the demo
platform, the tests, and the measurement record. This page is the integration reference: what
each graph takes and returns, how each lane is wired, where every threshold lives, and how to
run the whole path from a clean checkout. The lane pages under [final_models/](final_models/)
are the record behind every number here.

## Get it running

```bash
git lfs pull                         # the five graphs under final_models/ are Git LFS objects
uv sync                              # macOS development and harness environment
make guards                          # the seven guards, the import contracts, the guard self-test
make test                            # the test suite, excluding the gpu and slow marks
uv run python scripts/run_pipeline.py --config configs/pipeline/synthetic_demo.yaml --headless
uv run python scripts/run_pipeline.py --config configs/pipeline/replay_demo.yaml --headless
make demo                            # demo platform on 127.0.0.1:8700
```

The first pipeline command needs no weights: it replays generated tracker observations through
the rules, the alert store and the latency table. The second runs the committed clip through
all six stages on the models of record, with both event verifiers attached, and prints the
per-stage p50 and p95. Without `git lfs pull` every graph is a short pointer file, the replay
config refuses to start, and the demo picker lists every lane as disabled. Verify a weight in
place with:

```bash
cd final_models/<lane> && shasum -a 256 -c model.onnx.sha256
```

`make demo` starts the local single-page demo platform: upload a video or run the browser
camera, pick a detector and a pose checkpoint, and watch boxes, skeletons, track identifiers
and candidate events stream live, then download the annotated file with per-stage latency
from that run. The two event verifiers attach to every demo run from their lane directories;
the re-identification lane is reached through the pipeline configuration, not the picker. The
browser camera is watch-only: frames are processed in memory and nothing from the camera is
written to disk. That claim covers only that path. The separate bring-up script for a physical
camera, `scripts/live_camera.py` in [docs/LIVE_CAMERA.md](docs/LIVE_CAMERA.md), does write
annotated frames to the directory it is given.

## The five lanes

| Lane | Model of record | Checkpoint | ONNX | Headline metric | Exam | Page |
|---|---|---|---|---|---|---|
| eyes | RT-DETRv4-S person v2, seed 0 | `345c30a5` | `40262049` | AP 0.50:0.95 0.8356 | person_v2 dev-val | [eyes](final_models/eyes/README.md) |
| joints | RTMPose-m person v2, seed 0 epoch 38 | `a0a035fe` | `6ba96285` | keypoint AP 0.774115 with ground-truth boxes | COCO val2017 | [joints](final_models/joints/README.md) |
| fall | fall_v3 | `3d2d17ee` | `67954c04` | window AUC 0.79276 | multicam, all cameras | [fall](final_models/fall/README.md) |
| seizure | convulsion_v2_real | `715d172c` | `07c399be` | window AUC 0.69841 | wusahzu testsplit | [seizure](final_models/seizure/README.md) |
| reid | OSNet-x1_0 MEVID Phase A, epoch 30 | `66623239` | `60164a99` | mAP 0.204782, general protocol | MEVID test split | [reid](final_models/reid/README.md) |

The table shows the first eight characters of each digest; the full values are in each lane's
`model.onnx.sha256`, `model.json` and page. Every weight is the prototype class: internal
demonstration material, not a medical device, and no number above is clinical detection
performance. A number is quotable only with the hash beside it. What each lane trained on,
in one line each, with the full record on its page:

- eyes trained on person_v2, 55,200 rendered warehouse images from the NVIDIA PhysicalAI
  SmartSpaces MTMC_Tracking_2025 scenes, and was measured on a held-out scene. Its page records
  that a detector trained entirely on synthetic warehouse renders transfers poorly to a real
  operating room, with recall holding up better than precision.
- joints trained on COCO 2017 person keypoints only, from the body7 initialisation `e48f03d0`,
  which scores 0.764499 on the same protocol and which the model of record beat.
- fall trained on 881 windows of track and pose features from three public, licence-reviewed
  acted-fall sets, never on imagery.
- seizure trained on 1,106 windows from eight WU-SAHZU epilepsy monitoring unit videos, never
  on imagery.
- reid trained on the MEVID training split over an ImageNet-1k initialisation. Its result sits
  below the external same-clothes floor of 0.39 and the published general-protocol figure of
  0.271; it is the lane's first measured embedder and its page records the two collapsed runs
  that preceded it.

The two event lanes were measured on track trees from their own pinned detectors, named on
their pages, not from the eyes model of record. The replay configuration drives them off the
eyes and joints models of record; the two AUC values above hold only under the pinned
detectors, and the 2026-09-07 refits on the new trees were measured and not promoted.
[final_models/README.md](final_models/README.md) carries that reasoning in full.

## The graphs

Every graph is opset 17 ONNX with a dynamic batch axis, read off the committed files with
onnxruntime. Every backend runs on the CPU execution provider: on macOS the CoreML provider
crashes inside the run call on the detector graphs at onnxruntime 1.28.0, and the RT-DETRv4
and re-identification backends refuse anything but `cpu` and `cuda`.

| Lane | Inputs | Outputs | Preprocessing, as the backend does it |
|---|---|---|---|
| eyes | `images` (N,3,640,640) float32; `orig_target_sizes` (N,2) int64 as width, height | `labels` (N,300) int64; `boxes` (N,300,4) float32, xyxy in original frame pixels; `scores` (N,300) float32 | BGR to RGB, Pillow bilinear resize to 640x640 with no letterbox, divide by 255, no mean or standard deviation. `src/vividly/models/rtdetr_backend.py`, `preprocess_bgr`; `scripts/eval/backend_parity_rtdetr.py` asserts parity with the evaluator's path. |
| joints | `input` (N,3,256,192) float32 | `simcc_x` (N,17,384) float32; `simcc_y` (N,17,512) float32 | The rtmlib contract: affine crop of each box to 192x256, decode outside the graph by argmax over the SimCC bins at split ratio 2.0, then the affine back-projection. The backend feeds the frame in OpenCV BGR order; the joints page records the measured effect of feeding RGB instead. |
| fall | `windows` (N,64,8) float32 | `logits` (N) float32 | 64 steps of eight track features per step: `vy`, `vx`, `h_rel`, `aspect`, `angle_norm`, `angle_valid`, `kpc`, `present` (`src/vividly/verifier/features.py`). Window 8.0 s at stride 1.5 s, policy contain, read from `report.json`. |
| seizure | `windows` (N,64,10) float32 | `logits` (N) float32 | The same eight features plus `scene_n_tracks` and `scene_births`. Window 12.0 s at stride 3.0 s, policy overlap, read from `report.json`. |
| reid | `images` (N,3,256,128) float32 | `embeddings` (N,512) float32 | Resize the BGR crop to 128x256 with bilinear interpolation, BGR to RGB, divide by 255, subtract the ImageNet mean and divide by the ImageNet standard deviation, HWC to CHW. The rows are L2-normalised after the graph. `src/vividly/models/reid_backend.py`. |

A verifier score is `sigmoid(logit / temperature)` clamped to [0, 1]. The temperature is not in
the graph: it is `calibration.temperature` in the lane's `report.json`, 1.0251 for fall and
2.377 for seizure. The window geometry is read from the same file and never restated in a
pipeline configuration.

## Wiring a lane into the pipeline

`src/vividly/models/factory.py` is the only place a backend is named. `input_size` is written
as width then height everywhere, in configurations and in the `model.json` sidecars, and a
repository-relative `onnx` path is resolved against the repository root by the runner. The
fragments below are the ones [configs/pipeline/replay_demo.yaml](configs/pipeline/replay_demo.yaml)
runs.

```yaml
detector:
  backend: rtdetr_onnx
  onnx: final_models/eyes/model.onnx
  input_size: [640, 640]
  device: cpu
  score_thr: 0.3
  name: rtdetrv4-s-person-v2-40262049

pose:
  backend: rtmlib
  onnx: final_models/joints/model.onnx
  input_size: [192, 256]
  device: cpu
  name: rtmpose-m-person-v2-6ba96285

verify:
  policy: annotate
  heads:
    seizure_candidate_v0:
      run_dir: final_models/seizure
      confirm_threshold: 0.8
    fall_then_still_v0:
      run_dir: final_models/fall
      confirm_threshold: 0.95
```

The event heads load from a directory that holds `report.json` beside the graph, either a
training run with its `verifier.onnx` or a lane directory with `model.onnx`, and a graph whose
sha256 differs from the digest in the report is refused. Each head accepts `run_dir`,
`confirm_threshold`, `pre_pad_s`, `persist_k` and `hyst_hi`, and an unknown key is an error.
A firing whose rule has no head, the egress head included, is stamped `unavailable`, never
scored by guess. A seizure head scores only once the engine's scene hook has run, because its
two scene channels count every track in the room, not only the firing one. The thresholds
above are the operating points the lane pages record.

The re-identification stage is off by default and no shipped configuration enables it on real
footage. Point a pipeline config at [configs/pipeline/reid.yaml](configs/pipeline/reid.yaml)
with `reid_config:`, or carry a `reid:` block inline, never both; set `enabled: true` and
`embeddings.source: embedder`; and give the embedder block:

```yaml
embedder:
  backend: onnx
  model_path: final_models/reid/model.onnx
  train_data: [mevid]
  pretrain_data: [imagenet-1k]
  device: cpu
  name: osnet-x1-0-mevid-a1-60164a99
```

`train_data` and `pretrain_data` are required with no defaults, because for re-identification
weights the lineage is the licence question, and the backend refuses the banned lineages
listed in `src/vividly/models/reid_backend.py`. The stage enrols a track when the book reaps
it, probes every later track against the session gallery, and writes a match at or above
`tau` as a pending proposal in the alert store. Nothing merges until a person confirms on the
dashboard, and the gallery is emptied on every exit path.

## Running it

```bash
uv run python scripts/run_pipeline.py --config <config> [--headless] [--max-frames N]
```

Exit code 0 means the run completed, 2 is a usage error and 3 is a configuration error, raised
before anything is opened. A run writes `<output.dir>/<run_name>/alerts.sqlite` and
`<output.dir>/<run_name>/latency.jsonl`, and prints p50 and p95 per stage for the six stages
`source`, `detect`, `pose`, `track`, `rules` and `render`. Three source shapes exist:

- a clip: `source: {path: <file>, rate: <null | 1.0 | 2.0>}`, where `rate` only paces playback
  and `t_source` is always frame index divided by the clip's frame rate;
- an image sequence: `source: {images: <dir>, fps: <Hz>, pattern: "*.jpg", rate: ...}`, where
  `fps` is mandatory because no frame directory records its rate;
- zero footage: `detections_from: <jsonl>` replaces the detect, pose and track stages with
  replayed tracker observations, so the run needs no weights, no OpenCV and no network.

The shipped configurations under `configs/pipeline/`:

| File | What it runs |
|---|---|
| `replay_demo.yaml` | the committed clip on eyes, joints and the two event verifiers, dashboard on 127.0.0.1:8600 |
| `synthetic_demo.yaml` | zero footage through the rules and the alert store, dashboard on 8601 |
| `egress_demo.yaml` | the egress head on a 220 s synthetic scene with exit zones, dashboard on 8603 |
| `synthetic_reid_demo.yaml` | the re-identification stage on synthetic embeddings, dashboard on 8602 |
| `rules.yaml` | every rule threshold, loaded by the others through `rules_config:` |
| `reid.yaml` | the re-identification settings, off by default |

## Where the thresholds live

Every value in this list is PROVISIONAL until the calibration protocol in
[docs/CALIBRATION.md](docs/CALIBRATION.md) has its preconditions met, and no firing count from
any run is a false-alarm rate.

1. The 21 rule values, the three egress clocks among them, in `configs/pipeline/rules.yaml`,
   loaded through `rules_config:` or an inline `rules:` block, never both.
2. Each verifier head's `confirm_threshold`, with `pre_pad_s`, `persist_k` and `hyst_hi`, under
   `verify.heads.<rule_id>` in the pipeline configuration.
3. The detector's `score_thr` in the `detector:` block.
4. The tracker gates in the `tracker:` block: `track_activation_threshold` and `score_floor`
   for BoT-SORT, with `high_conf_det_threshold` on the ByteTrack baseline.
5. The keypoint confidence floor `kp_conf_min` in the `tracks:` block.
6. The re-identification values in `configs/pipeline/reid.yaml`, `tau` and
   `confidence_at_tau` among them; the file tags each one PROVISIONAL and separates them from
   the entries it labels sizing rather than thresholds.

## Environments and guards

```bash
uv sync                                                          # macOS development and harness
uv sync --extra cuda --no-group mac --locked                     # Linux GPU training and evaluation
uv sync --group linux-harness --no-default-groups                # Linux extraction, onnxruntime only
UV_PROJECT_ENVIRONMENT=.venv-agpl   uv sync --extra eval-agpl --no-default-groups
UV_PROJECT_ENVIRONMENT=.venv-curate uv sync --extra curate --no-default-groups
```

The `cuda` extra conflicts with `eval-agpl` and with `curate` by declaration, which is the
enforcement of the isolation rule. [TRAINING.md](TRAINING.md) says which environment each
reproduction step runs in. `make guards` runs the seven scripts under `scripts/guards/`
(`check_no_fisheye`, `check_no_fisheye_primitives`, `check_agpl_imports`, `check_face_deps`,
`check_train_data_paths`, `check_config_registry`, `check_reid_weight_sources`), then
`uv run lint-imports` for the import contracts, then `bash scripts/guards/selftest.sh`, which
proves each guard against rejection cases. `make test` runs `uv run pytest -m "not gpu and not
slow"`, and `make lint` runs `ruff check` and `ruff format --check`.

## Where the rest lives

| Document | What it holds |
|---|---|
| [POLICY.md](POLICY.md) | every standing rule, the carve-outs, the acquisition law, and the file that enforces each |
| [TRAINING.md](TRAINING.md) | how each lane is reproduced, command by command, with the artefacts each step produces |
| [DATASETS.md](DATASETS.md) | the registry summarised: bucket, licence, status, use and acquisition record per dataset |
| [final_models/](final_models/) | the model record: index across the five lanes, one page per lane, curves and figures |
| [research/](research/) | the evidence documents behind the numbers, one flat folder with an index |
| [docs/PIPELINE.md](docs/PIPELINE.md) | the replay pipeline: sources, tracks, rules, verifier seam, alert store, dashboard |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | the protocol that turns a provisional threshold into a calibrated one |
| [docs/LIVE_CAMERA.md](docs/LIVE_CAMERA.md) | camera bring-up against a network video recorder rig |
| [docs/COUNSEL_QUESTIONS.md](docs/COUNSEL_QUESTIONS.md) | the counsel register, one section per question identifier |
| [docs/LICENCE_ALLOWLIST.md](docs/LICENCE_ALLOWLIST.md) | recorded licence exceptions and the re-identification weight-source rules |

## Repository map

| Path | What lives there |
|---|---|
| `src/vividly/` | the package: `models/` `data/` `sources/` `tracks/` `rules/` `verifier/` `events/` `eval/` `reid/` `dashboard/` `demo/` `schemas/` `media/` |
| `configs/` | everything tunable: `pipeline/` `events/` `eval/` `datasets/` `reid/` `checkpoints/` `convert/` |
| `scripts/` | the entry points, grouped by lane: `detector/` `pose/` `events/` `reid/` `eval/` `data/` `pod/` `analysis/`, plus `guards/` |
| `final_models/` | the shipped weights, their sidecars, their pages, their curves and their figures |
| `tests/` | the test suite and its committed synthetic fixtures; no real footage is in version control |
| `third_party/` | the vendored-trainer recipe: pins, patches and wrapper training configurations |
| `data/` | ignored by version control: `train_ok/` `train_prototype/` `eval_only/`, physically separate by licence bucket |
| `checkpoints/` | ignored by version control: upstream initialisation weights fetched by the reproduction steps |
| `outputs/` | ignored by version control: training runs, evaluation dumps and pipeline runs |

Every runner, trainer, extractor and evaluator under `scripts/` that reads a recorded recipe
takes `--config`, and the recipe rather than the script source carries every dataset path,
hyperparameter and threshold. Guards, analysis tools, instrument graders, weight soups,
parity probes and the demo platform launcher sit outside the rule and take plain flags.

## Guardrails

Eleven standing rules hold this repository together, most of them enforced by a guard, an import
contract or the dependency resolver rather than by review: no fisheye imagery, no facial
recognition, no vital-signs extraction, no simulated or acted medical events in training, AGPL
isolation, evaluation-only data never trains, commodity cameras only, recorded checkpoint
provenance, no unmeasured performance claim, no number without its hash, and refused weight
sources for re-identification. Each is stated in full in [POLICY.md](POLICY.md), with its
enforcing file named by path. Prove them with:

```bash
make guards                          # every guard, the import contracts, and the guard self-test
```

## Attribution

COCO 2017 annotations are used under CC BY 4.0, with attribution to the COCO Consortium; the
images are governed by their individual Flickr terms. Test fixtures are CC BY 2.0 images from
COCO val2017 with per-image attribution recorded in
[tests/fixtures/ATTRIBUTION.md](tests/fixtures/ATTRIBUTION.md). Every other dataset carries its
attribution line word for word in its registry entry under `configs/datasets/`, summarised in
[DATASETS.md](DATASETS.md). No registered dataset is redistributed from this repository.

## Data attribution

Every registered dataset held on disk whose licence requires attribution is credited here to
the party that collected and published it. The full attribution line, the licence text and the
licence URL are in that dataset's registry entry under `configs/datasets/`, which is the legal
record; this table is the credit itself, and
`tests/test_registry.py::test_attribution_required_entries_are_credited` fails if an entry that
owes attribution is missing from it.

| Dataset | Credited to |
|---|---|
| coco | COCO Consortium. Lin et al., Microsoft COCO: Common Objects in Context, ECCV 2014 |
| physicalai_smartspaces | NVIDIA (Physical AI / Metropolis, AI City Challenge organizers) |
| ajile12 | Peterson et al. (Brunton lab, University of Washington) |
| caucafall | J. C. Eraso, E. Munoz, M. Munoz, J. Pinto (Universidad del Cauca) |
| gmdcsa24 | Ekram Alam et al. (GMDCSA-24 authors) |
| le2i_fdd | Imen Charfi, Johel Miteran, Julien Dubois (Le2i / ImViA, U. Bourgogne) |
| multicam_fall | E. Auvinet, C. Rougier, J. Meunier, A. St-Arnaud, J. Rousseau (DIRO, UdeM) |
| ur_fall | Bogdan Kwolek & Michal Kepski, University of Rzeszow / AGH |
| mvor | CAMMA, University of Strasbourg / IHU Strasbourg (Srivastav et al.) |
| wu_sahzu_emu | Y. Xu, J. Wang and colleagues (Westlake University and Zhejiang University, VSViG project), recorded at the Second Affiliated Hospital of Zhejiang University School of Medicine epilepsy monitoring unit |
| wu_sahzu_emu_trainsplit | the same recording team and unit as wu_sahzu_emu, of which this entry is the train-side split |
| mevid | Kitware (MEVID annotation release), over the MEVA corpus (mevadata.org). Davila, Du, et al., MEVID: Multi-view Extended Videos with Identities for Video Person Re-Identification, WACV 2023 |
