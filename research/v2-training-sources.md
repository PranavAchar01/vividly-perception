# Training dependencies: pins, licences and local changes

Recorded 2026-08-31. One row per dependency: where it comes from, the exact pin, the licence
as verified from the raw file and never from a repository badge, and what was changed.
Verification tags: **[V]** means checked first-hand, that is file and line in a pinned clone,
bytes on local disk, or a live URL probe; **[R]** means a third party's claim at a URL that
was opened and read. The underlying audits of the architecture rows cited file and line in
pinned clones that are not carried in this repository, so those [V] tags cannot be
re-opened from a clone. What can be checked here is `third_party/PINS.yaml`, the
registry entries under `configs/datasets/` and the provenance cards inside
`third_party/configs/*.yml`, which carry the pins, the licence readings and the deltas.

## 1. Architectures and training recipes, the two comparison arms

| | Model A | Model B (challenger) |
|---|---|---|
| Architecture | D-FINE-S (10M params) | RT-DETRv4-S (10.43M params) |
| Recipe | DEIM (CVPR 2025) | RT-DETRv4 (ECCV 2026), DINOv3 teacher **disabled** |
| Papers | arXiv:2412.04234 (DEIM), arXiv:2410.13842 (D-FINE) | arXiv:2510.25257 |
| Repo | github.com/Intellindust-AI-Lab/DEIM | github.com/RT-DETRs/RT-DETRv4 |
| Pin | `09d35d53d39ee3145a1e61e3a989b28b9468d1dd` [V] | `55fefaaed7efe2a5f72d0a18fd4e05965e35c292` [V] |
| License | Apache-2.0 (body word for word; Intellindust header prepended 2025-07-21, no terms change) [V] | Apache-2.0, stock unmodified [V] |
| Init checkpoint | DEIM-D-FINE-S, 49.0 AP COCO | RT-DETRv4-S, 49.8 AP COCO |
| Checkpoint lineage | ImageNet (PPHGNetV2-B0 stage1), then COCO train2017. **No Objects365** [V] | Same backbone, then COCO train2017 only (proved by the shipped training log: 118,272 images per epoch, which is train2017 with the final partial batch dropped, 118,287 images in 3,696 full batches of 32, and nowhere near an Objects365 stage). **No Objects365** [V] |
| sha256 | `a089c9eb798f8a0322e05f757fe8e8188eee34aa578788a35ec253bcec8a6b38`, recorded at first pod download into `ckpt/MANIFEST.sha256` because Google Drive is not content-addressed, and carried in `third_party/PINS.yaml` | `238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee`, recorded the same way and carried in `third_party/PINS.yaml` and `final_models/eyes/README.md` |

Also audited but **not vendored**: the original D-FINE repository (Peterande/D-FINE at
`956d170`, Apache-2.0 [V]). DEIM vendors its own complete copy of the D-FINE model code, so
one clone per arm suffices [V].

What was changed, in full (each delta is justified in the config's provenance card,
`third_party/configs/*.yml`, and composed through the repositories' own config loaders
with the effective values asserted before commit [V]):

- `num_classes: 1`, `remap_mscoco_category: False`, person = **category id 0**
  (both trainers index the class head by the raw category_id when remap is off [V];
  a 1-based id triggers a device assertion, per DEIM #12, #25, #109 and D-FINE #10, #37 [R]).
- 48-epoch fine-tune schedule scaled coherently by the repositories' own formulas
  (flat 28 = 4 + 48//2; aug policy [4, 28, 40]; stop_epoch 40 = epoches minus
  no_aug 8), batch 32 as in the recipe, lr halved to 2e-4/1e-4, marked provisional at
  the time (section 6 records that it was carried unchanged into the recorded runs).
- Model B only: the `teacher_model` block is absent, which is the only disable mechanism the
  code supports [V], with `distill_teacher_dim: 0` and losses without `'distill'`, so
  **no DINOv3 weights are ever downloaded and no click-wrap is ever accepted** [V]. What
  remains is counsel item C-1: the released init was *trained with* a DINOv3 teacher, proved
  by the live loss_distill in the shipped log [V], though no DINOv3 tensor is in the file,
  because the teacher is skipped at serialization by the solver's `state_dict` method in
  det_solver.py [V].
- 8 patches, apply-verified against pristine pins [V] and carried under
  `third_party/patches/`: the DEIM torchvision at or above 0.21
  hook shim (a hard blocker; upstream PR #47 unmerged [V]); the export dummy batch moved from
  32 to 1 and the opset from 16 to 17 (LayerNorm needs 17 or later, per D-FINE #153 and
  #302 [R]); `weights_only=False` on checkpoint loads (the torch at or above 2.6 default flip);
  and, on RT-DETRv4 only, the startup FLOPs profile moved onto a CPU copy, the evaluation
  position embeddings registered as non-persistent buffers, and the legacy TorchScript
  exporter pinned at opset 17 in place of the dynamo exporter.
- Deliberate non-changes, checked and recorded in `third_party/PINS.yaml`:
  the DDF `batch_scale = 8/N` **cancels exactly** in the loss's convex mix, so no
  single-GPU compensation is owed [V]; the O(N²) GIoU out-of-memory risk (RT-DETRv4 #22 [R])
  is a prepared contingency, not a pre-emptive rewrite; GO-LSD stays on, because it is
  always on inside the `'local'` loss and free at inference [V].

Integration rules from the audits that govern how these are run: use plain `python
train.py` with `-t`, because torchrun would wrap the run in DDP and empty negative frames
would desynchronise the denoising branch (DEIM #32 [R]), and `-t` sets `pretrained=False`
past the `get_rank()` crash [V]; never pass `-d cuda:0`, which autocast rejects [V, rtv4];
the final artefact is `best_stg2.pth` (maintainer, DEIM #56 [R]); watch for
`./NaN.pth`, because the trainers continue through NaN [V]; on NaN, the first fallback is
bf16 autocast and the second is dropping AMP, since fp16 AMP is the recipe but also the
documented NaN suspect [R]; training and validation preprocessing is a plain stretch
resize to 640x640 [V], so every downstream ONNX evaluation and CPU benchmark of these
students must stretch-resize too, because the repositories' own onnx_inf.py letterboxes and
disagrees with training (D-FINE #208, #358 [R][V]).

## 2. Box fusion, training side only, nothing here ships

| Component | Package | License | Role |
|---|---|---|---|
| WBF | `ensemble-boxes==1.0.9` (reference implementation, `dev` dependency group) | MIT [V] | fusion behind `src/vividly/eval/wbf.py`, the module the I1 test-time-augmentation row of `research/eyes-instrument-protocol.md` is fused with; paper arXiv:1910.13302 |

The fusion behaviour that matters for reading fused scores is pinned by
`tests/test_wbf.py`: WBF rescales a cluster's score by the weight of the views that
contributed over the total weight, so a box seen by one of two views keeps half its score,
and a fixed cut on fused scores is therefore not the operating point the same cut gives on
one view [V].

At the time of this record the tree also carried a teacher pseudo-labelling stack (four
open-vocabulary detection and segmentation models hosted natively by `transformers`, with
the same fusion module as its consensus step). It produced no label that any model of
record trained on, its filter threshold was never swept, and it has since been removed
from the repository together with the `transformers` dependency it needed; no surviving
dependency rests on its licence readings.

## 3. Datasets in this build

| Dataset | License | Role | State |
|---|---|---|---|
| NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2025 slice) | CC BY 4.0 (the registry entry records it word for word) | training corpus (synthetic warehouses; robots excluded) | **Built**: 55,200-image train and 1,800-image dev-val manifests, person id 0, visible-region boxes as released (section 7e house convention) [V on disk] |
| MEVA | CC BY 4.0. The published sentence reads "All MEVA data is available for use under a CC BY-4.0 license", verified word for word on mevadata.org [V] | not a training or exam source of any model of record | considered as a real-footage pool for a teacher probe and a frozen detection exam; the probe ran on four public clips and the exam was designed and never built, and neither the clip list nor the design is carried in this repository. No MEVA entry exists under `configs/datasets/`; the licence reading here is the one `configs/datasets/mevid.yaml` relies on, because the MEVID re-identification chips are cut from MEVA video |
| MVOR | CC BY-NC-SA 4.0 (EVAL_ONLY bucket) | local probe exam (real clinical-adjacent multi-view) | **Built**: 2,196 images and 4,699 boxes, every frame verified on disk [V]; local only, never a pod fetch |
| SmartSpaces val scenes (Hospital_000, Lab_000, W015/W016) | same CC BY 4.0 | exam-side, frozen | untouched by selection; never fetched to a pod; excluded from every manifest [V by construction and by script refusal] |

Dev-val rule: Warehouse_005 is held out by scene from training, frame-thinned by a factor of
5, so per-epoch checkpoint selection never touches exam material.

## 4. Runtime and tooling pins

- torch `>=2.7` with torchvision `>=0.22` from the cu128 index (Blackwell sm_120;
  no older torch can drive the 5090). Trainer environments are built per repository by
  `scripts/pod/setup_pod.sh` with recorded freezes. No Blackwell run of either trainer had
  been published when this was audited (zero tracker mentions [V searched]); the person_v2
  runs recorded in this repository were the first, and the probe seed served as the smoke
  test.
- `faster-coco-eval`, the trainers' COCOeval replacement, is their requirement and lives in
  their environments; `pycocotools` lives in the project environment.
- onnx and onnxsim, which the export scripts hard-require [V]; export at opset 17,
  with 18 as the recorded fallback if the dynamo exporter engages (rtv4 #28 [R]).
- ffmpeg via subprocess argv (house rule; ffmpeg-python is banned).
- gdown for checkpoint fetch, and the huggingface_hub CLI with include-only patterns; the
  two recorded hugging-face traps from the MVOR fetch are baked into the script.

## 5. Deliberately not used, and why

- **DEIMv2**: a separate repository, relicensed as non-commercial on 2026-08-24 (its
  LICENSE.md section 3 bans commercial use and section 5 extends that to weights [R],
  confirmed by a recon-verified diff of the relicensing commit), and DINOv3-based. Banned,
  and stale Apache-era clones must not be laundered in.
- **Every `*_obj365` and `*_obj2coco` checkpoint** in both families, on Objects365
  terms; the D-FINE README itself warns they "should not be assumed to be
  commercially cleared" [V]. 9 of the 14 released D-FINE checkpoints are tainted;
  only plain `*_coco` files are admitted.
- **DINOv3 weights and the associated click-wrap**: never downloaded, never accepted (see
  C-1 above for what remains).
- **BEDLAM** (non-commercial plus an anti-surveillance clause); **anything AGPL or GPL**
  (POLICY.md, rule 5); **WILDTRACK** (the page currently grants no licence at all [V]);
  **SOMPT22** (the linked Drive folder is not publicly readable and the standing acquisition
  rule admits self-serve anonymous download only, so no access request is made and the set is
  not acquired; DATASETS.md); **SAM 3** (exception only, C-2); and **online Mean-Teacher
  semi-supervision** (a 4 to 8 GPU assumption plus documented confirmation-bias fragility).

## 6. Gaps, named rather than hidden

- TIDE error decomposition is adopted in the method (section 2.1 of
  `research/v2-training-stack.md`) but is not wired into the exam grader, and no number on
  any lane page depends on it.
- A teacher pseudo-labelling pipeline was committed at the time of this record and later
  removed from the repository. Its filter threshold was never swept and no model of record
  was trained on its output; the box-fusion module it shared with the I1
  test-time-augmentation row survives as `src/vividly/eval/wbf.py`.
- The lr halving in both wrapper configs is a fine-tune default, marked provisional in
  `third_party/configs/rtv4_s_person_v2_noteacher.yml` and
  `third_party/configs/deim_s_person_v2.yml`. It was carried unchanged into the recorded
  runs, including the eyes model of record (RT-DETRv4-S seed 0, checkpoint `345c30a5`,
  dev-val AP 0.8356, `research/eyes-rtdetrv4-training-record.md`), and no page records a
  revision of it.

Chain of custody for every claim above that a reader can re-open here: `third_party/PINS.yaml`,
the registry entries under `configs/datasets/`, and the provenance cards inside
`third_party/configs/*.yml`. MEVA has no registry entry of its own; its licence reading
above is the one `configs/datasets/mevid.yaml` and DATASETS.md rest on.
