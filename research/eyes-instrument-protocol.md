# Eyes v2: instruments and next-spend protocol

Dated 2026-09-05.

**Status.** Every item declared below has since been run or cut, and the queue is closed:
research/eyes-seed-spread-and-soup.md records that the eyes GPU queue is complete. The outcome
of each declared item is in the closing status section at the foot of this page. The
declarations themselves are left as they were written.

**Operating conditions, pre-declared before any run.** These were the conditions at the time of
writing. One rented pod GPU was available and jobs ran sequentially on it; the Mac processor
could be used freely. WU-SAHZU extraction was permitted on the pod because nothing from it is
published: the raw footage is deleted once the de-identified track sequences have been
retrieved, and it is never made public. That supersedes the stale local-extraction-only header
carried by the superseded WU-SAHZU extraction configuration, which is no longer in the tree;
the surviving one is configs/events/extract_wu_sahzu_v2c.yaml. GPU work was declared to queue
behind the re-identification chain until that chain exited on its own.

Context: model of record 345c30a5 dev-val AP 0.8356; DEIM floor 0.0012 (3 seeds, 0.8329 at
`5185d500`, 0.8341 at `e98de401` and 0.8333 at `32a783a3`, recorded in
research/eyes-noise-floor.md); pre-registered call bar 0.8368, with 0.8380 as the margin that
would be convincing. Dev-val is synthetic (SmartSpaces Warehouse_005) and is never quoted as
deployment evidence.

## I1: precision instrument for the acute-event rule set (first GPU minutes after re-identification)

Single-view --test-only dumps for `345c30a5` (RT-DETRv4 venv) and `e98de401` (DEIM venv) on
the 1,800-image dev-val produce COCO JSON plus a checkpoint-sha sidecar, which the Mac
pycocotools grader then reads: precision, recall and false positives per image at score 0.3,
**0.4 (the demo point)** and 0.5, IoU 0.5, per camera. Every row carries the label: synthetic
SmartSpaces dev-val, not deployment evidence and not evidence for the acute-event rule set.
Optional in the same session: hflip-only dumps for the pre-declared protocol row (fuse.py
defaults, iou_thr 0.55, no grid search), reported as an ensemble and test-time-augmentation
protocol, not a checkpoint, and not promotable.

## I2: ONNX export, Mac grader and CPU bench

Status: done. The export, the split-level parity and the processor bench are graded in
research/eyes-instrument-grading.md and summarised in the closing status section below.

The RT-DETRv4 trainer's `tools/deployment/export_onnx.py --check` at trainer pin 55fefaae, run
on the `345c30a5` config (opset 17, batch-1, patch
third_party/patches/rtdetrv4/0001-export-onnx-batch1-opset17.patch), records a sha256, after
which the Mac runs scripts/eval/eval_detector_v2_onnx.py (onnxruntime CPU, pycocotools,
per-camera clustered bootstrap) to give parity against the trainer's 0.8356 and CPU ms per
frame at 640, the first real number for that requirement. Then, internal only: one look each for `345c30a5` and the
init `238a3f65` on MVOR (CC BY-NC-SA; graded on the Mac only, never uploaded to any cloud
machine, never quoted externally), which sets the field to beat for any future real-footage
claim.

## I3: the training slot queue (one job at a time on the 5090)

1. **B1 seed 1** (pre-registered next spend): RT-DETRv4-S, --seed=1, about 13 h, peak
   26,521 MiB. Per-epoch curve into the eyes lane curve directory
   final_models/eyes/curves/, under a run name that carries the seed, as the seed 0 curve
   already there does.
2. **B1 seed 2 and the soup rider:** --seed=2 (about 13 h), then uniform-3 and greedy-3
   weight soups of the three seeds (tensors written to both 'model' and 'ema.module', with
   'updates' preserved), each scored with --test-only (about 48 s). The bar for any soup or
   seed claim: the best single seed plus max(0.0012, measured 3-seed spread). A seed below
   0.8353 revisits the provisional crown.
3. **B2-DINOv2T (gated, after the seeds):** re-enable the trainer's foundation-feature
   distillation with DINOv2-B/14 (Apache-2.0, PINS.yaml entry with sha256 at first download),
   teacher input 560x560 (the 40x40 grid matches the student's stride-16), loss_distill 5,
   micro-batched no_grad teacher, 15-minute dry-run VRAM gate of 30,500 MiB. It runs only if
   budget remains after the seeds; the adversarial review's point estimate is +0.001 (range
   -0.002 to +0.003), with roughly a 25% chance of clearing 0.8368.

Pod preparation was required first, because the workspace was blank after migration.
scripts/pod/setup_pod.sh and TRAINING.md are the record of how the pod is staged.

---

## Addendum, 2026-09-05

Mac CPU only; nothing here was run, trained or downloaded. Four corrections, all sourced to
**research/eyes-dinov2-teacher-arm.md**, which read the RT-DETRv4 tree at the PINS.yaml pin
`55fefaaed7efe2a5f72d0a18fd4e05965e35c292` (third_party/PINS.yaml, `repos.rtdetrv4.pin`) at
file and line. The declared terms above are left in place so the change is visible.
**Where this addendum and the text above disagree, this addendum governs.**

### (a) Teacher input is 280x280 at stride-32, which supersedes the 560x560 line

I3 item 3 says the teacher input is 560x560, with a 40x40 grid matching the student's
stride-16. The parenthetical premise is **false at this pin**: the student never taps
stride-16.

- `use_encoder_idx: [2]` over `feat_strides: [8, 16, 32]`
  (both keys in RTV4/configs/base/dfine_hgnetv2.yml, never overridden anywhere in the include
  chain, and confirmed in B1's own dumped config, outputs/pod_backup/train_rtv4_s0_final.log
  line 4, a run artefact under a gitignored output directory and not in version control)
  makes `encoder_idx_for_distillation = 2`, which is set in `HybridEncoder`
  (RTV4/engine/rtv4/hybrid_encoder.py).
  The projected student feature is therefore the **stride-32** map: **20x20 at 640 input**.
  [reconnaissance finding 1, research/eyes-dinov2-teacher-arm.md, headline section]
- The teacher has no 560 knob; it derives its own input size from the student's
  (`DINOv2TeacherModel`, RTV4/engine/rtv4/dinov2_teacher.py). With
  `target_downsample_factor: 32`, 640/32 = 20 and 20 by 14 gives **280x280 in** and **20x20 out**, an exact match the criterion
  does not resize at all. With `target_downsample_factor: 16` the teacher takes 560x560 in
  and emits 40x40 out, which the criterion then **bilinearly downsamples from 40 to 20**
  (`RTv4Criterion.loss_distillation`, RTV4/engine/rtv4/rtv4_criterion.py), so the extra
  resolution is averaged away.
  [reconnaissance section 6, the arithmetic table in research/eyes-dinov2-teacher-arm.md]

**Recorded: the B2 arm runs `target_downsample_factor: 32`.** It is written into
third_party/configs/rtv4_s_person_v2_dinov2t.yml, which is the only place the value lives,
because it is a config key rather than a launch flag. 560x560 is not deleted as an option,
but it is a *different question*, namely whether a higher-resolution teacher, average-pooled
to the student grid, is a better target than a low-resolution one, so it would be a **second
pre-declared arm at roughly double the B2 spend**, not this one. Making the student genuinely
tap stride-16 would need `use_encoder_idx: [1]`, which moves the AIFI transformer from S5 to
S4: four times the token count, a silently different meaning for the COCO-pretrained
`encoder.encoder.0.*` tensors, and a breach of the bake-off's minimal-delta law. The
reconnaissance recommendation, adopted, is **not to do it**.
[research/eyes-dinov2-teacher-arm.md, section 6]

### (b) The teacher runs full-batch in fp32, and the dry-run VRAM gate is the protection

I3 item 3 lists a micro-batched no_grad teacher as if it were a setting. **It is not a config
key and no such code exists at this pin**: it is a patch that would have to be written.
[reconnaissance finding 2, research/eyes-dinov2-teacher-arm.md, headline section]

- `teacher_model(samples)` is called **once on the whole batch of 32** under `no_grad`
  in `train_one_epoch` (RTV4/engine/solver/det_engine.py), and that call sits **outside**
  the autocast block, which opens only on the lines immediately after it, so the teacher
  runs in **fp32**.
- B1 seed 0 peaked at 26,521 MiB with no teacher (this document, I3 item 1), leaving about
  4 GB of headroom under the declared 30,500 MiB gate. The reconnaissance teacher-overhead
  figures are explicitly **estimates, not measurements**: about 1 to 2 GiB at 280x280 and
  about 3 to 5 GiB at 560x560, the latter plausibly *over* the gate.
  [research/eyes-dinov2-teacher-arm.md, section 8]

**Because those are estimates, the declared 15-minute dry-run VRAM gate at or below
30,500 MiB is the governing instrument, not the estimate.** It is implemented in
scripts/pod/eyes_chain.sh as a superset of sleeping 900 seconds and then reading nvidia-smi:
the peak is polled every 30 s across the window and the **maximum** is taken, because a
single sample at t=15 min can miss the peak. Over the gate, the run is killed and marked
`B2_DINOV2T_SKIPPED_VRAM_GATE`, which is **the declared outcome, not a failure**, and the
chain ends clean. Levers before any retry, least damage first: tdf 16 to 32, already 32;
autocast or `.half()` on the frozen teacher; micro-batching the teacher call. **Reducing
`total_batch_size` is not on that list**, because it changes the recipe and breaks
comparability with the B1 seeds. [research/eyes-dinov2-teacher-arm.md, section 8]

### (c) `--test-only` writes no detections, so the I1 dumps re-run the eval loop

I1 above describes the precision instrument as producing `--test-only` dumps. That phrasing
is misleading about the mechanism, and the difference matters because it is where the
instrument's numbers actually come from.

In **both** trainers at their pins, `--test-only` routes to `DetSolver.val()`
(the `--test-only` branch in train.py), and `val()` writes exactly one artefact,
`torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir/"eval.pth")` (the body of
`DetSolver.val` in rtv4 engine/solver/det_solver.py, and the identical body of the same
method in DEIM). `COCOeval.eval` is the **accumulate** output: the precision, recall and
scores ndarrays over the IoU by recall by area by maxDets grid. It contains **no boxes, no
image ids and no per-detection scores**. The detections *are* built, in pycocotools
results-list format (the results-list preparation step in rtv4
engine/data/dataset/coco_eval.py), but that list is a local in `CocoEvaluator.update()`,
handed to `loadRes` and dropped. **Nothing persists it, so there is no per-detection file to
convert.**

The I1 dumps therefore come from `scripts/detector/dump_detections_v2.py run`, which
**re-walks the trainer's own eval loop** (engine/solver/det_engine.py) with the
trainer's own config, dataloader, EMA weights and postprocessor, and writes that list to
disk. It is not a re-implementation: every tensor comes from the trainer's objects. Each dump
carries a `.sidecar.json` with the checkpoint sha256, per POLICY.md, rule 10. The full
evidence is the header of that script, verified 2026-09-05.

**Launch-time watch item, since closed.** `run` mode had not executed when this was written.
The declared proof was a comparison of the dumped detections against the training log's
`best_stat` for the same checkpoint, with an instruction to stop on a mismatch. It ran: the
four dumps graded in research/eyes-instrument-grading.md all carry `mode: run`, and the
comparison was closed off the pod, where the identity dump for `345c30a5` re-graded to AP
**0.8355529764** against the training log's `best_stat` of `0.8355529763877473`, a delta of
exactly **0.0**.

### (d) Soup family guard: B1 and B2 are different soup families

The I3 item 2 soup rider covers the **B1 seeds only**, which in the event were seeds 0 and 1,
because seed 2 was cut on 2026-09-06. A B2 checkpoint must never be
souped with them, and the tooling already refuses it. This section records *why*, so nobody
overrides the refusal thinking it is a bug.

- B2 builds a projector B1 does not have: `distill_teacher_dim: 768` constructs an
  `nn.Linear` from 256 to 768 in the encoder's constructor (hybrid_encoder.py), so a B2
  checkpoint carries `encoder.feature_projector.0.{weight,bias}`, that is **796 tensors, not
  B1's 794**. B1's checkpoint was confirmed on disk to have no such key
  (outputs/pod_backup/b1_best_stg2_345c30a5.pth, a run artefact under a gitignored output
  directory and not in version control, 794 tensors, read 2026-09-05).
  [research/eyes-dinov2-teacher-arm.md, sections 2 and 7]
- `scripts/detector/soup_v2.py` rejects a B1 and B2 mix on its key-set check
  (its `check_same_arch` function, exit code 4). **That is correct behaviour**, not an
  obstacle.
- Separately and independently, `--family` is **required** by the same script's `main` and is
  checked against recorded sha256 values in its `check_family` function, because DEIM
  `e98de401` and RT-DETRv4 `345c30a5` were measured to be tensor-layout identical, 794 keys
  with zero key-set, shape or dtype difference, so a *cross-architecture* soup would load,
  average and save silently
  while producing a meaningless model. The structural check is necessary but not sufficient.
  Do not add a quieter override than the existing banner-printing one.
- Related, and the reason the same fact bites at export time in the opposite direction:
  the trainer's `tools/deployment/export_onnx.py` calls `load_state_dict` with default
  `strict=True`, so
  **a B2 checkpoint exported through a `distill_teacher_dim: 0` config fails**. Export and
  score B2 through the B2 config; the projector is skipped at eval by the `self.training`
  guard in the encoder's forward pass (hybrid_encoder.py), so the ONNX graph is identical
  either way, and the acceptance-parity harness needs no work.
  [research/eyes-dinov2-teacher-arm.md, section 7]

### What did not change

The B2 arm's other declared terms stand: `loss_distill: 5` (the published S recipe's own
value, the `loss_distill` key in RTV4/configs/rtv4/rtv4_hgnetv2_s_coco.yml), DINOv2-B/14
Apache-2.0 with a PINS.yaml
entry whose sha256 is recorded at first download (now written: third_party/PINS.yaml
`repos.dinov2` and `checkpoints.dinov2_vitb14`), the 30,500 MiB gate, the condition that the
arm runs only if budget remains after the seeds, and the adversarial review's point estimate
(+0.001, range -0.002 to +0.003, roughly a 25% chance of clearing 0.8368). The call bar
0.8368 and the 0.8380 convincing margin are untouched. Dev-val remains synthetic and is never
quoted as deployment evidence.

---

## Closing status of the declared queue

Every item above has been run or cut. The numbers here are read from
research/eyes-instrument-grading.md, research/eyes-seed-spread-and-soup.md and
final_models/eyes/README.md, which are the records that hold them.

**I1, the precision instrument: run.** Four dumps were produced by
`scripts/detector/dump_detections_v2.py run`, identity and horizontal flip for both
`345c30a5` and `e98de401`, 540,000 rows over 1,800 dev-val images each, and graded on the Mac.
The precision, recall and false-positives-per-image table at score 0.3, 0.4 and 0.5 is in
research/eyes-instrument-grading.md; at the 0.4 demo point `345c30a5` reads precision 0.9317,
recall 0.9306 and 1.279 false positives per image, and the two detectors are
indistinguishable there. The optional fusion row was run at the declared defaults and lost:
minus 0.0423 AP for `345c30a5` (0.7933 against 0.8356) and minus 0.0427 for `e98de401`
(0.7914 against 0.8341). It stays a protocol row, not a checkpoint and not promotable.

**I2, export, grader and bench: done.** The ONNX export of `345c30a5`, sha256
`4026204967aa01a7680f7ce0c6169463d654a519f8093383a10537321f56fd49`, scores
**0.8355776016** on the full dev-val split against the trainer's 0.8355529764, a delta of
+0.0000246, which is the accumulated 32-bit floating point drift over the split. Per image,
every ONNX box has a twin in the checkpoint's dump at intersection over union at or above
0.9964 with a median box disagreement of 0.006 px. The processor bench reads mean 55.04 ms and
p50 **54.77 ms** for `session.run` in isolation at 640x640, batch 1, onnxruntime 1.28.0 CPU
execution provider. The authorised MVOR look for `345c30a5` was spent and is recorded in
research/eyes-instrument-grading.md, internal only, with its numbers not restated outside that
page; the look for the initialisation weight `238a3f65` remains unspent. It was blocked on
2026-09-05 for want of a checkpoint or graph on the grading machine, and that blocker no longer
holds: a graph of `238a3f65` now exists, ONNX
`b4c3ff04a89b4938bfa6fe39ee458ea00fa9b52e407954fa4b21af5d65e53390`, exported from the pinned
RT-DETRv4 source and run through this repository's `rtdetr_onnx` backend on the processor in
research/eyes-coupling-probe.md. The look is therefore unspent by choice rather than for want
of an artefact.

**I3 item 1, B1 seed 1: ran, not promoted.** Checkpoint `17362dca`, dev-val AP **0.8348**,
recorded as a seed replication. The per-epoch curve this item required was not produced:
`final_models/eyes/curves/` holds `detector_b1_rtv4s_seed0.csv` and the three DEIM files only,
and final_models/eyes/README.md records that no per-epoch CSV exists for `17362dca`. The
0.8348 is below the 0.8353 promotion bar named in item 2, and the record that followed leaves
the slot where it was: `345c30a5` at 0.8356 is the model of record and seed 1 is recorded, not
a displacement.

**I3 item 2, seed 2 and the soup rider: seed 2 cut, soup run and not promoted.** Seed 2 was
cut on 2026-09-06 before any seed-1 or soup score existed, so no three-seed soup and no
greedy soup exist. What ran is a uniform-2 soup over seeds 0 and 1, checkpoint `0d80cb4c`,
dev-val AP **0.830**, which is about 0.007 below the bar and below both of its own
ingredients. With seed 2 cut the spread the bar uses is the two-seed spread 0.00075, so the
bar applied was 0.8356 plus max(0.0012, 0.00075), that is **0.8368**.

**I3 item 3, B2-DINOv2T: ran to completion, not promoted.** The gated arm ran for 16.8 hours
with a frozen DINOv2 ViT-B/14 teacher at repository pin
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` and teacher weights sha256
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`. Checkpoint `9ecb4627`,
dev-val AP **0.8345** against the 0.8368 bar. The dry-run video-memory gate did not fire: peak
23,758 MiB against the declared 30,500 MiB. Measured against the forecast on this page, the
difference from seed 0 is minus 0.0011, inside the review's stated range of -0.002 to +0.003
and below its point estimate of +0.001.

**Lane conclusion.** Three independent full trainings, two plain and one with a foundation
teacher, land within 0.0011 of one another, and the teacher run sits below both plain seeds.
The conclusion of record is that DINOv2-B distillation adds nothing on this training set, that
0.8356 at `345c30a5` is the data-era ceiling for this corpus, and that the next lever for this
lane is data rather than architecture or teachers.
