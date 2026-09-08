# v2 research conclusions: architectures, method, teachers, data (2026-08-31)

Combined conclusions of the research pass for the v2 top-view lane, whose scope is
recorded in POLICY.md, section 1. This document combines four research areas: detection
architectures, training methodology, pose and event verifiers, and non-fisheye
closed-circuit television datasets. Every load-bearing licence claim below was then
re-verified against the raw LICENSE file, not the repository page, before adoption-grade
status was granted.

Verification legend: **[V]** = the raw licence or file was fetched and read in this
pass. **[R]** = reported by repo page/model card, raw file
not fetched. **[C]** = paper/blog claim, not independently replicated. Anything [R]/[C]
that later becomes load-bearing gets a raw fetch at adoption time; nothing ships on [C].

Project policy is unchanged: the standing rules, the registry and licence buckets, the
counsel gates and the top-view process gates are all carried by `POLICY.md`, which is the
single source for them, and they bind every item here; `TRAINING.md` is the reproduction
record. Code under AGPL and GPL never enters, including for internal-only tooling, and
there is no separate path for internal tools.

---

## 1) The conclusions, per lane

### Eyes (person detection), the position before the bake-off: replace RTMDet-m with D-FINE trained under the DEIM recipe

Recorded outcome, after this document was written: the bake-off inverted this call. The
eyes model of record is RT-DETRv4-S person v2 seed 0, checkpoint `345c30a5`, ONNX
`40262049`, AP 0.50:0.95 0.8356 on person_v2 dev-val, against the DEIM arm's best seed
`e98de401` at 0.8341 (`final_models/eyes/README.md`). The primary named below is the arm
that lost; the challenger named below is the shipped detector.

**Primary at the time: D-FINE-S (M as the capacity check) trained with the DEIM v1 recipe.**
Both Apache-2.0 from raw LICENSE fetches [V]
(https://raw.githubusercontent.com/Peterande/D-FINE/master/LICENSE,
https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIM/main/LICENSE).
Why this pair leads the permissive field:

- D-FINE (ICLR 2025 Spotlight): S = 48.5 AP on COCO at 10M params and 25 GFLOPs, with a
  CNN (HGNetv2) backbone, the best CPU profile in the cohort, plus built-in GO-LSD
  self-distillation and an official ONNX exporter. AP is 0-to-1 box quality against
  human labels, and the eyes model of record holds AP 0.50:0.95 0.8356 on person_v2
  dev-val (checkpoint `345c30a5`, ONNX `40262049`), so these COCO numbers are context
  rather than a comparison: the v2 lane is graded on its own pinned split.
- DEIM (CVPR 2025) is a training recipe over D-FINE (Dense O2O matching plus a
  matchability-aware loss): DEIM-D-FINE-S = 49.0 AP, and it is the only cohort
  member with a published crowded-scene result, 57.5 AP on CrowdHuman at L size,
  which is +1.5 over baseline. Crowded, overlapping, seated people are exactly the
  waiting-room failure mode, and DETR-family models have no NMS stage to suppress
  genuinely overlapping boxes, which was the crowd failure of the previous RTMDet and
  NMS stack.
- Training cost is the lowest in class: the paper trains L-class to 53.2 AP in a
  single day on one 4090, and the per-experiment 5090 rentals here, at about $0.69 per
  hour or about $17 per day, cover S-class fine-tunes in hours.

**Challenger in the same bake-off, and the arm that won it: RT-DETRv4-S.** Apache-2.0,
stock unmodified text as a single LICENSE for the whole tree, confirmed by a raw LICENSE
fetch [V]; the repository is https://github.com/RT-DETRs/RT-DETRv4 and the pin is recorded
in `third_party/PINS.yaml` under `repos.rtdetrv4`. Highest
permissive small-model COCO AP (49.8) because Meta's DINOv3 foundation model teaches it
at training time through feature injection, at no inference-time cost. It is 9 months
old, from a new group, with no crowd evidence, so it enters as challenger, not primary.

**Conditional third: RF-DETR-Nano and RF-DETR-Small** (Apache-2.0 core; the XL and 2XL
plus-tier models are under Roboflow's PML-1.0 platform licence and never enter the
build). A community CPU measurement of about 180 ms per frame at 320 px, roughly 5
frames per second, sits at the floor of the budget here, so it is adopted only if its
DINOv2 features prove decisively more robust on oblique CCTV in the bake-off.

**Refused on licence, permanently:** DEIMv2, the strongest small model on the technical
numbers at 50.9 AP and 9.7M parameters, whose LICENSE.md states "Commercial Use of the
DEIMv2 Materials is not permitted" as published [V]; every Ultralytics-lineage YOLO
through YOLO26 (AGPL-3.0 [V]); YOLOv10, YOLOv12 and YOLOv13 (AGPL); YOLO-NAS
(non-commercial weights); YOLO-World (GPL-3.0).
The permissive frontier in 2025 to 2026 is entirely DETR-family.

**Standing risks:** (a) No public CPU-ONNX latency figure exists for D-FINE, DEIM or
RT-DETRv4. The deformable-attention GridSample operator is the unknown on x86, so the
in-house CPU benchmark in section 6 was the first engineering task and the gate this
document set before any training spend. RT-DETRv2's discrete-sampling option, which removes grid_sample, is the
fallback if GridSample proves slow. (b) Objects365-pretrained checkpoints carry a
dataset restriction for commercial use, so fine-tuning starts from COCO-lineage weights
only, and the checkpoint-provenance rule in POLICY.md, rule 8, records the lineage.
(c) The DEIM organisation (Intellindust) relicensed its v2 line as non-commercial and
sells exceptions; DEIM v1's Apache grant is irrevocable, and the v1 commit is vendored
and pinned now, with no upstream maintenance expected.

### Pose, the position before the retrain: keep stock RTMPose-m, instrument first, retrain only when the tripwire fires

Recorded outcome, after this document was written: the lane retrained. The joints model of
record is RTMPose-m person v2 seed 0 epoch 38, checkpoint `a0a035fe`, ONNX `6ba96285`,
keypoint AP 0.774115 on COCO val2017 with ground-truth boxes, against the body7
initialisation `e48f03d0` at 0.764499 (`final_models/joints/README.md`). The shipped joints
graph is that fine-tuned export, not stock weights.

RTMPose-m remains unbeaten at this deployment point: 75.8 AP COCO at 13.6M params with a
**measured 11.06 ms single-thread CPU ONNX latency**, the only candidate anywhere with
verified CPU numbers at this quality, Apache-2.0 end to end including the mm-free
`rtmlib` runtime [V]. Everything newer is worse here: ViT-backbone models cost an
estimated 3 to 5 times as much on CPU with no published benchmark; the 2025 occlusion
state of the art (BMPv2) is GPL-3.0; Meta's Sapiens is CC-BY-NC.

- **First action under that plan, at no training spend:** ship stock weights through rtmlib-style ONNX
  loading, and build the retrain tripwire, a per-track keypoint-confidence dashboard
  segmented by posture (standing, seated, lying) on the project's own cameras. The
  literature documents COCO-trained pose failing on overhead, aerial and heavily
  occluded views, and one 2026 occlusion study found supervised keypoints missing
  nearly half of falls, but mild-oblique CCTV has not been quantified anywhere. The
  dashboard is therefore the only decisive measurement, and it settles whether
  retraining is funded at all.
- **When the tripwire fires, as the plan then stood:** the mm training stack is unmaintained upstream (mmpose
  has gone more than 12 months without a release, and mmcv wheels for torch 2.11 on
  cu128 do not exist), so the fine-tune path is (1) a plain-PyTorch reimplementation of
  the SimCC head, with the backbone initialised from the stock checkpoint and validated
  bit for bit against rtmlib inference before any training, fine-tuned on annotated CCTV
  frames with a COCO replay buffer against forgetting; and (2) a ViTPose-B or ViTPose-L
  teacher (Apache-2.0, native Hugging Face training path) fine-tuned on the same data,
  then distilled into the RTMPose-m student in the DWPose manner, for the documented
  +1.7 to +4 AP at identical inference cost.
  Neither branch was taken. The lane trained inside the vendored pose repository at pin
  `759b39c1`, with the recipe `third_party/configs/rtmpose_m_person_v2_ft.py` installed
  into it, which is the path `TRAINING.md` records under "Joints: pose". There was no
  plain-PyTorch SimCC reimplementation and no ViTPose teacher. The packaging premise above
  was refuted as well: `research/pose-stack.md` records, tagged [V], a prebuilt `mmcv-2.2.0`
  wheel that is an exact match for torch 2.11.0+cu128, with the architecture chain closed
  through sm_120, so the mm stack was installable after all and the lane used it.

### Fall verifier, the position before training: MiniRocket-family features with a linear head

Recorded outcome, after this document was written: not adopted. The fall model of record is
fall_v3, checkpoint `3d2d17ee`, ONNX `67954c04`, window AUC 0.79276 on the multicam
all-cameras exam over 192 videos. It is a small temporal convolutional network, hidden
width 64, three blocks, dropout 0.15, over 8 track channels
(`final_models/fall/README.md`), which is the default this subsection proposed to replace.

The verifier seam and 64-step feature windows (box geometry + 17 keypoints +
confidences) are unchanged; the model inside the seam changes from a small multilayer
perceptron chosen by default to the family that leads small-data time-series
classification benchmarks:
**MultiRocket and MiniRocket random-convolution features with a ridge or linear head.**
In the largest controlled benchmark, the 2024 bake-off redux over more than 112
datasets, MultiRocket with Hydra matches the large ensembles at about one hundredth of
their compute, and it performs best at small-dataset scale, which is the scale of the
labelled fall windows here.

- Deployment is simple and licence-clean: the MiniRocket transform is a set of
  deterministic fixed convolutions, reimplementable in about 200 lines or exportable
  from tsai's PyTorch module (Apache-2.0, raw LICENSE verified in this pass
  [V]); aeon (BSD-3 [V]) is the training-side reference. Inference is microseconds per
  window on CPU.
- Backup if the linear ceiling appears past about 2 to 5k labelled windows: a small
  skeleton graph convolutional network from permissive lineage only, following ST-GCN's
  BSD-2 pattern or DeGCN (MIT), reimplemented rather than depended on. CTR-GCN
  (CC-BY-NC) and SkateFormer (an MIT tag contradicted by a research-only README) are
  refused on licence [V].
- Evaluation rule: subject-disjoint splits plus occlusion splits, always. Public fall
  benchmarks carry 4 to 10 subjects and are near-saturated, so their 99% figures carry
  no information about an out-of-distribution waiting room, and the product risk is
  false positives (children playing, bending, sleeping) that no public benchmark
  measures. An internal negatives-rich evaluation set is part of the exam build rather
  than an addition to it.

### Seizure verifier, the position before training: spectral hand-priors with a MiniRocket hybrid, and VSViG as the external lever

Recorded outcome, after this document was written: not adopted either. The seizure model of
record is convulsion_v2_real, checkpoint `715d172c`, ONNX `07c399be`, window AUC 0.69841 on
the wusahzu testsplit over 16 videos. It is the same small temporal convolutional network,
hidden width 64, three blocks, dropout 0.15, over 10 track and scene channels, with no
spectral and no MiniRocket features (`final_models/seizure/README.md`).

With 8 patients, added capacity is the binding risk and priors carry the work. The
convulsive signature is rhythmic limb oscillation, which lives in frequency space:

- **Primary:** band-limited oscillation-energy features (2 to 6 Hz limb and keypoint
  velocity spectra by Welch's method) concatenated with MiniRocket features, feeding a
  ridge or logistic head, under leave-one-patient-out cross-validation over the 8
  patients. Patient-disjoint numbers only, as always. The bar is the frozen model of
  record's measured window AUC of 0.69841 on the wusahzu testsplit, convulsion_v2_real,
  checkpoint `715d172c`, ONNX `07c399be`. Window AUC is the probability that a true event
  window outranks a quiet window, where 0.5 is chance.
- **The published precedent for this lane:** VSViG (ECCV 2024), a 1.4M-param
  skeleton-patch model built specifically for video seizure detection, Apache-2.0
  [V], reporting 5.1 s detection latency after EEG onset with no false detections on
  its epilepsy-monitoring cohort. **The released dataset (WU-SAHZU-EMU-Video) is the
  `wu_sahzu_emu` set already in the registry**, admitted 2026-08-26 as an
  EVAL_ONLY exam (14 patients, 33 seizures, counsel flag CQ-23) with the
  disjoint-patient trainsplit entry and the permanent exam-side patient allocation
  recorded in POLICY.md, rule 4. There is therefore no new data here; the new adoptions
  are (a) the VSViG architecture as the challenger to the spectral-hybrid primary, and
  (b) its probabilistic onset labelling, an exponential ramp from 0 to 1 between EEG
  onset and clinical onset in place of a binary window label, which rewards firing
  early and which the extraction configurations can emit against the same Label.xlsx
  onsets already parsed. The existing allocation rule, that exam patients never train,
  is untouched.
- **Teacher, never shipped:** an I3D two-stream video model following the Boyne 2025
  Epilepsia recipe (leave-one-patient-out AUC 0.988 on 25 patients), run offline over
  the archive to mine hard negatives and weak labels. Its documented cross-site
  degradation is the standing warning about the eight-patient regime here.
- **Calibration and abstention, both event lanes:** temperature scaling, then a
  split-conformal abstain band (MAPIE, BSD-3 [V]). With 8 patients the conformal
  coverage guarantee is weak under patient shift, so it is an engineering target rather
  than a guarantee.

### Teachers and auto-labellers (training-side only, nothing here ships)

Recorded outcome, after this document was written: the pseudo-labelling pipeline described
here was written and ran only as a probe on four public clips. It produced no label that
any model of record trained on, its filter threshold was never swept, and it has since been
removed from the repository together with the `transformers` dependency it needed
(`research/v2-training-sources.md`, section 2). The eyes lane trained on converted
SmartSpaces boxes with the distillation teacher block absent from the configuration rather
than disabled, so no teacher weights were ever downloaded (`final_models/eyes/README.md`,
"Recipe"). The licence findings below stand, and the ledger in section 4 rests on them.

- **Box teacher 1: MM-Grounding-DINO** (Swin-T and Swin-B): Apache-2.0 confirmed by a
  raw mmdetection LICENSE fetch in this pass [V]; 50.6 zero-shot COCO AP;
  weights downloadable; text-prompted pseudo-labelling, with prompts such as "person"
  and "person lying down", runs comfortably on one 5090.
- **Teacher 2: SAM 2 by default, with SAM 3 only under a documented exception.**
  Instance masks are the strongest tool for separating adjacent, overlapping and seated
  bodies where box teachers merge them into one region. Amended by the 2026-08-31
  simulation-to-real review: SAM 3's custom "SAM License" fails the
  permissive allowlist. Beyond the indemnification and passthrough terms read earlier,
  Meta may modify the agreement unilaterally with immediate effect (licence section 8) and
  termination requires deleting the materials and ceasing use (section 6); section 5(a) does confirm
  that derivative works, that is the pseudo-labels produced here, belong to the
  licensee [V]. SAM 2 is Apache-2.0 [V] and becomes the default mask teacher, and SAM 3
  is used only under an explicit documented exception if SAM 2's masks prove
  insufficient (counsel item C-2 in section 5, amended).
- **Voter 3 and tie-break:** OWLv2 (Apache-2.0 [R], from the Google Hugging Face card,
  with a raw fetch at adoption), which is purpose-built for pseudo-label self-training,
  and Florence-2 (MIT [V]) as the low-cost second opinion. Teacher boxes are fused with
  weighted box fusion, and cross-teacher disagreement is kept as a quality-assurance
  signal at no extra cost.
- **Event auto-labeller: Qwen3-VL-8B-Instruct**, Apache-2.0 on the official model
  card [V], with native video and timestamp grounding, and it fits the 5090 in BF16.
  Its labels are weak supervision with human adjudication, never gold, and they pass
  through the same provenance and registry gates as any label source. A per-size
  licence check is mandatory before switching sizes, because the Qwen2.5 generation
  mixed research licences into some sizes.
- **Refused teachers:** every API-only model (Grounding DINO 1.5 and later, T-Rex2,
  DINO-X), because hospital frames never leave the premises regardless of licence;
  Rex-Omni (a research-licensed Qwen base); Sapiens (CC-BY-NC, and a non-commercial
  grant contaminates teacher use for a commercial product under this policy); and
  Molmo 2, which carries an Apache tag but which Ai2 states is trained on datasets
  "subject to academic and non-commercial research use only", so it is skipped, since
  Qwen3-VL covers the need.

### Backbone policy

Released checkpoints are preferred over from-scratch pretraining, permissive first.
**DINOv2 and Perception Encoder (PE-Spatial)** are the approved foundation trunks:
DINOv2 Apache-2.0 [V]; PE Apache-2.0 confirmed in this pass through the GitHub
licence API (file `LICENSE.PE`) [V]. **DINOv3 is training-side only** pending counsel:
the raw LICENSE.md [V] grants commercial use and states that derivatives belong to the
licensee, with licence passthrough if DINO Materials or derivatives are distributed;
whether a distilled student counts as a derivative is the open legal question (counsel
item C-1). A reported display duty of the form "Built with DINOv3" was checked against
the raw licence in this pass and does not exist; only research publications
must acknowledge use. SigLIP2 (Apache [V]) serves curation and embedding, not
detection. In-domain MAE pretraining is a gated experiment only, because the evidence
for it comes from domains far from web imagery, such as industrial defects, rather than
people in rooms.

---

## 2) The training method (adoption order)

The method is built from what the 2024 to 2026 evidence supports at this scale:
one 5090, few gold labels, a large licence-gated unlabelled pool, CPU-ONNX students.

1. **Evaluation harness before any training.** Frozen v2 exam built
   first; 3 paired seeds minimum for decisions, 5 for headline claims; TIDE error
   decomposition (MIT [V]) on every exam run; clustered and bootstrapped error bars by
   camera and patient, since events cluster and naive error bars understate the spread;
   a label-error audit of the exam before trusting deltas (test sets average at least
   3.3% label errors in published audits). Measured seed spread on COCO-scale data is
   about 0.1 to 0.3 AP and will be larger on the small exam here. **No difference
   between two arms below about twice the measured noise floor is ever called a
   result.** This closes all four measurement defects of the previous lane in one move.
2. **License ledger.** Every dependency's LICENSE URL, fetch date and SHA recorded in
   the registry documents at adoption time: the raw file, never the repository page, as
   DEIMv2 and SkateFormer both showed. No continuous-integration involvement; it lives
   with the registry.
3. **Foundation-teacher offline pseudo-labelling** of the unlabelled CCTV pool
   (MM-GDINO, OWLv2, SAM 2 and Florence-2, fused with weighted box fusion; SAM 3 by
   exception only, section 7e). The filter threshold is swept as a
   first-class hyperparameter, since the OWLv2 self-training result is the canonical
   evidence that filtering is the main lever. A stratified 1 to 2% human
   quality-assurance sample estimates pseudo-label error before any training consumes
   it. Teacher precision and recall are measured on the labelled frames held here
   first, because zero-shot teachers degrade on oblique views and that measurement sets
   the ceiling on the whole move.
   Expected value from the literature, translated conservatively: +5 to +12 AP over
   gold-only training, with a wide range because no published result covers this exact
   viewpoint shift.
   Offline labelling is preferred over online Mean-Teacher semi-supervision because the
   online frameworks assume 4 to 8 GPUs and carry documented confirmation-bias
   fragility; at most 1 to 2 self-training rounds, ever.
4. **The student training core is the DEIM recipe as published**, adapted to one GPU by
   the linear learning-rate-to-batch scaling rule the configurations themselves state,
   with bf16 automatic mixed precision on Blackwell and the exponential moving average
   always on. A warmup-stable-decay schedule replaces cosine once data grows
   continuously: training extends when new footage lands and a decay branch is taken to
   release, and the checkpoint-anytime property suits a corpus that keeps
   growing. Schedule-Free AdamW is a one-day comparison against the noise floor, not a
   default swap. torch.compile gets one afternoon, never days.
5. **Occlusion-targeted copy-paste augmentation**, ablated against the noise floor:
   simple copy-paste is worth +1.4 to +2.5 AP in the literature including on strong
   models, the crowd-specific variant more than 2% in crowded scenes, and Occlusion
   Copy and Paste targets person-to-person occlusion, which is exactly this case.
   Strong-early and weak-late scheduling follows RT-DETRv2. Mild rotation, plus or
   minus 10 to 15 degrees, and perspective jitter for oblique views; no published
   evidence exists for oblique-specific augmentation, so this is where the ablation
   harness is required.
6. **Distillation:** GO-LSD comes at no cost inside D-FINE; large-teacher query-level
   knowledge distillation in the QSKD manner, worth +4 to +5 AP on weak students and
   discounted to +1 to +3 on a strong DEIM baseline, follows once the base recipe is
   stable; two-stage knowledge distillation in the DWPose manner for pose when
   that lane retrains.
7. **Data-centric loop:** embed the pool with SigLIP2 or DINOv2, deduplicate
   near-identical frames, since CCTV temporal redundancy is very high; stratify by
   camera, time and occupancy; spend human labels through simple uncertainty and
   diversity selection over foundation embeddings, because the 2026 evidence has
   bespoke active-learning systems performing worse than this at low budgets; and cap
   useful repetition of the small gold set at about 4 effective
   epochs, transferring the data-constrained scaling result with stated caution.
8. **Synthetic people for pose pretraining only:** PeopleSansPeople (Apache-2.0 [V])
   with cameras biased to oblique-high elevations, where the published few-label gain
   is +2.9 to +4.6 keypoint AP in exactly this regime. This is synthetic people in
   scenes, not acted medical events, so the bar on simulated or acted medical events in
   POLICY.md, rule 4, is untouched, and it still enters through
   a registry admission decision before use (owner call, section 5). BEDLAM stays out
   (non-commercial plus an anti-surveillance clause). No diffusion-generated people.
9. **Event heads:** frozen trunk features feed tiny heads (the verifier picks in section 1),
   real events only, clustered evaluation. No change to the rules seam.

### The frontier-lab paper trail

| Lab | Paper | What it gave v2 | Lands in |
|---|---|---|---|
| Google DeepMind | OWLv2 / OWL-ST self-training (NeurIPS 2023, arXiv 2306.09683) | teacher pseudo-labelling scales detection; the filter threshold is the first-class control (their 0.1 to 0.7 sweep) | section 2.3, the whole teacher pipeline |
| Anthropic | Adding Error Bars to Evals (2024, arXiv 2411.00640) | clustered SEs, paired seed tests, power planning | section 2.1, the noise-floor rule |
| Meta | Llama-3-style staged data curriculum | broad, then domain, then small high-quality last | section 2 ordering (COCO and SmartSpaces, then pseudo-labels, then gold) |
| Meta | DINOv2 / DINOv3 / SAM 3 / Perception Encoder | frozen foundation features at no training cost; concept-prompted masks | section 1 backbone policy, teacher 2, RT-DETRv4's injection pattern |
| Google DeepMind | SigLIP2 (2025) | curation and deduplication embeddings | section 2.7 |
| Google (Brain) | Simple Copy-Paste (CVPR 2021) plus crowd variants | +1.4 to +2.5 AP occlusion augmentation | section 2.5 |
| OpenAI / DeepMind | Kaplan scaling laws (2020) / Chinchilla (2022) / data-constrained scaling (NeurIPS 2023, arXiv 2305.16264) | quality over steps; about a 4-epoch repeat budget; and the negative result: no scaling-law fitting at n=2 model sizes | section 2.7 plus the not-worth-pursuing entries on muP and on scaling-law fitting at this size |
| OpenAI/MSR lineage | muP / Tensor Programs V (arXiv 2203.03466) | evaluated and rejected at this scale, with reasons | the not-worth-pursuing entry on muP |
| LLM-pretraining line | WSD / river-valley schedules (ICLR 2025, arXiv 2410.05192) | checkpoint-anytime training for continuously growing footage | section 2.4 |
| Microsoft | Florence-2 (MIT) | low-cost teacher tie-breaker | section 1 teachers |
| Alibaba Qwen | Qwen3-VL-8B (Apache) | the literal VLM in the plan: offline event auto-labeller | section 1 teachers |

Why no vision-language-model architecture is cloned: the models here run on CPU with no
cloud connectivity (POLICY.md, rule 7) at 10 to 20M params, three orders of magnitude
below any vision-language model. What transfers from the frontier method at that size
is the method itself, that is the data engine, teacher-to-student distillation,
schedules and evaluation statistics, which is what this section records.

**Not worth pursuing, on the evidence:** muP and scaling-law fitting at this size; FP8
and FP4 training; torch.compile as a load-bearing dependency; online DETR-SSOD
frameworks as the first move; from-scratch DINO-style self-supervised learning on CCTV;
research-grade active-learning acquisition functions; Muon, SOAP and Shampoo optimizers
for detection (watchlist only: no DETR evidence, and the AlgoPerf Shampoo result needed
a tuning harness that does not exist here); BEDLAM-lineage and diffusion people; and
anything AGPL in the toolchain, a verified hazard, since LightlyTrain, the obvious tool
for distilling DINOv2 into a smaller model, is AGPL-3.0 with a $5k per year commercial
licence, so the pattern is replicated with Apache-licensed parts instead.

---

## 3) Datasets (non-fisheye CCTV)

Every licence below was fetched and quoted from the primary page or file in the dataset
review of 2026-08-31 unless marked as reported. All admissions go through the registry
gates, the acquisition interlocks (`--yes` plus `VIVIDLY_ALLOW_FETCH=1`), and an owner
decision each; nothing below is admitted by this document. Acquisition is self-serve
anonymous download only, and no owner is contacted and no form is submitted, under
POLICY.md, section 3.

**Lean-start scoping (owner, 2026-08-31):**
the lists below are the licensed menu, not the shopping list. The first bake-off
pulls about 40 to 70 GB in total: the three exam sets (about 10 GB) plus a curated
SmartSpaces slice (hospital plus a few indoor scenes, about 30 to 60 GB, fetched scene
by scene). COCO already on disk covers generic volume. MEVA enters only if the week-1
teacher check passes, and then as a clip-level slice; Open Images only if a concrete
gap shows; VIRAT only after counsel; PeopleSansPeople renders only if the pose tripwire
fires. Every pull is sized to the experiment that needs it, with nothing fetched in
advance.

### Train-side candidates (TRAIN_OK-candidate, registry admission queued)

1. **NVIDIA PhysicalAI-SmartSpaces**, the anchor. CC BY 4.0 [V] on the official Hugging
   Face card. About 280 hours across about 1,800 simulated ceiling and wall security
   cameras in indoor spaces **including hospital scenes**, with boxes, track
   identifiers, camera calibration and
   depth. Synthetic humans (NVIDIA Omniverse): no consent risk, direct download, no
   gate. Exactly the target geometry; the only cost is the synthetic-to-real gap, which
   the teacher-pseudo-labelled real data below is there to close.
2. **MEVA**, the real-video pool. CC BY 4.0 [V] (mevadata.org), 328 hours of real
   elevated security-style footage, 11 of 29 cameras indoor, filmed with about 100
   paid, IRB-consented actors on a closed facility, the only large real corpus that is
   both licence-clean and ethics-clean. It has no native person boxes, which is
   acceptable: it is the designated **pseudo-labelling pool** for the section 2.3 teacher
   pipeline. (MEVID's 1.7M boxes over the same footage exist but are identity-labelled
   re-identification annotations; using them with identity fields stripped is counsel
   item C-6, not assumed.) Recorded outcome: MEVA itself was never admitted to the
   registry. The only MEVA-derived entry is `mevid`, bucket TRAIN_PROTOTYPE, status
   available, acquired on 2026-09-07 as three archives pinned by sha256 and used as the
   re-identification Phase A training corpus rather than as a detection pseudo-labelling
   pool (`DATASETS.md`, `configs/datasets/mevid.yaml`).
3. **VIRAT ground**, the rare real-CCTV set whose signed agreement explicitly
   permits commercial use ("lawful research and commercial purposes", quoted from the
   agreement PDF [V]). Outdoor only, about 8.5 h, person boxes included. Caveats: no
   redistribution, and Kitware may terminate at will with a destroy-copies duty, so
   counsel item C-4 comes before it becomes load-bearing. The agreement is a signed form,
   which the acquisition rule in POLICY.md, section 3 does not permit, so the set stays
   out and never reached the registry.
4. **Open Images V7 person subset** (with the COCO already held and LVIS masks), the
   generic-viewpoint volume filler. Annotations CC BY 4.0 [V]; images are Flickr with
   Google's explicit "verify each image yourself" disclaimer, so it is admitted only
   once an attribution and takedown posture is settled (counsel item C-7).
5. **PeopleSansPeople** (Apache-2.0 [V]) rendered at 20 to 60 degrees surveillance
   pitch, the elevated-pose gap closer (section 2.8, decision F-1), since no permissive
   elevated-pose dataset exists anywhere. Plus **CAVIAR** (CC BY-SA [V], 2003-era
   indoor corridor and lobby views) as a small real supplement, only if counsel accepts
   the ShareAlike question (C-5).

### The v2 detection exam (frozen before training, EVAL_ONLY entries)

- IndoorCrowd (2026): **excluded at admission time on 2026-08-31, with no bytes
  fetched.** Its FAIR Noncommercial Research License card states the licence
  "prohibits surveillance and any re-identification of individuals" [V], a use-based
  restriction that plausibly reaches a patient-monitoring product even as an
  engineering-only exam. Recorded in CQ-24; it re-enters only if counsel affirmatively
  clears it. The indoor exam surface was instead assigned on that date to MVOR, the
  held-out SmartSpaces val scenes and CrowdHuman, which was already on disk.
- **SOMPT22**: 6 to 8 m pole-mounted outdoor surveillance, MOT format, treated as
  non-commercial [V]. The outdoor elevated counterpart. Recorded outcome: SOMPT22 was never
  acquired and has no registry entry in this repository, because the linked folder is not
  publicly readable and the acquisition rule admits self-serve anonymous download only, so
  no bytes were fetched and it carries no exam surface (`DATASETS.md`).
- **MVOR**: 732 multi-view frames from a real hospital interior (an operating
  room), CC BY-NC-SA [V]. Small, and the only public probe of clinical-environment
  domain shift; it grades transfer and never trains.
- Legacy exams held at that date: MOT20, CrowdHuman-val, and footage collected by this
  project as it accrues. CrowdHuman's non-commercial terms were re-read word for word [V]
  and its registry entry was EVAL_ONLY permanently.

Both of those legacy entries were later removed from the registry, so neither carries an
exam surface in this repository: the indoor exam surface is MVOR and the held-out
SmartSpaces val scenes, and DATASETS.md, section "Considered and not used", records that
no v2 lane uses `crowdhuman` and that `mot20` was never used.

### Findings the sweep established

- **True hospital waiting-room imagery does not exist publicly.** The closest public
  artefact (LookDeep) released only aggregated prediction CSVs; the frames are PHI
  and stayed private. The waiting-room distribution (seated rows, wheelchairs,
  gurneys, children on laps, long-dwell occlusion) must come from synthetic scripting
  plus, eventually, a consented partner-hospital collection under IRB and BAA, which is
  a strategic owner decision (F-3), not a download.
- **Real indoor perspective CCTV with a commercial licence is effectively MEVA alone.**
  Everything else at the right viewpoint (MMPTRACK, Toyota Smarthome, PIROPO, HDA+,
  IndoorCrowd, MTMMC) is research-only or consent-scoped, so exam or quarantine at
  most.
- **Permissive elevated-view 2D pose data does not exist.** Render it
  (PeopleSansPeople) or negotiate BEDLAM commercially: BEDLAM's default licence
  prohibits surveillance use outright [V], so it is unusable here without a negotiated
  licence.
- **Never list, attractive on viewpoint and excluded on ethics:** DukeMTMC (terminated
  following its institutional review controversy), Brainwash, Oxford Town Centre,
  WILDTRACK (no licence and unconsented filming), Mall and Grand Central (unconsented
  public CCTV), and all GTA-derived sets (non-commercial plus Rockstar EULA taint). A
  hospital customer audits exactly this, so none of it ever enters, evaluation
  included. Live public webcams are never scraped: that is the Brainwash pattern with
  GDPR and biometric-statute exposure on top.
- Roboflow and Kaggle CCTV sets tagged CC BY carry an uploader-asserted licence only,
  until a per-set provenance audit says otherwise, and are treated as conflicting as a
  class.

---

## 4) The licence ledger (adoption-grade items)

| Component | Role | License | Status |
|---|---|---|---|
| D-FINE | student detector | Apache-2.0 | [V] raw |
| DEIM v1 | training recipe | Apache-2.0 | [V] raw; vendor/pin now |
| RT-DETRv4 | challenger detector | Apache-2.0 | [V] raw |
| RF-DETR core (Nano to Large) | conditional detector | Apache-2.0 | [V] repo; raw fetch if adopted |
| RTMPose / mmpose | pose student | Apache-2.0 | [V] |
| rtmlib | pose runtime | Apache-2.0 | [V] |
| ViTPose (repo) | pose teacher | Apache-2.0 | [V] repo; HF weight cards re-check per-card |
| MM-Grounding-DINO (mmdetection) | box teacher | Apache-2.0 | [V] raw |
| OWLv2 | box teacher voter | Apache-2.0 | [R] HF card; raw fetch at adoption |
| Florence-2 | tie-break teacher | MIT | [V] card |
| SAM 2 | mask teacher (default) | Apache-2.0 | [V] raw (2026-08-31 review) |
| SAM 3 | mask teacher, exception only | custom SAM License, fails the allowlist (unilateral modification section 8, terminate and delete section 6) | [V] raw; counsel item C-2 (amended) |
| DINOv2 | backbone/embeddings | Apache-2.0 | [V] raw |
| Perception Encoder | backbone alternative | Apache-2.0 (`LICENSE.PE`) | [V] GitHub licence API |
| DINOv3 | training-side teacher only | custom DINOv3 License | [V] raw; counsel item C-1; no display-duty clause |
| SigLIP2 | curation embeddings | Apache-2.0 | [V] |
| Qwen3-VL-8B-Instruct | event auto-labeller | Apache-2.0 | [V] card; re-verify per size |
| tsai | MiniRocket PyTorch module | Apache-2.0 | [V] raw |
| aeon | TSC reference impl | BSD-3-Clause | [V] |
| MAPIE | conformal calibration | BSD-3-Clause | [V] |
| VSViG code | seizure architecture and labelling scheme | Apache-2.0 | [V] repo; its dataset is the registered `wu_sahzu_emu` entry (CQ-23) |
| TIDE | error decomposition | MIT | [V] raw |
| PeopleSansPeople | synthetic pose pretrain | Apache-2.0 | [V] quoted; was registered, never materialised, entry since removed, see F-1 |

**Refused list (never enters, licence-verified):** DEIMv2 (non-commercial, as
published); Ultralytics YOLO at all versions including YOLO26 (AGPL); YOLOv10, YOLOv12
and YOLOv13 (AGPL); YOLO-NAS weights (non-commercial); YOLO-World (GPL); BMPv2 and
BBoxMaskPose (GPL); Sapiens (CC-BY-NC); CTR-GCN
(CC-BY-NC); SkateFormer (research-only README over an MIT tag); LightlyTrain (AGPL);
NVILA weights (CC-BY-NC-SA); the Llama vision line (EU multimodal carve-out plus
community terms); Gemma (custom terms with remote-restriction rights); Rex-Omni
(research-licensed base); BEDLAM (non-commercial plus anti-surveillance); and all
API-only teachers (PHI never leaves the premises, independent of licence).

---

## 5) Counsel items and owner decisions queued

- **C-1 (counsel):** the DINOv3 distilled-student question. Is a student trained on
  DINOv3 outputs a "derivative work of the DINO Materials" carrying passthrough? If
  yes or unclear, RT-DETRv4's own Apache weights (its authors ship them regardless)
  and the DINOv3-teacher experiments here stay training-side and internal only, which
  is the current default. Nothing blocks on this.
- **C-2 (counsel, amended 2026-08-31 on a raw read of the licence):** SAM 3 is demoted to
  exception only. That raw read surfaced a unilateral-modification clause
  (section 8, effective immediately) and a terminate-and-delete clause (section 6) on top of the
  indemnification and passthrough terms, so it fails the permissive allowlist outright.
  SAM 2 (Apache-2.0 [V]) is the default mask teacher and needs no counsel. C-2 matters
  only if SAM 2's masks prove insufficient and the SAM 3 exception is wanted.
- **C-3 (counsel, standing):** CQ-23 already covers the WU-SAHZU and VSViG clinical
  footage (commercial use of models influenced by it); no new dataset admission is
  needed, because the set is registered. Any v2 training use stays inside the existing
  permanent exam-side patient allocation recorded in POLICY.md, rule 4.
- **F-1 (owner):** registry admission for PeopleSansPeople synthetic pose
  pretraining (synthetic people, not medical events, so the bar on simulated or acted
  medical events in POLICY.md, rule 4, is untouched, while data admission remains an
  owner decision under project policy). Recorded outcome: the question is moot.
  `peoplesanspeople` was registered, never materialised, and its registry entry has since
  been removed; the synthetic corpus of record is `physicalai_smartspaces` (`DATASETS.md`,
  "Considered and not used").
- **C-4 (counsel):** the VIRAT agreement grants commercial use, but the licence is
  terminable at will with a destroy-all-copies duty; does "copies" reach trained
  weights? An answer is needed before VIRAT becomes load-bearing, and since it is
  optional volume, nothing blocks.
- **C-5 (counsel):** CAVIAR is CC BY-SA; does ShareAlike propagate to a model trained
  on it? The prevailing view is no, and the point is unsettled. The set is skipped
  unless cleared, since it is supplementary rather than structural.
- **C-6 (counsel + policy):** MEVID annotation licence verification, and sign-off
  that using its boxes with identity fields stripped satisfies the no-re-ID posture.
  Superseded on the re-identification lane: MEVID annotations trained the model of record,
  OSNet-x1_0 MEVID Phase A epoch 30, checkpoint `66623239`, ONNX `60164a99`,
  general-protocol mAP 0.204782 and rank-1 0.490506 on the MEVID test split
  (`final_models/reid/README.md`), under the prototype weight class and the conflict note
  recorded in `configs/datasets/mevid.yaml`. MEVA itself is not used at all.
- **C-7 (counsel):** Open Images/COCO Flickr substrate posture for a medical product
  (attribution/takedown pipeline), plus the DPIA-style review of biometric/privacy
  statutes that applies to any real-people training data regardless of copyright.
- **C-8 (counsel, sharpens C-6, from the 2026-09-03 re-identification research
  pass):** MEVID has no licence at all, rather than an unverified one: the GitHub API
  returns `license: null` and the README states no terms, and MEVA's CC-BY-4.0 covers
  the video, not the annotation layer.
  No clarification is requested: POLICY.md, section 3 bars contacting a dataset owner, so
  the contradiction is recorded rather than resolved and the conservative reading governs.
  The registry entry `configs/datasets/mevid.yaml` carries bucket `TRAIN_PROTOTYPE` and
  status `available` with the three archive digests recorded, and every weight the data
  touches stays in the prototype class. Full finding: research/reid-data-licences.md and
  the research/reid-licence-*.md set. The same pass supersedes part of the premise
  behind F-1: PeopleSansPeople's 28 humans are RenderPeople scans that Unity states it
  had no right to redistribute, RenderPeople's own terms prohibit computer-vision
  research use without written consent, and the set carries no identity labels, so the
  F-1 admission question is moot for re-identification and close to moot for pose.
- **F-2 (owner, standing default is no):** whether the GPL ban ever relaxes for
  internal-only offline tooling, which would unlock BMPv2 as an occlusion
  pseudo-labeller. The recommendation is to keep the blanket ban.
- **F-3 (owner, strategic):** partner-hospital data collection (IRB scope, consent
  signage against opt-in, BAA, on-premises annotation). The dataset sweep's central
  finding is that no download substitutes for it; timing is the owner's call.
- **F-4 (owner):** registry admissions batch for the section 3 picks. **Partially executed
  on 2026-08-31 as the lean-data batch, whose acquisition rule is now recorded in
  POLICY.md, section 3:** PhysicalAI-SmartSpaces (TRAIN_OK, 10-scene RGB slice) registered
  with the licence text recorded word for word, alongside MVOR as an EVAL_ONLY exam
  (CQ-24), with SOMPT22 considered for the same exam role under that decision; IndoorCrowd
  considered and excluded (no-surveillance licence clause, no bytes fetched). Of those two
  exam candidates only `mvor` reached status available; SOMPT22 was never acquired and has
  no registry entry in this repository (`DATASETS.md`). Still open under F-4:
  MEVA (gated on the teacher check that was never run), VIRAT (blocked by the signed
  agreement, C-4), and the Open Images person subset. Acquisition runs through the
  standard interlocks.

---

## 6) The engineering queue as it stood on 2026-08-31 (before any training spend)

The five items are recorded as they were written. Each one carries its outcome where
this repository records one.

1. **CPU-ONNX latency benchmark** of D-FINE-N, D-FINE-S and RT-DETRv4-S at 640 and
   960 px on the target x86 part (onnxruntime and OpenVINO). No published measurement
   exists; the GridSample operator is the unknown; this gate decides the detector
   shortlist. Cost: $0, CPU only.
2. **Teacher check:** MM-GDINO, with OWLv2, precision and recall on the labelled
   perspective frames already held, which sets the ceiling on the pseudo-labelling move
   before any pool labelling runs. Outcome: the teacher pipeline ran only as a probe and
   has since been removed from the repository (`research/v2-training-sources.md`, section
   2), so this check never ran; the eyes lane trained without a teacher
   (`final_models/eyes/README.md`).
3. **The v2 detection exam is built, not adopted** (the 2026-08-31 simulation-to-real
   conclusion, section 7e): the primary frozen exam is annotated from scratch on MEVA (CC BY
   4.0 [V], the only licence-clean real substrate) at about 2,000 to 3,000 frames
   across at least 60 to 80 distinct camera-recordings, visible-region convention, 100%
   double-annotated; recording diversity, not frame count, is the binding constraint
   (4.3x precision at equal cost). The sets named on 2026-08-31 are secondary probe
   surfaces, each labelled for what it can and cannot say: MVOR (732 frames, a
   real-hospital-interior probe, never the exam), SOMPT22 (real elevated CCTV, but
   outdoor), and SmartSpaces val (a synthetic surface and selection validation only, so a
   synthetic score is never quoted as deployment evidence). Until the MEVA exam exists,
   probe results carry a label marking them a probe rather than an exam. The exam is
   frozen before the first training run, as project policy requires.
   Outcome: the MEVA exam was never annotated, and SOMPT22 was never acquired and has no
   registry entry in this repository. The eyes headline of record, AP 0.50:0.95 0.8356 for
   checkpoint `345c30a5`, ONNX `40262049`, is measured on person_v2 dev-val, 1,800 images
   and 33,755 boxes from the SmartSpaces slice with Warehouse_005 held out by scene. That is
   the synthetic selection surface this item describes, and `final_models/eyes/README.md`
   records it as the pinned selection split with no exam consulted, so the number is a
   selection and comparison number and not deployment evidence.
4. **Noise floor:** 3 seeds of the unmodified DEIM-D-FINE-S recipe on the initial v2
   training set; the measured spread becomes the significance bar for every later
   comparison. Cost: about 3 GPU-days, roughly $50. Outcome: measured. The DEIM three-seed
   spread is 0.0012, and the two RT-DETRv4-S seeds land 0.00075 apart, seed 0 `345c30a5` at
   0.8356 against seed 1 `17362dca` at 0.8348 (`research/eyes-seed-spread-and-soup.md`).
5. **Seizure v2kp re-extraction** (`emit_keypoints: true`): CPU only, and it unblocks
   the spectral features for the new verifier regardless of every other decision. Outcome:
   done. The re-extracted track tree is named as `outputs/events/wu_sahzu_tracks_v2kp` by
   the `configs/events/lopo_limbgate*.yaml` configurations; `outputs/` is a build location
   that is not in version control.

Projected first-milestone spend under this plan, since superseded by the recorded runs,
through a first bake-off conclusion (D-FINE with DEIM
against RT-DETRv4, 3 seeds each plus a pseudo-label sweep): roughly $150 to $250 in
5090 hours at about $0.69 per hour. Every run gets its dry-run VRAM gate and written
cost projection first, per `TRAINING.md`.

## 7) Deployment research (2026-08-31): tracking, CPU deployment, temporal and alerts, night

Four research passes on the non-training aspects, under the same evidence law ([V] primary
fetched, [R] repository-reported, [C] unreplicated, [M] measured on real code in the same
pass, [V-local] re-verified against the installed package or raw source before adoption,
[V-negative] a primary-source search that found nothing, [V-a] a figure or claim taken
from the cited paper's own text, not reproduced here, and [V/C] a source that was fetched
whose figure is its own unreplicated claim). This section records only the
conclusions that change the plan.

### 7a) Tracker: ByteTrack out, BoT-SORT without CMC in, on a verified safety defect

- **The defect, [V-local] re-verified in the installed `trackers` package
  (site-packages, the low-score selection in bytetrack/tracker.py):** the Roboflow
  ByteTrack port carries
  no low-score floor. `low_indices = np.where(~high_mask)` admits every detection below
  the high threshold into the second association pass, including a raw DETR's
  low-confidence tail (D-FINE and DEIM emit exactly 300 boxes, with no NMS and no
  threshold [V]; about 97% of the sub-threshold tail is background per D-FINE issue
  #300 [R]). Upstream ByteTrack
  hardcodes `scores > 0.1` [V]; the port dropped it. BoT-SORT's port has the floor
  ([V-local] the same selection in botsort/tracker.py reads
  `(confidences > 0.1) & (~high_mask)`) and
  restricts the low-score pass to currently-tracked tracks, so a low-confidence
  detection can never bind to a long-lost track.
- **Measured consequence [M]:** with a raw 300-query tail and a motionless occluded
  person, the floor-less ByteTrack track was dragged 515.6 px across the frame,
  about 5.7 person-widths, which is exactly the fall scenario. OC-SORT, BoT-SORT and
  C-BIoU: 0.0 px.
- **Buffer units [V-local] (the buffer arithmetic in the tracker base class,
  base.py):** `lost_track_buffer` counts frames at 30
  frames per second, so the effective duration in seconds is the value divided by 30
  regardless of stream frame rate. The default of 30 is one
  second of identity memory, and every tracker at defaults loses a still person at 10 s
  [M]. A waiting room needs minutes: 1800 (60 s) to 9000 (300 s), values 20 to 100
  times outside the library's own tuned range, so the burn-in here must price the
  re-attachment risk (no published precision-against-buffer curve exists [V-negative]).
- **Adopted:** switch the shipped tracker to `BoTSORTTracker` with `enable_cmc=False`
  (static cameras; CMC off saves 7.0 ms/frame [M]), which is the same Apache-2.0
  library and one line; port OC-SORT's velocity-noise damping
  (`process_noise[4:,4:] *= 0.01` [V]),
  the entire mechanism behind its 3.5x lower stationary drift and 180 s identity
  survival [M]; add an explicit score floor, starting at 0.1, between detector and
  tracker; deduplicate the DETR's per-class duplicate boxes, since its flattened top-k
  can select one box under several labels [V]. Candidate original work: a zero-velocity
  (ZUPT-style) gate for stationary tracks, which no MOT paper has published
  [V-negative] and which the measured drift justifies.
- **Rejected:** every end-to-end transformer tracker (1 to 10 s per frame on CPU,
  CUDA-only deformable operators, non-commercial weights only [V]); appearance-based
  auto-binding inside the tracker, which POLICY.md, section 4 bars beyond the tracker's own
  short re-association window, and which is also worse on long gaps, since
  motion loses 0.8 IDF1 where appearance loses 6.8 when the
  occlusion limit stretches from 30 to 120 frames [C, MotionTrack Table 4]; StrongSORT
  and BoxMOT (GPL and AGPL [V]). MOT benchmark numbers carry a caution: MOT20 deletes
  static people from ground truth, up to 41% of the humans in one sequence [V], so the
  public benchmarks never scored the population that matters here. Metrics: AssA, AssRe
  and AssPr plus a longest-correct-run survival curve; MOTA, MT, ML and Frag are barred
  from the dashboard, being identity-agnostic or about 99% detection-driven [V].

### 7b) CPU deployment: the envelope, the one decisive operator, and the flag hazard

- **No CPU latency figure exists for D-FINE, DEIM or RT-DETRv4 anywhere**
  [V-negative; one circulating figure of 91.3 ms on CPU traced back to a paper whose
  table is GPU-only, so the snippet was fabricated]. The only real x86 anchor at about
  42 AP is PicoDet-L-640 at 62.5 ms on
  an i7-10750H with 12 threads [V]. D-FINE-N matches its AP at 2.4x fewer but denser
  FLOPs, which is plausible and unproven on CPU; measuring it is the first week-1 item.
- **The decisive operator is GridSample** (deformable attention): the ORT x86 kernel is
  scalar C++ with no SIMD, parallel over about 32 channels, while OpenVINO
  just-in-time compiles an AVX-512 or AVX2 kernel parallel over output elements
  [V, both source trees]. Benchmark
  native OpenVINO against ORT FP32 first, then profile GridSample's share of CPU time:
  under 10% the runtime choice does not matter, at 40 to 80% it is the whole decision.
- **Quantization flag hazard:** ORT `quantize_static` defaults to S8S8, which has no
  optimized x86 GEMM outside Arrow Lake and Lunar Lake, so not on Xeon SPR, EMR or GNR
  and not on Zen 4 or 5
  [V, MLAS platform.cpp dispatch]. Use QDQ with `activation_type=QUInt8,
  weight_type=QInt8` and backbone-only INT8, since GridSample has no INT8 kernel in
  either runtime [V]; expect 1.2 to 1.5x, not 2 to 4x [V, Lenovo controlled bench].
  FP16 and BF16 on ORT CPU do nothing, because there are no x86 kernels [V], and
  AMX-BF16 needs a Xeon with OpenVINO.
- **Scheduling:** one shared session per model, narrow workers of 1 to 2 threads,
  `allow_spinning=0`, `set_denormal_as_zero=1`, one process per camera with disjoint
  core pinning if isolating [V]; never batch across cameras, since Intel's CPU plugin
  itself refuses to auto-batch [V] and an 8x batch measured +15% [R]. The 2.2x
  multi-session contention problem is real and unfixed upstream (ORT #17011 [V]).
- **Ingest:** pull the cameras' substream at about 640x360, about a 9x decode
  reduction, which lands at detector-native size; the FFmpeg default LGPL build decodes
  H.264 and HEVC legally, since the GPL parts are optional filters [V]; integrated-GPU
  QuickSync offload uses an MIT-licensed driver stack; avoid F-suffix parts without an
  integrated GPU in the machine specification; use dual-channel RAM.
  Envelope: 8 cameras at 3 frames per second on a 16-core machine is plausible only
  with the detector interval at N=5, the RTMPose authors' own shipped default [V], and
  pose gated on active tracks. Elastic degradation is load-bearing, not an
  optimization.
- **Not worth pursuing:** the ORT OpenVINO execution provider (a four-minor-version
  downgrade), NCNN and MNN, anything from Ultralytics (AGPL), DL Streamer (LGPL plus
  patent disclaimers in writing), dynamic quantization (measured slower than FP32
  before VNNI [V]), quantization-aware training (0 to 3 mAP over post-training
  quantization at full fine-tune cost [V]), and single-class head shrinking (0.76% of
  parameters).

### 7c) Temporal robustness and alert policy: the spectral null, n=8 statistics, the false-alarm budget

- **The spectral null, the highest-severity finding.** A 5-frame moving average at
  15 frames per second places an exact spectral null on 3.0 Hz [C, exact filter math],
  the centre of the clinically validated 2 to 6 Hz convulsive band. The one
  video-validated seizure system monitors exactly that band at 32 frames per second,
  with more than 4 s of persistence, 94% sensitivity over 1,661
  nights and 0.05 false alarms per night [V, Epilepsia 2020]. Savitzky-Golay and
  SmoothNet cost -24 to
  -40 dB at 3 Hz plus 533 to 1067 ms of structural lag [V/C]. Adopted: two-path
  keypoints. Raw keypoints, band-passed 1 to 7 Hz rather than low-passed, feed the
  event verifiers, and smoothing exists only on the dashboard display branch.
  **No smoother enters the alert path.** Capture at 30 frames per second is worth
  considering if CPU allows, because 6 Hz sits close to the Nyquist edge at 15 frames
  per second.
- **Detector flicker:** no temporal variant of any RT-DETR-family model exists
  [V-negative]. Adopted: tracklet-level evidence accumulation with decrement
  (the MR2-ByteTrack rescore rule; naive score carrying dropped precision from 62 to
  59, while accumulate-and-decrement recovered above baseline at under 0.2% of detector
  cost [V]), reimplemented from the paper, since the reference repository has no
  licence [V].
- **Statistics at eight patients:** patient-level conformal coverage above 88.9%
  is mathematically unavailable at k=8 (Dunn et al., Theorem 9: nontrivial only if
  k > 1/alpha - 1 [V]), and frames do not rescue it, since record-wise
  cross-validation inflates about 2.3x in the
  published record and the effective n falls to k/ICC, about 8 [V, Saeb et al.].
  Adopted: state coverage as it is (alpha > 1/9), camera-conditional, since a camera is
  an atom and a new patient is not [V, Vovk Proposition 4]; the Brier score with a
  bootstrap confidence interval instead of expected calibration error, which is
  undetectable below 10% miscalibration at n<500 [V]; a sealed patient holdout; and
  `crepes` (BSD-3) rather than TorchCP (LGPL). An M=5 ensemble of tiny verifiers is the
  one shift-robustness lever CPU can afford [V, Ovadia].
- **False-alarm budget, the product number:** the ship target is at most 0.1 false
  alarms per camera-day, with a hard ceiling of 0.5 and a stretch target of 0.05 on the
  worst camera, because false alarms cluster: 81 of 81 fell in 6 of
  22 subjects in the best published system [V]. Certification: a silent burn-in of
  8 cameras over 14 days with at most 3 false alarms certifies below 0.07 false alarms
  per camera-day at 95%
  one-sided [C, exact Poisson]; the sequential probability ratio test is used only for
  post-deployment regression watch. Positive predictive value must
  clear about 0.5 to 0.7 or staff stop responding [V, Huegli and Manzey]. The missing
  variable is the waiting-room true-event base rate, which is unpublished
  [V-negative], so the burn-in measures it first.
- **Cascade design corrections from real fall data** (227 filmed real falls, Lancet
  2013 [V]): only 3% of real falls are slips, 41% are low-momentum weight-shift
  failures, and 12% happen while sitting down. Adopted: discriminate on post-event
  state (floor-level torso plus an immobility dwell of 30 to 60 s; the clinical
  long-lie threshold starts at 60 minutes, so this dwell is conservative [R]),
  de-weight impact-spike features, and never
  exclude sit-down kinematics. Persistence and delay gates are the strongest published
  false-alarm control, at 75 to 89% alarm reductions with no measured safety cost [V].
- **Degradation training boundary:** codec-degradation augmentation on the detector and
  pose lanes is validated, recovering +0.24 mAP at JPEG Q15 with no clean-accuracy cost
  [V], and it is never applied to the motion or spectral verifiers: the one published
  test collapsed the motion stream from 38.96 to 9.26 at CRF 40 [V]. It also does not
  generalise across corruption families [V, Geirhos], so each family is simulated
  separately. The operating specification carries a bitrate floor, with the knee at
  about JPEG Q15 or H.264 CRF 40 [V], and B-frames off.

### 7d) Night and infrared: the night mode here is the benign kind, and pose is the fragile stage

- **Magnitude:** matched-condition night costs about 4 LAMR (EuroCity Persons,
  controlled [V]), and an RGB-pretrained detector applied to active near-infrared with
  no adaptation lost
  0.053 mAP, 7% relative [V, RGB-NIR stereo], because security-camera night mode
  is active near-infrared, that is well-exposed monochrome with edges and texture
  intact, rather than thermal. Nearly
  every large infrared degradation figure in the literature is long-wave infrared
  thermal, which is the wrong modality
  [V]. Occlusion costs far more than darkness, about 38 miss-rate points against 4 to 7
  [V, NightOwls]. The larger
  hazard is the mixed distribution together with the day-to-night cut-filter switch
  mid-stream [V, TJU-DHD], so the transition is tested and the mode flag is considered
  as a model input.
- **Adopted augmentation stack** (albumentations, MIT): random grayscale
  desaturation, the published near-infrared bridge, worth +3 rank-1 on a real
  near-infrared benchmark on its own
  [V]; illumination attenuation with alpha in [0.6, 0.9] plus Gaussian sigma of about
  0.01, worth +12.4 pp mAP cross-domain with no target data [V, EAT 2026]; Brooks
  shot and read-noise sampling for the high-gain far field [V]; and clean data stays
  mixed in.
- **Rejected: enhancement preprocessing in the pipeline.** In the regime closest to
  this one, where the detector has seen the domain and the image is already adequate,
  measured gains are zero
  to negative: two of three enhancers reduced accuracy on ExDark against a
  low-light-trained detector
  [V, MAET]. It also costs CPU that is not available, and all three candidate
  repositories fail the
  licence gate outright (Zero-DCE is CC-BY-NC; SCI and Retinexformer carry no licence
  [V]).
- **Pose is the night bottleneck, not detection:** same-model drops of 52 to 90% for
  pose under severe low light against 20 to 25% for detection [V, ExLPose and DEKR];
  no active near-infrared pose benchmark exists at all [V-negative]. Adopted: the event
  cascades fall back to box and motion evidence when keypoint confidence drops, rather
  than firing on degraded keypoints, and the posture-segmented pose-confidence
  dashboard, the tripwire in section 1, gains a night and day split.
- **Data position:** no indoor, near-infrared, permissively licensed person dataset
  exists [V-negative]. Night robustness is measured on the project's own cameras,
  burn-in first and engineering second, and the partner-hospital collection (F-3)
  remains the only full remedy.
  NightOwls, LLVIP, KAIST and NTU are all non-commercial or unlicensed, so exam
  quarantine at most.

### Queue additions from the 2026-08-31 deployment research (folded into section 6's gates)

1. Tracker swap, score floor and buffer configuration on the replay pipeline; re-run
   the rules burn-in on the equivalent negative footage (v2 clips) with the new tracker.
2. CPU benchmark arms on day zero: ORT against OpenVINO at FP32 on D-FINE-N and
   D-FINE-S, GridSample time share, S8S8 against U8S8, 1 to 8 concurrent sessions with
   and without spinning, and substream against
   main-stream decode cost. PicoDet-L is the calibration control, and p50, p95 and p99
   are measured at sustained thermal state.
3. Two-path keypoint plumbing (raw -> verifiers, smoothed -> display) + the 1-7 Hz
   band-pass; verify the seizure verifier's input path carries NO smoothing.
4. RTMO-s CPU benchmark at 1, 2, 4, 8 and 16 people; no published measurement exists,
   and the result could restructure the whole pose budget.

### 7e) Simulation to real, and the exam build (the final 2026-08-31 pass)

Recorded outcome for the exam build: the from-scratch MEVA exam was never annotated, and
the eyes lane's headline is measured on person_v2 dev-val, the held-out SmartSpaces scene
Warehouse_005, which `final_models/eyes/README.md` records as the pinned selection split
with no exam consulted. The box convention and leakage rules below did land: the converter
pinned by `tests/test_convert_smartspaces.py` writes the visible-region convention this
subsection settles.

- **The exam is built, not adopted.** All four candidate exam sets fail as the primary
  exam: SOMPT22 (non-commercial and outdoor), MVOR (non-commercial and 732 frames),
  IndoorCrowd (non-commercial, with labels generated
  by SAM 3 and GroundingSAM, which are the same teacher families used here, so grading
  a teacher-distilled student on them is circular and shared teacher blind spots become
  invisible), and
  SmartSpaces val (synthetic, and so structurally unable to measure the
  simulation-to-real gap) [V].
  The primary exam is annotation from scratch on MEVA (CC BY 4.0, verified against the
  published text [V]; the existing MEVA boxes are activity and re-identification
  annotations rather than exhaustive person labels, so
  frames are labelled from scratch), about 2,000 to 3,000 frames from at least 60 to 80
  distinct camera-recordings, at about 8 to 15 people per frame, giving 20 to 35k
  instances, which supports about plus or minus 2 AP absolute and
  paired-model resolution well under 0.5 AP. Simulation findings that size it:
  video frames are not independent, so naive confidence intervals understate by 3 to
  6 times and the bootstrap is taken by
  recording; recording diversity beats frame count by 4.3x at equal cost; and the stock
  101-point COCO AP has a floor of about 0.5 AP on paired comparisons, so all-points AP
  is used or differences above 0.5 AP are required [C, simulation]. MEVA is one site
  with a fixed actor cast, and Caltech to CityPersons shows 8.7x cross-site degradation
  [V-a], so the exam is labelled within site and a second cross-site exam, on the
  partner-hospital footage under F-3, is planned rather than optional.
- **Box convention: visible region** (the smallest box on visible pixels, unioned
  across disconnected parts). Six reasons, strongest first: every input held here is
  already visible-convention (SmartSpaces `2d_bounding_box_visible`, COCO
  mask-derived, Open Images as published [V]); the CityPersons authors declined to
  define amodal boxes for seated people, on the ground that no uniform alignment can be
  applied [V-a], and
  a waiting room is seated people; amodal human-to-human agreement is about 0.60 IoU
  against 0.88 for
  visible, so an amodal exam measures annotator noise; a convention mismatch hides at
  AP50 and destroys AP75, with a band at 25 to 50% occlusion scoring AP50=1.000 and
  AP75=0.000 [V-a]; D-FINE's distribution head sharpens onto image boundaries and
  DEIM's matchability-aware loss suppresses learning on chronically low-IoU matches,
  both adverse to
  amodal targets [V-a]; and torchvision's SanitizeBoundingBoxes, called twice in the
  default D-FINE and DEIM pipeline, silently deletes boxes extending past the frame, so
  an edge-seated amodal box becomes a training negative [V-a, empirically verified].
  Mitigations in the protocol: a minimum scored height of 25 px, ignore-regions for
  crowd slivers, and a per-instance ordinal visibility grade (above 90, 70 to 90, 50 to
  70, 30 to 50, and 30% or below) as
  non-scoring metadata. Corrected number: the convention-mismatch cost is about -11%
  median relative AP. The -45% figure in circulation came from a misread table, where
  the 0.220 was a flower-class cell [V].
- **Simulation-to-real recipe, ranked with measured effect sizes:** (1) label 500 to
  1,000 real frames and fine-tune after synthetic pretraining with all layers unfrozen
  (+10 to +29 AP in the
  low-real regime; freezing the backbone cost up to 13.5 pp [V-a]); (2) teacher
  pseudo-label quality dominates (+15.0 mAP from a stronger labeller against +3.5 from
  feature alignment [V-a]); (3) randomise renderer appearance and match geometry
  (lighting randomisation is worth +6.1 AP, while wrong geometry is severe, at 48.9
  against 5.2
  [V-a]); (4) recall-first low-threshold auto-labelling, at about 0.2 to 0.4 rather than
  0.8; (5) weighted box fusion
  (+0.75 to +3.4 mAP); then DINOv2 feature alignment, copy-paste, and two-stage
  training only, since mixed batches measured 3.6 AP worse than pretraining followed by
  fine-tuning [V-a].
  AI City leaderboards are not simulation-to-real evidence: the 2024 winner trained
  nothing, the 2023 winner swapped to a real-data detector for the one real scene, and
  the only top-three team that fine-tuned on synthetic placed last of the three [V-a].
- **Skipped, after checking the evidence:** frequency-domain adaptation (no detection
  experiments in the source paper), the 0.618 golden-ratio mixing rule (tabular
  statistics, no vision), heavy codec augmentation at normal CCTV bitrates (about 1% F1,
  while the Q15 and CRF 40 knee from section 7c still bounds the operating specification; both
  hold, in different
  regimes), flying distractors, and Semi-DETR self-training for now (an 8-GPU recipe
  for +1.8 to +3.6). Demographic note: the person-detection skin-tone gap is a
  localization-precision effect, near zero at IoU 0.5, while the age gap is the large
  one, with children undetected 20.14% more often [V-a]. A waiting room has children,
  so the exam is stratified by age where visible.
- **Tooling:** CVAT (MIT; 1,585 SPDX headers audited across the repository, all MIT
  [V]) with SAM 2 in the labelling loop. **CVAT's bundled YOLOv7 serverless function is
  never deployed** (GPL-3.0 [V]). cleanlab and ObjectLab serve as a safety net only, at
  15 to 50% precision and missing up to 66% of errors [V-a]; capture-recapture on the
  double-annotation
  passes bounds what both annotators missed. A teacher box-convention probe runs before
  any fusion: composite an opaque occluder over the lower fraction f in
  {0, 0.2, 0.4, 0.6} of a visible
  person and track the bottom edge. The unoccluded frame is its own amodal ground
  truth, so the probe is causal [V-a design].
- **Leakage rule for the v2 splits:** split at camera-viewpoint level and then session
  level, never by frame or clip; at most 1 exam frame per 5-minute clip; never
  interpolate exam boxes;
  deduplicate on person crops rather than whole frames, because fixed-camera
  whole-frame embeddings saturate above 0.95; perceptual hashing and SSIM are at chance
  as near-duplicate detectors (AUROC about 0.5), so
  self-supervised embeddings are used instead (0.92) with a 0.90 flag threshold and a
  0.95 drop threshold, calibrated on the project's own
  footage; and the exam is deduplicated against itself, since a 21% near-duplicate
  prevalence was found in a real curated set [V-a].
