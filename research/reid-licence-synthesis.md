# v2 re-identification training corpus: licensing and suitability

Dated 2026-09-02. Scope: the appearance-embedding training corpus for session-scoped, top-view and perspective closed-circuit television re-identification.

This is the 2026-09-02 lead report of the licence pass. `research/reid-data-licences.md` is the synthesis of record, and where the two differ that document governs.

Evidence tags: `[V]` primary source opened or measured first-hand; `[R]` reported by a secondary source that was opened directly; `[C]` inference.

**Coverage at the time of writing.** This report covered COCO and the permissive datasets. Classic re-identification sets, tracking sets, synthetic sets, codebases and the training-scale literature were open gaps when it was written. Those gaps are now closed by the specialist reports beside it, `research/reid-licence-classic-benchmarks.md`, `research/reid-licence-mot-family.md`, `research/reid-licence-synthetic.md`, `research/reid-licence-permissive-recent.md`, `research/reid-licence-open-images.md` and `research/reid-scale-literature.md`, and consolidated in `research/reid-data-licences.md`. Everything below is either first-hand verification or a verified finding of the one specialist pass that had reported at the time. A section still marked as a gap is a gap of this report, not of the record, and none is filled by inference.

The headline COCO measurement was made twice and independently: once against the on-disk copy of `instances_train2017.json` and once against COCO's own servers. The two counts agree exactly (68.87% NonCommercial, 32.27% NoDerivs, 26.07% clean), which makes it the strongest evidence in this document.

---

## 1. Registry bucket table

| Dataset | Bucket | Commercial training | Basis |
|---|---|---|---|
| **PhysicalAI-SmartSpaces** (MTMC 2024/2025/2026) | **TRAIN_OK** | Permitted | `cc-by-4.0` in HF card front-matter and API `[V]` |
| **COCO, licence-filtered to ids {4,5,7}** | **TRAIN_OK** (tighter than today) | Permitted | Per-image licence ids measured `[V]` |
| COCO, unfiltered | **TRAIN_OK** in the registry; recommend demote | 68.87% NonCommercial | Measured twice independently `[V]` |
| **TVRID** (ICPR 2026 top-view) | **TRAIN_OK**, low value | Permitted | Zenodo API `license: cc-by-4.0` `[V]` |
| **MEVA** | **TRAIN_OK** | Permitted | `mevadata.org` plus full CC BY 4.0 legal code `[V]` |
| **MEVID** (identity layer on MEVA) | **TRAIN_PROTOTYPE** since 2026-09-07 | Conflicting | GitHub `license: null`, no README terms `[V]`; the prototype grant and the conflict note are in `configs/datasets/mevid.yaml` |
| DanceTrack | EVAL_ONLY | Prohibited | "non-commercial research purposes only" `[V]` |
| PersonPath22 | EVAL_ONLY | Prohibited | CC-BY-NC-4.0 on data; Apache-2.0 is code only `[V]` |
| JRDB | EVAL_ONLY | Prohibited | CC BY-NC-SA 3.0 `[V]` |
| CrowdHuman | EVAL_ONLY | Prohibited | Licence text recorded in a registry entry since removed; the terms and the reason for the removal are kept in DATASETS.md under "Considered and not used" `[V]` |
| MOT17 / MOT20 | EVAL_ONLY | **No licence exists** | Page carries no licence text `[V]` |
| SportsMOT | EVAL_ONLY | Unknown | GitHub `license: null` `[V]` |
| BDD100K | EVAL_ONLY | Code BSD-3; data terms unverified | `[V]` code only |
| **DukeMTMC** and all derivatives | **LEGAL_HOLD** | **Never** | Retracted 2019, IRB breach `[V]` |
| Market-1501 / MSMT17 / CUHK03 | EVAL_ONLY | Assume prohibited | Canonical page DNS-dead `[V]`; terms unverified |
| LUPerson-derived weights | **LEGAL_HOLD** | Prohibited | "commercial usage is forbidden" `[V]` in-repo |
| BEDLAM / AGORA | LEGAL_HOLD | Prohibited | Non-commercial research licence `[V]` |
| ClonedPerson | LEGAL_HOLD | Unlicensed | GitHub `license: null` `[V]` |
| RandPerson / UnrealPerson | TRAIN_PROTOTYPE | Code Apache-2.0; **data terms unverified** | `[V]` code only |
| LTCC, PRCC, RegDB, SYSU-MM01, iLIDS-VID, PRID2011, MARS, PathTrack, TAO, WildTrack, PersonX, MOTSynth | Gap in this report, closed later | See the specialist reports | Not researched in this pass. LTCC through MARS are covered in `research/reid-licence-classic-benchmarks.md`, PathTrack, TAO and WILDTRACK in `research/reid-licence-mot-family.md`, PersonX and MOTSynth in `research/reid-licence-synthetic.md` |

---

## 2. The headline finding: COCO is 69% non-commercial

The measurement was taken on the on-disk `instances_train2017.json` and repeated independently against `images.cocodataset.org`. **The two counts agree exactly.**

| Aggregate (train2017, n=118,287) | Count | % |
|---|---|---|
| **NonCommercial (ids 1,2,3)** | **81,459** | **68.87%** |
| NoDerivs (ids 3,6) | 38,176 | 32.27% |
| **Commercial and derivative-OK (4,5,7,8)** | **30,836** | **26.07%** |
| US Government Work (id 8) | **0** | 0.00% |

**COCO's own terms, as published** `[V]`: *"The COCO Consortium does not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use."* Only the **annotations** are CC BY 4.0. The pixels arrive under no grant from COCO at all.

Two consequences specific to this project:

- **NoDerivs bites the pipeline directly.** CC BY-ND 2.0 section 3 grants reproduction and collective-work rights and **omits the derivative-works clause entirely** `[V]`. Cropping a person out and compositing them is the paradigm derivative. 32% of the feedstock carries it.
- **CC 2.0 has no cure period.** section 7 terminates automatically on breach; the 30-day cure in CC BY **4.0** section 6(b) does not exist in 2.0 `[V]`. COCO's annotations are curable; COCO's images are not.

**Measured on the render pipeline's own filters** (`iscrowd=0`, polygon segmentation, bbox height at or above 80px, the person-crop filter the render pipeline applied; that module is no longer in this repository) `[V]`:

- 130,332 person patches qualify, of which **only 40,177 (30.8%) come from commercial-permitting images**
- Dropping NoDerivs too leaves **33,697 patches from ids {4,5,7}**, or **21,883** under the strict {4,7} pool

**Why this matters more than it looks:** the v1 corpus consumed **22,144 training identities** `[V]` in-repo. The licence-clean pool of 21,883 to 33,697 patches is the same size as the pool already used. **The licence filter is close to free.** It removes the CQ-2 exposure at essentially no cost in identity count.

One caveat the completed specialist pass surfaced: id 5 (CC BY-SA), which is the 11,814-patch difference between the two pools above, raises the same ShareAlike-propagation question already open as **C-5** for CAVIAR. The strict {4,7} pool of 21,883 sidesteps it.

---

## 3. The corpus already held is a re-identification corpus

`PhysicalAI-SmartSpaces` is registered as a *detection* set. Measured on disk, it is a strong *re-identification* set that has not been used as one.

**Verified from the HF dataset card** `[V]` (`license: cc-by-4.0` in front-matter and API):

| Edition | Scenes | Cameras | **Person identities** | 2D boxes | Size |
|---|---|---|---|---|---|
| MTMC_Tracking_2024 | 90 | 953 | **2,481** | 135M | 217 GB (no depth) |
| MTMC_Tracking_2025 (a 10-scene slice is held locally) | 23 | 504 | 292 | 73M | 3.31 TB / ~70 GB no-depth |
| MTMC_Tracking_2026 | 28 | 353 | 901 | 33.2M | 2.94 TB / ~50 GB no-depth |

The ground-truth format is explicitly re-identification-shaped `[V]`: `<camera_id> <obj_id> <frame_id> <xmin> <ymin> <width> <height>`, where *"obj_id: Consistent numeric identifier for each object **across cameras**."*

**Measured on the local slice** `[V]`:

- **274 train-side identities**, 30.2M person crop instances, 17.9M at or above 48x96px
- At the **1 Hz cadence already extracted to JPEG on disk** (64,200 frames, every 30th frame): **597,903 crops at or above 48x96 and 434,494 at or above 64x128**, median **2,315 crops per identity**, median **22 to 41 cameras per identity**
- **Zero video decoding required**, because the exact frames are already extracted for the detection corpus
- Concurrent people per camera view: **median 8, p90 18, p99 23, max 30**, which is precisely waiting-room scale

**The honest weakness is the identity axis.** 274 identities against Market-1501's 751 `[R]`. A test of whether the identities are visually distinct clustered per-identity 3D body-size signatures and found **273 distinct signatures for 274 identities at 4dp, but only 199 at 2dp** `[V]`. That is consistent with continuous per-instance scale randomization, so **the test is inconclusive about mesh and texture reuse** `[C]`, and the registry's existing warning that the distinct mesh count is undisclosed stands. A visual audit of N crops per identity is the actual test and it is cheap. That audit ran on 2026-09-03 and its result is recorded in `research/reid-smartspaces-mesh-audit.md`.

**The remedy is already on the shelf:** MTMC_Tracking_2024 is the same CC BY 4.0 repository, **2,481 identities**, 90 scenes, and only 217 GB because it carries no depth maps. Its scene-type mapping is undisclosed in the card `[V]`, though the errata confirm scenes 071 to 080 are retail space.

---

## 4. Real footage: one clean option, one blocked

**MEVA, TRAIN_OK, verified twice.** *"All MEVA data is available for use under a CC BY-4.0 license"* `[V]`, with the full CC BY 4.0 legal code published at `mevadata.org/resources/MEVA-data-license.txt` `[V]`. 328 hours, indoor and outdoor, filmed at MUTC with *"a team of over 100 actors"* `[V]`, consented, paid and IRB-covered. Ethically the cleanest real corpus available.

**MEVID has no declared annotation licence, and this is a firm finding.** 158 identities, 598 outfits, 8,092 tracklets, 33 camera views `[R]`. `Kitware/MEVID` returns **`license: null`** from the GitHub API, and the README contains **no licence or terms section**, only IARPA/ODNI and USAF funding acknowledgements `[V]`. The annotations ship from the MEVA S3 bucket, but MEVA's CC BY 4.0 covers the *video*, not the annotation layer. This sharpens existing counsel item **C-6** from "verify" to "no license currently exists". The C-numbered items are defined in `research/v2-training-stack.md`, section 5; the counsel register carries this question as CQ-29 in `docs/COUNSEL_QUESTIONS.md`.

**Status of that gap.** The MEVID README publishes `mevadata@kitware.com` and Kitware sells support contracts `[V]`. Under the dataset acquisition rule in POLICY.md, section 3 the owner is not contacted, no licence clarification is requested, and the ambiguity is recorded in the registry and resolved by counsel at ship time. That is where it now sits. `configs/datasets/mevid.yaml` records status available, bucket TRAIN_PROTOTYPE and `commercial_use: conflicting` under a prototype grant decided 2026-09-07, so the 158 consented-actor identities train prototype-class weights only and the licence question stays open. The re-identification model of record, OSNet-x1_0 MEVID Phase A with checkpoint sha256 beginning `66623239` and ONNX sha256 beginning `60164a99`, was trained on this corpus; `final_models/reid/README.md` carries its measured numbers.

**TVRID, a real find of limited use.** ICPR 2026 competition set: **CC BY 4.0 on Zenodo** `[V]` (API `license: {"id": "cc-by-4.0"}`), open access, 86 to 88 identities (the paper says 86, Zenodo says 88 `[V]`, and the discrepancy is flagged), **four synchronized overhead RealSense D455 cameras**, 15 FPS, 640x480, with a pre-extracted person-crop archive. Exactly this geometry.

Two caveats decide its role. **The RGB track is saturated: the top three entries scored 100.0 / 100.0 / 99.84 CMC@1 and 100.0 / 99.98 / 99.92 mAP** `[V]`. And **no consent, IRB, ethics or GDPR statement appears anywhere in the paper** `[V]`, so it is real people with an undocumented consent posture. Use it as a small on-geometry sanity check, never as an exam, and flag the ethics gap.

**DukeMTMC, LEGAL_HOLD, permanently.** Taken down June 2019 after a Financial Times investigation; over 2,000 students filmed on campus in 2014 without informed consent; the authors acknowledged breaching Duke's IRB guidelines `[V]`. It continues to circulate through mirrors and *"continues to be used, even for commercial research"* `[V]`. **Any mirror's licence tag is meaningless.** This taints DukeMTMC-reID, DukeMTMC-VideoReID and any weight whose lineage passes through them, which is the same rule already applied to LUPerson.

---

## 5. Training codebases

Verified via GitHub API `[V]`:

| Codebase | License | Last push | Outcome |
|---|---|---|---|
| **torchreid** (`KaiyangZhou/deep-person-reid`) | **MIT** | 2026-01-09 | **Adopt.** Not archived, 4,905 stars, 163 open issues |
| FastReID (`JDAI-CV/fast-reid`) | Apache-2.0 | **2024-07-30** | Dormant for about 2 years; avoid for new work |
| **BoxMOT** | **AGPL-3.0** | 2026-09-02 | **Barred** by the AGPL isolation rule, POLICY.md, rule 5 |

**The decisive evidence is local, not on GitHub.** The v1 export provenance record of the earlier stack, written to `outputs/train/reid_v1_e12.onnx.json` under `outputs/`, which holds build outputs that are not in version control, records torchreid 1.4.0 at rev `f8cd150` running under **torch 2.13.0** `[V]`. That clears the torch 2.7 floor for Blackwell sm_120 support by a wide margin, so the sm_120 question is answered by a real run rather than by issue archaeology. OSNet is pure PyTorch; torchreid's only compiled component is the optional Cython rank library, which falls back to numpy.

**Recommendation:** keep torchreid, pinned to the canonical MIT source. POLICY.md, rule 11 already forbids the GPL-3.0-labelled `kadirnar/osnet` repack. The v1 recipe, that is OSNet x1_0, ID cross-entropy plus batch-hard triplet, PK 16x4 = batch 64, margin 0.3, lr 3.5e-4, random erasing 0.3 `[V]`, is standard and already implemented, with a torch-free data layer, since removed from the tree, that any new corpus can feed by emitting `{identity, crop}` manifest rows.

**Gap:** CLIP-ReID, TransReID, SOLIDER, centroids-reid, light-reid, and the zero-shot DINOv2/SigLIP-as-embedder question were not verified. The last of these is the most important unanswered question here, because an Apache-2.0 backbone that works zero-shot would moot the corpus decision.

---

## 6. Scale: what the task actually requires

**Gap:** the systematic scaling-ablation review did not complete. What follows rests on first-hand verification plus in-repo history, and should be treated as calibrated judgement, not a literature finding.

Three verified data points bound the problem:

1. **The gallery is small.** Median 8, p99 23 concurrent people per camera `[V]`. A session-scoped gallery is tens of people, not the 750 identities of a Market-1501 test split.
2. **The task is easy when clothing is fixed.** TVRID's 86-identity, same-session, overhead RGB track was solved to **100% mAP** `[V]`. Session-scoped re-identification without clothing change is close to saturated.
3. **The corpus was never the binding constraint.** v1 trained on 22,144 identities and 123,161 crops and still returned a **false-merge of 0.167** `[V]`. Its rank-1 figures were measured on a render holdout produced by the compositing harness and on a fisheye held-out set, both retired under POLICY.md, rule 1, so they are not quoted here.

**Working target:** roughly **1,000 to 3,000 identities at 50 to 200 crops each**, prioritizing viewpoint and lighting diversity per identity over raw identity count. MTMC_2024's 2,481 identities sits squarely in that band. Reference points: Market-1501 751 train identities and 12,936 images, MSMT17 1,041 and 32,621 `[R]`.

**The uncomfortable conclusion.** v1 and v1b both failed the same criterion, **false-merge at or below 0.01 against an observed 0.167, 16 times over**, while every other axis improved substantially. Both outcomes rest on **3 to 6 return probes** `[V]`, which the v1b write-up itself flags. **The binding constraint today is the instrument, not the corpus.** Buying a bigger corpus before widening the probe set spends GPU money to move a needle that cannot currently be read.

---

## 7. Ranked recommendation

### Option 1. SmartSpaces as the primary corpus, 2025 slice now, 2024 next

*Superseded on 2026-09-03. The mandatory audit this option proposes ran and failed: `research/reid-smartspaces-mesh-audit.md` found roughly 35 to 45 distinct avatar builds recycled under 231 identity labels, and `research/reid-data-licences.md` records the resulting demotion to positive-pair-only view-invariance pretraining and session-structure prototyping. Read what follows as the proposal that audit tested.*

Extract crops from the **1 Hz frames already on disk**: 274 identities by about 2,315 crops = **597,903 crops, zero decode, zero new fetch, zero new counsel questions**. Then extend the existing registry entry to a scene subset of **MTMC_Tracking_2024** for **2,481 identities** under the same CC BY 4.0.

**Pros:** licence-clean TRAIN_OK; no real people, so no consent or biometric exposure; correct geometry (elevated indoor multi-camera); identity labels consistent across cameras by construction; a same-instant multi-camera view of one person is exactly the viewpoint-invariance signal v1 was built to buy; `derived_from` stays valid since the parent is TRAIN_OK.

**Cons:** synthetic-only domain gap; the registry's **must-beat-real-baseline caveat** applies unchanged (across AI City 2023 to 2025 no top team shipped a synthetic-trained detector to real scenes); 2024 is 217 GB against a ~70 GB storage cap, so a scene subset is required; 2024's scene-type mapping is undisclosed; **mesh and texture diversity was unaudited when this was written and the body-size test was inconclusive**, which is the concern the 2026-09-03 audit noted above went on to confirm.

**Counsel flags:** none new. The CC BY 4.0 attribution obligation is already recorded.

**Do first, before any GPU spend:** extract about 20 crops for each of the 274 identities and inspect them. If meshes visibly repeat across identity labels, the metric-learning loss is being fed contradictory targets and the identity count is effectively far below 274.

### Option 2. Licence-filtered COCO compositing, the incumbent, corrected

Keep the pipeline; add a one-line licence filter to ids **{4,5,7}** (33,697 patches) or strictly **{4,7}** (21,883).

**Pros:** the pipeline, the PK sampler and the torch-free data layer all existed and were tested when this was written, though the crop-and-composite module named in section 2 is no longer in this repository; v1 proved the approach trains a usable embedder; identity count exceeds any real re-identification set that could legally be touched; **it strictly reduces current exposure at almost no cost**, since v1 only consumed 22,144 identities anyway.

**Cons:** composites are not real footage, and v1 and v1b both failed the false-merge ceiling on this data; id 5 (ShareAlike) reopens the C-5 propagation question; **attribution is currently absent and a CC 2.0 breach is uncurable**; about 21% of the safe COCO images have since drifted to ARR or NC on Flickr `[V]`, so the 2014 grant survives (CC BY 2.0 section 7 bars withdrawal `[V]`) but only if what it said can be proved, which means **snapshot and hash the licence metadata now**.

**Counsel flags:** CQ-2 narrows rather than widens. A new question arises on whether internal-only composites avoid triggering ND: CC's own guidance says ND and BY are *"triggered only when works or adaptations of works are publicly shared"* while *"the NonCommercial (NC) restriction applies to all uses"* `[V]`.

### Option 3. Hybrid: SmartSpaces pretrain, then licence-clean COCO composite fine-tune, with MEVA held for later

Pretrain on SmartSpaces for viewpoint invariance and fine-tune on filtered COCO composites for appearance diversity. MEVID's 158 consented-actor identities were later admitted for prototype-class training only, under the 2026-09-07 grant in `configs/datasets/mevid.yaml`; nothing trained on them ships until counsel resolves the licence question.

**Pros:** covers both weak axes, since SmartSpaces is identity-poor but viewpoint-rich while filtered COCO is identity-rich but viewpoint-synthetic; every component is TRAIN_OK today; MEVA remains available unlabelled through the existing teacher pipeline regardless of MEVID's status.

**Cons:** two-stage training doubles the experiment matrix on a single 5090; **with a 3-probe instrument the stages' contributions cannot be distinguished**; MEVID may never clear for commercial use.

---

## 8. Counsel flags and gaps

**New items:**

- **C-8, MEVID has no licence.** Not "unverified": GitHub reports `license: null` and the README states no terms `[V]`. The registry now carries the entry as TRAIN_PROTOTYPE with `commercial_use: conflicting` under a prototype grant decided 2026-09-07 (`configs/datasets/mevid.yaml`), and the licence question itself stays open for counsel at ship time.
- **CQ-25, COCO per-image licence filtering.** Adopting the {4,5,7} filter and an attribution manifest converts CQ-2 from an open advisory into a mitigated one. Worth recording.
- **CQ-26, TVRID ethics.** CC BY 4.0, but real people with no documented consent, IRB or GDPR posture `[V]`.

**Registry changes proposed here, with their status:**

- Applied. `configs/datasets/physicalai_smartspaces.yaml` carries `reid` in its `task:` list and records the measured identity and crop counts.
- Applied for weights. `scripts/guards/check_reid_weight_sources.py` refuses `dukemtmc` and its derivatives alongside `luperson`.
- Not adopted. `coco` remains TRAIN_OK in `configs/datasets/coco.yaml`, and the filtered subset proposed here as a `derived_from: [coco]` entry was never materialised; DATASETS.md records `coco_reid_filtered` under considered and not used.
- Not adopted. PersonPath22, JRDB and DanceTrack are recorded under considered and not used in DATASETS.md, with their licence terms and the reason each was refused, rather than carried as EVAL_ONLY registry entries.

**Gaps not filled, and not papered over:** classic re-identification terms (Market-1501, MSMT17, CUHK03, LTCC, PRCC, RegDB, SYSU-MM01, iLIDS-VID, PRID2011, MARS), closed later, see `research/reid-licence-classic-benchmarks.md`; synthetic sets (PersonX, MOTSynth, JTA, MTA, and the **GTA end-user-licence asset-provenance question**, which is the single most likely place for a hidden problem), closed later, see `research/reid-licence-synthetic.md`; diffusion-generated person sets and their generator licences; the training-scale ablation literature; zero-shot DINOv2/SigLIP as a re-identification embedder; **PeopleSansPeople's underlying character-asset provenance**, which is currently TRAIN_OK on the strength of Apache-2.0 code without the assets having been checked; and commercial data vendors, which for a clinical product may be worth more for the consent paperwork than the pixels.
