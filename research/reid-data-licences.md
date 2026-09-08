# v2 re-identification training corpus: research synthesis

2026-09-03. Synthesis of eight research reports (one lead report and seven
specialist reports, run 2026-09-02 and 2026-09-03) on the licensing, ethics, scale
requirements, and recipe for the v2 session-scoped re-ID embedder, the
re-identification lane. The full reports are kept in
the `research/reid-licence-*.md` set, plus `research/reid-scale-literature.md`, which
carries the SCALE report; every claim here traces to one of them, and each
of them tags claims `[V]` (primary source opened by that report), `[R]`
(secondary), `[C]` (inference). Where the lead report declared a gap, the
specialist report that later filled it is named. This is not legal advice; counsel
items are routed by the conventions in `docs/COUNSEL_QUESTIONS.md`, which holds the
CQ-numbered items, while the C-numbered items live in `research/v2-training-stack.md`
section 5.

Report key, each label naming the file that holds the report: LEAD =
`research/reid-licence-synthesis.md`; CLASSIC =
`research/reid-licence-classic-benchmarks.md`; MOT =
`research/reid-licence-mot-family.md`; SCALE = `research/reid-scale-literature.md`;
SYNTH = `research/reid-licence-synthetic.md`; COCO = `research/reid-licence-coco.md`;
OI = `research/reid-licence-open-images.md`; PERM =
`research/reid-licence-permissive-recent.md`.

---

## 1. Recommended bucket assignment (consolidated)

Every gap the lead report left is now filled. The basis column names the report
carrying the exact licence quote. These are recommended assignments, not a transcript
of the registry: of the datasets named below only three are registry entries, and they
carry `configs/datasets/physicalai_smartspaces.yaml` at TRAIN_OK,
`configs/datasets/coco.yaml` at TRAIN_OK and `configs/datasets/mevid.yaml` at
TRAIN_PROTOTYPE.

### TRAIN_OK (commercial training permitted)

| Dataset | License | Notes | Basis |
|---|---|---|---|
| **PhysicalAI-SmartSpaces** (HF acquisition only) | CC-BY-4.0 | Primary corpus; caveats in section 2 are mandatory reading | LEAD, MOT, SYNTH |
| **COCO filtered to licence ids {4,5,7}** | CC-BY/BY-SA/no-known 2.0 | Registered as its own `derived_from: [coco]` entry; an attribution manifest is required; id 8 has zero images | COCO |
| **Open Images V7** | imgs CC-BY-2.0 (100.00% verified), anns CC-BY-4.0 | ~1.03M person boxes in train; attribution metadata ships complete; snapshot the CSVs now (14.6% link rot) | OI |
| **MEVA** | CC-BY-4.0, full legal code published | Consented paid actors, IRB, licence disclosed in the consent briefing itself | LEAD, MOT, COCO |
| **TVRID** | CC-BY-4.0 (Zenodo API) | 86 to 88 top-view IDs; ethics gap raising CQ-26; RGB track saturated (100.0 CMC@1), so it is a sanity check only, never an exam | LEAD, PERM |
| **CHIRLA** | CC-BY-4.0 (HF + PMC) | 22 IDs / 7 cameras / 7 months / clothing changes / 963,554 identity boxes; the best long-horizon signal legally available; gated (contact info) | PERM |
| **Visym CAP + PIP-175k** | CC-BY-4.0 | 780 paid consenting collectors, elevated third-person view, tracks around primary actors | PERM |
| **WiseNET** | CC0 | 77 tracks, written consent, indoor multi-camera; a validation set, too small to train | MOT |
| **SDG-SynHuman** (NVIDIA) | OpenMDW-1.1 ("without restriction", outputs unencumbered) | 4,050 human assets, no boxes/IDs; an asset pool, not a labelled corpus | SYNTH |
| VIRAT | explicit commercial grant | Legally OK, technically useless for re-ID (no global identity namespace); termination-with-derivative-destruction clause referred to counsel | MOT |

### TRAIN_PROTOTYPE (permissive-but-flagged)

| Dataset | Why not TRAIN_OK | Basis |
|---|---|---|
| RandPerson, restricted to the 5,114 random-UV and original-clothing identities | Apache-2.0 code; data grant unstated; the filter drops the 2,886 of 8,000 IDs that wear textures from scraped web images, leaving the 5,000 random-UV identities plus the 114 with the original clothing models; MakeHuman's bundled assets are CC0, but its community clothing assets sit outside that grant and are unenumerated (SYNTH section 3) | SYNTH, MOT |
| PersonX | MIT on repo, but models "textured by scanning real people" and ship outside the MIT'd repo; scenes withheld | SYNTH |
| COCO unfiltered | 68.87% NC / 32.27% ND measured twice independently, so a demotion for re-ID use is recommended here; not adopted, `coco` remains TRAIN_OK in `configs/datasets/coco.yaml`, and the filtered subset above was never materialised | LEAD, COCO |
| **MEVID** | CC BY 4.0 covers the underlying MEVA video only; the MEVID annotation release ships no LICENSE file, so `commercial_use` is recorded as conflicting and the conflict is carried rather than resolved. Registered on 2026-09-07 as bucket TRAIN_PROTOTYPE, status available, under a prototype grant (`configs/datasets/mevid.yaml`) | LEAD, MOT, PERM |

### EVAL_ONLY (benchmarks only, nothing trains)

DanceTrack, PersonPath22 (data CC-BY-NC-4.0 + anti-biometric request), JRDB
(CC-BY-NC-SA-3.0), CrowdHuman, MOT17/MOT20 (site is HTTP 410; recovered terms
CC-BY-NC-SA-3.0, now no grant at all), SportsMOT (NC + broadcast footage the
authors never owned), BDD100K (members-only commercial class), SoccerNet
(NDA-gated), Market-1501 / MSMT17 / CUHK03 / MARS / iLIDS-VID / PRID2011 / LTCC
/ PRCC / Celeb-reID (CLASSIC: zero of 16 classic sets cleanly usable; "no
licence" is default-deny, not permission). SYSU-MM01 is the single classic set
with a drafted commercial-approval path (iSEE, and the only one with signed
subject consent): a route that could be opened, not a dataset to touch.

### LEGAL_HOLD / deny-list (never fetch, never train, purge on sight)

| Dataset | Reason | Basis |
|---|---|---|
| **DukeMTMC + every `Duke*` derivative** | Retracted 2019, IRB breach, institution-level takedown; mirrors' licence tags are void | CLASSIC, LEAD |
| **LUPerson / LUPerson-NL** + every weight pretrained on them (SOLIDER, PASS, PersonViT, HAP, TransReID-SSL) | "commercial usage is forbidden" (README is the entire licence); 4M to 10M YouTube-scraped non-consenting people; HF mirror's MIT tag is fabricated | CLASSIC, SCALE |
| **All GTA-V-derived** (MOTSynth, JTA, MTA, GPR/GPR+/FineGPR, VC-Clothes, GTA-Human) | Rockstar ToS section 2.2/2.3/5.4/6.1(3) independently prohibit; four separate clauses; active enforcer | SYNTH |
| **ClonedPerson, UnrealPerson** | DeepFashion contamination ("non-commercial research purposes only" reaching "any portion of derived data"); ClonedPerson's paper misstates DeepFashion as Apache-2.0 | SYNTH, MOT |
| **BEDLAM, AGORA** | MPI licence prohibits "commercial, pornographic, military, or surveillance" use by name; AGORA extends the bar to training models for it | SYNTH, PERM |
| **CCVID** | Upstream FVG terms breached three ways; source imagery since privacy-retracted (FVG-B face-blurring) | CLASSIC |
| **SLP** (in-bed pose) | NC + "not intended for use... for incorporation into a product", a named clinical-product exclusion | PERM |
| PeopleSansPeople binaries | 28 RenderPeople humans Unity had no right to redistribute (RenderPeople ToS section 4.2 prohibits CV research use without written consent); no identity labels anyway | SYNTH |
| MALS, CCUP, OmniFall, SynBody, NTU RGB+D, URFD*, Project Aria | research-only / NC / no grant (details in PERM section 2) | PERM |

*URFD = the existing `ur_fall` exam entry, already on the evaluation side in the
fall lane; its NC licence is one more reason it must never train.

MEVID sat on this deny-list when this synthesis was written on 2026-09-03, on the reading
that MEVA's CC BY 4.0 does not reach the annotation layer, which is counsel item C-8. On
2026-09-07 it was registered instead as bucket TRAIN_PROTOTYPE with status available
(`configs/datasets/mevid.yaml`), and the re-identification model of record, OSNet-x1_0
checkpoint `66623239`, ONNX `60164a99`, was trained on it in the prototype weight class.
The undeclared annotation licence is carried in that entry as an open counsel item rather
than resolved, and the actor check-in photographs package is barred outright under
POLICY.md, rule 2 and is absent from the entry's acquisition urls.

---

## 2. The corpus decision (ranked)

### #1: SmartSpaces as primary corpus, **demoted 2026-09-03 after the audit failed**

> **Audit result (2026-09-03, research/reid-smartspaces-mesh-audit.md):**
> caveat 4 below was executed, all 231 identities on the 5 detector-train scenes
> eyeballed via camera-spread contact sheets, and it failed: the corpus is
> roughly 35 to 45 distinct avatar builds recycled under 231 identity labels, with
> look-alike pairs inside single scenes (002:1 vs 002:7). Identity-discriminative
> embedder training on SmartSpaces is excluded; what survives is positive-pair
> view-invariance pretraining and session/tracker structure prototyping only.
> MTMC_2024's 2,481 IDs inherit the suspicion, so sample-audit before any fetch.
> The plan below is retained as the record of what was proposed; #2 and #3 now
> carry the embedder lane.

Measured on the local slice (LEAD, all `[V]` on-disk): 274 train-side
identities; 597,903 crops at or above 48x96 at the 1 Hz frames already extracted for the
detector corpus (zero decode, zero new fetch); median 2,315 crops/ID across a
median 22 to 41 cameras/ID; concurrent people per view median 8 / p99 23, which is
waiting-room scale. Scale-up: MTMC_2024 = 2,481 IDs / 90 scenes / 217 GB
(no-depth) under the same CC-BY-4.0, which sits squarely in the 2,000 to 3,000-ID
saturation band (section 4).

Mandatory caveats, each with an owner:

1. **Acquisition-path rule (registry and fetcher):** the same bytes carry
   CC-BY-4.0 on HuggingFace and a "no commercial or production purpose"
   agreement via the AI City challenge portal (MOT `[V]`). Acquisition is
   HuggingFace only, with provenance documented, and no registration for the
   challenge portal is made. This wording goes into the registry entry exactly as
   the source states it.
2. **NVIDIA responsible-AI counsel flag (escalate to counsel):** the card's
   Responsible-AI metadata says the dataset "is not authorized for use in
   biometric, facial recognition, demographic profiling, or surveillance
   systems targeting real individuals" and names patients among affected
   groups. CC-BY-4.0 is irrevocable and field-of-use-unrestricted, so this is
   stated publisher intent rather than a copyright bar, but it names this
   product category. Escalate before any shipped weight; a session-scoped
   embedder over synthetic people trained for leave-and-return continuity is
   arguably not "targeting real individuals," and that argument is one for
   counsel to make.
3. **Cosmos-Transfer scenes are a re-ID hazard (corpus builder):** several
   scenes are CT2.5 re-renderings NVIDIA discloses as "single-view rather than
   multi-view-consistent, so the same object may look subtly different across
   simultaneous camera views" (SYNTH `[V]`; the registry entry records the 2026
   edition's re-render pairs 001/003/005/007/021/024 from its admission-time
   card read). For a cross-camera identity objective that is label noise by
   construction. Exclude CT2.5 re-renders from any re-ID manifest (they may
   stay in detection manifests, subject to the existing leakage pairing rule).
4. **Mesh-diversity audit before GPU spend (owner, about an hour):** the lead
   report's body-size-signature test was inconclusive (273 distinct at 4dp, 199
   at 2dp, consistent with per-instance scale randomization). Extract ~20 crops
   for each of the 274 IDs and eyeball for mesh/texture reuse across identity
   labels. If meshes repeat across labels, effective identity count is far
   below 274 and the metric loss gets contradictory targets.
   **Done 2026-09-03 and failed: see the demotion banner above and
   research/reid-smartspaces-mesh-audit.md for the full record.**

### #2: compositing feedstock, migrate to Open Images V7; COCO survives only as the corrected filtered subset

The incumbent COCO composite pipeline gets a one-line licence filter and a
demotion. Measured (COCO `[V]`, computed from COCO's own JSON, cross-validated
three ways): train2017 has 64,115 person images, 69.20% NC, 32.35% ND; the
commercial-and-derivatives pool {4,5,7} is 16,513 images / 69,372 crops (id 8
is declared but used by zero images). Under the render pipeline's own quality
filters (iscrowd=0, polygon seg, at or above 80px) that pool is 33,697 patches, or 21,883
under the strict {4,7} pool that sidesteps ShareAlike (LEAD `[V]`). CC 2.0 has
no cure period, so an uncured attribution failure terminates permanently, and
about 21% of the safe images have since drifted to restrictive display on Flickr
(the 2014 grant survives section 7 irrevocability, but only where the original wording
can be proved): **snapshot and hash the licence metadata now**, and build the
attribution manifest via `flickr.com/photo.gne?id={photo_id}` before anything ships.

Open Images V7 dominates COCO as feedstock (OI `[V]`, full enumeration): 100.00%
of all 1,910,098 boxed images carry CC-BY-2.0 (zero NC/ND/SA), attribution
fields (Author, AuthorProfileURL, Title, OriginalLandingURL) are 0.00% missing,
and the person class alone is ~1.03M boxes / 248K images in train, about 15 times COCO's
clean person pool of 16,513 images / 69,372 crops, with compliance reduced to a
template render. Caveats: no identity labels (feedstock only, identity comes from
compositing), Google disclaims per-image verification (warranty stack is empty
end-to-end), and MIAP's age/gender attributes are off-limits by stated Google
position. Snapshot the metadata CSVs now (14.6% link rot).

### #3: hybrid schedule and the cheap unlocks

*(2026-09-03 update: after the mesh audit, the SmartSpaces stage of this
schedule shrinks to positive-pair-only view-invariance pretraining, with no
identity-discriminative loss on its labels; the composite and real stages carry
the identity supervision.)*

SmartSpaces pretrain (viewpoint invariance) then licence-clean composite fine-tune
(appearance diversity). Three cheap follow-on items were identified, of which only
the third is reachable: (a) the MEVID annotation licence, 158
consented-actor identities in exactly this geometry, which stays with counsel
under C-8 and is not pursued with the owner; (b) **CHIRLA**, the only legal
months-horizon clothing-change signal, whose CC-BY-4.0 assessment stands but whose
download is gated behind shared contact information, and POLICY.md, section 3 refuses
gated sets outright, so it is recorded as an option for an owner decision rather than
an unlock to pursue; (c) TVRID + WiseNET as clean on-geometry validation sets, which
are near-free. UnrealPerson Table 4 (SCALE `[V]`) is the precedent that synthetic
pretrain + real fine-tune beats both ImageNet and real-data pretraining (94.0/84.7 vs
91.4/76.8 Market R1/mAP).

---

## 3. The instrument comes before the corpus

v1 and v1b both failed one criterion, false-merge at or below 0.01 against an observed 0.167,
and both verdicts rest on 3 to 6 return probes (LEAD `[V]` in-repo). Every corpus
decision above is downstream of a measurement the current instrument cannot make.
**Widen the probe instrument before any re-ID GPU spend**, alongside the mesh
audit. The highest-value experiment that remains unpublished (SCALE
`[V-negative]`): DIR@FAR binned by gap length in the minutes-to-hours,
clothing-constant band, which this instrument can produce.

## 4. Training-scale laws (SCALE, quantitative)

- **~5 img/ID is a hard floor** (LUPerson Table 6 `[V]`: at a fixed ~1,200-image
  budget, 75 IDs by 17 img/ID = 64.6 mAP vs 751 IDs by 1.7 img/ID = 26.4, because
  triplet mining collapses below ~5). Above ~8 img/ID the depth-against-breadth
  question stops mattering. Target 8 to 17 or more img/ID.
- **Camera and viewpoint diversity beats identity count by 2 to 4 times** (UnrealPerson
  Table 2 `[V]`: 800 to 3,000 IDs at fixed cams = +3.1 to 4.8 R1; 6 to 34 cams at fixed
  IDs = +9.7 to 13.5 R1). Spend budget on camera configurations.
- **Identity saturation at ~2,000 to 3,000** (UnrealPerson section 5 `[V]`: "plateau in
  direct transfer at 3,000 pedestrians"; RandPerson Fig 10 even shows a slow
  decline past the plateau on the far-domain target). MTMC_2024's 2,481 IDs is
  in-band; buying more IDs than that buys nothing.
- **The height gap is the decisive factor, not overhead placement in itself**
  (CARGO `[V]`: Ground to Ground 82.14 R1, Aerial to Aerial 80.00, Aerial to Ground 43.13,
  so training matches the deployed view). Street-trained models land at 28.8 to
  32.8 R1 on elevated indoor CCTV (GRID, UnrealPerson Fig 5); adding 6 indoor
  virtual cameras to the training set bought +4.4 R1.
- **Same-camera is the easy regime the field throws away** (Market-1501 counts
  same-camera matches as junk `[V]`; readmitting them: 93.05 to 96.44 R1 with no
  model change; clothing-constant short-term re-ID is 97.76 R1 cross-camera in
  Anytime-ReID vs 36.75 after clothing change, so session scoping eliminates the
  61-point cliff).
- **Background is ~61% of the measured domain gap** (SBSGAN `[V]`: t-SNE L1
  26.20 to 10.19 on background removal; soft masks beat hard masks by +4.7 R1, so
  carried objects are kept). Blending quality is irrelevant (Simple Copy-Paste `[V]`).
  Binding the same ID label across multiple target-like camera renders is where
  transfer gain comes from (PTGAN `[V]`: 2.0 to 37.5 R1; "6.3 transferred images
  ≈ 1 annotated image").
- **Zero-shot foundation embedders do not work** (three independent sources
  `[V]`: frozen CLIP scores below a plain ImageNet Swin-B across 29 re-ID sets;
  frozen DINOv2 ties it; task training adds +25 points; DINOv3 is about 4 times worse
  than DINOv2 at instance identity). The MegaDescriptor template, an Apache-2.0
  backbone with ArcFace over identity data held in-house, is the recipe that
  resolves both the accuracy and licensing constraints at once.

## 5. Recipe (what changes in the training configuration)

- **torchreid (MIT) stays**, canonical `KaiyangZhou/deep-person-reid` only
  (the weight-provenance deny-list, POLICY.md, rule 11, already bars the
  GPL-labelled repack); the v1 provenance record shows it ran under torch 2.13.0,
  so the sm_120 question is answered by a real run. FastReID is dormant (~2 years);
  BoxMOT is AGPL and is therefore barred from the runtime by the AGPL isolation
  rule (POLICY.md, rule 5).
- **BNNeck recipe as baseline** (Bag of Tricks `[V]`: BNNeck alone +2.1 R1 /
  +4.0 mAP; 256x128, PK 16x4, Adam 3.5e-4, warmup 10 ep, margin 0.3).
- **Drop random erasing and color jitter for cross-domain training** (removing
  REA: +13.9 R1 M to D `[V]`; disabling CJ: +0.6 mAP, and clothing color is the
  session-scoped system's strongest cue). Score every augmentation on a
  held-out domain, never in-domain.
- **Product levers ordered by measured effect size** (SCALE section 8): (1) session/
  temporal gating of the candidate set; (2) threshold calibration against an
  explicit FAR target, with separate thresholds for short-gap vs long-gap
  matches; (3) one-time per-camera BatchNorm adaptation (+0.9 HOTA); (4) only
  then embedder quality (+~2.2 IDF1). The gallery/threshold work is worth more
  than the corpus.
- 32GB is comfortably sufficient (TransReID trained ViT-B/16 on one V100).

## 6. Counsel and registry actions

New counsel items: **C-8**, carried as CQ-29 in `docs/COUNSEL_QUESTIONS.md`: the MEVID
annotation release ships no licence statement of its own, a firm finding rather than an
unverified one, and the CC BY 4.0 grant reaches only the MEVA video beneath it. It was on
LEGAL_HOLD when this was written; the registry now carries
it at TRAIN_PROTOTYPE under a prototype grant, as section 1 records, and C-8 stays
open until counsel resolves it. **CQ-25**
COCO per-image licence filter plus attribution manifest converts CQ-2 to
mitigated. **CQ-26** TVRID: CC-BY-4.0 but real people with no documented
consent/IRB/GDPR posture. **CQ-27 (new)** the NVIDIA SmartSpaces
responsible-AI statement, "not authorized for biometric/surveillance targeting
real individuals", against a session-scoped synthetic-trained embedder. **CQ-28
(new)** composite derivative-work exposure: no decided AI case involves
retained, visually recognizable derivative images (Ross is the closest analogue
and the defendant lost); keep composites internal-only so ND/SA/attribution
sharing triggers stay dormant.

Registry edits proposed here, each with the outcome it reached. **Applied:**
`configs/datasets/physicalai_smartspaces.yaml` carries `reid` in its `task:` list,
together with the measured identity and crop counts, the HuggingFace-only acquisition
rule and the CT2.5 scene exclusion. **Applied for weights:**
`scripts/guards/check_reid_weight_sources.py` refuses `dukemtmc` and `luperson` with
their lineage backbones. **Not adopted:** the COCO filtered subset was never
registered as a `derived_from: [coco]` entry, and DATASETS.md records
`coco_reid_filtered` under "Considered and not used" as never materialised. **Not
adopted:** PersonPath22 and JRDB were never registered as EVAL_ONLY entries; the same
DATASETS.md table records `personpath22` and `jrdb` with their non-commercial clauses
and the reason each was refused.

## 7. Gaps still open (declared, not papered over)

Diffusion-generated person sets and generator-licence flow-through (FLUX/SD3
and similar) are an open item; makehumancommunity.org per-asset licences for
community clothing (site unreachable); commercial vendor terms (Nexdata catalog
404s, Simuletic licence text unpublished, and the vendor category has consolidated
sharply, with Datagen closed and Synthesis AI absorbed); Tian et al. CVPR 2018
background-bias PDF (403 everywhere); LAION v. Kneschke appeal; Casual
Conversations v2 full terms; Dongguk RegDB copyright form; frozen DINOv2 against
OSNet tracker comparison (unpublished). None block the section 2 plan.
