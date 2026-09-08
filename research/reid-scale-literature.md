# How much data a session-scoped re-identification embedder needs

Quantitative literature review, evidence tagged, for a same-camera, same-day,
clothing-constant hospital waiting room.

This is one specialist report in the re-identification licence set, from the pass run
on 2026-09-02 and 2026-09-03.

`research/reid-licence-synthesis.md` is the lead report, and
`research/reid-data-licences.md` is the synthesis of record.

Evidence tags. `[V]` means the primary source was opened and the results table read
directly, most of them as rendered PDF page images; this is the strongest tag. `[V-f]` means
the primary source was opened but the table was extracted by a fetch summariser and not
verified against the rendered page. `[R]` means a secondary source. `[C]` means an inference
or a computation from the tagged evidence. `[V-negative]` records a search that returned
nothing.

**Read this caveat before quoting anything.** Fetch summarisation produced two fabricated
tables during this review: a nonexistent CARGO to AG-ReID transfer table, and a garbled
version of UnrealPerson Table 2. Every headline number tagged `[V]` below was re-read from
the source page itself. A number tagged `[V-f]` needs a second reading before it is quoted.

---

## 0. Summary: the five findings that drive the decision

1. A corpus needs roughly 5 to 17 images per identity, after which identity count is what
   buys accuracy; at a fixed image budget, depth beats breadth heavily at small scale.
   LUPerson Table 6 is a fixed-budget experiment that the paper does not label as one: at
   roughly 1,200 Market-1501 images, 75 identities at 17 images each gives 64.6 mAP and 85.5
   rank-1, while 751 identities at 1.7 images each gives 26.4 mAP and 47.5 rank-1. `[V]` Same
   budget, 38.2 mAP in favour of depth. The gap closes to 3.6 mAP at 30 percent of the budget
   and to 0.1 at 90 percent.
2. Camera and viewpoint diversity beats identity count by a factor of 2 to 4. UnrealPerson:
   800 to 3,000 identities at a fixed 34 cameras gives plus 3.1 to plus 4.8 rank-1; 6 to 34
   cameras at a fixed 800 identities gives plus 9.7 to plus 13.5 rank-1. `[V]` Budget is
   better spent on camera configurations than on identities.
3. The problem this lane has is the one the benchmarks explicitly discard. Market-1501 treats
   relevant images captured in the same camera as the query as junk `[V]`. When same-camera
   matches were admitted, the same model moved from 93.05 to 96.44 rank-1, a 49 percent
   relative error reduction with no model change `[V-f]`.
4. Zero-shot foundation models do not work as re-identification embedders. Frozen CLIP at
   47.5 mean accuracy is worse than a plain ImageNet Swin-B at 53.5 across 29
   re-identification datasets; frozen DINOv2 at 54.3 ties it; task-training the same backbone
   adds 25 points `[V]`. The DINOv2, DINOv3 and SigLIP2 papers contain no person
   re-identification evaluations.
5. The licensing position is the inverse of the accuracy position, and it is the real
   constraint. Market-1501: "DO NOT distribute or use it for commercial purpose." MSMT17:
   "will not be used for any commercial purpose in any form." DukeMTMC was retracted in 2019.
   LUPerson, which SOLIDER, PASS and PersonViT are pretrained on: "commercial usage is
   forbidden." `[V-f]` Every high-accuracy off-the-shelf weight is commercially encumbered.

---

## 1. Scale: identities against images

### 1.1 Canonical training-set sizes, all verified

Read directly from RandPerson section 5.1 (ACM MM 2020), which states all four splits in one
paragraph `[V]`, cross-checked against the MSMT17 paper `[V-f]` and PersonX Table 1 `[V]`.

| Dataset | Train IDs | Train images | img/ID | Test IDs | Cameras | Scene |
|---|---|---|---|---|---|---|
| Market-1501 | 751 | 12,936 | 17.2 | 750 (19,732 img) | 6 | outdoor campus |
| MSMT17 | 1,041 | 32,621 | 31.3 | 3,060 (93,820 img) | 15 | 12 outdoor + 3 indoor |
| DukeMTMC-reID | 702 | 16,522 | 23.5 | 1,110 (19,889 img) | 8 | outdoor |
| CUHK03-NP (det.) | 767 | 7,365 | 9.6 | 700 (6,732 img) | 2 | indoor campus |
| MARS (video) | 625 | 1,191,003 boxes total | | 636 | 6 | outdoor |
| RandPerson (synth subset) | 8,000 | 132,145 | 16.5 | | 19 | 8 outdoor + 3 indoor |
| UnrealPerson (synth) | 3,000 | 120,000 | 40.0 | | 34 | 4 virtual |
| ClonedPerson (synth) | 4,826 | 763,953 | 158.3 | 795 | 24 | 6 |
| LUPerson (unlabelled) | ~219,848 est. | 4,180,243 | ~19.0 | | 46,260 scenes | streetview |

The working rules of thumb are confirmed: Market-1501 is 17.2 images per identity and MSMT17
is 31.3. `[C from V]`

**A documentation inconsistency worth recording.** The original Market-1501 ICCV 2015 paper
states 750 and 751 identities for train and test respectively `[V, read directly]`, which is
the opposite of the released data. Every subsequent paper, including RandPerson's `[V]`, uses
751 train identities and 12,936 images. The data, not the paper text, is the record.

Also from Market-1501 directly `[V]`: 1,501 identities, 32,668 bounding boxes, 6 cameras,
2,793 plus 500K distractors, 3,368 query images, and on average 14.8 cross-camera ground
truths for each query.

### 1.2 The identity-count ablations and where they saturate

UnrealPerson (arXiv:2012.04268) Table 2 is the cleanest identity-count ablation in the
literature. It was read as a page image `[V]`. Direct transfer, ResNet-50 with CBN, 256x128,
batch 64.

| #IDs | #Cams | Market R1/mAP | Duke R1/mAP | MSMT17 R1/mAP |
|---|---|---|---|---|
| 800 | 6 (random textures) | 52.0 / 26.2 | 41.4 / 22.4 | 18.5 / 6.1 |
| 800 | 6 (real textures) | 64.5 / 37.9 | 54.3 / 33.8 | 20.7 / 6.9 |
| 800 | 6 (+acc +hard) | 65.2 / 38.8 | 56.7 / 36.6 | 21.9 / 7.6 |
| 800 | 16 | 69.9 / 42.5 | 61.0 / 38.4 | 26.3 / 9.0 |
| 800 | 22 | 71.1 / 43.9 | 61.9 / 41.1 | 30.3 / 10.7 |
| 800 | 28 | 73.7 / 46.5 | 62.9 / 41.6 | 33.4 / 12.6 |
| 800 | 34 | 74.9 / 48.2 | 64.9 / 42.9 | 35.4 / 13.3 |
| 1,500 | 34 | 75.7 / 50.7 | 67.3 / 46.6 | 36.3 / 13.9 |
| 2,000 | 34 | 76.8 / 52.0 | 69.0 / 48.0 | 37.9 / 14.7 |
| 3,000 | 34 | 79.0 / 54.3 | 69.7 / 49.4 | 38.5 / 15.3 |

The decomposition `[C from V]`: identities 800 to 3,000, a factor of 3.75, at a fixed camera
count gives plus 4.1, plus 4.8 and plus 3.1 rank-1. Cameras 6 to 34, a factor of 5.7, at a
fixed 800 identities gives plus 9.7, plus 8.2 and plus 13.5 rank-1. Camera diversity is worth
2 to 4 times more per unit of effort than identity count.

Saturation is stated explicitly in UnrealPerson section 5, Open Problems `[V]`: on the
quantity of synthesised identities, the algorithm reaches a plateau in direct transfer at
3,000 pedestrians, a number smaller than MSMT17, the real-world re-identification dataset.

Section 4.2 records that adding more pedestrians beyond 3,000 hardly improves performance
`[V]`. Figure 4 sweeps 600 to 3,000 identities at 34 cameras and the curves flatten by roughly
2,000 to 2,500.

RandPerson (arXiv:2006.12774) Figure 10 `[V, read as a page image; it is a figure and not a
table, so no cell values are quoted from it]`. The x-axis runs 1,000 to 8,000 characters, all
imported into the same 11 scenes. The paper records that the scores increase gradually as the
number of characters increases when evaluated on CUHK03-NP, Market-1501 and DukeMTMC-reID,
but that on MSMT17 the performance of the trained model slowly decreases, most likely because
the domain gap between RandPerson and MSMT17 is the largest of the four, so more training data
does not help and instead leads gradually to overfitting.

Its Figure 9, scenes 1 to 11 `[V]`, records that most of the scores increase progressively as
the number of scenes increases, while saturation may be reached after a certain number of
scenes. The curves rise steeply to about 4 scenes and then flatten.

Negative finding: no paper asks directly how many identities are enough in person
re-identification. The closest artefacts are the two ablations above plus the LUPerson
small-scale table below. `[V-negative]`

### 1.3 The fixed-budget depth against breadth experiment

LUPerson (arXiv:2012.03753) Table 6(a), MGN on Market-1501, read as a page image `[V]`. Two
settings at matched image budgets:

- small-scale p percent: p percent of the 751 identities with all of their images, that is,
  deep with few identities.
- few-shot p percent: all 751 identities with p percent of the images each, that is, shallow
  with many identities.

| Budget | approx. imgs | Deep: IDs / img-per-ID | mAP / R1 | Shallow: IDs / img-per-ID | mAP / R1 | mAP delta |
|---|---|---|---|---|---|---|
| 10% | 1,294 | 75 / 17.3 | 64.6 / 85.5 | 751 / 1.7 | 26.4 / 47.5 | plus 38.2 |
| 30% | 3,881 | 225 / 17.2 | 81.9 / 93.7 | 751 / 5.2 | 78.3 / 92.1 | plus 3.6 |
| 50% | 6,468 | 376 / 17.2 | 85.8 / 94.9 | 751 / 8.6 | 84.2 / 93.9 | plus 1.6 |
| 70% | 9,055 | 526 / 17.2 | 88.8 / 95.9 | 751 / 12.1 | 88.4 / 95.5 | plus 0.4 |
| 90% | 11,642 | 676 / 17.2 | 90.5 / 96.4 | 751 / 15.5 | 90.4 / 96.3 | plus 0.1 |
| 100% | 12,936 | 751 / 17.2 | 91.0 / 96.4 | | | |

The `approx. imgs` column is computed here as 12,936 times the budget fraction, not read from
the paper. The paper's own count for the 10 percent row is 1,170 images, quoted in the next
paragraph, so the 1,294 figure and the 1,170 figure are not two readings of one measurement.

The paper records `[V]` that for the small-scale setting on Market-1501, which contains only
1,170 images for 75 persons at a percentage of 10, MGN with its pre-training model achieves
64.6 mAP, which is 11.5 mAP higher than the ImageNet supervised counterpart. It gives the
mechanism as well: quite limited images for each identity may weaken the ability of the hard
triplet loss. `[V]`

Two conclusions for this build `[C from V]`:

- There is a hard floor at roughly 5 images per identity. Below it, at 1.7 images per
  identity, triplet mining collapses and 38 mAP is lost. Above roughly 8 images per identity
  the depth against breadth question stops mattering.
- 75 identities gives 85.5 percent rank-1 where 751 gives 96.4 percent, that is, 89 percent of
  the full-data rank-1 from 10 percent of the identities. For a session-scoped gallery of tens
  of people, this is an encouraging number.

### 1.4 LUPerson pretraining deltas, verified

LUPerson holds 4,180,243 images over 46,260 scenes with roughly 219,848 estimated identities,
from 50,534 filtered YouTube streetview videos. Pretraining: MoCoV2, ResNet-50, 256x128, 8
V100 GPUs, 200 epochs, batch 2560, learning rate 0.3 `[V]`.

Table 5, MGN baseline, ImageNet-supervised against LUPerson-unsupervised, mAP / cmc1 `[V]`:

| Target | ImageNet sup. | LUPerson unsup. | mAP delta |
|---|---|---|---|
| CUHK03 | 70.5 / 71.2 | 74.7 / 75.4 | plus 4.2 |
| Market-1501 | 87.5 / 95.1 | 91.0 / 96.4 | plus 3.5 |
| DukeMTMC | 79.4 / 89.0 | 82.1 / 91.0 | plus 2.7 |
| MSMT17 | 63.7 / 85.1 | 65.7 / 85.5 | plus 2.0 |

The paper records `[V]` that all three methods gain more than 4.2, 3.5, 2.7 and 2 percent in
terms of mAP on CUHK03, Market-1501, DukeMTMC and MSMT17 respectively.

The gain is a factor of 3 larger in the small-data regime `[V]`: at 10 percent of the Market
identities, ImageNet-supervised gives 53.1 mAP against LUPerson at 64.6 mAP, a gain of 11.5.
The paper states that the performance improvement is even more significant with small-scale
training samples.

Table 7, pretraining data scale, MGN, mAP / cmc1 `[V]`:

| Scale | CUHK03 | Market1501 | DukeMTMC | MSMT17 |
|---|---|---|---|---|
| 12.5% | 69.2 / 69.0 | 89.5 / 95.9 | 80.2 / 89.1 | 61.5 / 82.3 |
| 25% | 72.1 / 71.5 | 90.1 / 96.2 | 80.9 / 90.1 | 63.1 / 83.3 |
| 50% | 74.1 / 74.5 | 90.8 / 96.4 | 81.6 / 90.4 | 65.5 / 84.7 |
| 100% | 74.7 / 75.4 | 91.0 / 96.4 | 82.1 / 91.0 | 65.7 / 85.5 |

A factor of 8 less pretraining data, 522k images against 4.18M, costs only 1.5 mAP on Market.
`[C]` The paper records that performance tends to saturate when the data scale is large
enough, and that a larger network capacity would be necessary to leverage more data. `[V]`

---

## 2. Short horizon and same camera: does a weak embedder suffice?

### 2.1 Market-1501 structurally excludes this problem

Read directly from the ICCV 2015 paper, section 3.3 `[V]`: the search process is performed in
a cross-camera mode, so relevant images captured in the same camera as the query are treated
as junk. Junk images have zero influence on the re-identification accuracy `[V]`. Every
Market-1501 number ever published is therefore a lower bound on same-camera performance. `[C]`

### 2.2 The one paper that put same-camera matches back in

arXiv:2008.05255 `[V-f]` records that in the previous CMC calculation a gallery image under
the same camera as the query is considered a junk image, whereas in its application images
from the same camera are used as effective information.

| Protocol | Rank-1 | mAP |
|---|---|---|
| Standard Market-1501 (same-camera = junk) | 93.05 | 81.60 |
| Same-camera matches admitted | 96.44 | 84.30 |

A gain of 3.39 rank-1 and 2.70 mAP with no model change, a 49 percent relative reduction in
rank-1 error. `[C]` This is the cleanest published measurement of how much easier the
same-camera problem is.

### 2.3 Camera bias: the field's central difficulty is constant in this deployment

Exploring the Camera Bias of Person Re-identification (arXiv:2502.10195) `[V-f, fetched
directly]`. Its definition is that the feature distribution is biased towards the camera
labels of the samples, so samples from the same camera tend to gather closer in the feature
space. Normalised mutual information camera bias runs 15 to 32 percent across benchmarks, and
29 to 34 percent for unsupervised models.

The critical observation is that the distances between samples within the same camera are more
skewed to the left, that is, within-camera distributions are more separable than across-camera
ones. `[V-f]`

The entire cross-camera re-identification research programme exists to suppress a nuisance
factor that is constant in this deployment. This lane sits structurally on the easy side of
the field's hardest problem. `[C]`

### 2.4 Tracking ablations: appearance is worth about 1 IDF1 at short gaps

| Paper | Setting | IDF1 delta from appearance | HOTA delta | Tag |
|---|---|---|---|---|
| BoT-SORT (2206.14651) | MOT17-val | plus 0.54 | plus 0.06 | `[V-f]` |
| BoT-SORT | MOT17-test | plus 0.7 | plus 0.4 | `[V-f]` |
| BoT-SORT | MOT20-test | plus 1.2 | plus 0.7 | `[V-f]` |
| ByteTrack (2110.06864) | MOT17-val (static cam) | plus 1.2, with 36 percent more ID switches | | `[V-f]` |
| ByteTrack | BDD100K (driving) | plus 5.9 | | `[V-f]` |
| Deep OC-SORT (2302.11813) | MOT17-val | plus 2.92 | plus 1.53 | `[V-f]` |
| Deep OC-SORT | MOT20-val | minus 0.14, net negative | minus 0.02 | `[V-f]` |
| StrongSORT (2202.13514), weak to strong embedder | MOT17-val | plus 2.2 | plus 1.5 | `[V-f]` |
| StrongSORT EMA feature bank | MOT17-val | plus 0.4 | minus 0.1 | `[V-f]` |
| Hybrid-SORT (2308.00783) | MOT17-test | plus 0.3 | plus 0.4 | `[V-f]` |
| Hybrid-SORT | DanceTrack-test | plus 4.4 | plus 3.5 | `[V-f]` |

Median gain 1.2 IDF1 across all eleven rows, or 0.7 IDF1 across the nine MOT17 and MOT20
rows once the two settings outside this domain are set aside, ByteTrack on BDD100K driving
and Hybrid-SORT on DanceTrack. The mean across all eleven rows is 1.78 IDF1. `[C]`

Two ablations reframe this:

- BoT-SORT Table 2: appearance alone gives 70.0 IDF1 and intersection over union alone gives
  81.5 IDF1 `[V-f]`. A state-of-the-art re-identification embedder without motion is 11.5 IDF1
  worse than bounding-box overlap.
- BoT-SORT Table 1: camera-motion compensation is worth 1.62 IDF1, three times more than the
  entire re-identification branch at 0.54 `[V-f]`.
- The StrongSORT AFLink module, which performs global tracklet linking and is the closest
  analogue to this task, is the Appearance-Free Link: it performs global association without
  appearance information, worth 0.2 IDF1 and 0.0 HOTA `[V-f]`.

### 2.5 These numbers are not the evidence base for this deployment

The gain of 0.7 to 1.2 IDF1 is small precisely because motion covers the gap at 1 second. At
the gaps this deployment faces, minutes to hours with the person out of the room, motion
carries no information at all and appearance is 100 percent of the signal.

SUSHI (arXiv:2212.03038) decomposes cue importance by temporal range `[V-f,
figure-derived]`: removing appearance costs 3.9 IDF1 at long range, levels 7 to 9, against 1.2
IDF1 at short range, levels 1 to 3. The marginal value of appearance grows by roughly a factor
of 3 from short to long range. The multi-object tracking literature systematically understates
what appearance is worth for this lane.

GHOST (arXiv:2206.04656) records `[V-f]` that while the appearance embeddings of one identity
barely vary between two consecutive frames, the embeddings of the same identity before and
after an occlusion can show larger distances. Its fix is separate matching thresholds for
active and inactive tracks, worth plus 0.2 HOTA and plus 0.5 IDF1. Its larger gain is
on-the-fly batch-normalisation domain adaptation to the current camera, plus 0.9 HOTA and plus
1.7 IDF1, a factor of 4 more. On a fixed hospital camera that can be calibrated once, offline.

### 2.6 Gap-length dependence: a genuine gap in the literature

Negative finding `[V-negative]`: no one has published re-association accuracy as a function of
gap length in the minutes-to-hours, clothing-constant band. The literature is bimodal.

- Multi-object tracking ablates 1 to 5 seconds. The only maximum-age sweep found
  (arXiv:2103.16178) runs 30 to 150 frames: graph matching moves 68.0 to 70.0 IDF1, a gain of
  2.0, but the naive Hungarian matcher gets worse past 100 frames, 68.1 to 67.3, so a long
  memory with a weak matcher is actively harmful `[V-f]`.
- Re-identification jumps straight to days and months with clothing change.

The closest analogue is Anytime ReID (arXiv:2509.16635), AT-USTC, 270 people over 21 months
`[V-f]`:

| Scenario | Rank-1 | mAP |
|---|---|---|
| Daytime, short-term (clothing constant) | 97.76% | 87.97% |
| Daytime, long-term (clothing changed) | 36.75% | 25.89% |

A drop of 61 points that is almost entirely the clothing change, which session scoping
structurally eliminates. The 97.76 percent figure is a cross-camera number. `[C]`

### 2.7 Open-set retrieval is where a weak embedder hurts

Liao et al., Open-set Person Re-identification (arXiv:1408.0872) `[V-f]`. Table 4, DIR in
percent at operating points, OPeRID v1.0 with 200 persons, 7,413 images and 6 cameras:

| Algorithm | DIR@FAR=1%, R=1 | @FAR=1%, R=10 | @FAR=10%, R=1 | @FAR=10%, R=10 |
|---|---|---|---|---|
| KISSME | 1.82 | 1.92 | 9.99 | 11.46 |
| MAHAL | 1.89 | 1.99 | 10.50 | 11.97 |
| LADF | 1.53 | 1.74 | 9.11 | 10.82 |
| RRDA (best) | 3.99 | 4.35 | 14.51 | 16.72 |

The bridge to the closed-set literature `[V-f]`: when FAR is 100 percent, the corresponding DIR
is the traditional identification rate of the closed-set identification task, and open-set
person re-identification performance drops significantly as the value of FAR decreases.

This survives the deep era. Adversarial Open-World Re-ID (arXiv:1807.10482) reaches 43.85
percent TTR at 0.1 percent FTR on Market-1501 `[V-f]`. MICRO-TRACK (arXiv:2409.03879), the
closest deployed analogue, moves from 0.790 to 0.964 accuracy purely by moving the confidence
threshold `[V-f]`.

Every query in a waiting room is open-set: most people entering frame are new arrivals, not
returning patients. This, not ranking, is where embedder quality is realised, as the separation
margin between genuine and impostor score distributions. `[C]`

One lead carrying only an `[R]` tag and worth verifying: arXiv:2111.09056 (WACVW 2022, Dublin
Airport, 5 cameras with timestamps) is reported to gain 37.43 percent mAP and 30.22 percent
rank-1 from restricting the gallery to a 30-minute temporal window `[R, primary source returned
403, unverified]`. If it holds, temporal gating beats every embedder upgrade in this report.

---

## 3. Transfer: how far street-trained embedders degrade

### 3.1 The spread is large and method dependent

Market-1501 to MSMT17 rank-1, the same nominal task, in four different papers:

| Source model | R1 | mAP | Paper | Tag |
|---|---|---|---|---|
| GoogLeNet (2018) | 3.4 | 1.0 | MSMT17 paper Table 4 | `[V-f]` |
| ResNet-50 + softmax | 12.4 | 3.6 | RandPerson Table 5 | `[V]` |
| ResNet-50 + CBN | 25.7 | 9.6 | UnrealPerson Table 3 | `[V]` |
| QAConv-GS | 45.9 | 17.2 | arXiv:2104.01546 Table 1 | `[V]` |

A spread of 13 times depending on method. `[C]` No single cross-dataset number should be cited
as the transfer gap.

Reverse direction, MSMT17 to Market-1501: 50.0 / 23.1 `[V, RandPerson Table 3]`; 74.4 / 45.4
`[V, UnrealPerson Table 3]`; 79.1 / 49.5 `[V, QAConv-GS]`.

The asymmetry matters more than the magnitude `[C]`: training on the diverse dataset (MSMT17,
15 cameras, indoor and outdoor) and testing on the easy one costs almost nothing, 79.1 against
79.2 within-dataset. Training on the easy one (Market, 6 outdoor campus cameras) and testing on
the diverse one is catastrophic. The direction of the shift dominates.

Market and Duke in both directions with a strong baseline, Bag of Tricks Table 3 `[V-f]`:
Market to Duke 41.4 / 25.7 against within-Duke 86.4 / 76.4, a factor of 0.48 on rank-1 and 0.34
on mAP; Duke to Market 54.3 / 25.5 against within-Market 94.5 / 85.9, a factor of 0.57 on
rank-1 and 0.30 on mAP. mAP degrades roughly a factor of 2 faster than rank-1: retrieval ranking
quality degrades before top-1 does. `[C]`

### 3.2 Camera-height shift, isolated cleanly

CARGO and VDT (arXiv:2403.14513, CVPR 2024) `[V]`. 5,000 identities, 108,563 images, 13
cameras: 8 on the ground at 5 m and 5 aerial drones up to 75 m. Same model, same identities,
supervised on both views.

| Method | Ground to ground R1/mAP | Aerial to aerial R1/mAP | Aerial to ground R1/mAP |
|---|---|---|---|
| ViT baseline | 82.14 / 71.34 | 80.00 / 64.47 | 43.13 / 40.11 |
| VDT (theirs) | 82.14 / 71.59 | 82.50 / 66.83 | 48.12 / 42.76 |
| AGW | 81.25 / 71.66 | 67.50 / 56.48 | 43.57 / 40.90 |
| BoT | 77.68 / 66.47 | 65.00 / 49.79 | 36.25 / 32.56 |

Ground to ground is 82.14 rank-1 against aerial to ground at 43.13, so the height gap alone
costs 39.0 rank-1 points, a factor of 0.53. Aerial to aerial is 80.00, so overhead cameras are
not intrinsically hard. The mismatch is what costs the accuracy. `[C]`

Direct implication for the top-view lane: a model trained on elevated or top-view data works
nearly as well as a ground model does on ground data, while adapting a street model across the
height gap does not. Build for the view that is actually deployed.

PRAI-1581 (arXiv:1908.05024), 39,461 images, 1,581 identities, unmanned aerial vehicles at 20 to
60 m `[V]`. Supervised aerial re-identification tops out at OSNet 42.10 mAP and 54.40 rank-1,
where the same authors' supervised ground pipeline reaches Market-1501 71.44 / 86.49, an
intrinsic difficulty gap of roughly 32 points. The paper records that the upright assumption of
person images in traditional surveillance cameras cannot hold, which leads to poor part
alignment.

Practical consequence: the horizontal-stripe pooling of PCB, MGN and AlignedReID, which
dominates ground leaderboards, loses most of its advantage overhead. On PRAI the spread between
plain-identity and part-based methods is only about 5 rank-1 points. `[V/C]`

### 3.3 Elevated indoor closed-circuit television, the closest analogue to this camera

GRID holds 250 identities and 1,275 gallery images from 8 cameras in a subway station.
UnrealPerson describes it as a dataset collected in an underground station where the cameras
are located at high angles of view `[V]`.

UnrealPerson Figure 5, direct transfer rank-1 on GRID `[V, read as a page image]`:

| Training set | Market | Duke | MSMT17 | Unreal (no indoor) | Unreal + 6 indoor cams |
|---|---|---|---|---|---|
| GRID R1 | 28.8 | 32.8 | 30.8 | 32.0 | 36.4 |

Every street-trained model lands at 28.8 to 32.8 percent rank-1 on elevated indoor
closed-circuit television. Adding 6 indoor virtual cameras to the synthetic training set gains
4.4 rank-1 over the same synthetic set without them. `[V]`

With dedicated multi-source domain-generalisation research (domain-generalisation survey
arXiv:2506.12413 Table 3, Protocol-1) `[V]`, GRID rank-1 climbs from 29.3 (DIMN, CVPR 2019) to
roughly 60 (ReNorm and BAU, 2024), but only with 4 to 5 source datasets and dedicated
domain-generalisation methods, and GRID is consistently the hardest of the four targets.

**A deflation that changes how these numbers read** `[C]`. The GRID gallery is 1,275 images,
Market's is 19,732, and CARGO's is roughly 2,400 identities. A rank-1 of 60 on a 1,000-image
gallery is worth far less than 60 on a 20,000-image gallery. Only CARGO-scale numbers should
inform a deployment forecast. This is why the AG-ReID figure of roughly 82 percent aerial to
ground looks so much better than CARGO's 43 percent: 189 test identities against roughly 2,400.

### 3.4 Viewpoint (azimuth): PersonX

PersonX (CVPR 2019, arXiv:1812.02162) `[V, read as page images]`. 1,266 identities, 6 cameras,
36 viewpoints at 10 degree intervals, giving 273,456 images. Its subsection conclusions are that
missing viewpoints compromises training, that missing continuous viewpoints is more detrimental
than missing random viewpoints, and that when limited training viewpoints are available,
left and right orientations allow models to be better trained than front and back orientations.

Figure 5C, training on one orientation only: left 94.3 mAP against front 87.7 and back 88.1 on
PersonX12 `[V]`, with the paper putting the mAP gap as large as 6 percent. Figure 5D sweeps 36,
27, 18, 12, 9 and 6 remaining viewpoints; continuous removal drops mAP from roughly 96.5 to
roughly 90, while random removal stays at roughly 94 to 95.

The most relevant finding for this lane `[V]`: the highest re-identification accuracy is
achieved when the viewpoint of the true match is similar to the query, because under the same
viewpoint the query and the true match differ only in illumination and background. In a single
fixed camera the viewpoint is approximately matched, which is the easy regime.

Gap `[C]`: azimuth is well studied and elevation is not. The CARGO aerial to ground split is the
closest controlled camera-height experiment that exists, and it confounds height with resolution
and drone motion. A clean pitch sweep would be a genuine contribution.

---

## 4. Synthetic to real: the numbers, and what is actually claimed

### 4.1 RandPerson (ACM MM 2020), read directly from the PDF `[V]`

Setup `[V]`: ResNet-50, open-reid, softmax cross-entropy only, 256x128, random horizontal flip,
batch 32, SGD momentum 0.9, weight decay 5e-4, learning rate 0.001 on the backbone and 0.01 on
new layers. The subset is 132,145 images sampled at random from the 8,000 identities.

Table 3, tested on Market-1501:

| Training set | R1 | mAP |
|---|---|---|
| Market-1501 (within-dataset) | 82.2 | 58.9 |
| CUHK03 | 33.1 | 12.9 |
| DukeMTMC-reID | 44.9 | 19.0 |
| MSMT17 | 50.0 | 23.1 |
| SOMAset (synth) | 4.5 | 1.3 |
| SyRI (synth) | 29.0 | 10.8 |
| PersonX (synth) | 44.0 | 20.4 |
| RandPerson (synth) | 55.6 | 28.8 |
| RandPerson + Market-1501 | 87.2 | 70.9 |

Table 5, tested on MSMT17: MSMT17 60.8 / 30.7; Duke 18.3 / 5.5; Market 12.4 / 3.6; PersonX 11.7
/ 3.6; SyRI 16.4 / 4.4; RandPerson 20.1 / 6.3.
Table 2, CUHK03-NP: CUHK03 21.5 / 19.8; MSMT17 11.6 / 10.6; Market 7.2 / 6.2; Duke 6.6 / 5.9;
RandPerson 13.4 / 10.8.
Table 4, Duke: Duke 74.2 / 53.7; MSMT17 53.9 / 31.3; Market 34.6 / 18.1; RandPerson 47.6 / 27.1.

The claim as published `[V]`: models trained on virtual data generalise well to unseen target
images, surpassing models trained on various real-world datasets including CUHK03, Market-1501
and DukeMTMC-reID, and almost MSMT17.

Read precisely `[C]`, synthetic beats every real source dataset in the cross-dataset setting,
and loses to MSMT17 as a source on Duke. It does not beat within-dataset training: 55.6 against
82.2 rank-1 on Market.

### 4.2 UnrealPerson Table 3, read directly `[V]`

| Training set | Market R1/mAP | Duke R1/mAP | MSMT17 R1/mAP |
|---|---|---|---|
| Market | 91.4 / 76.8 | 56.7 / 36.5 | 25.7 / 9.6 |
| Duke | 72.4 / 41.9 | 82.1 / 67.5 | 35.8 / 13.1 |
| MSMT17 | 74.4 / 45.4 | 67.1 / 46.8 | 72.5 / 42.4 |
| SyRI | 48.5 / 22.6 | 38.9 / 18.2 | 21.8 / 5.7 |
| PersonX | 58.7 / 32.7 | 49.4 / 28.9 | 22.2 / 7.9 |
| RandPerson | 64.7 / 39.3 | 59.4 / 38.4 | 20.0 / 6.8 |
| Unreal | 79.0 / 54.3 | 69.7 / 49.4 | 38.5 / 15.3 |

Synthetic-only to MSMT17 gives 38.5 rank-1, beating Market to MSMT17 at 25.7 and Duke to MSMT17
at 35.8. It beats all real sources on Market, 79.0 against 74.4, and on Duke, 69.7 against 67.1,
and remains far below within-dataset training at 91.4 on Market. `[V/C]`

Table 4, the recipe that wins: synthetic pretrain followed by real fine-tune `[V]`.

| Pre-train | Fine-tune | Market R1/mAP |
|---|---|---|
| ImageNet | Market | 91.4 / 76.8 |
| MSMT17 | Market | 93.7 / 82.5 |
| Unreal | Market | 94.0 / 84.7 |

A gain of 2.6 rank-1 and 7.9 mAP over ImageNet pretraining. The same pattern holds on Duke
(86.8 / 74.2) and MSMT17 (74.5 / 46.0). `[V]`

### 4.3 ClonedPerson (CVPR 2022), the strongest synthetic-beats-real claim

5,621 identities and 887,766 images, of which 4,826 identities and 763,953 images are training
data, over 24 cameras and 6 scenes `[V, abstract]`. Table 2, QAConv 2.0 `[V-f, read via an HTML
mirror, cells not verified against the rendered page]`:

| Training data | CUHK03-NP R1/mAP | Market R1/mAP | MSMT17 R1/mAP |
|---|---|---|---|
| CUHK03 | | 65.5 / 34.5 | 42.3 / 13.4 |
| Market-1501 | 15.1 / 15.1 | | 41.0 / 14.6 |
| MSMT17 | 18.5 / 19.2 | 76.3 / 47.9 | |
| ClonedPerson | 22.6 / 21.8 | 84.5 / 59.9 | 49.1 / 18.5 |

ClonedPerson to Market at 84.5 rank-1 beats the real MSMT17-trained model at 76.3 by 8.2 rank-1
and 12.0 mAP. `[V-f]`

**A protocol caveat that changes the reading** `[C]`. This uses QAConv 2.0, a purpose-built
generalisable matcher, not a plain embedding with a cosine distance, so the synthetic-beats-real
claim is entangled with the matcher. The RandPerson plain-softmax ResNet-50 result, 55.6 against
50.0 on Market, is the like-for-like version, a much narrower margin of 5.6 rank-1.

Diversity beats volume again `[V-f, ClonedPerson Table 4]`: UnrealPerson at 3,000 identities and
120k images beats UnrealPerson at 6,799 identities and 1,256k images on MSMT17, 52.0 against
44.8 rank-1 with TransMatcher. ClonedPerson Figure 6 shows images per cluster saturating at
about 5, while cluster count improves monotonically.

---

## 5. Losses and recipes

### 5.1 Bag of Tricks (Luo et al., arXiv:1903.07071), Table 1, verified `[V-f]`

| Model | Market R1 / mAP | Duke R1 / mAP |
|---|---|---|
| Baseline-S | 87.7 / 74.0 | 79.7 / 63.7 |
| + warmup | 88.7 / 75.2 | 80.6 / 65.1 |
| + random erasing (REA) | 91.3 / 79.3 | 81.5 / 68.3 |
| + label smoothing | 91.4 / 80.3 | 82.4 / 69.3 |
| + last stride = 1 | 92.0 / 81.7 | 82.6 / 70.6 |
| + BNNeck | 94.1 / 85.7 | 86.2 / 75.9 |
| + center loss | 94.5 / 85.9 | 86.4 / 76.4 |

BNNeck is the single biggest trick, worth 2.1 rank-1 and 4.0 mAP. `[C]`

Exact hyperparameters `[V-f]`: 256x128; P equals 16 identities times K equals 4 images, giving a
batch of 64; Adam; base learning rate 3.5e-4; linear warmup from 3.5e-5 to 3.5e-4 over 10
epochs; decay at 40 and 70; 120 epochs in total; triplet margin 0.3; centre-loss weight 0.0005.

### 5.2 Modern variants

| Recipe | Config | Market R1/mAP | MSMT17 R1/mAP | Tag |
|---|---|---|---|---|
| FastReID (2006.02631) | R101-ibn + non-local, GeM, BNNeck, 384x128, P=4 and K=16, circle plus triplet, 18k iters | 96.3 / 90.3 | 85.1 / 65.4 (MGN R50-ibn) | `[V-f]` |
| TransReID (2102.04378) | ViT-B/16, 256x128, batch 64, 4 img/ID, SGD 0.008 cosine, FP16 | 95.0 / 88.2 | 83.3 / 64.9 | `[V-f]` |
| CLIP-ReID (2211.13977) | ViT-B/16, 2-stage prompt learning, batch 64, P=16 and K=4, Adam 3.5e-4 | | ~89.7 / 75.8 (+SIE+OLP) | `[V-f]` |
| Circle Loss (2002.10857) | gamma 128, m 0.25 | 94.2 / 84.9 (R50), 96.1 / 87.4 (MGN) | 76.3 / 50.2, 76.9 / 52.1 | `[V-f]` |
| PersonViT (2408.05398) | ViT-B/16, LUPerson MIM and DINO pretrain | 95.0 mAP / 97.6 R1 | 80.8 mAP / 92.0 R1 | `[V-f]` |

Circle Loss beats AM-Softmax by 1.8 rank-1 and 1.1 mAP on Market with a ResNet-50 `[V-f]`. Note
the unusual FastReID setting of P equals 4 and K equals 16, which gives much deeper per-identity
batches than the 16 by 4 of Bag of Tricks, consistent with the depth finding in section 1.3.

### 5.3 Compute: 32 GB is sufficient

TransReID trained ViT-B/16 at 256x128, batch 64, on one Nvidia Tesla V100 GPU with FP16 `[V-f]`.
A V100 carries 32 GB. The rented Blackwell-class pod this lane trains on also carries 32 GB, is
considerably faster and supports BF16, so it covers this comfortably.
LUPerson pretraining used 8 V100 GPUs at batch 2560 `[V]`, but this lane fine-tunes rather than
pretrains.

### 5.4 Small-data regime

The strongest evidence is LUPerson, in sections 1.3 and 1.4 above: 75 identities and 1,170
images give 64.6 mAP and 85.5 rank-1 with good pretraining, against 53.1 mAP with ImageNet only
`[V]`. Intra-camera-only supervision reaches 83.3 percent rank-1 on Market against 91.3 percent
fully supervised, an 8-point gap for discarding all cross-camera labels `[R]`.

---

## 6. Augmentation and compositing

### 6.1 The most actionable finding: random erasing helps in-domain and harms transfer

In-domain (arXiv:1708.04896) `[V-f]`: Market IDE 83.14 to 85.24 rank-1; SVDNet 84.41 to 87.08;
CUHK03-Labeled TriNet 49.86 to 58.14.

Cross-domain, Bag of Tricks Table 3 `[V-f, fetched directly and confirmed independently a second
time]`:

| Config | Market to Duke R1/mAP | Duke to Market R1/mAP |
|---|---|---|
| With REA (and all other tricks) | 27.5 / 15.0 | 47.4 / 21.4 |
| Without REA | 41.4 / 25.7 | 54.3 / 25.5 |

Removing random erasing gains 13.9 rank-1 and 10.7 mAP on Market to Duke, and 6.9 and 4.1 on
Duke to Market. The paper records that random erasing does harm to models in the cross-domain
re-identification task. `[V-f]`

Mechanism `[C]`: random erasing forces the model to spread evidence across all image regions
including the background, which is exactly the domain-specific signal it should be ignoring.
Every augmentation decision is scored on a held-out domain, never in-domain.

### 6.2 Colour jitter harms re-identification, and doubly so here

LUPerson Table 2, CUHK03 transfer `[V, read as a page image]`: default 73.4 mAP / 74.0 cmc1;
with colour jitter disabled 74.0 / 74.6; with colour jitter disabled and random erasing added
74.7 / 75.4. The paper records that performance is boosted by 0.6 percent in mAP and 0.6 percent
in cmc1 after colour jitter is disabled, because person re-identification relies heavily on
colour information, for example the colour of clothes and of bags, for association, while colour
jitter harms that.

For a clothing-constant, session-scoped system, clothing colour is the single strongest cue, so
it is not jittered. `[C]`

LUPerson Table 3, random-erasing strength as a maximum area `[V]`: 0.0 gives 73.2; 0.2 gives
74.1; 0.4 gives 74.4; 0.6 gives 74.7; 0.8 gives 73.3. The paper records the optimal random
erasing strength on existing person re-identification methods as 0.4.

### 6.3 Background is roughly 60 percent of the domain gap, and compositing attacks it directly

SBSGAN (ICCV 2019, arXiv:1908.09086) `[V]`. Market to Duke, IDE with DenseNet-121:

| Training data | mAP | R-1 |
|---|---|---|
| Original | 17.7 | 33.5 |
| Hard-mask (background zeroed, JPPNet) | 21.5 | 38.6 |
| Style-transferred (PTGAN) | 22.7 | 42.9 |
| Soft-mask | 22.3 | 43.3 |
| Soft-mask 2-Domains | 23.5 | 44.2 |

t-SNE L1 domain distance between Market and Duke `[V]`: original images 26.20, hard-masked
images with the background removed 10.19.

Roughly 61 percent of the measured inter-domain gap is background, not person. Masking the
background in train and test alone gains 5.1 rank-1, and replacing it with a target-like
background gains 9.4 rank-1. `[C]`

**A caveat recorded by the authors** `[V]`. Hard masks strip carried objects such as bags, which
the paper records as significant to person re-identification and to be retained rather than
removed. Soft masking beats hard masking by 4.7 rank-1, 43.3 against 38.6. Masks are feathered
and carried objects are kept.

### 6.4 Re-rendering source images into target-like camera conditions: 2.0 to 37.5 rank-1

MSMT17 and PTGAN paper Table 3, CUHK03 to PRID `[V-f]`:

| Training set | cam1/cam2 R-1 | R-10 |
|---|---|---|
| CUHK03 (raw) | 2.0 | 11.5 |
| CUHK03 rendered to cam1 | 18.0 | 43.5 |
| CUHK03 rendered to cam2 | 17.5 | 53.0 |
| CUHK03 rendered to cam1 plus CUHK03 rendered to cam2 | 37.5 | 72.5 |

An improvement of 18.75 times with no target labels. The mechanism recorded in the source: each
person image has two transferred samples, one on camera1 and one on camera2, and because those
two samples carry the same person identity label, the training data forces the network to gain
robustness to the style variations between camera1 and camera2. `[V-f]`

Direct read-across: render each identity into multiple target-like camera configurations and
bind the identity label across them. The gain comes from that binding, not from realism. `[C]`

Data efficiency `[V-f]`: 6.3 transferred images are equivalent to 1 annotated image.

### 6.5 Copy and paste: blending does not matter

Simple Copy-Paste (arXiv:2012.07177) `[V-f]`: plus 1.0 to plus 1.5 box AP across five backbones;
LVIS mask AP 33.7 to 36.0, with rare classes gaining 3.3, so the gains are largest where the
data is scarcest. Table 10 records that composing without any blending has similar performance,
both at 50.0 box AP and 43.7 mask AP.

Photorealistic harmonisation is therefore not required for composites to be useful. `[C]`
Precedent: Cut, Paste and Learn (arXiv:1708.01642) `[R]`.

**Not verified.** Tian et al., Eliminating Background-Bias for Robust Person
Re-identification (CVPR 2018), returned HTTP 403 on every attempt and has no arXiv mirror. A
secondary source `[R]` reports that it augments training with person images on random
backgrounds, which is the most on-point citation for compositing in re-identification. The PDF
is still to be obtained.

The standard stack, for the record `[V-f]`: resize to 256x128, pad 10px, random crop, random
horizontal flip at p 0.5, ImageNet normalisation, random erasing at p 0.5 with area 0.02 to 0.4
and aspect 0.3 to 3.33. Erasing and any colour jitter are dropped where transfer matters.

---

## 7. Foundation models zero-shot, and the licensing constraint

### 7.1 The answer is no, and the evidence is consistent across three independent sources

The best-quality evidence is WildlifeDatasets and MegaDescriptor (WACV 2024, arXiv:2311.09118),
read as page images `[V]`. The protocol is exactly re-identification retrieval: a
1-nearest-neighbour classifier within a deep embedding space using cosine similarity. Frozen
features, no training, over 29 re-identification datasets.

| Backbone | Mean accuracy (29 datasets) |
|---|---|
| ImageNet Swin-B (2021 classifier) | 53.5 |
| Frozen CLIP ViT-L/14-336 | 47.5 |
| Frozen DINOv2 ViT-L/14-518 | 54.3 |
| MegaDescriptor-L (re-ID-trained) | 79.3 |

The means are computed `[C]` from the `[V]` table. The paper records that DINOv2, the
state-of-the-art foundation model for almost any vision task, performs poorly in animal
re-identification even with a much higher input size of 518 by 518 and a larger backbone. `[V]`

Frozen CLIP is worse than a plain ImageNet classifier, frozen DINOv2 ties it, and task-training
the same backbone class adds 25 absolute points. `[C]`

Object re-identification (arXiv:2605.26383) `[V-f]`, EPIC-Kitchens, frozen features with a
cosine distance: CLIP ViT-B/32 0.355 mAP; DINOv2 ViT-B/14 0.416; DINOv3 ViT-B/16 0.110; SAM3
0.451. The paper records that zero-shot methods fail, with the best baseline achieving only 45.3
percent mAP. **DINOv3 is roughly 4 times worse than DINOv2 at instance identity here, so newer
is not better for this task.**

Text-based person search (arXiv:2412.20646), read as a page image `[V]`, records that prior
studies have indicated that directly transferring knowledge from CLIP to the text-based person
search task by freezing the model parameters is unfeasible. It gives a worked failure case: two
images of the same person score 0.34 similarity while one of them matches a different person at
0.76.

### 7.2 The one direct frozen person re-identification benchmark, and why to distrust it

arXiv:2601.20598 (Person Re-ID in 2025, a single-author preprint, January 2026) `[V-f]`, Table 2,
mAP in percent, frozen rows:

| Model | MSMT17 | Market-1501 | Duke | CUHK03 | GRID | LasT | Celeb |
|---|---|---|---|---|---|---|---|
| CLIP-B/16 (frozen) | 0.11 | 0.43 | 0.26 | 0.29 | 0.98 | 0.28 | 0.68 |
| CLIP-L/14 (frozen) | 0.14 | 0.50 | 0.29 | 0.30 | 1.99 | 0.49 | 0.82 |
| SigLIP2-384 (frozen) | 4.56 | 5.97 | 8.36 | 2.77 | 5.72 | 14.01 | 15.32 |
| DINOv2-B/14 (frozen) | 0.37 | 1.71 | 1.03 | 0.32 | 4.69 | 4.12 | 3.68 |
| DINOv2-L/14 (frozen) | 0.39 | 1.40 | 0.86 | 0.43 | 3.86 | 4.70 | 3.79 |
| OSNet (supervised) | 3.37 | 83.57 | 19.63 | 5.86 | 21.85 | 2.73 | 3.06 |

The paper states that self-supervised models do not work for zero-shot re-identification, with
DINOv2 managing only 0.3 to 4.7 percent mAP, and that language alignment brings robustness, with
SigLIP2 beating supervised specialists on CelebReID, 14.2 against 5.8, and on LasT, 13.7 against
8.7.

**These magnitudes are a loose lower bound, not a measurement** `[C]`.

- Random-chance mAP on Market-1501 is roughly 0.1 to 0.5 percent. CLIP at 0.37 percent is
  indistinguishable from random, which is implausible for a colour-dominated task.
- The paper has no experimental setup section at all: no resolution, no preprocessing, no
  statement about L2 normalisation, no distance metric, no code and no rank-1 table.
- The likely silent cause: default CLIP and DINOv2 preprocessing is a resize followed by a 224
  by 224 centre crop, which on a 128x64 pedestrian crop cuts off head and legs and squares the
  aspect ratio. Re-identification needs an aspect-preserving 256x128.

Also checked and confirmed absent `[V-f]`: DINOv2 (arXiv:2304.07193), DINOv3 (arXiv:2508.10104)
and SigLIP 2 (arXiv:2502.14786) contain no person re-identification evaluations. Their
instance-retrieval results are on Oxford and Paris landmarks, Met artworks and AmsterTime, which
is category-with-one-instance retrieval and is structurally different from one category with
thousands of instances distinguished by low-level appearance. A high Oxford-Hard mAP is not
evidence of re-identification capability. No CLIP-ReID, SOLIDER, PersonViT or Instruct-ReID
paper reports a frozen baseline either.

Unclosed gap `[V-negative]`: no published frozen DINOv2 against OSNet IDF1 or HOTA tracker
comparison was located; the search was not exhaustive, so its existence cannot be ruled out.

### 7.3 Licensing: model weights are fine, the datasets are the problem

| Asset | License | Commercial? | Source |
|---|---|---|---|
| DINOv2 (code and weights) | Apache-2.0 | yes | raw LICENSE read directly `[V-f]` |
| DINOv2 biology variants | CC BY-NC / FAIR Noncommercial | no | dinov2 README `[V-f]` |
| DINOv3 | custom DINOv3 License, gated, non-transferable | yes, conditioned | LICENSE.md read directly `[V-f]` |
| CLIP (OpenAI) | MIT | yes | `[V-f]` |
| OpenCLIP | MIT | yes | `[V-f]` |
| SigLIP / SigLIP 2 | Apache-2.0 | yes | Hugging Face model cards `[V-f]` |
| EVA-CLIP | unresolved, 404 on LICENSE | unknown | `[V-f]` |
| torchreid / OSNet code | MIT | yes, code only | `[V-f]` |
| FastReID code | Apache-2.0 | yes, code only | `[V-f]` |

DINOv3 specifics, read directly from the licence `[V-f]`: royalty-free but non-transferable;
publication acknowledgement required; an ITAR, military and weapons field-of-use bar; no
Llama-style monthly-active-user threshold; and nothing restricting surveillance or biometric
identification, which is permissive, although GDPR, BIPA and the EU AI Act bind independently.
Every Hugging Face checkpoint is gated behind contact-information sharing.

**The dataset licences are the actual constraint** `[V-f]`.

| Dataset | Terms as published |
|---|---|
| Market-1501 | "This dataset should be used for research only. Please DO NOT distribute or use it for commercial purpose." |
| MSMT17 | "All the dataset will be used for the purpose of scientific researches only... will not be used for any commercial purpose in any form." Also: "No images in MSMT17 could be published or shown in any form of publication, document, or demonstration." Not even a sample in a slide. Performance numbers may be reported. |
| DukeMTMC | Retracted in April and June 2019. The website returns a DNS error. arXiv:2108.02922 (NeurIPS 2021 Datasets and Benchmarks) records that a creator acknowledged in June 2019 that institutional review board guidelines had been violated. |
| LUPerson | "LUPerson can only be used for research, commercial usage is forbidden." |

LUPerson is the concealed constraint: SOLIDER, PASS, PersonViT, HAP and TransReID-SSL are all
pretrained on it, and an Apache-2.0 licence on the code does not make those weights commercially
clean. Per arXiv:2108.02922 `[V]`, the PyTorch and Keras maintainers both decline to say whether
weights launder non-commercial training data, and refer the question to legal counsel.

| Route | Accuracy | Data provenance |
|---|---|---|
| Frozen DINOv2 or SigLIP2 (Apache-2.0) | unacceptable, at most 6 mAP measured | clean |
| OSNet, FastReID or CLIP-ReID weights | excellent, roughly 85 to 95 mAP | Market, MSMT17 and Duke, encumbered |
| SOLIDER, PersonViT or PASS | excellent, roughly 94 to 95 mAP | LUPerson, encumbered |
| DINOv3 | possibly worse than DINOv2 for identity | custom and gated, unresolved |

The MegaDescriptor recipe is the template that resolves both constraints `[V]`: take an
Apache-2.0 backbone, DINOv2 or SigLIP2, add ArcFace, and train on identity data that is owned or
licensed. That is 25 points over frozen, measured rather than assumed, and it avoids Market,
MSMT17 and LUPerson entirely. Benchmarking on Market and MSMT17 remains permissible, since the
MSMT17 terms at clause (4) explicitly permit reporting performance.

---

## 8. What this means for the hospital waiting room

The waiting-room problem is materially easier than the benchmark, and the literature supports
that quantitatively:

- Same-camera admission: 3.39 rank-1, a 49 percent relative error reduction, with no model
  change `[V-f]`.
- Clothing-constant short-term: 97.76 percent rank-1 even cross-camera `[V-f]`.
- Camera bias, the field's central nuisance, is constant in this deployment `[V-f]`.
- Viewpoint is approximately matched, the regime PersonX identifies as easiest `[V]`.
- 75 identities gives 85.5 percent rank-1 where 751 gives 96.4 percent `[V]`.

Priority order, by measured effect size:

1. Session and temporal gating of the candidate set. `[R, 37 mAP, unverified]` Structurally the
   largest lever, and free.
2. Threshold calibration against an explicit false-accept-rate target. MICRO-TRACK moves 0.790
   to 0.964 accuracy from the threshold alone `[V-f]`. Separate thresholds are used for
   short-gap and long-gap matches, following GHOST `[V-f]`.
3. One-time camera adaptation, that is, batch-normalisation recalibration on this project's own
   footage: plus 0.9 HOTA and plus 1.7 IDF1, close to free on a fixed camera `[V-f]`.
4. Then embedder quality, worth roughly 2.2 IDF1 for a weak to strong swap, following StrongSORT
   `[V-f]`.

Corpus design, if one is built:

- Target 8 to 17 images per identity or more; below roughly 5 the triplet loss collapses, at a
  cost of 38 mAP `[V]`.
- Spend budget on camera configurations rather than identities, a 2 to 4 times better return
  `[V]`.
- Roughly 2,000 to 3,000 identities is the documented saturation point `[V]`.
- Composite with soft, feathered masks; keep carried objects; render each identity across many
  target-like backgrounds and heights with the identity label bound across the variants `[V]`.
- Drop random erasing and colour jitter where transfer matters, worth 13.9 rank-1 and 0.6 mAP
  respectively `[V-f]/[V]`.
- Validate augmentation choices only on a held-out domain.

Compute: TransReID trained ViT-B/16 at 256x128, batch 64, FP16, on one 32 GB V100 `[V-f]`. The
rented Blackwell-class pod this lane trains on, also 32 GB, is comfortable.

Three hazards the benchmarks do not warn about:

1. Clothing is less constant than assumed. Coats come off, gowns go on, blankets appear. That
   assumption is load-bearing: it is the difference between 97.76 percent and 36.75 percent
   rank-1 `[V-f]`.
2. A waiting room resembles DanceTrack more than Market-1501: many seated people in similar dark
   outerwear. Where appearance is uninformative, appearance-based trackers collapse rather than
   degrade, with FairMOT at 39.7 HOTA against motion-only OC-SORT at 54.6 `[V-f]`.
3. The small-gallery effect. A GRID rank-1 of roughly 60 sits on a 1,275-image gallery. Only
   CARGO-scale numbers forecast a deployment.

The highest-value experiment available `[V-negative]`: no one has published re-association
accuracy against gap length in the minutes-to-hours, clothing-constant band. Multi-object
tracking ablates gaps of at most 5 seconds and re-identification jumps to days. A DIR at a fixed
false-accept rate, binned by gap length and measured on this project's own footage, would be both
the number this lane needs and a genuine contribution. The second highest: a carefully engineered
frozen baseline, DINOv2-B and SigLIP2-384 at an aspect-preserving 256x128 with L2 normalisation,
measured on Market. It is a small amount of work and would be the best available evidence on a
question the field has assumed away.
