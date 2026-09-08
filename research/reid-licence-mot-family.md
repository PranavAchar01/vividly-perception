# Licence and ethics due diligence: MOT and person-identification datasets for a commercial hospital re-identification embedder

This is one specialist report in the re-identification licence set. `research/reid-licence-synthesis.md` is the lead report, and `research/reid-data-licences.md` is the synthesis of record.

Dated 2026-09-02. Evidence tags: `[V]` primary source opened and read, `[R]` secondary only, `[C]` inference.
This is evidence-gathering, not legal advice. Operative quotations are excerpted for legal analysis.

## Bottom line

Of the 13 targets, **four clear the commercial bar**. Only **two** of those are also technically suitable for training a re-identification embedder.

| # | Dataset | Commercial training | Cross-camera IDs | Geometry fit | Verdict |
|---|---|---|---|---|---|
| **13a** | **NVIDIA PhysicalAI-SmartSpaces** (HF) | permitted, CC BY 4.0 | yes | indoor CCTV, **hospital scenes** | **primary** |
| **9/10** | **MEVA and MEVID** | CC BY 4.0 on the MEVA video; the MEVID annotation layer is undeclared, see the caveats below | yes | caution: elevated oblique, 37% indoor | **secondary** |
| **13b** | **WiseNET** | permitted, **CC0** | caution: within-set only | indoor | small, clean supplement |
| **11** | VIRAT Ground | permitted, explicit | none | caution: outdoor elevated | legally acceptable, **useless for re-identification** |
| 1 | MOT17 / MOT20 | refused, non-commercial, so **no grant** | none | MOT20 elevated | reject |
| 2 | MOTSynth | refused, no grant plus **GTA-V intellectual property** | none | outdoor only | reject |
| 3 | SportsMOT | refused, CC BY-NC 4.0 | none | broadcast | reject |
| 4 | PersonPath22 | refused, CC-BY-NC-4.0 | none, within-video only | mixed | reject |
| 5 | PathTrack | refused, no grant, and **dead** | none | 71% handheld | reject |
| 6 | TAO / TAO-Person / Amodal | refused, no grant, 7 mixed upstreams | none | none elevated | reject |
| 7 | BDD100K MOT | refused, members only | none | dashcam | reject |
| 8 | WILDTRACK | refused, research and evaluation only | yes | elevated | reject on licence |
| 12 | AI City (portal) / CityFlow / CAMPUS | refused / refused / no grant | none | none | reject |

## Item 13: NVIDIA PhysicalAI-SmartSpaces

**Primary URLs [V]:** `huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces`, `/raw/main/README.md`, `/resolve/main/croissant.json`, `/api/datasets/nvidia/PhysicalAI-SmartSpaces`

**Licence [V].** The dataset card YAML and the HF API both give `license: cc-by-4.0`. Ungated, not disabled. An adversarial grep of the card found **no** occurrence of "non-commercial", "research only", "academic", "prohibited", or "restricted".

**Why it wins:** synthetic (Omniverse plus Cosmos Transfer), so zero subject-privacy exposure; **indoor warehouse, hospital, retail and lab**; `obj_id` is documented as *"Consistent numeric identifier for each object across cameras"* `[V]`, which is true cross-camera identity supervision; 1080p/30fps with full intrinsics, extrinsics and homographies shipped.

| Subset | Scenes | Hours | Cameras | **Person IDs** | 2D boxes |
|---|---|---|---|---|---|
| MTMC_Tracking_2024 | 90 | 212 | 953 | **2,481** | 135M |
| MTMC_Tracking_2025 | 23 | 42 | 504 | 292 | 73M |
| MTMC_Tracking_2026 | 28 | 28.7 | 353 | **901** | 33.2M |

About **3,674 synthetic person identities**, comparable in identity count to MSMT17, with far more views per identity. The 2025 split includes an explicit `Hospital_000` scene.

**Camera geometry, confirmed `[V]`**, from the card's Responsible-AI metadata: *"Camera viewpoints follow industrial CCTV conventions (overhead and oblique)"*, the quotation truncated there because only the overhead-and-oblique half bears on this point. This is exactly the target geometry, and it is documented rather than inferred.

### Two things that must reach counsel

**1. A split-licence trap.** The *same data* carries two incompatible licences depending on the acquisition path `[V]`. The AI City Challenge 2022 to 2024 Dataset License Agreement PDF states: *"You may not use the DATASET or any models developed using the DATASET for any commercial or production purpose."* The HuggingFace release of the identical bytes is CC BY 4.0. **HuggingFace-only acquisition must be mandated and the provenance documented:** an engineer who registers for the challenge portal binds the company to the non-commercial term regardless of the parallel CC BY grant. From 2025 onward the challenge pages themselves point at the CC BY 4.0 release.

**2. NVIDIA disclaims this exact use case.** The card's own responsible-AI metadata says `[V]`: *"The dataset, including the hidden target, is not authorized for use in biometric, facial recognition, demographic profiling, or surveillance systems targeting real individuals"*, and lists as out-of-scope *"clinical decision-making, production surveillance, biometric or facial recognition."* CC BY 4.0 is irrevocable and contains no field-of-use restriction, so the copyright grant stands, but this is a stated publisher position squarely against a commercial hospital re-identification product. It is a reputational and contractual-interpretation risk, not a clean copyright block. **It must be escalated.**

**Also:** exclude `Warehouse_026` and `_027` (2026), which are real-world captures, though test-only and unlabelled. NVIDIA states that every individual in that real footage *"provided a signed, written consent agreement"* `[V]`.

## MEVA (item 10) and MEVID (item 9): the best *real* footage option

**MEVA licence [V]**, quoted from `mevadata.org` and `mevadata.org/resources/MEVA-data-license.txt`:
> "All MEVA data is available for use under a CC BY-4.0 license"

and the licence file header: *"…MEVA dataset by Kitware Inc. and the Intelligence Advanced Research Projects Activity (IARPA) is licensed under a Creative Commons Attribution 4.0 International License."* The full unmodified CC BY 4.0 legal code follows, with **no non-commercial rider, no field-of-use restriction, and no click-through**.

**The annotation-licence trap was checked and cleared `[V]`.** MEVA annotations live at `gitlab.kitware.com/meva/meva-data-repo`; its `LICENSE` file is byte-identical to the MEVA *data* licence, not a code licence, which is the opposite of the usual pattern.

**Ethics, the strongest in this entire review `[V]`.** From the WACV 2021 paper: the collection was overseen by an institutional review board, and *"Included in the protocol and consent briefing was a sample license that may be used for the data, a CC-BY 4.0, to make it fully clear the intent and potential future use of the data."* Hired professional actors signed consent forms approved by that board **after being shown the CC BY 4.0 licence**. The authors named *"prevention of commercial use"* as a defect they deliberately engineered around.

One caveat is worth stating plainly: collection occurred at an access-controlled facility to *"minimize"*, not eliminate, non-participants in frame. The phrase "zero non-consenting individuals" must not appear in customer-facing collateral.

**MEVID identity statistics, every figure in the brief verified exactly `[V]`:** **158 identities**, **8,092 tracklets** (mean 592.6 frames), **33 camera views** across 17 locations, **598 outfits** (1 to 9 changes per actor), more than 1.7M boxes. The train and test splits are **disjoint in both actors and cameras**, which is unusually honest, so scores should be expected to be pessimistic (best baseline 27.1% mAP; the cloth-changing subset collapses to 1.2% to 5.7%).

**Camera geometry, computed from the shipped KRTD calibration files `[V]`:** mounting height **4.0 m to 16.1 m**, depression angle **-1.1 degrees to 35 degrees** (median about 15 degrees). This is **elevated oblique CCTV, not top-view**. The two indoor cameras with published calibration sit at about 4 m with 12 degrees to 16 degrees down-tilt, a wall or ceiling-corner mount, close to typical ward-corridor installs. Cross-referencing clip filenames against MEVA's camera status table, **about 37% of MEVID's supporting video is indoor** (classroom, entrance, stairwell), materially more than the paper's prose suggests.

### Three MEVID caveats

- **The annotation licence is inherited, not stated `[C]`.** There is no LICENSE file inside the `mevid-annotations/` S3 prefix and the README never says "license." The chain (mevadata.org, then the AWS Open Data registry, then the bucket objects) is strong but is one inference step. Confirming it with the publisher would be the cheapest risk reduction in this report; under the standing acquisition rule in POLICY.md, section 3 no dataset owner is contacted, so this stays an open item for counsel rather than an action.
- **The repository `Kitware/MEVID` has no licence at all `[V]`** (`license: null`, four 404s on LICENSE variants). The baseline code must not be vendored. The canonical repository is `Kitware/MEVID`; the `CVRL/MEVID` URL sometimes cited for it returns 404.
- **Biometric privacy law is the real blocker, not copyright.** MEVID redistributes about 43 GB of person crops *and* `mevid-v1-actor-checkin-photos.zip`, which are identifiable face portraits. A CC BY copyright grant conveys nothing about data-protection consent for a given processing purpose. Under BIPA and GDPR Art. 9 this is special-category biometric data. Route it to privacy counsel, and never redistribute the crops or the portraits.

## Items 1 to 12: detail

### 1. MOT17 and MOT20: the site now returns HTTP 410

**The most consequential single finding `[V]`:** `motchallenge.net/`, `/data/`, `/terms/` and `/license/` all return **HTTP 410 Gone**. Only static stubs and the ZIP downloads survive. **The licence text lived on the homepage that is now gone.**

Recovered from the archive, where it was present until at least 2026-03-24 `[V]`:
> "The datasets provided on this page are published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License."

So the widely-held belief is correct, but the statement is no longer published. Live sources grant nothing: both data pages return zero matches for licence, copyright or attribution; **neither `MOT17.zip` (33,960 entries) nor `MOT20.zip` (13,461 entries) contains any LICENSE or README**, verified by reading the ZIP central directories over HTTP range requests; and the MOT16 and MOT20 papers contain zero hits for licence, consent, ethics, or GDPR. The devkit `MOTChallengeEvalKit` has **no licence file**; `TrackEval` is MIT but ships no data.

The only live legal text reachable from those pages is the TUM Impressum, which permits reuse *"nur mit unserer Einwilligung"*, that is, only with the publisher's consent `[V]`. **Verdict: prohibited, degrading to unclear with no grant**, and there is no counterparty left to license from (the contact is `motchallenge-eol@`).

**Provenance premise corrected `[V]`:** the PETS, TUD, ETH, KITTI and AVG-TownCentre aggregation is **MOT15, not MOT17**. MOT16 and MOT17 Table II mark 12 of 14 sequences `new`; only MOT17-05 and MOT17-06 are third-party (Ess et al., CVPR 2008).

**Identity suitability:** MOT17 gives **546 trainable identities** (test ground truth withheld), single-camera, and the protocol explicitly reassigns a new identity when a person re-enters, which is actively hostile to re-identification. **MOT17 has zero sequences that are both indoor and elevated with public labels.** MOT20 is the best technical fit anywhere in this set, *"All sequences are filmed in high resolution from an elevated viewpoint"* `[V]`, about 2,215 training identities of which about 344 are indoor, elevated and crowded, **and it is the one most clearly barred**. Ethics: non-consenting crowds in train stations and stadium exits, with no consent or review-board statement anywhere.

### 2. MOTSynth: trades a privacy problem for an intellectual-property problem

**No licence statement found** `[V]` on the paper, the AImageLab project page, the MOTChallenge MOTSynth page, or inside the distributed archives. The repository's MIT LICENSE is scoped by its own words to *"this software"*, which is the exact code-against-data trap. Distribution *from* motchallenge.net means it most likely inherits CC BY-NC-SA 3.0 `[C]`.

**The disqualifier is asset provenance `[V]`.** The paper states that it used Script Hook V and *"579 pedestrian models, provided by the GTA-V game"*, and offers **no legal basis whatsoever**. Take-Two's Terms of Service grant a licence *"for your personal, non-commercial enjoyment"* and bar users from *"create derivative works, commercialize, or otherwise exploit the Services (including the Content)"*, with a recently added clause barring use to *"develop, train, enhance, or provide source material for"* generative systems. Take-Two has a documented enforcement record (OpenIV 2017; re3/reVC and other DMCA campaigns from 2021) `[R]`.

**MOTSynth is often pitched as the safe synthetic fallback. It is arguably the most legally exposed option here**, because Take-Two is an active, well-resourced enforcer that still exists, unlike the defunct MOTChallenge. Technically: 9,519 identities but from only 579 base meshes; **zero indoor content**; and the paper concedes that near-bird's-eye viewpoints *"are rare in MOTSynth dataset."*

### 3. SportsMOT: a clear non-commercial term, plus a defective chain of title

**Licence `[V]`**, repository README: *"SportsMOT is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License."* The official HF mirror (`MCG-NJU/SportsMOT`, tag `cc-by-nc-4.0`) adds a gated checkbox: *"I agree to use this dataset for non-commerical use ONLY."* The terms also bar redistribution without written permission.

**Worse than the non-commercial term:** the clips are YouTube identifiers, scraped from *"Olymplic Games, NCAA Championship, and NBA on YouTube"* `[V]`. Nanjing University does not own IOC, NBA or Premier League broadcast footage and cannot license it, so even a negotiated commercial licence from that university would not cure it `[C]`. Upstream MultiSports repeats the same trap: an MIT LICENSE file over CC BY-NC data.

It is technically irrelevant in any case: about 1,280 usable identities, within-clip only, and the paper's own framing is that *"the objects appearances in sports scenes are less distinguishable"*. It is a motion-association benchmark that deliberately suppresses appearance cues.

### 4. PersonPath22: two independent blockers

**Correction:** PersonPath22 is **arXiv:2211.02175, ECCV 2022** `[V]`; arXiv:2211.11704, sometimes cited for it, is a SLAM paper.

**The two-licence split confirmed `[V]`.** The repository's only LICENSE file is Apache-2.0, which covers code. The data licence exists **only as README prose**: *"The retrieved videos and their annotations are licensed under the CC-BY-NC-4.0 License."* AWS Open Data independently lists CC BY-NC 4.0.

**The second blocker names this exact use case `[V]`**, on both the homepage and in the paper: *"…request to not use the data to identify or generate biometric information of the people in the videos."* It is phrased as a request, so it is not independently enforceable, but it is indefensible in a customer security review.

**Chain-of-title gap `[V]`:** of 236 videos, Amazon hosts 179 and admits it has *"no right to re-distribute"* the other 57 (PathTrack, MEVA, VIRAT, Pixabay), yet asserts CC-BY-NC over "the retrieved videos." The documented PathTrack retrieval URL now returns 404.

**Disqualifying for re-identification regardless `[V]`:** *"we assume that the same person does not appear in different videos"*, so identities are within-video only, across 12,150 tracks (7,534 foreground), annotated at 5 FPS. One strategically useful note: the *underlying* stock licences (Pexels, Pixabay, Fillerstock, about 78% of the corpus) do permit commercial use, and Amazon's non-commercial term is a downstream restriction. Direct licensing from the stock platforms is a legitimate alternative path; Amazon's annotations are not.

### 5. PathTrack: no licence, and dead

**No licence statement found `[V]`.** Exact term counts over the full paper: `license 0, copyright 0, consent 0, ethic 0, privacy 0`. The archived project page has no terms section, and the paper's dataset footnote is an unfilled placeholder (`http://www.project-website.com`). The blanket ETH CVL notice, *"All data is only for research purposes"*, does not list PathTrack, so the result is research-only if it applies and **no grant at all** if it does not. Both branches prohibit.

**Provenance `[V]`:** *"we queried and crawled YouTube with 23 phrases"*, and ETH redistributed the **actual video pixels**, not identifiers. **The download is dead:** the release zip and its parent directory both return 404, and `trace.ethz.ch` fails DNS resolution. There are 16,287 trajectories, but **71% come from moving handheld cameras** and the annotations are deliberately noisy.

### 6. TAO, TAO-Person and TAO-Amodal: no data licence, seven mixed upstreams

TAO's LICENSE opens with an unusually explicit scope limiter `[V]`: *"NOTE: This license applies to the code in this repository."* Nothing then covers the data: the README has **zero** occurrences of "license", `taodataset.org` has no licence section, and the paper has zero hits for licence, ethics, consent or privacy. **No licence statement found.**

TAO aggregates **Charades, LaSOT, ArgoVerse, AVA, YFCC100M, BDD-100K and HACS** and provides **no per-source licence table and no statement that licences differ**, which is itself the finding. **BDD100K alone is a hard commercial blocker** (see item 7), and AVA is sourced from copyrighted feature films that Google never held redistribution rights to.

TAO's documentation carries a **video deletion request mechanism** `[V]`, an admission that people are featured without consent and that rights-holders may not have granted rights, plus a mutable corpus with no versioning.

**TAO-Amodal's sole licence assertion is a one-word `mit` tag on a gated HF repository** `[V]`; the GitHub repository returns `license: null` with no LICENSE file. That tag is not supportable upstream. **TAO-Person** is defined in SiamMOT, not GTR, and is concentrated in exactly the two worst upstreams, AVA and BDD `[V]`.

**Disqualifying independent of licensing:** annotation is **1 FPS** and federated and sparse, so there is no temporal continuity for hard-positive mining, and unlabelled people become false negatives in a contrastive objective.

### 7. BDD100K MOT: the premise needs correcting

**Current licence `[V]`**, live at the `bdd-data.berkeley.edu/download.html` click-through gate:
> "Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement; and permission to use, copy, modify and distribute this software for commercial purposes (such rights not subject to transfer) to BDD and BAIR Commons members and their affiliates, is hereby granted…"

**Correction:** BDD100K did **not** move *to* non-commercial. Diffing archived snapshots `[V]`, the change landed between 2021-11-30 and 2022-02-24 and **widened** the commercial class from *"BDD member and its affiliates"* to *"BDD and BAIR Commons members and their affiliates."* The public commercial door was shut the whole time. No CC-BY-NC phase was found in any primary source.

Note *"such rights not subject to transfer"*: a member cannot pass commercial rights on by contract. The GitHub BSD-3 file is code-only and is a trap. Site health: `www.bdd100k.com` and `doc.bdd100k.com` are both **DNS-dead** today.

**Identity figures corrected `[V]`:** the figure of 131K identities is dominated by cars. **Person tracks number about 24K** (440K boxes), all within-clip dashcam, so there are **zero cross-camera pairs**. Forward-facing, eye-level, moving platform: near worst-case mismatch.

### 8. WILDTRACK: the dataset page grants nothing, and the parent page prohibits

**The EPFL WILDTRACK page grants no licence at all `[V]`.** The complete page text has no licence, terms, copyright notice, or data-use agreement, and **that is the finding.**

The parent index `epfl.ch/labs/cvlab/data/` carries a site-wide notice that binds it `[V]`:
> "Unless explicitely distributed under a GPL licence, the data that can be dowloaded from this site should only be used for research and evaluation purposes." … "For any form of non GPL commercial usage, prior approval from EPFL should be obtained in writing."

**Verdict: prohibited** absent written EPFL approval, which is stronger and more actionable than mere absence of a grant.

**The worst ethics profile of the review `[V]`.** A full-text keyword scan of the CVPR paper returns **0 hits** for consent, ethic, IRB, privacy, anonym, blur and GDPR. It was filmed *"in a public open area"* outside ETH Zurich with *"unscripted dense groups of pedestrians"*, and the title itself says **Unscripted**. Faces are visible and unblurred in the paper's own figures, and annotation was crowdsourced to Mechanical Turk. **DukeMTMC, the closest analogue, was withdrawn in 2019 over exactly this profile `[C]`.** Identifiable people filmed in the EU for a biometric purpose puts GDPR Art. 9 squarely in play.

Technically it is the best real geometric match among items 1 to 12 (7 static overlapping cameras *"positioned above the humans' average height"*, with genuine cross-camera identities), but **313 identities is far too few** to train an embedder, and it is a single outdoor daylight scene.

### 11. VIRAT: the only one of items 1 to 12 that permits commercial use

**Licence `[V]`**, from the VIRAT Video Dataset Usage Agreement PDF:
> "You may use the Dataset for lawful research and commercial purposes."

**Verdict: permitted.** Redistribution is prohibited except to co-signers, plus *"a few video frames or short video clips"* in publications.

**Two risks to escalate `[V]`:**
1. **Termination at will, with destruction of derivatives.** Kitware may terminate *"at any time, without notice"*, whereupon the licensee must *"return or destroy all copies, derivatives and related documentation."* Trained weights are plausibly a derivative, so this could reach shipped production models.
2. The agreement obliges the licensee to *"protect against the disclosure of Personally Identifiable Information"* and *"not to take any action to seek out the PII"*, which is in visible tension with training a biometric identifier.

**Ethics, and it runs the opposite way to the usual assumption `[V]`.** From viratdata.org: *"Actions performed by directed actors were minimized; most were actions performed by the general population."* The footage is predominantly of **non-consenting members of the public**, with consented actors deliberately minimised. There is no review-board or consent statement anywhere.

**It fails hard on identity labels `[V]`.** Annotations are per-clip KPF, and `id1` is renumbered independently in every clip, so there is **no global identity namespace**, and the word "identity" appears 0 times in the Release 2.0 documentation. Counting the whole public DIVA corpus: **1,074 person track instances across 119 clips, median 6 per clip, zero cross-camera pairs.** Tracks also fragment (*"sometimes, tracks fragment into multiple segments"*). Person heights are 10 px to 200 px. **Legally usable, technically near-useless for re-identification.**

### 12. CityFlow, CAMPUS, AI City portal

- **CityFlow: vehicles only `[V]`.** 40 cameras, 10 intersections, no person tracks. Governed by the NVIDIA agreement, which prohibits, and moot in any case. **Drop.**
- **CAMPUS: no licence found, dead host, wrong geometry `[V]`.** The CVPR 2016 paper gives no URL and no licence for CAMPUS itself, and `web.cs.ucla.edu` returns a host-level 403. Cameras are *"mounted around 1.5m-2m above ground"*, which is eye-level rather than elevated CCTV, mostly outdoor, with roughly 60 to 100 identities in total. Beware a **name collision** with the Belagiannis TPAMI 2016 "Campus" pose dataset (3 cameras, 3 people). **Drop.**

## Item 13: the rest of the permissive-licence sweep

Beyond the two winners above, the sweep covered HuggingFace systematically (by licence tag across about 30 identity tokens), recent conference releases, indoor multi-camera sets, hospital and operating-room sets, and the synthetic re-identification family.

**WiseNET: CC0, and ethically the cleanest thing found.**
`[V]` The 4TU.ResearchData record `10.4121/uuid:c1fb5962-…` returns licence **`CC0`** (`creativecommons.org/publicdomain/zero/1.0/`). The Data in Brief article (CC BY) states `[V]`: *"Each people that appear in the video sets signed informed written consent before participating."* Six indoor cameras across 6 spaces in a university building, 11 video sets, 77 people tracks, identities consistent across cameras **within a set** but not globally. 1280x720 at 30fps. 10.7 GB. Camera mounting height is undocumented `[C]`. **Too small to pretrain on, at about 77 tracks, but CC0 plus written consent plus indoor multi-camera makes it an ideal clean validation and fine-tuning set with no legal overhead whatsoever.**

**Rejected, with reasons worth recording:**

| Dataset | Finding |
|---|---|
| **MTMMC** (CVPR 2024, 3,669 identities, 16 cameras, RGB and thermal) | Agreement obtained and read `[V]`. section 1(a) *"non-commercial, research, or academic purposes only"*; **section 2: derived *models* "can only be used for academic purposes"**; section 6.5 bars using the data *"to extract or process biometric identifiers or biometric information."* Plus audit rights, deletion on demand, and an **unauthorised-redistribution breach** that suspended access until Feb 2026. **Comprehensively barred.** |
| **MMPTrack** (Microsoft, 28 subjects, 5 indoor scenes, cross-camera identities) | Gated: signed terms and conditions emailed to the organisers; *"The datasets can only be used for research purpose"* `[V]`. Ethically good (*"paid and signed agreement to release their data"*), but research-only, and access runs through a dormant 2021 workshop mailbox. |
| **MVOR** (operating room, ideal geometry) | **CC-BY-NC-SA 4.0** `[V]`: *"the dataset cannot be used for commercial purposes."* Also only 732 frames and 3 cameras. |
| **IndoorCrowd** (arXiv 2604.02032, Apr 2026) | Doubly barred `[V]`: *"non-commercial computer vision research"*, and it *"prohibits surveillance and any re-identification of individuals."* Faces are pre-blurred. |
| **CrowdTrack** (2025, Apache-2.0 at source) | Genuinely Apache-2.0 `[V]`, but *"shot mainly from the first-person view"*, with 5,185 within-video trajectories. Wrong geometry, and the video provenance is undocumented. |
| **PeopleSansPeople** (Unity, Apache-2.0) | Apache-2.0 confirmed `[V]`, but the human assets are *"from RenderPeople"* (28 commercial scanned assets), and it emits only boxes, segmentation and COCO pose, with **no persistent identity labels**. Not a re-identification source. |
| **MTA** (GTA-V, about 2,800 identities) | `LICENSE.md` is **CC BY-NC 4.0** `[V]`, plus the same Take-Two overhang as MOTSynth. |
| **JRDB, SOMPT22, DIVOTrack, PETS2009, EPFL Terrace** | Research-only or no grant; JRDB is robot-mounted at about 1 m, eye-level, which is the wrong geometry. |

**RandPerson is the one near-miss worth a decision.** 8,000 identities, 19 cameras across 11 scenes, *"camera networks similar to real surveillance systems"* with varied height and angle, indoor and outdoor `[V]`. Critically, its characters come from **MakeHuman**, whose `LICENSE.md` puts bundled assets under **CC0** and states `[V]`: *"the MakeHuman project makes no claim whatsoever over output such as… Renderings"*, so the *person pixels* have genuinely clean provenance, unlike every GTA-derived set. However, the repository's Apache-2.0 covers only code (the root is `generateCode/`, `trainCode/` and `randperson.py`, while the data sits on Google Drive and Baidu with **no stated licence**) `[V]`, and the backgrounds were *"downloaded… from the Unity3D Asset Store"* `[V]`, which carries its own end-user licence agreement. Since re-identification training crops persons, tight cropping largely sidesteps the background exposure `[C]`. It is the only large synthetic identity pool with clean character provenance, so its licence position is worth resolving with counsel; the standing acquisition rule in POLICY.md, section 3 means no request is made to the authors.

**ClonedPerson must be rejected, and the reason is a false licence claim in a peer-reviewed paper.** The CVPR 2022 paper states `[V]`: *"We use both DeepFashion (Apache License 2.0) … and DeepFashion2 images for our virtual data creation."* **DeepFashion is not Apache-2.0.** Its CUHK MMLab page states `[V]`: *"The DeepFashion is available for non-commercial research purposes only"*, and users *"agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data."* ClonedPerson's cloned outfit textures **are** derived data. The repository also has **no licence file at all** `[V]`. It is contaminated at the root. UnrealPerson has the same structure, Apache-2.0 on code with no data grant `[V]`.

## Seven recurring traps, for the audit memo

1. **The GitHub LICENSE file covers code only, and this fired at least 6 times.** TrackEval, motsynth-baselines, MultiSports, TAO, PersonPath22, RandPerson and UnrealPerson all ship permissive LICENSE files scoped by their own words to *"this software."* An audit that greps for LICENSE files reaches the **opposite** of the correct conclusion.
2. **No licence is not a permissive licence.** MOTSynth, PathTrack, TAO, ClonedPerson, CAMPUS and the WILDTRACK page grant nothing. Under copyright, silence is default-deny.
3. **Licences vanish.** MOTChallenge's terms went to HTTP 410 while the ZIPs stayed downloadable. **Archive the licence page and the API metadata with a date and a hash for anything relied on.**
4. **The arXiv "view license" badge is not the dataset licence.** It misled on both ClonedPerson (CC0) and MTMMC (CC BY-NC-SA). It covers the *paper*.
5. **HuggingFace re-uploads launder licences.** `Lekim89/MOT17`, `ThanosNagda/msmt17-reid-dataset`, `ShantyCam/wildtrack`, and third-party `PhysicalAI-SmartSpaces` copies. A re-uploader cannot enlarge the original grant. Use first-party repositories only.
6. **A permissive licence does not cure a defective chain of title.** SportsMOT (broadcast footage), MOTSynth and MTA (Take-Two), ClonedPerson (DeepFashion), and PersonPath22's 57 external videos.
7. **Anti-biometric clauses target this exact use case.** PersonPath22, MTMMC section 6.5, IndoorCrowd, VIRAT's personally-identifiable-information clause, and NVIDIA's responsible-AI statement all restrict biometric and re-identification use *independently of* the copyright grant. Copyright clearance is necessary but not sufficient.

## Recommended path

1. **Pretrain on `nvidia/PhysicalAI-SmartSpaces`**, from HuggingFace only and never the challenge portal. About 3,674 synthetic identities, indoor CCTV geometry, hospital scenes, and zero privacy exposure. Ship a CC BY 4.0 NOTICE crediting NVIDIA and cite the 8th and 9th AI City Challenge papers, without implying endorsement.
2. **Fine-tune on MEVID** for real-world cloth-changing cross-camera signal. This is the path the lane took: MEVID is registered as bucket TRAIN_PROTOTYPE, status available, under a prototype grant decided 2026-09-07 (`configs/datasets/mevid.yaml`), and the re-identification model of record, OSNet-x1_0 checkpoint `66623239` and ONNX `60164a99`, was trained on it in the prototype weight class. The undeclared annotation licence stays an open counsel item on that entry rather than a resolved question. The face crops are not pending clearance: the actor check-in photographs package is barred outright under POLICY.md, rule 2 and is deliberately absent from the entry's acquisition urls.
3. **Validate on WiseNET** (CC0, consented, indoor), which carries no legal overhead at all.
4. **Resolve RandPerson's licence position with counsel**, since it is the only large synthetic identity pool with CC0 character provenance.
5. **Collect consented in-domain data.** For a hospital product this is the only fully clean route and the only source with the correct mounting height, scrubs, beds, wheelchairs and personal protective equipment, and the consent and review-board basis is needed for clinical and privacy review regardless.

**Four questions for counsel:** (a) NVIDIA's CC BY 4.0 grant against its stated non-authorisation for biometric and surveillance use; (b) whether the withdrawn MOTChallenge CC offer still binds prior downloaders; (c) VIRAT's destroy-all-derivatives-on-termination-without-notice clause against shipped weights; (d) whether MEVID's released face imagery can be processed for a commercial biometric purpose under BIPA and GDPR Art. 9.

**Method note:** search coverage was capped partway through, and the later findings rest on direct fetches against primary sources, which is the stronger method. Fetch failures are disclosed inline: `web.cs.ucla.edu` 403, `openaccess.thecvf.com` 403, `doc.bdd100k.com` and `www.bdd100k.com` DNS-dead, Rockstar's mod-policy page unreachable, and `makehumancommunity.org` unreachable, so the MakeHuman terms were taken from the project's own repository instead.
