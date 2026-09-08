# Commercial licence due diligence: person datasets (2023 to 2026)
### For a hospital patient-monitoring computer-vision product

This is one specialist report in the re-identification licence set, from the pass run on 2026-09-02 and 2026-09-03.

`research/reid-licence-synthesis.md` is the lead report, and `research/reid-data-licences.md` is the synthesis of record.

Evidence tags: `[V]` = the primary source was opened and the licence read; `[R]` = secondary source; `[C]` = inference.

**Bottom line:** the commercially-licensed, identity-labelled person-data pool is very small. Three clusters clear the bar cleanly: **CHIRLA**, **TVRID**, and **MEVA / Visym CAP / PIP-175k**. Nearly everything the field actually trains on (Market-1501, MOT17/20, MSMT17, DanceTrack, JRDB, NTU RGB+D, BEDLAM, AGORA, SLP) is either explicitly non-commercial or **carries no licence at all**, and the unlicensed group is the larger hidden risk, not the smaller one.

---

## Priority 1: commercially permitted, with identity or tracking labels

| Dataset | Licence text | Scale / identities | Fit |
|---|---|---|---|
| **TVRID** (ICPR 2026) | **`"Creative Commons Attribution 4.0 International"`** on the Zenodo record **[V]**. No added usage restriction stated. | **86 to 88 identities** (the paper says 86, Zenodo says 88, and the discrepancy is flagged), 4 synchronized Intel RealSense D455, paired RGB+Depth 640x480, 20.9 GB. Ships both raw frames and pedestrian-centered crops. | **Best geometric match in the report.** Top-view (ceiling-mounted) RGB-D, framed explicitly as *privacy-preserving* re-ID. |
| **CHIRLA** (Nature Sci Data) | HF field `cc-by-4.0`; PMC confirms it covers the dataset not just the article: *"publicly available under the CC-BY license."* **[V]** | **963,554 identity-annotated bboxes** / 596,345 frames, **22 identities, 7 cameras**, 10 sequences over **7 months**, about 5.5 h at 30 fps, 1080x720, **includes clothing changes**. | Best long-term re-ID signal. Gated (`gated: auto`, contact info required). |
| **MEVA** | *"'Multiview Extended Video with Activities' (MEVA) dataset by Kitware Inc. and the Intelligence Advanced Research Projects Activity (IARPA) is licensed under a Creative Commons Attribution 4.0 International License"*, followed by the full unmodified CC BY 4.0 text, **no field-of-use restriction** **[V]** | 328 h ground video (516 GB) plus 4.6 h UAV; 64 h eval plus 120.2 h training annotations; EO plus thermal IR plus 2 drones | About 100 paid consenting actors, IRB. See the consented-actor section. |
| **Visym CAP** | **`"Creative Commons Attribution 4.0 International (CC BY 4.0)"`** **[V, two independent primary sources]** | 780 collectors / 33 countries; 288 GB plus 386 GB; **bbox tracks around primary actors** plus temporal segments | Third-person **looking down on the scene from above**, which matches hospital mount geometry. |
| **Visym PIP-175k** | **`"Creative Commons Attribution 4.0 International (CC BY 4.0)"`** **[V]** | 215,250 activity instances, 66 classes (subset of 37 MEVA activities), 150+ contributors / 44 countries | Composes cleanly with MEVA. |
| **CAUCAFall** | **`"CC BY 4.0"`** on Mendeley Data **[V]** | 10 subjects, 5 fall types plus 5 ADLs, 1080x960 at 23 fps, **per-frame segmentation labels** (YOLO-ready) | Uncontrolled home environment, occlusions, lighting change. |
| **Simuletic CCTV Fall / Lying-Down** | *"CC BY 4.0. You are free to use and adapt this data, provided you give appropriate credit to Simuletic."* **[V]** | 112 images, synthetic, bbox plus 17-kp COCO skeletons, about 95% fallen | The publisher states that no real individuals are depicted, which removes the GDPR and privacy exposure. Overhead CCTV. Directly on-point. |
| **Simuletic CCTV Aggressive Poses / Fight** | `cc-by-4.0` **[V]** | Under 1K synthetic, YOLOv8-Pose | Same pipeline. |
| **UP-Fall 3D Skeletons** (Zenodo) | **`"Creative Commons Attribution 4.0 International"`** **[V]** | 5 subjects, 2 camera angles | **Derived skeleton release only.** The original UP-Fall video licence is unverified and must not be assumed to flow. |
| **pat2echo fall/posture** | *"Creative Commons Attribution 4.0 International (CC BY 4.0)"* **[V]** | 113 images plus 10 videos (about 31 min), **1 subject** | The publisher states that no synthetic, scraped or third-party data is included and that every video and image is an original recording of the consenting subject. Essex ethics approval. Tiny but spotless. |
| **lookdeep/ai-norms-2024** | `CC0-1.0`, confirmed via GitHub API **[V]** | 300+ high-risk-fall hospital patients, 1,000+ patient-days | **Not training data.** It is an aggregated per-patient-hour prediction CSV, with no video or crops. Useful as a deployed-system benchmark. |

**Smaller permissive HF items (verified, mostly demo-scale):** `Voxel51/reid-people-tracking` (Apache-2.0, has `global_id`/`camera_id`, **101 samples**); `Mikiee/Person_Detection_and_Re-Identification_from_Low_Altitude_UAV` (cc-by-4.0, 1,624 imgs, **2 identities**); `issai/OpenThermalPose2` (MIT, thermal pose, relevant for privacy-preserving night monitoring); `vrg-prague/OCHuman-Pose` (MIT, occluded pose); `fei213/SurveillanceVQA-589K` (MIT).

**Permissive tag, unverified upstream. Do not rely on these:** `Voxel51/MPII_Human_Pose_Dataset` (bsd-2-clause) and `testdummyvt/cocohumanparts` (MIT) tag the *annotation packaging*, not image rights; MPII images came from YouTube, COCO from per-photo Flickr licences. The same applies to `MCG-NJU/Tracking-Any-Granularity` (cc-by-4.0, card states no source-video provenance) and third-party KAIST mirrors.

**Structural finding on Hugging Face:** the HF API was swept across `person re-identification`, `pedestrian`, `MOT`, `reid`, `surveillance`, `crowd`, `fall detection`, `patient`, `CCTV`, `gait`, `tracking`, `human pose`, `person detection`, plus licence-filtered sweeps of `task_categories:object-detection`. Outside CHIRLA, **every** permissively-licensed person dataset is under about 2,000 images or has 2 identities or fewer. The dominant commercial pattern is **a non-commercial sample on HF, with the real licence sold off-platform**.

---

## Priority 2: exclusion list

### Explicitly non-commercial, with the licence text

| Dataset | License text |
|---|---|
| **DanceTrack** | *"The dataset of DanceTrack is available for **non-commercial** research purposes only."* Split licence: annotations are CC BY 4.0, code is MIT, **the dataset is neither** **[V]** |
| **SportsMOT** | *"Creative Commons Attribution-NonCommercial 4.0 International License"* plus *"You agree not to distribute the SportsMOT dataset without prior written permission"* **[V]** |
| **PersonPath22** (Amazon) | *"The retrieved videos and their annotations are licensed under the CC-BY-NC-4.0 License."* **[V]** |
| **JRDB / JRDB-Pose / JRDB-Act** | *"Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License"* **[V]**; 60K frames, 2.4M 2D plus 1.8M 3D boxes, 600K poses |
| **BEDLAM** (MPI) | *"…sole purpose of performing non-commercial scientific research…"* plus *"Any other use, in particular any use for commercial, pornographic, military, or **surveillance**, purposes is prohibited."* Commercial path: `smpl@max-planck-innovation.de` **[V]** |
| **AGORA** (MPI) | Same, and further: *"This license also prohibits the use of the Data & Software to train methods/algorithms/neural networks/etc. for commercial, pornographic, military, **surveillance**, or defamatory use of any kind."* **[V]** |
| **SynBody** | *"SynBody is released under a CC BY-NC-SA 4.0 License."* **[V]**; 1.2M images, 10K body models |
| **NTU RGB+D / 120** | *"released for academic research only… non-commercial purposes"* plus *"any of the following will be considered illegal: redistribution, derivation or generation of a new dataset from this dataset, and commercial usage"* **[V]** |
| **SLP** (in-bed pose) | *"Non-commercial Purposes means research, teaching, scientific publication and personal experimentation"* plus *"**not intended for use by, with, or for the diagnosis of human subjects for incorporation into a product**."* **[V]** |
| **UR Fall Detection (URFD)** | *"licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License and is intended for non-commercial academic use."* **[V]** |
| **Project Aria Pilot** (Meta) | *"you may not use the Dataset to develop any product that is currently (or that you intend to make or that becomes) available for any commercial use."* **[V]** |
| **Casual Conversations v2** (Meta) | License field: **`"Limited; see full license language for use"`**; the page states that a model may be trained only on certain labels and refers to the licence permissions. Positioned as a fairness *evaluation* set. The full agreement is not retrievable, so it is **restricted, with commercial status unverified** **[V]** |
| **Simuletic non-commercial half** | `CCTV-Pedestrian-1K-Person-Attribute-Dataset`, `CCTV-Smoke-Fire`, `ATM_Robbery`, `Weapon_Rifles`, `Airport_Drone`, and 2 more, all `cc-by-nc-4.0` **[V]** |
| **UniDataPro face re-ID** | `cc-by-nc-nd-4.0` **[V]**; 670K images; **ND** also forbids derivatives |

> **Three traps worth naming explicitly.**
> **(a) SLP** is the in-bed pose dataset of record (109 subjects, 14,715 samples, RGB plus LWIR plus depth plus pressure, 7 hospital subjects) and is squarely off-limits: non-commercial *and* carrying a no-clinical-product clause.
> **(b) AGORA and BEDLAM** name **surveillance** as a prohibited purpose, and the clause reaches *training models* for it. A patient-monitoring product plausibly reads as surveillance, and even a negotiated MPI commercial license may not cover it, so the scope has to be settled explicitly.
> **(c) OmniFall.** Its arXiv page shows a **CC BY-SA 4.0** badge that reads as commercially usable. That covers the *paper*. The dataset's HF field is **`cc-by-nc-4.0`**: *"The annotations and split definitions are released under CC BY-NC-SA 4.0. The original video data belongs to their respective owners and should be obtained from the original sources."* **[V]** Doubly blocked: non-commercial annotations plus 8 constituent lab datasets each under its own mostly non-commercial license. 14,982 videos / 33.3 h / 32,840 segments, the most attractive fall benchmark going, and unusable.

### No licence at all, which carries higher risk than a non-commercial tag

Absent a licence grant, default copyright applies and there is **no commercial right**. Each primary page was opened and no grant was found:

- **Market-1501**: no licence, no terms. The page carries only a citation request and a contact address, `liangzheng06@gmail.com` **[V]**
- **MOT17**: the page carries only a note directing questions about missing files or dead links to `motchallenge-eol@vision.in.tum.de`, and nothing else **[V]**
- **MOT20**: no licence, no Creative Commons reference; only a TUM *Impressum* footer **[V]**
- **MOTSynth**: the repo's MIT licence covers **the baseline code, not the dataset**; no dataset licence located; **GTA V**-derived, so the EULA governs regardless **[V for absence]**
- **VisDrone**: no licence terms on the dataset repo **[V for absence]**
- **WILDTRACK** (EPFL, 7-cam multi-view pedestrian): no licence or commercial policy **[V for absence]**
- **SoccerNet / SoccerNet-Tracking**: no LICENSE file (GitHub API: `license: None`), and access is NDA-gated, with the site requiring a non-disclosure agreement form before any video data is downloaded. The NDA text is not public; the DevKit MIT licence covers code only; the underlying content is third-party broadcast footage **[V]**
- **ENTIRe-ID** (2024, **13,540 identities, 37 cameras, 4.45M images**): the repo contains only `README.md` and no LICENSE; the project site's CC BY-SA 4.0 notice **covers the webpage, not the dataset** **[V]**
- **AG-VPReID 2025** (3,027 IDs, 13,511 tracklets, 3.7M frames): no licence statement in the challenge paper **[V for absence]**
- **MSMT17**: the terms page is a frameset only; unverified
- **GTA-Human**: no authoritative licensed repo located; unverified; the same GTA V EULA problem applies
- **DukeMTMC / DukeMTMC-reID**: **withdrawn** over privacy and consent objections **[R]**. Any model trained on it is a live liability.

**MEVID sits in between.** 158 identities, 598 outfits, 8,092 tracklets (about 590 frames on average), 33 camera views, 17 locations. The repo has **no LICENSE file** and directs inquiries to `mevadata@kitware.com` **[V]**. Its source video *is* MEVA (CC BY 4.0, verified), so inheritance is plausible, but that is `[C]` inference and not a grant. It is otherwise the strongest identity-labelled, consent-backed re-ID corpus in existence, and the unstated annotation licence is the one gap to close with counsel.

---

## Priority 3: consented-actor datasets

**MEVA, the strongest provenance artefact in this review.** The homepage states: *"All MEVA data is available for use under a CC BY-4.0 license; the general MEVA data license is available here."* The linked licence file itself was pulled: it opens with the attribution statement above and then reproduces the **full, unmodified CC BY 4.0 legal text with no added field-of-use restriction** **[V]**. Consent posture **[R, WACV 2021 paper]**: IRB-overseen, about 100 actors over several weeks, informed-consent briefing in advance, and **the CC BY 4.0 licence was disclosed in the consent briefing itself**, so participants knew the downstream commercial use. That is exactly the artefact a hospital DPIA or procurement reviewer wants.

**Visym CAP**: CC BY 4.0 **[V]**; *"Every subject in this dataset has consented to their personally identifable information to be shared publicly for the purpose of advancing computer vision research."* Non-consented bystanders are face-blurred and plates anonymized. Visym **pays** collectors via an app that records and annotates simultaneously, capturing only consenting subjects by design. Free access: `aws s3 ls --no-sign-request s3://consented-activities-of-people/train/`.

**Visym PIP-175k**: CC BY 4.0, identical consent language **[V]**.

**pat2echo**: single consenting adult, named institutional ethics approval, CC BY 4.0. Trivial scale, but a validation set that can be cited without caveats.

**Consented but unusable:** Casual Conversations v1/v2 (restrictive licence), Project Aria (non-commercial).

---

## Priority 4: commercial vendors

**Nexdata**: [nexdata.ai](https://www.nexdata.ai/datasets/computervision), [datarade.ai/data-providers/nexdata/profile](https://datarade.ai/data-providers/nexdata/profile).
The published statement **[V]**: *"All the data are collected with proper authorization and with clear copyright, and **can be used for commercial purposes**."* Plus GDPR and CCPA compliance and ISO27001/ISO27701. Live catalog categories confirmed by scraping their CV index **[V]**: **Person Re-Identification**, Human Body Tracking, Human Body Attribute, Human Body Detection/Segmentation, Human Pose; the vertical list includes **Intelligent Healthcare**. Product line **[R, search-index snapshots]**, with the separator in each catalog title normalised to a comma: *"10,034 People, Re-ID Data in Real Surveillance Scenes"* (bboxes plus **15 body attributes**), *"11,130 People, Real Surveillance Re-ID"*, *"2,769 People CCTV Re-ID in Europe"*, *"208,914 Bounding Boxes, Person Attributes in Surveillance"*, *"21,404 Images, In-Home Human Posture and Activity Recognition"*.
**Every** product URL tried (`/966`, `/1038`, `/1129`, `/1132`, `/1160`, `/1348`) and `/terms` returns **HTTP 404** **[V]**; their HF org was renamed `Nexdata` to `Nexdata-AI` and **the re-ID datasets are gone from it** (the remaining samples are `cc-by-nc-sa-4.0`). The catalog has been restructured, and whether the products are retired or re-slugged is unknown. Also Chinese-origin (Nexdata/Datatang): PIPL and cross-border transfer questions stack on top of GDPR.

**Simuletic**: [simuletic.com](https://simuletic.com/datasets/cctv-fall-incident-detection-dataset), [pricing](https://simuletic.com/pricing). **The closest commercial fit to this product.** Overhead-CCTV synthetic humans, fall and lying-down, 17-kp COCO pose, YOLOv8 and YOLO11-Pose ready, marketed for *"detect patients falling in hospitals or care homes (privacy-safe)."* Free CC BY 4.0 samples allow evaluation before contracting. Tiers: Free / Student and Research / **Corporate and Business** (*"1,000 images for commercial use"*, *"Commercial licence"*, about $69/mo) / Custom Enterprise **[V]**.
**The actual commercial licence text is published nowhere**, so redistribution, model-training and derivative scope are all unverified. Note also the publisher's own disclaimer: *"Not intended as a sole basis for biometric identification, legal evidence, or safety-critical decisions without independent human review."* Patient fall detection is arguably safety-critical.

**SKY ENGINE AI / Omnihuman**: [omnihuman.ai](https://www.omnihuman.ai/), *"millions of photorealistic… synthetic individuals"* from 216 adult identities x 6 ethnicities x 5 age ranges x 60 hair variants; healthcare, human detection and crowd monitoring are listed **[V]**. No licensing or ownership terms published, so unverified.

**Anyverse**: [anyverse.ai](https://anyverse.ai/), a physically-accurate render engine, strong on in-cabin human sensing, which is the nearest analogue to indoor person monitoring. No published data licence, so unverified.
**Rendered.ai**: [rendered.ai](https://rendered.ai/), synthetic-data PaaS on NVIDIA Omniverse Replicator; sensor breadth over human-centric coverage. Terms unverified.
**Mindtech Global / Chameleon**: [mindtech.global](https://www.mindtech.global/), synthetic humans specifically for training **CCTV**; healthcare among verticals; In-Q-Tel backed. **[R]** acquired by **Synthera Corp, 2025-03-10**; current availability unverified. (A same-search hit about "Chameleon Technology" acquired 2025-07-02 is a **different** company, UK smart metering. Do not conflate the two.)

**Unidata**: [unidata.pro/datasets](https://unidata.pro/datasets/) **[V]**, genuinely consented sourcing (*"collected on the Prolific platform… ensuring participant consent"*), datasets labelled *"Commercial"* but **no licence text published**. The catalog is **biometric and anti-spoofing, not re-ID**, with no CCTV-angle or healthcare data. Low fit.

### Gone. Verify before anyone cites them
- **Datagen**: ceased operations **March 2024** despite about $20M remaining (raised $72M) **[R]**
- **Synthesis AI**: dissolved; acquired by **Globant, September 2025** **[R]**
- **AI.Reverie**: absorbed **[R]**
- **Protege**: `protege.ai` **307-redirects to a GoDaddy parked for-sale page [V]**
- **TrainingData.pro**: **ECONNREFUSED (84.54.37.150:443)** on repeated attempts **[V]**; site down

The single-purpose synthetic-human data vendor category has consolidated hard since 2024. Confirm a vendor is currently operating before building a data strategy on it.

### Services, not off-the-shelf
**Scale AI, Appen, Sama, Innodata, Defined.ai, Shaip** offer custom collection and annotation; several advertise healthcare verticals (Appen's *"Custom Human Data Services"* with credentialed medical professionals; Sama has annotated *"RGB security footage to identify people and doors"* **[R]**). **None** surfaced an off-the-shelf person re-ID or CCTV-angle product. These are **bespoke-commission routes**: the cleanest legal path, because the commissioner owns the output, and the most expensive. License terms not verified; Shaip's CV collection URL returned 404.

**Datarade marketplace** **[V]**: the video-data providers were enumerated (Nexdata, DataSeeds.AI, Unidata, Spire Light, InfoBay AI, Datoric, MFour, Avant-Garde, Babbl). **No surveillance, CCTV or healthcare video providers are listed** beyond Nexdata. Egocentric video is well covered (Datoric 100,000 h; Avant-Garde; Unidata 4,050 h), which is the wrong viewpoint for this product.

**Not found or not verified:** Getty, Shutterstock AI data and Cognata surfaced nothing relevant to person re-ID; Infinity AI / Infinity Fitness could not be located as an operating vendor. Treat these as unverified rather than absent.

---

## Recommendation

**Use now:** TVRID (top-view RGB-D, best geometry match), then MEVA plus CAP plus PIP-175k (consented backbone), then CAUCAFall plus the Simuletic free samples (fall-specific).

**Gated, and therefore in the same category as the contract routes below:** CHIRLA. The licence is clean and the fit is the best long-term re-identification signal in this report; what stops it is the gate rather than the licence, since the download is `gated: auto` with contact information required, and the acquisition rule in POLICY.md, section 3 refuses gated sets outright and forbids completing a sign-up or submitting a form. It is recorded as an option for an owner decision, not as work to schedule.

**Contract routes, which sit outside this repository's acquisition channel:** Nexdata, Simuletic Business, SKY ENGINE, and bespoke commission via Appen, Sama or Scale. Every one of them requires contacting a publisher, which the acquisition rule in POLICY.md, section 3 forbids for any lane, so none of them is a step this repository takes. They are recorded here as options for an owner decision, not as work to schedule.

**Licence gaps to close with counsel rather than with the publisher:** MEVID (high value, unstated annotation licence), BEDLAM (a commercial licensing route exists, but the surveillance carve-out has to be settled), and the UP-Fall source corpus. Under the acquisition rule in POLICY.md no dataset owner is contacted; the ambiguity is recorded in the registry entry and resolved by counsel at ship time.

**Highest-priority internal audit:** the *unlicensed* set, that is Market-1501, MOT17/20, MSMT17, MOTSynth, VisDrone, WILDTRACK, ENTIRe-ID, SoccerNet, and especially **withdrawn DukeMTMC**. If any pretrained backbone in the stack came from these, and most public re-ID and MOT checkpoints did, that exposure exists **today**, independent of what is licensed going forward.

**Two clauses to put in front of counsel:** AGORA and BEDLAM prohibit training for *surveillance* by name; SLP explicitly excludes incorporation into a diagnostic product. Both plausibly reach a hospital patient-monitoring system.
