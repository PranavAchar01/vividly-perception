# Licence due-diligence: synthetic person re-identification data for a commercial hospital-monitoring product

Evidence tags: `[V]` = the primary source was opened and quoted; `[R]` = secondary; `[C]` = inference.
Research date 2026-09-02. Quotes are literal. Not legal advice.

---

## 0. Outcome table

| # | Dataset | License | Asset provenance | IDs / images | Commercial training |
|---|---|---|---|---|---|
| 1 | **PersonX** | MIT `[V]` | Unity; models textured **by scanning real people** `[V]`; scenes not distributed `[V]` | 1,266 / 273,456 `[V]` | **Unresolved.** MIT is clean, provenance is not |
| 2 | **RandPerson** | Apache-2.0 `[V]` | MakeHuman and Unity3D; **5,772 scraped internet images** as clothing textures `[V]` | 8,000 / 1,801,816 `[V]` | **Unresolved.** Best permissive option, texture provenance unresolved |
| 3 | **UnrealPerson** | Apache-2.0 `[V]` | MakeHuman, Mixamo and UE4; clothing from **DeepFashion and CCP** `[V]` | 3,000 / 120k paper; 4.8k / 1.2M release `[V]` | **Rejected.** Imports DeepFashion's non-commercial bar |
| 4 | **ClonedPerson** | **No licence** `[V]` | MakeHuman community and Unity3D; outfits cloned from **DeepFashion and DeepFashion2** `[V]` | 5,621 / 887,766 `[V]` | **Rejected.** No grant, and a non-commercial source |
| 5 | **SyRI / SOMAset** | Not reachable `[V]` | UE4, or Blender and MakeHuman `[R]` | SyRI 100 / 56k to 1.68M `[V]`; SOMAset 50 / 100,000 `[V]` | **Unresolved** (obsolete anyway) |
| 6 | **MOTSynth / JTA** | MOTSynth: **none published** `[V]`; JTA: **CC BY-NC 4.0** `[V]` | **GTA V** `[V]` | MOTSynth 9,519 / 1.38M frames `[V]`; JTA 512 videos `[V]` | **Rejected** |
| 7 | **MTA** | Non-commercial declaration `[V]` | **GTA V** `[V]` | 2,800+ IDs, 6 cameras `[V]` | **Rejected** |
| 7b | **GPR+ / FineGPR** | GPL-3.0 badge **plus an explicit NC clause** `[V]` | **GTA5** `[V]` | 808 / 475,104 `[V]` | **Rejected** |
| 8 | **PeopleSansPeople** | Apache-2.0 `[V]` | **28 RenderPeople scans** `[V]` | 28 humans / 490k images `[V]` | **Rejected.** No identity labels, unlicensed humans |
| 9 | **NVIDIA PhysicalAI-SmartSpaces** | **CC-BY-4.0** `[V]` | Omniverse and Cosmos Transfer `[V]` | **2,481 + 292 + 901 persons**, 1,800 cameras `[V]` | **Licence permits, publisher intent objects. Since demoted on visual grounds: research/reid-smartspaces-mesh-audit.md excludes it for identity-discriminative embedder training** |
| 9b | NVIDIA SDG-SynHuman | OpenMDW-1.1 `[V]` | Omniverse `[V]` | 4,050 human assets, no boxes `[V]` | **Cleanest licence found** |
| 10 | **BEDLAM / AGORA** | MPI non-commercial plus **anti-surveillance** `[V]` | Commercial scans `[R]` | not recorded | **Rejected** |
| 11 | **MALS / CCUP** | MALS research-only `[V]`; CCUP none `[V]` | Diffusion `[V]` | MALS 1.5M; CCUP 6,000 / 1.18M `[V]` | **Rejected** |
| 12 | **Renderpeople** | Proprietary `[V]` | not applicable | not applicable | **Explicitly prohibits ML training** `[V]` |

---

## 1. The central finding: the GTA V chain is dead, and not marginally

Every GTA-derived set (JTA, MOTSynth, MTA, GPR/GPR+/FineGPR) fails at the root, and no downstream licence can cure it, because the dataset authors never held rights to grant.

**Rockstar Terms of Service** (rockstargames.com/legal, "Last Updated: February 28, 2025") `[V]`:

> §2.2: "we grant you a limited, non-exclusive, non-transferable, non-sublicensable, revocable license to access and use the Services … for your **personal, non-commercial enjoyment**."

> §2.3: "you may not, sell, copy …, distribute, … modify, create derivative works, **commercialize, or otherwise exploit the Services (including the Content)** … unless subject to separate, express written terms provided by Rockstar."

> §5.4: "**Rockstar reserves all rights to and ownership of all Custom Content**". §5.3 defines Custom Content to include "**screenshots, videos, recordings of in-game audio, gameplay clips**," adding "You may only use Custom Content with the Services and/or only as authorized by us."

> §6.1(3): "You will use the Services for your own personal, non-commercial use. **You will not commercially exploit the Services**."

There is also an AI clause, though its precise scope is narrow `[V]`:

> "nothing in the limited license … authorizes the use of the Services (including the Content) … to develop, train, enhance, or provide source material for … any **Generative AI Tools**; and any such uses are hereby explicitly prohibited."

**Nuance `[C]`:** "Generative AI Tools" is defined as technology that "create[s] or generate[s] content." A re-identification embedder is discriminative, since it emits a feature vector rather than content, so it arguably falls outside section 2.3's AI sentence. **That does not change the outcome.** section 2.3's "commercialize, or otherwise exploit," section 5.4's ownership reservation over screenshots, and section 6.1(3) independently and unambiguously bar the use.

**The fan-content policy does not rescue it** (support.rockstargames.com, "Last Updated: January 10, 2025") `[V]`:

> "Generally, Take-Two Interactive does not object to our fans using materials for **non-commercial uses**…"
> "'Non-commercial' means that you don't make money through the game footage you post or use the material as part of a promotion for a product or service."
> "this policy is intended to address **occasional, non-commercial in-game content use by individual members of the Rockstar fan community**."

**What the dataset publications actually assert.** None assert any right. The JTA paper's only statement is "by exploiting the highly photorealistic video game Grand Theft Auto V developed by Rockstar North", with **no** discussion of EULA, permission or legal basis anywhere in the paper `[V]`. MOTSynth is identical `[V]`. The obligation is instead pushed onto downloaders:

- **JTA** `[V]`: `LICENSE.md` is the unmodified CC BY-NC 4.0 text, plus the required declaration: "I declare that I will use the JTA Dataset for research and educational purposes only, since I am aware that **commercial use is prohibited**. I also undertake to purchase a copy of Grand Theft Auto V."
- **MTA** `[V]`: "With this email we declare that we will use the MTA Dataset for research and educational purposes only, since we are aware that **commercial use is prohibited**. We also undertake to purchase a copy of Grand Theft Auto V."
- **FineGPR** `[V]`: "The FineGPR Dataset and FineGPR-C caption are made available for **non-commercial purposes only**." The reach into trained weights is explicit: "You will not, directly or indirectly, reproduce, use, or convey the FineGPR dataset … **or any work product or data derived therefrom, for commercial purposes**." Its ethics section adds: "we do not consider the datasets for developing non-research systems without further professional processing or augmentation."
- **MOTSynth** `[V]`: **no licence at all.** The distribution page (motchallenge.net/data/MOTSynth-MOT-CVPR22/) was opened in a browser and returns zero matches for "license" or "terms". Only the *baselines code* repository is MIT. Absence of a licence is absence of a grant `[C]`.

**Outcome:** excluded. This is not a risk to price; it is a use the rightsholder has affirmatively prohibited in four separate clauses.

---

## 2. The second trap: DeepFashion contaminates ClonedPerson and UnrealPerson

Both datasets paint real clothing onto 3D humans. The clothing comes from a dataset that forbids exactly this.

**ClonedPerson** `[V]`: "For cloning clothes from real-world person images to virtual characters, we use images from both DeepFashion and DeepFashion2, with a total of **409k images** as our source data."

**UnrealPerson** `[V]`: "we use real-world clothing images of two datasets, **Clothing-co-parsing and DeepFashion**, as the clothing texture images for the generated humans."

**The DeepFashion agreement** (mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html, read in full via browser) `[V]`:

> "The DeepFashion is available for **non-commercial research purposes only**."
> "You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images **and any portion of derived data**."
> "All images of the DeepFashion are obtained from the Internet which are **not property of MMLAB**, The Chinese University of Hong Kong."

**The ClonedPerson paper misstates this.** Quoted from the paper `[V]`: *"We use both DeepFashion (Apache License 2.0) and DeepFashion2 images for our virtual data creation."* DeepFashion is **not** Apache-2.0; it is a signed, password-gated, non-commercial agreement. This is a factual error in a peer-reviewed CVPR paper, and relying on it would not survive diligence `[C]`.

"**Any portion of derived data**" is the operative phrase: a rendered 3D character wearing a cloned DeepFashion outfit is derived data on its face `[C]`. DeepFashion2 has **no licence at all** on its repository and is form-gated for the unzip password `[V]`. Clothing-co-parsing is Apache-2.0 `[V]`, but consists of "2,098 high-resolution street fashion photos" of real people whose copyright the authors never held `[C]`.

**ClonedPerson has no licence of its own.** Verified two ways `[V]`: the GitHub API returns `license: NONE`, and the repository root contains only `README.md, dataloader, img, pipeline`, with **no LICENSE file**. Default copyright, all rights reserved. **Excluded.**

---

## 3. The permissive candidates, and what is actually underneath them

### PersonX: MIT, but the models are derived from real human scans

`LICENCE` is the unmodified MIT text, "Copyright (c) 2022 sxzrt" `[V]`. The paper then says `[V]`:

> "In PersonX, the characters and objects are realistic, because the texture and material of these models are **mapped and generated from the real world by scanning real people and objects**."

> "PersonX has **1,266 hand-crafted identities** including 547 females and 719 males." … "The materials of the clothes (color and texture) are mapped from images of real clothes."

Two structural caveats `[C]`: (a) MIT licences "the Software," and the 3D models ship via Google Drive and Baidu, **not** in the MIT-licensed repository, so the grant's attachment to the models is arguable; (b) the scenes are deliberately withheld, and the README says "there are many free sources in Unity Asset Store. If you want to use the scenes used in this work, please send an email" `[V]`, which reads as the authors not holding redistribution rights to them. Scale: 1,266 identities, 273,456 boxes, 6 cameras `[V]`.

### RandPerson: Apache-2.0, the strongest permissive option, with one open thread

Apache-2.0 `LICENSE` in repository `[V]`. "The RandPerson dataset is generated by MakeHuman and Unity3D": 1,801,816 images, 8,000 identities, 19 cameras, 11 scenes; a 132,145-image subset on Google Drive `[V]`.

The provenance thread `[V]`: "**8,000 different 3D characters, including 114 characters with the original clothing models, 2,886 with UV texture maps from web images, and 5,000 with random UV maps.**" And: "The first is to search and download images from the Internet, and use them directly as new UV texture maps. Accordingly, **5,772 images are downloaded**."

So about 2,886 of 8,000 identities wear textures made from unlicensed scraped web images `[C]`. The paper's only permission statement is narrower than it first appears `[V]`: "**Some illustration figures in this paper** are created with additional public resources as labeled, thanks to the MakeHuman and Unity3D Asset Store where permissions are provided for free usage." That sentence is scoped to *figures in the paper*, not to the dataset assets.

**Mitigation `[C]`:** the 5,000 procedurally-random-UV characters carry no third-party imagery. A filtered subset is defensible; the full set is not.

### UnrealPerson: Apache-2.0, but DeepFashion-contaminated

Apache-2.0 confirmed via authenticated `gh api`; an earlier unauthenticated call returned `null` under rate-limiting, and the repository does have a LICENSE `[V]`. Download.md `[V]`: "Our synthesized data … is generated with **Makehuman, Mixamo, and UnrealEngine 4**. We provide 1.2M images of 4.8K identities, captured from 4 unreal environments." Paper: "We choose 4 scenes from **UE4 marketplace**" `[V]`; the toolkit requires three purchased UE Marketplace plugins `[V]`. Blocked by the DeepFashion chain in section 2.

### MakeHuman: the clean-rebuild path

This is the most important constructive finding. `LICENSE.md` `[V]`:

> "These assets have been released under **CC0 1.0 Universal**. … it is the intention of the MakeHuman project that anyone can do whatever they want with it."
> "As the assets have been released under CC0, **there is no limitation on what you can do with this combined output**."
> "the MakeHuman project makes **no claim whatsoever over output** such as: Exports to files (FBX, OBJ, DAE, MHX2...) … **Renderings** … We regard these things as your data, which is yours to handle as you see fit."

The code is AGPL; the **bundled** assets and all output are CC0. Note the carve-out `[V]`:

> "what is discussed here are only assets **bundled in the MakeHuman distribution**. If you use a third part asset, such as one downloaded from the asset repositories, it is your own responsibility to make sure you abide by its specific license."

That carve-out is precisely where RandPerson (179 community clothing models) and ClonedPerson ("3D clothes models available in the MakeHuman community") sit `[C]`. **Gap:** makehumancommunity.org was unreachable (connection refused), so the per-asset licences of those community clothing items could not be enumerated `[V]`.

---

## 4. PeopleSansPeople: Apache-2.0 that does not reach the humans, and no identity labels

**License confirmed as unmodified** `[V]`: Apache-2.0, "Copyright 2021 Unity Technologies."

**The humans are RenderPeople scans.** The README caption reads `[V]`: "our 3D human assets **from RenderPeople** with their original clothing textures." The repository contains `Assets/People/Controllers/renderpeople_master_01.controller`, and the shipped binary's product name is literally `HDRP RenderPeople 2020.1.17f1` `[V]`. Unity says outright, in the open-source environment README `[V]`:

> "**In lieu of 3rd party assets for which we do not own redistribution license**, we provide the following example assets/content:"
>
> - "4 Unity-branded assets with different clothing colour but the same appearance."
> - "529 Unity groceries textures."
> - "8 Unity-owned animation clips."

and in the paper `[V]`: "**due to RenderPeople redistribution and licensing policies, we do not provide direct access to the 3D human assets**."

So the git repository ships **4** Unity-owned humans, while the **28** RenderPeople humans exist only inside the prebuilt binaries. Apache-2.0 grants only what Unity had to give, and Unity states it did not have this `[C]`. The binaries README even contradicts itself: "licensed under the Apache License, Version 2.0 … **All rights reserved for Unity Technologies**" `[V]`.

**It does not emit persistent person identifiers.** Labels are only "2D bounding box, human keypoints, semantic and instance segmentation masks" plus 3D boxes `[V]`. Perception's `instance_id` is a per-frame handle from a monotonic counter assigned at `Awake()`; PSP re-samples which human spawns each iteration (`OnIterationStart`/`OnIterationEnd`) and re-randomizes clothing albedo, mask and normal every iteration: "Randomizes the material texture of tagged objects **at the start of each scenario iteration**" `[V]`. There is no cross-frame track identifier, no cross-camera association, and the generator is single-camera.

**Outcome:** taking that route would carry the RenderPeople exposure and return no identity labels. Excluded.

---

## 5. NVIDIA: the best technical fit, with a serious caveat

### PhysicalAI-SmartSpaces: CC-BY-4.0, cross-camera person identifiers, and a hospital scene

License confirmed two ways `[V]`: the HF API returns `cardData.license: cc-by-4.0`, and the README frontmatter is exactly `license: cc-by-4.0`. No LICENSE file, no gating, no click-through.

**It has real re-identification identity labels** `[V]`:

> "`<obj_id>`: **Consistent numeric identifier for each object across cameras**."

Annotation type is "2D & 3D bounding boxes, **multi-camera tracking IDs**" `[V]`. Scale `[V]`:

| Edition | Persons | Scenes | Cameras | 2D boxes |
|---|---|---|---|---|
| MTMC_Tracking_2024 | **2,481** | 90 | 953 | 135M |
| MTMC_Tracking_2025 | **292** | 23 | 504 | 73M |
| MTMC_Tracking_2026 | **901** | 28 | 353 | 33.2M |

"This dataset is synthetically generated with Omniverse and Cosmos Transfer" `[V]`, and the 2025 val split includes a `Hospital_000` scene `[V]`.

**Three caveats to weigh `[V]`:**

1. **NVIDIA's Croissant Responsible-AI metadata says**: "The dataset … **is not authorized for use in biometric, facial recognition, demographic profiling, or surveillance systems targeting real individuals**," and lists among risks "biometric re-identification … that disproportionately affects workers, **patients**, customers." CC-BY-4.0 is irrevocable and field-of-use-unrestricted, so as a copyright licence this cannot be clawed back `[C]`, but the publisher has placed this exact use case in its non-authorised set, in machine-readable form, and named patients. **This is a counsel item and must be cleared by counsel before the corpus is built on.**
2. Scenes 001/003/005/007/021/024 are Cosmos Transfer 2.5 re-renderings, and NVIDIA discloses that CT2.5 "is single-view rather than multi-view-consistent, so the same object may look subtly different across simultaneous camera views" `[V]`, which is a direct hazard to a re-identification objective `[C]`.
3. The 2026 edition includes two **real-world** test scenes (Warehouse_026/027) with signed consent, ground truth withheld `[V]`.

### Isaac Sim human assets: Apache-2.0 does not cover them

`github.com/isaac-sim/IsaacSim/LICENSE`, quoted `[V]`:

> "The Isaac Sim software in this repository is covered under the Apache 2.0 License terms below. Building or using the software requires additional components licenced under other terms. These additional components include dependencies such as the Omniverse Kit SDK, **as well as 3D models and textures**."

Those fall under the **Isaac Sim Additional Software and Materials License** (Last Modified June 9, 2025) `[V]`, which grants only "install and use copies of the Software in systems with NVIDIA GPUs," prohibits "**Modify or create derivative works of the Software**," and reserves "**no other license or right is granted to you by implication, estoppel or otherwise**". It contains **no** AI-training prohibition, but equally **no output grant** `[V]`. Under a closed grant with an express reservation, silence is not permission `[C]`. It also carries a Mission Critical Applications disclaimer naming "**medical, life support**" uses `[V]`.

### SDG-SynHuman: the most permissive licence found

`nvidia/PhysicalAI-WorldModel-Synthetic-Digital-Human-Scenes`, OpenMDW-1.1 `[V]`: "**This dataset is ready for commercial/non-commercial use.**" 4,050 unique digital human assets, 236,937 clips. OpenMDW-1.1, quoted `[V]`: "permission is hereby granted … **to deal in the Model Materials without restriction**" and "**This agreement does not impose any restrictions or obligations with respect to any use, modification, or sharing of any outputs generated by using the Model Materials.**" No boxes and no track identifiers, so labels must be supplied locally `[C]`.

---

## 6. Scanned-human vendors: the ML-training prohibition is explicit

**Renderpeople** (renderpeople.com/general-terms-and-conditions/, v. 13.12.2024) `[V]`, section 4.2 chapeau: "**The use of the 3D data for the following purposes is prohibited unless there is express written consent from Renderpeople.**" section 4.2(a):

> "**Use of the 3D data for Computer Vision Research. This includes but is not limited to (a) the creation of datasets, the development and training of neural networks and machine learning models, the development of real-time simulations**"

section 4.2(b) reaches derived data, "This includes … 3D data created with or based on products from Renderpeople", and defines third parties to include "companies associated with the Licensee's company, as well as subsidiaries" `[V]`. **This is the clause that explains why PeopleSansPeople ships without its humans.** It is a consent gate rather than an absolute bar, since Renderpeople sells a separate computer-vision and machine-learning licence.

**Others** `[V]`, except the AXYZ terms, which are secondary evidence `[R]` as section 10 records: the **AXYZ Extended License** uniquely lists "**AI (artificial intelligence) training**" as a permitted use and section 3.8 grants dataset distribution, which is the cleanest self-serve scan path. **HumanAlloy** sells a purpose-built AI-training licence over 4,700 real subjects **with signed model releases covering biometric data and third-party sublicensing**, and is the only vendor addressing biometric consent. **Reallusion** gates "Use for AI training and deep learning" to its Enterprise tier. **Unity Asset Store** (updated 04.12.2024) has the broadest bar: "training an artificial intelligence or machine learning model … for data sets … whether for commercial or non-commercial purposes." **Epic Fab** restricts only "NoAI"-tagged content and only for *generative* AI, with an express carve-out for programs that "generate tags to classify visual input content", which describes a re-identification embedder. **Daz3D**'s clause is tied to "auto-generating new content," so a discriminative embedder arguably falls outside.

**BEDLAM and AGORA** (MPI licence template) `[V]`:

> "To use the Data & Software for the sole purpose of performing **non-commercial** scientific research, non-commercial education, or non-commercial artistic projects"
> "Any other use, in particular any use for **commercial, pornographic, military, or surveillance, purposes is prohibited**."

AGORA adds explicitly `[V]`: "This license also prohibits the use of the Data & Software **to train methods/algorithms/neural networks/etc. for commercial, pornographic, military, surveillance, or defamatory use of any kind.**" That clause names this product category twice over. **Excluded.**

---

## 7. Diffusion-generated data (item 11)

Verified directly `[V]`: **MALS** (1.5M diffusion-generated image-text pairs) has an MIT code repository, but the README states "**Note that MALS can only be used for research; any commercial usage is forbidden.**" **CCUP** (6,000 identities, 1,179,976 images, 100 cameras, ICME 2025) has repository `license: NONE` and no grant.

**Gap, stated rather than guessed `[C]`:** the broader diffusion sweep was not run, so this review has no coverage of additional 2024-2026 diffusion re-identification datasets and did not read the generator-model licence texts themselves (CreativeML OpenRAIL-M Attachment A, SDXL RAIL++-M, SD3 Community License revenue threshold, FLUX.1 [dev] non-commercial). **The structural point stands independently `[C]`:** FLUX.1 [dev] is non-commercial and FLUX.1 [schnell] is Apache-2.0; SD3 and 3.5's Community License has a revenue threshold; OpenRAIL-M carries use restrictions that flow to outputs and derivatives. Any diffusion-generated corpus must be cleared against its *generator's* licence as well as its own, and a hospital monitoring application is exactly the kind of use OpenRAIL-style restrictions target. This remains an open question.

---

## 8. The transfer question: does synthetic pretraining actually work?

**Yes, and the published evidence is strong: synthetic sources beat real ones for direct transfer.**

**ClonedPerson (CVPR 2022), Table 2, "Direct cross-dataset evaluation results of models trained on different datasets with QAConv 2.0"** `[V]`:

| Training set | CUHK03-NP R1/mAP | Market-1501 R1/mAP | MSMT17 R1/mAP |
|---|---|---|---|
| CUHK03 (real) | n/a | 65.5 / 34.5 | 42.3 / 13.4 |
| Market-1501 (real) | 15.1 / 15.1 | n/a | 41.0 / 14.6 |
| MSMT17 (real) | 18.5 / 19.2 | 76.3 / 47.9 | n/a |
| **ClonedPerson (synthetic)** | **22.6 / 21.8** | **84.5 / 59.9** | **49.1 / 18.5** |

Synthetic ClonedPerson beats the largest real source (MSMT17) on Market-1501 by **+8.2 R1 and +12.0 mAP**.

**Table C, the full synthetic comparison** `[V]` (QAConv): RandPerson 8,000 identities and 1,801k images gives Market 74.5/46.9 and MSMT 40.6/14.0; RandPerson\* gives 81.6/56.4 and 46.8/17.6; UnrealPerson 3,000/120k gives 80.6/55.9 and 49.5/19.3; ClonedPerson 4,826/763k gives 84.5/59.9 and 49.1/18.5. With TransMatcher, ClonedPerson reaches **84.8/62.3 on Market** and **51.6/20.8 on MSMT17**.

**UnrealPerson (CVPR 2021), Table 3, direct transfer** `[V]`:

| Training set | Market R1/mAP | MSMT R1/mAP |
|---|---|---|
| Market (supervised) | *91.4 / 76.8* | 25.7 / 9.6 |
| MSMT17 (real) | 74.4 / 45.4 | *72.5 / 42.4* |
| SyRI | 48.5 / 22.6 | 21.8 / 5.7 |
| PersonX | 58.7 / 32.7 | 22.2 / 7.9 |
| RandPerson | 64.7 / 39.3 | 20.0 / 6.8 |
| **UnrealPerson** | **79.0 / 54.3** | **38.5 / 15.3** |

**Most relevant to the recipe used here, Table 4, pretrain then fine-tune** `[V]`:

| Pretraining | Fine-tune Market R1/mAP | Fine-tune Duke R1/mAP |
|---|---|---|
| ImageNet | 91.4 / 76.8 | 82.1 / 67.5 |
| MSMT17 (real) | 93.7 / 82.5 | 85.7 / 72.1 |
| **Unreal (synthetic)** | **94.0 / 84.7** | **86.8 / 74.2** |

Synthetic pretraining beats both ImageNet and real-data pretraining after fine-tuning.

**MOTSynth (ICCV 2021), re-identification on MOT17** `[V]`: Market1501-trained 64.6 mAP / 91.9 R1; MOTSynth split-1 71.3 / 91.4; split-4 **75.2 / 92.8**, a reported +6.9 mAP improvement over Market1501, with the reported conclusion that a re-identification model trained on MOTSynth is an adequate replacement for models trained on the real data.

**RandPerson (ACM MM 2020), weaker baseline** `[V]`: on Market, RandPerson 55.6/28.8 against PersonX 44.0/20.4 and SyRI 29.0/10.8; on MSMT17, RandPerson 20.1/6.3, SyRI 16.4/4.4, PersonX 11.7/3.6.

**Caveats `[C]`:** absolute numbers are strongly method-dependent (RandPerson on Market is 55.6 R1 in 2020 against 74.5 to 81.6 with QAConv 2.0 in 2022), so compare only within a table. MSMT17 transfer remains weak in absolute terms (about 49 R1) for every synthetic source. And the generalization gap that makes synthetic data win is a *domain-shift* result on street and campus surveillance; hospital scenes (gowns, wheelchairs, beds, staff uniforms, occlusion by equipment) are a further shift that none of these benchmarks measures.

---

## 9. Recommendations

1. **Exclude outright:** JTA, MOTSynth, MTA, GPR/GPR+/FineGPR (Rockstar ToS section 2.2/2.3/5.4/6.1); ClonedPerson (no licence, plus DeepFashion); UnrealPerson (DeepFashion); BEDLAM and AGORA (explicit anti-surveillance and anti-commercial-training); MALS (research-only); CCUP (no licence); PeopleSansPeople binaries (unlicensed RenderPeople humans, and no identity labels anyway).
2. **NVIDIA PhysicalAI-SmartSpaces is a counsel item, and it has since been demoted on technical grounds.** On licence and label coverage it was the best technical fit, with 2,481+ cross-camera person identifiers, a hospital scene and CC-BY-4.0, and NVIDIA's own metadata declares it "not authorized" for biometric re-identification and names patients as an affected group, so counsel must clear it before it is built on. The mesh-diversity audit of 2026-09-03 (research/reid-smartspaces-mesh-audit.md) then found roughly 35 to 45 distinct avatar builds recycled under 231 identity labels and excluded the corpus from identity-discriminative embedder training; research/reid-data-licences.md records that demotion. What remains permitted is the narrow set of roles the audit names.
3. **Best clean-build path:** MakeHuman **bundled** (CC0) assets, locally produced textures and locally produced Unity or Unreal scenes, rendered in-house. MakeHuman disclaims all output rights, so the result is owned outright. Pair with **AXYZ Extended** or **HumanAlloy** (which has biometric-scope model releases) for scan-grade identity diversity.
4. **RandPerson is the weakest-risk off-the-shelf permissive set**, at Apache-2.0 and 8,000 identities, provided the 2,886 identities built from scraped web textures are dropped. That filter leaves 5,114: the 5,000 with random UV maps plus the 114 with the original clothing models. The 5,000 carry no third-party imagery at all, which is the mitigation section 3 records; the 114 turn on the MakeHuman clothing question in the same section, because community assets fall outside the bundled-CC0 grant and their per-asset licences could not be enumerated. Restricting to the 5,000 alone is the conservative reading of the same filter.
5. **Two exposures no copyright licence resolves `[C]`:** biometric statutes (BIPA/TX/WA; GDPR Art. 9) attach to identity-matching of real people regardless of asset licensing, and PersonX's models are explicitly textured from **scans of real people**. Hospital deployment compounds this with a vulnerable population.
6. **Keep a per-frame provenance ledger** mapping every training image to its source asset, licence text and licence version-date. Terms are moving fast: Rockstar 28.02.2025, Renderpeople 13.12.2024, Unity Asset Store 04.12.2024, Fab 01.10.2024, Reallusion 01.08.2025, NVIDIA Isaac assets 09.06.2025.

## 10. Verified gaps

makehumancommunity.org was unreachable (connection refused), so the per-asset licences of the community clothing used by RandPerson and ClonedPerson are unverified. The SyRI and SOMAset dataset pages were not reached; their counts come from the PersonX and UnrealPerson comparison tables `[V]` and disagree with each other on SyRI's image count (1,680,000 against 56,000). The AXYZ end-user licence agreement was not opened directly, so its terms are recorded here as secondary evidence `[R]`. The 2024-2026 diffusion re-identification sweep and the generator-model licence texts (section 7) fall outside the coverage of this review and were not read.
