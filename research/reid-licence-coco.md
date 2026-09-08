# COCO image licensing and permissive person datasets: commercial due diligence

This is one specialist report in the re-identification licence set.

`research/reid-licence-synthesis.md` is the lead report, and `research/reid-data-licences.md` is the synthesis of record.

**Prepared 2026-09-02.** Evidence tags: **[V]** = the primary source was opened and read or quoted directly; **[R]** = secondary source; **[C]** = inference or reasoning, not a quote. Not legal advice.

## Bottom line

COCO's annotations are cleanly licensed. **The images are not licensed by COCO at all.** COCO's own Terms of Use disclaim ownership and point at Flickr. The actual distribution, computed from COCO's own annotation files: **68.87% of train2017 images and 69.20% of train2017 person images carry NonCommercial licences**, and **32.35% of person images carry NoDerivs**. Filtering to commercially-usable, derivative-permitting licences leaves **16,513 of 64,115 person images (25.8%)** and **69,372 of 262,465 person crops (26.4%)**. Compositing crops from the other 74% or so into a commercial product was the single largest licence exposure measured here. The crop-and-composite module against which that exposure was measured is not in this repository, so every passage below about compositing is a record of that measurement rather than a description of a current step.

**Open Images V7 is the clean substitute**: **100.00%** of its validation images were verified as CC BY 2.0 with **0.00% missing attribution metadata**. **MEVA** supplies consented actors and closed-circuit viewpoints under CC BY 4.0; the **MEVID** identity layer built on that video ships no licence statement of its own, as section 2.2 records, so the CC BY 4.0 grant covers the MEVA video only.

---

# Part 1: the COCO image licence situation

## 1.1 Annotations: CC BY 4.0 [V]

The Terms of Use are not on the rendered page (it is a JS tab app); the content lives at `https://cocodataset.org/dataset/termsofuse.htm`, which was fetched directly. The text in full:

> **"Annotations & Website**
> The annotations in this dataset along with this website belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License."

## 1.2 Images: COCO explicitly conveys nothing [V]

From the same page, the section that governs the whole exposure:

> **"Images**
> The COCO Consortium does not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use. The users of the images accept full responsibility for the use of the dataset, including but not limited to the use of any copies of copyrighted images that they may create from the dataset."

Note the last clause: it anticipates that the user will create copies, that is crops and composites, and pushes that risk entirely onto the user.

Also stated at `https://cocodataset.org/dataset/download.htm` [V]: **"By downloading this dataset, you agree to our Terms of Use."**

Supporting facts, each verified against the source:
- The `annotations_trainval2017.zip` central directory contains **exactly 6 members, all `.json`. There is no LICENSE file** in the distribution. [V]
- `images.cocodataset.org/LICENSE`, `/LICENSE.txt`, `/README.txt`, `/license.txt` all return **404**. [V]
- Images are hosted under a **DMCA notice-and-takedown regime**, not an affirmative licence. CVDF's takedown policy [V]: *"the Common Visual Data Foundation ("CVDF") may be required to remove content it hosts in response to an official DMCA request."* CVDF's stated mission is *"to enable open community-driven research in computer vision through the creation of **academic** datasets"* [V].

## 1.3 The 8 licence ids, copied from COCO's own JSON [V]

This table was extracted directly from `annotations/captions_train2017.json` inside the official zip (a byte-identical array appears in `image_info_unlabeled2017.json`):

| id | name | url |
|---|---|---|
| 1 | Attribution-NonCommercial-ShareAlike License | `http://creativecommons.org/licenses/by-nc-sa/2.0/` |
| 2 | Attribution-NonCommercial License | `http://creativecommons.org/licenses/by-nc/2.0/` |
| 3 | Attribution-NonCommercial-NoDerivs License | `http://creativecommons.org/licenses/by-nc-nd/2.0/` |
| 4 | Attribution License | `http://creativecommons.org/licenses/by/2.0/` |
| 5 | Attribution-ShareAlike License | `http://creativecommons.org/licenses/by-sa/2.0/` |
| 6 | Attribution-NoDerivs License | `http://creativecommons.org/licenses/by-nd/2.0/` |
| 7 | No known copyright restrictions | `http://flickr.com/commons/usage/` |
| 8 | United States Government Work | `http://www.usa.gov/copyright.shtml` |

All 8 URLs currently resolve (HTTP 200). [V] Note every licence is **version 2.0**, not 4.0. This matters (see section 1.8).

## 1.4 The distribution: exact counts, computed from primary data [V]

**No published source carries these numbers, so they were computed here.** Method: HTTP range requests against `annotations_trainval2017.zip` to read the zip central directory, then fetching and inflating only the needed members in memory (18.8 MB for `captions_train2017.json`, 71 MB for `person_keypoints_train2017.json`). Read-only; nothing modified.

### train2017, all images (n = 118,287)

| id | licence | count | % |
|---|---|---:|---:|
| 1 | CC BY-NC-SA | 32,878 | 27.80% |
| 2 | CC BY-NC | 16,397 | 13.86% |
| 3 | CC BY-NC-ND | 32,184 | 27.21% |
| 4 | CC BY | 19,470 | 16.46% |
| 5 | CC BY-SA | 10,901 | 9.22% |
| 6 | CC BY-ND | 5,992 | 5.07% |
| 7 | No known copyright restrictions | 465 | 0.39% |
| 8 | US Government Work | **0** | 0.00% |

- **NonCommercial (1,2,3): 81,459 = 68.87%**
- **NoDerivs (3,6): 38,176 = 32.27%**
- Commercial-permitted (4,5,6,7,8): 36,828 = 31.13%
- **Commercial and derivatives permitted (4,5,7,8): 30,836 = 26.07%**
- ShareAlike (1,5): 43,779 = 37.01%

### train2017, person images (n = 64,115), the numbers that matter here

| id | licence | count | % |
|---|---|---:|---:|
| 1 | CC BY-NC-SA | 17,913 | 27.94% |
| 2 | CC BY-NC | 8,950 | 13.96% |
| 3 | CC BY-NC-ND | 17,506 | 27.30% |
| 4 | CC BY | 10,322 | 16.10% |
| 5 | CC BY-SA | 5,809 | 9.06% |
| 6 | CC BY-ND | 3,233 | 5.04% |
| 7 | No known copyright restrictions | 382 | 0.60% |
| 8 | US Government Work | 0 | 0.00% |

- **NonCommercial: 44,369 = 69.20%**
- **NoDerivs: 20,739 = 32.35%**
- **Commercial and derivatives permitted: 16,513 = 25.76%**

### train2017, person instances and crops (n = 262,465)

- NonCommercial: 180,031 = **68.59%**
- NoDerivs: 84,333 = **32.13%**
- **Commercial and derivatives permitted: 69,372 = 26.43%**

### Other splits, for completeness [V]

| subset | n | NC | ND | keep {4,5,7,8} |
|---|---:|---:|---:|---:|
| val2017 all | 5,000 | 69.50% | 33.20% | 1,279 (25.58%) |
| val2017 person | 2,693 | 68.88% | 31.86% | 696 (25.84%) |
| train+val all | 123,287 | 68.89% | 32.31% | 32,115 (26.05%) |
| unlabeled2017 | 123,403 | 68.74% | 32.08% | 32,441 (26.29%) |

The distribution is remarkably stable across every split and is essentially identical for person and non-person images. **License id 8 (US Government Work) is declared but used by zero images anywhere in COCO 2017.** [V]

## 1.5 Cross-validation of the computation [V]

Three independent checks, all exact:

1. **Person instance count.** The stream parsed **262,465** person annotations, matching COCO's published figure for train2017 exactly.
2. **GitHub issue #81 (2017).** A commenter reported *"there are 123403 images with license annotations in the JSON, but only 6614 (about 5%)..."* The unlabeled2017 counts above give licence 6 = 6,140 plus licence 7 = 474, summing to **exactly 6,614**. (The commenter mislabelled these as "unrestricted or USgov"; the arithmetic confirms the file is unchanged since 2017.)
3. **exposing.ai person total.** The train2017 person images (64,115) plus val2017 person images (2,693) = **66,808**, matching exposing.ai's independently derived "Total Flickr Photos: 66,808" for COCO's person class exactly. [V]

## 1.6 The one published third-party audit: exposing.ai [V]

`https://exposing.ai/coco/` and `https://adam.harvey.studio/creative-commons/`. The underlying data file was pulled rather than the chart, because a summarizer conflated the audit's Flickr-wide table with its COCO table.

From the audit:
> "Based on an analysis of only the "person" class, which is the category most closely associated with potential surveillance applications, **57,245 out of 59,144 images used a Creative Commons license, and of those 57,245 Creative Commons licenses over 99% required attribution, for which 0% was provided.**"

> "**COCO provides a clear example of large-scale CC violation by its clear omission of attribution, denying the creators of 328,000 images of their legally required credit.** ... It may also be considered a commercial use of non-commercial images."

The audit's raw `licenses.csv` (Flickr API, October 2019), for COCO's person class:

| licence | count |
|---|---:|
| CC BY-NC-SA | 16,325 |
| CC BY-NC-ND | 15,396 |
| CC BY | 9,150 |
| CC BY-NC | 8,386 |
| CC BY-SA | 5,127 |
| CC BY-ND | 2,861 |
| License status unknown | 376 |
| Public Domain | 52 |
| CC Zero | 51 |
| US government work | 22 |

Derived (all arithmetic reconciles to the audit's stated totals): NC = 40,107 (**67.81%** of 59,144); ND = 18,257 (**30.87%**); commercial and derivative permitted = 14,402 (**24.35%**). **These live-Flickr figures agree with COCO's own recorded metadata to within about 1.5 percentage points**, which independently validates both.

The audit's `copyrights.csv` shows the rot: of 59,144 still-live person photos, **1,398 (2.36%) had become "Copyright" (all rights reserved)**, and separately **7,664 of the original 66,808 (11.47%) had been removed from Flickr entirely** by October 2019.

## 1.7 Licence drift in 2026: original measurement [V]

The audit's data is seven years old, so the drift was measured fresh here. COCO images whose **recorded licence is commercially safe** were sampled and each photo's **current** licence re-checked using Flickr's own embedded licence field.

*(A first attempt gave a spurious 20% "all rights reserved" rate; the string "All Rights Reserved" appears in Flickr page boilerplate even on CC BY pages, and the page read was truncated at 400 KB on 522 KB pages. The corrected method is the one reported below.)*

| sample | n | same licence today | now ARR or NonCommercial | deleted/private |
|---|---:|---:|---:|---:|
| val2017 person, ids 4/5/7/8 | 150 | 76.0% | **18.0%** [11.9 to 24.1] | 4.0% |
| train2017 all, ids 4/5 | 200 | 70.5% | **23.5%** [17.6 to 29.4] | 3.5% |
| **pooled** | **350** | **72.9%** | **21.1%** | 3.7% |

Brackets are 95% confidence intervals. Drift is dominated by photographers switching to All Rights Reserved (61 of the 74 drifted).

**This does not void the granted rights, but it destroys the evidence for them.** CC BY 2.0 section 7 [V]: *"the Licensor reserves the right to release the Work under different license terms or to stop distributing the Work at any time; provided, however that any such election **will not serve to withdraw this License**."* The 2014 grant survives. But roughly one in five of the images relied on now shows a restrictive licence to anyone who looks, and **COCO's stored licence id is a third-party assertion, not a signed record.** [C] Snapshot and archive the provenance evidence now; it degrades every year.

## 1.8 What the CC 2.0 terms actually say [V]

The readings below come from the operative legal codes, not the summary deeds.

**NonCommercial** (CC BY-NC 2.0 section 4(b)):
> "You may not exercise any of the rights granted to You in Section 3 above **in any manner that is primarily intended for or directed toward commercial advantage or private monetary compensation.**"

**NoDerivs** (CC BY-ND 2.0 section 3). The grant is *"to reproduce the Work, to incorporate the Work into one or more Collective Works... to distribute copies..."* Compare CC BY 2.0 section 3, which additionally grants *"**to create and reproduce Derivative Works**."* **The ND grant simply omits the derivative-works right.** [C] On the plain text of the 2.0 licences, cropping a person and compositing them into a synthetic scene is outside the ND grant entirely, not merely a restriction on sharing.

That reading is **stricter than Creative Commons' own** [V], which says the ND restriction is *"triggered only when works or adaptations of works are publicly shared."* The conservative posture is to exclude ND (32.35% of person images) regardless.

**Attribution** (CC BY 2.0 section 4): a licensee who distributes or publicly displays the Work or any Derivative Work must *"give the Original Author credit reasonable to the medium or means You are utilizing by conveying the name (or pseudonym if applicable) of the Original Author if supplied; the title of the Work if supplied; to the extent reasonably practicable, the Uniform Resource Identifier..."*

**Termination, and the trap** (CC BY 2.0 section 7):
> "This License and the rights granted hereunder will **terminate automatically upon any breach** by You of the terms of this License."

**CC 2.0 has no cure period.** CC 4.0 section 6(b) added one [V]: rights reinstate *"automatically as of the date the violation is cured, provided it is cured within 30 days of Your discovery of the violation."* So COCO's **annotations** (CC BY 4.0) are curable; the **images** (all CC 2.0) are not. [C] An uncured attribution failure terminates the image licence permanently and turns a licensee into an infringer.

## 1.9 COCO ships zero attribution, but it is recoverable [V]

A COCO image record contains no author field at all:

```
{"license": 4, "file_name": "000000397133.jpg", "coco_url": "...",
 "height": 427, "width": 640, "date_captured": "2013-11-14 17:02:52",
 "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
 "id": 397133}
```

The `flickr_url` points at the CDN, not the photo page, so the photographer is not directly derivable. It **is** recoverable via `https://www.flickr.com/photo.gne?id={photo_id}`, which redirects to `/photos/{username}/{id}/`. Tested on 4 images: **4/4 resolved** (maggiew, larrylawfer, ensh, lusciousblopster). The method comes from GitHub issue #81, comment dated 2025-05-30, which also supplies a `jq` filter for commercial-only selection. Cost is roughly one HTTP request per image, so about 16,500 requests for the filtered person subset.

## 1.10 The claim that Flickr prohibits commercial use is wrong [V]

This is repeated widely, including in COCO issue #551. The Flickr Terms of Use do contain the restrictive language:

> "you may not use the Services or the Flickr Materials to engage in any of the following prohibited activities: the collection, copying or distribution of any portion of the Flickr Materials; any **resale, commercial use, commercial exploitation**, distribution... modifying or otherwise making any derivative uses...; **scraping** or otherwise using any data mining, robots or similar data gathering or extraction methods..."

But "Flickr Materials" is a defined term, and the definition excludes user photos:

> "The Services and all images, software, platforms, tools, graphics, data, text, code, the Marks... and other content and materials available on the Services **(excluding User Content)**... (collectively, the "**Flickr Materials**")"

And separately: *"**You retain all intellectual property rights in and to any User Content you post**... including the copyright in and to your photos and videos. SmugMug does not claim any ownership."*

**The Flickr ToU contains no mention of "Creative Commons" anywhere.** [V] So COCO's instruction to "abide by the Flickr Terms of Use" points at a document that says nothing about the licences that actually govern the photos. [C] The real terms are the per-photo CC 2.0 licences. The anti-scraping clause is also largely academic in this case: COCO did the scraping in 2014, and the pixels arrive from Google Cloud, not Flickr.

License id 7 ("No known copyright restrictions") is also not a warranty. Flickr Commons [V], with the source's own capitalisation retained: *"IF YOU MAKE USE OF A PHOTO FROM THE COMMONS, YOU ARE REMINDED TO CONDUCT AN INDEPENDENT ANALYSIS OF APPLICABLE LAW BEFORE PROCEEDING WITH A PARTICULAR NEW USE."*

## 1.11 Authoritative commentary and legal framework

### The one peer-reviewed paper on exactly this question [V]

*"Can I use this publicly available dataset to build commercial AI software? Most likely not"*, ACM TOSEM 2022 (arXiv:2111.02374). The full text was extracted and read. On COCO:

> "**MS COCO dataset only licenses their annotations under a standard license, they refer to the Flickr terms of use for the license of the images contained in their dataset.**"

> "In the case of the MS COCO dataset, only the annotations are licensed under CC 4.0, since users of the dataset typically require the images along with the annotation, **using MS COCO dataset under any commercial scenario might result in potential license violation.** However, if only the annotations from MS COCO dataset is used to build commercial AI software, they are allowed to do so."

Its Table 3 marks MS COCO **not permitted** across all three commercial scenarios: distributing the dataset, releasing a product with embedded AI models, and commercializing model output. Overall finding: *"all of our studied datasets might not be suitable to build commercial AI software due to a high risk of potential license violations."* Result 5 specifically flags **modification**: *"4 out of the 6 studied publicly available datasets might result in potential license compliance violation if the dataset is modified... for CIFAR-10, ImageNet, MS COCO and Cityscapes."* The crop-and-composite step in this pipeline is exactly that.

### COCO maintainers have never answered the question [V]

Issue #551, *"Is it possible to use COCO dataset for training models for commercial purposes?"* (Nov 2021), drew one reply, from another user, saying only that the same question remained open with no clear answer anywhere. **No maintainer response.** The only maintainer statement on licensing anywhere (issue #81, 2017) is that all images were obtained under a Creative Commons licence from the Flickr website and that the licence of each image can be found in the annotation files, followed by a pointer to the DMCA takedown policy.

### Creative Commons' own position [V]

From `creativecommons.org/using-cc-licensed-works-for-ai-training-2/` and CC's detailed legal analysis PDF:

> "**CC licenses apply only when copyright permission is required. If exceptions or limitations apply, then the CC license terms don't apply.**"

> "The attribution (BY) condition, the ShareAlike (SA) condition, and the NoDerivatives (ND) restriction are **all triggered only upon public sharing**... **In contrast, the NonCommercial (NC) restriction applies to all uses requiring permission under copyright.**"

> "If AI training data includes the NonCommercial restriction, then following the NC restriction would require that **all stages, from copying the data during training to sharing the trained model, must not be for commercial gain**."

> "Following the NoDerivatives restriction would require that ND-licensed content not be used as training data."

CC also warns its own guidance is deliberately over-restrictive: *"following this guidance will almost certainly lead to **overcompliance** as a matter of copyright law and the Creative Commons licenses themselves. It assumes the most restrictive version of the facts and the law."*

[C] Note the structural implication: **if training is fair use, the NC condition never engages**, because the user was never operating under the licence. That is CC's own reasoning. But CC is not a court, and CC launched **CC Signals** in June 2025 [V] precisely because it concluded existing licence conditions do not cleanly govern AI training, which is a tacit admission the NC-binds-training theory is weak.

### Condition against covenant: why a CC breach is infringement, not just breach [V]

*Jacobsen v. Katzer*, 535 F.3d 1373 (Fed. Cir. 2008). The full opinion was retrieved via the Caselaw Access Project. The Federal Circuit **vacated** a district court holding that public-licence terms were mere covenants:

> "If, however, a license is limited in scope and the licensee acts outside the scope, the licensor can bring an action for copyright infringement."

> "The Artistic License also uses the traditional language of conditions by noting that the rights to copy, modify, and distribute are granted '**provided that**' the conditions are met."

> "**Copyright holders who engage in open source licensing have the right to control the modification and distribution of copyrighted material.**"

[C] CC BY 2.0 section 4 uses the identical structure: *"The license granted in Section 3 above is expressly made subject to and limited by the following restrictions."* So exceeding NC or ND scope is infringement, which exposes the user to statutory damages, fees, and injunctive relief rather than contract damages. Creative Commons appeared as amicus in *Jacobsen*.

### The only appellate authority construing NC, and why it does not help [V]

*Great Minds v. FedEx Office & Print Services*, 886 F.3d 91 (2d Cir., 21 March 2018). Full opinion retrieved. Holding:

> "We hold that, in view of the absence of any clear license language to the contrary, licensees may use third-party agents such as commercial reproduction services in furtherance of their own permitted noncommercial uses. **Because FedEx acted as the mere agent of licensee school districts** when it reproduced Great Minds' materials... we conclude that FedEx's activities did not breach the license or violate Great Minds' copyright. We therefore AFFIRM."

[C] **The rationale is agency, and it does not transfer.** This project would be the principal making commercial use for its own account, not a vendor executing a noncommercial licensee's instructions. *Great Minds v. Office Depot*, 945 F.3d 1106 (9th Cir., 27 Dec 2019) is the parallel case [V, citation verified via CourtListener; holding not verified].

### AI training case law, 2023 to 2026

| case | status | relevance |
|---|---|---|
| **Thomson Reuters v. Ross Intelligence** (D. Del., 11 Feb 2025) | **Decided; fair use rejected.** 3d Cir. No. 25-2153 **argued 11 June 2026, undecided** [V] | **Closest analogue that exists**: commercial, **non-generative**, intermediate copying to build a competing tool. Defendant lost. |
| **Bartz v. Anthropic** (N.D. Cal., 23 June 2025) | Decided; settled; **final approval 20 July 2026**, appeal being filed [V] | Training held *"exceedingly transformative"*, but *"piracy of otherwise available copies is inherently, irredeemably infringing."* |
| **Kadrey v. Meta** (N.D. Cal., 25 June 2025) | Decided on the record presented; case ongoing [V] | Introduces **market dilution**; expressly generative-specific. |
| **Getty v. Stability (UK)** [2025] EWHC 2863 (Ch), 4 Nov 2025 | Decided [V] | *"the model weights are not themselves an infringing copy and they do not store an infringing copy."* |
| **Getty v. Stability (US, D. Del.)** | **Voluntarily dismissed 18 Aug 2025**; no decision on the merits [V] | |
| NYT v. OpenAI; Andersen v. Stability; Concord v. Anthropic; Disney/Warner v. Midjourney | All **pending**, no decisions on the merits [V] | |

Key holdings from the opinions, reported rather than quoted [V]:
- *Ross* records that only non-generative AI was before the court, and holds that **the effect on a potential market for AI training data is enough** on its own.
- *Bartz* holds that the purpose and character of using copyrighted works to train large language models to generate new text was quintessentially transformative.
- *Kadrey* states that the decision does not stand for the proposition that Meta's use of copyrighted materials to train its language models is lawful, and stands only for the proposition that these plaintiffs made the wrong arguments.

**[C] Transfer analysis for a discriminative hospital CV model:**
- **Favorable:** *Kadrey*'s market-dilution theory, the strongest 2025 plaintiff theory, is generative-specific. A person detector floods no market with photographs. *Getty UK* suggests weights that do not store the works are a weak target.
- **Unfavorable:** *Ross* is the closest factual match and the defendant lost, and its *"potential market for AI training data"* holding transfers cleanly. A licensed person-imagery market plainly exists (see Part 2).
- **The distinctive exposure here:** no decided case involves a defendant that **created and retained visually recognizable derivative images**. Composites of this kind may retain the photographer's expression (pose, lighting, subject, styling). That is a conventional derivative-work claim, far less protected by *Bartz*-style transformativeness reasoning, and it gets materially worse if composites are ever distributed, shipped in an SDK, or used in marketing.

### EU and UK [V]

- **DSM Directive Art. 4(1)** permits commercial TDM on "lawfully accessible" works, but **Art. 4(3)** conditions it on the use "**not** [having] been expressly reserved by their rightholders in an appropriate manner, such as machine-readable means in the case of content made publicly available online."
- [C] **Does CC BY-NC count as an Art. 4(3) reservation?** Genuinely unsettled. For: CC licences are machine-readable (RDFa on Flickr), exactly what Recital 18 contemplates, and NC on its face reserves commercial use. Against: Art. 4(3) requires rights be expressly reserved, and an NC tag applied years before Art. 4 existed is not a targeted TDM opt-out. Assume the risk is live.
- **EU AI Act Art. 53(1)(c)** requires a policy to "identify and comply with... a reservation of rights expressed pursuant to Article 4(3)". [C] But Art. 53 binds providers of **general-purpose AI models**, defined in Art. 3(63) as displaying "significant generality" across "a wide range of distinct tasks". **A hospital person-detection model almost certainly falls outside this**, so Art. 53 likely does not apply. Do not let anyone assume it does.
- **UK CDPA s.29A** is **non-commercial research only**, and s.29A(2)(b) makes later commercial use of a s.29A copy infringing. Useless here. Note s.29A(5) voids contrary contract terms: where Parliament wanted an exception to override contract, it said so. There is no equivalent attached to US fair use.

## 1.12 The filtered-subset workflow, and exactly what survives it

Filtering is available. COCO records a per-image licence id, so the selection is one line. Verified `jq` recipe from GitHub issue #81 (2025):

```bash
jq -r '[.licenses[] | select(.name | test("NonCommercial") | not)] as $ok
       | .images[] | select(.license as $l | $ok | map(.id) | contains([$l]))
       | [(.flickr_url | split("/")[-1] | split("_")[0]), .file_name] | @csv' \
   instances_train2017.json
```

Note that filter keeps id 6 (CC BY-ND). To exclude ND as well, select ids **{4, 5, 7, 8}**.

**What remains** (all [V], computed here):

| subset | total | keep {4,5,7,8} | % | keep {4,5,6,7,8} incl. ND | % |
|---|---:|---:|---:|---:|---:|
| train2017 all images | 118,287 | **30,836** | 26.07% | 36,828 | 31.13% |
| train2017 person images | 64,115 | **16,513** | 25.76% | 19,746 | 30.80% |
| train2017 person crops | 262,465 | **69,372** | 26.43% | 82,434 | 31.41% |
| val2017 person images | 2,693 | 696 | 25.84% | 838 | 31.12% |

**Three caveats [C]:**
1. **ShareAlike.** 5,809 of the 16,513 kept person images are CC BY-SA. SA is triggered only on public sharing of an adaptation, so internal training is fine, but publishing the composites would oblige licensing them CC BY-SA. Dropping id 5 leaves **10,704 person images**, and dropping SA plus keeping only id 4 gives **10,322**.
2. **Attribution is mandatory and uncured.** Every kept image except the 382 under id 7 requires attribution. The manifest (section 1.9) has to exist before any distribution, and CC 2.0 offers no cure period.
3. **21% drift.** About one in five of the kept images now shows a restrictive licence on Flickr (section 1.7). The granted rights survive, but the evidence has to be archived.

---

# Part 2: permissively-licensed person datasets

## 2.1 Open Images V7: the clean substitute [V]

Google's exact statements, from `https://storage.googleapis.com/openimages/web/factsfigures_v7.html`:

> "**The annotations are licensed by Google LLC under CC BY 4.0 license. The images are listed as having a CC BY 2.0 license.**"

> "**Note: while we tried to identify images that are licensed under a Creative Commons Attribution license, we make no representations or warranties regarding the license status of each image and you should verify the license for each image yourself.**"

That disclaimer is the same risk-shift COCO makes. **The underlying data is nonetheless structurally different, and that was verified directly.** The V7 validation image metadata CSV (15 MB) was downloaded and every row counted:

- **41,620 / 41,620 = 100.00% carry `https://creativecommons.org/licenses/by/2.0/`.** Zero NC. Zero ND. Zero ARR.
- Attribution fields **`Author`, `AuthorProfileURL`, `OriginalLandingURL`, `Title`: 0.00% missing.**

The metadata schema ships everything CC BY requires:
```
ImageID, Subset, OriginalURL, OriginalLandingURL, License, AuthorProfileURL,
Author, Title, OriginalSize, OriginalMD5, Thumbnail300KURL, Rotation
```

[C] **This is the single most important operational difference in the whole report.** COCO supplies a licence integer and a dead CDN link. Open Images supplies the author name, the author's profile URL, the photo's landing page, the title, and the licence URL, for every image. CC BY compliance is a template render, not a 16,500-request scraping project. And there is no NC or ND problem to filter around at all.

**Scale, from Google's tables [V]:** about 9M images total; 16M boxes over 600 classes on 1.9M images; box images = 1,743,042 train / 41,620 val / 125,436 test.

**Person content, counted directly from the V5 box CSVs [V]:**

| split | Person boxes | images with 1 or more Person boxes | images with 1 or more person-related boxes |
|---|---:|---:|---:|
| validation | 16,753 | 7,144 (17.2%) | 11,724 (28.2%) |
| test | 53,385 | 21,688 (17.3%) | 35,471 (28.3%) |

Related classes present: Human body, Human arm, Human head, Man, Human leg, Human face, Human hand, Woman, Girl, Boy. The percentages above are computed against the split sizes stated earlier in this section, 41,620 validation and 125,436 test box images. [V] The train split has since been counted directly rather than extrapolated: `research/reid-licence-open-images.md` reports **1,034,721 Person boxes on 248,384 images**, a rate of 14.25%, all CC BY 2.0. That is roughly **15x the commercially-clean person imagery available from COCO**, whose clean person pool is 16,513 images and 69,372 crops, with attribution included. **Superseded:** this report originally extrapolated a 19.2% rate to the 1,743,042 train box images and estimated roughly 330,000 train images with Person boxes and on the order of 800,000 person boxes. Against the direct count that estimate is about 33 percent high on images and low by about a factor of 1.3 on boxes.

**Caveats [C]:** no person identity or tracking labels (still images only); Google's disclaimer means residual licence-rot risk remains, though none appeared in 41,620 validation rows; the validation split was verified exhaustively and the train split was not re-verified.

## 2.2 MEVA and MEVID: consented actors, CCTV viewpoints, person identities, CC BY 4.0 [V]

**This is the closest match to a hospital-monitoring use case that exists under a commercial licence.**

**MEVA** (`https://mevadata.org/`), created by Kitware and IARPA on the DIVA program:
> "**All MEVA data is available for use under a CC BY-4.0 license**"

The licence file at `https://mevadata.org/resources/MEVA-data-license.txt` is the full CC BY 4.0 International legal code, headed: *"'Multiview Extended Video with Activities' (MEVA) dataset by Kitware Inc. and the Intelligence Advanced Research Projects Activity (IARPA) is licensed under a Creative Commons Attribution 4.0 International License."* [V]

Scale: **328 hours (516 GB) of ground camera data** plus 4.6 hours of UAV data, hosted free via the AWS Public Dataset Program. Filmed with paid actors; the site publishes "Actor check-in photos" and GPS data associated with the actors. [V]

**MEVID** (`https://github.com/Kitware/mevid`, WACV 2023), a person re-ID identity layer on MEVA:
> "we label the identities of **158 unique people wearing 598 outfits taken from 8,092 tracklets, average length of about 590 frames, seen in 33 camera views** from the very large-scale MEVA person activities dataset."

> "4 outfits/identity vs. 2 outfits/identity in CCVID, 33 view-points across 17 locations... **10 million frames** vs. 3 million for LS-VID. Being based on the MEVA video dataset, we also inherit data that is **intentionally demographically balanced to the continental United States**."

Splits: train 104 identities / 485 outfits / 6,338 tracklets (30.5 GB); test 54 identities / 113 outfits / 1,754 tracklets (13 GB). Actor check-in photos separately available (600 MB).

**Caveat [V]:** the MEVID GitHub repo has **no LICENSE file** and its README has no licence section. GitHub's API reports `license: None`. MEVID annotations are distributed through MEVA's S3 bucket, so CC BY 4.0 is strongly implied but **not separately stated**. [C] Written confirmation of the annotation licence is the precondition for commercial reliance on MEVID. Under the acquisition rule in POLICY.md the publisher is not contacted, so the gap stays recorded in the registry entry and is resolved by counsel at ship time.

## 2.3 Verified non-commercial exclusion list [V]

Knowing what to exclude is as valuable as knowing what to use. All verified from primary sources:

| dataset | actual licence | verdict |
|---|---|---|
| **DanceTrack** | *"The annotations of DanceTrack are licensed under a Creative Commons Attribution 4.0 License. The dataset of DanceTrack is available for **non-commercial** research purposes only."* | **Excluded.** Identical annotations-vs-data trap as COCO. |
| **PersonPath22** (amazon-science/tracking-dataset) | *"This provided code is licensed under the Apache-2.0 License. **The retrieved videos and their annotations are licensed under the CC-BY-NC-4.0 License.**"* | **Excluded.** |
| **JRDB / JRDB-Pose** | *"Published under Creative Commons **Attribution-NonCommercial-ShareAlike 3.0** License."* | **Excluded.** |
| **BEDLAM** | *"Data & Software Copyright License for **non-commercial scientific research purposes**"* | **Excluded** despite being fully synthetic. |
| **AGORA** | *"Data & Software Copyright License for **non-commercial scientific research purposes**"* | **Excluded.** |
| **MOT17 / MOT20** | **No licence statement found** on motchallenge.net dataset pages | **Treat as no grant.** Absence of a licence is not permission. |

[C] The pattern is consistent and worth recording: **the tracking and re-ID literature is overwhelmingly non-commercial**, and several datasets deliberately license permissive *annotations* over restricted *data*, exactly as COCO does. A dataset's headline licence badge is never read without checking what it covers.

## 2.4 Gaps in Part 2 that were not closed

Discovery was capped partway through by the search limit for this pass. **Not covered at the time, and worth a follow-up pass:**
- Hugging Face Hub systematic sweep for permissively-licensed person tracking and re-ID datasets.
- Named commercial data vendors (Scale, Appen, Sama, Defined.ai, TrainingData.pro, Datarade listings) and synthetic-human vendors (Datagen, Synthesis AI, Mindtech, Anyverse, Rendered.ai, Infinity AI). **Purchasing is a legitimate option here, and no vendor was priced or verified.** Note that *Ross* makes the existence of this market a fact that cuts against a fair use defense.
- SoccerNet, SportsMOT, NTU RGB+D, MOTSynth, and the fall-detection sets (UP-Fall, Le2i, CAUCAFall), which are directly relevant to hospital monitoring. This gap has since closed: SportsMOT is reviewed in `research/reid-licence-mot-family.md`, MOTSynth in `research/reid-licence-synthetic.md`, SoccerNet, NTU RGB+D and UP-Fall in `research/reid-licence-permissive-recent.md`, and Le2i and CAUCAFall in `research/fall-dataset-licences.md`, the last two being registered entries (`configs/datasets/le2i_fdd.yaml`, `configs/datasets/caucafall.yaml`).
- Meta's Casual Conversations v2 consent terms.
- LAION v. Kneschke (LG Hamburg, 27 Sept 2024) and its appeal, the most on-point EU decision on Art. 3 against Art. 4 for image datasets.

One correction worth flagging: a citation to "ML Genealogy" surfaced during the preemption research and **does not appear to exist** (zero hits in a full-text opinions search). It is not carried into any memo.

---

# Recommendations

**Outcome, recorded after the fact.** Recommendation 2 was not adopted: the filtered subset became `coco_reid_filtered`, which DATASETS.md records under "Considered and not used" as a planned derived subset of COCO person crops that was never materialised. Recommendation 4 was not adopted: there is no Open Images entry under `configs/datasets/`, so the person corpus was never migrated. Recommendation 5 was adopted for MEVID only, registered on 2026-09-07 at bucket TRAIN_PROTOTYPE under a prototype grant with the undeclared annotation licence carried as an open counsel item (`configs/datasets/mevid.yaml`); MEVA itself was never registered. The reasoning below is left as written.

1. **COCO's stored licence id is not a grant of permission.** 68.9% of the corpus and 69.2% of person images are NonCommercial. If any of that reached commercial weights, the exposure is quantified now; the counts in section 1.4 give the exact denominator.
2. **If COCO is kept, filter to ids {4, 5, 7, 8}** and accept 16,513 person images and 69,372 crops. Drop id 5 as well if composites may ever be published, leaving 10,704.
3. **Build the attribution manifest before shipping anything.** CC 2.0 has no cure period; an uncured attribution failure terminates the licence permanently. Archive current Flickr licence evidence immediately, since it is drifting at roughly 21%.
4. **Migrate the person corpus to Open Images V7.** 100% CC BY 2.0, complete attribution shipped, no NC or ND to filter, and roughly 15x more commercially-clean person imagery. This is the highest-leverage single change available.
5. **Add MEVA/MEVID** for identity labels and realistic surveillance viewpoints under CC BY 4.0, with the unstated MEVID annotation licence recorded in the registry entry as an open question for counsel.
6. **Keep composites internal.** Every restrictive condition that is triggered by *sharing* (ND, SA, and the attribution obligation itself) stays dormant while the composites never leave this infrastructure and only weights ship. *Getty UK* supports the view that weights not storing the works are a weak target.
7. **Refer the crop-and-composite step to counsel specifically.** It is the part of the pipeline with no favorable precedent, because no decided case involves retained, visually recognizable derivative images.
