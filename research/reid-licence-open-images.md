# Open Images V7: commercial licensing due diligence

This is one specialist report in the re-identification licence set, from the pass run on 2026-09-02 and 2026-09-03.

`research/reid-licence-synthesis.md` is the lead report, and `research/reid-data-licences.md` is the synthesis of record.

## Correction before anything else

**No sentence in which Open Images disclaims owning the copyright of the images appears anywhere on the Open Images site.** Every page was checked. Wording of that shape was anticipated in advance and is not present, so it is not quotable: using it would put a fabricated quotation into a legal memo. The actual disclaimer is different and is reproduced in full below.

## 1. The licence statements, in full

The entire licensing disposition of Open Images is **one three-sentence block**. No terms page, no EULA, no click-through, no licence file in the data buckets.

**[V]** `https://storage.googleapis.com/openimages/web/factsfigures_v7.html`, section heading `Licenses` (anchor `#licenses`), complete and unedited:

> "**Licenses**
> The annotations are licensed by Google LLC under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. The images are listed as having a [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) license.
> **Note:** while we tried to identify images that are licensed under a Creative Commons Attribution license, we make no representations or warranties regarding the license status of each image and you should verify the license for each image yourself."

That is the whole thing.

- **[V] "Annotations" is one undifferentiated blanket grant**: boxes, segmentations, image-level labels, visual relationships, localized narratives, and point-level labels. **There is no separate or differing licence for any annotation type** anywhere on the page.
- **[V] Byte-identical across V4, V5, V6, V7.** All four pages were fetched and diffed. V2 and V3 pages have no `Licenses` section at all.
- **[V] `factsfigures.html` (unversioned) is stale and serves the V6 page**, banner "latest version of Open Images (V6 - released Feb 2020)". Cite `factsfigures_v7.html`.

**[V] Pages containing no licence text** (each fetched, then grepped for licen, copyright, warrant, CC BY):

| Page | Result |
|---|---|
| `download_v7.html` | **No licence section at all**, only a sample CSV row showing a `License` column. This is where an engineer actually goes. |
| `index.html` | nothing |
| `extended.html` (MIAP) | nothing |
| `news.html` | nothing; no takedown or removal announcements |
| `github.com/cvdfoundation/open-images-dataset` | **No licence text whatsoever**, and this CVDF S3 mirror hosts the actual 561 GB of images most commercial users download |
| `raw.githubusercontent.com/openimages/dataset/master/LICENSE` | Apache 2.0, "Copyright 2016 The Open Images Authors", which **covers the downloader code, not the data.** Do not let this be mistaken for a data licence. |

## 2. Image selection by licence, audited rather than taken from the prose

**[R]** V4 paper (`https://arxiv.org/abs/1811.00982`): "they have been collected from Flickr without a predefined list of class names or tags, leading to natural class statistics and avoiding an initial design bias" and "The images have a Creative Commons Attribution license that allows to share and adapt the material".

**[V] Full enumeration of Google's own image-metadata CSVs:**

| File | Rows | CC BY 2.0 |
|---|---|---|
| `.../2018_04/train/train-images-boxable-with-rotation.csv` | 1,743,042 | **100.00000%** |
| `.../2018_04/validation/validation-images-with-rotation.csv` | 41,620 | **100.0000%** |
| `.../2018_04/test/test-images-with-rotation.csv` | 125,436 | **100.0000%** |

Every row carries `https://creativecommons.org/licenses/by/2.0/`; landing hosts 100% Flickr. **Zero NC / ND / SA / All-Rights-Reserved contamination** in the boxed-image universe (1,910,098 images). The recorded licence is uniformly the most permissive CC variant, the one that expressly permits commercial use.

## 3. Person-class counts

Google **publishes no per-class box counts**, and there is no Person row on the facts page. The official annotation CSVs were stream-counted instead.

**Method validated [V]:** the split totals reproduce Google's published Table 2 exactly: train 14,610,229, val 303,980, test 937,327.

**Person (`/m/01g317`):**

| Split | Boxes | Images | group-of | depiction |
|---|---|---|---|---|
| train | 1,034,721 | 248,384 | 58,748 | 112,674 |
| validation | 16,753 | 7,144 | 2,233 | 1,857 |
| test | 53,385 | 21,688 | 6,858 | 6,524 |
| **Total** | **1,104,859** | **277,216** | 67,839 | 121,055 |

**Full person family, all splits:**

| Class | MID | Boxes | Images |
|---|---|---|---|
| Person | /m/01g317 | 1,104,859 | 277,216 |
| Man | /m/04yx4 | 1,453,318 | 393,161 |
| Woman | /m/03bt1vf | 786,655 | 276,669 |
| Boy | /m/01bl7v | 92,143 | 48,391 |
| Girl | /m/05r655 | 211,817 | 102,381 |
| Human body | /m/02p0tk3 | 221,299 | 67,987 |
| **Human face** | /m/0dzct | **1,060,312** | 344,043 |
| Human head | /m/04hgtk | 238,296 | 64,272 |
| **Sum** | | **5,168,699** | union **867,737** |

**[C]** Person alone = **6.97%** of all 15,851,536 V7 boxes; the person family = **32.61%**. A third of Open Images' entire box budget is human, so a person-class diligence failure contaminates a third of the dataset.

**Annotation caveats [V], quoted from the facts page:**
> "We have drawn bounding boxes for human body parts and the class "Mammal" only for 95,335 images, due to the overwhelming number of instances (1,327,596 on the 95,335 images)."

> "we capped four categories: "clothing" to 441k instances, "person" to 149k, "woman" to 117k, "man" to 114k. In total we annotated segmentation masks for 769k instances of "person"+"man"+"woman"+"boy"+"girl"+"human body"."

**[V] Reproducibility note:** V7's download page serves train boxes from the `/v6/` bucket and val and test boxes from `/v5/`. Box annotations are unchanged since V5; V7's addition is point-level labels. The V7 person boxes are literally the V5 and V6 person boxes.

**[V] V7 totals:** 9,178,275 images (train 9,011,219 / val 41,620 / test 125,436); 15,851,536 boxes on 600 classes; 1,910,098 boxed images; 2,785,498 segmentations on 350 classes; 3,284,280 relationships; 675,155 localized narratives; 66,391,027 point labels; 61,404,966 image-level labels on 20,638 classes.

## 4. Person identity and tracking labels: none present

**[V]** Header row of every annotation schema:

| File | Schema |
|---|---|
| train bbox | `ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,XClick1X..XClick4Y` |
| segmentation | `MaskPath,ImageID,LabelName,BoxID,BoxXMin,BoxXMax,BoxYMin,BoxYMax,PredictedIoU,Clicks` |
| VRD | `ImageID,LabelName1,LabelName2,XMin1..YMax2,RelationshipLabel` |
| image labels | `ImageID,Source,LabelName,Confidence` |
| point labels | `ImageId,X,Y,Label,EstimatedYesNo,Source,YesVotes,NoVotes,UnsureVotes,TextLabel` |

**No identity field, no person ID, no track ID, no re-identification key, no name.** `BoxID` is a within-image index, not an identity. Still images only; there is no Open Images video or tracking release (Google's AVA and YouTube-8M are separate projects with separate terms).

## 5. MIAP, the person-only extension, with a flag

**[V]** `extended.html`, quoted: "We provide additional annotations for 100,000 (70k from training and 30k from validation/test) images that contain at least one person bounding box in the original annotations." Attributes are "the human perceived gender presentation... and perceived age range".

**[V] Counted directly:** 472,186 boxes on 100,000 images (train 373,386/70,000; val 23,940/7,410; test 74,860/22,590). 100% `LabelName=/m/01g317`. Schema adds only `GenderPresentation,AgePresentation`, and still **no identity field**.

**[V] Google's explicit stated position, quoted:**
> "Note, we do not support or condone the building or deployment of gender and/or age classifiers from this dataset."

**[C]** Not a licence term (MIAP has no separate licence; it falls under the blanket CC BY 4.0), so not legally binding, but it is documented licensor intent, and shipping a commercial gender or age classifier trained on it is a reputational and EU AI Act (biometric categorization) exposure.

**[V]** Localized Narratives has its own grant (`google.github.io/localized-narratives`): "All the annotations available through this website are released under a CC BY 4.0 license." Note that LN includes **annotator voice recordings** (OGG), which is human voice data with a separate privacy character.

## 6. Licence rot, measured here because no published study exists

**No quantitative study of Open Images licence accuracy exists.** A search found none, and that absence is recorded rather than guessed around. The nearest published work (Harvey, `adam.harvey.studio/creative-commons/` **[R]**) covers COCO and MegaFace but **does not mention Open Images**; its finding is that the dominant CC failure in ML datasets is *omitted attribution*, not NC misuse ("over 99% required attribution, for which 0% was provided").

Rot was therefore measured live (2026-09-02):

**[V] Link rot, n=520 random validation images, HEAD on `OriginalURL`:** **76 dead (14.6%)**, of which 65 answered 404 and 11 answered 410. 95% CI approximately 11.6% to 17.6%.

**[V] License drift, n=200 random Flickr landing pages, parsing live licence from page HTML:**

| Current state | Count | % |
|---|---|---|
| Still CC BY 2.0 | 167 | 83.5% |
| Now CC BY 4.0 (upgraded, still permissive) | 1 | 0.5% |
| HTTP 404, gone | 24 | 12.0% |
| HTTP 403, private, restricted or throttled | 6 | 3.0% |
| Reachable, no licence in HTML | 2 | 1.0% |

**Of 168 readable licences: 167 still CC BY 2.0, 1 upgraded, zero downgrades.** Rule-of-three 95% upper bound on the downgrade rate: **~1.79%**.

*Caveats on the record:* validation split only; n=200 for the licence check; the six 403 answers are ambiguous (genuinely private against Flickr throttling) and were not disambiguated; sub-1.8% is not zero.

## 7. The clause that settles the relicensing worry

**[V]** CC BY 2.0 legal code section 7 (`creativecommons.org/licenses/by/2.0/legalcode.en`), quoted:

> "Subject to the above terms and conditions, the license granted here is perpetual (for the duration of the applicable copyright in the Work). Notwithstanding the above, Licensor reserves the right to release the Work under different license terms or to stop distributing the Work at any time; provided, however that any such election will not serve to withdraw this License (or any other license that has been, or is required to be, granted under the terms of this License), and this License will continue in full force and effect unless terminated as stated above."

**[C]** A CC BY 2.0 grant already made is **irrevocable**. A photographer flipping to All Rights Reserved in 2027 does not retroactively strip a grant validly made in 2016; deletion from Flickr does not revoke it either. The 14.6% rot is an **availability and attribution-verification** problem, not a rights problem.

**[V]** CC BY 2.0 deed: "copy and redistribute the material in any medium or format for any purpose, **even commercially**... The licensor cannot revoke these freedoms as long as you follow the license terms." And, critically: "No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as **publicity, privacy, or moral rights** may limit how you use the material."

**[V]** CC BY 2.0 section 5: "LICENSOR OFFERS THE WORK AS-IS AND MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE WORK... INCLUDING, WITHOUT LIMITATION, WARRANTIES OF TITLE, MERCHANTIBILITY, FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT..."

**[C] The warranty stack is empty end to end:** the uploader disclaims title warranty to Google under section 5; Google disclaims warranty to the downloader on the facts page. **Nobody in the chain warrants title.** All residual risk lands on the downstream commercial trainer.

## 8. Bottom line [C]

1. **Annotations are clean.** CC BY 4.0 from Google LLC, commercial use permitted, attribution to Google required. Low risk; attribute in the model card.
2. **The recorded image licence is uniformly permissive**: 100.000% CC BY 2.0 across all 1,910,098 boxed images, verified by full enumeration. No NC filtering needed.
3. **The irreducible risk is that Google's disclaimer is literally true**: nobody verified per-image that the Flickr uploader actually held the copyright they purported to license ("Flickr washing"). Google says so and directs the downloader to verify. That is not performable at 1.9M scale, and nobody has published an error rate for Open Images.
4. **Relicensing is legally a non-issue** (section 7 irrevocability), empirically 0/168 downgrades, ceiling ~1.8%.
5. **Rot at ~14.6% degrades future re-verification.** Snapshot the image-metadata CSVs now: they carry `Author`, `AuthorProfileURL`, `Title`, `OriginalLandingURL`, exactly what CC BY attribution requires and exactly what vanishes as pages 404.
6. **Attribution is the obligation most likely to be breached** and the one the field routinely fails. Whether training weights is a "Derivative Work" requiring per-image attribution is unsettled; the conservative posture is to ship the attribution manifest.
7. **Person-specific overlay:** ~1/3 of all boxes, identifiable people from consumer Flickr, and copyright licensing conveys **no publicity or privacy rights**, as CC says itself. That is a separate diligence track (BIPA, GDPR, EU AI Act) the CC BY analysis does not reach. If MIAP age or gender attributes are in play, Google has said on the record that it does not condone building those classifiers.

Primary sources opened: [factsfigures_v7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), [download_v7](https://storage.googleapis.com/openimages/web/download_v7.html), [index](https://storage.googleapis.com/openimages/web/index.html), [factsfigures (stale V6)](https://storage.googleapis.com/openimages/web/factsfigures.html), [extended/MIAP](https://storage.googleapis.com/openimages/web/extended.html), [extras](https://storage.googleapis.com/openimages/web/extras.html), [news](https://storage.googleapis.com/openimages/web/news.html), [CVDF mirror](https://github.com/cvdfoundation/open-images-dataset), [openimages/dataset LICENSE](https://raw.githubusercontent.com/openimages/dataset/master/LICENSE), [CC BY 2.0 deed](https://creativecommons.org/licenses/by/2.0/), [CC BY 2.0 legalcode](https://creativecommons.org/licenses/by/2.0/legalcode.en), [Localized Narratives](https://google.github.io/localized-narratives/), [V4 paper](https://arxiv.org/abs/1811.00982), [Harvey, CC Biometrics](https://adam.harvey.studio/creative-commons/), annotation CSVs under `storage.googleapis.com/openimages/{v5,v6,v7,2018_04,open_images_extended_miap}/`
