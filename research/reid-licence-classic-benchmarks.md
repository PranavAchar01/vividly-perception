# Licence and ethics review: classic person re-identification benchmarks

This is one specialist report in the re-identification licence set, from the pass run on 2026-09-02 and 2026-09-03.

`research/reid-licence-synthesis.md` is the lead report, and `research/reid-data-licences.md` is the synthesis of record.

Context: commercial hospital-monitoring product, re-identification embedder training.

**Bottom line: of the 16 datasets examined, none is cleanly usable for commercial model training. One (SYSU-MM01) has a written path to a negotiated commercial licence. Everything else is either explicitly non-commercial, or has no licence at all, which is worse rather than better.**

---

## Summary verdict table

The table has 18 rows and covers 16 datasets: DukeMTMC is one row covering reID and
VideoReID, and i-LIDS (underlying) and LUPerson-NL are variants of the rows above them.
Every count of 16 elsewhere on this page uses that rule.

| Dataset | Commercial training | Redistribution | Licence exists? | Worst finding |
|---|---|---|---|---|
| Market-1501 | **Prohibited** | Prohibited | readme.txt only, not on project page | Tsinghua campus, no consent |
| DukeMTMC (+reID, VideoReID) | **Prohibited** (CC BY-NC-SA) | Permitted-but-ShareAlike, moot | Yes | **Retracted April to June 2019; IRB violated** |
| MSMT17 | **Prohibited** | Prohibited (even modified) | Yes (signed agreement) | Official page 404; PKU campus |
| CUHK03 | **Silent, no licence** | Silent | **No** | No grant of rights at all |
| CUHK-SYSU | **Silent, no licence** | Silent | **No** | Street snaps plus movies and television |
| MARS | **Silent, no licence** | Silent | **No** | Auto-detected distractors |
| iLIDS-VID | **Prohibited** | Prohibited | Yes (one sentence) | Derivative of Crown-copyright CCTV |
| i-LIDS (underlying) | **Prohibited** | Prohibited | Yes (6-pp EULA) | UK airport CCTV; **no longer obtainable** |
| PRID2011 | **Silent, no licence** | Silent | **No** | EU street surveillance, GDPR |
| LTCC | **Prohibited** | Prohibited | Yes (signed agreement) | No consent statement found |
| PRCC | **Prohibited** | Silent / all rights reserved | Yes (page notice) | No consent statement |
| RegDB | **Unclear**, terms unretrievable | Unclear | Behind unretrievable form | Terms gated, copies circulate off-channel |
| SYSU-MM01 | **Prohibited without prior approval**, negotiable | Prohibited without prior approval | Yes (signed agreement) | **Best consent posture in the set** |
| CCVID | **Prohibited** (via FVG upstream) | Prohibited upstream | **No licence on CCVID itself** | Upstream FVG terms appear breached |
| VC-Clothes / Real28 | **Unclear, misapplied licence** | Unclear | Apache-2.0 *code header*, misapplied | GTA V EULA exposure |
| Celeb-reID | **Prohibited** | Prohibited | Yes (README note) | Web-crawled celebrity photos |
| LUPerson | **Prohibited** | Silent | README sentence only, **no LICENSE file** | **YouTube scrape, 4M non-consenting people** |
| LUPerson-NL | **Silent, no licence** | Silent | **No** | Same YouTube scrape, 10M images |

---

## 1. Market-1501

**Sources opened:** `https://zheng-lab-anu.github.io/Project/project_reid.html` (200) [V]; `https://raw.githubusercontent.com/niruhan/market1501/master/Market-1501-v15.09.15/readme.txt` [V]; `http://www.liangzheng.com.cn/...`, which returned a **DNS failure** [V]

**Critical structural finding:** the official project page carries **no licence text whatsoever**. A grep of the rendered page for licen, commercial, distribut, copyright, terms and restrict returned zero hits [V]. The operative terms exist **only inside the `readme.txt` shipped in the dataset zip**:

> "This dataset should be used for research only. Please DO NOT distribute or use it for commercial purpose."

[V, read raw, character-exact]

- **Commercial training: prohibited** [V]
- **Redistribution: prohibited** [V]
- **Ethics:** the readme records collection with six cameras, in an open environment, at Tsinghua University [V, readme]. Supermarket forecourt, DPM-detected passers-by. No consent, no IRB, no notice mentioned anywhere. No retraction.
- **Note on scepticism:** because the restriction lives only in a zip readme, most downstream users never see it, and every mirror strips it. That does not weaken it, since it is still the author's stated term, but it explains why so many commercial actors have used this dataset unaware. The author moved to ANU, and the ANU GitHub Pages site is his own, so it is authoritative despite the dead original domain. [C]

---

## 2. DukeMTMC and DukeMTMC-reID: the retraction record

**Sources opened:** `http://vision.cs.duke.edu/DukeMTMC/`, which returned a **DNS failure, curl exit 000** [V]; `https://exposing.ai/duke_mtmc/` [V]; `https://arxiv.org/pdf/2108.02922` (Peng et al., NeurIPS 2021 D&B) read in full [V]; licence files at `sxzrt/DukeMTMC-reID_evaluation` and `Yu-Wu/DukeMTMC-VideoReID` [V]; EdScoop [R]

### The licence, exact text from `LICENSE_DukeMTMC.txt` [V]

> "The DukeMTMC dataset is made available under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license."

> "NonCommercial — You may not use the material for commercial purposes."

> "ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original."

> "If you publish data derived from DukeMTMC, the original dataset creators should first be notified. Any future dataset derived from DukeMTMC should include "DukeMTMC" in the title, as well as link to the original project website (http://vision.cs.duke.edu/DukeMTMC/)."

`LICENSE_DukeMTMC-reID.txt` and `LICENSE_DukeMTMC-VideoReID.txt` carry **identical CC BY-NC-SA 4.0 terms**, each stacking their own attribution protocol on top of Ristani et al. [V]

### Who took it down, when, why

- **Trigger:** the MegaPixels and Exposing.ai report by Adam Harvey, together with a *Financial Times* investigation. Exposing.ai records that, in response to that report and that investigation, Duke University terminated the Duke MTMC dataset in May 2019 [V].
- **Takedown actor: Duke University**, not the authors alone. The Exposing.ai log records 2 June 2019 as the date on which Duke University appears to have shut the Duke MTMC dataset project down [V].
- Peng et al. date the retraction to **April 2019**, with the apology in June [V]. Both dates hold: removal in April, public apology in June.
- **Why:** dataset was being used by SenseTime/SenseNets and Chinese military academies. Exposing.ai names CloudWalk, Hikvision, Megvii (Face++), SenseNets, SenseTime, NUDT, and the PLA Army Engineering University as acknowledged users [V].
- **Collateral:** the multi-target multi-camera tracking and person re-identification challenge workshop was cancelled, the organisers citing unforeseen circumstances and the test data not being available [V, recorded on exposing.ai from reid-mct.github.io/2019].

### The authors' statement

A statement by one of the dataset's authors appeared in the *Duke Chronicle* on **14 June 2019** [R, EdScoop reporting the Chronicle; the Chronicle original was not reached]. It recorded a belief that IRB guidelines were being followed on the video-analysis project, accepted full responsibility for the mistakes, and apologised to everyone who was recorded and to Duke for the consequences.

**Two IRB conditions breached** [R, EdScoop]: (1) recording was cleared by the IRB for **indoor locations only**, but was conducted at **eight outdoor campus locations**; (2) data was to stay with the researchers and be shared **only on direct request**, but was **posted publicly** on Duke's website. The same statement said that posters had been placed at all entrance points to the recording area, as directed [R].

Peng et al. add a governance point [V]: a creator of DukeMTMC apologised in June 2019, noting a violation of IRB guidelines, but that explanation did not appear in official channels.

### Were the derivatives withdrawn? **No.**

Peng et al. record [V] that the two most popular derivatives, DukeMTMC-ReID and DukeMTMC-VideoReID, are still available for download, and that both contain a cropped and edited subset of the videos from DukeMTMC.

Table 2 [V]: for the original, no locations were found where it is still available; for the derivatives, two derived datasets remain available with images from the original; and the original page returns a DNS error, which was independently reproduced in this pass [V].

Seven or more derivatives catalogued: DukeMTMC-ReID, DukeMTMC-VideoReID, DukeMTMC-Attribute, DukeMTMC4ReID, DukeMTMC Group, DukeMTMC-SI-Tracklet, Occluded-DukeMTMC [V]. Exposing.ai adds **QMUL-SurvFace**, so the Duke recordings were carried onward into a *face recognition* dataset [V]. For Occluded-DukeMTMC, the dataset itself is no longer available, but a script that converts DukeMTMC-ReID, which is still available, into Occluded-DukeMTMC remains available [V].

**Retraction was ineffective:** 72 of 73 DukeMTMC uses were through derivative datasets [V].

### New finding against the 2021 literature

`github.com/layumi/DukeMTMC-reID_evaluation`, the **original author's** repository, now returns **404** [V]. The dataset and its licence survive through third-party forks (`sxzrt/DukeMTMC-reID_evaluation`, 200 [V]) and `Yu-Wu/DukeMTMC-VideoReID` (200 [V]). So DukeMTMC-reID has been *partially* withdrawn by its own creator since the Peng et al. survey, and what remains in circulation is fork-hosted.

- **Commercial training: prohibited, the hardest refusal in the set.** CC BY-NC-SA is explicit; ShareAlike is additionally viral over adaptations; and the dataset is retracted by the institution whose IRB clearance was violated. [V]
- **Ethics:** about 2,000 non-consenting students and visitors, filmed outdoors in breach of IRB. Exposing.ai records that over 2,000 students and visitors who happened to be walking to class in 2014 will remain in every downloaded copy [V].
- **Assessment [C]:** Duke arguably never held the rights to grant CC BY-NC-SA, because the collection itself exceeded its IRB authorisation. For a *hospital* product this is the single worst dataset to be caught with: it is the canonical cited example of surveillance-dataset harm, and reputational exposure exceeds legal exposure.

---

## 3. MSMT17

**Sources opened:** `http://www.pkuvmc.com/publications/msmt17.html`, which returned **404** [V]; `https://www.pkuvmc.com/dataset.html`, which returned 200 but renders only a personal homepage banner [V]; `https://www.pkuvmc.com/agreement/RELEASE_AGREEMENT-MSMT17.pdf`, **200, downloaded and read directly** [V]; CVPR 2018 paper [V]

**The full operative text, exactly as published** [V, read from the PDF itself, not a summary]:

> "The MSMT17 dataset, constructed by National Engineering Laboratory for Video Technology (NELVT), is now available for the academic purpose only on a case-by-case basis. The NELVT at Peking University is serving as the technical agent for distribution of the dataset and reserves the copyright of all the images and subjective assessment results in the dataset. Any researcher who requests the MSMT17 dataset must sign this agreement and thereby agrees to observe the restrictions listed in this document."

> "(2) The database will not be further distributed, published, copied, or further disseminated in any way or form whatsoever, regardless of whether the dataset is modified or not. This includes further distributing, copying or disseminating to a different facility or organizational unit in the requesting university, organization, or company."

> "(3) All the dataset will be used for the purpose of scientific researches only. The MSMT17 dataset, in whole or in part, will not be used for any commercial purpose in any form."

> "(4) The performance computed on MSMT17 can be reported and released. No images in MSMT17 could be published or shown in any form of publication, document, or demonstration."

> "(5) The final explanation of this agreement refers to NELVT."

- **Commercial training: prohibited** [V]. Clause 3 is unambiguous and covers the dataset in whole or in part.
- **Redistribution: prohibited** [V]. Clause 2 is the strictest in the set: it survives modification and bars sharing even **between teams inside one company**.
- **Clause 4 is easily overlooked:** *metrics* may be published, but no MSMT17 image may be shown in any paper, document or **demonstration**. A product demonstration built on it would breach this even if training were somehow allowed. [V]
- **Availability: the official download page is dead (404).** The agreement PDF is still served, but there is no working request path from the paper's stated URL. In practice MSMT17 now circulates almost entirely through unauthorised mirrors. [V]
- **Ethics:** the paper records a 15-camera network deployed on campus, of which 12 cameras are outdoor [V, paper]. Peking University campus, 12 outdoor and 3 indoor cameras, over multiple days. **The same consent posture as DukeMTMC, non-consenting students under campus surveillance, but without the retraction.** [C]
- **Mirror metadata:** `huggingface.co/datasets/xianpeijie/MSMT17_V1`, 428 downloads, **no licence metadata at all** [V]. Downloading from a mirror grants none of the agreement's rights; the signed agreement is the *only* grant NELVT offers. Possessing a mirror copy is possession without a licence. [C]

---

## 4. CUHK03 / CUHK-SYSU

**Sources opened:** `http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html`, 200 [V]; `http://mmlab.ie.cuhk.edu.hk/projects/CUHK_Identification.html`, **404** [V]; `http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html`, 200 [V]

### CUHK03
**No licence text found at `http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html`.** [V] The page is a bare table of CUHK01/02/03 with Download / Paper / BibTex links, plus a CMC-computation FAQ. No terms, no copyright notice, no agreement, no request gate.

The frequently-cited line about the dataset being collected from the CUHK campus, about referring to CUHK regulation for its use, and about the data being sensitive and its use respecting student privacy appears in **third-party** documentation (torchreid docs, NEU's awesome-reid-dataset) [R]. **It could not be located on any CUHK-controlled page.** Treat it as unverified, not as a licence.

### CUHK-SYSU
The only operative sentence on the page [V]:

> "Dataset is available upon request (sli [at] ee.cuhk.edu.hk)."

No licence text otherwise. [V]

- **Commercial training: silent, no licence.** [V]
- **Redistribution: silent.** [V]
- **This is the most dangerous posture, not the safest.** [C] Absent any licence, default copyright applies: the rights-holder has granted *nothing*, so copying and training are unlicensed by default. A silent dataset is legally worse than a non-commercial one, because a non-commercial licence at least states the boundary. Commercial counsel will treat both as blockers.
- **Ethics:** CUHK03 holds 1,360 identities, six cameras, the CUHK campus, and non-consenting students. CUHK-SYSU is worse in kind [V, page]: the page divides the set by image source into street snaps and movies, records that street-snap images were collected with hand-held cameras across hundreds of scenes, and names movies and television dramas as the second source. That second half layers **third-party film copyright** on top of the depicted persons' rights, and CUHK plainly cannot license studio footage onward. [C]
- No retraction found for either. No HuggingFace mirror surfaced for CUHK03 [V].

---

## 5. MARS

**Sources opened:** `https://zheng-lab-anu.github.io/Project/project_mars.html`, 200 [V]; `http://www.liangzheng.com.cn/Project/project_mars.html`, **DNS failure** [V]

**No licence text found at `https://zheng-lab-anu.github.io/Project/project_mars.html`.** [V] The page contains a description, a benchmark comparison table, baseline code links, and Baidu and Google Drive download links, plus only a request to cite the authors' work [V].

- **Commercial training: silent, no licence.** [V]
- **Redistribution: silent.** [V]
- **Do not infer that Market-1501's terms carry over.** MARS is by the same author and is described on the page as an extension of the Market-1501 dataset, collected with six near-synchronised cameras on the Tsinghua University campus [V], so the same non-commercial intent is highly likely, but **it is nowhere stated for MARS**, and inferring a licence that was never granted is not a defence. [C]
- **Ethics:** Tsinghua campus, 1,191,003 bounding boxes, produced by DPM and GMMCP, that is by fully automatic detection [V]. It uniquely contains **3,248 distractors** [V]: people auto-detected and retained in the release who are not even benchmark identities. Nobody consented, and nobody curated them out.

---

## 6. iLIDS-VID and the underlying i-LIDS: the most restrictive terms in the set

**Sources opened:** `https://xiatian-zhu.github.io/downloads_qmul_iLIDS-VID_ReID_dataset.html`, 200 [V]; i-LIDS User Guide PDF, Publication No. 29/09, © CROWN COPYRIGHT 2009 [V, extracted directly]; **the actual commercial EULA** at `https://assets.publishing.service.gov.uk/media/5a7ac62ce5274a34770e7097/ilids-commercial-licence.pdf`, 6 pp, downloaded and read [V]; `https://www.gov.uk/government/publications/i-lids-end-user-licence-agreement-commercial` [V]

### Layer 1, iLIDS-VID (QMUL), exact text [V]

> "The dataset is intended for research purposes only and as such cannot be used commercially."

> "For downloading this data set, we have assumed that you have the right to access the i-LIDS MCTS scenario."

### Layer 2, i-LIDS User Guide, exact text [V]

> "Distribution of i-LIDS datasets is restricted to VA manufacturers and relevant academic research groups. Applications for datasets cannot be accepted from other organisations or individuals not connected with VA development or evaluation."

> "i-LIDS data remains Crown copyright. The End User Licence Agreement permits that it may be disassembled and processed in any way such as to contribute to the development of Video Analytics algorithms. It must not be redistributed to any third party. i-LIDS imagery may be used in academic, but not commercial exhibition."

### Layer 3, the actual EULA, exact text [V]

> "Permitted Purpose: research and/or educational purposes within the Licensee's organisation"

> "1.1 In consideration of the Licensee receiving the i-LIDS® Dataset as a potential research tool, the Licensor hereby grants to the Licensee a non-transferable, non-exclusive licence to: 1.1.1 Install and allow Permitted Users to access the i-LIDS® Dataset; 1.1.2 use and allow Permitted Users to use the i-LIDS® Dataset for the Permitted Purpose; 1.1.3 create new, or extend existing Ground-Truth Data; and 1.1.4 use still or video images from the i-LIDS® Dataset, excluding the Documentation, in academic papers and/or conference presentations, provided that the Licensor consents in advance to this use in each case."

> "1.3 The Licensee shall not, and shall procure that the Permitted Users shall not: … 1.3.2 use all or part of the i-LIDS® Dataset and/or Documentation in commercial advertising or promotion of any kind … 1.3.5 sell, rent out, lend, lease, sub-license or in an other way transfer the i-LIDS® Dataset and/or the Documentation in any form to any third party … 1.3.6 create any derivative works from the i-LIDS® Dataset … 1.3.7 disclose, sell, rent out, lend, lease, sub-license, broadcast, transmit or in an other way transfer details of any person, place or vehicle contained in the i-LIDS® Dataset and/or Documentation, including vehicle registration numbers."

> "1.2 … all Intellectual Property in the i-LIDS® Dataset, the Documentation and any New Release and New Version belongs and shall belong to the Licensor and the Licensee shall have no rights in the i-LIDS® Dataset and Documentation other than the right to use it in accordance with the terms of this Licence."

### Findings

- **Commercial training: prohibited at both layers.** [V] One nuance is worth getting right: the User Guide restricts academic against commercial **exhibition**, and the EULA does contemplate commercial video-analytics manufacturers as licensees. But the **Permitted Purpose is defined as research or educational purposes within the licensee's organisation**, and training a shipped commercial product falls outside it. The reference to video-analytics manufacturers must not be read as a commercial-product grant. [V + C]
- **Redistribution: prohibited** (1.3.5), and **derivative works are prohibited** (1.3.6). [V]
- **The legitimacy of iLIDS-VID itself is questionable [C]:** iLIDS-VID *is* a derivative work of i-LIDS MCTS, which clause 1.3.6 forbids. QMUL sidesteps this by assuming, in its own wording, that the downloader already holds the right to access the i-LIDS MCTS scenario, which pushes the compliance burden onto the downloader.
- **Those rights can no longer be obtained.** The gov.uk page was **withdrawn 3 March 2015**, recording that as of 30 November 2014 CAST was no longer distributing i-LIDS datasets [V]. So the precondition QMUL assumes is now **impossible for any new user to satisfy**. Anyone downloading iLIDS-VID today is, on QMUL's own framing, unlicensed. [C]
- **Ethics:** real UK **airport arrival hall CCTV** of the travelling public, captured under a Home Office and CPNI counter-terrorism programme. Crown copyright. No consent, no notice, no opt-out. Clause 1.3.7 forbids transferring details of any person contained in the data, and an identity embedder is arguably a compressed transfer of exactly that. [V for the clause, C for the inference]

---

## 7. PRID2011

**Source opened:** `https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/prid11`, 200, full text read [V]

**No licence text found at that URL.** [V] The only usage statement is a request to cite the authors' paper. The sole access restriction is that the cleaned dataset and the MAT-file results download directly, while the videos are behind a personal contact request, with the download link itself labelled as restricted access [V]. The only occurrence of the word licence on the page belongs to the TYPO3 content-management footer, not to the dataset.

- **Commercial training: silent, no licence.** [V]
- **Redistribution: silent**, with raw videos gated behind personal contact. [V]
- **Ethics:** the page records that the set was created in co-operation with the Austrian Institute of Technology, and that the images were extracted from multiple person trajectories recorded by two different static surveillance cameras [V]. Real street surveillance of the Austrian public. No consent statement.
- **Distinct legal exposure [C]:** this is EU-origin biometric data of EU data subjects. GDPR Art. 9 treats biometric data for unique identification as a special category; there is no consent, and no legitimate-interest assessment is published. For a commercial product this is the dataset most likely to attract a regulator rather than a licensor.

---

## 8. LTCC (Long-Term Cloth-Changing)

**Sources opened:** `https://naiq.github.io/LTCC_Perosn_ReID.html`, 200 [V]; the release agreement PDF `https://naiq.github.io/images/arxiv2020_LTCC/Restrictions for The Use of LTCC Dataset.pdf`, downloaded and read [V]; ACCV 2020 paper [V]

**The agreement, exactly as published** [V]:

> "Fudan University owns copyright of the LTCC Dataset and serves as the source for this data."

> "1. Requests: All requests for the LTCC Dataset records are submitted to Fudan University in writing."

> "2. Redistribution and Modification: The entire or part of the LTCC Dataset will not be modified and further distributed, published, copied or disseminated in any way or form, either for profit or not, except Fudan University. This refers also to further distribution of the dataset records to any other facility or unit within the requesting organization, other than the one mentioned in the request."

> "3. Commercial Use: LTCC Dataset is made available exclusively for research purposes, and may not be used in any commercial activity. In particular, the LTCC Dataset or its part may not be used to promote any product or technology."

The project page adds that dataset records are made available to researchers only after the receipt and acceptance of a completed and signed database release agreement [V].

- **Commercial training: prohibited** [V]. The second sentence of clause 3, barring use to promote any product or technology, independently blocks even marketing use of results.
- **Redistribution: prohibited**, including **between units inside one organisation**, and whether for profit or not. [V]
- **Ethics:** a grep of the ACCV 2020 paper for consent, ethics, IRB, privacy, volunteer and permission returned **zero hits** [V]. 152 identities, 12 camera views, collected over months on campus. No consent posture is documented anywhere.
- Minor: the canonical page URL contains a typographical error, Perosn for Person, which is worth noting so the page can be found again.

---

## 9. PRCC

**Source opened:** `http://www.isee-ai.cn/~yangqize/clothing.html`, 200, full text read [V]

**Exactly as published**, reproducing the original's spelling error [V]:

> "The images and the corresponding annotation results can only be used for ACADEMIC PURPOSES. NO COMERCIAL USE is allowed."

> "Copyright © Intelligence Science and System Lab (iSEE), Sun Yat-sen University. All rights reserved."

- **Commercial training: prohibited**, and stated as emphatically as the set gets, since the authors capitalised it. [V]
- **Redistribution: silent on its face, but the reservation of all rights means no redistribution right is granted.** [V + C]
- **Access is ungated:** unlike LTCC and SYSU-MM01, PRCC downloads straight from Google Drive and Baidu with no signed agreement [V]. Ease of access is not permission.
- **Ethics:** 221 identities, 33,698 images; cameras A and B record the same clothes in different rooms, and camera C records different clothes on a different day [V]. **No consent statement** appears anywhere on the page. This is conspicuous [C]: the *same lab* (iSEE, SYSU) states explicitly for SYSU-MM01 that every subject signed a privacy licence. The silence for PRCC, from a lab that documents consent when it has it, should be read as an absence of documented consent rather than as an oversight.

---

## 10. RegDB: terms not retrievable

**Sources opened:** `http://dm.dongguk.edu/link.html`, **200 but unusable** [V]; `https://raw.githubusercontent.com/mangye16/Visible-Thermal-Person-Re-Identification/master/README.md` [V]

**No licence text retrievable at `http://dm.dongguk.edu/link.html`.** [V] The page returns an approximately 3 KB frameset and image-map shell (`<title>CGCV Laboratory</title>`, image-map navigation, an iframe resizer script) with **no dataset listing and no terms in the served HTML**. The real terms live in a copyright form behind that interface, which could not be obtained without entering the publisher's request process.

The access route, exactly as published in the de-facto distribution README [V]:

> "The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form."

> "(Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1) on their website.)"

- **Commercial training: unclear, gated behind an unretrieved copyright form.** [V] Given that every peer dataset in this class is non-commercial, and that Dongguk gates access behind a signed copyright form at all, the base rate strongly suggests non-commercial. **Treat it as prohibited until the form is obtained and read.** [C]
- **Redistribution: unclear, and there is a warning sign.** The README and associated pages of the de-facto distribution offer a **private download link by email** [R] rather than routing users to Dongguk. That is redistribution outside the rights-holder's channel [C]. It means most RegDB copies in circulation, very likely including any obtained now, did **not** come with a Dongguk licence.
- **Ethics:** RegDB is a paired visible and thermal capture (412 people at 10 + 10 images each) with the structure of a **recruited, staged laboratory collection**, which would make it the second-best consent posture here, but **no primary consent statement was found**, so it cannot be confirmed. [C]
- **Action:** this is the one dataset where obtaining the copyright form would materially change the answer, and the standing acquisition rule in POLICY.md bars contacting the publisher to get it.

---

## 11. SYSU-MM01: the only negotiable one

**Sources opened:** `https://github.com/wuancong/SYSU-MM01` [V]; agreement PDF `https://raw.githubusercontent.com/wuancong/SYSU-MM01/master/agreement/agreement.pdf`, downloaded and read [V]

**The agreement, exactly as published** [V]:

> "Every pedestrian captured in the dataset has signed a privacy license to allow the images to be used for scientific research and shown in research papers."

> "1. Redistribution: Without prior approval from iSEE, SYSU-MM01 dataset will not be further distributed, published, copied or disseminated in any way, whether for profit or not. This includes further distributing, copying or disseminating to a different facility or organizational unit within the requesting university, organization or company."

> "2. Modification: Without prior approval from iSEE, SYSU-MM01 dataset cannot be modified."

> "3. Commercial Use: Without prior approval from iSEE, commercial use of SYSU-MM01 dataset will not be allowed."

> "4. Publication Requirements: In no case should the images be used in a way that could reasonably cause the original subject embarrassment or mental anguish."

> "6. Indemnification: Researcher agrees to indemnify, defend and hold harmless iSEE and its officers, employees and agents … from any and all losses, expenses, damages, demands and/or claims based upon any such injury or damage (real or alleged) and shall pay all damages, claims, judgements or expenses resulting from Researcher's use of SYSU-MM01 dataset."

- **Commercial training: prohibited by default, but uniquely, conditioned on prior approval from iSEE.** [V] This is the **only dataset in the entire set with a drafted path to commercial use.** Every restriction, on redistribution, modification and commercial use alike, is qualified the same way. Note that the standing acquisition rule in POLICY.md, section 3 bars contacting a dataset owner at all, so this path is drafted but closed to this project.
- **Redistribution: prohibited without approval**, including inside one company. [V]
- **Ethics: the best consent posture of all 16.** Subjects **signed a privacy licence** [V]. The scope needs care [C]: consent was given for scientific research and for display in research papers, **not** for training a commercial hospital-monitoring product. So even here the consent does not reach this use case, and the iSEE approval would be as much an ethical re-consent question as a licensing one.
- Clause 6 shifts all liability to the requesting researcher.

---

## 12. CCVID, VC-Clothes, Celeb-reID

### 12a. CCVID: no licence, and the upstream terms appear breached

**Sources opened:** `https://github.com/guxinqian/Simple-CCReID` README [V]; MSU FVG page `http://cvlab.cse.msu.edu/frontal-view-gaitfvg-database.html` [V]; **FVG-B Release Agreement** (Google Drive `1zogtwkeVUeT2h7EH7zvYjOxa5FsL0qtf`) downloaded and read [V]

**No licence text found at `https://github.com/guxinqian/Simple-CCReID`.** [V] The README's entire dataset section is two download links (BaiduYun + GoogleDrive) and a BibTeX entry. No terms, no agreement, no gate.

**But CCVID is reconstructed from FVG** (MSU's Front-View Gait database), and **FVG has a signed release agreement** [V]:

> "The dataset will be made available for research purposes only after the receipt of a completed and signed dataset release agreement form (this form). All requests for the dataset must be submitted to the Principal Investigator."

> "2. Redistribution: Without prior approval from MSU's Principal Investigator, the dataset – in whole or in part – cannot be further distributed, published, copied, or disseminated in any way or form. This dataset cannot be used by any other facility or organizational unit, other than the one requesting the dataset."

> "3. Modification and Commercial use: Without prior approval from MSU, the dataset, in whole or in part, may not be modified or used for commercial purposes. The license granted herein is specifically for internal research purposes only, and requestor shall not duplicate or use this dataset for any other purposes."

> "4. Publication: If images and videos need to be shown in papers and demos or disseminated in any way, only the imagery of subjects #100-#147 in session #1 may be used. Moreover, no captured data should be published, presented, or annotated in a way that can embarrass, discomfort, or anguish the original subject. Cropped face images of a subject cannot be published or displayed in public without receiving prior permission from the subject … Under no circumstance should an attempt be undertaken to determine the identity of a subject."

**Three compounding problems** [C, grounded in the quoted text]:
1. CCVID is redistributed on **open Google Drive / Baidu links with no gate**, while FVG clause 2 bars redistribution absent MSU PI approval. No approval is evidenced anywhere in the CCVID repo.
2. CCVID is a **modification** of FVG, re-cropped into 347,833 boxes and re-annotated with clothes labels, and clause 3 bars modification absent MSU approval.
3. CCVID's entire purpose is **re-identification**, against the instruction in FVG clause 4 that no attempt should be undertaken to determine the identity of a subject.

**And MSU has since privacy-retracted the original imagery** [V]: the FVG page now serves only **FVG-B**, a version of the database in which the face area is blurred far enough that a state-of-the-art face recognition algorithm fails to recognise it, done to better protect the privacy of the subjects. CCVID contains the **unblurred** originals.

- **Commercial training: prohibited via upstream FVG.** [V/C]
- **Redistribution: prohibited upstream.** [V]
- **Assessment [C]:** CCVID is the highest-risk item after DukeMTMC. It carries no licence of its own, and the licence it *should* carry forbids nearly everything CCVID does. Do not use it.

### 12b. VC-Clothes and Real28: a misapplied licence

**Sources opened:** `https://wanfb.github.io/dataset.html` [V]; `https://wanfb.github.io/resources/licence.txt` [V]

The page claims that users should follow the licence and are free to share, create and adapt the VC-Clothes and Real28 dataset in the manner the licence specifies [V].

**The actual licence file, in full and exactly as published** [V]:

> "# Copyright (C) 2019 Fangbin Wan, Yang Wu, Xuelin Qian, Yanwei Fu, Fudan University
> # Licensed to the Apache Software Foundation (ASF) under one or more
> # contributor license agreements.
> # The ASF licenses this file to You under the Apache License, Version 2.0
> # (the "License"); you may not use this file except in compliance with
> # the License. …"

- **This is not a dataset licence.** [V + C] It is the boilerplate **Apache-2.0 source-file header**, copied and pasted. It names the **Apache Software Foundation** as the licensor of a Fudan University dataset, which is incoherent. Apache-2.0 nominally permits commercial use, and an incautious reviewer could read this as the one clearance in the set. It is not one: a file header is not a grant, the named licensor has no relationship to the data, and the patent-grant and NOTICE machinery of Apache-2.0 does not map onto image data.
- **Commercial training: unclear, effectively unusable.** [C] Any use would rest on a licence the authors demonstrably pasted by accident.
- **Ethics, two separate problems** [V for facts, C for the exposure]: **VC-Clothes is synthetic, rendered from Grand Theft Auto V**, which imports Take-Two and Rockstar end-user-licence exposure over commercial exploitation of game-derived assets; that end-user licence was not opened in this pass and is flagged for counsel. **Real28 is real people**, 28 identities over 3 days on 4 cameras, indoor and outdoor, with **no consent statement on the page** [V].

### 12c. Celeb-reID

**Source opened:** `https://raw.githubusercontent.com/Huang-3/Celeb-reID/master/README.md` [V]

**Exactly as published** [V]:

> "Note: The two datasets should be used for research only. Please DO NOT distribute or use it for commercial purpose."

(Covers both Celeb-reID and Celeb-reID-light.)

- **Commercial training: prohibited.** [V]
- **Redistribution: prohibited.** [V]
- **Ethics, and the authors could not grant even the research licence they wrote** [C]: the images are **crawled street-snapshot photographs of celebrities from websites** [R, from the source papers and secondary descriptions; the README itself does not describe provenance]. Every image carries a third-party photographer's copyright, none of the 1,052 subjects consented, and for a *commercial* product right-of-publicity exposure attaches to identifiable celebrities in a way it does not for anonymous pedestrians. The dataset creators held no rights in the underlying photographs and so had nothing to license onward.

---

## 13. LUPerson and LUPerson-NL

**Sources opened:** `https://github.com/DengpanFu/LUPerson` README [V]; `https://raw.githubusercontent.com/DengpanFu/LUPerson/main/LICENSE`, **404** [V]; `.../LUPerson/main/LUP/README.md` [V]; `https://github.com/DengpanFu/LUPerson-NL` README [V]; `.../LUPerson-NL/main/LUP-NL/README.md` [V]; HuggingFace API [V]

### The operative sentence, and exactly where it appears

In **`README.md`** of `DengpanFu/LUPerson`, in the LUPerson Dataset section, as its own **bolded** line immediately after the dataset description and immediately before the pointer to `./LUP`:

> **"LUPerson can only be used for research, commercial usage is forbidden."**

[V, read raw and exact]

**There is no LICENSE file in the repository**: the raw URL returns 404 [V]. **That one README sentence is the entire licence.** Nothing in `./LUP` adds terms; that directory contains only the reconstruction scripts. [V]

- **LUPerson, commercial training: prohibited.** [V]
- **LUPerson, redistribution: silent**, with no statement either way. [V]

### LUPerson-NL is *not* covered by that sentence

A grep of the LUPerson-NL README for the word commercial returned **zero occurrences** [V]. `LUP-NL/README.md` contains only build scripts and download links, no terms [V].

- **LUPerson-NL, commercial training: silent, no licence.** [V]
- **Do not read across [C]:** same authors and same method, so the same restriction was almost certainly intended, but it is not written. Silence is not permission; it is default copyright, that is, no grant at all. Either way the answer is that the set is unusable.

### The ethics finding, the largest in this report

Both datasets are built by **scraping YouTube**. Both `LUP/README.md` and `LUP-NL/README.md` state that every video's YouTube key is listed in `vnames.txt`, and that downloading the raw videos requires `youtube-dl` [V].

LUPerson reports 4M images of over 200K identities [V]. LUPerson-NL reports 10M images of over 430K identities extracted from 21K street-view videos [V].

- **About 630,000 non-consenting members of the public**, harvested from street-view videos uploaded by third parties. No consent, no notice, no opt-out, no institutional review board, and, unlike the campus datasets, no institution that could even in principle have obtained consent. [V for the numbers, C for the characterisation]
- **The construction pipeline breaches YouTube's terms of service**, which prohibit downloading content except through YouTube-provided means. [C]
- **Copyright sits with the video uploaders**, not the dataset authors; as with Celeb-reID, the authors licensed something they never owned. [C]
- **The distribution model moves the breach downstream** [C]: shipping YouTube keys and detection pickles rather than images means the downloader performs the scrape, so the terms-of-service breach lands on the downloading organisation rather than on the authors. A processed copy is also offered via Baidu [V].

### Mirror metadata, verified

| HF mirror | Licence metadata | Downloads |
|---|---|---|
| `wahdan2003/LUPerson-NL` | **`license: mit`** | 923 [V] |
| `tuandunghcmut/LUPerson` | none | 206 [V] |
| `aveocr/Market-1501-v15.09.15.zip` | none | 533 [V] |
| `xianpeijie/MSMT17_V1` | none | 428 [V] |

`wahdan2003/LUPerson-NL` is tagged **MIT** [V]. **That tag is supplied by a third-party uploader** and flatly contradicts the authors' own statement that commercial usage is forbidden. An unauthorised mirror's licence tag has no legal effect, because the uploader cannot grant rights they never held. Any justification of LUPerson use that rests on an MIT tag on HuggingFace is therefore void. [V + C]

---

## Dead / moved / withdrawn URLs (verified in this pass)

| URL | Status |
|---|---|
| `http://vision.cs.duke.edu/DukeMTMC/` | **DNS failure** [V] |
| `http://www.liangzheng.com.cn/` (Market-1501, MARS) | **DNS failure** [V] |
| `http://www.pkuvmc.com/publications/msmt17.html` | **404** [V] |
| `http://mmlab.ie.cuhk.edu.hk/projects/CUHK_Identification.html` | **404** [V] |
| `https://github.com/layumi/DukeMTMC-reID_evaluation` | **404** (author's own repo removed) [V] |
| gov.uk i-LIDS EULA page | **Withdrawn 3 March 2015**; distribution ceased 30 Nov 2014 [V] |
| `http://cvlab.cse.msu.edu/` FVG | Replaced by **FVG-B** (faces blurred for privacy) [V] |
| `https://www.pkuvmc.com/agreement/RELEASE_AGREEMENT-MSMT17.pdf` | Still live (200); the agreement outlives the dataset page [V] |

---

## Recommendations

1. **Nothing here clears commercial training as it stands.** An absence of a licence, as with CUHK03, CUHK-SYSU, MARS, PRID2011, LUPerson-NL and CCVID, must not be read as permissive. It is the weakest position, not the strongest, because no rights were granted at all.
2. **Quarantine immediately if present in any training set:** DukeMTMC and all `Duke*` derivatives (retracted, in breach of the governing review-board approval, CC BY-NC-SA), CCVID (upstream terms breached, source imagery privacy-retracted), LUPerson and LUPerson-NL (YouTube scrape covering about 630K non-consenting people), and iLIDS-VID (Crown-copyright closed-circuit footage that can no longer be licensed). For a **hospital** product these four create board-level exposure, not merely licence exposure.
3. **The one drafted path to a licensed real dataset is SYSU-MM01.** Its agreement conditions the commercial bar on prior approval from iSEE, and it is the only set whose subjects signed a privacy licence. The consent covers research publication rather than commercial deployment, so the approval needed would be an ethical re-consent question as much as a licensing one. That path is closed here in any case: POLICY.md, section 3 bars contacting a dataset owner. [V/C]
4. **RegDB stays out.** Its terms sit behind a Dongguk copyright form that was not retrievable, and POLICY.md, section 3 bars requesting it, so the conservative reading governs and the set is treated as prohibited.
5. **Audit mirror provenance.** Four HuggingFace mirrors of restricted datasets are in active use with 200 to 900 downloads each, one of them carrying a third-party MIT tag [V]. Data obtained from HuggingFace, Kaggle or Academic Torrents rather than under a signed agreement is possession without a licence, whatever the mirror's metadata says.
6. **Two items need counsel rather than more research:** the Rockstar and Take-Two end-user licence position on the Grand Theft Auto V derived VC-Clothes, and the GDPR Article 9 position on PRID2011's European-subject biometric data. Both are outside what primary-source dataset pages can answer.

**Method note on the standard of evidence:** every quotation above marked [V] was read from the primary artefact. The MSMT17, LTCC, SYSU-MM01, i-LIDS and FVG agreements were extracted from the PDFs themselves, and the Market-1501, PRCC, Celeb-reID and LUPerson wording was pulled as raw text, because summarisers were silently truncating and paraphrasing the operative sentences. A claim available only at second hand, namely the Duke apology wording, the CUHK regulation line cited for CUHK03, RegDB's private-link distribution and Celeb-reID's crawl provenance, is tagged [R] and marked as such rather than presented as verified. Every block quotation of licence or agreement text on this page is reproduced character for character from the artefact named beside it, so punctuation the surrounding prose does not use, namely the two em dashes in the DukeMTMC deed clauses and the two en dashes in the FVG redistribution clause, stands as the source wrote it and is not edited to match house style.
