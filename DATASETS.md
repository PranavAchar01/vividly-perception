# Datasets

Every dataset this repository knows about is one YAML entry under `configs/datasets/`, loaded
and validated by `src/vividly/data/registry.py`. Those entries are the legal record: they carry
the licence text copied word for word, the licence URL, the conflict notes, the counsel-question
identifiers, the terms-acceptance record and the event grants. This document summarises them so
the shape of the corpus can be read at a glance. It never replaces them, and where the two
disagree the YAML entry governs.

Twelve entries are registered. A thirteenth file, `configs/datasets/_template.yaml`, documents
the schema and is not a dataset; the loader skips any file whose name begins with an underscore.

## How to read the table

**Bucket** is the licence class, and it decides which physical tree the data may live in and
what may be trained from it.

| Bucket | Meaning | Physical tree |
|---|---|---|
| TRAIN_OK | may train shippable weights | `data/train_ok/` |
| TRAIN_PROTOTYPE | may train prototype weights only, under a recorded grant | `data/train_prototype/` |
| EVAL_ONLY | measures only, never trains anything | `data/eval_only/` |
| LEGAL_HOLD | never fetched and never resolved to a path | none |

**Status** is one of planned, requested, available, converted, blocked or dropped. Available and
converted mean the bytes are on disk and hashed, or that the entry records why hashes are
impractical.

**Redistributable** is the `redistribute` flag: whether the licence permits re-publishing the
bytes from this repository. No registered dataset is redistributable, so nothing in
`configs/datasets/` may be mirrored outward.

**Acquisition record** is how the bytes were obtained and, where a human accepted terms, when
that was recorded. The standing rule is that datasets are acquired only by self-serve anonymous
download: no dataset owner is contacted, and no sign-up, form or licence clarification request
is ever made. Licence ambiguities are recorded in the registry entry and resolved by counsel at
ship time. The full rule is in [POLICY.md](POLICY.md).

## Registered datasets

| Name | Bucket | Licence | Redistributable | Status | Used for | Acquisition record |
|---|---|---|---|---|---|---|
| `coco` | TRAIN_OK | annotations CC BY 4.0; images under the Flickr Terms of Use, since the COCO Consortium does not own the image copyright | no | available | detector and pose training, and the COCO val2017 keypoint exam | anonymous http download, 2026-08-07; the zips were deleted after extraction and the four registered annotation JSONs are pinned by sha256 |
| `physicalai_smartspaces` | TRAIN_OK | CC BY 4.0 | no | available | detector and tracking training on the train-side scenes; the val scenes are frozen exam material; re-identification use is limited to view-invariance pretraining | Hugging Face only, permanently, because the same bytes carry incompatible terms through the challenge portal; terms recorded 2026-08-31; per-file hashing is impractical at this file count, so per-scene sizes and file counts are recorded instead |
| `ajile12` | TRAIN_PROTOTYPE | CC-BY-4.0 (DANDI:000055, OpenAccess) | no | available | matched-context negative diet for the seizure verifier, drawn only from labelled-normal spans | anonymous HTTPS streaming from the DANDI S3 bucket; terms recorded 2026-08-28; the harvest manifest that carries a sha256 for each of the 110 session files is itself pinned by sha256 |
| `caucafall` | TRAIN_PROTOTYPE | CC BY 4.0 | no | available | interim prototype fall verifier training, and quiet-footage negatives | anonymous Mendeley download; terms recorded 2026-08-27; 1 archive pinned by sha256 |
| `gmdcsa24` | TRAIN_PROTOTYPE | CC BY 4.0 on the Zenodo deposit, MIT on the GitHub repository | no | available | interim prototype fall verifier training | anonymous Zenodo download plus the interval CSVs at a pinned GitHub commit; terms recorded 2026-08-27; 9 files pinned by sha256 |
| `le2i_fdd` | TRAIN_PROTOTYPE | CC BY-NC-SA 3.0 per the dat@UBFC record | no | available | interim prototype fall verifier training | anonymous http download from dat@UBFC; terms recorded 2026-08-26; 1 archive pinned by sha256 |
| `wu_sahzu_emu_trainsplit` | TRAIN_PROTOTYPE | Apache-2.0 per the Hugging Face dataset card | no | available | interim prototype convulsion verifier training on four convulsive patients, disjoint from the exam patients | the same acquisition as `wu_sahzu_emu`; the four train-side patients were copied into the prototype tree; 8 files pinned by sha256 |
| `mevid` | TRAIN_PROTOTYPE | CC BY 4.0 for the underlying MEVA video; undeclared for the MEVID annotation release, which ships no LICENSE file | no | available | re-identification Phase A training corpus for the appearance embedder, and its held-out test split as that lane's evaluation surface | anonymous http download of three archives from the public MEVA S3 bucket; terms recorded 2026-09-07; 3 archives pinned by sha256 on first download, recorded below |
| `mvor` | EVAL_ONLY | CC BY-NC-SA 4.0 | no | available | frozen detection exam on real hospital-interior multi-camera footage | anonymous http download from the CAMMA public S3 bucket, with the annotations file from the project repository; terms recorded 2026-08-31; 2 files pinned by sha256 |
| `ur_fall` | EVAL_ONLY | CC BY-NC-SA 4.0, non-commercial academic use | no | available | held-out fall exam; never trains | anonymous http download from the dataset page; terms recorded 2026-08-26; 172 files pinned by sha256 |
| `multicam_fall` | EVAL_ONLY | none stated; a citation request only, so all rights are reserved by default | no | available | held-out multi-view fall exam, and the confounder class the burn-in found; never trains | anonymous http download; terms recorded 2026-08-26; 3 files pinned by sha256 |
| `wu_sahzu_emu` | EVAL_ONLY | Apache-2.0 per the Hugging Face dataset card | no | available | held-out seizure exam, the only real clinical seizure footage held; never trains | anonymous Hugging Face download; terms recorded 2026-08-26; 40 files pinned by sha256 |

Open counsel questions are recorded per entry and are listed in
[docs/COUNSEL_QUESTIONS.md](docs/COUNSEL_QUESTIONS.md). The five acted-fall entries
(`caucafall`, `gmdcsa24`, `le2i_fdd`, `multicam_fall` and `ur_fall`) carry CQ-22, the
non-commercial research exam `mvor` carries CQ-24, and the two WU-SAHZU-EMU entries carry
CQ-23. `ajile12` is clinical epilepsy-monitoring footage as well, but its entry carries no
counsel-question identifier. The re-identification licence questions on the synthetic corpus
are recorded as CQ-27 in that entry's conflict note. `mevid` holds the most exposed licence
position in the registry, and its conflict note deliberately keeps CQ-11 and CQ-20 out of
`counsel_questions`, because listing them would demand a clearance date that nobody has made.

### MEVID archive hashes, recorded 2026-09-07

The three upstream archives were downloaded once on the training pod from the public MEVA
bucket and hashed as they arrived. The training and annotation archives were hashed from the
files on disk; the test archive was hashed from the download stream and unpacked in the same
pass, so its size is the length the bucket declares for it. The rows are in
`configs/datasets/mevid.yaml` under `files`, and `status` moved from planned to available in
the same edit.

| archive | size in bytes | sha256 |
|---|---|---|
| `upstream/mevid-v1-bbox-train.tgz` | 32,684,603,246 | `7277b794d40f611308f2d1b8588c9f4acc49c1c3fdeb11135304a52a347f03d1` |
| `upstream/mevid-v1-bbox-test.tgz` | 13,974,475,339 | `4d4ec2672324c26b9e4b128c550cf0de3964c0cf81ce37b5e7a4c71e25fc221d` |
| `upstream/mevid-v1-annotation-data.zip` | 12,211,153 | `18a91691a0ccb3b79302c55643840bc5b7f16a9ed813d93bbe91edd151f982e8` |

Unpacked on the pod: 104 training identities in 3,609,156 chip files and 54 test identities in
1,186,251 chip files, matching the counts the registry entry carries from the upstream README.

## Attribution

Every registered dataset requires attribution. The credit line each entry reproduces exactly is in
its YAML `licensing.prototype_grant.attribution` or `provenance.citation` field; the collectors
credited are listed below, with `wu_sahzu_emu` and `wu_sahzu_emu_trainsplit` sharing one credit
line because the second is the train-side split of the first:

- COCO 2017: COCO Consortium. Lin et al., Microsoft COCO: Common Objects in Context, ECCV 2014.
- PhysicalAI-SmartSpaces: NVIDIA (Physical AI / Metropolis, AI City Challenge organizers).
- AJILE12: Peterson et al. (Brunton lab, University of Washington), Scientific Data 9, 184 (2022).
- CAUCAFall: J. C. Eraso, E. Munoz, M. Munoz, J. Pinto (Universidad del Cauca), Data in Brief 45 (2022) 108610.
- GMDCSA-24: Ekram Alam et al. (GMDCSA-24 authors), Data in Brief 57 (2024) 110892.
- Fall Detection Dataset: Imen Charfi, Johel Miteran, Julien Dubois (Le2i / ImViA, U. Bourgogne).
- MVOR: CAMMA, University of Strasbourg / IHU Strasbourg (Srivastav et al.), MICCAI-LABELS 2018.
- UR Fall Detection Dataset: Bogdan Kwolek & Michal Kepski, University of Rzeszow / AGH.
- Multiple Cameras Fall Dataset: E. Auvinet, C. Rougier, J. Meunier, A. St-Arnaud, J. Rousseau (DIRO, UdeM), Technical report 1350, July 2010.
- WU-SAHZU-EMU-Video: Yankun Xu et al. (Westlake University / Zhejiang University; VSViG project), recorded at the Second Affiliated Hospital of Zhejiang University School of Medicine EMU.
- MEVID: Kitware (MEVID annotation release), over the MEVA corpus (mevadata.org). Davila, Du, et al., WACV 2023, arXiv:2211.04656; MEVA is CC BY 4.0.

No registered dataset is redistributed from this repository.

## Considered and not used

Each row below was a registry entry that has been removed. The entry is gone, but the reason it
was refused or retired is kept here, because a refusal is a record.

| Name | Licence as recorded | Reason it is not used |
|---|---|---|
| `crowdhuman` | non-commercial research and educational use only, with an explicit no-distribution clause | web-crawled crowd detection set. No v2 lane uses it; its converter, its fixture tree and its tests go with the entry. |
| `peoplesanspeople` | Apache-2.0 | synthetic generator registered but never materialised. The synthetic corpus of record is `physicalai_smartspaces`. |
| `jrdb` | CC BY-NC-SA 3.0 | registration requires an institutional email and a named academic supervisor, which the standing acquisition rule bars, and the domain is egocentric and eye-level rather than the target geometry. |
| `mot20` | unverified; assumed to follow the MOTChallenge norm of CC BY-NC-SA until read from the live site | street-level oblique tracking benchmark, never used. Tracker plumbing is checked at the target geometry instead. |
| `mpii` | annotations and code under the Simplified BSD licence; images no commercial use, since the authors do not hold the image copyright | single-person 16-keypoint pose crops, never used. COCO val2017 is the pose exam. |
| `dancetrack` | conflicting: cc-by-4.0 on the author's dataset card, MIT on the repository covering code only, and no stated dataset terms on the project page | adversarial appearance benchmark for the retired re-identification v0 work; no v2 lane uses it. |
| `personpath22` | CC-BY-NC-4.0 for the data and annotations; the Apache-2.0 file covers the code only | dropped: the data grant is non-commercial, the authors ask that the data not be used to identify or generate biometric information, and identities are within-video only. |
| `coco_reid_filtered` | per-image Flickr licences filtered to CC BY 2.0, CC BY-SA 2.0 and no known copyright restrictions; annotations CC BY 4.0 | planned derived subset of COCO person crops for re-identification; never materialised. |
| `sompt22` | annotations CC BY 4.0; the dataset itself is non-commercial research only, and the stricter reading governs | pole-mounted outdoor surveillance multi-pedestrian tracking set (Simsek, Cigla, Kayabol, ECCV 2022 Workshops). Requested as a frozen detection exam and never acquired: the linked Drive folder is not publicly readable and the standing acquisition rule admits self-serve anonymous download only, so no access request is made and no registry entry remains. |

Seven further entries covered overhead-fisheye imagery, real or rendered; they were removed
under [POLICY.md](POLICY.md), rule 1, together with the held-out split file that covered
three of them, and nothing in this repository reaches them.
