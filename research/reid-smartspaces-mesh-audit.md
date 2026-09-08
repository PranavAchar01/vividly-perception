# SmartSpaces mesh-diversity audit: the re-identification precondition gate, not cleared

Dated 2026-09-03. The method of review was a visual pass over every sheet, under the
mandatory precondition in research/reid-data-licences.md section 2
caveat 4 and the re-identification admission notes in the registry entry.

**Rejected for identity-discriminative embedder training.** The identity labels
are real, but the appearances are not: the corpus is roughly **35 to 45 distinct avatar
builds, that is combinations of body mesh, outfit and hair, recycled under 231 identity
labels**, including look-alike pairs inside the same scene. SmartSpaces is therefore
demoted from primary re-identification training corpus to the narrow roles in the
consequences section below.

## Method

- Input: the 5 detector-train scenes (Warehouse_000 to Warehouse_004) of the
  MTMC_Tracking_2025 slice; per-scene `ground_truth.json` (frame, then objects with
  `object id` and per-camera `2d bounding box visible`), joined to the 1 Hz frames already
  on the training pod for the person_v2 detector corpus. Warehouse_005 is excluded, because
  it is the detector dev-val scene.
- Script: `scripts/reid/smartspaces_mesh_audit.py`, an argparse copy of the pod run. For
  every identity, up to 10 crops are sampled round-robin across its cameras, tallest first
  per camera, tiled into contact sheets, with rows sorted by a mean-HSV torso signature so
  look-alike identities land adjacent.
- Review: all 20 sheets, 231 rows, that is every identity with at least 1 crop of at least
  48x96, read visually. Artefacts: `sheet_{00..19}.jpg` and `stats.json`, written to the
  directory given to `--out`, which TRAINING.md sets to
  `outputs/reid/smartspaces_mesh_audit`. That location is a build output and is not in
  version control; regenerate the artefacts with the script.

## Counts (measured in this audit, 5 scenes)

| scene | person IDs (at least 1 crop at 48x96 or larger) | crops 48x96 or larger | crops 64x128 or larger |
|---|---|---|---|
| Warehouse_000 | 50 | 50,846 | 29,113 |
| Warehouse_001 | 46 | 142,689 | 110,369 |
| Warehouse_002 | 45 | 66,539 | 42,041 |
| Warehouse_003 | 42 | 109,256 | 81,720 |
| Warehouse_004 | 48 | 120,253 | 93,712 |
| **total** | **231** | **489,583** | **356,955** |

Reconciliation with the research pass, which reported 274 IDs and 597,903 crops: that
measurement spanned all 6 train-split scenes, while this audit excludes Warehouse_005, the
detector dev-val scene, and 274 minus 231 is 43, which is about one scene's ID count. The two
measurements agree.

## Findings

1. **Avatar recycling across identity labels is pervasive.** Nearly every appearance
   recurs under 2 to 9 different `object id` values, across scenes and within them.
   Documented examples, given as scene and id from the sheets:
   - Heavy-set male, salmon coverall or polo: at least 9 ids across 5 scenes (002:1, 004:3,
     002:7, 004:13, 003:3, 000:0, 001:4, 001:13, 000:38; lighter-pink variants 001:15,
     000:30, 003:19; lilac variant 003:14, 000:17, 001:25).
   - Long-haired figure, cream sweater and khakis: 5 ids in 5 scenes (004:45, 003:8,
     001:43, 002:9, 000:47).
   - Blue shirt, cream high-visibility vest and auburn bob: 5 ids (004:17, 000:8, 001:34,
     003:30, 002:17).
   - White high-visibility suit and auburn hair: at least 6 ids (004:7, 001:30, 002:4,
     002:19, 001:40, 000:28, 003:44).
   - Pink top and red trousers: at least 5 ids (002:21, 001:29, 004:38, 000:9, 003:35).
   - Overalls and white tee, in heavy and slim builds: at least 8 ids (004:19, 001:32,
     000:42, 002:25, 003:34, 002:26, 003:39, 001:18, 000:48).
   - **Same-scene look-alike pairs exist**, 002:1 against 002:7 and 002:4 against 002:19:
     visually identical, differently labelled, in the same scene.
2. **The distinct appearance pool is about 35 to 45.** The exact mesh count is not
   recoverable from imagery, because NVIDIA does not disclose it, but after sorting the 231
   rows by appearance the sheets collapse into that many families. This matches the
   inconclusive body-size-signature test, 199 distinct at two decimal places, read
   pessimistically.
3. **Texture artefacts.** Several avatars render as featureless matte-black bodies or heads
   (003:11, 002:39, 004:49, 002:3, 000:4, 003:16), and 001:6 shows one cartoonish
   untextured head. These are additional label-free appearance collisions.
4. Crop quality is otherwise good: 1 Hz sampling, 18 to 45 cameras per identity, and
   occlusion-realistic crowding.

## Consequences

- **Identity-discriminative embedder training on SmartSpaces is excluded.** With about 40
  effective appearances, a metric loss receives contradictory targets, that is identical
  inputs under different labels, which is irreducible noise, and the effective vocabulary
  sits about 50 times below the 2,000 to 3,000 identity saturation band the scale
  literature calls for. This is exactly the failure mode the precondition was designed to
  catch.
- **What SmartSpaces re-identification data remains good for:** first,
  positive-pair-only view-invariance pretraining, where same-identity cross-camera
  positives stay valid and no negatives are mined; second, session, gallery and
  tracker-association structure prototyping, where look-alike distractors are a legitimate
  stress test, with the explicit note that appearance cues are expected to fail there; and
  nothing else in the re-identification lane.
- **Detection and tracking training are unaffected**, because their labels are boxes and
  per-scene tracks, which remain correct regardless of appearance reuse.
- **MTMC_Tracking_2024, with 2,481 IDs, inherits the suspicion.** Same engine, same avatar
  economics, and 2,481 labels do not promise 2,481 appearances. Do not spend the 217 GB
  fetch or any GPU time on it before running this same audit on a 2-scene sample.
- **The corpus plan re-ranks**, and research/reid-data-licences.md section 2 is updated
  accordingly: the licence-clean compositing path, with Open Images V7 feedstock, and the
  real-data unlocks (CHIRLA, Visym CAP, TVRID, MEVA and MEVID) carry the embedder lane,
  while SmartSpaces drops to the auxiliary roles above.
- The CQ-27 responsible-AI question now gates a much smaller surface, because no shipped
  embedder will train on SmartSpaces identity labels, but it stays open because detection
  and tracking use continues and the auxiliary roles above still touch the bytes.
