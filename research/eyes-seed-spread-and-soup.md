# Eyes B1 seed runs and uniform-2 soup verdict (2026-09-06)

Scope: the B1 RTv4-S person_v2 seed-replication arm and its soup, run on pod
`6j8legxaq76pr8` (RTX 5090) by `scripts/pod/eyes_chain.sh`. Surface for every
number here: **dev-val** (`person_v2_devval.json`, the pinned selection split).
No exam was looked at by anything on this page. The bar was declared before any
seed-1 or soup score existed, and it is stated in full under the soup section
below: the best single seed plus the larger of 0.0012 and the observed seed
spread. The seed-2 cut was taken on 2026-09-06.

## Seeds

| run | ckpt | sha256 (8) | dev-val AP | source of the number |
|---|---|---|---|---|
| seed 0 (baseline, model of record) | b1_best_stg2_345c30a5.pth | 345c30a5 | **0.8356** | trainer best_stat 0.8355529764, reproduced exactly by the I1 dump grade (research/eyes-instrument-grading.md) |
| seed 1 | b1_best_stg2_seed1.pth | 17362dca | **0.8348** | trainer best_stat 0.8348031819826661 (epoch 47/48, B1_SEED1.log); --test-only COCOeval prints 0.835 (testonly_seed1.log) |
| seed 2 | not run | not run | not run | **Cut 2026-09-06 at about 05:00Z** (spending-lid deadline; about $11 of pod time to resurrect later) |

The seed spread between seed 0 and seed 1 is **0.00075**, under the 0.0012 noise floor from the
DEIM/RTv4 A-vs-B calibration. Two full 48-epoch trainings from the same COCO
init land 0.00075 apart: the recipe is highly reproducible, and dev-val is at
a plateau near 0.835.

## Uniform-2 soup

Built by `scripts/detector/soup_v2.py uniform` over seeds 0+1 (692 tensors averaged, 102
copied, averaged weights written into both `model` and `ema.module`):

- artefact: the soup checkpoint, identified by sha256
  `0d80cb4c860bd737ee098291cce5bfdcbf53fe9ccac82c47443d59cfd169cf43`
- dev-val COCOeval (testonly_soup_uniform2.log): **AP 0.830**, AP50 0.958,
  AP75 0.881, small 0.118, medium 0.714, large 0.928

Bar (declared before any of the scores above existed): the best single seed plus the larger
of 0.0012 and the seed spread, that is 0.8356 plus max(0.0012, 0.00075) = **0.8368**.

## Verdict

**Soup not promoted.** 0.830 is about 0.007 below the bar and below both single
seeds, worse in every size bucket than seed-1 alone (small 0.118 vs 0.124,
medium 0.714 vs 0.721, large 0.928 vs 0.930). Greedy soup was skipped under
the same cut and would change nothing: greedy starts from the best single
ingredient and only keeps additions that help, and the only possible addition
just measured minus 0.005.

The model of record stands: seed 0 `345c30a5`, dev-val AP 0.8356 (ONNX
`4026204967aa…`, parity passed; the deployable artefact is unchanged).

Reading: weight-averaging pays when ingredients sit in one loss basin
(fine-tunes from a shared, already-adapted init). These seeds are full
retrainings from the generic COCO init whose data order and augmentation draws
differ from step one, so they land in nearby but different basins, and the
midpoint is slightly off both. Implication for a resurrected seed-2: its value
would be a third draw at a better single seed, and at a wider observed spread,
not a soup ingredient.

## Recording defect in the run's AP files

The run's `soups/seed_aps.txt` and `soup_uniform2.ap` are polluted: inside
`test_only()`, `raise_ulimit` echoed "ulimit -n = 524288" to stdout, which is
the channel callers capture, so the echo rode ahead of the (correct) parsed
AP. `parse_ap`'s regex itself matched faster_coco_eval's COCOeval table on
first live contact. Authoritative numbers are the COCOeval tables in the run logs
`testonly_seed1.log` and `testonly_soup_uniform2.log`, which are pod-side logs not
carried in this repository and are quoted above. Fixed on 2026-09-06 in
`scripts/pod/eyes_chain.sh` (`raise_ulimit >&2`); the .ap files are left as they were
written, because result artefacts are never hand-edited.

## B2-DINOv2T verdict (appended 2026-09-06, chain state ALL_DONE)

The gated distillation arm ran to completion (06:30:44Z to 23:17:24Z, 16.8 h,
about $11.6): RTv4-S student with a frozen DINOv2 ViT-B/14 teacher (pin 7764ea0f,
weights sha 0b8b82f8…, load-clean assertion passed; VRAM peak 23,758 MiB vs
30,500 gate).

- Final: **dev-val best_stat 0.8345451735294738** (epoch 47/48),
  ckpt `b2_best_stg2_9ecb4627.pth`, sha256
  `9ecb46272c838cf9032d53db73e491dc82de7a1d9c9a6e098b517d106b5382a3`.
- Bar (declared in the B2 recon before launch): beat best single seed by the
  soup-rider margin, giving **0.8368**. 0.8345 < 0.8368, so the arm is
  **not promoted**.

Four-arm convergence, one table (all dev-val, one look each, exams untouched):

| arm | AP | ckpt |
|---|---|---|
| seed 0 (model of record) | **0.8356** | 345c30a5 |
| seed 1 | 0.8348 | 17362dca |
| B2-DINOv2T (teacher) | 0.8345 | 9ecb4627 |
| uniform-2 soup | 0.830 | 0d80cb4c |

Three independent full trainings, two plain and one with a foundation teacher,
land within 0.0011 of each other, and the teacher run sits below both plain
seeds. Conclusion of record: DINOv2-B distillation adds nothing on this
training set; 0.8356 (345c30a5) is the data-era ceiling and the final model of
record across B1 and B2. The next lever for eyes is data (multi-room, own-camera),
not architecture or teachers. The eyes GPU queue is complete.

## Chain state after the verdict

After SOUP_UNIFORM3_OK at 06:20:09Z, B2_FETCH_TEACHER hit its intended
first-contact stop 2 s later: PINS.yaml `repos.dinov2.pin` still read TODO, the
stage cloned facebookresearch/dinov2, printed HEAD
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`, and refused to proceed unpinned, as
the pinning rule requires. The pin was recorded in `third_party/PINS.yaml` and
the chain relaunched into B2_FETCH_TEACHER and then B2_DINOV2T (the weights
sha256 is established and committed at first download under the same rule).
