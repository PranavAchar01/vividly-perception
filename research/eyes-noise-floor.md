# v2 detector model A: noise floor locked (3 seeds, dev-val)

2026-09-02. DEIM-D-FINE-S person fine-tune per
`third_party/configs/deim_s_person_v2.yml` (provenance card in the file);
the trainer is vendored DEIM @ pin 09d35d5 on the 5090 pod (torch 2.11.0+cu128).
Corpus: SmartSpaces person_v2 (55,200 training images and 828,230 boxes; dev-val
1,800 / 33,755, held out by scene, Warehouse_005). Selection surface:
dev-val only, because variants are selected on the train-side validation split and never on
an exam (POLICY.md, section 5, measurement discipline). Curves: the per-epoch series
`final_models/eyes/curves/detector_deim_seed0.csv`, `detector_deim_seed1.csv` and
`detector_deim_seed2.csv`, extracted from the training logs by
`scripts/analysis/extract_training_curves.py` and plotted into
`final_models/eyes/validation_curves.png`.

## The number that matters

Fine-tuned dev-val AP (IoU .50:.95, best epoch, best_stg2 artefact):

| seed | best AP | best epoch | best_stg2 sha256, first 8 | wall | cost |
|---|---|---|---|---|---|
| 0 | 0.8329 | 47 | `5185d500` | 13.7 h | about $12.2 |
| 1 | 0.8341 | 47 | `e98de401` | 12.7 h | about $11.3 |
| 2 | 0.8333 | 47 | `32a783a3` | 13.4 h | about $11.9 |

The noise floor is a spread of 0.0012 (0.8329 to 0.8341), mean 0.8334. Under the
measurement discipline in POLICY.md, section 5, no A-against-B difference inside 0.0012 is
ever called a result.

Baseline on the same split, transforms and evaluator: untouched COCO base
`deim_dfine_s_coco.pth` (`a089c9eb`), zero-shot via
`third_party/configs/deim_s_person_v2_zeroshot_eval.yml` (80-class heads intact, remap off,
EMA off, see the config header): AP 0.584, with AP-small 0.003 and AR-small
0.035, so the base model is effectively blind to the small, distant people this
domain lives on.

The fine-tune gain is +0.249 AP, about 200x the seed spread.

## Run-shape notes (all three seeds identical)

- Curve: about 0.70 after epoch 1, rising to a 0.79 to 0.80 peak at epoch 3, then a
  deliberate dip when the augmentation policy engages (epoch 4), then monotone
  recovery, step-ups at augmentation ramp-down (28) and augmentation off (40), and
  the best value at the final epoch (47) in all three seeds. Still climbing at 48:
  a schedule-length lever exists and was not taken, because the runs that followed
  changed one variable at a time (POLICY.md, section 5) and none of them was the
  schedule; `research/eyes-seed-spread-and-soup.md` records the eyes queue as complete
  with data named as the next lever.
- Pre-augmentation peaks varied more across seeds (0.794 to 0.800) than finals did
  (0.833 to 0.834): the schedule converges seeds, and the augmentation-off tails agree.
- Operational facts recorded in the wrapper configurations: the launching shell must
  raise `ulimit -n` (the container soft file-descriptor limit kills the first eval);
  train workers 16 and val 8 (data starvation at the recipe's 4); AMP stable
  (no NaN.pth in any seed).

## Caveats (standing)

- Dev-val is the steering split. No frozen exam surface exists for the eyes lane: the
  detection exam once designed on MEVA footage was never built, and its design is not
  carried in this repository.
- These are prototype-lane weights; the checkpoint provenance rule (POLICY.md, rule 8)
  applies before anything ships.

## Model B, RT-DETRv4-S

Model B is the RT-DETRv4-S no-teacher run, B1 at seed 0, trained against the same
person_v2 corpus and the same dev-val split as the three seeds above. Two upstream
defects are carried as vendored patches under `third_party/patches/rtdetrv4/`: 0003 for
a device mismatch in the FLOPs profiler and 0004 for an evaluation position embedding
that was never moved to CUDA. Validation dataloader workers are set to 0, which is a
throughput setting and does not move the measured numbers. The completed result for
this arm is dev-val AP 0.8356 at checkpoint `345c30a5` against a bar of 0.8353, which is the
0.8341 best seed above at checkpoint `e98de401` plus the 0.0012 noise floor; it is recorded in
research/eyes-rtdetrv4-training-record.md.
