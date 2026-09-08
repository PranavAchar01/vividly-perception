# B1 RT-DETRv4-S no-teacher rival run: clears the bar by a narrow margin

Dated 2026-09-03. B1 clears the pre-registered comparison. Final
dev-val AP 0.8356 against the required 0.8353 (the incumbent at 0.8341 plus a 0.0012
noise margin). The margin is +0.0003, the smallest margin the rule admits, so the
outcome recorded below is stated conservatively.

## The run

| item | value |
|---|---|
| model | RT-DETRv4-S, no-teacher recipe (10.2M params) |
| config | `third_party/configs/rtv4_s_person_v2_noteacher.yml`, installed by `scripts/pod/setup_pod.sh` into the trainer's own `configs/rtv4/` directory, against which the vendored trainer is invoked as `RTV4/configs/rtv4/rtv4_s_person_v2_noteacher.yml` (val `num_workers: 0`) |
| init | `rtv4_s_coco.pth` sha256 `238a3f65…` (COCO-pretrained release) |
| data | person_v2: 55,200 training images and 828,230 boxes; dev-val 1,800 images and 33,755 boxes (Warehouse_005 held out by scene) |
| schedule | 48 epochs, batch 32, AMP, seed 0; about 13.0 h on the RTX 5090 pod (about $9.0 at $0.69/h) |
| best checkpoint | `best_stg2.pth` sha256 `345c30a539c2b6bb665302825f418322284b98f6cbcf55d58d84ffeb75b16e7e` (epoch 47; backed up to `outputs/pod_backup/b1_best_stg2_345c30a5.pth`) |
| stage-1 best | `best_stg1.pth` sha256 `365b6925…` (pre-no-aug phase) |
| log | `outputs/pod_backup/train_rtv4_s0_final.log`, a build output that is not in version control |

## Final dev-val numbers (epoch 47, checkpoint 345c30a5)

- AP .50:.95 = 0.8356 (exact 0.8355529763877473, the `best_stat` line)
- AP50 0.958, AP75 0.883
- AP small 0.128, medium 0.722, large 0.931
- AR@100 0.860 (AR50 0.977, AR75 0.907)

## Against the DEIM incumbent (model A noise floor, research/eyes-noise-floor.md)

| run | best dev-val AP | checkpoint |
|---|---|---|
| DEIM seed 0 | 0.8329 | 5185d500 |
| DEIM seed 1 (incumbent) | 0.8341 | e98de401 |
| DEIM seed 2 | 0.8333 | 32a783a3 |
| noise floor | 0.8334 plus or minus 0.0006 | 3 seeds |
| **bar for B1** | **0.8353** | the incumbent plus 0.0012, which is two sigma |
| **B1 seed 0** | **0.8356** | **345c30a5** |

Trajectory note: B1 tracked DEIM seed 1's per-epoch curve almost exactly for 40
epochs (never more than about 0.002 apart), then out-gained it in the final no-aug
phase (+0.0073 over epochs 41 to 47, against the incumbent's roughly +0.006). Curves:
`final_models/eyes/validation_curves.png` and
`final_models/eyes/champion_vs_baseline.png`, rendered from the per-epoch series in
`final_models/eyes/curves/*.csv` (columns `epoch`, `train_loss`, `val_ap`) by
`scripts/analysis/plot_final_curves.py`.

## Outcome

1. B1 (RTv4-S no-teacher, 345c30a5) is the v2 detector's model of
   record under the pre-registered rule (a single qualifying run must clear the
   incumbent plus two sigma; it did, 0.8356 above 0.8353).
2. The win is one seed against a three-seed floor, by 0.0003. B1's own seed
   noise was measured afterwards: seed 1 scored 0.8348 at checkpoint 17362dca, a
   seed spread of 0.00075, and seed 2 was cut on 2026-09-06
   (research/eyes-seed-spread-and-soup.md). DEIM e98de401 is retained as the
   fallback and nothing that only B1 can do is promised.
3. Practical read: the two recipes are statistically equivalent on person_v2
   dev-val; B1 clears the rule, not by a margin that changes product behaviour.
   The GPU queue moves on rather than spending 26 h to sharpen a tie.

## Startup check (recorded at launch, 2026-09-03 07:23 UTC)

Missed and unmatched keys at init were exactly the class heads
(`denoising_class_embed`, `enc_score_head`, `dec_score_head.0-2`), which is correct
for COCO to person-only transfer; everything else loaded from 238a3f65.
