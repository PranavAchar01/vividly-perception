# RTMPose-m person-keypoints fine-tune: v2 pose lane, run 1 (recipe reproduction).
#
# This recipe produced the joints model of record: seed 0, best checkpoint at epoch 38,
# sha256 a0a035fe, keypoint AP 0.774115 on COCO val2017 with ground-truth boxes, beside
# seed 1 (8d990435, 0.774000) and seed 2 (5bcca5bd, 0.774057). All three ran on
# 2026-09-04 (final_models/joints/README.md, research/pose-seed-floor-and-champion.md,
# research/pose-p0-training-record.md).
#
# The file sets no randomness key. The seed is supplied at launch through
# --cfg-options, so reproducing seed 0 means passing randomness.seed=0 explicitly;
# the file is not deterministic on its own.
#
# Installed into <mmpose>/configs/body_2d_keypoint/rtmpose/coco/ by the command
# TRAINING.md gives for this lane, where _base_ resolves against the vendored repo at
# pin 759b39c1 (research/pose-stack.md). Every path below is the pod layout described
# in TRAINING.md, not a repository path.
#
#   .venv/bin/python tools/train.py \
#       configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_person_v2_ft.py \
#       --cfg-options randomness.seed=0
#
# What run 1 established: the sm_120 stack and this scaled schedule reproduce
# paper-grade COCO keypoint quality from the body7 init. The gate was the body7 init's
# own AP measured on the identical protocol, 0.764499; the run scored 0.774115, a gain
# of 0.009616 (final_models/joints/README.md, before-and-after table).
# Top-view adaptation data is a separate, later, gated decision; nothing here
# trains on MEVA or any pseudo-label.
#
# Provenance card: deltas from the published recipe
# (rtmpose-m_8xb256-420e_coco-256x192.py at pin 759b39c1); everything not listed
# is inherited unchanged (codec sigma (4.9,5.66), split_ratio 2.0, AdamW wd 0.05,
# stage-2 pipeline switch, EMA hook, RandomHalfBody/Albumentations aug):
#   max_epochs 420 -> 60; stage2 30 -> 5; cosine begin 210 -> 30 (max//2, the
#   recipe's own formula); val_interval 10 -> 2
#       fine-tune from the fully-converged body7 init needs a fraction of the
#       from-scratch schedule; every phase moves by the recipe's own ratios
#       (stage2 7% of run, cosine over the back half). Run 1 completed and the
#       three-seed floor is 0.77406 with a spread of 0.000115, so this schedule is
#       the recipe of record, not a proposal.
#   base_lr 4e-3 -> 1e-4
#       two standard factors composed: ~10x down for fine-tuning a converged
#       init, then linear batch scaling 256/1024 (auto_scale_lr base_batch_size
#       is 1024 in the recipe; this recipe runs one 5090 at batch 256, so the scaling is
#       baked in here rather than left to a launch flag). 1e-4 is the measured value
#       behind all three seeds.
#   warmup 1000 iters -> 500
#       scaled with schedule length; body7 init needs little warmup.
#   train/val batch 256/64 kept; workers set for the 48-vCPU pod (throughput only;
#       seeds 1 and 2 ran with train workers 32 instead, which is the one further
#       difference between the three runs, per
#       research/pose-seed-floor-and-champion.md).
#   init_cfg -> load body7 checkpoint
#       /workspace/ckpt/rtmpose-m_simcc-body7_...pth, sha256 e48f03d0..., recorded in
#       full on final_models/joints/README.md and in scripts/pose/soup_checkpoints.py.
#       Counsel flag on its 7-dataset ancestry is recorded in research/pose-stack.md,
#       in the counsel section on licensing exposure, and POLICY.md, rule 8, checkpoint
#       provenance, applies before any ship.
#   dataset -> /workspace/data/coco2017 person_keypoints (118,287 train imgs /
#       262,465 kpt anns, verified 2026-09-01). Val uses GT bboxes (deterministic,
#       no external detections artifact); published-table comparisons must note
#       the protocol difference. This val loop is the trainer-side protocol, COCO
#       val2017 with ground-truth boxes, which is the pose exam surface named in
#       DATASETS.md. The deployed-path protocol, the exported graph over a frozen
#       500-image COCO subset with detector boxes, is a different quantity and is
#       recorded in final_models/joints/README.md and TRAINING.md; the two numbers
#       are never subtracted from each other.
_base_ = ["./rtmpose-m_8xb256-420e_coco-256x192.py"]

max_epochs = 60
stage2_num_epochs = 5
base_lr = 1e-4

train_cfg = dict(max_epochs=max_epochs, val_interval=2)

optim_wrapper = dict(optimizer=dict(lr=base_lr))

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

model = dict(
    init_cfg=dict(
        type="Pretrained",
        checkpoint="/workspace/ckpt/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth",
    ),
)

data_root = "/workspace/data/coco2017/"

train_dataloader = dict(
    num_workers=16,
    dataset=dict(
        data_root=data_root,
        ann_file="annotations/person_keypoints_train2017.json",
        data_prefix=dict(img="train2017/"),
    ),
)

val_dataloader = dict(
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        ann_file="annotations/person_keypoints_val2017.json",
        bbox_file=None,  # GT-bbox eval; see provenance card
        data_prefix=dict(img="val2017/"),
    ),
)

val_evaluator = dict(ann_file=data_root + "annotations/person_keypoints_val2017.json")

custom_hooks = [
    dict(
        type="EMAHook", ema_type="ExpMomentumEMA", momentum=0.0002, update_buffers=True, priority=49
    ),
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline={{_base_.train_pipeline_stage2}},
    ),
]
