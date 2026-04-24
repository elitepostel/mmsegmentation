_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

crop_size = (256, 256)
data_root = 'data/train_dataset_for_students_clean'

dataset_type = 'BaseSegDataset'

metainfo = dict(
    classes=('background', 'class_1', 'class_2'),
    palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0]],
)

model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            dict(
                type='DiceLoss',
                loss_weight=1.0,
            ),
        ],
    ),
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(
            img_path='img/train',
            seg_map_path='labels/train',
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(
            img_path='img/val',
            seg_map_path='labels/val',
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(
            img_path='img/test',
            seg_map_path='labels/test',
        ),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])

train_cfg = dict(type='IterBasedTrainLoop', max_iters=8000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        save_best='mDice',
        rule='greater',
        max_keep_ckpts=3,
    ),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, weight_decay=0.01),
)

param_scheduler = [
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=0, end=8000, by_epoch=False)
]

work_dir = './work_dirs/practicum/segformer_b0_ce_dice'
