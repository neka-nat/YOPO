# dataset settings
dataset_type = 'HouseCat6DDataset'
data_root = 'data/housecat6d/'
scale = (1096, 852)

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Load9DPoseAnnotations', with_bbox=True),
    dict(type='Resize', scale=scale, keep_ratio=True),
    dict(type='Pack9DPoseInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='Load9DPoseAnnotations', with_bbox=True),
    dict(
        type='Pack9DPoseInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'intrinsic', 'models_info_path'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),

    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split='train',
            pipeline=train_pipeline,
            backend_args=backend_args),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='HouseCat6DMetric',
    format_only=False)
test_evaluator = val_evaluator
