# dataset settings
dataset_type = 'NOCSDataset'
data_root = 'data/nocs/'
scale = (640, 480)

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepthImageFromFile', backend_args=backend_args),
    dict(type='Load9DPoseAnnotations', with_bbox=True),
    dict(type='Resize', scale=scale, keep_ratio=True),
    dict(type='ConvertDepthToPointCloud'),
    dict(type='Pack9DPoseInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepthImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='Load9DPoseAnnotations', with_bbox=True),
    dict(type='ConvertDepthToPointCloud'),
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
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                split='camera_train',
                pipeline=train_pipeline,
                backend_args=backend_args),
            dict(
                type=dataset_type,
                data_root=data_root,
                split='real_train',
                pipeline=train_pipeline,
                backend_args=backend_args)
        ])
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
        # split='overfit',
        split='real_test',
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='NOCSMetric',
    format_only=False)
test_evaluator = val_evaluator
