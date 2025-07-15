# dataset settings     create time 2024-11-3
# 该数据处理方法主要针对输入为5帧连续帧样本 --- posetrack数据集
dataset_type = 'opera.PosetrackVideoPoseDatasetV2'   # 单样本包含多帧图片---连续五帧 pre_pre_frame, pre_frame, now_frame, next_frame, next_next_frame
data_root = '/root/autodl-tmp/datasets/posetrack17/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='mmdet.LoadMulImageFromFileForPosetrackFrames5', to_float32=True), # done
    dict(type='opera.LoadAnnotations',
         with_bbox=True,
         with_keypoint=True,
         with_area=True),
    dict(
        type='mmdet.MulPhotoMetricDistortionForFrames5', # done
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='opera.MulKeypointRandomAffineForFrames5', # done
        max_rotate_degree=30.0,
        max_translate_ratio=0.,
        scaling_ratio_range=(1., 1.),
        max_shear_degree=0.,
        border_val=img_norm_cfg['mean'][::-1]),
    dict(type='opera.MulRandomFlip', flip_ratio=0.5), # done
    dict(
        type='mmdet.AutoAugment',
        policies=[
            [
                dict(
                    type='opera.MulResize', # done
                    img_scale=[(400, 1000), (1000, 1000)],
                    multiscale_mode='range',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='opera.MulResize', # done
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='opera.MulRandomCrop', # done
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='opera.MulResize',
                    img_scale=[(400, 1000), (1000, 1000)],
                    multiscale_mode='range',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='mmdet.Normalize', **img_norm_cfg), # done
    dict(type='mmdet.Pad', size_divisor=1), # done
    dict(type='mmdet.FixDefaultFormatBundle', num_frames=5,
         extra_keys=['gt_keypoints', 'gt_areas']), # done
    dict(type='mmdet.FixCollect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas']),
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='mmdet.LoadMulImageFromFileForPosetrackFrames5', to_float32=True),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(type='mmdet.RandomFlip'),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=1),
            dict(type='mmdet.MulImageToTensor', keys=['img_prev_prev','img_prev','img_now','img_next','img_next_next']), # 5帧图像
            dict(type='mmdet.FixCollect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'jsons/posetrack_train_fixed.json',
        img_prefix=data_root + 'images_renamed/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        samples_per_gpu=1,
        ann_file=data_root + 'jsons/posetrack_val_fixed.json',
        img_prefix=data_root + 'images_renamed/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        samples_per_gpu=1,
        ann_file=data_root + 'jsons/posetrack_val_fixed.json',
        img_prefix=data_root + 'images_renamed/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='keypoints',jsonfile_prefix='2025_2_7_swin_num_frames_5_posetrack17')
