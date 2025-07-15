# dataset settings     create time 2024-11-18
# 本配置文件用于利用coco数据集来预训练时序模型
# resnet50 --- done coco_rle_15keypoints.pth
# hrnetw48 --- todo
# swin-L ----- running
dataset_type = 'opera.CocoVideoPoseDataset5'   # 单样本包含多帧图片---连续五帧
data_root = '/root/autodl-tmp/datasets/coco/' 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='mmdet.LoadMulImageFromFileForCocoFrames5', to_float32=True), # 单样本包含5帧图像
    dict(type='opera.LoadAnnotations',
         with_bbox=True,
         with_keypoint=True,
         with_area=True),
    dict(
        type='mmdet.MulPhotoMetricDistortionForFrames5',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='opera.MulKeypointRandomAffineForFrames5',
        max_rotate_degree=30.0,
        max_translate_ratio=0.,
        scaling_ratio_range=(1., 1.),
        max_shear_degree=0.,
        border_val=img_norm_cfg['mean'][::-1]),
    dict(type='opera.MulRandomFlip', flip_ratio=0.5),
    dict(
        type='mmdet.AutoAugment',
        policies=[
            [
                dict(
                    type='opera.MulResize',
                    img_scale=[(400, 1000), (1000, 1000)],
                    multiscale_mode='range',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='opera.MulResize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='opera.MulRandomCrop',
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
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=1),
    dict(type='mmdet.FixDefaultFormatBundle', num_frames=5,
         extra_keys=['gt_keypoints', 'gt_areas']),
    dict(type='mmdet.FixCollect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas']),
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='mmdet.LoadMulImageFromFileForCocoFrames5', to_float32=True),
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
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_train2017.json',
        img_prefix=data_root + 'images/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='keypoints',jsonfile_prefix='2025_2_1_petr_r50_num_frame_5_bs16_20e_coco_rle')
evaluation = dict(interval=1, metric='keypoints',jsonfile_prefix='petr_swin_num_frame_5_bs16_20e_coco_rle')
