_base_ = [
    '../../_base_/datasets/posetrack_video_keypoint.py', '../../_base_/default_runtime.py'
]
model = dict(
    type='opera.VideoPoseV1',
    # 加载预训练权重  ----- 在posetrack单帧上使用rle训练好的模型权重 map=74
    init_cfg=dict(type='Pretrained', checkpoint='work_dirs/2024_10_9_videopose_r50_20e_coco_fuse_time_v1/epoch_8.pth'),
    backbone=dict(
        type='mmdet.ResNet',
        input_type='mul_frames',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='opera.VideoPoseHeadV10',
        num_keypoints=15,
        num_query=300,
        num_classes=1,  # only person
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_kpt_refine=True,
        as_two_stage=True,
        transformer=dict(
            type='opera.VideoPoseTransformerV10',
            num_keypoints=15,
            encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='opera.VideoPoseTransformerDecoderV10',
                num_keypoints=15,
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        # 使用不同的线性层，当前帧使用预训练权重初始化，辅助帧随机初始化
                        dict(
                            type='opera.MulFramesMultiScaleDeformablePoseAttentionV4',
                            num_points=15,
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 
                                     'ffn', 'norm'))),
            hm_encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='mmcv.SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5.0),
        # decoder使用
        loss_kpt=dict(type='opera.RLELoss', loss_weight=1.0),
        # encoder使用
        loss_kpt_rpn=dict(type='opera.RLELoss', loss_weight=1.0),
        # hm使用
        loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='opera.PoseHungarianAssignerV10',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=5.0),
            kpt_cost=dict(type='opera.RLECost', weight=1.0))),
    test_cfg=dict(max_per_img=100))  # set 'max_per_img=20' for time counting
# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=8,
    grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
