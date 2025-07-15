_base_ = [
    '../../_base_/datasets/coco_video_keypoint_num_frames_5.py', '../../_base_/default_runtime.py'
]
# 添加时间-2025-1-30
# 使用coco数据集来模拟时序数据微调petr原始训练好的resnet50模型权重
model = dict(
    type='opera.PETR',
    # 使用原始petr在coco上训练得到的权重，17个关节点
    init_cfg=dict(type='Pretrained', checkpoint='pretrain_checkpoints/petr_r50_16x2_100e_coco.pth'),
    backbone=dict(
        type='mmdet.ResNet',
        input_type='mul_frames', # 使用单样本多帧输入数据，单样本使用5帧数据
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='opera.VideoPoseHeadMulFrames',
        num_frames=5,
        num_keypoints=17,
        num_query=300,
        num_classes=1,  # only person
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_kpt_refine=True,
        as_two_stage=True,
        transformer=dict(
            type='opera.VideoPoseTransformerMulFrames',
            num_keypoints=17,
            num_frames=5,
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
                type='opera.VideoPoseTransformerDecoderV2_1', # 适用于5帧输入
                num_keypoints=17,
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
                        # 时空解码 --- 5帧
                        dict(
                            type='opera.MulFramesMultiScaleDeformablePoseAttentionNumFrames5',
                            num_points=17,
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
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            refine_decoder=dict(
                type='mmcv.DeformableDetrTransformerDecoderV1_2',
                num_layers=2,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        # 时空解码 --- 5帧
                        dict(
                            type='mmcv.MulFramesMultiScaleDeformableAttentionNumFrames5',
                            embed_dims=256,
                            im2col_step=128)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='mmcv.SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        # 需要平衡下cls-loss 与 rle-loss
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_kpt=dict(type='opera.RLELoss', loss_weight=1.0),
        loss_kpt_rpn=dict(type='opera.RLELoss', loss_weight=1.0),
        loss_oks=dict(type='opera.OKSLoss', num_keypoints=17, loss_weight=0.0),
        loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=0.0),
        loss_kpt_refine=dict(type='opera.RLELoss', loss_weight=1.0),
        loss_oks_refine=dict(type='opera.OKSLoss', num_keypoints=17, loss_weight=0.0)),
    train_cfg=dict(
        assigner=dict(
            type='opera.PoseHungarianAssigner',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            kpt_cost=dict(type='opera.KptL1Cost', weight=70.0),
            oks_cost=dict(type='opera.OksCost', weight=7.0))),
    test_cfg=dict(max_per_img=100))  # set 'max_per_img=20' for time counting
# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-5,
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
lr_config = dict(policy='step', step=[7])
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
