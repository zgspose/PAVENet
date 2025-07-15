_base_ = [
    '../../_base_/datasets/posetrack18_video_keypoint.py', '../../_base_/default_runtime.py'
]
# backbone + encoder 训练
# pose-decoder 训练 （时空解码  ------ 三帧只有一份task_query,使用三个不同的线形层得到不同帧的相对偏移以及回归不同帧的参考点位置偏移）
# joint-decoder 训练 （时空解码  ------ 三帧只有一份task_query,初始位置使用pose输出的当前帧结果，使用三个不同的线形层得到不同帧的相对偏移以及回归不同帧的参考点位置偏移）
# end-to-end
model = dict(
    type='opera.VideoPoseV1',
    # 加载预训练权重  ----- 使用coco模拟时序训练好的权重
    init_cfg=dict(type='Pretrained', checkpoint='work_dirs/2025_2_7_swin_num_frames_3_posetrack17/epoch_10_best.pth'),
    backbone=dict(
       type='mmdet.SwinTransformer',
        num_frames=3,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='opera.VideoPoseHeadMulFrames',
        num_frames=3,  # 3帧输入
        num_keypoints=15,
        num_query=300,
        num_classes=1,  # only person
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_kpt_refine=True,
        as_two_stage=True,
        transformer=dict(
            type='opera.VideoPoseTransformerMulFrames', # 支持多帧输入，目前只支持3帧和5帧
            num_keypoints=15,
            num_frames=3, # 3帧输入
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
                type='opera.VideoPoseTransformerDecoderV2', # 适用于3帧输入
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
                        # 时空解码 --- 3帧
                        dict(
                            type='opera.MulFramesMultiScaleDeformablePoseAttentionNumFrames3',
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
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            refine_decoder=dict(
                type='mmcv.DeformableDetrTransformerDecoderV1',
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
                        # 时空解码 --- 3帧
                        dict(
                            type='mmcv.MulFramesMultiScaleDeformableAttentionNumFrames3',
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
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5),
        loss_kpt=dict(type='opera.RLELoss', loss_weight=1.0),
        loss_kpt_rpn=dict(type='opera.RLELoss', loss_weight=1.0),
        loss_oks=dict(type='opera.OKSLoss', num_keypoints=15, loss_weight=0.0),
        loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=0.0),
        # loss_kpt_refine=dict(type='mmdet.L1Loss', loss_weight=80.0),
        loss_kpt_refine=dict(type='opera.RLELoss', loss_weight=1.0),
        loss_oks_refine=dict(type='opera.OKSLoss', num_keypoints=15, loss_weight=0.0)),
    train_cfg=dict(
        assigner=dict(
            type='opera.PoseHungarianAssigner',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            kpt_cost=dict(type='opera.KptL1Cost', weight=70.0),
            oks_cost=dict(type='opera.OksCost', num_keypoints=15, weight=7.0))),
    test_cfg=dict(max_per_img=20))  # set 'max_per_img=20' for time counting
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
    cumulative_iters=16,
    grad_clip=dict(max_norm=0.1, norm_type=2))
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[10])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
