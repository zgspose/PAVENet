# Copyright (c) Hikvision Research Institute. All rights reserved.
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Linear, bias_init_with_prob, constant_init, normal_init,
                      build_activation_layer)
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.dense_heads import AnchorFreeHead
import torch.distributions as distributions

from opera.core.bbox import build_assigner, build_sampler
from opera.core.keypoint import gaussian_radius, draw_umich_gaussian
from opera.models.utils import build_positional_encoding, build_transformer
from ..builder import HEADS, build_loss
from easydict import EasyDict
from collections import Counter

# 添加时间 2024-10-21 适用于只在Pose-decoder中进行时空编码模块 joint-decoder 单帧 rle_loss
# match_loss: cls loss + l1 loss
# tatal_loss: cls loss + rle loss
# 截断joint-decoder,只测试pose-decoder结果
@HEADS.register_module()
class VideoPoseHeadV12_1(AnchorFreeHead):
    """Head of `End-to-End Multi-Person Pose Estimation with Transformers`.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_kpt_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the keypoint regression head.
            Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): ConfigDict is used for
            building the Encoder and Decoder. Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_kpt (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_oks (obj:`mmcv.ConfigDict`|dict): Config of the
            regression oks loss. Default `OKSLoss`.
        loss_hm (obj:`mmcv.ConfigDict`|dict): Config of the
            regression heatmap loss. Default `NegLoss`.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        with_kpt_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to True.
        train_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_kpt_fcs=2,
                 num_keypoints=17,
                 transformer=None,
                 sync_cls_avg_factor=True,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_kpt=dict(type='L1Loss', loss_weight=70.0),
                 loss_oks=dict(type='OKSLoss', loss_weight=2.0),
                 loss_hm=dict(type='CenterFocalLoss', loss_weight=4.0),
                 as_two_stage=True,
                 with_kpt_refine=True,
                 train_cfg=dict(
                     assigner=dict(
                         type='PoseHungarianAssigner',
                         cls_cost=dict(type='FocalLossCost', weight=2.0),
                         kpt_cost=dict(type='KptL1Cost', weight=70.0),
                         oks_cost=dict(type='OksCost', weight=7.0))),
                 loss_kpt_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_kpt_refine=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_oks_refine=dict(type='opera.OKSLoss', loss_weight=2.0),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            # assert loss_kpt['loss_weight'] == assigner['kpt_cost'][
            #     'weight'], 'The regression L1 weight for loss and matcher ' \
            #     'should be exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='mmdet.PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_kpt_fcs = num_kpt_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.as_two_stage = as_two_stage
        self.with_kpt_refine = with_kpt_refine
        self.num_keypoints = num_keypoints
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        else:
            raise RuntimeError('only "as_two_stage=True" is supported.')
        self.loss_cls = build_loss(loss_cls)
        self.loss_kpt = build_loss(loss_kpt)
        self.loss_kpt_rpn = build_loss(loss_kpt_rpn)
        self.loss_kpt_refine = build_loss(loss_kpt_refine)
        self.loss_oks = build_loss(loss_oks)
        self.loss_oks_refine = build_loss(loss_oks_refine)
        self.loss_hm = build_loss(loss_hm)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and keypoint branch of head."""
        # encoder、pose-decoder所使用分类子网络
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        # encoder、pose-deocder所使用回归关节点以及偏移子网络
        kpt_branch = []
        kpt_branch.append(Linear(self.embed_dims, 512))
        kpt_branch.append(nn.ReLU())
        for _ in range(self.num_kpt_fcs):
            kpt_branch.append(Linear(512, 512))
            kpt_branch.append(nn.ReLU())
        kpt_branch.append(Linear(512, 2 * self.num_keypoints))
        kpt_branch = nn.Sequential(*kpt_branch)
        
        # pose-decoder用于预测关节点的sigma
        dec_fc_sigma_branch = []
        for _ in range(self.num_kpt_fcs):
            dec_fc_sigma_branch.append(Linear(self.embed_dims, self.embed_dims))
        dec_fc_sigma_branch.append(Linear_with_norm(self.embed_dims, 2 * self.num_keypoints, norm=False))
        dec_fc_sigma_branch = nn.Sequential(*dec_fc_sigma_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last kpt_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_kpt_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.kpt_branches = _get_clones(kpt_branch, num_pred)
            self.dec_fc_sigma_branches = _get_clones(dec_fc_sigma_branch, num_pred)
            
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.kpt_branches = nn.ModuleList(
                [kpt_branch for _ in range(num_pred)])

        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)
        

        refine_kpt_branch = []
        for _ in range(self.num_kpt_fcs):
            refine_kpt_branch.append(Linear(self.embed_dims, self.embed_dims))
            refine_kpt_branch.append(nn.ReLU())
        refine_kpt_branch.append(Linear(self.embed_dims, 2))
        refine_kpt_branch = nn.Sequential(*refine_kpt_branch)
        
        # joint-decoder用于预测关节点的sigma
        refine_fc_sigma_branch = []
        for _ in range(self.num_kpt_fcs):
            refine_fc_sigma_branch.append(Linear(self.embed_dims, self.embed_dims))
        refine_fc_sigma_branch.append(Linear_with_norm(self.embed_dims, 2, norm=False))
        refine_fc_sigma_branch = nn.Sequential(*refine_fc_sigma_branch)
        
        if self.with_kpt_refine:
            num_pred = self.transformer.refine_decoder.num_layers
            self.refine_kpt_branches = _get_clones(refine_kpt_branch, num_pred)
            self.refine_fc_sigma_branches = _get_clones(refine_fc_sigma_branch, num_pred) 
        self.fc_hm = Linear(self.embed_dims, self.num_keypoints)
        
        # add flow model
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
        
        # pose-decoder中使用的流模型
        dec_prior = distributions.MultivariateNormal(torch.zeros(2) + 0.5, torch.eye(2))
        self.dec_flow = RealNVP(nets, nett, masks, dec_prior)
        
        # joint-decoder中使用的流模型
        prior = distributions.MultivariateNormal(torch.zeros(2) + 0.5, torch.eye(2))
        self.flow = RealNVP(nets, nett, masks, prior)
        

    def init_weights(self):
        """Initialize weights of the PETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.kpt_branches:
            constant_init(m[-1], 0, bias=0)
        # initialization of keypoint refinement branch
        if self.with_kpt_refine:
            for m in self.refine_kpt_branches:
                constant_init(m[-1], 0, bias=0)
        # initialize bias for heatmap prediction
        bias_init = bias_init_with_prob(0.1)
        normal_init(self.fc_hm, std=0.01, bias=bias_init)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            outputs_classes (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should include background.
            outputs_kpts (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, K*2].
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (N, h*w, num_class). Only when
                as_two_stage is Ture it would be returned, otherwise
                `None` would be returned.
            enc_outputs_kpt (Tensor): The proposal generate from the
                encode feature map, has shape (N, h*w, K*2). Only when
                as_two_stage is Ture it would be returned, otherwise
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)  # 这里实际上batch-size大小 为 batch-size // 3
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id // 3]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_kpt, hm_proto, memory = \
                self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    kpt_branches=self.kpt_branches \
                        if self.with_kpt_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches \
                        if self.as_two_stage else None  # noqa:E501
            )
        # 多层pose-decoder迭代的pose-token
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_kpts = []
        outputs_sigmas = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp_kpt = self.kpt_branches[lvl](hs[lvl])
            sigma = self.dec_fc_sigma_branches[lvl](hs[lvl])
            
            assert reference.shape[-1] == self.num_keypoints * 2
            tmp_kpt += reference
            outputs_kpt = tmp_kpt.sigmoid()
            output_sigma = sigma.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_kpts.append(outputs_kpt)
            outputs_sigmas.append(output_sigma)
            
        
        outputs_classes = torch.stack(outputs_classes)
        outputs_kpts = torch.stack(outputs_kpts)
        outputs_sigmas = torch.stack(outputs_sigmas)
        
        if hm_proto is not None:
            # get heatmap prediction (training phase)
            hm_memory, hm_mask = hm_proto
            hm_pred = self.fc_hm(hm_memory)
            hm_proto = (hm_pred.permute(0, 3, 1, 2), hm_mask)

        if self.as_two_stage:
            new_mlvl_masks = [mlvl_mask[1::3] for mlvl_mask in mlvl_masks]
            new_memory = memory[:, 1::3]
            # 返回dec_sigma 参与pose-decoder损失计算
            return outputs_classes, outputs_kpts, outputs_sigmas, \
                enc_outputs_class, enc_outputs_kpt.sigmoid(), \
                hm_proto, new_memory, new_mlvl_masks
        else:
            raise RuntimeError('only "as_two_stage=True" is supported.')
        
    def forward_refine(self, memory, mlvl_masks, refine_targets, losses,
                       img_metas):
        """Forward function.

        Args:
            mlvl_masks (tuple[Tensor]): The key_padding_mask from
                different level used for encoder and decoder,
                each is a 3D-tensor with shape (bs, H, W).
            losses (dict[str, Tensor]): A dictionary of loss components.
            img_metas (list[dict]): List of image information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        kpt_preds, kpt_targets, area_targets, kpt_weights = refine_targets
        pos_inds = kpt_weights.sum(-1) > 0
        if pos_inds.sum() == 0:
            pos_kpt_preds = torch.zeros_like(kpt_preds[:1])
            pos_img_inds = kpt_preds.new_zeros([1], dtype=torch.int64)
        else:
            pos_kpt_preds = kpt_preds[pos_inds]
            pos_img_inds = (pos_inds.nonzero() / self.num_query).squeeze(1).to(
                torch.int64)
            
        hs, init_reference, inter_references = self.transformer.forward_refine(
            mlvl_masks,
            memory,
            pos_kpt_preds.detach(),
            pos_img_inds,
            kpt_branches=self.refine_kpt_branches if self.with_kpt_refine else None,  # noqa:E501
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_kpts = []
        outputs_sigmas = []
        outputs_scores = []

        # 这部分修改，新添加了sigma的操作
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            
            tmp_kpt = self.refine_kpt_branches[lvl](hs[lvl])
            
            tmp_sigma = self.refine_fc_sigma_branches[lvl](hs[lvl]).sigmoid()
            score = 1 - tmp_sigma
            score = torch.mean(score, dim=2, keepdim=True)
            outputs_score = score
            outputs_scores.append(outputs_score)
            
            assert reference.shape[-1] == 2
            tmp_kpt += reference
            outputs_kpt = tmp_kpt.sigmoid()
            outputs_kpts.append(outputs_kpt)
            
            outputs_sigma = tmp_sigma
            outputs_sigmas.append(outputs_sigma)
            
        outputs_kpts = torch.stack(outputs_kpts)
        outputs_sigmas = torch.stack(outputs_sigmas)
        outputs_scores = torch.stack(outputs_scores)

        if not self.training:
            return outputs_kpts, outputs_scores, outputs_sigmas

        batch_size = mlvl_masks[0].size(0)
        factors = []
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            factor = mlvl_masks[0].new_tensor(
                [img_w, img_h, img_w, img_h],
                dtype=torch.float32).unsqueeze(0).repeat(self.num_query, 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        factors = factors[pos_inds][:, :2].repeat(1, kpt_preds.shape[-1] // 2)

        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        num_total_pos = kpt_weights.new_tensor([outputs_kpts.size(1)])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        pos_kpt_weights = kpt_weights[pos_inds]
        pos_kpt_targets = kpt_targets[pos_inds]
        pos_kpt_targets_scaled = pos_kpt_targets * factors
        pos_areas = area_targets[pos_inds]
        pos_valid = kpt_weights[pos_inds, 0::2]
        
        # 修改，将原始的l1 loss修改为rle loss,这里输出的joint-decoder_layer共有两层
        for i, (kpt_refine_preds, kpt_refine_sigma) in enumerate(zip(outputs_kpts, outputs_sigmas)):
            # 如果预测的结果为空，没有人的情况下
            if pos_inds.sum() == 0:
                loss_kpt = kpt_refine_preds.sum() * 0
                losses[f'd{i}.loss_kpt_refine'] = loss_kpt
                continue
            # kpt rle loss
            bs = kpt_refine_preds.size(0)
            all_gts = pos_kpt_targets.reshape(bs, -1, 2)
            gt_weights = pos_kpt_weights.reshape(bs, -1, 2)
            pred = kpt_refine_preds
            sigmas = kpt_refine_sigma
            bar_mu = (pred - all_gts) / sigmas
            log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(bs, -1, 1)
            nf_loss = torch.log(sigmas) - log_phi 
            pred_jts = pred
            sigma = sigmas
            
            output = EasyDict(
                pred_jts=pred_jts,
                sigma=sigma,
                nf_loss=nf_loss
            )
            labels = EasyDict(
                target_uv=all_gts,
                target_uv_weight=gt_weights
            )
            loss_kpt = self.loss_kpt_refine(output, labels)
            losses[f'd{i}.loss_kpt_refine'] = loss_kpt

        return losses

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_keypoints=None,
                      gt_areas=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_keypoints (list[Tensor]): Ground truth keypoints of the image,
                shape (num_gts, K*3).
            gt_areas (list[Tensor]): Ground truth mask areas of each box,
                shape (num_gts,).
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        memory, mlvl_masks = outs[-2:]
        outs = outs[:-2]
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, gt_keypoints, gt_areas, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypoints, gt_areas,
                                  img_metas)
        losses_and_targets = self.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses, refine_targets = losses_and_targets
        # get pose refinement loss
        losses = self.forward_refine(memory, mlvl_masks, 
                                     refine_targets, losses, img_metas)
        return losses

    @force_fp32(apply_to=('all_cls_scores', 'all_kpt_preds'))
    def loss(self,
             all_cls_scores,
             all_kpt_preds,
             all_sigma_preds,
             enc_cls_scores,
             enc_kpt_preds,
             enc_hm_proto,
             gt_bboxes_list,
             gt_labels_list,
             gt_keypoints_list,
             gt_areas_list,
             img_metas,
             gt_bboxes_ignore=None):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_kpt_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (x_{i}, y_{i}) and shape
                [nb_dec, bs, num_query, K*2].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map, has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_kpt_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, K*2). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                    ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_keypoints_list = [
            gt_keypoints_list for _ in range(num_dec_layers)
        ]
        all_gt_areas_list = [gt_areas_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        # 计算pose-decoder损失
        losses_cls, losses_kpt, losses_oks, kpt_preds_list, kpt_targets_list, \
            area_targets_list, kpt_weights_list = multi_apply(
                self.loss_single, all_cls_scores, all_kpt_preds, all_sigma_preds, 
                all_gt_labels_list, all_gt_keypoints_list,
                all_gt_areas_list, img_metas_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_kpt = \
                self.loss_single_rpn(
                    enc_cls_scores, enc_kpt_preds, binary_labels_list,
                    gt_keypoints_list, gt_areas_list, img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_kpt'] = enc_losses_kpt

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_kpt'] = losses_kpt[-1]
        loss_dict['loss_oks'] = losses_oks[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_kpt_i, loss_oks_i in zip(
                losses_cls[:-1], losses_kpt[:-1], losses_oks[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_kpt'] = loss_kpt_i
            loss_dict[f'd{num_dec_layer}.loss_oks'] = loss_oks_i
            num_dec_layer += 1

        # losses of heatmap generated from P3 feature map
        hm_pred, hm_mask = enc_hm_proto
        loss_hm = self.loss_heatmap(hm_pred, hm_mask, gt_keypoints_list,
                                    gt_labels_list, gt_bboxes_list)
        loss_dict['loss_hm'] = loss_hm

        return loss_dict, (kpt_preds_list[-1], kpt_targets_list[-1],
                           area_targets_list[-1], kpt_weights_list[-1])

    def loss_heatmap(self, hm_pred, hm_mask, gt_keypoints, gt_labels,
                     gt_bboxes):
        assert hm_pred.shape[-2:] == hm_mask.shape[-2:]
        num_img, _, h, w = hm_pred.size()
        # placeholder of heatmap target (Gaussian distribution)
        hm_target = hm_pred.new_zeros(hm_pred.shape)
        for i, (gt_label, gt_bbox, gt_keypoint) in enumerate(
                zip(gt_labels, gt_bboxes, gt_keypoints)):
            if gt_label.size(0) == 0:
                continue
            gt_keypoint = gt_keypoint.reshape(gt_keypoint.shape[0], -1,
                                              3).clone()
            gt_keypoint[..., :2] /= 8
            assert gt_keypoint[..., 0].max() <= w  # new coordinate system
            assert gt_keypoint[..., 1].max() <= h  # new coordinate system
            gt_bbox /= 8
            gt_w = gt_bbox[:, 2] - gt_bbox[:, 0]
            gt_h = gt_bbox[:, 3] - gt_bbox[:, 1]
            for j in range(gt_label.size(0)):
                # get heatmap radius
                kp_radius = torch.clamp(
                    torch.floor(
                        gaussian_radius((gt_h[j], gt_w[j]), min_overlap=0.9)),
                    min=0, max=3)
                for k in range(self.num_keypoints):
                    if gt_keypoint[j, k, 2] > 0:
                        gt_kp = gt_keypoint[j, k, :2]
                        gt_kp_int = torch.floor(gt_kp)
                        draw_umich_gaussian(hm_target[i, k], gt_kp_int,
                                            kp_radius)
        # compute heatmap loss
        hm_pred = torch.clamp(
            hm_pred.sigmoid_(), min=1e-4, max=1 - 1e-4)  # refer to CenterNet
        loss_hm = self.loss_hm(hm_pred, hm_target, mask=~hm_mask.unsqueeze(1))
        return loss_hm

    def loss_single(self,
                    cls_scores,
                    kpt_preds,
                    sigma_preds,
                    gt_labels_list,
                    gt_keypoints_list,
                    gt_areas_list,
                    img_metas):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i}) and
                shape [bs, num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        
        # gt匹配
        cls_reg_targets = self.get_targets(cls_scores_list, kpt_preds_list,
                                           gt_labels_list, gt_keypoints_list,
                                           gt_areas_list, img_metas)
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         area_targets_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        kpt_targets = torch.cat(kpt_targets_list, 0)
        kpt_weights = torch.cat(kpt_weights_list, 0)
        area_targets = torch.cat(area_targets_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt keypoints accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale keypoints
        factors = []
        for img_meta, kpt_pred in zip(img_metas, kpt_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = kpt_pred.new_tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0).repeat(
                                              kpt_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape(-1, kpt_preds.shape[-1])
        sigma_preds = sigma_preds.reshape(-1, sigma_preds.shape[-1])
        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        # kpt rle loss
        # 得到与gt匹配的pred
        pos_inds = kpt_weights.sum(-1) > 0
        if pos_inds.sum() == 0:
            pos_kpt_preds = torch.zeros_like(kpt_preds[:1])
            loss_kpt = pos_kpt_preds.sum() * 0
        else:
            pos_kpt_preds = kpt_preds[pos_inds]
            pos_kpt_targets = kpt_targets[pos_inds]
            pos_kpt_weights = kpt_weights[pos_inds]
            pos_kpt_sigmas = sigma_preds[pos_inds]
            bs = pos_kpt_preds.size(0)
            pos_kpt_preds = pos_kpt_preds.reshape(bs, -1, 2)
            all_gts = pos_kpt_targets.reshape(bs, -1, 2)
            gt_weights = pos_kpt_weights.reshape(bs, -1, 2)
            pos_kpt_sigmas = pos_kpt_sigmas.reshape(bs, -1, 2)
            bar_mu = (pos_kpt_preds - all_gts) / pos_kpt_sigmas
            log_phi = self.dec_flow.log_prob(bar_mu.reshape(-1, 2)).reshape(bs, -1, 1)
            nf_loss = torch.log(pos_kpt_sigmas) - log_phi 
            output = EasyDict(
                pred_jts=pos_kpt_preds,
                sigma=pos_kpt_sigmas,
                nf_loss=nf_loss
            )
            labels = EasyDict(
                target_uv=all_gts,
                target_uv_weight=gt_weights
            )
            loss_kpt = self.loss_kpt(output, labels)
        # keypoint oks loss
        pos_inds = kpt_weights.sum(-1) > 0
        factors = factors[pos_inds][:, :2].repeat(1, kpt_preds.shape[-1] // 2)
        pos_kpt_preds = kpt_preds[pos_inds] * factors
        pos_kpt_targets = kpt_targets[pos_inds] * factors
        pos_areas = area_targets[pos_inds]
        pos_valid = kpt_weights[pos_inds, 0::2]
        if len(pos_areas) == 0:
            loss_oks = pos_kpt_preds.sum() * 0
        else:
            assert (pos_areas > 0).all()
            loss_oks = self.loss_oks(
                pos_kpt_preds,
                pos_kpt_targets,
                pos_valid,
                pos_areas,
                avg_factor=num_total_pos)

        return loss_cls, loss_kpt, loss_oks, kpt_preds, kpt_targets, \
            area_targets, kpt_weights

    def get_targets(self,
                    cls_scores_list,
                    kpt_preds_list,
                    gt_labels_list,
                    gt_keypoints_list,
                    gt_areas_list,
                    img_metas):
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            kpt_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (x_{i}, y_{i}) and shape [num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3).
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all
                    images.
                - kpt_targets_list (list[Tensor]): Keypoint targets for all
                    images.
                - kpt_weights_list (list[Tensor]): Keypoint weights for all
                    images.
                - area_targets_list (list[Tensor]): area targets for all
                    images.
                - num_total_pos (int): Number of positive samples in all
                    images.
                - num_total_neg (int): Number of negative samples in all
                    images.
        """
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         area_targets_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, kpt_preds_list, 
             gt_labels_list, gt_keypoints_list, gt_areas_list, img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, kpt_targets_list,
                kpt_weights_list, area_targets_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           kpt_pred,
                           gt_labels,
                           gt_keypoints,
                           gt_areas,
                           img_meta):
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            kpt_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (x_{i}, y_{i}) and
                shape [num_query, K*2].
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_keypoints (Tensor): Ground truth keypoints for one image with
                shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v, ..., \
                    p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas (Tensor): Ground truth mask areas for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor): Label weights of each image.
                - kpt_targets (Tensor): Keypoint targets of each image.
                - kpt_weights (Tensor): Keypoint weights of each image.
                - area_targets (Tensor): Area targets of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = kpt_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(cls_score, kpt_pred, gt_labels,
                                             gt_keypoints, gt_areas, img_meta)
        sampling_result = self.sampler.sample(assign_result, kpt_pred,
                                              gt_keypoints)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_labels.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(num_bboxes)

        img_h, img_w, _ = img_meta['img_shape']

        # keypoint targets
        kpt_targets = torch.zeros_like(kpt_pred)
        kpt_weights = torch.zeros_like(kpt_pred)
        pos_gt_kpts = gt_keypoints[sampling_result.pos_assigned_gt_inds]
        pos_gt_kpts = pos_gt_kpts.reshape(pos_gt_kpts.shape[0],
                                          pos_gt_kpts.shape[-1] // 3, 3)
        valid_idx = pos_gt_kpts[:, :, 2] > 0
        pos_kpt_weights = kpt_weights[pos_inds].reshape(
            pos_gt_kpts.shape[0], kpt_weights.shape[-1] // 2, 2)
        pos_kpt_weights[valid_idx] = 1.0
        kpt_weights[pos_inds] = pos_kpt_weights.reshape(
            pos_kpt_weights.shape[0], kpt_pred.shape[-1])

        factor = kpt_pred.new_tensor([img_w, img_h]).unsqueeze(0)
        pos_gt_kpts_normalized = pos_gt_kpts[..., :2]
        pos_gt_kpts_normalized[..., 0] = pos_gt_kpts_normalized[..., 0] / \
            factor[:, 0:1]
        pos_gt_kpts_normalized[..., 1] = pos_gt_kpts_normalized[..., 1] / \
            factor[:, 1:2]
        kpt_targets[pos_inds] = pos_gt_kpts_normalized.reshape(
            pos_gt_kpts.shape[0], kpt_pred.shape[-1])

        area_targets = kpt_pred.new_zeros(
            kpt_pred.shape[0])  # get areas for calculating oks
        pos_gt_areas = gt_areas[sampling_result.pos_assigned_gt_inds]
        area_targets[pos_inds] = pos_gt_areas

        return (labels, label_weights, kpt_targets, kpt_weights,
                area_targets, pos_inds, neg_inds)

    def loss_single_rpn(self,
                        cls_scores,
                        kpt_preds,
                        gt_labels_list,
                        gt_keypoints_list,
                        gt_areas_list,
                        img_metas):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i}) and
                shape [bs, num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, kpt_preds_list,
                                           gt_labels_list, gt_keypoints_list,
                                           gt_areas_list, img_metas)
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         area_targets_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        kpt_targets = torch.cat(kpt_targets_list, 0)
        kpt_weights = torch.cat(kpt_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt keypoints accross all gpus, for
        # normalization purposes
        # num_total_pos = loss_cls.new_tensor([num_total_pos])
        # num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape(-1, kpt_preds.shape[-1])
        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        loss_kpt = self.loss_kpt_rpn(
            kpt_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)

        return loss_cls, loss_kpt

    @force_fp32(apply_to=('all_cls_scores', 'all_kpt_preds'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_kpt_preds,
                   all_sigma_preds,
                   enc_cls_scores,
                   enc_kpt_preds,
                   hm_proto,
                   memory,
                   mlvl_masks,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_kpt_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (x_{i}, y_{i}) and shape
                [nb_dec, bs, num_query, K*2].
            enc_cls_scores (Tensor): Classification scores of points on
                encode feature map, has shape (N, h*w, num_classes).
                Only be passed when as_two_stage is True, otherwise is None.
            enc_kpt_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, K*2). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Defalut False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 3-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box. The third item is an (n, K, 3) tensor
                with [p^{1}_x, p^{1}_y, p^{1}_v, ..., p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        cls_scores = all_cls_scores[-1]
        kpt_preds = all_kpt_preds[-1]
        sigma_preds = all_sigma_preds[-1]
        # cls_scores = enc_cls_scores
        # kpt_preds = enc_kpt_preds

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            kpt_pred = kpt_preds[img_id]
            sigma_pred = sigma_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            # TODO: only support single image test
            # memory_i = memory[:, img_id, :]
            # mlvl_mask = mlvl_masks[img_id]
            proposals = self._get_bboxes_single(cls_score, kpt_pred, sigma_pred,
                                                img_shape, scale_factor,
                                                memory, mlvl_masks, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           kpt_pred,
                           sigma_pred,
                           img_shape,
                           scale_factor,
                           memory,
                           mlvl_masks,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            kpt_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (x_{i}, y_{i}) and
                shape [num_query, K*2].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5],
                    where the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with
                    shape [num_query].
                - det_kpts: Predicted keypoints with shape [num_query, K, 3].
        """
        OKS_SCORE = 0.5
        OKS_SIGMAS = [.26, .79, .79, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]
        assert len(cls_score) == len(kpt_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexs = cls_score.view(-1).topk(max_per_img)
            det_labels = indexs % self.num_classes
            bbox_index = indexs // self.num_classes
            kpt_pred = kpt_pred[bbox_index]
            sigma_pred = sigma_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            kpt_pred = kpt_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        # ----- results after pose decoder -----
        # det_kpts = kpt_pred.reshape(kpt_pred.size(0), -1, 2)

        # ----- results after joint decoder (default) -----
        # import time
        # start = time.time()
        # refine_targets = (kpt_pred, None, None, torch.ones_like(kpt_pred))
        # refine_outputs = self.forward_refine(memory, mlvl_masks,
        #                                      refine_targets, None, None)
        # end = time.time()
        # print(f'refine time: {end - start:.6f}')
        
        # refine_outputs: 包含2层decoder-layer关节点的输出结果以及每个关节点的得分score,这里只使用最后一层的结果
        det_kpts = kpt_pred.reshape(max_per_img, self.num_keypoints, -1)
        det_kpts_sigmas = sigma_pred.reshape(det_kpts.shape) # shape: 100, 15, 2
        det_kpts_score = 1 - det_kpts_sigmas
        det_kpts_score = torch.mean(det_kpts_score, dim=-1, keepdim=True) # shape： 100, 15, 1

        det_kpts[..., 0] = det_kpts[..., 0] * img_shape[1]
        det_kpts[..., 1] = det_kpts[..., 1] * img_shape[0]
        det_kpts[..., 0].clamp_(min=0, max=img_shape[1])
        det_kpts[..., 1].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_kpts /= det_kpts.new_tensor(
                scale_factor[:2]).unsqueeze(0).unsqueeze(0)

        # use circumscribed rectangle box of keypoints as det bboxes
        x1 = det_kpts[..., 0].min(dim=1, keepdim=True)[0]
        y1 = det_kpts[..., 1].min(dim=1, keepdim=True)[0]
        x2 = det_kpts[..., 0].max(dim=1, keepdim=True)[0]
        y2 = det_kpts[..., 1].max(dim=1, keepdim=True)[0]
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=1)
        
        # poseur 调整了det_kpts_score  --------------------------------------------------------------------------------------------------------------------------------------
        output_regression_sigma = det_kpts_sigmas
        output_regression = det_kpts
        
        output_regression_p = self.get_p(output_regression_sigma)
        p_to_coord_index = 5
        output_regression = (output_regression * output_regression_p**p_to_coord_index) / (output_regression_p**p_to_coord_index + 1e-10)

        output_regression_score = output_regression_p
    
        det_kpts_score = output_regression_score
        det_kpts = output_regression
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # follow rle and poseur
        # scores = scores + torch.mean(det_kpts_score, dim=1).squeeze(-1) + torch.max(det_kpts_score, dim=1)[0].squeeze(-1)
        # det_bboxes_scores = scores * torch.mean(det_kpts_score, dim=1).squeeze(-1)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)
        
        # 关节点得分为pose置信度 * 预测关节点置信度
        kpt_scores = scores[:, None, None] * det_kpts_score
        # kpt_scores = det_kpts_score

        det_kpts = torch.cat(
            (det_kpts, kpt_scores), dim=2)
        
        # nms
        keep, _ = oks_nms(det_kpts, scores, OKS_SCORE, np.array(OKS_SIGMAS) / 10.0)
        tmp_det_kpts = []
        tmp_det_labels = []
        tmp_det_bboxes = []
        for keep_ in keep:
            tmp_det_kpts.append(det_kpts[keep_])
            tmp_det_bboxes.append(det_bboxes[keep_])
            tmp_det_labels.append(det_labels[keep_])
        det_bboxes = torch.stack(tmp_det_bboxes, dim=0)
        det_labels = torch.stack(tmp_det_labels, dim=0)
        det_kpts = torch.stack(tmp_det_kpts, dim=0)

        return det_bboxes, det_labels, det_kpts
    
    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The third item is ``kpts`` with shape
                (n, K, 3), in [p^{1}_x, p^{1}_y, p^{1}_v, p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list
    
    def get_p(self, output_regression_sigma, p_x = 0.2):
        output_regression_p = (1- torch.exp(-(p_x/output_regression_sigma)))
        output_regression_p = output_regression_p[:,:,0] * output_regression_p[:,:,1]
        output_regression_p = output_regression_p[:,:,None]
        return output_regression_p * 0.7


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)

        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        return self.log_prob(x)

def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())


def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))



class Linear_with_norm(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear_with_norm, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y
    
def oks_nms(poses, scores, thresh, sigmas=None, in_vis_thre=None):
    poses = poses.cpu().numpy()
    scores = scores.cpu().numpy()
    if len(poses) == 0: return []
    areas = (np.max(poses[:, :, 0], axis=1) - np.min(poses[:, :, 0], axis=1)) * \
            (np.max(poses[:, :, 1], axis=1) - np.min(poses[:, :, 1], axis=1))
    poses = poses.reshape(poses.shape[0], -1)

    order = scores.argsort()[::-1]

    keep = []
    keep_ind = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        oks_ovr = oks_iou(poses[i], poses[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)
        inds = np.where(oks_ovr <= thresh)[0]
        nms_inds = np.where(oks_ovr > thresh)[0]
        nms_inds = order[nms_inds + 1]
        keep_ind.append(nms_inds.tolist())
        order = order[inds + 1]

    return keep, keep_ind

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg >= in_vis_thre) and list(vd >= in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

