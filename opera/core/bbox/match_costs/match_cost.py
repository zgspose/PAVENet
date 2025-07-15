# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import numpy as np

from .builder import MATCH_COST
import math

from easydict import EasyDict


@MATCH_COST.register_module()
class KptL1Cost(object):
    """KptL1Cost.

    Args:
        weight (int | float, optional): loss_weight.

    Examples:
        >>> from opera.core.bbox.match_costs.match_cost import KptL1Cost
        >>> import torch
        >>> self = KptL1Cost()
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with normalized coordinates
                (x_{i}, y_{i}), which are all in range [0, 1]. Shape
                [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with normalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].

        Returns:
            torch.Tensor: kpt_cost value with weight.
        """
        kpt_cost = []
        for i in range(len(gt_keypoints)):
            kpt_pred_tmp = kpt_pred.clone()
            valid_flag = valid_kpt_flag[i] > 0
            valid_flag_expand = valid_flag.unsqueeze(0).unsqueeze(
                -1).expand_as(kpt_pred_tmp)
            kpt_pred_tmp[~valid_flag_expand] = 0
            cost = torch.cdist(
                kpt_pred_tmp.reshape(kpt_pred_tmp.shape[0], -1),
                gt_keypoints[i].reshape(-1).unsqueeze(0),
                p=1)
            avg_factor = torch.clamp(valid_flag.float().sum() * 2, 1.0)
            cost = cost / avg_factor
            kpt_cost.append(cost)
        kpt_cost = torch.cat(kpt_cost, dim=1)
        return kpt_cost * self.weight


@MATCH_COST.register_module()
class RLECost(object):
    """RLECost.

    Args:
        weight (int | float, optional): loss_weight.

    Examples:
        >>> from opera.core.bbox.match_costs.match_cost import KptL1Cost
        >>> import torch
        >>> self = KptL1Cost()
    """

    def __init__(self, weight=1.0, size_average=True):
        self.weight = weight
        self.amp = 1 / math.sqrt(2 * math.pi)
        self.size_average = size_average
        self.num_joints = 15
    
    def get_nf_loss(self, pred_jts, sigma, target, flow_model):
        num_joints = self.num_joints
        BATCH_SIZE = sigma.size(0) # num_query 300
        gt_uv = target.reshape(pred_jts.shape)
        bar_mu = (pred_jts - gt_uv) / sigma
        # (B, K, 1)
        with torch.no_grad():
            log_phi = flow_model.log_prob(bar_mu.reshape(-1, 2)).reshape(BATCH_SIZE, num_joints, 1)
        nf_loss = torch.log(sigma) - log_phi
        return nf_loss # shape: 300, 15, 2
    
    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)
    
    def get_rle_loss(self, output, labels):
        # shape: 300, 15, 2
        nf_loss = output.nf_loss
        # shape: 300, 15, 2
        pred_jts = output.pred_jts
        # shape: 300, 15, 2
        sigma = output.sigma
        # gt
        gt_uv = labels.target_uv.reshape(pred_jts.shape)
        # 关节点是否可见
        gt_uv_weight = labels.target_uv_weight.reshape(pred_jts.shape)
        nf_loss = nf_loss * gt_uv_weight[:, :, :1]
        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob
            
        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum(dim=(1, 2), keepdim=True).squeeze(dim=-1) / loss.shape[1]  
        
        return loss

    def __call__(self, kpt_pred, sigma_pred, gt_keypoints, valid_kpt_flag, flow_model):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with normalized coordinates
                (x_{i}, y_{i}), which are all in range [0, 1]. Shape
                [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with normalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].

        Returns:
            torch.Tensor: kpt_cost value with weight.
        """
        # 遍历每一个GT，计算300个预测值与GT的rle-loss
        kpt_cost = []
        output = EasyDict(
            pred_jts=None,
            sigma=None,
            nf_loss=None)
        labels = EasyDict(
            target_uv=None,
            target_uv_weight=None,
        )
        
        for i in range(len(gt_keypoints)):
            # 克隆一份，防止影响原数据
            num_query, num_joints, _ = kpt_pred.shape
            kpt_pred_tmp = kpt_pred.clone() # shape: 300, 15, 2
            sigma_pred_tmp = sigma_pred.clone().reshape(num_query, num_joints, -1) # shape: 300, 15, 2
            valid_flag = valid_kpt_flag[i] > 0 # shape: 15
            valid_flag_expand = valid_flag.unsqueeze(0).unsqueeze(
                -1).expand_as(kpt_pred_tmp)
            # 去除不可见的关节点的预测值
            # kpt_pred_tmp[~valid_flag_expand] = 0
            # sigma_pred_tmp[~valid_flag_expand] = 0
            # 处理数据，gt复制300份
            output.pred_jts = kpt_pred_tmp
            # shape: num_query, num_joints, 2
            gt = gt_keypoints[i:i+1].repeat(num_query, 1, 1)
            output.nf_loss = self.get_nf_loss(kpt_pred_tmp, sigma_pred_tmp, gt, flow_model)
            output.sigma = sigma_pred_tmp
            labels.target_uv = gt
            labels.target_uv_weight=valid_flag_expand
            # 计算
            cost = self.get_rle_loss(output, labels)
            
            avg_factor = torch.clamp(valid_flag.float().sum() * 2, 1.0)
            # 归一化
            cost = cost / avg_factor
            kpt_cost.append(cost)
        kpt_cost = torch.cat(kpt_cost, dim=1)
        return kpt_cost * self.weight


@MATCH_COST.register_module()
class OksCost(object):
    """OksCost.

    Args:
        weight (int | float, optional): loss_weight.

    Examples:
        >>> from opera.core.bbox.match_costs.match_cost import OksCost
        >>> import torch
        >>> self = OksCost()
    """

    def __init__(self, num_keypoints=17, weight=1.0):
        self.weight = weight
        if num_keypoints == 17:
            self.sigmas = np.array([
                .26,
                .25, .25,
                .35, .35,
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89], dtype=np.float32) / 10.0
        # 添加，适用于posetrack 15个关节点
        elif num_keypoints == 15:
            self.sigmas = np.array([
                .26,
                .79, .79,
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89], dtype=np.float32) / 10.0
            
        elif num_keypoints == 14:
            self.sigmas = np.array([
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89,
                .79, .79], dtype=np.float32) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag, gt_areas):
        """
        Args:
            kpt_pred (Tensor): Predicted keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_query, K, 2].
            gt_keypoints (Tensor): Ground truth keypoints with unnormalized
                coordinates (x_{i}, y_{i}). Shape [num_gt, K, 2].
            valid_kpt_flag (Tensor): valid flag of ground truth keypoints.
                Shape [num_gt, K].
            gt_areas (Tensor): Ground truth mask areas. Shape [num_gt,].

        Returns:
            torch.Tensor: oks_cost value with weight.
        """
        sigmas = torch.from_numpy(self.sigmas).to(kpt_pred.device)
        variances = (sigmas * 2)**2

        oks_cost = []
        assert len(gt_keypoints) == len(gt_areas)
        for i in range(len(gt_keypoints)):
            squared_distance = \
                (kpt_pred[:, :, 0] - gt_keypoints[i, :, 0].unsqueeze(0)) ** 2 + \
                (kpt_pred[:, :, 1] - gt_keypoints[i, :, 1].unsqueeze(0)) ** 2
            vis_flag = (valid_kpt_flag[i] > 0).int()
            vis_ind = vis_flag.nonzero(as_tuple=False)[:, 0]
            num_vis_kpt = vis_ind.shape[0]
            assert num_vis_kpt > 0
            area = gt_areas[i]

            squared_distance0 = squared_distance / (area * variances * 2)
            squared_distance0 = squared_distance0[:, vis_ind]
            squared_distance1 = torch.exp(-squared_distance0).sum(
                dim=1, keepdim=True)
            oks = squared_distance1 / num_vis_kpt
            # The 1 is a constant that doesn't change the matching, so omitted.
            oks_cost.append(-oks)
        oks_cost = torch.cat(oks_cost, dim=1)
        return oks_cost * self.weight
