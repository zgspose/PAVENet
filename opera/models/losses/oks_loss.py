# Copyright (c) Hikvision Research Institute. All rights reserved.
import numpy as np
import mmcv
import torch
import torch.nn as nn
import math
from mmdet.models.losses.utils import weighted_loss

from ..builder import LOSSES


def oks_overlaps(kpt_preds, kpt_gts, kpt_valids, kpt_areas, sigmas):
    sigmas = kpt_preds.new_tensor(sigmas)
    variances = (sigmas * 2)**2

    assert kpt_preds.size(0) == kpt_gts.size(0)
    kpt_preds = kpt_preds.reshape(-1, kpt_preds.size(-1) // 2, 2)
    kpt_gts = kpt_gts.reshape(-1, kpt_gts.size(-1) // 2, 2)

    squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + \
        (kpt_preds[:, :, 1] - kpt_gts[:, :, 1]) ** 2
    assert (kpt_valids.sum(-1) > 0).all()
    squared_distance0 = squared_distance / (
        kpt_areas[:, None] * variances[None, :] * 2)
    squared_distance1 = torch.exp(-squared_distance0)
    squared_distance1 = squared_distance1 * kpt_valids
    oks = squared_distance1.sum(dim=1) / kpt_valids.sum(dim=1)

    return oks


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def oks_loss(pred,
             target,
             valid=None,
             area=None,
             linear=False,
             sigmas=None,
             eps=1e-6):
    """Oks loss.

    Computing the oks loss between a set of predicted poses and target poses.
    The loss is calculated as negative log of oks.

    Args:
        pred (torch.Tensor): Predicted poses of format (x1, y1, x2, y2, ...),
            shape (n, K*2).
        target (torch.Tensor): Corresponding gt poses, shape (n, K*2).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Returns:
        torch.Tensor: Loss tensor.
    """
    oks = oks_overlaps(pred, target, valid, area, sigmas).clamp(min=eps)
    if linear:
        loss = 1 - oks
    else:
        loss = -oks.log()
    return loss


@LOSSES.register_module()
class OKSLoss(nn.Module):
    """OKSLoss.

    Computing the oks loss between a set of predicted poses and target poses.

    Args:
        linear (bool): If True, use linear scale of loss instead of log scale.
            Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 linear=False,
                 num_keypoints=17,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(OKSLoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        if num_keypoints == 17:
            self.sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=np.float32) / 10.0
        elif num_keypoints == 15:
            self.sigmas = np.array([
                .26, .25, .25, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=np.float32) / 10.0
        elif num_keypoints == 14:
            self.sigmas = np.array([
                .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
                .79, .79
            ]) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')

    def forward(self,
                pred,
                target,
                valid,
                area,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            valid (torch.Tensor): The visible flag of the target pose.
            area (torch.Tensor): The area of the target pose.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * oks_loss(
            pred,
            target,
            weight,
            valid=valid,
            area=area,
            linear=self.linear,
            sigmas=self.sigmas,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class RLELoss(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, loss_weight=1.0, joints_weight=None, OUTPUT_3D=False, size_average=True):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)
        self.loss_weight = loss_weight
        self.joints_weight = joints_weight

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels, num_valid):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = labels.target_uv.reshape(pred_jts.shape)
        gt_uv_weight = labels.target_uv_weight.reshape(pred_jts.shape)

        nf_loss = nf_loss * gt_uv_weight[:, :, :1] 

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            # return loss.sum() / (loss.shape[0] * loss.shape[1])
            # return loss.sum() / len(loss)
            return (loss.sum() / num_valid) * self.loss_weight
        else:
            return loss.sum()