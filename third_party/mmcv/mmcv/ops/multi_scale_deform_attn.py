# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable

from mmcv import deprecated_api_warning
from mmcv.cnn import constant_init, xavier_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner import BaseModule
from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class MultiScaleDeformableAttnFunction(Function):

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index,\
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


def multi_scale_deformable_attn_pytorchV1(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """
    # 多尺度特征tokens
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # bs*num_heads, num_levels, embed_dims, num_querys, num_points
    sampling_value = torch.stack(sampling_value_list, dim=1).reshape(bs, num_heads, num_levels, embed_dims, num_queries, num_points).permute(0, 4, 1, 2, 5, 3)

    return sampling_value


@ATTENTION.register_module()
class MultiScaleDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (torch.Tensor): The positional encoding for `key`. Default
                None.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

# 添加时间 2024-10-25 适用于VideoPoseHeadV18
@ATTENTION.register_module()
class MultiScaleDeformableAttentionV18(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                query_time_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (torch.Tensor): The positional encoding for `key`. Default
                None.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos + query_time_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

# TODO 适用于时空解码器 修改时间 2024-10-9
@ATTENTION.register_module()
class MulFramesMultiScaleDeformableAttentionV1(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        # 前一帧
        self.pre_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.pre_attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        # 当前帧
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        # 后一帧
        self.next_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.next_attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        
        
        self.pre_value_proj = nn.Linear(embed_dims, embed_dims)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.next_value_proj = nn.Linear(embed_dims, embed_dims)
        
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.pre_sampling_offsets, 0.)
        constant_init(self.sampling_offsets, 0.)
        constant_init(self.next_sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.pre_sampling_offsets.bias.data = grid_init.view(-1)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        self.next_sampling_offsets.bias.data = grid_init.view(-1)
        
        constant_init(self.pre_attention_weights, val=0., bias=0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.next_attention_weights, val=0., bias=0.)
        
        xavier_init(self.pre_value_proj, distribution='uniform', bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.next_value_proj, distribution='uniform', bias=0.)
        
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MulFramesMultiScaleDeformableAttentionV1')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (torch.Tensor): The positional encoding for `key`. Default
                None.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        
        # TODO 切分为三份
        bs = bs // 3
        # 提取不同帧的特征
        pre_value = value[0::3]
        now_value = value[1::3]
        next_value = value[2::3]
        pre_value = self.pre_value_proj(pre_value)
        now_value = self.value_proj(now_value)
        next_value = self.next_value_proj(next_value)
        
        pre_value = pre_value.view(bs, num_value, self.num_heads, -1)
        now_value = now_value.view(bs, num_value, self.num_heads, -1)
        next_value = next_value.view(bs, num_value, self.num_heads, -1)
        
            
        pre_sampling_offsets = self.pre_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        now_sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        next_sampling_offsets = self.next_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        pre_attention_weights = self.pre_attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        now_attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        next_attention_weights = self.next_attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        
        # BUG ------可能会出现错误
        pre_attention_weights_sum = torch.exp(pre_attention_weights).sum(-1, keepdim=True)
        now_attention_weights_sum = torch.exp(now_attention_weights).sum(-1, keepdim=True)
        next_attention_weights_sum = torch.exp(next_attention_weights).sum(-1, keepdim=True)
        sum_all = pre_attention_weights_sum + now_attention_weights_sum + next_attention_weights_sum
        # BUG
        
        pre_attention_weights = pre_attention_weights.softmax(-1)
        now_attention_weights = now_attention_weights.softmax(-1)
        next_attention_weights = next_attention_weights.softmax(-1)
        

        pre_attention_weights = pre_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        now_attention_weights = now_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        next_attention_weights = next_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            
            pre_sampling_locations = reference_points[:, :, None, :, None, :] \
                + pre_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            
            now_sampling_locations = reference_points[:, :, None, :, None, :] \
                + now_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            
            next_sampling_locations = reference_points[:, :, None, :, None, :] \
                + next_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
                
        elif reference_points.shape[-1] == 4:
            pre_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + pre_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
                
            now_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + now_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
                
            next_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + next_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
            
        if torch.cuda.is_available() and value.is_cuda:
            pre_output = MultiScaleDeformableAttnFunction.apply(
                pre_value, spatial_shapes, level_start_index, pre_sampling_locations,
                pre_attention_weights, self.im2col_step)
            now_output = MultiScaleDeformableAttnFunction.apply(
                now_value, spatial_shapes, level_start_index, now_sampling_locations,
                now_attention_weights, self.im2col_step)
            next_output = MultiScaleDeformableAttnFunction.apply(
                next_value, spatial_shapes, level_start_index, next_sampling_locations,
                next_attention_weights, self.im2col_step)
        else:
            pre_output = multi_scale_deformable_attn_pytorch(
                pre_value, spatial_shapes, level_start_index, pre_sampling_locations,
                pre_attention_weights, self.im2col_step)
            now_output = multi_scale_deformable_attn_pytorch(
                now_value, spatial_shapes, level_start_index, now_sampling_locations,
                now_attention_weights, self.im2col_step)
            next_output = multi_scale_deformable_attn_pytorch(
                next_value, spatial_shapes, level_start_index, next_sampling_locations,
                next_attention_weights, self.im2col_step)
            
        # TODO start: 融合多帧不同参考点对应的feature_token
        pre_output = pre_output.reshape(bs, num_query, self.num_heads, -1)
        now_output = now_output.reshape(bs, num_query, self.num_heads, -1)
        next_output = next_output.reshape(bs, num_query, self.num_heads, -1)
        output = pre_output * (pre_attention_weights_sum / sum_all) + now_output * (now_attention_weights_sum / sum_all) + next_output * (next_attention_weights_sum / sum_all)
        output = output.flatten(-2, -1)
        # TODO end

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

# 适用于joint-时空解码器 修改时间 2024-10-26 ---- 不同帧共享同一个task_quey, 使用不同的帧的全连接回归相对偏移，然后融合不同帧的特征信息
@ATTENTION.register_module()
class MulFramesMultiScaleDeformableAttentionV17(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        # 前一帧
        self.pre_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.pre_attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        # 当前帧
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        # 后一帧
        self.next_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.next_attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        
        
        self.pre_value_proj = nn.Linear(embed_dims, embed_dims)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.next_value_proj = nn.Linear(embed_dims, embed_dims)
        
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.pre_sampling_offsets, 0.)
        constant_init(self.sampling_offsets, 0.)
        constant_init(self.next_sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.pre_sampling_offsets.bias.data = grid_init.view(-1)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        self.next_sampling_offsets.bias.data = grid_init.view(-1)
        
        constant_init(self.pre_attention_weights, val=0., bias=0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.next_attention_weights, val=0., bias=0.)
        
        xavier_init(self.pre_value_proj, distribution='uniform', bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.next_value_proj, distribution='uniform', bias=0.)
        
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MulFramesMultiScaleDeformableAttentionV17')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                query_time_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (torch.Tensor): The positional encoding for `key`. Default
                None.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2, 3)

        # num_gts, num_querys, embed_dims
        bs, num_query, _ = query.shape
        bs, num_value, num_frames, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None, None], 0.0)
        
        # TODO 切分为三份
        # bs = bs // 3
        # 提取不同帧的feature_tokens
        pre_value = value[:, :, 0]
        now_value = value[:, :, 1]
        next_value = value[:, :, 2]
        # num_gts, num_tokens, embed_dims
        pre_value = self.pre_value_proj(pre_value)
        now_value = self.value_proj(now_value)
        next_value = self.next_value_proj(next_value)
        
        pre_value = pre_value.view(bs, num_value, self.num_heads, -1)
        now_value = now_value.view(bs, num_value, self.num_heads, -1)
        next_value = next_value.view(bs, num_value, self.num_heads, -1)
        
        pre_sampling_offsets = self.pre_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        now_sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        next_sampling_offsets = self.next_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        pre_attention_weights = self.pre_attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        now_attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        next_attention_weights = self.next_attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        
        # BUG ------可能会出现错误
        pre_attention_weights_sum = torch.exp(pre_attention_weights).sum(-1, keepdim=True)
        now_attention_weights_sum = torch.exp(now_attention_weights).sum(-1, keepdim=True)
        next_attention_weights_sum = torch.exp(next_attention_weights).sum(-1, keepdim=True)
        sum_all = pre_attention_weights_sum + now_attention_weights_sum + next_attention_weights_sum
        # BUG
        
        pre_attention_weights = pre_attention_weights.softmax(-1)
        now_attention_weights = now_attention_weights.softmax(-1)
        next_attention_weights = next_attention_weights.softmax(-1)
        

        pre_attention_weights = pre_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        now_attention_weights = now_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        next_attention_weights = next_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            
            pre_sampling_locations = reference_points[:, :, None, :, None, :] \
                + pre_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            
            now_sampling_locations = reference_points[:, :, None, :, None, :] \
                + now_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            
            next_sampling_locations = reference_points[:, :, None, :, None, :] \
                + next_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
                
        elif reference_points.shape[-1] == 4:
            pre_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + pre_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
                
            now_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + now_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
                
            next_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + next_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
            
        if torch.cuda.is_available() and value.is_cuda:
            pre_output = MultiScaleDeformableAttnFunction.apply(
                pre_value, spatial_shapes, level_start_index, pre_sampling_locations,
                pre_attention_weights, self.im2col_step)
            now_output = MultiScaleDeformableAttnFunction.apply(
                now_value, spatial_shapes, level_start_index, now_sampling_locations,
                now_attention_weights, self.im2col_step)
            next_output = MultiScaleDeformableAttnFunction.apply(
                next_value, spatial_shapes, level_start_index, next_sampling_locations,
                next_attention_weights, self.im2col_step)
        else:
            pre_output = multi_scale_deformable_attn_pytorch(
                pre_value, spatial_shapes, level_start_index, pre_sampling_locations,
                pre_attention_weights, self.im2col_step)
            now_output = multi_scale_deformable_attn_pytorch(
                now_value, spatial_shapes, level_start_index, now_sampling_locations,
                now_attention_weights, self.im2col_step)
            next_output = multi_scale_deformable_attn_pytorch(
                next_value, spatial_shapes, level_start_index, next_sampling_locations,
                next_attention_weights, self.im2col_step)
            
        # TODO start: 融合多帧不同参考点对应的feature_token
        pre_output = pre_output.reshape(bs, num_query, self.num_heads, -1)
        now_output = now_output.reshape(bs, num_query, self.num_heads, -1)
        next_output = next_output.reshape(bs, num_query, self.num_heads, -1)
        output = pre_output * (pre_attention_weights_sum / sum_all) + now_output * (now_attention_weights_sum / sum_all) + next_output * (next_attention_weights_sum / sum_all)
        output = output.flatten(-2, -1)
        # TODO end

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@ATTENTION.register_module()
class MulFramesMultiScaleDeformableAttentionNumFrames3(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 num_frames=3,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_frames = num_frames
        # 前一帧
        self.pre_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.pre_attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        # 当前帧
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        # 后一帧
        self.next_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.next_attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.pre_sampling_offsets, 0.)
        constant_init(self.sampling_offsets, 0.)
        constant_init(self.next_sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.pre_sampling_offsets.bias.data = grid_init.view(-1)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        self.next_sampling_offsets.bias.data = grid_init.view(-1)
        
        constant_init(self.pre_attention_weights, val=0., bias=0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.next_attention_weights, val=0., bias=0.)
        
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MulFramesMultiScaleDeformableAttentionV17')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                query_time_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (torch.Tensor): The positional encoding for `key`. Default
                None.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2, 3)

        # num_gts, num_querys, embed_dims
        bs, num_query, _ = query.shape
        bs, num_value, num_frames, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask.transpose(1, 2)[..., None], 0.0)
        
        # TODO 切分为三份
        value = self.value_proj(value)
        # 提取不同帧的feature_tokens
        pre_value = value[:, :, 0]
        now_value = value[:, :, 1]
        next_value = value[:, :, 2]
        # num_gts, num_tokens, embed_dims
        
        pre_value = pre_value.view(bs, num_value, self.num_heads, -1).contiguous()
        now_value = now_value.view(bs, num_value, self.num_heads, -1).contiguous()
        next_value = next_value.view(bs, num_value, self.num_heads, -1).contiguous()
        
        pre_sampling_offsets = self.pre_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        now_sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        next_sampling_offsets = self.next_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        pre_attention_weights = self.pre_attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        now_attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        next_attention_weights = self.next_attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        
        # BUG ------可能会出现错误
        pre_attention_weights_sum = torch.exp(pre_attention_weights).sum(-1, keepdim=True)
        now_attention_weights_sum = torch.exp(now_attention_weights).sum(-1, keepdim=True)
        next_attention_weights_sum = torch.exp(next_attention_weights).sum(-1, keepdim=True)
        sum_all = pre_attention_weights_sum + now_attention_weights_sum + next_attention_weights_sum
        # BUG
        
        pre_attention_weights = pre_attention_weights.softmax(-1)
        now_attention_weights = now_attention_weights.softmax(-1)
        next_attention_weights = next_attention_weights.softmax(-1)
        

        pre_attention_weights = pre_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        now_attention_weights = now_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        next_attention_weights = next_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            # 获取不同辅助帧的初始参考点
            pre_reference_points = reference_points[:bs]
            now_reference_points = reference_points[bs:bs*2]
            next_reference_points = reference_points[bs*2:]
            
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            
            pre_sampling_locations = pre_reference_points[:, :, None, :, None, :] \
                + pre_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            
            now_sampling_locations = now_reference_points[:, :, None, :, None, :] \
                + now_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            
            next_sampling_locations = next_reference_points[:, :, None, :, None, :] \
                + next_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
                
        elif reference_points.shape[-1] == 4:
            pre_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + pre_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
                
            now_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + now_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
                
            next_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + next_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
            
        if torch.cuda.is_available() and value.is_cuda:
            pre_output = MultiScaleDeformableAttnFunction.apply(
                pre_value, spatial_shapes, level_start_index, pre_sampling_locations,
                pre_attention_weights, self.im2col_step)
            now_output = MultiScaleDeformableAttnFunction.apply(
                now_value, spatial_shapes, level_start_index, now_sampling_locations,
                now_attention_weights, self.im2col_step)
            next_output = MultiScaleDeformableAttnFunction.apply(
                next_value, spatial_shapes, level_start_index, next_sampling_locations,
                next_attention_weights, self.im2col_step)
        else:
            pre_output = multi_scale_deformable_attn_pytorch(
                pre_value, spatial_shapes, level_start_index, pre_sampling_locations,
                pre_attention_weights, self.im2col_step)
            now_output = multi_scale_deformable_attn_pytorch(
                now_value, spatial_shapes, level_start_index, now_sampling_locations,
                now_attention_weights, self.im2col_step)
            next_output = multi_scale_deformable_attn_pytorch(
                next_value, spatial_shapes, level_start_index, next_sampling_locations,
                next_attention_weights, self.im2col_step)
            
        # TODO start: 融合多帧不同参考点对应的feature_token
        pre_output = pre_output.reshape(bs, num_query, self.num_heads, -1)
        now_output = now_output.reshape(bs, num_query, self.num_heads, -1)
        next_output = next_output.reshape(bs, num_query, self.num_heads, -1)
        output = pre_output * (pre_attention_weights_sum / sum_all) + \
                 now_output * (now_attention_weights_sum / sum_all) + \
                 next_output * (next_attention_weights_sum / sum_all)
        output = output.flatten(-2, -1)
        # TODO end

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

# 适用于joint-时空解码器 修改时间 2024-10-26 ---- 不同帧共享同一个task_quey, 使用不同的帧的全连接回归相对偏移，然后融合不同帧的特征信息 5帧
@ATTENTION.register_module()
class MulFramesMultiScaleDeformableAttentionNumFrames5(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in  
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        
        # 前前帧
        self.pre_pre_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.pre_pre_attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        # 前一帧
        self.pre_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.pre_attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        # 当前帧
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        # 后一帧
        self.next_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.next_attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        # 后后帧
        self.next_next_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.next_next_attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        
        # self.pre_value_proj = nn.Linear(embed_dims, embed_dims)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        # self.next_value_proj = nn.Linear(embed_dims, embed_dims)
        
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.pre_pre_sampling_offsets, 0.)
        constant_init(self.pre_sampling_offsets, 0.)
        constant_init(self.sampling_offsets, 0.)
        constant_init(self.next_sampling_offsets, 0.)
        constant_init(self.next_next_sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.pre_pre_sampling_offsets.bias.data = grid_init.view(-1)
        self.pre_sampling_offsets.bias.data = grid_init.view(-1)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        self.next_sampling_offsets.bias.data = grid_init.view(-1)
        self.next_next_sampling_offsets.bias.data = grid_init.view(-1)
        
        constant_init(self.pre_pre_attention_weights, val=0., bias=0.)
        constant_init(self.pre_attention_weights, val=0., bias=0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.next_attention_weights, val=0., bias=0.)
        constant_init(self.next_next_attention_weights, val=0., bias=0.)
        
        # xavier_init(self.pre_value_proj, distribution='uniform', bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        # xavier_init(self.next_value_proj, distribution='uniform', bias=0.)
        
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MulFramesMultiScaleDeformableAttentionV17_v2')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                query_time_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (torch.Tensor): The positional encoding for `key`. Default
                None.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2, 3)

        # num_gts, num_querys, embed_dims
        bs, num_query, _ = query.shape
        bs, num_value, num_frames, _ = value.shape
        # print('num_value:', num_value)
        # print('spatial:', spatial_shapes)
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask.transpose(1, 2)[..., None], 0.0)
        
        # num_gts, num_tokens, num_frames, embed_dims
        value = self.value_proj(value)
        # 提取不同帧的feature_tokens
        pre_pre_value = value[:, :, 0]
        pre_value = value[:, :, 1]
        now_value = value[:, :, 2]
        next_value = value[:, :, 3]
        next_next_value = value[:, :, 4]
        
        pre_pre_value = pre_pre_value.view(bs, num_value, self.num_heads, -1).contiguous()
        pre_value = pre_value.view(bs, num_value, self.num_heads, -1).contiguous()
        now_value = now_value.view(bs, num_value, self.num_heads, -1).contiguous()
        next_value = next_value.view(bs, num_value, self.num_heads, -1).contiguous()
        next_next_value = next_next_value.view(bs, num_value, self.num_heads, -1).contiguous()
        
        pre_pre_sampling_offsets = self.pre_pre_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        pre_sampling_offsets = self.pre_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        now_sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        next_sampling_offsets = self.next_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        next_next_sampling_offsets = self.next_next_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        pre_pre_attention_weights = self.pre_pre_attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        pre_attention_weights = self.pre_attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        now_attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        next_attention_weights = self.next_attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        next_next_attention_weights = self.next_next_attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
              
        # maybe exist BUG ------可能会出现错误
        pre_pre_attention_weights_sum = torch.exp(pre_pre_attention_weights).sum(-1, keepdim=True)
        pre_attention_weights_sum = torch.exp(pre_attention_weights).sum(-1, keepdim=True)
        now_attention_weights_sum = torch.exp(now_attention_weights).sum(-1, keepdim=True)
        next_attention_weights_sum = torch.exp(next_attention_weights).sum(-1, keepdim=True)
        next_next_attention_weights_sum = torch.exp(next_next_attention_weights).sum(-1, keepdim=True)
        
        sum_all = pre_pre_attention_weights_sum + pre_attention_weights_sum + now_attention_weights_sum + next_attention_weights_sum + next_next_attention_weights_sum
        # BUG

        pre_pre_attention_weights = pre_pre_attention_weights.softmax(-1)
        pre_attention_weights = pre_attention_weights.softmax(-1)
        now_attention_weights = now_attention_weights.softmax(-1)
        next_attention_weights = next_attention_weights.softmax(-1)
        next_next_attention_weights = next_next_attention_weights.softmax(-1)


        pre_pre_attention_weights = pre_pre_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        pre_attention_weights = pre_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        now_attention_weights = now_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        
        next_attention_weights = next_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        next_next_attention_weights = next_next_attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            pre_pre_reference_points = reference_points[:bs]
            pre_reference_points = reference_points[bs:bs*2]
            now_reference_points = reference_points[bs*2:bs*3]
            next_reference_points = reference_points[bs*3:bs*4]
            next_next_reference_points = reference_points[bs*4:]
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            
            pre_pre_sampling_locations = pre_pre_reference_points[:, :, None, :, None, :] \
                + pre_pre_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
                
            pre_sampling_locations = pre_reference_points[:, :, None, :, None, :] \
                + pre_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            
            now_sampling_locations = now_reference_points[:, :, None, :, None, :] \
                + now_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            
            next_sampling_locations = next_reference_points[:, :, None, :, None, :] \
                + next_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
                
            next_next_sampling_locations = next_next_reference_points[:, :, None, :, None, :] \
                + next_next_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
                
        elif reference_points.shape[-1] == 4:
            pre_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + pre_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
                
            now_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + now_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
                
            next_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + next_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
            
        if torch.cuda.is_available() and value.is_cuda:
            pre_pre_output = MultiScaleDeformableAttnFunction.apply(
                pre_pre_value, spatial_shapes, level_start_index, pre_pre_sampling_locations,
                pre_pre_attention_weights, self.im2col_step)
            
            pre_output = MultiScaleDeformableAttnFunction.apply(
                pre_value, spatial_shapes, level_start_index, pre_sampling_locations,
                pre_attention_weights, self.im2col_step)
            
            now_output = MultiScaleDeformableAttnFunction.apply(
                now_value, spatial_shapes, level_start_index, now_sampling_locations,
                now_attention_weights, self.im2col_step)
            
            next_output = MultiScaleDeformableAttnFunction.apply(
                next_value, spatial_shapes, level_start_index, next_sampling_locations,
                next_attention_weights, self.im2col_step)
            
            next_next_output = MultiScaleDeformableAttnFunction.apply(
                next_next_value, spatial_shapes, level_start_index, next_next_sampling_locations,
                next_next_attention_weights, self.im2col_step)
        else:
            pre_output = multi_scale_deformable_attn_pytorch(
                pre_value, spatial_shapes, level_start_index, pre_sampling_locations,
                pre_attention_weights, self.im2col_step)
            now_output = multi_scale_deformable_attn_pytorch(
                now_value, spatial_shapes, level_start_index, now_sampling_locations,
                now_attention_weights, self.im2col_step)
            next_output = multi_scale_deformable_attn_pytorch(
                next_value, spatial_shapes, level_start_index, next_sampling_locations,
                next_attention_weights, self.im2col_step)
            
        # TODO start: 融合多帧不同参考点对应的feature_token
        pre_pre_output = pre_pre_output.reshape(bs, num_query, self.num_heads, -1)
        pre_output = pre_output.reshape(bs, num_query, self.num_heads, -1)
        now_output = now_output.reshape(bs, num_query, self.num_heads, -1)
        next_output = next_output.reshape(bs, num_query, self.num_heads, -1)
        next_next_output = next_next_output.reshape(bs, num_query, self.num_heads, -1)
        
        output = pre_pre_output * (pre_pre_attention_weights_sum / sum_all) + \
                 pre_output * (pre_attention_weights_sum / sum_all) + \
                 now_output * (now_attention_weights_sum / sum_all) + \
                 next_output * (next_attention_weights_sum / sum_all) + \
                 next_next_output * (next_next_attention_weights_sum / sum_all)
                 
        output = output.flatten(-2, -1)
        # TODO end

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


# 适用于joint 单帧 修改时间 2024-10-26 ----- 每帧共享一个task_query, 然后单独去提取joint-token, 为了能更好的区分不同帧的task_query, 因此，加入每帧的时间位置编码
@ATTENTION.register_module()
class MulFramesMultiScaleDeformableAttentionV18(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MulFramesMultiScaleDeformableAttentionV18')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                query_time_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (torch.Tensor): The positional encoding for `key`. Default
                None.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            # 每帧的task_query添加单独的时间编码
            query = query + query_pos + query_time_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2, 3)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
