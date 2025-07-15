# Copyright (c) Hikvision Research Institute. All rights reserved.
from .dkdetr_head import DKDETRHead
from .inspose_head import InsPoseHead
from .petr_head import PETRHead
from .soit_head import SOITHead
from .videopose_head_v1 import VideoPoseHeadV1 # 在原始的petr的pose-decoder中对前后帧也进行pose-query  ---- 使用前后帧pose-token与当前帧pose-token进行交互，升级当前帧的pose-token
from .videopose_head_v2 import VideoPoseHeadV2  # 保持原始petr的encoder和pose-decoder ---- pose-decoder只对当前帧处理，添加了pose-tracking模块
from .videopose_head_v3 import VideoPoseHeadV3 # 修改pose-tracking提取策略
from .videopose_head_v4 import VideoPoseHeadV4 # pose-tracking中辅助帧提取pose-token过程中加入self-attn
from .vedpose_head import VedPoseHead
from .vedpose_head_v2 import VedPoseHeadV2
from .vedpose_head_v3 import VedPoseHeadV3
from .vedpose_head_test import VedPoseHeadTest
from .videopose_head_v5 import VideoPoseHeadV5 # pose-decoder之后添加单样本多帧pose时序信息融合
from .videopose_head_v6 import VideoPoseHeadV6 # encoder之后添加单样本多帧特征时序信息融合
from .videopose_head_v8 import VideoPoseHeadV8 # pose-decoder使用时空解码， joint-decoder之后进行时空双分支并行融合时序
from .videopose_head_v9 import VideoPoseHeadV9
from .videopose_head_v10 import VideoPoseHeadV10
from .videopose_head_v11 import VideoPoseHeadV11
from .videopose_head_v12 import VideoPoseHeadV12
from .videopose_head_v12_1 import VideoPoseHeadV12_1
from .videopose_head_v13 import VideoPoseHeadV13
from .videopose_head_v13_1 import VideoPoseHeadV13_1
from .videopose_head_v14 import VideoPoseHeadV14
from .videopose_head_v14_1 import VideoPoseHeadV14_1
from .videopose_head_v14_2 import VideoPoseHeadV14_2

from .videopose_head_v16 import VideoPoseHeadV16
from .videopose_head_v17_1 import VideoPoseHeadV17_1
from .videopose_head_v17_2 import VideoPoseHeadV17_2
from .videopose_head_v17_3 import VideoPoseHeadV17_3
from .videopose_head_v17_4 import VideoPoseHeadV17_4
from .videopose_head_v17_5 import VideoPoseHeadV17_5
from .videopose_head_v17_5_test import VideoPoseHeadV17_5_test
from .videopose_head_v17_5_freeze import VideoPoseHeadV17_5_freeze



from .videopose_head_v11_9_1_freeze import VideoPoseHeadV11_9_1_freeze
from .videopose_head_v11_9_1_unfreeze import VideoPoseHeadV11_9_1_unfreeze

from .videopose_head_v11_9_1 import VideoPoseHeadV11_9_1
from .videopose_head_v11_9_2_freeze import VideoPoseHeadV11_9_2_freeze
from .videopose_head_v11_9_2_unfreeze import VideoPoseHeadV11_9_2_unfreeze
from .videopose_head_v11_9_3_freeze import VideoPoseHeadV11_9_3_freeze
from .videopose_head_v14_2_num_frame_5 import VideoPoseHeadV14_2_num_frame_5
from .videopose_head_V11_10_1 import VideoPoseHeadV11_10_1
from .videopose_head_v11_13_5_unfreeze import VideoPoseHeadV11_13_5_unfreeze
from .videopose_head_v11_19 import VideoPoseHeadV11_19
from .videopose_head_v11_21_v1 import VideoPoseHeadV11_21_v1
from .videopose_head_v11_21_v2 import VideoPoseHeadV11_21_v2
from .videopose_head_v11_21_v3 import VideoPoseHeadV11_21_v3


from .videopose_head_v17_6 import VideoPoseHeadV17_6
from .videopose_head_v17_2_test import VideoPoseHeadV17_2_test
from .videopose_head_v18_1 import VideoPoseHeadV18_1
from .videopose_head_v18_2 import VideoPoseHeadV18_2
from .videopose_head_mul_frames import VideoPoseHeadMulFrames
__all__ = ['DKDETRHead', 'InsPoseHead', 'PETRHead', 'SOITHead', 'VedPoseHeadTest', 'VedPoseHeadV2', 'VedPoseHeadV3', 'VideoPoseHeadV11_9_1','VideoPoseHeadV14_2',
           'VideoPoseHeadV1', 'VideoPoseHeadV2', 'VedPoseHead',
           'VideoPoseHeadV3', 'VideoPoseHeadV4', 'VideoPoseHeadV5',
           'VideoPoseHeadV6', 'VideoPoseHeadV8', 'VideoPoseHeadV9',
           'VideoPoseHeadV10', 'VideoPoseHeadV11', 'VideoPoseHeadV12', 'VideoPoseHeadV12_1',
           'VideoPoseHeadV13', 'VideoPoseHeadV13_1','VideoPoseHeadV14', 'VideoPoseHeadV14_1',
           'VideoPoseHeadV16', 'VideoPoseHeadV17_1', 'VideoPoseHeadV17_2', 'VideoPoseHeadV18_1', 'VideoPoseHeadV18_2',
           'VideoPoseHeadV17_2_test', 'VideoPoseHeadV17_3', 'VideoPoseHeadV17_4', 'VideoPoseHeadV17_5', 'VideoPoseHeadV17_6',
           'VideoPoseHeadV17_5_test', 'VideoPoseHeadV17_5_freeze',
           'VideoPoseHeadV11_9_1_freeze', 'VideoPoseHeadV11_9_2_freeze', 'VideoPoseHeadV11_9_3_freeze', 'VideoPoseHeadV14_2_num_frame_5',
           'VideoPoseHeadV11_10_1', 'VideoPoseHeadV11_9_2_unfreeze', 'VideoPoseHeadV11_9_1_unfreeze', 'VideoPoseHeadV11_13_5_unfreeze', 'VideoPoseHeadV11_19',
           'VideoPoseHeadV11_21_v1', 'VideoPoseHeadV11_21_v2', 'VideoPoseHeadV11_21_v3', 'VideoPoseHeadMulFrames'
           ]
