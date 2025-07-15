# Copyright (c) Hikvision Research Institute. All rights reserved.
from .builder import DATASETS, PIPELINES, build_dataset, build_dataloader
from .coco_pose import CocoPoseDataset
from .crowd_pose import CrowdPoseDataset
from .objects365 import Objects365
# 单样本数据包含连续的三帧 默认使用的数据集
from .posetrack_video_pose import PosetrackVideoPoseDataset
# 单样本数据包含连续的五帧
from .posetrack_video_pose_num_frames_5 import PosetrackVideoPoseDatasetV2
from .posetrack_pose import PosetrackPoseDataset
from .pipelines import *
from .utils import replace_ImageToTensor


from .coco_video_pose import CocoVideoPoseDataset
from .coco_video_pose_num_frames_5 import CocoVideoPoseDataset5


__all__ = [
    'DATASETS', 'PIPELINES', 'build_dataset', 'build_dataloader',
    'CocoPoseDataset', 'CrowdPoseDataset', 'Objects365', 'PosetrackVideoPoseDataset', 'PosetrackPoseDataset', 'PosetrackVideoPoseDatasetV2',
    'replace_ImageToTensor', 'CocoVideoPoseDataset', 'CocoVideoPoseDataset5'
]
