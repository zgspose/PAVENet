# Copyright (c) OpenMMLab. All rights reserved.
from .auto_augment import (AutoAugment, BrightnessTransform, ColorTransform,
                           ContrastTransform, EqualizeTransform, Rotate, Shear,
                           Translate)
from .compose import Compose
from .formatting import (Collect, DefaultFormatBundle, ImageToTensor, FixCollect, FixDefaultFormatBundle, MulImageToTensor, # add ---- 添加时间 2024-8-02
                         ToDataContainer, ToTensor, Transpose, to_tensor, ImageToTensorV2, DefaultFormatBundleV2)
from .instaboost import InstaBoost
from .loading import (FilterAnnotations, LoadAnnotations, LoadImageFromFile, LoadMulImageFromFileForCocoFrames3, LoadMulImageFromFileForCocoFrames5, LoadMulImageFromFileForPosetrackFrames3, LoadMulImageFromFileForPosetrackFrames5, # add ---- 添加时间 2024-8-02
                      LoadImageFromWebcam, LoadMultiChannelImageFromFiles,
                      LoadPanopticAnnotations, LoadProposals)
from .test_time_aug import MultiScaleFlipAug
from .transforms import (Albu, CopyPaste, CutOut, Expand, MinIoURandomCrop,
                         MixUp, Mosaic, Normalize, Pad, PhotoMetricDistortion, MulPhotoMetricDistortionForFrames3, MulPhotoMetricDistortionForFrames5, # add ---- 添加时间 2024-8-02
                         RandomAffine, RandomCenterCropPad, RandomCrop, MulRandomCrop, # add ---- 添加时间 2024-8-02
                         RandomFlip, RandomShift, Resize, SegRescale,
                         YOLOXHSVRandomAug, RandomFlipV2, ResizeV2, MulRandomCropV2, NormalizeV2, PadV2)

__all__ = [
    'Compose', 'to_tensor', 'ToTensor', 'ImageToTensor', 'ToDataContainer', 'FixCollect', 'FixDefaultFormatBundle', 'MulImageToTensor',
    'Transpose', 'Collect', 'DefaultFormatBundle', 'LoadAnnotations',
    'LoadImageFromFile', 'LoadImageFromWebcam', 'LoadPanopticAnnotations',
    'LoadMultiChannelImageFromFiles', 'LoadProposals', 'FilterAnnotations',
    'MultiScaleFlipAug', 'Resize', 'RandomFlip', 'Pad', 'RandomCrop', 'MulRandomCrop',
    'Normalize', 'SegRescale', 'MinIoURandomCrop', 'Expand',
    'PhotoMetricDistortion', 'Albu', 'InstaBoost', 'RandomCenterCropPad',
    'AutoAugment', 'CutOut', 'Shear', 'Rotate', 'ColorTransform',
    'EqualizeTransform', 'BrightnessTransform', 'ContrastTransform',
    'Translate', 'RandomShift', 'Mosaic', 'MixUp', 'RandomAffine',
    'YOLOXHSVRandomAug', 'CopyPaste', 'RandomFlipV2', 'ResizeV2', 'MulRandomCropV2', 'NormalizeV2', 'ImageToTensorV2',
    'PadV2', 'DefaultFormatBundleV2',
    # 单样本多帧图像加载
    'LoadMulImageFromFileForCocoFrames3', 'LoadMulImageFromFileForCocoFrames5', 
    'LoadMulImageFromFileForPosetrackFrames3', 'LoadMulImageFromFileForPosetrackFrames5',
    # 单样本多帧图像增强
    'MulPhotoMetricDistortionForFrames3', 'MulPhotoMetricDistortionForFrames5',
]
