# End-to-End Multi-Person Pose Estimation with Pose-Aware Video Transformer

This repo is the official implementation for **End-to-End Multi-Person Pose Estimation with Pose-Aware Video Transformer** [arxiv](https://arxiv.org/abs/2511.13208). The paper has been accepted to [AAAI 2026](https://aaai.org/conference/aaai/aaai-26/).


## Introduction

Existing multi-person video pose estimation methods typically adopt a two-stage pipeline: detecting individuals in each frame, followed by temporal modeling for single
person pose estimation. This design relies on heuristic operations such as detection, RoI cropping, and non-maximum suppression (NMS), limiting both accuracy and efficiency.
In this paper, we present a fully end-to-end framework for multi-person 2D pose estimation in videos, effectively eliminating heuristic operations. A key challenge is to associate individuals across frames under complex and overlapping temporal trajectories. To address this, we introduce a novel Pose-Aware Video transformEr Network (PAVE-Net),
which features a spatial encoder to model intra-frame relations and a spatiotemporal pose decoder to capture global dependencies across frames. To achieve accurate temporal association, we propose a pose-aware attention mechanism that enables each pose query to selectively aggregate features corresponding to the same individual across consecutive frames. Additionally, we explicitly model spatiotemporal dependencies among pose keypoints to improve accuracy. Notably, our approach is the first end-to-end method for multi-frame 2D human pose estimation. Extensive experiments show that PAVE-Net substantially outperforms prior image-based end-to-end methods, achieving a 6.0 mAP improvement on PoseTrack2017, and delivers accuracy competitive with state-of-the-art two-stage video based approaches, while offering significant gains in efficiency.

![PAVENet](demo/pipeline.png)

## Weights Download
The pretrained model weights have been released and are available for download at: [Resnet50](https://drive.google.com/drive/folders/1TSTbrm6jjq4LaK1CxqUNobcb6nF4LbMb?usp=drive_link) and [Swin-L](https://drive.google.com/drive/folders/17b9Oy9Sk_k5OjdzhKbaU4b2Qdr-4ksf3?usp=drive_link).
## Quantitative Performance
The following figure demonstrates the accuracy advantage of the method we proposed over the current advanced end-to-end algorithms based on static images.
| Method               | Backbone       | Head  |Shoulder| Elbow| Wrist |  Hip  | Knee  | Ankle | Mean |
|----------------------|----------------|-------|-------|-------|-------|-------|-------|-------|-------|
| **Image-Based**      |                |       |       |       |       |       |       |       |       |
| PETR (2022)          | ResNet-50      | 80.5  | 80.8  | 71.3  | 62.1  | 73.4  | 68.5  | 61.2  | 71.7  |
| GroupPose (2023)     | ResNet-50      | 82.4  | 82.1  | 73.3  | 64.3  | 74.4  | 70.7  | 63.7  | 73.6  |
| PETR (2022)          | HRNet-W48      | 82.4  | 83.2  | 74.4  | 70.8  | 74.5  | 72.3  | 66.9  | 75.4  |
| GroupPose (2023)     | HRNet-W48      | 83.3  | 84.3  | 77.8  | 70.3  | 75.6  | 72.8  | 66.8  | 76.3  |
| PETR (2022)          | Swin-L         | 83.3  | 84.3  | 78.3  | 71.3  | 76.4  | 73.4  | 67.6  | 76.8  |
| GroupPose (2023)     | Swin-L         | 83.9  | 84.7  | 78.8  | 70.6  | 77.5  | 74.4  | 68.7  | 77.4  |
| **Video-Based**      |                |       |       |       |       |       |       |       |       |
| PAVE-Net (Ours)      | ResNet-50      | 86.5  | 87.4  | 78.9  | 69.3  | 78.2  | 73.8  | 65.8  | 77.7  |
| PAVE-Net (Ours)      | HRNet-W48      | 87.1  | 88.4  | 80.9  | 73.9  | 80.3  | 76.9  | 69.9  | 80.1  |
| PAVE-Net (Ours)      | Swin-L         | 88.2  | 89.1  | 81.7  | 74.8  | 81.6  | 78.5  | 71.8  | 81.3  |

The following figure shows the speed advantage of the method we proposed over the current advanced two-stage algorithms, especially when there are a large number of people. Note: All results were obtained using a single A800 GPU, and the unit is milliseconds (ms).
<div align="center">
<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="5"><strong>Number of Persons (ms)</strong></th>
    </tr>
    <tr>
      <th>Method</th>
      <th>1</th><th>3</th><th>5</th><th>10</th><th>20</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="6"><strong>Two-Stage (Top-Down)</strong></td></tr>
    <tr><td>DCPose</td><td>150</td><td>204</td><td>292</td><td>431</td><td>721</td></tr>
    <tr><td>DSTA</td><td>122</td><td>181</td><td>265</td><td>418</td><td>631</td></tr>
    <tr><td colspan="6"><strong>End-to-End</strong></td></tr>
    <tr>
      <td>PETR</td>
      <td colspan="5" align="center">116</td>
    </tr>
    <tr>
      <td>GroupPose</td>
      <td colspan="5" align="center">89</td>
    </tr>
    <tr>
      <td><strong>PAVE-Net (Ours)</strong></td>
      <td colspan="5" align="center">153</td>
    </tr>

  </tbody>
</table>
</div>

## Visualizations
Here are some qualitative results from both the PoseTrack dataset and real-world scenarios:
![Result 1](demo/vs1.png)
![Result 2](demo/vs2.png)

## Usage and Install 
To download some auxiliary materials, please refer to [DCPose](https://github.com/Pose-Group/DCPose).

Follow the [PETR](https://github.com/hikvision-research/opera) to install the mmcv and mmdetection.
### Training
```
python tools/train.py --cfg your_config.yaml
```
### Evaluation
```
python tools/test.py --cfg your_config.yaml
```
### Citations
If you find our paper useful in your research, please consider citing:
```
@inproceedings{yu2025end,
  title={End-to-End Multi-Person Pose Estimation with Pose-Aware Video Transformer},
  author={Yu, Yonghui and Cai, Jiahang and Wang, Xun and Yang, Wenwu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## Acknowledgment
Our codes are mainly based on [PETR](https://github.com/hikvision-research/opera). Part of our code is borrowed from [DSTA](https://github.com/zgspose/DSTA)、[DCPose](https://github.com/Pose-Group/DCPose) and [RLE](https://github.com/jeffffffli/res-loglikelihood-regression). Many thanks to the authors!
