# Copyright (c) Hikvision Research Institute. All rights reserved.
import mmcv
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
from mmdet.core.visualization import color_val_matplotlib
from mmdet.core import bbox_mapping_back, multiclass_nms
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors.detr import DETR

from opera.core.keypoint import bbox_kpt2result, kpt_mapping_back
from ..builder import DETECTORS


@DETECTORS.register_module()
class PETR(DETR):
    """Implementation of `End-to-End Multi-Person Pose Estimation with
    Transformers`"""

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypoints,
                      gt_areas,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            gt_keypoints (list[Tensor]): Each item are the truth keypoints for
                each image in [p^{1}_x, p^{1}_y, p^{1}_v, ..., p^{K}_x,
                p^{K}_y, p^{K}_v] format.
            gt_areas (list[Tensor]): mask areas corresponding to each box.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_keypoints,
                                              gt_areas, gt_bboxes_ignore)
        return losses

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`.
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3),
                scale_factor=(1., 1., 1., 1.)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas=dummy_img_metas)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, dummy_img_metas, rescale=True)
        return bbox_list

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            img (list[torch.Tensor]): List of multiple images.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox and keypoint results of each image
                and classes. The outer list corresponds to each image.
                The inner list corresponds to each class.
        """
        import time
        start = time.time()
        batch_size = len(img_metas)
        assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
            f'mode is supported. Found batch_size {batch_size}.'
        
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)

        end = time.time()
        print(f'Inference time: {end - start}')

        bbox_kpt_results = [
            bbox_kpt2result(det_bboxes, det_labels, det_kpts,
                            self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_kpts in results_list
        ]
        return bbox_kpt_results

    def merge_aug_results(self, aug_bboxes, aug_kpts, aug_scores, img_metas):
        """Merge augmented detection bboxes and keypoints.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4).
            aug_kpts (list[Tensor] or None): shape (n, K, 2).
            img_metas (list): meta information.

        Returns:
            tuple: (bboxes, kpts, scores).
        """
        recovered_bboxes = []
        recovered_kpts = []
        for bboxes, kpts, img_info in zip(aug_bboxes, aug_kpts, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            kpts = kpt_mapping_back(kpts, img_shape, scale_factor, flip,
                                    flip_direction)
            recovered_bboxes.append(bboxes)
            recovered_kpts.append(kpts)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        kpts = torch.cat(recovered_kpts, dim=0)
        if aug_scores is None:
            return bboxes, kpts
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, kpts, scores

    def aug_test(self, imgs, img_metas, rescale=False):
        feats = self.extract_feats(imgs)
        aug_bboxes = []
        aug_scores = []
        aug_kpts = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x, img_meta)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=False)

            for det_bboxes, det_labels, det_kpts in bbox_list:
                aug_bboxes.append(det_bboxes[:, :4])
                aug_scores.append(det_bboxes[:, 4])
                aug_kpts.append(det_kpts[..., :2])

        merged_bboxes, merged_kpts, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_kpts, aug_scores, img_metas)

        merged_scores = merged_scores.unsqueeze(1)
        padding = merged_scores.new_zeros(merged_scores.shape[0], 1)
        merged_scores = torch.cat([merged_scores, padding], dim=-1)
        det_bboxes, det_labels, keep_inds = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            return_inds=True)
        det_kpts = merged_kpts[keep_inds]
        det_kpts = torch.cat(
            (det_kpts, det_kpts.new_ones(det_kpts[..., :1].shape)), dim=2)

        bbox_kpt_results = [
            bbox_kpt2result(det_bboxes, det_labels, det_kpts,
                            self.bbox_head.num_classes)
        ]
        return bbox_kpt_results

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=10,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'.
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'.
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, keypoint_result = result
            segm_result = None
        else:
            bbox_result, segm_result, keypoint_result = result, None, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # draw keypoints
        keypoints = None
        if keypoint_result is not None:
            keypoints = np.vstack(keypoint_result)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = self.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            keypoints,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def imshow_det_bboxes(self,
                          img,
                          bboxes,
                          labels,
                          segms=None,
                          keypoints=None,
                          class_names=None,
                          score_thr=0,
                          bbox_color='green',
                          text_color='green',
                          mask_color=None,
                          thickness=2,
                          font_size=8,
                          win_name='',
                          show=True,
                          wait_time=0,
                          out_file=None):
        """Draw bboxes and class labels (with scores) on an image.

        Args:
            img (str or ndarray): The image to be displayed.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
                (n, 5).
            labels (ndarray): Labels of bboxes.
            segms (ndarray or None): Masks, shaped (n,h,w) or None.
            keypoints (ndarray): keypoints (with scores), shaped (n, K, 3).
            class_names (list[str]): Names of each classes.
            score_thr (float): Minimum score of bboxes to be shown. Default: 0.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
                The tuple of color should be in BGR order. Default: 'green'.
                text_color (str or tuple(int) or :obj:`Color`):Color of texts.
                The tuple of color should be in BGR order. Default: 'green'.
            mask_color (str or tuple(int) or :obj:`Color`, optional):
                Color of masks. The tuple of color should be in BGR order.
                Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            show (bool): Whether to show the image. Default: True.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param. Default: 0.
            out_file (str, optional): The filename to write the image.
                Default: None.

        Returns:
            ndarray: The image with bboxes drawn on it.
        """
        assert bboxes.ndim == 2, \
            f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
        assert labels.ndim == 1, \
            f' labels ndim should be 1, but its ndim is {labels.ndim}.'
        assert bboxes.shape[0] == labels.shape[0], \
            'bboxes.shape[0] and labels.shape[0] should have the same length.'
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
            f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
        img = mmcv.imread(img).astype(np.uint8)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]
            if keypoints is not None:
                keypoints = keypoints[inds, ...]

        num_keypoint = keypoints.shape[1]
        if num_keypoint == 14:
            colors_hp = [(169, 209, 142), (255, 255, 0), (169, 209, 142),
                         (255, 255, 0), (169, 209, 142), (255, 255, 0),
                         (0, 176, 240), (252, 176, 243), (0, 176, 240),
                         (252, 176, 243), (0, 176, 240), (252, 176, 243),
                         (236, 6, 124), (236, 6, 124)]
        elif num_keypoint == 15:
            colors_hp = [(169, 209, 142), (255, 255, 0), (169, 209, 142),
                         (255, 255, 0), (169, 209, 142), (255, 255, 0),
                         (0, 176, 240), (252, 176, 243), (0, 176, 240),
                         (252, 176, 243), (0, 176, 240), (252, 176, 243),
                         (236, 6, 124), (236, 6, 124),(252, 176, 243)]
        elif num_keypoint == 17:
            colors_hp = [(236, 6, 124), (236, 6, 124), (236, 6, 124),
                         (236, 6, 124), (236, 6, 124), (169, 209, 142),
                         (255, 255, 0), (169, 209, 142), (255, 255, 0),
                         (169, 209, 142), (255, 255, 0), (0, 176, 240),
                         (252, 176, 243), (0, 176, 240), (252, 176, 243),
                         (0, 176, 240), (252, 176, 243)]
        else:
            raise ValueError(f'unsupported keypoint amount {num_keypoint}')
        colors_hp = [color[::-1] for color in colors_hp]
        colors_hp = [color_val_matplotlib(color) for color in colors_hp]

        if num_keypoint == 14:
            edges = [
                [0, 2],
                [2, 4],
                [1, 3],
                [3, 5],  # arms
                [0, 1],
                [0, 6],
                [1, 7],  # body
                [6, 8],
                [8, 10],
                [7, 9],
                [9, 11],  # legs
                [12, 13]
            ]  # neck
            ec = [(169, 209, 142),
                  (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),
                  (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),
                  (252, 176, 243), (252, 176, 243), (236, 6, 124)]
        elif num_keypoint == 15:
            edges = [
                [0, 2],
                [0, 1],
                [1, 3],
                [1, 4],
                [3, 5],  
                [4, 6],
                [3, 9],  
                [4, 10],
                [5, 7],
                [6, 8],
                [9, 11],  
                [10, 12],
                [11, 13],
                [12, 14]
            ] 
            ec = [(169, 209, 142),
                  (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),
                  (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),
                  (252, 176, 243), (252, 176, 243), (236, 6, 124), (236, 6, 124), (236, 6, 124)]
        elif num_keypoint == 17:
            edges = [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],  # head
                [5, 7],
                [7, 9],
                [6, 8],
                [8, 10],  # arms
                [5, 6],
                [5, 11],
                [6, 12],  # body
                [11, 13],
                [13, 15],
                [12, 14],
                [14, 16]
            ]  # legs
            ec = [(236, 6, 124), (236, 6, 124), (236, 6, 124), (236, 6, 124),
                  (169, 209, 142),
                  (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),
                  (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),
                  (252, 176, 243), (252, 176, 243)]
        else:
            raise ValueError(f'unsupported keypoint amount {num_keypoint}')
        ec = [color[::-1] for color in ec]
        ec = [color_val_matplotlib(color) for color in ec]

        img = mmcv.bgr2rgb(img)
        width, height = img.shape[1], img.shape[0]
        img = np.ascontiguousarray(img)

        EPS = 1e-2
        fig = plt.figure(win_name, frameon=False)
        plt.title(win_name)
        canvas = fig.canvas
        dpi = fig.get_dpi()
        # add a small EPS to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

        # remove white edges by set subplot margin
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis('off')

        polygons = []
        color = []
        for i, (bbox, label, kpt) in enumerate(zip(bboxes, labels, keypoints)):
            bbox_int = bbox.astype(np.int32)
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            # polygons.append(Polygon(np_poly))
            # color.append(bbox_color)
            # label_text = class_names[
            #     label] if class_names is not None else f'class {label}'
            # if len(bbox) > 4:
            #     label_text += f'|{bbox[-1]:.02f}'
            # get left-top corner of all keypoints
            bbox_int[0] = np.floor(kpt[:, 0].min()).astype(np.int32)
            bbox_int[1] = np.floor(kpt[:, 1].min() - 30).astype(np.int32)
            label_text = f'{bbox[-1]:.02f}'
            # ax.text(
            #     bbox_int[0],
            #     bbox_int[1],
            #     f'{label_text}',
            #     bbox={
            #         'facecolor': 'black',
            #         'alpha': 0.8,
            #         'pad': 0.7,
            #         'edgecolor': 'none'
            #     },
            #     color=text_color,
            #     fontsize=font_size,
            #     verticalalignment='top',
            #     horizontalalignment='left')
            for j in range(kpt.shape[0]):
                ax.add_patch(
                    Circle(
                        xy=(kpt[j, 0], kpt[j, 1]),
                        radius=2,
                        color=colors_hp[j]))
            for j, e in enumerate(edges):
                poly = [[kpt[e[0], 0], kpt[e[0], 1]],
                        [kpt[e[1], 0], kpt[e[1], 1]]]
                np_poly = np.array(poly).reshape((2, 2))
                polygons.append(Polygon(np_poly))
                color.append(ec[j])
            if segms is not None:
                color_mask = mask_colors[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

        plt.imshow(img)

        p = PatchCollection(
            polygons, facecolor='none', edgecolors=color, linewidths=thickness)
        ax.add_collection(p)

        stream, _ = canvas.print_to_buffer()
        buffer = np.frombuffer(stream, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        img = rgb.astype('uint8')
        img = mmcv.rgb2bgr(img)

        if show:
            # We do not use cvc2 for display because in some cases, opencv will
            # conflict with Qt, it will output a warning: Current thread
            # is not the object's thread. You an refer to
            # https://github.com/opencv/opencv-python/issues/46 for details
            if wait_time == 0:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        plt.close()
        return img
