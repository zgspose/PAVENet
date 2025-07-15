# Copyright (c) Hikvision Research Institute. All rights reserved.
import contextlib
import io
import itertools
import logging
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
import os
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCOeval
from mmdet.datasets import CocoDataset
from .builder import DATASETS
from ..core.posetrack_utils import evaluate_simple
import scipy.io as sio
import json


@DATASETS.register_module()
class PosetrackPoseDataset(CocoDataset):
    """
    PoseTrack keypoint indexes::
        0: 'nose',
        1: 'head_bottom',
        2: 'head_top',
        3: 'left_shoulder',
        4: 'right_shoulder',
        5: 'left_elbow',
        6: 'right_elbow',
        7: 'left_wrist',
        8: 'right_wrist',
        9: 'left_hip',
        10: 'right_hip',
        11: 'left_knee',
        12: 'right_knee',
        13: 'left_ankle',
        14: 'right_ankle'
    """

    CLASSES = ('person', )

    FLIP_PAIRS = [[3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10],
                  [11, 12],
                  [13, 14]]
                  
    
    def __init__(self,
                 *args,
                 skip_invaild_pose=True,
                 **kwargs):
        super(PosetrackPoseDataset, self).__init__(*args, **kwargs)
        self.skip_invaild_pose = skip_invaild_pose

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['keypoint_fields'] = []
        results['area_fields'] = []

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox, mask and keypoint annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map, keypoints, areas. "masks" are raw \
                annotations and not decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_keypoints = []
        gt_areas = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            # skip invalid pose annotation
            if ann['num_keypoints'] == 0 and self.skip_invaild_pose:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_keypoints.append(ann.get('keypoints', None))
                gt_areas.append(ann.get('area', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_keypoints = np.array(gt_keypoints, dtype=np.float32)
            gt_areas = np.array(gt_areas, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_keypoints = np.zeros((0, 51), dtype=np.float32)
            gt_areas = np.array([], dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            keypoints=gt_keypoints,
            areas=gt_areas,
            flip_pairs=self.FLIP_PAIRS)

        return ann

    def _kpt2json(self, results):
        """Convert keypoint detection results to COCO json style."""
        bbox_json_results = []
        kpt_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, kpt = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # kpt results
                # some detectors use different scores for bbox and kpt
                kpts = kpt[label]
                kpt_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['score'] = float(kpt_score[i])
                    data['category_id'] = self.cat_ids[label]
                    i_kpt = kpts[i].reshape(-1)
                    data['keypoints'] = i_kpt.tolist()
                    kpt_json_results.append(data)
        return bbox_json_results, kpt_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 4 types of results: proposals, bbox predictions, mask
        predictions, keypoint_predictions, and they have different data types.
        This method will automatically recognize the type, and dump them to
        json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            if isinstance(results[0][-1][0],
                          np.ndarray) and results[0][-1][0].ndim == 3:
                json_results = self._kpt2json(results)
                result_files['bbox'] = f'{outfile_prefix}.bbox.json'
                result_files['proposal'] = f'{outfile_prefix}.bbox.json'
                result_files['keypoints'] = f'{outfile_prefix}.keypoints.json'
                # 不需要bbox结果
                # mmcv.dump(json_results[0], result_files['bbox'])
                mmcv.dump(json_results[1], result_files['keypoints'])
            else:
                json_results = self._segm2json(results)
                result_files['bbox'] = f'{outfile_prefix}.bbox.json'
                result_files['proposal'] = f'{outfile_prefix}.bbox.json'
                result_files['segm'] = f'{outfile_prefix}.segm.json'
                mmcv.dump(json_results[0], result_files['bbox'])
                mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def evaluate(self,
                 results,
                 metric='keypoints',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 save_dir='work_dirs/vedpose_r50_100e_coco_6_3_v1'):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'keypoints', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        # do eval with api
        # 加载预测结果
        coco_dt = self.coco.loadRes(result_files['keypoints'])
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.params.maxDets = [30]
        coco_eval.params.imgIds = list(coco_dt.imgToAnns.keys())
        coco_eval.evaluate()
        # 获取处理后的结果
        out_data = coco_eval.out_data
        output_dir = os.path.join(save_dir, 'val_set_json_results')
        self.create_folder(output_dir)
        annot_dir = 'DcPose_supp_files/posetrack17_annotation_dirs/jsons/val/'
        out_filenames, L = self.video2filenames(annot_dir)
        # 处理没有gt的帧的结果
        for key in out_data.keys():
            new_data = []
            data = out_data[key]
            arrs = data[0]['image']['name'].split('/')
            num_frames = L['images/' + key]
            frame_ids = [i for i in range(1, num_frames+1)]
            # 获取已经预测的frame_id
            count = 0
            used_frame_ids = [d['img_num'][0] for d in data]
            for frame_id in frame_ids:
                if frame_id not in used_frame_ids:
                    annorect = []
                    img_sfx = arrs[0] + '/' + arrs[1] + '/' + str(frame_id).zfill(8) + '.jpg'
                    annorect.append({
                        'annopoints': [{'point': [{
                            'id': [0],
                            'x': [0],
                            'y': [0],
                            'score': [-100.0],
                        }]}],
                        'score': [0],
                        'track_id': [0]})
                    new_data.append({
                        'image': {'name': img_sfx},
                        'imgnum': [frame_id],
                        'annorect': annorect
                        
                    })
                    count += 1
                else:
                    new_data.append(data[frame_id-count-1])
            out_data[key] = new_data
            
        print_log("=> saving files for evaluation")
        #### saving files for evaluation
        for vname in out_data.keys():
            vdata = out_data[vname]
            outfpath = os.path.join(output_dir, out_filenames[os.path.join('images', vname)])

            write_json_to_file({'annolist': vdata}, outfpath)

        # run evaluation
        # AP = self._run_eval(annot_dir, output_dir)[0]
        # output_dir = "work_dirs/test/val_set_json_results"
        AP = evaluate_simple.evaluate(annot_dir, output_dir, eval_track=False)[0]
        print(AP)

        name_value = [
            ('Head', AP[0]),
            ('Shoulder', AP[1]),
            ('Elbow', AP[2]),
            ('Wrist', AP[3]),
            ('Hip', AP[4]),
            ('Knee', AP[5]),
            ('Ankle', AP[6]),
            ('Mean', AP[7])
        ]

        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
    
    def create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
    def video2filenames(self, annot_dir):
        pathtodir = annot_dir

        output, L = {}, {}
        mat_files = [f for f in os.listdir(pathtodir) if
                    os.path.isfile(os.path.join(pathtodir, f)) and '.mat' in f]
        json_files = [f for f in os.listdir(pathtodir) if
                    os.path.isfile(os.path.join(pathtodir, f)) and '.json' in f]

        if len(json_files) > 1:
            files = json_files
            ext_types = '.json'
        else:
            files = mat_files
            ext_types = '.mat'

        for fname in files:
            if ext_types == '.mat':
                out_fname = fname.replace('.mat', '.json')
                data = sio.loadmat(
                    os.path.join(pathtodir, fname), squeeze_me=True,
                    struct_as_record=False)
                temp = data['annolist'][0].image.name

                data2 = sio.loadmat(os.path.join(pathtodir, fname))
                num_frames = len(data2['annolist'][0])
            elif ext_types == '.json':
                out_fname = fname
                with open(os.path.join(pathtodir, fname), 'r') as fin:
                    data = json.load(fin)

                if 'annolist' in data:
                    temp = data['annolist'][0]['image'][0]['name']
                    num_frames = len(data['annolist'])
                else:
                    temp = data['images'][0]['file_name']
                    num_frames = data['images'][0]['nframes']


            else:
                raise NotImplementedError()
            video = os.path.dirname(temp)
            output[video] = out_fname
            L[video] = num_frames
        return output, L
    
def write_json_to_file(data, output_path, flag_verbose=False):
    with open(output_path, "w") as write_file:
        json.dump(data, write_file)
    if flag_verbose is True:
        print("Json string dumped to: %s", output_path)

       