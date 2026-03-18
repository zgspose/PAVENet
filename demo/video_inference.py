import torch
import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from mmcv.runner import load_checkpoint
from opera.models import build_model
from mmcv import Config
import warnings
warnings.filterwarnings('ignore')

KEYPOINT_COLORS = [
    (169, 209, 142), (255, 255, 0), (169, 209, 142),
    (255, 255, 0), (169, 209, 142), (255, 255, 0),
    (0, 176, 240), (252, 176, 243), (0, 176, 240),
    (252, 176, 243), (0, 176, 240), (252, 176, 243),
    (236, 6, 124), (236, 6, 124), (252, 176, 243)
]

EDGES = [
    [0, 2], [0, 1], [1, 3], [1, 4], [3, 5], [4, 6],
    [3, 9], [4, 10], [5, 7], [6, 8], [9, 11], [10, 12],
    [11, 13], [12, 14]
]

EDGE_COLORS = [
    (169, 209, 142), (169, 209, 142), (255, 255, 0), (255, 255, 0),
    (255, 102, 0), (0, 176, 240), (252, 176, 243), (0, 176, 240),
    (0, 176, 240), (252, 176, 243), (252, 176, 243), (236, 6, 124),
    (236, 6, 124), (236, 6, 124)
]

def init_model(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def preprocess_frames(frames, img_norm_mean, img_norm_std, device):
    processed_frames = []
    for frame in frames:
        img = frame.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - np.array(img_norm_mean, dtype=np.float32)) / np.array(img_norm_std, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).contiguous()
        processed_frames.append(img)
    
    img = torch.stack(processed_frames, dim=0).unsqueeze(0).to(device)
    
    h, w = frames[0].shape[:2]
    img_metas = [dict(
        batch_input_shape=(h, w),
        img_shape=(h, w, 3),
        scale_factor=(1.0, 1.0, 1.0, 1.0),
        flip=False,
        flip_direction='horizontal',
        pad_shape=(h, w, 3),
    )]
    
    return [img], [img_metas]

def draw_keypoints(frame, bbox_result, keypoint_result, score_thr=0.3):
    img = frame.copy()
    bboxes = np.vstack(bbox_result)
    keypoints = np.vstack(keypoint_result)
    if bboxes.shape[1] > 4:
        scores = bboxes[:, -1]
        keep = scores > score_thr
        bboxes = bboxes[keep]
        keypoints = keypoints[keep]
    for bbox, kpts in zip(bboxes, keypoints):
        kpts = kpts.reshape(-1, 3)
        valid = kpts[:, 2] > 0.3
        for j, (x, y, v) in enumerate(kpts):
            if v > 0.3:
                color = tuple(int(c) for c in KEYPOINT_COLORS[j])
                cv2.circle(img, (int(x), int(y)), 3, color, -1)
        for edge_idx, (i, j) in enumerate(EDGES):
            if valid[i] and valid[j]:
                pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                color = tuple(int(c) for c in EDGE_COLORS[edge_idx])
                cv2.line(img, pt1, pt2, color, 2)
    return img

def process_video(video_path, model, device, window_size=3, score_thr=0.3, out_path='output.mp4'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('Cannot open video file: ' + video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    img_norm_mean = model.cfg.get('img_norm_cfg', dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))['mean']
    img_norm_std = model.cfg.get('img_norm_cfg', dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))['std']
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    
    frame_buffer = []
    frame_idx = 0
    processed_count = 0
    
    print('Processing video: ' + str(total_frames) + ' frames...')
    print('Input: ' + str(frame_width) + 'x' + str(frame_height) + ' @ ' + str(fps) + ' fps, duration: ' + str(round(total_frames/fps, 2)) + 's')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_buffer.append(frame)
        
        if len(frame_buffer) >= window_size:
            try:
                imgs, img_metas = preprocess_frames(frame_buffer, img_norm_mean, img_norm_std, device)
                with torch.no_grad():
                    results = model(return_loss=False, rescale=True, img=imgs, img_metas=img_metas)
                bbox_result, keypoint_result = results[0]
                
                central_frame = frame_buffer[window_size // 2].copy()
                if len(bbox_result) > 0 and len(bbox_result[0]) > 0:
                    central_frame = draw_keypoints(central_frame, bbox_result, keypoint_result, score_thr)
                out.write(central_frame)
                processed_count += 1
            except Exception as e:
                print('Error at frame ' + str(frame_idx) + ': ' + str(e))
                out.write(frame_buffer[window_size // 2])
                processed_count += 1
            
            frame_buffer.pop(0)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print('Read ' + str(frame_idx) + '/' + str(total_frames) + ' frames, output ' + str(processed_count))
    
    cap.release()
    out.release()
    print('Done! Output: ' + str(processed_count) + ' frames, duration: ' + str(round(processed_count/fps, 2)) + 's')
    print('Video saved to: ' + out_path)

def parse_args():
    parser = argparse.ArgumentParser(description='PAVENet Video Inference')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file path')
    parser.add_argument('--video', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video file path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--window-size', type=int, default=3, help='Number of frames for temporal modeling')
    return parser.parse_args()

def main():
    args = parse_args()
    print('Initializing model with config: ' + args.config)
    print('Loading checkpoint: ' + args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    config_path = os.path.abspath(args.config)
    model = init_model(config_path, args.checkpoint, device)
    print('Processing video: ' + args.video)
    process_video(
        args.video,
        model,
        device,
        window_size=args.window_size,
        score_thr=args.score_thr,
        out_path=args.output
    )

if __name__ == '__main__':
    main()
