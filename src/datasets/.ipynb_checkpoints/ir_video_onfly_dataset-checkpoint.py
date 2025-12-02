# src/datasets/ir_video_onfly_dataset.py


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import os
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize_ir_frame(frame, eps=1e-6):
    # frame: ndarray HxW (uint8)
    # 1) 转 float32
    f = frame.astype(np.float32)
    # 2) 减去中位数再除以标准差（robust normalization）
    m = np.median(f)
    std = f.std()
    return (f - m) / (std + eps)

def read_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转灰度（若已灰度则保持）
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    return frames  # list of HxW uint8 arrays

class IRVideoOnFlyDataset(Dataset):
    """
    在线解码视频并构造 (C=2, T, H, W) 样本:
      - channel0: normalized intensity
      - channel1: frame difference (curr - prev), first diff = 0
    标签: 1=pick up (A006), 0=other
    切片逻辑: 随机起点（训练）或中心起点（验证）
    """
    def __init__(self, list_file, clip_len=8, resize=(128,128), mode="train"):
        """
        list_file: 每行 'full_path<TAB>label'
        clip_len: 帧数
        resize: (W,H)
        mode: "train"/"val"
        """
        self.samples = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p,l = line.split()
                self.samples.append((p, int(l)))
        self.clip_len = clip_len
        self.resize = resize
        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def _get_clip_from_frames(self, frames):
        T_total = len(frames)
        if T_total == 0:
            return None
        # resize frames and normalize
        frames_resized = [cv2.resize(f, self.resize) for f in frames]
        # choose start
        if self.mode == "train":
            if T_total >= self.clip_len:
                start = random.randint(0, T_total - self.clip_len)
                clip_frames = frames_resized[start:start+self.clip_len]
            else:
                # repeat last frame
                reps = (self.clip_len + T_total - 1) // T_total
                clip_frames = (frames_resized * reps)[:self.clip_len]
        else:
            # val: center crop
            if T_total >= self.clip_len:
                start = max(0, (T_total - self.clip_len)//2)
                clip_frames = frames_resized[start:start+self.clip_len]
            else:
                reps = (self.clip_len + T_total - 1) // T_total
                clip_frames = (frames_resized * reps)[:self.clip_len]
        # build channels
        norm_frames = [normalize_ir_frame(f) for f in clip_frames]  # float arrays
        diffs = []
        prev = None
        for f in norm_frames:
            if prev is None:
                diffs.append(np.zeros_like(f))
            else:
                diffs.append(f - prev)
            prev = f
        # stack channels: shape (T, H, W, 2)
        ch0 = np.stack(norm_frames, axis=0)  # (T,H,W)
        ch1 = np.stack(diffs, axis=0)
        # convert to (C, T, H, W)
        arr = np.stack([ch0, ch1], axis=0).astype(np.float32)
        return arr

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = read_video_frames(path)
        arr = self._get_clip_from_frames(frames)
        if arr is None:
            # 处理异常：返回 zeros
            c,t,h,w = 2, self.clip_len, self.resize[1], self.resize[0]
            return torch.zeros((c,t,h,w), dtype=torch.float32), int(label)
        # to tensor
        tensor = torch.from_numpy(arr)  # shape (C, T, H, W)
        return tensor, int(label)
