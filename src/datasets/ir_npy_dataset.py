# src/datasets/ir_npy_dataset.py
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class IRNpyDataset(Dataset):
    """
    每个样本为一个 .npy 文件，内容为 uint8 或 float32，shape (T,H,W) 或 (T,H,W,3)
    返回: tensor (C=1, T, H, W), label (int)
    """
    def __init__(self, root_dir, label_map=None, clip_len=16, transform=None):
        self.root_dir = str(root_dir)
        self.clip_len = int(clip_len)
        self.transform = transform
        self.label_map = label_map or {}  # e.g. {"A037":0, "A040":1, "A043":2}
        self.files = []
        # 遍历目录，收集 .npy 文件
        for dp, _, fns in os.walk(self.root_dir):
            for fn in fns:
                if fn.lower().endswith(".npy"):
                    self.files.append(os.path.join(dp, fn))
        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)

    def _infer_label_from_path(self, path):
        """
        根据文件名或路径推断动作编号（例如文件名包含 A037）
        若匹配到 label_map 对应的键则返回对应 id，否则返回 -1
        """
        p = path.replace("\\", "/")
        for k, v in self.label_map.items():
            if f"/{k}" in p or f"{k}_" in p or f"_{k}" in p or f"{k}." in p:
                return int(v)
            # 有些文件名像 S001C001P001R001A001_ir.npy -> 检测 "A037" 形式
            if k.lower() in p.lower():
                return int(v)
        return -1

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = np.load(path)  # (T,H,W) 或 (T,H,W,3)
        # 兼容：若为彩色 (T,H,W,3)，转为灰度取第0通道
        if arr.ndim == 4:
            arr = arr[..., 0]
        # dtype 处理 -> float32 [0,1]
        if arr.dtype == np.uint8:
            arr = arr.astype("float32") / 255.0
        else:
            arr = arr.astype("float32")
        T = arr.shape[0]
        # 抽取 clip_len 帧：若足够长则随机裁剪，否则重复最后一帧补齐
        if T >= self.clip_len:
            if self.clip_len == T:
                clip = arr
            else:
                start = np.random.randint(0, T - self.clip_len + 1)
                clip = arr[start:start + self.clip_len]
        else:
            reps = (self.clip_len + T - 1) // T
            clip = np.tile(arr, (reps, 1, 1))[:self.clip_len]
        # (T,H,W) -> (C=1, T, H, W)
        clip = np.expand_dims(clip, axis=0)
        clip = torch.from_numpy(clip)  # float32 tensor
        label = self._infer_label_from_path(path)
        if label < 0:
            # 若无法推断 label，设为 0（训练时可忽略或手动提供 label_map）
            label = 0
        return clip, int(label)
