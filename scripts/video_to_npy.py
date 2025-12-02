#!/usr/bin/env python3
"""
video_to_npy.py

功能：
- 将 data_processed/small_subset_custom 下的 .avi/.mp4 视频逐视频解码为 numpy 保存文件 (.npy)
- 输出为与 video 同目录下同名但后缀为 .npy（例如 S001..._ir.avi -> S001..._ir.npy）
- 默认将帧Resize到 128x128 并转为单通道（灰度），以节约存储与训练计算
- 仅处理尚未存在对应 .npy 的视频（避免重复处理）
"""

import os
import cv2
import numpy as np
from pathlib import Path

# 配置（可按需修改）
ROOT = "data_processed/small_subset_custom"   # 小子集目录（脚本输出的目录）
OUT_RES = (128, 128)   # 输出帧大小 (width, height)
TO_GRAY = True         # 是否转灰度（True -> 单通道）

def is_video_file(name):
    return name.lower().endswith((".avi", ".mp4", ".mkv", ".mov"))

def process_video(video_path, out_npy_path, out_res=OUT_RES, to_gray=TO_GRAY):
    """
    读取 video_path，按帧解码，resize -> (H,W) 或 (H,W,3)，返回 np.array shape (T,H,W)
    并保存到 out_npy_path（.npy）
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频：{video_path}")
        return False
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 可选：转灰度
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 得到 (H,W)
        # resize
        frame = cv2.resize(frame, out_res, interpolation=cv2.INTER_LINEAR)  # (W,H) -> cv2 uses (w,h)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        print(f"[WARN] 视频无帧：{video_path}")
        return False
    arr = np.stack(frames, axis=0)  # shape (T,H,W) 或 (T,H,W,3)
    # 除非你想节省空间并且仍要保留 uint8，否则可以直接保存 uint8
    np.save(str(out_npy_path), arr)
    print(f"[OK] 保存 {out_npy_path}  (frames={arr.shape[0]}, shape={arr.shape[1:]})")
    return True

def main(root=ROOT):
    root = Path(root)
    if not root.exists():
        print(f"[ERROR] 根目录不存在: {root}")
        return
    # 遍历目录
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if is_video_file(fn):
                video_path = Path(dirpath) / fn
                out_npy = video_path.with_suffix(".npy")
                if out_npy.exists():
                    print(f"[SKIP] 已存在 .npy：{out_npy}")
                    continue
                # 如果当前 video_path 是符号链接，np.save 会把 .npy 放在链接的目录（which is fine）
                try:
                    process_video(video_path, out_npy)
                except Exception as e:
                    print(f"[ERR] 处理失败 {video_path}: {e}")

if __name__ == "__main__":
    main()
