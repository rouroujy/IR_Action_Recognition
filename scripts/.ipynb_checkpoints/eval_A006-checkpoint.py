#!/usr/bin/env python3
# scripts/eval_A006.py
"""
批量评估脚本：对 val.txt（格式: path<TAB>label）里的样本做推理，
输出：总体 accuracy、混淆矩阵、precision/recall/f1（用 sklearn），
并把错误样本路径保存到 logs/misclassified.txt 便于可视化分析。
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import os, sys
from pathlib import Path
import argparse
import torch
import numpy as np
from tqdm import tqdm

# add project root to path if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.ir_motion_model import IRMotionNet
from src.datasets.ir_video_onfly_dataset import normalize_ir_frame
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def video_to_clip_tensor(video_path, clip_len=8, resize=(128,128)):
    import cv2
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None
    # center clip
    if len(frames) < clip_len:
        reps = (clip_len + len(frames) - 1) // len(frames)
        frames = (frames * reps)[:clip_len]
    elif len(frames) > clip_len:
        start = (len(frames) - clip_len) // 2
        frames = frames[start:start+clip_len]
    norm = [normalize_ir_frame(f) for f in frames]
    diffs = []
    prev = None
    for f in norm:
        if prev is None:
            diffs.append(np.zeros_like(f))
        else:
            diffs.append(f - prev)
        prev = f
    arr = np.stack([np.stack(norm,0), np.stack(diffs,0)], axis=0).astype(np.float32)  # (2,T,H,W)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,2,T,H,W)
    return tensor

def parse_list(file):
    lines = []
    with open(file, "r") as f:
        for l in f:
            l = l.strip()
            if not l: continue
            p, lab = l.split()
            lines.append((p, int(lab)))
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-list", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="checkpoints/ir_motion_best.pth")
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--resize", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] device =", device)
    model = IRMotionNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    samples = parse_list(args.val_list)
    y_true = []
    y_pred = []
    mis = []
    for p, lab in tqdm(samples, desc="Eval"):
        if not os.path.exists(p):
            print("[WARN] missing:", p)
            continue
        t = video_to_clip_tensor(p, clip_len=args.clip_len, resize=(args.resize, args.resize))
        if t is None:
            mis.append((p, lab, -1, "no_frames"))
            continue
        t = t.to(device)
        with torch.no_grad():
            logits = model(t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())
        y_true.append(lab)
        y_pred.append(pred)
        if pred != lab:
            mis.append((p, lab, pred, float(probs[pred])))

    # metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)

    os.makedirs("logs", exist_ok=True)
    with open("logs/misclassified.txt", "w") as f:
        for p, lab, pred, score in mis:
            f.write(f"{p}\t{lab}\t{pred}\t{score}\n")
    print(f"[DONE] misclassified saved to logs/misclassified.txt")

if __name__ == "__main__":
    main()
