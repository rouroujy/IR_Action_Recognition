#!/usr/bin/env python3
"""
scripts/collect_A006_list.py

扫描 data/ 下所有 NTU IR 子目录，收集 A006（pick up）视频路径，
并从非 A006 中随机采样负样本，生成 train/val 列表文件。

用法（在项目根运行）:
    python scripts/collect_A006_list.py --data-root data --out-dir data/labels_A006 --neg-sample-ratio 1 --val-split 0.1 --seed 42

参数:
 - data-root: 根数据目录（默认 data）
 - out-dir: 输出标签目录（会写 train.txt val.txt）
 - neg-sample-ratio: 负样与正样的比率（例如 1 -> 1:1）
 - val-split: 验证集占比
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import os, sys, random, argparse
from pathlib import Path

def find_all_videos(data_root):
    vids = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith((".avi", ".mp4", ".mkv")):
                vids.append(os.path.join(root, f))
    return vids

def is_A006(fn):
    # 文件名示例包含 A006 的形式
    fn_low = fn.lower()
    return "a006" in fn_low or "_a006" in fn_low or "a006_" in fn_low

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data", help="数据根目录")
    parser.add_argument("--out-dir", type=str, default="data/labels_A006", help="输出标签目录")
    parser.add_argument("--neg-sample-ratio", type=float, default=1.0, help="负样本比率（neg : pos）")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集比率")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    vids = find_all_videos(args.data_root)
    pos = [v for v in vids if is_A006(os.path.basename(v))]
    neg = [v for v in vids if not is_A006(os.path.basename(v))]

    print(f"[INFO] Found {len(pos)} positive (A006) videos, {len(neg)} negative videos in {args.data_root}")

    # 采样负样
    n_pos = len(pos)
    n_neg = min(len(neg), int(args.neg_sample_ratio * max(1, n_pos)))
    neg_sampled = random.sample(neg, n_neg) if n_neg>0 else []

    # 合并并标签化 1=pos,0=neg
    all_samples = [(p,1) for p in pos] + [(n,0) for n in neg_sampled]
    random.shuffle(all_samples)

    # split train/val
    n_total = len(all_samples)
    n_val = int(n_total * args.val_split)
    val = all_samples[:n_val]
    train = all_samples[n_val:]

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tfile = outdir / "train.txt"
    vfile = outdir / "val.txt"
    with open(tfile, "w") as f:
        for p,l in train:
            f.write(f"{p}\t{l}\n")
    with open(vfile, "w") as f:
        for p,l in val:
            f.write(f"{p}\t{l}\n")
    print(f"[DONE] train {len(train)} samples saved to {tfile}; val {len(val)} samples saved to {vfile}")

if __name__ == "__main__":
    main()
