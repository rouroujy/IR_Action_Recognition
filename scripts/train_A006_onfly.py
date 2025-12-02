#!/usr/bin/env python3
"""
scripts/train_A006_onfly.py

训练针对 A006(pick up) 的二分类模型，数据在线解码，不生成中间文件。
推荐先用 --max-samples 100 调试，确认工作流后再去掉该参数训练完整数据。

示例（调试）:
  python scripts/train_A006_onfly.py --train-list data/labels_A006/train.txt --val-list data/labels_A006/val.txt --epochs 3 --batch-size 2 --max-samples 200

示例（完整训练）:
  python scripts/train_A006_onfly.py --train-list data/labels_A006/train.txt --val-list data/labels_A006/val.txt --epochs 30 --batch-size 8
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os, sys, random, argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.datasets.ir_video_onfly_dataset import IRVideoOnFlyDataset
from src.models.ir_motion_model import IRMotionNet

def parse_list(fpath, max_samples=None):
    lines = []
    with open(fpath, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            lines.append((parts[0], int(parts[1])))
    if max_samples and len(lines) > max_samples:
        random.shuffle(lines)
        return lines[:max_samples]
    return lines

def collate_fn(batch):
    # batch: list of (tensor, label)
    clips = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    clips = torch.stack(clips, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return clips, labels

# train_one_epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total, correct = 0, 0
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():  # GPU AMP
                outputs = model(clips)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)

    return total_loss / total, correct / total if total > 0 else 0.0


# eval_one_epoch
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total, correct = 0, 0
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():  # GPU AMP
                    outputs = model(clips)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(clips)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * clips.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += clips.size(0)

    return total_loss / total, correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", type=str, required=True)
    parser.add_argument("--val-list", type=str, required=True)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--resize", type=int, default=128, help="短边 resize 大小 (square)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="调试用：限制最大样本数")
    args, _ = parser.parse_known_args()

    # ------------------ fix random seed ------------------
    import numpy as np, random, torch
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # -----------------------------------------------------


    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] device = {device}")

    # 读取列表（可限制样本数方便调试）
    train_lines = parse_list(args.train_list, max_samples=args.max_samples if args.max_samples>0 else None)
    val_lines = parse_list(args.val_list, max_samples=args.max_samples if args.max_samples>0 else None)

    # 临时把 train_lines 写到临时文件供 Dataset 读取
    tmp_train = "data/labels_A006/train_tmp.txt"
    tmp_val = "data/labels_A006/val_tmp.txt"
    Path("data/labels_A006").mkdir(parents=True, exist_ok=True)
    with open(tmp_train, "w") as f:
        for p,l in train_lines:
            f.write(f"{p}\t{l}\n")
    with open(tmp_val, "w") as f:
        for p,l in val_lines:
            f.write(f"{p}\t{l}\n")

    train_set = IRVideoOnFlyDataset(tmp_train, clip_len=args.clip_len, resize=(args.resize,args.resize), mode="train")
    val_set = IRVideoOnFlyDataset(tmp_val, clip_len=args.clip_len, resize=(args.resize,args.resize), mode="val")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                         shuffle=True, num_workers=args.num_workers,
                         collate_fn=collate_fn, pin_memory=True)


    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn, pin_memory=True)


    model = IRMotionNet(num_classes=2).to(device)
    #criterion = nn.CrossEntropyLoss()

    # 在读取 train list 后计算类权重（插入到 main() 在生成 train_lines 之后）
    from collections import Counter
    train_lines = parse_list(args.train_list, max_samples=args.max_samples if args.max_samples>0 else None)
    labels = [l for _, l in train_lines]
    cnt = Counter(labels)
    # weight for class i = total / (num_classes * count_i)
    total = sum(cnt.values())
    num_classes = 2
    weights = [total / (num_classes * cnt[i]) for i in range(num_classes)]
    import torch
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        torch.save(model.state_dict(), f"checkpoints/ir_epoch_{epoch}.pth")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/ir_motion_best.pth")
    print("[DONE] Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
