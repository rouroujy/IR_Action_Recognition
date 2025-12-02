# scripts/train_npy_small.py
import sys
from pathlib import Path
# 把项目根加入 Python 模块搜索路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

# local imports
from src.datasets.ir_npy_dataset import IRNpyDataset
from src.models.simple_ir_model import SimpleIRNet

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for clips, labels in loader:
        clips = clips.to(device)   # (B,1,T,H,W)
        labels = labels.to(device)
        outputs = model(clips)     # (B, num_classes)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)
    avg_loss = running_loss / total if total>0 else 0.0
    acc = correct / total if total>0 else 0.0
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * clips.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += clips.size(0)
    avg_loss = running_loss / total if total>0 else 0.0
    acc = correct / total if total>0 else 0.0
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser(description="Train SimpleIRNet on .npy IR clips (small subset).")
    parser.add_argument("--data-root", type=str, default="data_processed/small_subset_custom",
                        help="小子集目录，包含 .npy 文件")
    parser.add_argument("--clip-len", type=int, default=8, help="每个样本的时间长度 (帧数)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    # 兼容 Jupyter 的额外参数
    args, _ = parser.parse_known_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # label map（使用你提供的动作编号->语义映射）
    label_map = {"A037":0, "A040":1, "A043":2}
    num_classes = len(label_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    dataset = IRNpyDataset(args.data_root, label_map=label_map, clip_len=args.clip_len)
    # 如果数据太少，划分 train/val：按文件顺序划分 80/20
    n = len(dataset)
    if n == 0:
        print("[ERROR] 数据集为空，请确认 data_root 下有 .npy 文件。")
        return
    indices = list(range(n))
    split = max(1, int(0.8 * n))
    train_idx = indices[:split]
    val_idx = indices[split:]
    from torch.utils.data import Subset
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if val_idx else Subset(dataset, train_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SimpleIRNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    ensure_dir(args.save_dir)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        # 保存模型（按 val_acc）
        epoch_ckpt = os.path.join(args.save_dir, f"simple_ir_epoch{epoch}.pth")
        torch.save(model.state_dict(), epoch_ckpt)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_simple_ir.pth"))
    print("[DONE] Training finished. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
