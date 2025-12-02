#!/usr/bin/env python3
"""
prepare_small_subset.py  （为 rourou 定制版）

说明：
- 根据你提供的路径示例，脚本默认将源目录设置为：
    ~/data/nturgbd_ir_s001/nturgb+d_ir
  如果你想使用别的目录，可用 --src 指定。
- 脚本会从源目录中抽取极小子集（默认每个动作/被试取 3 个样本），
  优先创建符号链接（节省磁盘），若无法创建符号链接可用 --use-copy 强制复制。
- 脚本同时能处理两种常见组织结构：
  1) src 下直接是大量 .avi 文件（例如你的示例文件 S001C001P001R001A001_ir.avi）
  2) src/<action>/<subject>/<files...> 的层级结构
- 运行方式（在项目根目录下）:
    python scripts/prepare_small_subset.py
  或显式指定路径：
    python scripts/prepare_small_subset.py --src /home/yourname/data/nturgbd_ir_s001/nturgb+d_ir
"""

import os
import sys
import argparse
import shutil
import random
from pathlib import Path

# ========== 根据你给的示例路径设定默认 src ==========
# 你给出的示例文件路径： ~/data/nturgbd_ir_s001/nturgb+d_ir/S001C001P001R001A001_ir.avi
HOME = str(Path.home())
DEFAULT_SRC = os.path.join(HOME, "project","IR_Action_Recognition_bind","data", "nturgbd_ir_s001", "nturgb+d_ir")
DEFAULT_DST = "data_processed/small_subset_custom"   # 我给一个项目内的默认输出目录

# ========== 工具函数 ==========
def ensure_dir(p):
    """如果目录不存在就创建（递归创建）"""
    os.makedirs(p, exist_ok=True)

def try_symlink_or_copy(src, dst, use_copy=False):
    """
    优先尝试创建符号链接；如果失败或 use_copy=True 则复制文件。
    返回 'symlink' 或 'copied' 或 'exists'。
    """
    ensure_dir(os.path.dirname(dst))
    if os.path.lexists(dst):
        return "exists"
    if not use_copy:
        try:
            os.symlink(src, dst)
            return "symlink"
        except Exception:
            # 创建符号链接失败（常见于 Windows 挂载点或权限问题），回退到复制
            use_copy = True
    if use_copy:
        shutil.copy2(src, dst)
        return "copied"

def is_video_file(name):
    name = name.lower()
    return name.endswith((".avi", ".mp4", ".mov", ".mkv"))

def human_readable_size(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"

# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="为你的 NTU IR 数据准备一个小子集（定制版）。")
    parser.add_argument("--src", type=str, default=DEFAULT_SRC,
                        help=f"源数据目录（默认：{DEFAULT_SRC}）")
    parser.add_argument("--dst", type=str, default=DEFAULT_DST,
                        help=f"输出目录（默认：{DEFAULT_DST}）")
    parser.add_argument("--samples-per-file", type=int, default=3,
                        help="在每个检测到的组/被试下选取多少个样本（默认 3）")
    parser.add_argument("--use-copy", action="store_true",
                        help="强制复制文件到 dst（而不是创建符号链接）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()
    random.seed(args.seed)

    src = os.path.expanduser(args.src)
    dst = args.dst

    print(f"[INFO] 使用的源目录(src) = {src}")
    print(f"[INFO] 输出目录(dst) = {dst}")
    ensure_dir(dst)

    if not os.path.exists(src):
        print(f"[ERROR] 源目录不存在：{src}")
        sys.exit(1)

    # 显示目标盘剩余空间（粗略）
    try:
        du = shutil.disk_usage(dst)
        print(f"[INFO] 输出目录所在分区剩余空间: {human_readable_size(du.free)}")
    except Exception:
        pass

    # 检测 src 的组织结构
    entries = sorted(os.listdir(src))
    # 情形 A: src 下有大量 video 文件（像你给的示例），直接把这些作为样本池
    video_files = [f for f in entries if is_video_file(f)]
    # 情形 B: src 下是 action 子目录（action/subject/files）
    action_dirs = [d for d in entries if os.path.isdir(os.path.join(src, d))]

    total_handled = 0
    stats = {"symlink":0, "copied":0, "exists":0}

    if video_files and (not action_dirs):
        # 情形 A：直接处理 src 下的视频文件集合
        print(f"[INFO] 检测到 {len(video_files)} 个视频文件在 src 下（直接按文件抽样）。")
        n = min(len(video_files), args.samples_per_file)
        chosen = random.sample(video_files, n)
        for fname in chosen:
            s = os.path.join(src, fname)
            dst_dir = os.path.join(dst, "videos")
            ensure_dir(dst_dir)
            d = os.path.join(dst_dir, fname)
            method = try_symlink_or_copy(s, d, use_copy=args.use_copy)
            stats[method] = stats.get(method,0) + 1
            total_handled += 1
            print(f"  -> {fname}  [{method}]")
    else:
        # 情形 B：按 action/subject 层级遍历，兼容多层级结构
        print(f"[INFO] 源目录被解析为动作/子目录结构（action/subject/...），将对每个动作随机抽样。")
        # 遍历 action
        for action in action_dirs:
            action_path = os.path.join(src, action)
            # 列出被试或直接文件
            subj_entries = sorted(os.listdir(action_path))
            subjs = [d for d in subj_entries if os.path.isdir(os.path.join(action_path, d))]
            # 如果没有子目录（subj），则把 action_path 下的文件作为样本
            if not subjs:
                files = [f for f in subj_entries if is_video_file(f)]
                if not files:
                    # 尝试更深层查找
                    for root, _, filenames in os.walk(action_path):
                        for fn in filenames:
                            if is_video_file(fn):
                                files.append(os.path.join(root, fn))
                chosen = random.sample(files, min(len(files), args.samples_per_file)) if files else []
                for p in chosen:
                    src_path = p if os.path.isabs(p) else os.path.join(action_path, p)
                    dst_dir = os.path.join(dst, action)
                    ensure_dir(dst_dir)
                    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
                    method = try_symlink_or_copy(src_path, dst_path, use_copy=args.use_copy)
                    stats[method] = stats.get(method,0) + 1
                    total_handled += 1
                    print(f"  -> {os.path.basename(src_path)}  [{method}]")
            else:
                # 有被试目录，遍历每个被试，随机抽样
                # 为节省时间，如果被试过多，只随机取前 8 个被试
                if len(subjs) > 12:
                    sampled_subjs = random.sample(subjs, 8)
                else:
                    sampled_subjs = subjs
                for subj in sampled_subjs:
                    subj_path = os.path.join(action_path, subj)
                    files = [f for f in sorted(os.listdir(subj_path)) if is_video_file(f)]
                    # 若当前层没文件，递归查找一层
                    if not files:
                        for root, _, filenames in os.walk(subj_path):
                            for fn in filenames:
                                if is_video_file(fn):
                                    files.append(os.path.join(root, fn))
                        # 将其转为相对名
                        files = [os.path.relpath(p, subj_path) for p in files]
                    if not files:
                        continue
                    n = min(len(files), args.samples_per_file)
                    chosen = random.sample(files, n) if len(files) > n else files[:n]
                    for fname in chosen:
                        src_path = os.path.join(subj_path, fname) if not os.path.isabs(fname) else fname
                        dst_dir = os.path.join(dst, action, subj)
                        ensure_dir(dst_dir)
                        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
                        method = try_symlink_or_copy(src_path, dst_path, use_copy=args.use_copy)
                        stats[method] = stats.get(method,0) + 1
                        total_handled += 1
                        print(f"  -> {action}/{subj}/{os.path.basename(src_path)}  [{method}]")

    print("==== 完成摘要 ====")
    print(f"总共处理样本数量: {total_handled}")
    print(f"符号链接: {stats.get('symlink',0)}, 复制: {stats.get('copied',0)}, 已存在: {stats.get('exists',0)}")
    print(f"子集已放到: {os.path.abspath(dst)}")
    print("提示：如想强制复制请加参数 --use-copy；如果源目录不是默认路径，请用 --src 指定。")

if __name__ == "__main__":
    main()
