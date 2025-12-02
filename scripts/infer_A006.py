#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse, torch, cv2, numpy as np
from src.models.ir_motion_model import IRMotionNet
from src.datasets.ir_video_onfly_dataset import normalize_ir_frame

def video_to_clip_tensor(video_path, clip_len=8, resize=(128,128)):
    # read frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame.ndim==3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    # if frames < clip_len, repeat
    if len(frames) == 0:
        return None
    if len(frames) < clip_len:
        reps = (clip_len + len(frames) -1)//len(frames)
        frames = (frames * reps)[:clip_len]
    # pick center clip
    if len(frames) > clip_len:
        start = (len(frames)-clip_len)//2
        frames = frames[start:start+clip_len]
    norm = [normalize_ir_frame(f) for f in frames]
    diffs=[]
    prev=None
    for f in norm:
        if prev is None:
            diffs.append(np.zeros_like(f))
        else:
            diffs.append(f-prev)
        prev=f
    arr = np.stack([np.stack(norm,0), np.stack(diffs,0)], axis=0).astype(np.float32)  # (2,T,H,W)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,2,T,H,W)
    return tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--ckpt", default="checkpoints/ir_motion_best.pth")
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--resize", type=int, default=128)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IRMotionNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    t = video_to_clip_tensor(args.video, clip_len=args.clip_len, resize=(args.resize,args.resize))
    if t is None:
        print("[ERR] cannot read video")
        return
    t = t.to(device)
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1)[0]
        print(f"Prob(not pick): {probs[0].item():.4f}, Prob(pick): {probs[1].item():.4f}")

if __name__ == "__main__":
    main()
