# src/models/ir_motion_model.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import torch
import torch.nn as nn
import torch.nn.functional as F

class IRMotionNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # small 2D backbone for per-frame feature
        self.conv1 = nn.Conv2d(2, 24, kernel_size=3, padding=1)  # 2 input channels
        self.bn1 = nn.BatchNorm2d(24)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, C=2, T, H, W)
        B, C, T, H, W = x.shape
        x = x.view(B * T, C, H, W)      # (B*T, C, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)             # (B*T, 64,1,1)
        x = x.view(B, T, -1)            # (B, T, 64)
        x = x.mean(dim=1)               # temporal average (B,64)
        logits = self.fc(x)
        return logits
