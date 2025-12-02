# src/models/simple_ir_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleIRNet(nn.Module):
    """
    输入 x: (B, C=1, T, H, W)
    处理方式：把时间维度拆成 B*T 个帧做 2D 卷积，再时间平均
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B,1,T,H,W)
        B, C, T, H, W = x.shape
        x = x.view(B * T, C, H, W)           # (B*T, 1, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)                  # (B*T, 32, 1, 1)
        x = x.view(B, T, -1)                 # (B, T, 32)
        x = x.mean(dim=1)                    # 时间平均 -> (B, 32)
        logits = self.fc(x)                  # (B, num_classes)
        return logits
