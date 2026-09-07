"""mIoU 评价 (仅 3 类缺陷, 不含 Background)"""
import torch
import numpy as np
from config import cfg

class IoUMetric:
    """累积预测并计算 per-class IoU 和 mIoU。"""

    def __init__(self, num_classes=4, defect_classes=[1, 2, 3]):
        self.nc = num_classes
        self.defect_classes = defect_classes
        self.reset()

    def reset(self):
        self.inter = np.zeros(self.nc, dtype=np.int64)
        self.union = np.zeros(self.nc, dtype=np.int64)

    @torch.no_grad()
    def update(self, pred, target):
        """pred, target: (B, H, W) int64. 忽略 target=255 的像素。"""
        pred   = pred.cpu().numpy().astype(np.int64)
        target = target.cpu().numpy().astype(np.int64)
        # 排除 ignore_index=255 的像素
        valid = (target != 255)
        pred = pred[valid]
        target = target[valid]
        for c in range(self.nc):
            p = pred == c
            t = target == c
            self.inter[c] += (p & t).sum()
            self.union[c] += (p | t).sum()

    def compute(self):
        iou = {}
        for c in range(self.nc):
            if self.union[c] == 0:
                iou[c] = float('nan')
            else:
                iou[c] = self.inter[c] / self.union[c]

        # 缺陷类 mIoU
        defect_ious = [iou[c] for c in self.defect_classes
                       if not np.isnan(iou[c])]
        miou = np.mean(defect_ious) if defect_ious else 0.0
        return iou, miou

    def __str__(self):
        iou, miou = self.compute()
        names = cfg.class_names
        parts = [f'{names[c]}: {iou[c]:.4f}' for c in self.defect_classes]
        return ' | '.join(parts) + f' | mIoU: {miou:.4f}'