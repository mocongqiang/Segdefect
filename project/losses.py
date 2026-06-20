"""组合损失: Focal + Dice + Lovasz"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = torch.tensor(alpha or [1.0] * cfg.num_classes,
                                  dtype=torch.float32)
        self.gamma = gamma

    def forward(self, logits, target):
        # logits: (B, C, H, W), target: (B, H, W)
        self.alpha = self.alpha.to(logits.device)
        ce = F.cross_entropy(logits, target, weight=self.alpha,
                             reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce).mean()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        C = cfg.num_classes
        loss = 0.0
        for c in range(C):
            p = probs[:, c]
            t = (target == c).float()
            inter = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice = (2 * inter + self.smooth) / (union + self.smooth)
            loss += 1 - dice.mean()
        return loss / C


class LovaszSoftmax(nn.Module):
    """Lovasz-Softmax 直接优化 mIoU。"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _lovasz_grad(gt_sorted):
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if jaccard.numel() > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        C = cfg.num_classes
        loss = 0.0
        for c in range(C):
            if c == 0:   # 跳过 Background
                continue
            fg = (target == c).float()
            if fg.sum() == 0:
                continue
            errors = (1 - probs[:, c]).flatten()
            fg = fg.flatten()
            inds = torch.argsort(errors, descending=True)
            fg = fg[inds]
            errors = errors[inds]
            grad = self._lovasz_grad(fg)
            loss += torch.dot(F.relu(errors), grad)
        return loss / 3.0   # 只算 3 类缺陷


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal  = FocalLoss(alpha=cfg.class_weights, gamma=2.0)
        self.dice   = DiceLoss()
        self.lovasz = LovaszSoftmax()
        self.bce    = nn.BCEWithLogitsLoss()   # 多标签分类
        self.epoch  = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, seg_logits, cls_logits, target, cls_target):
        loss = cfg.w_focal * self.focal(seg_logits, target) \
             + cfg.w_dice  * self.dice(seg_logits, target)
        if self.epoch >= 5:
            loss += cfg.w_lovasz * self.lovasz(seg_logits, target)
        loss += cfg.w_cls * self.bce(cls_logits, cls_target)
        return loss
