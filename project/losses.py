"""组合损失: Focal + Dice + Lovasz + BCE(cls)

修改记录:
  - FocalLoss: alpha 注册为 buffer，自动跟随 device 迁移
  - DiceLoss: 只对缺陷类 [1,2,3] 计算，跳过 Background，与 mIoU 指标对齐
  - LovaszSoftmax: 延迟启用 + warmup 引入，避免损失跳变
  - CombinedLoss: BCEWithLogitsLoss 添加 pos_weight 平衡稀有类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg


class FocalLoss(nn.Module):
    """Focal Loss，alpha 注册为 buffer 自动跟随 device。"""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        alpha = alpha or [1.0] * cfg.num_classes
        # register_buffer: model.to(device) 时自动迁移，无需手动 .to()
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        # logits: (B, C, H, W), target: (B, H, W)
        ce = F.cross_entropy(logits, target, weight=self.alpha,
                             reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce).mean()
        return loss


class DiceLoss(nn.Module):
    """Dice Loss，对所有类别计算，缺陷类权重更高。

    包含 Background 以防止模型退化为全 Background 预测。
    使用 cfg.class_weights 保持与 FocalLoss 一致。
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        # 使用与 FocalLoss 相同的类别权重，保持一致性
        self.class_weights = cfg.class_weights

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        C = cfg.num_classes
        loss = 0.0
        weight_sum = 0.0
        for c in range(C):
            p = probs[:, c]
            t = (target == c).float()
            inter = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice = (2 * inter + self.smooth) / (union + self.smooth)
            w = self.class_weights[c]
            loss += w * (1 - dice.mean())
            weight_sum += w
        return loss / weight_sum


class LovaszSoftmax(nn.Module):
    """Lovasz-Softmax 直接优化 mIoU。只计算缺陷类。"""

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
        count = 0
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
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return loss / count


class CombinedLoss(nn.Module):
    """组合损失: Focal + Dice + Lovasz(延迟warmup) + BCE(cls)。"""

    def __init__(self):
        super().__init__()
        self.focal  = FocalLoss(alpha=cfg.class_weights, gamma=2.0)
        self.dice   = DiceLoss()
        self.lovasz = LovaszSoftmax()

        # 分类头: 根据各类别在数据集中的稀有程度设置 pos_weight
        # Stain(2) 极少 -> 3.0, Scratch(3) 少 -> 2.0, Oil(1) 正常 -> 1.0
        cls_pos_weight = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)

        self.epoch = 0

        # Lovasz warmup 参数: epoch >= lovasz_start 时开始线性 warmup
        self.lovasz_start = 4    # 第 5 个 epoch (0-indexed=4) 开始引入
        self.lovasz_warmup = 5   # 5 个 epoch 达到满权重

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, seg_logits, cls_logits, target, cls_target):
        loss = cfg.w_focal * self.focal(seg_logits, target) \
             + cfg.w_dice  * self.dice(seg_logits, target)

        # Lovasz warmup: 避免 epoch 4->5 时损失跳变
        if cfg.w_lovasz > 0 and self.epoch >= self.lovasz_start:
            warmup_factor = min(1.0, (self.epoch - self.lovasz_start + 1) / self.lovasz_warmup)
            loss += cfg.w_lovasz * warmup_factor * self.lovasz(seg_logits, target)

        if cfg.w_cls > 0 and cls_logits is not None:
            loss += cfg.w_cls * self.bce(cls_logits, cls_target)
        return loss
