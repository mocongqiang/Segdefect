"""
loss.py
Focal + Dice + Lovasz

修改：
1. Dice 不计算 Background
2. Focal 使用 class_weights
3. Lovasz 只优化缺陷类别
4. 去掉 cls 分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


# ==========================================================
# Focal Loss
# ==========================================================

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=None,
                 gamma=2.0):
        super().__init__()

        if alpha is None:
            alpha = cfg.class_weights

        self.gamma = gamma

        self.register_buffer(
            "alpha",
            torch.tensor(alpha, dtype=torch.float32)
        )

    def forward(self, logits, target):

        ce = F.cross_entropy(
            logits,
            target,
            weight=self.alpha,
            reduction="none"
        )

        pt = torch.exp(-ce)

        focal = ((1 - pt) ** self.gamma) * ce

        return focal.mean()


# ==========================================================
# Dice Loss（只计算缺陷）
# ==========================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):

        probs = F.softmax(logits, dim=1)

        loss = 0.0
        count = 0

        # 只计算 Oil / Stain / Scratch
        for c in cfg.defect_classes:

            pred = probs[:, c]

            gt = (target == c).float()

            inter = (pred * gt).sum(dim=(1, 2))

            union = pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))

            dice = (2 * inter + self.smooth) / (
                union + self.smooth
            )

            loss += (1 - dice.mean())

            count += 1

        return loss / count


# ==========================================================
# Lovasz Softmax
# ==========================================================

class LovaszSoftmax(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def lovasz_grad(gt_sorted):

        gts = gt_sorted.sum()

        intersection = gts - gt_sorted.float().cumsum(0)

        union = gts + (1 - gt_sorted).float().cumsum(0)

        jaccard = 1.0 - intersection / (union + 1e-8)

        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]

        return jaccard

    def forward(self, logits, target):

        probs = F.softmax(logits, dim=1)

        loss = 0.0

        valid = 0

        for c in cfg.defect_classes:

            fg = (target == c).float()

            if fg.sum() == 0:
                continue

            errors = (1 - probs[:, c]).reshape(-1)

            fg = fg.reshape(-1)

            perm = torch.argsort(errors, descending=True)

            errors = errors[perm]

            fg = fg[perm]

            grad = self.lovasz_grad(fg)

            loss += torch.dot(F.relu(errors), grad)

            valid += 1

        if valid == 0:
            return torch.tensor(
                0.0,
                device=logits.device,
                requires_grad=True
            )

        return loss / valid

# ==========================================================
# Combined Loss
# ==========================================================

class CombinedLoss(nn.Module):
    """
    Final Loss

    Loss =
        w_focal  * Focal
      + w_dice   * Dice
      + w_lovasz * Lovasz (warmup)

    已删除分类头(BCE)。
    """

    def __init__(self):
        super().__init__()

        self.focal = FocalLoss(
            alpha=cfg.class_weights,
            gamma=2.0
        )

        self.dice = DiceLoss()

        self.lovasz = LovaszSoftmax()

        self.epoch = 0

        # Lovasz 延迟启动
        self.lovasz_start = 15

        # 5 epoch 内逐渐增加
        self.lovasz_warmup = 5

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self,
                seg_logits,
                cls_logits=None,
                target=None,
                cls_target=None):

        loss = 0.0

        # -----------------------------
        # Focal
        # -----------------------------
        if cfg.w_focal > 0:
            loss += (
                cfg.w_focal *
                self.focal(seg_logits, target)
            )

        # -----------------------------
        # Dice
        # -----------------------------
        if cfg.w_dice > 0:
            loss += (
                cfg.w_dice *
                self.dice(seg_logits, target)
            )

        # -----------------------------
        # Lovasz
        # -----------------------------
        if (
            cfg.w_lovasz > 0
            and self.epoch >= self.lovasz_start
        ):

            factor = min(
                1.0,
                (self.epoch - self.lovasz_start + 1)
                / self.lovasz_warmup
            )

            loss += (
                cfg.w_lovasz
                * factor
                * self.lovasz(seg_logits, target)
            )

        return loss