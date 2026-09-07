"""
loss.py
Focal + Dice + Lovasz + Boundary Loss

修改：
1. Dice 不计算 Background
2. Focal 使用 class_weights
3. Lovasz 只优化缺陷类别
4. 去掉 cls 分支
5. 新增 Boundary Loss（边界感知，对细长缺陷有效）
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
            reduction="none",
            ignore_index=255
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

        # 忽略 ignore_index=255 的像素
        valid_mask = (target != 255).float()  # (B, H, W)

        loss = 0.0
        count = 0

        # 只计算 Oil / Stain / Scratch
        for c in cfg.defect_classes:

            pred = probs[:, c]

            gt = (target == c).float()

            # 排除 ignore 像素
            pred = pred * valid_mask
            gt = gt * valid_mask

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

        # 忽略 ignore_index=255 的像素
        valid_mask = (target != 255)

        loss = 0.0

        valid = 0

        for c in cfg.defect_classes:

            fg_all = (target == c).float()

            # 只保留有效像素
            fg = fg_all[valid_mask]

            if fg.sum() == 0:
                continue

            errors = (1 - probs[:, c])[valid_mask].reshape(-1)

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
# Boundary Loss（边界感知损失）
# ==========================================================

class BoundaryLoss(nn.Module):
    """Boundary Loss: 基于 Sobel 边界提取的边界感知损失。

    对每个缺陷类别：
    1. 从 GT mask 提取二值边界（Sobel 滤波器）
    2. 计算距离变换 (DT)，边界处=0，远离边界处值大
    3. 用 softmax 概率与 DT 做加权，聚焦边界区域

    参考: "Boundary Loss for Remote Sensing Imagery Semantic Segmentation" (ICCV 2019)
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0  # 边界提取阈值
        self.theta = theta    # DT 归一化范围

        # Sobel 卷积核（固定的，不参与训练）
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    @staticmethod
    def _compute_distance_transform(boundary: torch.Tensor) -> torch.Tensor:
        """近似距离变换：boundary 为 (B, H, W) 二值边界图，
        返回 (B, H, W) 浮点距离图，边界处=0，远离边界处值大。

        使用多次 max_pool2d 近似 Euclidean DT（计算高效，GPU 友好）。
        """
        # boundary: (B, H, W)，边界处=1
        # 反转：边界处=0，非边界处=1
        inv = 1.0 - boundary.unsqueeze(1).float()  # (B, 1, H, W)
        dist = inv.clone()
        # 多次 max pooling 膨胀，模拟距离传播
        kernel_size = 3
        for _ in range(15):  # 15 次 3x3 膨胀 ≈ 覆盖半径 ~15 像素
            dist = F.max_pool2d(
                dist, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            )
        # dist 现在是 0~1，边界处接近 0，远离边界处接近 1
        return 1.0 - dist.squeeze(1)  # (B, H, W)，边界处大，远离边界处小

    def get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """用 Sobel 滤波器提取 GT 边界。

        Args:
            mask: (B, H, W) 二值 mask

        Returns:
            boundary: (B, H, W) 二值边界
        """
        mask_f = mask.unsqueeze(1).float()  # (B, 1, H, W)
        gx = F.conv2d(mask_f, self.sobel_x, padding=1)
        gy = F.conv2d(mask_f, self.sobel_y, padding=1)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        boundary = (grad > self.theta0).float()
        return boundary.squeeze(1)  # (B, H, W)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes, H, W)
            target: (B, H, W) 语义分割 GT

        Returns:
            loss: scalar
        """
        probs = F.softmax(logits, dim=1)

        # 忽略 ignore_index=255 的像素
        valid_mask = (target != 255).float()  # (B, H, W)

        loss = 0.0
        count = 0

        for c in cfg.defect_classes:
            # 该类别的二值 GT
            gt_c = (target == c).float()  # (B, H, W)

            # 排除 ignore 像素
            gt_c = gt_c * valid_mask

            # 如果整个 batch 中该类别不存在，跳过
            if gt_c.sum() == 0:
                continue

            # 提取边界
            boundary = self.get_boundary(gt_c)  # (B, H, W)

            # 计算距离变换
            dt = self._compute_distance_transform(boundary)  # (B, H, W)

            # 归一化到 [0, 1]
            dt = dt / (dt.max() + 1e-8)
            dt = torch.clamp(dt - self.theta / dt.max(), min=0) if dt.max() > self.theta else dt
            dt = dt / (dt.max() + 1e-8)

            # 该类别的预测概率
            pred_c = probs[:, c]  # (B, H, W)

            # Boundary Loss = sum( |pred - gt| * DT * valid_mask )
            # 在边界附近 DT 大，损失贡献大
            boundary_loss = torch.abs(pred_c - gt_c) * dt * valid_mask
            loss += boundary_loss.mean()
            count += 1

        if count == 0:
            return torch.tensor(
                0.0,
                device=logits.device,
                requires_grad=True,
            )

        return loss / count


# ==========================================================
# Combined Loss
# ==========================================================

class CombinedLoss(nn.Module):
    """
    Final Loss

    Loss =
        w_focal    * Focal
      + w_dice     * Dice
      + w_lovasz   * Lovasz (warmup)
      + w_boundary * Boundary (warmup)

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

        self.boundary = BoundaryLoss()

        self.epoch = 0

        # Lovasz 延迟启动
        self.lovasz_start = 15

        # 5 epoch 内逐渐增加
        self.lovasz_warmup = 5

        # Boundary 延迟启动（先让模型学会大致分割，再优化边界）
        self.boundary_start = 10

        # 5 epoch 内逐渐增加
        self.boundary_warmup = 5

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

        # -----------------------------
        # Boundary
        # -----------------------------
        if (
            cfg.w_boundary > 0
            and self.epoch >= self.boundary_start
        ):
            b_factor = min(
                1.0,
                (self.epoch - self.boundary_start + 1)
                / self.boundary_warmup
            )
            loss += (
                cfg.w_boundary
                * b_factor
                * self.boundary(seg_logits, target)
            )

        return loss
