"""工具函数: 种子、EMA、学习率调度、滑窗推理"""
import os, random, math
import numpy as np
import torch
import torch.nn.functional as F
from config import cfg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential Moving Average."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v
            else:
                self.shadow[k] = v.clone()

    def apply_shadow(self, model):
        model.load_state_dict(self.shadow)

    def restore(self, model, backup):
        model.load_state_dict(backup)


def get_lr_scheduler(optimizer, epochs, warmup_epochs):
    """CosineAnnealingWarmup."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def sliding_window_inference(model, image, crop_size, overlap, device,
                             tta=True, num_classes=4, has_cls=False,
                             scales=None):
    """滑窗推理, 支持翻转 TTA + 多尺度 TTA。

    Args:
        model: 分割模型 (如果 has_cls=True, 返回 (seg_logits, cls_logits))
        image: (1, 3, H, W) tensor
        crop_size: int
        overlap: int
        device: torch.device
        tta: bool, 是否使用翻转 TTA
        num_classes: int
        has_cls: bool, 模型是否返回分类输出
        scales: list[float] | None, 多尺度因子, 如 [0.75, 1.0, 1.25], None 等价于 [1.0]

    Returns:
        prob_map: (H, W, C) numpy array, softmax 概率
    """
    scales = scales or [1.0]
    _, _, H, W = image.shape
    prob_sum = torch.zeros(H, W, num_classes, device=device)
    count_sum = torch.zeros(H, W, 1, device=device)

    for scale in scales:
        # ── 缩放图像 ──
        if scale == 1.0:
            scaled_img = image
        else:
            new_h, new_w = int(round(H * scale)), int(round(W * scale))
            scaled_img = F.interpolate(image, size=(new_h, new_w),
                                       mode='bilinear', align_corners=False)

        # ── 单尺度滑窗+翻转 TTA ──
        prob_map_scaled = _single_scale_inference(
            model, scaled_img, crop_size, overlap, device,
            tta=tta, num_classes=num_classes, has_cls=has_cls,
        )  # (scale_h, scale_w, C) numpy

        # ── 缩回原图尺寸 ──
        prob_t = torch.from_numpy(prob_map_scaled).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,C,H_s,W_s)
        prob_t = F.interpolate(prob_t, size=(H, W), mode='bilinear',
                               align_corners=False)  # (1,C,H,W)
        prob_t = prob_t[0].permute(1, 2, 0)  # (H,W,C)
        prob_sum += prob_t
        count_sum += 1.0

    prob_map = (prob_sum / count_sum).cpu().numpy()   # (H,W,C)
    return prob_map


@torch.no_grad()
def _single_scale_inference(model, image, crop_size, overlap, device,
                            tta=True, num_classes=4, has_cls=False):
    """单尺度滑窗推理 (含翻转 TTA)。

    Args:
        image: (1, 3, H, W) tensor
    其余参数同 sliding_window_inference。

    Returns:
        prob_map: (H, W, C) numpy array
    """
    _, _, H, W = image.shape
    stride = crop_size - overlap

    # 计算裁剪位置
    h_starts = list(range(0, max(H - crop_size, 0) + 1, stride))
    if h_starts[-1] + crop_size < H:
        h_starts.append(max(H - crop_size, 0))
    if not h_starts:
        h_starts = [0]

    w_starts = list(range(0, max(W - crop_size, 0) + 1, stride))
    if w_starts[-1] + crop_size < W:
        w_starts.append(max(W - crop_size, 0))
    if not w_starts:
        w_starts = [0]

    prob_sum = torch.zeros(H, W, num_classes, device=device)
    count    = torch.zeros(H, W, 1, device=device)

    flips = [(False, False)]
    if tta:
        flips += [(True, False), (False, True), (True, True)]

    for hs in h_starts:
        for ws in w_starts:
            he = min(hs + crop_size, H)
            we = min(ws + crop_size, W)
            ph = he - hs
            pw = we - ws

            patch = image[:, :, hs:he, ws:we]    # (1,3,ph,pw)
            # padding to crop_size
            if ph < crop_size or pw < crop_size:
                patch = F.pad(patch, (0, crop_size - pw, 0, crop_size - ph),
                              mode='reflect')

            for hflip, vflip in flips:
                p = patch.clone()
                if hflip:
                    p = torch.flip(p, dims=[3])
                if vflip:
                    p = torch.flip(p, dims=[2])

                out = model(p)
                seg_logits = out[0] if has_cls else out
                prob = F.softmax(seg_logits, dim=1)   # (1,C,cs,cs)

                if hflip:
                    prob = torch.flip(prob, dims=[3])
                if vflip:
                    prob = torch.flip(prob, dims=[2])

                prob = prob[0, :, :ph, :pw].permute(1, 2, 0)  # (ph,pw,C)
                prob_sum[hs:he, ws:we] += prob
                count[hs:he, ws:we]    += 1

    prob_map = (prob_sum / count).cpu().numpy()   # (H,W,C)
    return prob_map
