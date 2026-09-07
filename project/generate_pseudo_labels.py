"""生成高置信度伪标签（用于自训练 / self-training）。"""
import os
import glob
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import cfg
from data import load_image, get_val_transform
from model import build_model_custom
from utils import set_seed, sliding_window_inference


@torch.no_grad()
def predict_prob_map(model, img_tensor, device):
    """滑窗 + 全量 TTA 推理，返回 (H, W, C) 概率图。"""
    return sliding_window_inference(
        model,
        img_tensor,
        crop_size=cfg.infer_crop,
        overlap=cfg.infer_overlap,
        device=device,
        tta=cfg.tta_flips,
        num_classes=cfg.num_classes,
        has_cls=False,
        scales=cfg.tta_scales,
    )


def main():
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- 加载训练好的模型 ----------
    arch = cfg.arch
    encoder = cfg.encoder
    ckpt_path = os.path.join(cfg.ckpt_dir, 'best.pth')
    print(f'Loading {arch} / {encoder} from {ckpt_path}')

    model = build_model_custom(arch, encoder).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Mask2Former 需要屏蔽 mask 输入
    forward_orig = model.forward

    def forward_infer(pixel_values, **kwargs):
        kwargs.pop('mask_labels', None)
        kwargs.pop('class_labels', None)
        return forward_orig(pixel_values, **kwargs)

    model.forward = forward_infer

    # ---------- 目录准备 ----------
    pseudo_img_dir = cfg.pseudo_img_dir
    pseudo_mask_dir = cfg.pseudo_mask_dir
    os.makedirs(pseudo_img_dir, exist_ok=True)
    os.makedirs(pseudo_mask_dir, exist_ok=True)

    # ---------- 遍历测试集 ----------
    test_files = sorted(glob.glob(os.path.join(cfg.test_img, '*.jpg')))
    print(f'Test images: {len(test_files)}')

    val_transform = get_val_transform()
    threshold = cfg.pseudo_threshold
    stats = {'total_pixels': 0, 'valid_pixels': 0, 'class_counts': np.zeros(cfg.num_classes)}

    for path in tqdm(test_files, desc='Generating pseudo labels'):
        name = os.path.basename(path)
        img = load_image(path)
        H, W = img.shape[:2]

        # 复制原图到 pseudo_images（不额外占太多空间）
        dst_img = os.path.join(pseudo_img_dir, name)
        if not os.path.exists(dst_img):
            shutil.copy2(path, dst_img)

        # 推理
        augmented = val_transform(image=img)
        img_tensor = augmented['image'].unsqueeze(0).to(device)

        prob_map = predict_prob_map(model, img_tensor, device)
        prob_map = prob_map.cpu().numpy()  # (H, W, C)

        # 生成伪标签
        max_prob = prob_map.max(axis=2)    # (H, W)
        pred_cls = prob_map.argmax(axis=2).astype(np.int32)

        # 低置信度像素标记为 255（ignore）
        pseudo_mask = np.where(max_prob >= threshold, pred_cls, 255).astype(np.uint8)

        # 保存（使用灰度 PNG，值域 0/1/2/3/255）
        mask_name = os.path.splitext(name)[0] + '.png'
        cv2.imwrite(os.path.join(pseudo_mask_dir, mask_name), pseudo_mask)

        # 统计
        valid = max_prob >= threshold
        stats['total_pixels'] += H * W
        stats['valid_pixels'] += valid.sum()
        for c in range(cfg.num_classes):
            stats['class_counts'][c] += (pseudo_mask == c).sum()

    # ---------- 打印统计 ----------
    total = stats['total_pixels']
    valid = stats['valid_pixels']
    print('\n=== Pseudo Label Statistics ===')
    print(f'Total pixels:   {total:,}')
    print(f'Valid (>= {threshold}): {valid:,} ({valid / total * 100:.2f}%)')
    print(f'Ignored (< {threshold}): {total - valid:,}')
    for c in range(cfg.num_classes):
        name = cfg.class_names[c]
        cnt = int(stats['class_counts'][c])
        print(f'  {name:12s}: {cnt:>12,} pixels')
    print(f'\nSaved to:\n  Images: {pseudo_img_dir}\n  Masks:  {pseudo_mask_dir}')


if __name__ == '__main__':
    main()