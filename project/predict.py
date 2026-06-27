"""推理 + 生成提交文件"""
import os, glob, time
import numpy as np
import torch
from tqdm import tqdm

from config import cfg
from data import load_image, get_val_transform, _pad_to_divisor
from model import build_model
from rle import generate_submission
from utils import set_seed, sliding_window_inference


def load_models(device, ensemble_ckpts=None):
    """加载模型（支持多 checkpoint 集成）。

    Args:
        ensemble_ckpts: list of checkpoint paths, None 则使用 best.pth

    Returns:
        list of (model, weight) tuples
    """
    if ensemble_ckpts is None:
        ensemble_ckpts = [os.path.join(cfg.ckpt_dir, 'best.pth')]

    models = []
    for ckpt_path in ensemble_ckpts:
        print(f'Loading checkpoint: {ckpt_path}')
        model = build_model().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        models.append((model, 1.0))  # 等权重集成

    return models


def ensemble_inference(models, img_tensor, device):
    """多模型集成推理。

    Args:
        models: list of (model, weight) tuples
        img_tensor: (1, 3, H, W) tensor

    Returns:
        prob_map: (H, W, C) numpy array
    """
    prob_sum = None
    weight_sum = 0.0

    for model, weight in models:
        prob = sliding_window_inference(
            model, img_tensor,
            crop_size=cfg.infer_crop,
            overlap=cfg.infer_overlap,
            device=device,
            tta=cfg.tta_flips,
            num_classes=cfg.num_classes,
            has_cls=False,
            scales=cfg.tta_scales,
        )  # (H, W, C)

        if prob_sum is None:
            prob_sum = prob * weight
        else:
            prob_sum += prob * weight
        weight_sum += weight

    return prob_sum / weight_sum


def main():
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- 加载模型（支持集成） ----------
    # 单模型: ensemble_ckpts = None (默认使用 best.pth)
    # 多模型集成: ensemble_ckpts = ['best.pth', 'epoch_40.pth', 'epoch_50.pth']
    ensemble_ckpts = None
    models = load_models(device, ensemble_ckpts)
    print(f'Ensemble: {len(models)} model(s)')

    # ---------- 预处理 transform（循环外创建） ----------
    val_transform = get_val_transform()

    # ---------- 推理 ----------
    test_files = sorted(glob.glob(os.path.join(cfg.test_img, '*.jpg')))
    print(f'Test images: {len(test_files)}')

    predictions = []
    names = []

    for path in tqdm(test_files, desc='Predicting'):
        name = os.path.basename(path)
        img = load_image(path)
        H, W = img.shape[:2]

        # Pad 到 32 的倍数（DeepLabV3Plus encoder 下采样 32 倍）
        img_padded, pad_h, pad_w = _pad_to_divisor(img, divisor=32, return_pad=True)

        # 预处理
        augmented = val_transform(image=img_padded)
        img_tensor = augmented['image'].unsqueeze(0).to(device)  # (1,3,H',W')

        # 集成推理
        prob_map = ensemble_inference(models, img_tensor, device)  # (H', W', C)

        # 裁剪回原始尺寸
        if pad_h > 0:
            prob_map = prob_map[:-pad_h, :, :]
        if pad_w > 0:
            prob_map = prob_map[:, :-pad_w, :]

        pred = prob_map.argmax(axis=2).astype(np.uint8)   # (H, W)
        predictions.append(pred)
        names.append(name)

    # ---------- 生成提交 ----------
    sub_path = os.path.join(cfg.sub_dir, 'submission.csv')
    df = generate_submission(predictions, names, sub_path)

    # 验证
    print(f'\nSubmission preview:')
    print(df.head(12))
    print(f'Total rows: {len(df)} (expected {len(test_files) * 3})')

    # 检查非空 RLE 数量
    non_empty = (df['rle'] != '0 0').sum()
    print(f'Non-empty RLE: {non_empty} / {len(df)}')


if __name__ == '__main__':
    main()