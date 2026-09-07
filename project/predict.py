"""推理 + 生成提交文件（直接 resize 多尺寸融合版）

关键优化（已验证）：
- 直接 resize 优于滑窗：滑窗在窗口边缘产生拼接伪影，直接 resize 更稳定
- 768 是最优主尺寸，三尺寸加权融合效果最佳
- 禁用后处理：阈值/面积过滤会误删真实缺陷区域
- RLE 列优先编码（F-order）
"""
import os, glob, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import cfg
from data import load_image, get_val_transform
from model import build_model, build_model_custom
from rle import generate_submission
from utils import set_seed


def load_models(device, ensemble_configs=None, use_ensemble=False):
    """加载模型（支持单模型、多 checkpoint 集成）。

    Args:
        ensemble_configs: list of dicts, 每项包含 {arch, encoder, ckpt, weight}
        use_ensemble: bool, 是否使用集成模式

    Returns:
        list of (model, weight) tuples
    """
    if use_ensemble and ensemble_configs is not None:
        models = []
        for item in ensemble_configs:
            print(f'Loading ensemble model: {item["arch"]} + {item["encoder"]}')
            print(f'  Checkpoint: {item["ckpt"]}')
            model = build_model_custom(item['arch'], item['encoder']).to(device)
            ckpt = torch.load(item['ckpt'], map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            models.append((model, item['weight']))
        return models

    # 单模型
    default_ckpt = os.path.join(cfg.ckpt_dir, 'best.pth')
    print(f'Loading single model: {default_ckpt}')
    model = build_model().to(device)
    ckpt = torch.load(default_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return [(model, 1.0)]


def predict_single_scale(model, img_tensor, target_size, device, use_flip_tta=True):
    """单尺度正方形 resize 推理。

    将图像 resize 到 target_size×target_size（正方形），
    模型推理后 resize 回原尺寸。
    可选翻转 TTA（hflip + vflip + diagonal）。

    Args:
        model: 分割模型
        img_tensor: (1, 3, H, W) 原始尺寸 tensor
        target_size: int, 推理时的正方形边长
        device: torch.device
        use_flip_tta: bool, 是否使用翻转 TTA

    Returns:
        prob_map: (H, W, C) numpy array
    """
    _, _, H, W = img_tensor.shape
    num_classes = cfg.num_classes

    infer_h = target_size
    infer_w = target_size

    img_resized = F.interpolate(
        img_tensor, size=(infer_h, infer_w),
        mode='bilinear', align_corners=False
    )

    flips = [(False, False)]
    if use_flip_tta:
        flips += [(True, False), (False, True), (True, True)]

    prob_sum = torch.zeros(infer_h, infer_w, num_classes, device=device)

    for hflip, vflip in flips:
        x = img_resized
        if hflip:
            x = torch.flip(x, dims=[3])
        if vflip:
            x = torch.flip(x, dims=[2])

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)  # (1, C, ts, ts)

        # 翻转回去
        if hflip:
            probs = torch.flip(probs, dims=[3])
        if vflip:
            probs = torch.flip(probs, dims=[2])

        prob_sum += probs[0].permute(1, 2, 0)  # (ts, ts, C)

    # 平均所有翻转组合
    prob_map = prob_sum / len(flips)  # (ts, ts, C)

    # resize 回原始尺寸
    prob_t = prob_map.permute(2, 0, 1).unsqueeze(0)  # (1, C, ts, ts)
    prob_t = F.interpolate(
        prob_t, size=(H, W), mode='bilinear', align_corners=False
    )  # (1, C, H, W)
    return prob_t[0].permute(1, 2, 0).cpu().numpy()  # (H, W, C)


@torch.no_grad()
def multi_scale_predict(models, img_tensor, device):
    """多尺寸加权融合推理。

    对 cfg.infer_sizes 中每个尺寸分别推理，按 cfg.infer_weights 加权融合概率图。

    Args:
        models: list of (model, weight) tuples（模型级集成权重）
        img_tensor: (1, 3, H, W) tensor
        device: torch.device

    Returns:
        pred: (H, W) numpy array, 类别预测
    """
    _, _, H, W = img_tensor.shape
    sizes = cfg.infer_sizes
    weights = cfg.infer_weights
    num_classes = cfg.num_classes

    total_prob = np.zeros((H, W, num_classes), dtype=np.float32)

    for model, model_weight in models:
        for size, size_weight in zip(sizes, weights):
            prob = predict_single_scale(
                model, img_tensor, target_size=size,
                device=device, use_flip_tta=cfg.tta_flips,
            )  # (H, W, C) numpy
            w = model_weight * size_weight
            total_prob += prob * w

    # argmax 取类别
    return total_prob.argmax(axis=2).astype(np.uint8)


def main():
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- 加载模型 ----------
    use_ensemble = True
    models = load_models(
        device,
        ensemble_configs=cfg.ensemble_models if use_ensemble else None,
        use_ensemble=use_ensemble,
    )
    print(f'Ensemble: {len(models)} model(s)')

    # 显示推理配置
    print(f'Infer sizes: {cfg.infer_sizes}')
    print(f'Infer weights: {cfg.infer_weights}')
    print(f'TTA flips: {cfg.tta_flips}')
    total_inferences = len(models) * len(cfg.infer_sizes) * (4 if cfg.tta_flips else 1)
    print(f'Total forward passes per image: {total_inferences}')

    # ---------- 预处理 transform ----------
    val_transform = get_val_transform()

    # ---------- 推理 ----------
    test_files = sorted(glob.glob(os.path.join(cfg.test_img, '*.jpg')))
    print(f'Test images: {len(test_files)}')

    predictions = []
    names = []

    t0 = time.time()
    for path in tqdm(test_files, desc='Predicting'):
        name = os.path.basename(path)
        img = load_image(path)

        augmented = val_transform(image=img)
        img_tensor = augmented['image'].unsqueeze(0).to(device)

        # 多尺寸融合推理（直接 resize，无滑窗，无后处理）
        pred = multi_scale_predict(models, img_tensor, device)
        predictions.append(pred)
        names.append(name)

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed:.1f}s, avg {elapsed/len(test_files):.2f}s/image')

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