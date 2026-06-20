"""推理 + 生成提交文件"""
import os, glob, time
import numpy as np
import torch
from tqdm import tqdm

from config import cfg
from data import TestDataset, load_image
from model import build_model
from rle import generate_submission
from utils import set_seed, sliding_window_inference


def main():
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- 加载模型 ----------
    model = build_model().to(device)
    ckpt_path = os.path.join(cfg.ckpt_dir, 'best.pth')
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # ---------- 推理 ----------
    test_files = sorted(glob.glob(os.path.join(cfg.test_img, '*.jpg')))
    print(f'Test images: {len(test_files)}')

    predictions = []
    names = []

    for path in tqdm(test_files, desc='Predicting'):
        name = os.path.basename(path)
        img = load_image(path)
        H, W = img.shape[:2]

        # 预处理
        from data import get_val_transform
        t = get_val_transform()
        augmented = t(image=img)
        img_tensor = augmented['image'].unsqueeze(0).to(device)  # (1,3,H,W)

        # 滑窗推理
        prob_map = sliding_window_inference(
            model, img_tensor,
            crop_size=cfg.infer_crop,
            overlap=cfg.infer_overlap,
            device=device,
            tta=cfg.tta_flips,
            num_classes=cfg.num_classes,
        )   # (H, W, C)

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