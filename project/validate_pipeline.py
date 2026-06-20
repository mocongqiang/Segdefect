"""Pipeline 快速验证：3 epoch 训练 + 推理生成提交文件（~15 分钟）

目的：在本地确认数据/模型/训练/推理全流程无 bug 后，再迁移到服务器完整训练。
运行：python validate_pipeline.py

服务器完整训练：python train.py（RTX 3090 24GB, batch_size=8, 120 epoch）
"""
import os, sys, time, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- 临时覆盖配置：快速验证用 ----
from config import cfg

cfg.epochs           = 3
cfg.crop_sizes       = [512]                          # 只用最小 crop
cfg.batch_size       = 2
cfg.grad_accum_steps = 2
cfg.num_workers      = 0                              # Windows 上 0 最快
cfg.amp              = False
cfg.encoder          = 'timm-efficientnet-b1'          # 轻量编码器, 速度 3x
cfg.infer_crop       = 512                            # 验证/推理用 crop, 与训练一致
cfg.infer_overlap    = 128
# ----------------------------------

os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)

from config import cfg as _cfg   # noqa
from data import get_split, DefectDataset, DefectBank, make_sampler, collate_fn
from model import build_model
from losses import CombinedLoss
from metrics import IoUMetric
from utils import set_seed, EMA, get_lr_scheduler, sliding_window_inference
from logger import TrainLogger

# ---------- 轻量训练增强 (去掉重 CPU 变换) ----------
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_lightweight_train_transform(crop_size):
    """本地快速验证用：只保留 Flip/Rotate/ColorJitter，跳过重 CPU 增强。"""
    return A.Compose([
        A.RandomScale(scale_limit=(-0.3, 0.5), p=0.5),
        A.PadIfNeeded(min_height=crop_size, min_width=crop_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CoarseDropout(num_holes_range=(2, 8),
                        hole_height_range=(16, 64),
                        hole_width_range=(16, 64), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


# ---------- 验证：滑动窗口推理 ----------
@torch.no_grad()
def evaluate_sliding(model, loader, device, metric):
    """用滑动窗口验证 (与推理一致, 与训练 crop 尺寸匹配)。"""
    model.eval()
    metric.reset()
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Val', leave=False):
            masks = masks.to(device)
            for i in range(imgs.size(0)):
                single_img = imgs[i:i+1].to(device)
                prob = sliding_window_inference(
                    model, single_img, crop_size=cfg.infer_crop,
                    overlap=cfg.infer_overlap, device=device,
                    tta=False, num_classes=cfg.num_classes)
                pred = torch.from_numpy(prob.argmax(axis=2)).unsqueeze(0).to(device)
                metric.update(pred, masks[i:i+1])
    return metric.compute()


# ---------- 训练一个 epoch (无 amp) ----------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, ema):
    model.train()
    criterion.set_epoch(epoch)
    total_loss = 0.0
    n = 0
    accum = cfg.grad_accum_steps
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{cfg.epochs}')
    for step, (imgs, masks) in enumerate(pbar):
        imgs  = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks) / accum
        loss.backward()
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(model)
        total_loss += loss.item() * accum * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix(loss=f'{total_loss/n:.4f}')
    if (step + 1) % accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        ema.update(model)
    return total_loss / n


# ---------- 主流程 ----------
def main():
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Config: encoder={cfg.encoder}, epochs={cfg.epochs}, crop={cfg.crop_sizes}, '
          f'batch={cfg.batch_size}, workers={cfg.num_workers}, amp={cfg.amp}')

    # ==== 1. 数据加载 ====
    print('\n[1/5] Loading data ...')
    train_files, val_files = get_split()
    # 本地快速验证：只用前 200 张训练集
    train_files = train_files[:200]
    print(f'  Train: {len(train_files)} (subset), Val: {len(val_files)}')

    print('[1/5] Building defect bank ...')
    db = DefectBank(train_files)

    # 注入轻量 transform
    train_ds = DefectDataset(train_files, mode='train', defect_bank=db)
    train_ds.transforms = {cs: get_lightweight_train_transform(cs) for cs in cfg.crop_sizes}
    val_ds   = DefectDataset(val_files,   mode='val')
    sampler  = make_sampler(train_files)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            collate_fn=collate_fn)
    print('  OK – DataLoader created.')

    # ==== 2. 模型 + 损失 + 优化器 ====
    print('\n[2/5] Building model ...')
    model = build_model().to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = get_lr_scheduler(optimizer, cfg.epochs, cfg.warmup_epochs)
    ema = EMA(model, decay=cfg.ema_decay)
    print(f'  OK – {cfg.arch} + {cfg.encoder}, '
          f'{sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

    # ==== 日志 ====
    logger = TrainLogger(cfg, tag='validate')

    # ==== 3. 训练 3 epoch ====
    print(f'\n[3/5] Training {cfg.epochs} epochs ...')
    best_miou = 0.0
    metric = IoUMetric(num_classes=cfg.num_classes)
    t0 = time.time()

    for epoch in range(cfg.epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion,
                                   optimizer, device, epoch, ema)
        scheduler.step()

        backup = {k: v.clone() for k, v in model.state_dict().items()}
        ema.apply_shadow(model)
        iou_dict, val_miou = evaluate_sliding(model, val_loader, device, metric)
        ema.restore(model, backup)

        current_lr = scheduler.get_last_lr()[0]
        logger.log_epoch(epoch, avg_loss, iou_dict, val_miou, current_lr)

        print(f'  Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f} | {metric}')

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_miou': val_miou},
                       os.path.join(cfg.ckpt_dir, 'best.pth'))

    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.0f}s, Best mIoU: {best_miou:.4f}')

    # ==== 4. 加载 best model ====
    print('\n[4/5] Loading best checkpoint ...')
    ckpt = torch.load(os.path.join(cfg.ckpt_dir, 'best.pth'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f'  OK – epoch {ckpt["epoch"]+1}, val_mIoU={ckpt["val_miou"]:.4f}')

    # ==== 5. 推理 + 生成提交文件 ====
    print('\n[5/5] Running inference & generating submission ...')
    import glob
    from data import load_image, get_val_transform
    from rle import generate_submission

    test_files = sorted(glob.glob(os.path.join(cfg.test_img, '*.jpg')))
    print(f'  Test images: {len(test_files)}')
    model.eval()

    predictions, names = [], []
    val_t = get_val_transform()

    for path in tqdm(test_files, desc='  Infer'):
        name = os.path.basename(path)
        img = load_image(path)
        aug = val_t(image=img)
        img_t = aug['image'].unsqueeze(0).to(device)
        prob = sliding_window_inference(
            model, img_t, crop_size=cfg.infer_crop,
            overlap=cfg.infer_overlap, device=device,
            tta=cfg.tta_flips, num_classes=cfg.num_classes)
        predictions.append(prob.argmax(axis=2).astype(np.uint8))
        names.append(name)

    out_path = os.path.join(cfg.sub_dir, 'submission_validate.csv')
    df = generate_submission(predictions, names, out_path)
    non_empty = (df['rle'] != '0 0').sum()

    print(f'\n  Submission saved to: {out_path}')
    print(f'  Total rows: {len(df)}  (expected {len(test_files)*3})')
    print(f'  Non-empty RLE: {non_empty}/{len(df)}  '
          f'({"✅ Has detections" if non_empty > 0 else "⚠  All empty – check model"})')

    logger.close()

    print('\n' + '='*60)
    print('Pipeline validation PASSED ✅')
    print(f'Best mIoU: {best_miou:.4f} | Total time: {time.time()-t0:.0f}s')
    print('Ready to deploy to server for full training.')
    print('='*60)


if __name__ == '__main__':
    main()