"""训练脚本

服务器完整训练（RTX 3090 24GB）：
  修改 config.py → batch_size=8, num_workers=8, grad_accum_steps=1
  运行：python train.py
本地快速验证：
  运行：python validate_pipeline.py
"""
import os, sys, time, copy

# 减少 CUDA 内存碎片（OOM 时推荐开启）
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import cfg
from data import get_split, DefectDataset, DefectBank, make_sampler, collate_fn
from model import build_model
from losses import CombinedLoss
from metrics import IoUMetric
from utils import set_seed, EMA, get_lr_scheduler
from logger import TrainLogger

def evaluate(model, loader, device, metric):
    """验证：val_transform 已 pad 到 32 倍数，直接 forward 即可。"""
    model.eval()
    metric.reset()
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Val', leave=False):
            imgs  = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            pred = logits.argmax(dim=1)
            metric.update(pred, masks)
    return metric.compute()


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, epoch, ema):
    model.train()
    criterion.set_epoch(epoch)
    total_loss = 0.0
    n = 0

    accum_steps = cfg.grad_accum_steps
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{cfg.epochs}')
    for step, (imgs, masks) in enumerate(pbar):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast('cuda', enabled=cfg.amp):
            logits = model(imgs)
            loss = criterion(logits, masks) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

        total_loss += loss.item() * accum_steps * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix(loss=f'{total_loss/n:.4f}')

    # 处理最后不完整的累积步
    if (step + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update(model)

    return total_loss / n


def main():
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- 数据 ----------
    print('Loading data...')
    train_files, val_files = get_split()
    print(f'  Train: {len(train_files)}, Val: {len(val_files)}')

    print('Building defect bank for CopyPaste...')
    defect_bank = DefectBank(train_files)

    train_ds = DefectDataset(train_files, mode='train', defect_bank=defect_bank)
    val_ds   = DefectDataset(val_files,   mode='val')

    sampler = make_sampler(train_files)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=2,
                              shuffle=False, num_workers=cfg.num_workers,
                              pin_memory=True,
                              collate_fn=collate_fn)

    # ---------- 模型 ----------
    model = build_model().to(device)

    # ---------- 损失 / 优化器 / 调度器 ----------
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = get_lr_scheduler(optimizer, cfg.epochs, cfg.warmup_epochs)
    scaler = GradScaler('cuda', enabled=cfg.amp)

    # ---------- EMA ----------
    ema = EMA(model, decay=cfg.ema_decay)

    # ---------- 日志 ----------
    logger = TrainLogger(cfg, tag='train')

    # ---------- 训练循环 ----------
    best_miou = 0.0
    best_epoch = 0
    early_stop_counter = 0
    metric = IoUMetric(num_classes=cfg.num_classes)

    for epoch in range(cfg.epochs):
        # 训练
        avg_loss = train_one_epoch(model, train_loader, criterion,
                                   optimizer, scaler, device, epoch, ema)
        scheduler.step()

        # 用 EMA 权重验证
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        ema.apply_shadow(model)

        iou_dict, val_miou = evaluate(model, val_loader, device, metric)
        print(f'Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f} | '
              f'Val: {metric}')

        # 写日志
        current_lr = scheduler.get_last_lr()[0]
        logger.log_epoch(epoch, avg_loss, iou_dict, val_miou, current_lr)

        # 保存最佳模型
        if val_miou > best_miou + cfg.early_stop_min_delta:
            best_miou = val_miou
            best_epoch = epoch
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_miou': val_miou,
            }, os.path.join(cfg.ckpt_dir, 'best.pth'))
            print(f'  ★ Best mIoU: {best_miou:.4f}, saved.')
        else:
            early_stop_counter += 1

        # 恢复原始权重继续训练
        ema.restore(model, backup)

        # Early Stopping
        if early_stop_counter >= cfg.early_stop_patience:
            print(f'\nEarly stopping at epoch {epoch+1}: '
                  f'no improvement for {cfg.early_stop_patience} epochs. '
                  f'(Best: {best_miou:.4f} at epoch {best_epoch+1})')
            break

        # 每 20 epoch 存 checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(cfg.ckpt_dir, f'epoch_{epoch+1}.pth'))

    logger.close()
    print(f'\nTraining done. Best Val mIoU: {best_miou:.4f}')


if __name__ == '__main__':
    main()