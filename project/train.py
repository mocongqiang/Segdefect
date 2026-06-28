"""
训练入口
DeepLabV3+ Final Version

支持：
✔ AMP
✔ Cosine LR
✔ Warmup
✔ EarlyStopping
✔ Resume
✔ Defect-aware Crop
✔ CopyPaste
✔ 多尺度训练
✔ 最佳模型保存
"""

import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from tqdm import tqdm

from config import cfg

from data import (
    get_split,
    DefectDataset,
    DefectBank,
    collate_fn,
)

from model import build_model
from losses import CombinedLoss
from metrics import IoUMetric
from utils import (
    set_seed,
    get_lr_scheduler,
)
def evaluate(model, loader, criterion, device):
    """
    验证一个 epoch
    """

    model.eval()

    metric = IoUMetric()

    total_loss = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)

            dummy_cls = None
            dummy_target = None

            loss = criterion(
                logits,
                dummy_cls,
                masks,
                dummy_target
            )

            total_loss += loss.item()

            pred = logits.argmax(dim=1)

            metric.update(pred, masks)

    iou, miou = metric.compute()

    avg_loss = total_loss / len(loader)

    return avg_loss, iou, miou


def train():

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------
    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Device :", device)
    print("Model  :", cfg.arch)
    print("Encoder:", cfg.encoder)
    print("=" * 60)


    # --------------------------------------------------------
    # 数据集划分
    # --------------------------------------------------------
    train_files, val_files = get_split()

    print(f"Train : {len(train_files)}")
    print(f"Val   : {len(val_files)}")


    # --------------------------------------------------------
    # Defect Bank
    # --------------------------------------------------------
    defect_bank = DefectBank(train_files)


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    train_dataset = DefectDataset(
        train_files,
        mode="train",
        defect_bank=defect_bank
    )

    val_dataset = DefectDataset(
        val_files,
        mode="val"
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=cfg.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = build_model().to(device)

    print(model.__class__.__name__)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    criterion = CombinedLoss().to(device)


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------
    scheduler = get_lr_scheduler(
        optimizer,
        cfg.epochs,
        cfg.warmup_epochs
    )


    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------
    scaler = GradScaler(enabled=cfg.amp)


    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------
    start_epoch = 0

    if cfg.resume_from is not None:

        print(f"Resume from {cfg.resume_from}")

        ckpt = torch.load(
            cfg.resume_from,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(ckpt["model_state"])

        optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )

        scheduler.load_state_dict(
            ckpt["scheduler_state"]
        )

        scaler.load_state_dict(
            ckpt["scaler_state"]
        )

        start_epoch = ckpt["epoch"] + 1


    # --------------------------------------------------------
    # EarlyStopping
    # --------------------------------------------------------
    best_miou = 0.0

    patience = 0

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

        # ==========================================================
    # Train Loop
    # ==========================================================
    for epoch in range(start_epoch, cfg.epochs):

        print(f"\nEpoch [{epoch + 1}/{cfg.epochs}]")

        model.train()

        criterion.set_epoch(epoch)

        running_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            ncols=120
        )

        for step, (images, masks) in pbar:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast("cuda", enabled=cfg.amp):

                logits = model(images)

                loss = criterion(
                    logits,
                    None,
                    masks,
                    None
                )

                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            # ------------------------------
            # Gradient Accumulation
            # ------------------------------
            if (
                (step + 1) % cfg.grad_accum_steps == 0
                or
                (step + 1) == len(train_loader)
            ):

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.grad_clip
                )

                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad(set_to_none=True)

            running_loss += (
                loss.item() * cfg.grad_accum_steps
            )

            pbar.set_postfix(
                loss=f"{running_loss/(step+1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        scheduler.step()

        train_loss = running_loss / len(train_loader)

            # =====================================================
        # Validation
        # =====================================================
        val_loss, iou, miou = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"\nEpoch {epoch+1}/{cfg.epochs}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}"
        )

        print(
            f"Oil: {iou[1]:.4f}"
            f" | Stain: {iou[2]:.4f}"
            f" | Scratch: {iou[3]:.4f}"
            f" | mIoU: {miou:.4f}"
        )


        # =====================================================
        # Save Best Model
        # =====================================================
        if miou > best_miou + cfg.early_stop_min_delta:

            best_miou = miou
            patience = 0

            save_path = os.path.join(
                cfg.ckpt_dir,
                "best.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_miou": best_miou,
                },
                save_path
            )

            print(f"Best model saved -> {save_path}")

        else:

            patience += 1

            print(
                f"EarlyStopping "
                f"{patience}/{cfg.early_stop_patience}"
            )


        # =====================================================
        # Save Last
        # =====================================================
        last_path = os.path.join(
            cfg.ckpt_dir,
            "last.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_miou": best_miou,
            },
            last_path
        )


        # =====================================================
        # Early Stop
        # =====================================================
        if patience >= cfg.early_stop_patience:

            print("\nEarly Stopping Triggered.")

            break


    print("\nTraining Finished.")
    print(f"Best mIoU : {best_miou:.4f}")


if __name__ == "__main__":

    train()

