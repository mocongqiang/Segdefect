"""训练日志：记录每 epoch 的 loss + 各类别 IoU + mIoU 到 CSV。

输出目录：output/logs/YYYY-MM-DD_HH-MM-SS/train_log.csv
同时保存 config 快照 → config_snapshot.json
"""
import os
import csv
import json
import time
from datetime import datetime


class TrainLogger:
    """训练日志管理。

    用法：
        logger = TrainLogger(cfg, tag='train')
        for epoch in range(cfg.epochs):
            ...
            logger.log_epoch(epoch, avg_loss, iou_dict, miou, lr)
    """

    def __init__(self, cfg, tag='train'):
        self.cfg = cfg
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(cfg.log_dir, f'{tag}_{ts}')
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, 'train_log.csv')
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.csv_file)
        # 表头
        self.writer.writerow([
            'epoch', 'loss',
            'bg_iou', 'oil_iou', 'stain_iou', 'scratch_iou',
            'miou', 'lr', 'time_elapsed(s)'
        ])
        self.csv_file.flush()

        # 保存 config 快照
        cfg_snapshot = {}
        for k, v in cfg.__dict__.items():
            if not k.startswith('_'):
                try:
                    json.dumps(v)  # 只保存可序列化的
                    cfg_snapshot[k] = v
                except (TypeError, ValueError):
                    cfg_snapshot[k] = str(v)
        snap_path = os.path.join(self.log_dir, 'config_snapshot.json')
        with open(snap_path, 'w', encoding='utf-8') as f:
            json.dump(cfg_snapshot, f, indent=2, ensure_ascii=False)

        self.t0 = time.time()

        print(f'[Logger] Logging to {self.log_dir}')
        print(f'[Logger]   CSV: {self.csv_path}')

    def log_epoch(self, epoch, avg_loss, iou_dict, miou, lr):
        """写入一行 epoch 记录。"""
        elapsed = time.time() - self.t0
        row = [
            epoch + 1,
            f'{avg_loss:.4f}',
            f'{iou_dict.get(0, 0):.4f}',
            f'{iou_dict.get(1, 0):.4f}',
            f'{iou_dict.get(2, 0):.4f}',
            f'{iou_dict.get(3, 0):.4f}',
            f'{miou:.4f}',
            f'{lr:.6f}',
            f'{elapsed:.0f}',
        ]
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()
        print(f'[Logger] Log saved to {self.csv_path}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()