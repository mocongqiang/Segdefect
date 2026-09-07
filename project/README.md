# project/ — 代码目录说明

本目录为 Segdefect 项目的全部源码。所有脚本通过 `config.py` 自动定位仓库根目录
（`../data` 与 `../output`），因此在 `project/` 下直接 `python xxx.py` 运行即可。

---

## 文件一览

| 文件 | 职责 |
|---|---|
| `config.py` | 全局唯一配置 `cfg`（路径/超参/伪标签/推理/集成），并自动创建 output 子目录 |
| `data.py` | 数据加载与增强：`load_mask`（RGB/灰度/伪标签三种 Mask）、`load_image`、`defect_aware_crop` 缺陷感知裁剪、`DefectBank` CopyPaste、`get_split` 划分、`DefectDataset`/`TestDataset`、`collate_fn`（跨尺寸对齐） |
| `model.py` | `Mask2FormerSeg`：加载 Swin-Large 语义分割预训练，冻结早期层；`einsum` 低分辨率合成 + 统一上采样省显存；`build_model` / `build_model_custom` 支持 Mask2Former 与 smp 系列 |
| `losses.py` | `FocalLoss`（类别加权）、`DiceLoss`（仅缺陷类）、`LovaszSoftmax`、`BoundaryLoss`、`CombinedLoss`（加权组合 + 延迟/ramp 调度） |
| `metrics.py` | `IoUMetric`：逐类 IoU 与缺陷 mIoU，忽略 `255` 像素 |
| `train.py` | 训练入口：AMP + 梯度累积 + CosineWarmup + EarlyStopping + 断点续训，保存 `best.pth`/`last.pth` |
| `predict.py` | 推理 + 提交：直接 resize 多尺度融合 + 翻转 TTA + 模型集成 → `output/submissions/submission.csv` |
| `generate_pseudo_labels.py` | 用 `best.pth` 对测试集滑窗+多尺度 TTA 推理，`max_prob≥0.95` 的像素生成伪标签（其余 255），供自训练 |
| `validate_pipeline.py` | 快速验证：覆盖为 resnet18 + 6 epoch + 小 crop，跑通 训练→验证→推理→提交 全链路 |
| `rle.py` | `encode_rle`/`decode_rle`/`generate_submission`（F-order、1-indexed，空掩码 `0 0`） |
| `utils.py` | `set_seed`、`EMA`、`get_lr_scheduler`、`sliding_window_inference`（批量 patch + AMP + 翻转/多尺度 TTA） |
| `logger.py` | 训练 CSV 日志 + config 快照（`output/logs/<tag>_<时间戳>/`） |
| `requirement.txt` | Python 依赖清单 |
| `log.txt` / `log10.txt` | 运行日志备份（非主流程） |

---

## 运行方式

```bash
# 训练（Mask2Former Swin-Large）
python train.py

# 推理 + 生成提交
python predict.py

# 伪标签自训练数据生成
python generate_pseudo_labels.py

# 快速流程验证（轻量）
python validate_pipeline.py
```

> 提示：首次运行 `train.py` 需联网下载 HuggingFace 预训练权重
> （`facebook/mask2former-swin-large-cityscapes-semantic`）；`config.py` 已内置国内镜像。

---

## 代码调用关系

```
train.py ──▶ data.get_split / DefectBank / DefectDataset
         ──▶ model.build_model（Mask2FormerSeg）
         ──▶ losses.CombinedLoss
         ──▶ metrics.IoUMetric（验证）
predict.py ──▶ data / model.build_model_custom / rle.generate_submission
generate_pseudo_labels.py ──▶ utils.sliding_window_inference（多尺度+翻转 TTA）
validate_pipeline.py ──▶ train + 滑窗验证 + 推理 全流程演示
```

---

## 关键约定（改代码前必读）

1. **Mask 类别**：`0=BG, 1=Oil, 2=Stain, 3=Scratch`；`255` = ignore（伪标签低置信度）。
2. **loss 接口**：`CombinedLoss(seg_logits, cls_logits, target, cls_target)` —— 分类分支已移除，调用时传 `None`。
3. **Mask2Former 前向**：`model(x)` 直接返回 `(B,4,H,W)` logits（低分辨率合成后再上采样，显存友好）。
4. **训练冻结策略**：`Mask2FormerSeg.__init__` 冻结 encoder 的 `embeddings / stages.0 / stages.1`，其余可训。
5. **推理与训练 crop 不一致**：`train.py` 用 768/896 裁剪；`predict.py` 直接整图 resize；滑窗（1024/256）仅用于伪标签与验证脚本。
6. **断点续训**：最终版 `cfg.resume_from=None`（无兼容旧权重，从头训练）；训练后自动保存 `best.pth`，需续训时改回该路径。`config.py` 的 `_root` 基于文件位置计算 —— 移动目录后无需手动改路径。

---

## 显存参考

- 本地 RTX 4050 6GB：`batch_size=2 + grad_accum=4 + crop 768/896 + AMP` 可运行（多尺度 768/896 交替）。
- 服务器（大显存）：可调大 `batch_size` / `crop_sizes` / 增加 `infer_sizes`。
- OOM 时优先：`batch_size 2→1`、`crop_sizes` 移除 896、关闭 `amp` 前先确认驱动支持。