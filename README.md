# Segdefect — 表面缺陷语义分割

> 基于 **Mask2Former（Swin-Large）** 的多类别缺陷分割项目。
> 目标：从图像中像素级分割三类缺陷 —— **Oil（油渍）、Stain（污点）、Scratch（划痕）**，
> 并按 Kaggle 风格输出 **RLE（列优先 F-order、1-indexed）** 提交文件。
>
> 数据：训练集 960 张（1920×1080），测试集 240 张。
> 当前配置：`crop [768, 896]`、`batch 2 × grad_accum 4`、`epochs 20`、`AMP`、`ema 0.996`。

---

## 1. 主要特性

| 模块 | 说明 |
|---|---|
| 模型 | `Mask2Former` + `facebook/mask2former-swin-large-cityscapes-semantic`；冻结 patch embedding 与 stage0/1，微调 stage2/3 |
| 数据增强 | 缺陷感知随机裁剪、CopyPaste 缺陷贴图（2–4 个/图）、Elastic / ColorJitter / CoarseDropout 等（albumentations） |
| 训练技巧 | AMP 混合精度、梯度累积（×4）、Cosine Warmup（3 epoch）、EMA、EarlyStopping、断点续训（`resume_from=None`，可改回 `best.pth`） |
| 损失函数 | `0.35×Focal + 0.35×Dice + 0.10×Lovasz + 0.20×Boundary`，仅优化缺陷类；Focal 使用类别权重 `[0.01, 2.0, 3.0, 2.0]` |
| 自训练 | 高置信度伪标签（`threshold=0.95`），自动混入训练集 |
| 推理 | 多尺度直接 resize 融合 `[1024, 1152, 1280] × [1, 2, 1]` + 4 向翻转 TTA + 多模型集成 |
| 滑窗推理 | 伪标签生成 / 流程验证使用（crop 1024、overlap 256、多尺度 TTA） |

> 支持多架构：`Mask2Former`、`DeepLabV3Plus`、`Segformer`、`UnetPlusPlus`、`Unet`（见 `project/model.py`）。

---

## 2. 目录结构

```
Segdefect/
├── project/                        # 全部代码（模块说明见 project/README.md）
│   ├── config.py                   # 全局配置（路径 / 超参 / 推理参数）
│   ├── data.py                     # 数据加载、Mask 转换、增强、CopyPaste、Dataset
│   ├── model.py                    # Mask2Former 包装 + smp 模型
│   ├── losses.py                   # Focal / Dice / Lovasz / Boundary / CombinedLoss
│   ├── metrics.py                  # IoUMetric（缺陷类 mIoU）
│   ├── train.py                    # 训练主脚本
│   ├── predict.py                  # 推理 + 生成 submission.csv
│   ├── generate_pseudo_labels.py   # 伪标签生成（自训练数据）
│   ├── validate_pipeline.py        # 本地快速流程验证（轻量模型，跑通全链路）
│   ├── rle.py                      # RLE 编解码 / 提交生成
│   ├── utils.py                    # seed / EMA / 调度器 / 滑窗推理
│   ├── logger.py                   # 训练 CSV 日志 + config 快照
│   └── requirement.txt             # 依赖
├── data/
│   ├── train/images/               # 960 张训练图（train_*.jpg）
│   ├── train/masks/                # 960 张标注（train_*.png）
│   ├── train/pseudo_images/        # 伪标签图像（generate_pseudo_labels 生成）
│   ├── train/pseudo_masks/         # 伪标签 mask（0/1/2/3/255，255=忽略）
│   └── test/images/                # 240 张测试图（test_*.jpg）
│   └── test/sample_submission.csv
├── output/
│   ├── checkpoints/                # best.pth / last.pth
│   ├── logs/                       # train_log.csv + config_snapshot.json
│   └── submissions/                # submission.csv
├── README.md                       # 本文件
└── 修改.md / 实验报告.md / 报告要求.md    # 课程报告 / 调参记录文档
```

---

## 3. 数据与标注

- 图像：RGB JPG，1920×1080（`cfg.img_hw`）。
- Mask 支持三种形式，`data.load_mask` 自动识别：
  - RGB 标注：`(128,0,0)`→Oil(1)、`(0,128,0)`→Stain(2)、`(128,128,0)`→Scratch(3)
  - 灰度标注：`38→1、75→2、113→3`
  - 伪标签：灰度 `0/1/2/3/255`（`255` 为低置信度忽略像素）
- 训练/验证划分：15% 验证集（`val_ratio=0.15, seed=42`）→ 约 816 训练 / 144 验证。

---

## 4. 快速开始

```bash
# 0) 环境（推荐 Python 3.10）
pip install -r project/requirement.txt

# 1)（可选）快速流程验证 —— 轻量模型跑 6 epoch，确认全链路无 bug
python project/validate_pipeline.py

# 2) 完整训练 —— Mask2Former Swin-Large，输出 output/checkpoints/{best,last}.pth
python project/train.py

# 3) 推理并生成提交文件 —— output/submissions/submission.csv
python project/predict.py

# 4)（自训练循环，可选）用 best.pth 在测试集上生成高置信度伪标签
python project/generate_pseudo_labels.py
#    生成后 data/train/pseudo_images|pseudo_masks 就绪；
#    再跑 python project/train.py 时 cfg.use_pseudo=True 会自动混入伪标签数据重新训练。
```

> 国内网络：`config.py` 已设置 `HF_ENDPOINT=https://hf-mirror.com`（解决 HuggingFace 下载 WinError 10060 超时）。

---

## 5. 训练配置速览（`project/config.py`）

| 类别 | 关键参数 |
|---|---|
| 数据 | `crop_sizes=[768, 896]`、`defect_crop_prob=1.0`、`copypaste_prob=0.8` |
| 训练 | `epochs=20`、`batch_size=2`、`grad_accum_steps=4`、`lr=5e-6`、`weight_decay=1e-4`、`warmup_epochs=3` |
| 精度 | `amp=True`、`grad_clip=1.0` |
| 损失 | `w_focal=0.35`、`w_dice=0.35`、`w_lovasz=0.10`、`w_boundary=0.20`；`class_weights=[0.01, 2.0, 3.0, 2.0]` |
| 早停/EMA | `early_stop_patience=10`、`min_delta=0.001`、`ema_decay=0.996` |
| 断点续训 | `resume_from=None`（需续训时改回 `output/checkpoints/best.pth`） |
| 推理 | `infer_sizes=[1024,1152,1280]`、`infer_weights=[1,2,1]`、`tta_flips=True`、模型集成 `ensemble_models` |

Lovasz 从 epoch 15 起线性 ramp（5 epoch），Boundary 从 epoch 10 起 ramp；Dice 不计 Background。

---

## 6. 推理与提交

- `predict.py`：直接 **resize 多尺度融合**（非滑窗），避开滑窗拼接伪影。
  每图前向次数 = 模型数 × 3 尺寸 × 4 翻转。
- 后处理默认关闭：`cls_thresholds=None`、`min_area=None`、`fill_holes=False`（阈值/面积过滤会误删真实缺陷）。
- 提交格式：每个测试图 3 行，`id = 图片名_类别`（1/2/3），RLE 列优先（F-order）、1-indexed；空 mask 编码为 `0 0`。

---

## 7. 文档与历史

- `project/README.md` —— 代码目录模块级说明（推荐开发时查阅）。
- `修改.md` / `实验报告.md` / `报告要求.md` —— 调参过程记录与课程报告文档。
- 瘦身清理说明：旧版文档 `PROJECT_STRUCTURE.md`、`log*.txt` / `train.log` / `log/` 目录及旧方案权重均已删除，仓库只保留最终版本代码与产物。