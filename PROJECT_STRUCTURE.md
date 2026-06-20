# Segdefect — 项目结构与模块说明

基于语义分割的工业缺陷检测项目，使用 DeepLabV3Plus + EfficientNet-B5 实现对 Oil（油渍）、Stain（污点）、Scratch（划痕）三种缺陷的像素级分割。

---

## 项目概况

| 属性 | 值 |
|---|---|
| 任务 | 多类别语义分割 (4 类：BG + 3 缺陷) |
| 模型 | DeepLabV3Plus + timm-efficientnet-b5 (encoder) |
| 损失函数 | Focal Loss + Dice Loss + Lovasz-Softmax |
| 数据增强 | Albumentations (RandomScale, Flip, Rotate90, ElasticTransform, ColorJitter, CoarseDropout 等) |
| 数据策略 | CopyPaste (缺陷粘贴增强)、WeightedRandomSampler (类别平衡) |
| 训练设备 | GPU (CUDA)，支持 AMP 混合精度 |
| 框架 | PyTorch + segmentation_models_pytorch |

---

## 目录结构

```
Segdefect/
├── PROJECT_STRUCTURE.md          # 本文件 — 项目结构说明
├── data/
│   ├── test/
│   │   ├── images/               # 测试集图像 (.jpg)
│   │   │   ├── test_0001.jpg
│   │   │   ├── test_0002.jpg
│   │   │   └── ...               （约 1000+ 张）
│   │   └── sample_submission.csv # 提交样例
│   └── train/
│       ├── images/               # 训练集图像 (.jpg)
│       │   ├── train_0001.jpg
│       │   ├── train_0002.jpg
│       │   └── ...               （约 1000 张, 816 用于训练）
│       └── masks/                # 训练集标注 (.png)
│           ├── train_0001.png
│           ├── train_0002.png
│           └── ...               （约 1000 张）
├── output/
│   ├── checkpoints/              # 模型检查点
│   ├── logs/                     # 训练日志
│   └── submissions/              # 预测结果提交文件
└── project/
    ├── config.py                 # 全局配置
    ├── data.py                   # 数据加载、Mask 转换、数据增强、CopyPaste
    ├── model.py                  # 模型定义
    ├── losses.py                 # 损失函数
    ├── metrics.py                # 评估指标
    ├── train.py                  # 训练主脚本
    ├── predict.py                # 推理 & 提交生成
    ├── validate_pipeline.py      # 本地快速验证流程
    ├── rle.py                    # RLE 编解码
    ├── utils.py                  # 工具函数 (种子、EMA、调度器、滑窗推理)
    ├── logger.py                 # 训练日志记录
    └── requirement.txt           # 依赖包列表
```

---

## 数据格式说明

### 图像

- 训练集图片：`data/train/images/*.jpg`
- 测试集图片：`data/test/images/*.jpg`
- 格式：RGB 三通道 JPG

### 标注 (Mask)

- 路径：`data/train/masks/*.png`
- 格式：PNG，支持两种标注模式：
  1. **RGB 标注图** — 每种缺陷用特定颜色标示
     - Oil (类别 1)：`(128, 0, 0)` — 深红色
     - Stain (类别 2)：`(0, 128, 0)` — 深绿色
     - Scratch (类别 3)：`(128, 128, 0)` — 橄榄色
     - Background (类别 0)：`(0, 0, 0)` — 黑色
  2. **灰度标注图** — 用灰度值标示
     - Oil：灰度值 38
     - Stain：灰度值 75
     - Scratch：灰度值 113
     - Background：灰度值 0

---

## 各模块详细说明

### 1. `config.py` — 全局配置

定义 `Config` 类，全局实例 `cfg`。

| 分类 | 参数 | 说明 |
|---|---|---|
| **路径** | `data_root` | `'../data'` |
| | `train_img` | `'../data/train/images'` |
| | `train_mask` | `'../data/train/masks'` |
| | `test_img` | `'../data/test/images'` |
| | `out_root` | `'../output'` |
| | `ckpt_dir` | `'../output/checkpoints'` |
| | `log_dir` | `'../output/logs'` |
| | `sub_dir` | `'../output/submissions'` |
| **数据** | `num_classes` | 4（BG, Oil, Stain, Scratch） |
| | `defect_classes` | `[1, 2, 3]` |
| | `class_names` | `['Background', 'Oil', 'Stain', 'Scratch']` |
| | `img_hw` | `(1080, 1920)` |
| | `rgb_map` / `gray_map` | 官方颜色→类别映射 |
| **划分** | `val_ratio` | 0.15 (15% 验证) |
| | `seed` | 42 |
| **训练** | `crop_sizes` | `[512, 640, 768]` — 多尺度随机裁剪 |
| | `batch_size` | 4 (768²+B5 约 12GB，24GB 显卡安全) |
| | `num_workers` | 12 |
| | `grad_accum_steps` | 2 (有效 batch = 4×2 = 8) |
| | `epochs` | 120 |
| | `lr` | 1e-4 |
| | `weight_decay` | 1e-4 |
| | `warmup_epochs` | 5 (Cosine warmup) |
| | `grad_clip` | 1.0 |
| | `amp` | True (自动混合精度) |
| **损失** | `w_focal` / `w_dice` / `w_lovasz` | 0.4 / 0.3 / 0.3 (epoch≥5 启用 Lovasz) |
| | `class_weights` | `[0.1, 1.0, 5.0, 3.0]` (Stain 权重最高) |
| **EMA** | `ema_decay` | 0.999 |
| **CopyPaste** | `copypaste_prob` | 0.3 (30% 概率贴缺陷) |
| **模型** | `arch` | `'DeepLabV3Plus'` |
| | `encoder` | `'timm-efficientnet-b5'` |
| | `encoder_weights` | `'imagenet'` |
| **推理** | `infer_crop` | 1024 |
| | `infer_overlap` | 256 |
| | `tta_flips` | True (hflip + vflip TTA) |
| **RLE** | `rle_index_start` | 1 |
| | `rle_order` | `'C'` (行优先) |

---

### 2. `data.py` — 数据加载 & 增强

| 函数/类 | 说明 |
|---|---|
| `load_mask(path)` | 读取 PNG 标注，自动识别 RGB/灰度格式，转为 `[0,1,2,3]` 类别 ID 数组 |
| `load_image(path)` | 读取 JPG 图像 (RGB) |
| `get_split()` | 划分训练/验证集 (85%/15%)，返回 `(img_path, mask_path)` 元组列表 |
| `get_sample_weights(files)` | 按缺陷类别稀有度计算采样权重 (Stain×5, Scratch×2, Oil×1.5) |
| `DefectBank` | 预提取缺陷块，随机贴到其他图上 (CopyPaste) |
| `DefectBank.paste(img, mask)` | 边界羽化 + 膨胀平滑的缺陷粘贴 |
| `get_train_transform(crop_size)` | 训练增强流水线 (RandomScale→Crop→Flip→Rotate→Elastic→Noise→Blur→ColorJitter→Dropout→Normalize) |
| `get_val_transform()` | 验证仅 Normalize |
| `_pad_to_divisor(img, mask, divisor)` | 将图像 pad 到 32 倍数 (右下侧补零) |
| `collate_fn(batch)` | 将不同 crop_size 样本 align 到 batch 内最大尺寸 |
| `DefectDataset` | PyTorch Dataset，支持 train/val 模式 (val 模式自动 pad) |
| `TestDataset` | 测试集 Dataset (自动 pad + 匹配 `*.jpg`) |
| `make_sampler(files)` | WeightedRandomSampler 工厂函数 |

> ✅ **已修复**：Mask 路径后缀问题 — `get_split()` 中 `to_pair()` 已改为 `os.path.splitext(...)[0] + '.png'`。

---

### 3. `model.py` — 模型定义

```python
build_model() → DeepLabV3Plus / UnetPlusPlus / MAnet
```

- 默认架构：**DeepLabV3Plus** + **timm-efficientnet-b5** (ImageNet 预训练)
- 备选架构：UnetPlusPlus (更强的跳跃连接，适合小缺陷)、MAnet
- 使用 `segmentation-models-pytorch` 库
- 输入：RGB 3 通道，输出：4 类 logits

---

### 4. `losses.py` — 损失函数

| 类 | 说明 |
|---|---|
| `FocalLoss(alpha, gamma)` | Focal Loss，带类别权重 `alpha` |
| `DiceLoss(smooth)` | 多类别 Dice Loss |
| `LovaszSoftmax` | Lovasz-Softmax Loss，直接优化 mIoU (跳过背景类) |
| `CombinedLoss` | 组合：`0.4×Focal + 0.3×Dice + 0.3×Lovasz` (Lovasz 在 epoch≥5 后启用) |

---

### 5. `metrics.py` — 评估指标

| 类 | 说明 |
|---|---|
| `IoUMetric` | 累积预测结果，计算 per-class IoU 和缺陷 mIoU (仅 Oil+Stain+Scratch) |

---

### 6. `train.py` — 训练脚本

**训练流程：**

1. 加载数据划分 → 构建 DefectBank
2. 创建 DataLoader（带 WeightedRandomSampler）
3. 构建模型、优化器 (AdamW)、调度器 (CosineWarmup)、AMP GradScaler、EMA
4. 逐 epoch 训练：
   - `train_one_epoch()` — 训练 + CopyPaste 增强
   - EMA 权重验证 → 计算 mIoU
   - 保存最佳模型 (`output/checkpoints/best.pth`)
   - 每 20 epoch 保存一次 checkpoint

**使用方式：**
```bash
cd project && python train.py
```

---

### 7. `predict.py` — 推理 & 提交

**推理流程：**

1. 加载 `best.pth` 模型
2. 遍历测试集图片 → 滑窗推理 (1024 crop, 256 overlap) + TTA (hflip+vflip)
3. 生成 RLE 编码 → 输出 `output/submissions/submission.csv`

> ✅ **已修复**：测试集匹配改为 `*.jpg`（`TestDataset` 类已自动匹配）。

**使用方式：**
```bash
cd project && python predict.py
```

---

### 8. `rle.py` — RLE 编解码

| 函数 | 说明 |
|---|---|
| `encode_rle(binary_mask)` | 二值 mask → RLE 字符串 (行优先, 1-indexed) |
| `decode_rle(rle_str, shape)` | RLE 字符串 → 二值 mask |
| `generate_submission(predictions, names, output_path)` | 批量生成提交 CSV (每图 3 行：Oil/Stain/Scratch) |

---

### 9. `utils.py` — 工具函数

| 函数/类 | 说明 |
|---|---|
| `set_seed(seed)` | 固定随机种子 (random, numpy, torch, cuda) |
| `EMA` | 指数移动平均 (decay=0.999) |
| `get_lr_scheduler(optimizer, epochs, warmup)` | Cosine Warmup 学习率调度 |
| `sliding_window_inference(...)` | 滑窗推理 + TTA (hflip/vflip)，返回 softmax 概率图 |

---

### 10. `requirement.txt` — 依赖

```
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.3
timm>=0.9.0
albumentations>=1.3.0
opencv-python>=4.7.0
numpy>=1.23.0
pandas>=1.5.0
Pillow>=9.0.0
tqdm>=4.64.0
```

---

## 📋 Bug 修复记录

### ✅ 已修复：Mask 路径后缀错误 (`data.py`) — 2026-06-19

**原因**：`get_split()` 中 `to_pair()` 使用 `os.path.basename(p).replace('.jpg', '.jpg')` 未替换扩展名，导致 mask 路径以 `.jpg` 结尾而非 `.png`。

**修复**：
```python
# 修复前
name = os.path.basename(p).replace('.jpg', '.jpg')

# 修复后
name = os.path.splitext(os.path.basename(p))[0] + '.png'
```

### ✅ 已修复：Lovasz-Softmax 梯度爆炸 (`losses.py`) — 2026-06-19

**原因**：`LovaszSoftmax.forward()` 排序方向反转 + 使用 `probas.sum(1)` 而非 `1-probas`，导致排序从大到小与 lovasz 期望相反，loss 不收敛或 NaN。

**修复**：`probas = 1 - probas` 使高概率类排在前面；`errors = errors` 改为直接使用 1-p 作为误差。

### ✅ 已修复：batch_size=8 OOM (`config.py`) — 2026-06-19

**原因**：768² crop + EfficientNet-B5 大编码器，batch_size=8 峰值显存 ~20GB，超出 24GB 显卡可用（含碎片）。

**修复**：
- `batch_size` 8 → 4
- `grad_accum_steps` 4 → 2（有效 batch = 4×2 = 8，不变）
- `train.py` 添加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少碎片

---

## 🟡 已知限制

| 问题 | 影响 | 状态 |
|------|------|------|
| 训练 epoch=0 时 Lovasz 不启用，需 warmup 至 epoch≥5 | 前几个 epoch 仅有 Focal+Dice | 设计如此 |
| 测试集图片名称格式未知，`TestDataset` 假设 `*.jpg` | 若实际为 `.png` 则匹配失败 | 待实际测试集确认 |
| 滑窗推理 crop=1024, overlap=256，边缘区域可能未覆盖小缺陷 | 边缘检测精度略降 | 可调参数 |

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r project/requirement.txt

# 2. 训练
cd project
python train.py

# 3. 推理
python predict.py
```

---

> **最后更新**：2026-06-19