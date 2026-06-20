"""全局配置"""
import os

# HuggingFace 国内镜像（解决 WinError 10060 连接超时问题）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

class Config:
    # ===================== 路径 =====================
    data_root   = '../data'
    train_img   = os.path.join(data_root, 'train/images')
    train_mask  = os.path.join(data_root, 'train/masks')
    test_img    = os.path.join(data_root, 'test/images')
    sample_sub  = os.path.join(data_root, 'sample_submission.csv')

    out_root    = '../output'
    ckpt_dir    = os.path.join(out_root, 'checkpoints')
    log_dir     = os.path.join(out_root, 'logs')
    sub_dir     = os.path.join(out_root, 'submissions')

    # ===================== 数据 =====================
    num_classes      = 4          # 0=BG, 1=Oil, 2=Stain, 3=Scratch
    defect_classes   = [1, 2, 3]
    class_names      = ['Background', 'Oil', 'Stain', 'Scratch']
    img_hw           = (1080, 1920)   # H, W

    # 官方 RGB 映射
    rgb_map = {
        (0,   0,   0  ): 0,
        (128, 0,   0  ): 1,
        (0,   128, 0  ): 2,
        (128, 128, 0  ): 3,
    }
    # 灰度映射 (RGB→Gray 后的值, 已验证)
    gray_map = {0: 0, 38: 1, 75: 2, 113: 3}

    # ===================== 划分 =====================
    val_ratio = 0.15
    seed      = 42

    # ===================== 训练 =====================
    # ── 本地 (RTX 4050 6GB) ──   ── 服务器 (Linux, 大显存) ──
    crop_sizes    = [512, 640, 768]   # 同左（多尺度随机裁剪）
    batch_size    = 4                 # 768²+B5 约需 12GB，4 安全
    num_workers   = 12                # 服务器 12 核
    grad_accum_steps = 2              # 有效 batch = 4×2 = 8
    epochs        = 50                # 120→50（Epoch 28 已达最佳，50 留足缓冲）
    lr            = 1e-4
    weight_decay  = 1e-4
    warmup_epochs = 5
    grad_clip     = 1.0
    amp           = True              # 同左

    # 损失权重
    w_focal  = 0.4
    w_dice   = 0.3
    w_lovasz = 0.3          # epoch >= 5 才启用（修复 Lovasz 实现后提前）

    # 类别权重 (BG, Oil, Stain, Scratch)
    class_weights = [0.1, 1.0, 5.0, 3.0]

    # Early Stopping（Epoch 28 后持续过拟合，Loss 已失能）
    early_stop_patience = 15         # 15 epoch mIoU 不涨就停
    early_stop_min_delta = 0.001     

    # EMA
    ema_decay = 0.999

    # CopyPaste
    copypaste_prob = 0.3

    # ===================== 模型 =====================
    arch            = 'DeepLabV3Plus'
    encoder         = 'timm-efficientnet-b5'
    encoder_weights = 'imagenet'

    # ===================== 推理 =====================
    # 本地验证: 512/128, 服务器完整预测: 1024/256
    infer_crop    = 1024              # validate_pipeline.py 会覆盖为 512
    infer_overlap = 256               # validate_pipeline.py 会覆盖为 128
    tta_flips     = True       # hflip + vflip

    # ===================== RLE =====================
    rle_index_start = 1        # 像素编号从 1 开始
    rle_order       = 'C'      # 行优先 (row-major)

cfg = Config()

# 创建目录
for d in [cfg.ckpt_dir, cfg.log_dir, cfg.sub_dir]:
    os.makedirs(d, exist_ok=True)