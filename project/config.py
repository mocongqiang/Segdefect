"""全局配置"""
import os

# HuggingFace 国内镜像（解决 WinError 10060 连接超时问题）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

class Config:
    # ===================== 路径 =====================
    _proj_dir   = os.path.dirname(os.path.abspath(__file__))
    _root       = os.path.dirname(_proj_dir)          # Segdefect/
    data_root   = os.path.join(_root, 'data')
    train_img   = os.path.join(data_root, 'train/images')
    train_mask  = os.path.join(data_root, 'train/masks')
    test_img    = os.path.join(data_root, 'test/images')
    sample_sub  = os.path.join(data_root, 'sample_submission.csv')

    out_root    = os.path.join(_root, 'output')
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


    pseudo_threshold = 0.95

    pseudo_img_dir = os.path.join(
        data_root,
        'train',
        'pseudo_images'
    )
 # ===================== 伪标签 (Self-Training) 

    pseudo_mask_dir = os.path.join(
        data_root,
        'train',
        'pseudo_masks'
    )

    use_pseudo = True


    # ===================== 划分 =====================
    val_ratio = 0.15
    seed      = 42

    # ===================== 训练 =====================
    # ── 本地 (RTX 4050 6GB) ──   ── 服务器 (Linux, 大显存) ──
    crop_sizes    = [768, 896]    # 保持 768/896 多尺度，兼顾显存和分辨率
    batch_size    = 2                 
    num_workers   = 8                
    grad_accum_steps = 4              
    epochs        = 20                
    lr            = 5e-6          # 【修改1】提高学习率 2e-5 -> 3e-5，跳出局部最优
    weight_decay  = 1e-4
    warmup_epochs = 3
    grad_clip     = 1.0
    amp           = True              

    # 损失权重
    # 【修改2】调整 Loss 比例，增加 Dice 和 Lovasz，降低 Focal 防止初期不稳
    w_focal  = 0.35
    w_dice   = 0.35          
    w_lovasz = 0.10          
    w_boundary = 0.20        
    w_cls = 0.0            


    # Dice Warmup: 关闭，从第 0 个 Epoch 就开始强力学
    # 【修改3】关闭 Dice Warmup
    dice_warmup_start = 0
    dice_warmup_end = 0

    # 类别权重 (BG, Oil, Stain, Scratch)
    # 【修改4】极度放大稀有类别权重，打破模型只预测 Oil 的惰性
    class_weights = [0.01, 2.0, 3.0, 2.0]  # Oil 提到 10.0

    # Early Stopping
    early_stop_patience = 10         
    early_stop_min_delta = 0.001

    # EMA
    ema_decay = 0.996
    
    # 断点续训
# resume_from = None?????????????????????????? best.pth?
    # ????????resume_from = os.path.join(ckpt_dir, 'best.pth')
    resume_from = None
    # CopyPaste
    copypaste_prob = 0.8
    defect_crop_prob = 1.0

    # ===================== 模型 =====================
    arch            = 'Mask2Former'
    encoder         = 'facebook/mask2former-swin-large-cityscapes-semantic'  
    encoder_weights = 'imagenet'  

    # ===================== 推理 =====================
    # 正方形推理，与训练 crop（768/896）保持一致的正方形分布
    # 1024×1024 / 1152×1152 / 1280×1280
    infer_sizes    = [1024, 1152, 1280]
    infer_weights = [
        1.0,
        2.0,
        1.0,
    ]
    tta_flips     = True

    # 滑窗推理参数（伪标签生成 / validate_pipeline 用）
    infer_crop     = 1024
    infer_overlap  = 256
    # 多尺度 TTA（滑窗推理用）
    tta_scales     = [0.75, 1.0, 1.25, 1.5]

    # ===================== 后处理 =====================
    # cls_thresholds = {1: 0.65,  
    #                   2: 0.55,  
    #                   3: 0.50}  

    # min_area = {1: 150,   
    #             2: 50,    
    #             3: 80}  
    cls_thresholds = None

    min_area = None  

    fill_holes = False

    # ===================== RLE =====================
    rle_index_start = 1        
    rle_order       = 'F'      

    # ===================== 集成 =====================
    ensemble_models = [
        {
            'arch': 'Mask2Former',
            'encoder': 'facebook/mask2former-swin-large-cityscapes-semantic',
            'ckpt': os.path.join(
                os.path.join(_root, 'output', 'checkpoints'),
                'best.pth'
            ),
            'weight': 1.0,
        },
    ]

cfg = Config()

# 创建目录
for d in [cfg.ckpt_dir, cfg.log_dir, cfg.sub_dir]:
    os.makedirs(d, exist_ok=True)