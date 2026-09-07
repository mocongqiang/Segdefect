"""模型定义: 支持 Mask2Former 和 smp 系列模型 (纯分割)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from config import cfg


# ======================================================================
#  DeepLabV3Plus (保留兼容)
# ======================================================================
class DefectModel(smp.DeepLabV3Plus):
    """纯分割模型，直接继承 smp.DeepLabV3Plus。"""
    pass


# ======================================================================
#  Mask2Former 包装类
# ======================================================================
class Mask2FormerSeg(nn.Module):
    """Mask2Former 语义分割包装，输出与 smp 模型兼容的 (B, num_classes, H, W) logit。"""

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        # 使用 Auto 系列导入，兼容旧版本 transformers
        from transformers import AutoModelForUniversalSegmentation, AutoConfig

        # 1. 加载配置文件
        config = AutoConfig.from_pretrained(model_name)
        # 2. 修改配置中的类别数
        config.num_labels = num_classes

        # 3. 加载模型，使用 ignore_mismatched_sizes=True
        self.model = AutoModelForUniversalSegmentation.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        self.num_classes = num_classes

        # 【解冻策略】：解冻 Swin-Large 的最后两个 Stage (Stage 2 和 Stage 3)
        # 冻结前两个 Stage 和 patch embedding，保留底层通用特征
        for name, param in self.model.model.pixel_level_module.encoder.named_parameters():
            if "stages.0" in name or "stages.1" in name or "embeddings" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True  # 解冻 stages.2, stages.3 及后续层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 输入图像

        Returns:
            logits: (B, num_classes, H, W) 分割 logits
        """
        B, _, H, W = x.shape

        # Mask2Former 前向
        outputs = self.model(pixel_values=x)

        # class_queries_logits: (B, num_queries, num_classes+1)  (含 no-object 类)
        # masks_queries_logits: (B, num_queries, H', W')  <- 注意这里是低分辨率 H', W'
        class_logits = outputs.class_queries_logits  # (B, Q, C+1)
        mask_logits = outputs.masks_queries_logits    # (B, Q, H', W')

        # 取每个 query 的预测类别概率 (排除最后一列 no-object)
        class_probs = F.softmax(class_logits[:, :, :self.num_classes], dim=-1)

        # 【关键优化 1】：先在低分辨率下做 einsum 加权合成！
        # 此时 mask_logits 是小尺寸的，显存占用极小！
        semantic_logits = torch.einsum('bqc,bqhw->bchw', class_probs, mask_logits)

        # 【关键优化 2】：将合成好的 4 类特征图插值回原始尺寸
        # 只插值 4 个通道，而不是插值 100 个通道，显存占用直接降到原来的 1/25！
        semantic_logits = F.interpolate(
            semantic_logits, size=(H, W), mode='bilinear', align_corners=False
        )

        return semantic_logits


# ======================================================================
#  架构映射
# ======================================================================
_ARCH_MAP = {
    'DeepLabV3Plus': smp.DeepLabV3Plus,
    'Segformer': smp.Segformer,
    'UnetPlusPlus': smp.UnetPlusPlus,
    'Unet': smp.Unet,
    'Mask2Former': 'Mask2Former',  # 特殊标记，由 build_model 处理
}


def build_model():
    """构建纯分割模型（使用 cfg 中的默认配置）。"""
    if cfg.arch == 'Mask2Former':
        return Mask2FormerSeg(
            model_name=cfg.encoder,
            num_classes=cfg.num_classes,
        )
    else:
        return DefectModel(
            encoder_name=cfg.encoder,
            encoder_weights=cfg.encoder_weights,
            in_channels=3,
            classes=cfg.num_classes,
        )


def build_model_custom(arch, encoder, encoder_weights='imagenet'):
    """根据指定架构和编码器构建模型（用于异构集成）。"""
    if arch == 'Mask2Former':
        return Mask2FormerSeg(
            model_name=encoder,
            num_classes=cfg.num_classes,
        )

    if arch not in _ARCH_MAP:
        raise ValueError(
            f"Unknown arch '{arch}', supported: {list(_ARCH_MAP.keys())}"
        )
    model_cls = _ARCH_MAP[arch]
    return model_cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=cfg.num_classes,
    )