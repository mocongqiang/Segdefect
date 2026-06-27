"""模型定义: 纯分割模型"""
import segmentation_models_pytorch as smp
from config import cfg


class DefectModel(smp.DeepLabV3Plus):
    """纯分割模型，直接继承 smp.DeepLabV3Plus。"""
    pass


def build_model():
    """构建纯分割模型。"""
    return DefectModel(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=3,
        classes=cfg.num_classes,
    )
