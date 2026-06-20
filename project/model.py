"""模型定义"""
import segmentation_models_pytorch as smp
from config import cfg

def build_model():
    """构建分割模型。

    默认: DeepLabV3+ + EfficientNet-B5
    备选: UnetPlusPlus + 同编码器 (更强的跳跃连接, 适合小缺陷)
    """
    kwargs = dict(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=3,
        classes=cfg.num_classes,
    )

    if cfg.arch == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(**kwargs)
    elif cfg.arch == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(**kwargs)
    elif cfg.arch == 'MAnet':
        model = smp.MAnet(**kwargs)
    else:
        raise ValueError(f'Unknown arch: {cfg.arch}')

    return model