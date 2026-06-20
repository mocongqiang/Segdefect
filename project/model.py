"""模型定义: 分割头 + 分类头"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from config import cfg


class DefectModel(nn.Module):
    """分割 + 多标签分类双任务模型。

    EfficientNet-B5 Encoder
    ├── DeepLabV3Plus Decoder → seg_logits (B, 4, H, W)
    └── GAP → FC(3) → cls_logits (B, 3)    ← 新增分类头
    """

    def __init__(self, arch='DeepLabV3Plus', encoder='timm-efficientnet-b5',
                 encoder_weights='imagenet', num_classes=4):
        super().__init__()
        # 构建分割模型，并拆出 encoder / decoder
        kwargs = dict(
            encoder_name=encoder, encoder_weights=encoder_weights,
            in_channels=3, classes=num_classes,
        )
        if arch == 'DeepLabV3Plus':
            seg_model = smp.DeepLabV3Plus(**kwargs)
        elif arch == 'UnetPlusPlus':
            seg_model = smp.UnetPlusPlus(**kwargs)
        elif arch == 'MAnet':
            seg_model = smp.MAnet(**kwargs)
        else:
            raise ValueError(f'Unknown arch: {arch}')

        self.encoder = seg_model.encoder
        self.decoder = seg_model.decoder
        self.seg_head = seg_model.segmentation_head

        # 分类头: GAP + FC(encoder_out_channels, 3)
        enc_out = self._get_encoder_out_channels()
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_out, 3),
        )

    def _get_encoder_out_channels(self):
        """返回 encoder 最深层输出通道数。"""
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.encoder(dummy)
            # feats 可能是 tensor 或 list，取最后一层
            if isinstance(feats, (list, tuple)):
                return feats[-1].shape[1]
            return feats.shape[1]

    def forward(self, x):
        features = self.encoder(x)          # encoder 多尺度特征
        decoder_out = self.decoder(features) if isinstance(features, (list, tuple)) else self.decoder(features)
        seg_logits = self.seg_head(decoder_out)  # (B, 4, H, W)

        # 用 encoder 最后一层特征做分类
        last_feat = features[-1] if isinstance(features, (list, tuple)) else features
        cls_logits = self.cls_head(last_feat)     # (B, 3)

        return seg_logits, cls_logits


def build_model():
    """构建 DefectModel（兼容旧接口）。"""
    return DefectModel(
        arch=cfg.arch, encoder=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        num_classes=cfg.num_classes,
    )
