"""RLE 编解码 (Kaggle 标准: F-order 列优先, 1-indexed)"""
import os
import numpy as np
from config import cfg

def encode_rle(binary_mask):
    """将二值 mask 编码为 RLE 字符串。

    Args:
        binary_mask: 2D numpy array, 值为 0/1

    Returns:
        str: "start1 length1 start2 length2 ..." 或 "0 0"
    """
    mask = np.asarray(binary_mask, dtype=np.uint8)
    if mask.sum() == 0:
        return '0 0'

    # Kaggle 标准：F-order flatten + compact encoding
    pixels = mask.flatten(order=cfg.rle_order)
    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + cfg.rle_index_start
    runs[1::2] -= runs[::2]

    return ' '.join(map(str, runs))


def decode_rle(rle_str, shape):
    """将 RLE 字符串解码为二值 mask。"""
    mask = np.zeros(np.prod(shape), dtype=np.uint8)
    if rle_str == '0 0' or rle_str == '':
        return mask.reshape(shape)
    nums = list(map(int, rle_str.split()))
    for i in range(0, len(nums), 2):
        start = nums[i] - cfg.rle_index_start
        length = nums[i + 1]
        mask[start:start + length] = 1
    return mask.reshape(shape, order=cfg.rle_order)


def generate_submission(test_predictions, test_names, output_path):
    """生成提交 CSV。

    Args:
        test_predictions: list of (H, W) array, 每张图的类别预测
        test_names: list of str, 图像名 (如 'test_0001.png')
        output_path: str
    """
    import pandas as pd
    rows = []
    for pred, name in zip(test_predictions, test_names):
        # 从文件名提取 id, 如 'test_0001.png' -> 'test_0001'
        img_id = os.path.splitext(name)[0]
        for cls in cfg.defect_classes:   # 1=Oil, 2=Stain, 3=Scratch
            binary = (pred == cls).astype(np.uint8)
            rle = encode_rle(binary)
            rows.append({
                'id':  f'{img_id}_{cls}',
                'rle': rle,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f'Submission saved to {output_path}, {len(df)} rows')
    return df