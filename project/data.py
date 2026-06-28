"""数据加载、Mask转换、数据增强、CopyPaste"""
import os, random, glob
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import cfg



# import os
# from config import cfg

# print("Train image path:", os.path.abspath(cfg.train_img))
# print("Train mask path :", os.path.abspath(cfg.train_mask))
# print("Exists images? ", os.path.exists(cfg.train_img))
# print("Exists masks?  ", os.path.exists(cfg.train_mask))
# print("Number of image files:", len(os.listdir(cfg.train_img)) if os.path.exists(cfg.train_img) else 0)

# ------------------------------------------------------------------
#  Mask 加载
# ------------------------------------------------------------------
def load_mask(path):
    """读取 mask 并转为 [0,1,2,3] 类别 ID。"""
    img = Image.open(path)
    if img.mode == 'P':
        img = img.convert('RGB')
    arr = np.array(img)

    if arr.ndim == 3:
        # RGB 标注图
        h, w = arr.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        out[(arr[..., 0] == 128) & (arr[..., 1] == 0)   & (arr[..., 2] == 0)]   = 1
        out[(arr[..., 0] == 0)   & (arr[..., 1] == 128) & (arr[..., 2] == 0)]   = 2
        out[(arr[..., 0] == 128) & (arr[..., 1] == 128) & (arr[..., 2] == 0)]   = 3
        return out
    else:
        # 灰度标注图 (值 0/38/75/113)
        out = np.zeros_like(arr, dtype=np.uint8)
        out[arr == 38]  = 1
        out[arr == 75]  = 2
        out[arr == 113] = 3
        return out


def load_image(path):
    """读取 RGB 图像。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def defect_aware_crop(
    image,
    mask,
    crop_size,
    defect_prob=0.8,
):
    """
    优先裁剪包含缺陷区域。

    80%概率：
        裁剪中心落在缺陷附近

    20%概率：
        完全随机裁剪
    """

    H, W = mask.shape

    # -----------------------------
    # 随机裁剪
    # -----------------------------
    if random.random() < (1 - defect_prob):

        x = random.randint(
            0,
            max(0, W - crop_size)
        )

        y = random.randint(
            0,
            max(0, H - crop_size)
        )

        return (
            image[y:y+crop_size,
                  x:x+crop_size],
            mask[y:y+crop_size,
                 x:x+crop_size]
        )

    # -----------------------------
    # 找所有缺陷像素
    # -----------------------------
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:

        x = random.randint(
            0,
            max(0, W - crop_size)
        )

        y = random.randint(
            0,
            max(0, H - crop_size)
        )

        return (
            image[y:y+crop_size,
                  x:x+crop_size],
            mask[y:y+crop_size,
                 x:x+crop_size]
        )

    # -----------------------------
    # 随机选择一个缺陷像素
    # -----------------------------
    idx = random.randint(
        0,
        len(xs)-1
    )

    cx = xs[idx]

    cy = ys[idx]

    # -----------------------------
    # 加一点随机偏移
    # -----------------------------
    cx += random.randint(-64,64)

    cy += random.randint(-64,64)

    x1 = np.clip(
        cx-crop_size//2,
        0,
        W-crop_size
    )

    y1 = np.clip(
        cy-crop_size//2,
        0,
        H-crop_size
    )

    return (

        image[
            y1:y1+crop_size,
            x1:x1+crop_size
        ],

        mask[
            y1:y1+crop_size,
            x1:x1+crop_size
        ]

    )

# ------------------------------------------------------------------
#  数据划分
# ------------------------------------------------------------------
def get_split():
    """返回 train_files, val_files (每个元素 = (img_path, mask_path))。"""
    all_imgs = sorted(glob.glob(os.path.join(cfg.train_img, '*.jpg')))
    rng = np.random.RandomState(cfg.seed)
    indices = rng.permutation(len(all_imgs))
    n_val = int(len(all_imgs) * cfg.val_ratio)

    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    def to_pair(p):
        name = os.path.basename(p).replace('.jpg', '.png')
        mp = os.path.join(cfg.train_mask, name)
        return (p, mp)

    train_files = [to_pair(all_imgs[i]) for i in train_idx]
    val_files   = [to_pair(all_imgs[i]) for i in val_idx]
    return train_files, val_files


def get_sample_weights(files):
    """根据缺陷类别出现频率计算采样权重 (Stain 高权重)。"""
    weights = []
    for img_path, mask_path in files:
        mask = load_mask(mask_path)
        unique = set(np.unique(mask).tolist())
        w = 1.0
        if 2 in unique:     # Stain 极少 → 大幅加权
            w *= 5.0
        if 3 in unique:     # Scratch
            w *= 5.0
        if 1 in unique:     # Oil
            w *= 1.5
        weights.append(w)
    return weights





# ------------------------------------------------------------------
#  CopyPaste 缺陷库
# ------------------------------------------------------------------
class DefectBank:
    """预提取 Stain / Scratch 缺陷块，训练时随机贴到其他图上。"""

    def __init__(self, files):
        self.bank = {1: [], 2: [], 3: []}
        for img_path, mask_path in files:
            mask = load_mask(mask_path)
            img  = load_image(img_path)
            for cls in [1, 2, 3]:
                ys, xs = np.where(mask == cls)
                if len(ys) < 5:
                    continue
                y1, y2 = max(0, ys.min() - 8), min(mask.shape[0], ys.max() + 9)
                x1, x2 = max(0, xs.min() - 8), min(mask.shape[1], xs.max() + 9)
                if (y2 - y1) < 5 or (x2 - x1) < 5:
                    continue
                self.bank[cls].append({
                    'img':  img[y1:y2, x1:x2].copy(),
                    'mask': (mask[y1:y2, x1:x2] == cls).astype(np.uint8),
                })
        for cls in [1, 2, 3]:
            print(f'  DefectBank class {cls}: {len(self.bank[cls])} patches')

    def paste(self, img, mask):
        """随机选一个缺陷块贴到 img/mask 上。"""
        # 优先选 Scratch (cls=3)，因为最难学
        cls_pool = []
        r = random.random()
        if r < 0.5 and len(self.bank[3]) > 0:
            cls_pool = self.bank[3]
        elif r < 0.8 and len(self.bank[2]) > 0:
            cls_pool = self.bank[2]
        else:
            all_items = []
            for c in [1, 2, 3]:
                all_items.extend([(c, item) for item in self.bank[c]])
            if not all_items:
                return img, mask
            cls, item = random.choice(all_items)
            cls_pool = None

        if cls_pool is not None:
            item = random.choice(cls_pool)
            # 根据采样的池子直接确定类别
            if cls_pool is self.bank[2]:
                cls = 2
            elif cls_pool is self.bank[3]:
                cls = 3
            else:
                cls = 1

        pimg  = item['img']
        pmask = item['mask']
        ph, pw = pimg.shape[:2]
        H,  W  = img.shape[:2]
        if ph >= H or pw >= W:
            return img, mask

        py = random.randint(0, H - ph)
        px = random.randint(0, W - pw)

        # 边界平滑: 对 pmask 做 1px 膨胀后羽化
        k = np.ones((3, 3), np.uint8)
        boundary = cv2.dilate(pmask, k, iterations=1) - pmask
        blend = cv2.GaussianBlur(boundary.astype(np.float32), (5, 5), 1.0)

        roi_img  = img[py:py+ph, px:px+pw].astype(np.float32)
        roi_mask = mask[py:py+ph, px:px+pw].copy()

        # 在 pmask 区域直接替换图像
        m3 = pmask[..., None] > 0
        roi_img = np.where(m3, pimg.astype(np.float32), roi_img)
        # 边界处混合
        roi_img = np.where(
            (blend > 0)[..., None] & ~m3,
            roi_img * (1 - blend[..., None] * 0.5) + pimg.astype(np.float32) * (blend[..., None] * 0.5),
            roi_img
        )
        img[py:py+ph, px:px+pw] = roi_img.astype(np.uint8)
        mask[py:py+ph, px:px+pw] = np.where(pmask > 0, cls, roi_mask)

        return img, mask


# ------------------------------------------------------------------
#  数据增强
# ------------------------------------------------------------------
def get_train_transform(crop_size):
    """
    训练增强（注意：这里已经不负责 RandomCrop）

    crop 已经由 defect_aware_crop 完成，
    transform 这里只负责几何增强、颜色增强、噪声增强。
    """

    return A.Compose([

        # -------------------------
        # 几何增强
        # -------------------------
        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),

        # -------------------------
        # Elastic
        # -------------------------
        A.OneOf([
            A.ElasticTransform(alpha=30, sigma=5),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=0.03),
        ], p=0.2),

        # -------------------------
        # Noise
        # -------------------------
        A.OneOf([
            A.GaussNoise(),
            A.ISONoise(),
        ], p=0.25),

        # -------------------------
        # Blur
        # -------------------------
        A.OneOf([
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ], p=0.15),

        # -------------------------
        # Brightness
        # -------------------------
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.03,
            p=0.4
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),

        # -------------------------
        # Dropout
        # -------------------------
        A.CoarseDropout(
            num_holes_range=(2,6),
            hole_height_range=(16,48),
            hole_width_range=(16,48),
            p=0.2
        ),

        # -------------------------
        # Normalize
        # -------------------------
        A.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        ),

        ToTensorV2()

    ])

def _pad_to_divisor(img, mask=None, divisor=32, return_pad=False):
    """将图像 (和 mask) pad 到 divisor 的倍数 (右下侧补零)。

    Args:
        return_pad: 如果为 True，额外返回 (pad_h, pad_w) 用于后续裁剪
    """
    h, w = img.shape[:2]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    if pad_h == 0 and pad_w == 0:
        if return_pad:
            return (img, mask, 0, 0) if mask is not None else (img, 0, 0)
        return (img, mask) if mask is not None else img

    if img.ndim == 3:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
                     mode='constant', constant_values=0)
    else:
        img = np.pad(img, ((0, pad_h), (0, pad_w)),
                     mode='constant', constant_values=0)

    if mask is not None:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)),
                      mode='constant', constant_values=0)
        if return_pad:
            return img, mask, pad_h, pad_w
        return img, mask

    if return_pad:
        return img, pad_h, pad_w
    return img


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


# ------------------------------------------------------------------
#  Dataset
# ------------------------------------------------------------------
class DefectDataset(Dataset):
    def __init__(self, files, mode='train', defect_bank=None):
        self.files = files
        self.mode = mode
        self.defect_bank = defect_bank
        self.transforms = {
            cs: get_train_transform(cs) for cs in cfg.crop_sizes
        } if mode == 'train' else None
        self.val_t = get_val_transform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, mask_path = self.files[idx]
        img  = load_image(img_path)
        mask = load_mask(mask_path)

        if self.mode == 'train':
            # CopyPaste
            if self.defect_bank and random.random() < cfg.copypaste_prob:
                img, mask = self.defect_bank.paste(img, mask)

            # 随机选 crop 尺寸
            cs = random.choice(cfg.crop_sizes)

            img,mask = defect_aware_crop(

                img,
                mask,
                crop_size=cs,
                defect_prob=cfg.defect_crop_prob
            )

            augmented = self.transforms[cs](

                image=img,

                mask=mask,

            )

            img_t = augmented["image"]

            mask_t = augmented["mask"].long()

            return img_t,mask_t

        else:
            img, mask = _pad_to_divisor(img, mask, divisor=32)
            augmented = self.val_t(image=img, mask=mask)
            img_t = augmented['image']
            mask_t = augmented['mask'].long()
            return img_t, mask_t


class TestDataset(Dataset):
    def __init__(self, image_dir):
        self.files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.transform = get_val_transform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        name = os.path.basename(path)
        img = load_image(path)
        h, w = img.shape[:2]
        img = _pad_to_divisor(img, divisor=32)
        augmented = self.transform(image=img)
        return augmented['image'], name, h, w


# ------------------------------------------------------------------
#  Sampler
# ------------------------------------------------------------------


# ------------------------------------------------------------------
#  Collate
# ------------------------------------------------------------------
def collate_fn(batch):
    """将不同 crop_size 的样本 align 到 batch 内最大尺寸。"""
    imgs, masks = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)

    padded_imgs, padded_masks = [], []
    for img, mask in zip(imgs, masks):
        if img.shape[1] != max_h or img.shape[2] != max_w:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(max_h, max_w),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
            mask = torch.nn.functional.interpolate(
                mask, size=(max_h, max_w), mode='nearest'
            ).squeeze(0).squeeze(0).long()
        padded_imgs.append(img)
        padded_masks.append(mask)
    return torch.stack(padded_imgs), torch.stack(padded_masks)
