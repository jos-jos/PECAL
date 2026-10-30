import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def get_transforms(cfg_augment, img_size=224, is_train=True):
    """
    根据 config.augment 配置返回 transform
    """
    if not is_train:
        # 验证集永远是：Resize + Normalize
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    aug_name = cfg_augment.name  # 'basic' or 'full' (strong)

    transforms = []

    # === 基础几何增强 (所有策略都包含) ===
    # 注意: scale=(0.08, 1.0) 是 ImageNet 标准，能大幅增加尺度多样性。
    # 你之前的 (0.8, 1.0) 太保守了，几乎没有尺度变化，建议改为 (0.2, 1.0) 适应农业场景。
    transforms.append(
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.2, 1.0), ratio=(0.75, 1.33))
    )
    transforms.append(A.HorizontalFlip(p=0.5))

    # === 强增强策略 (RandAugment) ===
    if aug_name == 'full' or aug_name == 'randaugment':
        # 模拟 RandAugment (N=2, M=9)
        # SomeOf(n=2): 每次随机选 2 个操作
        # p=1.0: 只要开启强增强，就一定执行这层逻辑
        transforms.append(
            A.SomeOf([
                # --- 几何类 (Geometric) ---
                # 旋转: 30度 (M=9 对应大约 30度)
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
                # 错切 (Shear): 模拟视角变化
                A.Affine(shear={'x': (-15, 15), 'y': (-15, 15)}, border_mode=cv2.BORDER_CONSTANT, cval=0, p=1),
                # 平移 (Translate): 模拟物体位置偏差
                A.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}, border_mode=cv2.BORDER_CONSTANT,
                         cval=0, p=1),

                # --- 像素类 (Photometric) ---
                # AutoContrast 的替代品，非常适合光照不均的田间图像
                A.Equalize(mode="cv", by_channels=True, p=1),
                # Solarize: 模拟过度曝光
                A.Solarize(threshold=128, p=1),
                # Posterize: 减少色阶，模拟传感器噪声或压缩
                A.Posterize(num_bits=4, p=1),
                # ColorJitter: 亮度/对比度/饱和度 (M=9 对应较强抖动)
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1),
                # Sharpen: 锐化
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], n=2, p=1.0)  # [核心参数] n=2 代表 RandAugment-N2
        )

    # ------------------------------------------------------------------
    # 3. 归一化 (Normalize)
    # ------------------------------------------------------------------
    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return A.Compose(transforms)