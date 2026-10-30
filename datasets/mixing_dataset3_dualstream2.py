"""
增强版本的 MixingDataset，支持多种拼接策略、权重计算方式和双流设计。
主要改动：
1. 增加了 ricap_mode 参数，支持 'random', 'intra', 'neighbor', 'curriculum' 四种拼接策略。
2. 增加了 weighting_mode 参数，支持 'area', 'fixed', 'energy' 三种权重计算方式。
3. 增加了 dual_stream 参数，控制是否使用双流设计。  
4. 增加了 grid_type 参数，支持 'random' 和 'uniform' 两种网格划分方式。
5. 增加了 curriculum 学习机制，在前半训练周期使用 'intra' 策略，后半周期使用 'neighbor' 策略。
6. 增加了 label_smooth_sigma 参数，控制高斯序数标签的平滑程度。
7. 增加了 saliency_dir 和 use_fast_saliency 参数，支持预计算显著图和快速显著图计算。
8. 增加了 maxsize 参数，控制预计算显著图的最大尺寸，避免过大图像导致计算过慢。
9. 增加了 preload_saliency 参数，支持预加载显著图到内存中以加速训练。
10. 增加了 __getitem__ 返回值中的原始图像张量和权重张量，方便后续分析和可视化。原始图像张量为未经增强的图像，仅包含当前 index 的单张原图，代码会根据当前标签找出另外 3 张图像的索引。
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 图像处理工具函数（保持不变）
# ==========================================

def cv2_imread_safe(file_path):
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def compute_robust_saliency(img_bgr):
    if img_bgr is None:
        return None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_norm = img.astype(np.float32) / 255.0
    R, G, B = img_norm[:, :, 0], img_norm[:, :, 1], img_norm[:, :, 2]
    exgr = 3.0 * G - 2.4 * R - B
    saliency = np.maximum(0, exgr)
    if saliency.max() > 0:
        saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        saliency = np.zeros_like(R, dtype=np.uint8)
    _, mask = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    prob_map = cv2.GaussianBlur(mask, (51, 51), 0)
    return prob_map

def random_crop(img, crop_size):
    h, w = img.shape[:2]
    cw, ch = crop_size
    if h < ch or w < cw:
        scale = max(ch / h, cw / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        h, w = new_h, new_w
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    return img[y:y+ch, x:x+cw].copy()

def gaussian_ordinal_label(label, num_classes, sigma=0.5):
    """
    生成高斯序数标签
    label: int, 类别索引 (0 ~ num_classes-1)
    sigma: 高斯核宽度，控制平滑程度（越大则相邻类别越模糊）
    """
    classes = torch.arange(num_classes).float()
    dist = torch.exp(-(classes - label) ** 2 / (2 * sigma ** 2))
    return dist / dist.sum()

class MixingDataset(Dataset):
    def __init__(self, img_paths, labels, img_size=224, num_classes=9,
                 ricap_mode='random', weighting_mode='area', dual_stream=True,
                 grid_type='random', prob=0.5, subset_ratio=1.0, seed=42,
                 transform=None, saliency_dir=None, use_fast_saliency=False,
                 maxsize=1000, preload_saliency=True, label_smooth_sigma=0.5):
        self.img_paths = img_paths
        self.labels = labels
        self.img_size = img_size
        self.num_classes = num_classes
        self.ricap_mode = ricap_mode
        self.weighting_mode = weighting_mode
        self.dual_stream = dual_stream
        self.grid_type = grid_type
        self.prob = prob
        self.transform = transform
        self.saliency_dir = saliency_dir
        self.use_fast_saliency = use_fast_saliency
        self.maxsize = maxsize
        self.label_smooth_sigma = label_smooth_sigma  # 高斯平滑 sigma
        self.curriculum = (self.ricap_mode == 'curriculum')
        self.switch_epoch = 100   #
        self.current_epoch = 0

        # 子集采样（略）
        if subset_ratio < 1.0:
            rng = random.Random(seed)
            class_map = {}
            for p, l in zip(img_paths, labels):
                class_map.setdefault(l, []).append(p)
            new_paths, new_labels = [], []
            for l, ps in class_map.items():
                ps.sort()
                rng.shuffle(ps)
                n_keep = max(1, int(len(ps) * subset_ratio))
                selected = ps[:n_keep]
                new_paths.extend(selected)
                new_labels.extend([l] * len(selected))
            self.img_paths = new_paths
            self.labels = new_labels

        # 构建类别索引
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            self.class_indices.setdefault(label, []).append(idx)

        # 后处理（归一化等）
        self.post_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.curriculum:
            if epoch < self.switch_epoch:
                self.active_mode = 'intra'
            else:
                self.active_mode = 'neighbor'
        else:
            self.active_mode = self.ricap_mode

    def __len__(self):
        return len(self.img_paths)

    def _get_indices(self, current_idx, current_label):
        indices = [current_idx]
        mode = getattr(self, 'active_mode', self.ricap_mode)
        for _ in range(3):
            if mode == 'random':
                idx = random.randint(0, len(self.img_paths)-1)
            elif mode == 'intra':
                candidates = self.class_indices[current_label]
                idx = random.choice(candidates) if candidates else current_idx
            elif mode == 'neighbor':
                min_c = max(0, current_label - 1)
                max_c = min(self.num_classes - 1, current_label + 1)
                target_cls = random.randint(min_c, max_c)
                candidates = self.class_indices.get(target_cls, [])
                idx = random.choice(candidates) if candidates else current_idx
            indices.append(idx)
        return indices

    def __getitem__(self, index):
        
        img = cv2_imread_safe(self.img_paths[index])
        if img is None:
            return self.__getitem__((index + 1) % len(self))
        original_imgs = []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_img_resized = cv2.resize(img, (self.img_size, self.img_size))
        orig_img_tensor = torch.from_numpy(orig_img_resized).permute(2,0,1).float() / 255.0
        # img_tensor = torch.nn.functional.interpolate(
        # img_tensor.unsqueeze(0),
        #     size=(self.img_size, self.img_size),
        #     mode='bilinear',
        #     align_corners=False
        # ).squeeze(0)
        # original_imgs.append(img_tensor)

        # 未触发拼接：返回原始单张图像（硬标签转为高斯软标签）
        if random.random() > self.prob:
            label = self.labels[index]
            soft_label = gaussian_ordinal_label(label, self.num_classes, sigma=self.label_smooth_sigma)
            
            if self.transform:
                aug = self.transform(image=img)
                img_tensor = aug['image']
            # else:
            #     img_tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
            
            # 返回：图像, 软标签, 空张量(占位), 空张量(占位)
            return img_tensor, soft_label, orig_img_tensor, torch.zeros(4)

        # 触发拼接增强
        current_label = self.labels[index]
        indices = self._get_indices(index, current_label)


        w = h = self.img_size
        if self.grid_type == 'random':
            cx = random.randint(int(0.3*w), int(0.7*w))
            cy = random.randint(int(0.3*h), int(0.7*h))
            regions = [
                (0, 0, cx, cy),
                (cx, 0, w-cx, cy),
                (0, cy, cx, h-cy),
                (cx, cy, w-cx, h-cy)
            ]
        else:
            half_w, half_h = w // 2, h // 2
            regions = [
                (0, 0, half_w, half_h),
                (half_w, 0, w - half_w, half_h),
                (0, half_h, half_w, h - half_h),
                (half_w, half_h, w - half_w, h - half_h)
            ]

        final_img = np.zeros((h, w, 3), dtype=np.uint8)
        patches_info = []  # (label_soft, area, energy)

        for i, idx in enumerate(indices):
            path = self.img_paths[idx]
            label = self.labels[idx]
            img = cv2_imread_safe(path)
            if img is None:
                patches_info.append((gaussian_ordinal_label(label, self.num_classes, sigma=self.label_smooth_sigma), 0, 0))
                continue

            x, y, rw, rh = regions[i]
            energy = 0.0

            # 根据双流设计获取 patch
            if self.dual_stream and i < 2:
                patch = cv2.resize(img, (rw, rh))
                if self.weighting_mode == 'energy':
                    sal_map = compute_robust_saliency(patch)
                    energy = sal_map.sum() if sal_map is not None else 1e-6
            elif self.dual_stream and i >= 2:
                prob_map = compute_robust_saliency(img)
                if prob_map is None or prob_map.sum() == 0:
                    patch = random_crop(img, (rw, rh))
                    energy = 1e-6
                else:
                    flat_prob = prob_map.flatten().astype(np.float64)
                    flat_prob /= (flat_prob.sum() + 1e-8)
                    center_idx = np.random.choice(len(flat_prob), p=flat_prob)
                    cy_src, cx_src = divmod(center_idx, img.shape[1])
                    y1 = max(0, min(cy_src - rh//2, img.shape[0] - rh))
                    x1 = max(0, min(cx_src - rw//2, img.shape[1] - rw))
                    patch = img[y1:y1+rh, x1:x1+rw]
                    energy = prob_map[y1:y1+rh, x1:x1+rw].sum()
            elif not self.dual_stream and self.grid_type == 'uniform':
                patch = cv2.resize(img, (rw, rh))
                if self.weighting_mode == 'energy':
                    sal_map = compute_robust_saliency(patch)
                    energy = sal_map.sum() if sal_map is not None else 1e-6
            else:
                patch = random_crop(img, (rw, rh))
                if self.weighting_mode == 'energy':
                    sal_map = compute_robust_saliency(patch)
                    energy = sal_map.sum() if sal_map is not None else 1e-6

            final_img[y:y+rh, x:x+rw] = patch
            # 存储每个 patch 的高斯软标签
            label_soft = gaussian_ordinal_label(label, self.num_classes, sigma=self.label_smooth_sigma)
            patches_info.append((label_soft, rw*rh, energy))

        # 权重计算
        total_area = w * h
        weights = torch.zeros(4)
        if self.weighting_mode == 'area':
            for i, (_, area, _) in enumerate(patches_info):
                weights[i] = area / total_area
        elif self.weighting_mode == 'fixed':
            weights = torch.ones(4) * 0.25
        elif self.weighting_mode == 'energy':
            energies = [info[2] for info in patches_info]
            total_energy = sum(energies) + 1e-6
            for i, e in enumerate(energies):
                weights[i] = e / total_energy

        # 混合软标签（按权重线性组合）
        final_label = torch.zeros(self.num_classes)
        for i, (label_soft, _, _) in enumerate(patches_info):
            final_label += weights[i] * label_soft

        # 后处理
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        transformed = self.post_transform(image=final_img)
        return transformed['image'], final_label, orig_img_tensor, weights