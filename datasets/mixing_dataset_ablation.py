"""
消融实验版 MixingDataset

在 mixing_dataset_n_global.py 基础上新增两个消融参数：

1. label_type: 软标签类型
   - 'gaussian': 高斯序数软标签（默认，原版行为）
   - 'onehot':  独热编码
   - 'uniform': 均匀标签平滑 (epsilon=0.1)

2. switch_epoch: 课程学习切换时机
   - 控制 curriculum 模式下从 intra → neighbor 的切换 epoch
   - 默认 100（原版行为），消融实验设为 50 / 100 / 150

同时保留 n_global 和 random_position 参数，确保可与 n_global 消融实验兼容。
其余逻辑（RICAP 网格划分、权重计算、双流返回原始图像）与原版完全一致。
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
# 图像处理工具函数（与原版完全一致）
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
    classes = torch.arange(num_classes).float()
    dist = torch.exp(-(classes - label) ** 2 / (2 * sigma ** 2))
    return dist / dist.sum()


class MixingDataset(Dataset):
    def __init__(self, img_paths, labels, img_size=224, num_classes=9,
                 ricap_mode='random', weighting_mode='area', dual_stream=True,
                 grid_type='random', prob=0.5, subset_ratio=1.0, seed=42,
                 transform=None, saliency_dir=None, use_fast_saliency=False,
                 maxsize=1000, preload_saliency=True, label_smooth_sigma=0.5,
                 n_global=2, random_position=False,
                 label_type='gaussian', switch_epoch=100,
                 local_crop_mode='guided'):
        """
        新增消融参数:
            label_type: 'gaussian' | 'onehot' | 'uniform'
            switch_epoch: curriculum 模式下 intra→neighbor 的切换 epoch
            (n_global, random_position 见 mixing_dataset_n_global.py)
            local_crop_mode: [packaged] 仅作用于"局部"图块的取窗方式
                'guided' —— 显著图引导采样（默认，等价原行为）
                'random' —— 均匀随机取窗，仍按 weighting_mode 计算该窗的响应能量

        ⚠️ 修复记录（2026-10-30）：Table 5 的 Random+Area / Random+Energy 两个配置
        依赖本参数，但它此前只加在了 mixing_dataset3_dualstream2_ablation.py（patch_mode
        链路），而 Table 5 走的是本文件（ablation 链路）→ 该参数被静默忽略，
        导致 Random+Area 与 Random+Energy 会训出完全相同的模型。
        """
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
        self.label_smooth_sigma = label_smooth_sigma
        self.local_crop_mode = local_crop_mode  # [packaged]
        self.n_global = n_global
        self.random_position = random_position
        self.label_type = label_type          # <-- 消融：软标签类型
        self.switch_epoch = switch_epoch      # <-- 消融：课程切换时机
        self.curriculum = (self.ricap_mode == 'curriculum')
        self.current_epoch = 0

        # 子集采样
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

        print(f"   [ablation] label_type={self.label_type}, "
              f"switch_epoch={self.switch_epoch}, "
              f"n_global={self.n_global}, random_position={self.random_position}")

    # ==========================================
    # 消融核心：软标签生成
    # ==========================================
    def _make_soft_label(self, label):
        """根据 label_type 生成软标签"""
        if self.label_type == 'onehot':
            sl = torch.zeros(self.num_classes)
            sl[label] = 1.0
            return sl
        elif self.label_type == 'uniform':
            # 均匀标签平滑: (1-eps)*onehot + eps/num_classes
            epsilon = 0.1
            sl = torch.ones(self.num_classes) * (epsilon / self.num_classes)
            sl[label] += 1.0 - epsilon
            return sl
        else:  # 'gaussian' (默认)
            return gaussian_ordinal_label(label, self.num_classes, sigma=self.label_smooth_sigma)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.curriculum:
            # 消融核心：switch_epoch 控制 intra→neighbor 切换时机
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

    def _extract_patch(self, img, rw, rh, is_global):
        """根据 is_global 布尔值提取单个 patch。"""
        energy = 0.0

        if is_global:
            patch = cv2.resize(img, (rw, rh))
            if self.weighting_mode == 'energy':
                sal_map = compute_robust_saliency(patch)
                energy = sal_map.sum() if sal_map is not None else 1e-6
        else:
            # [packaged] 局部图块的取窗方式：guided=显著图引导 / random=均匀随机
            if self.local_crop_mode == 'random':
                h_i, w_i = img.shape[:2]
                if h_i < rh or w_i < rw:
                    scale = max(rh / h_i, rw / w_i)
                    img = cv2.resize(img, (int(w_i * scale) + 1, int(h_i * scale) + 1))
                    h_i, w_i = img.shape[:2]
                y1 = random.randint(0, h_i - rh)
                x1 = random.randint(0, w_i - rw)
                patch = img[y1:y1 + rh, x1:x1 + rw].copy()
                if self.weighting_mode == 'energy':
                    response = compute_robust_saliency(img)
                    energy = (float(response[y1:y1 + rh, x1:x1 + rw].sum())
                              if response is not None else 1e-6)
                else:
                    energy = 1e-6
            else:
                prob_map = compute_robust_saliency(img)
                if prob_map is None or prob_map.sum() == 0:
                    patch = random_crop(img, (rw, rh))
                    energy = 1e-6
                else:
                    flat_prob = prob_map.flatten().astype(np.float64)
                    flat_prob /= (flat_prob.sum() + 1e-8)
                    center_idx = np.random.choice(len(flat_prob), p=flat_prob)
                    cy_src, cx_src = divmod(center_idx, img.shape[1])
                    y1 = max(0, min(cy_src - rh // 2, img.shape[0] - rh))
                    x1 = max(0, min(cx_src - rw // 2, img.shape[1] - rw))
                    patch = img[y1:y1 + rh, x1:x1 + rw]
                    energy = prob_map[y1:y1 + rh, x1:x1 + rw].sum()

        return patch, energy

    def __getitem__(self, index):

        img = cv2_imread_safe(self.img_paths[index])
        if img is None:
            return self.__getitem__((index + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_img_resized = cv2.resize(img, (self.img_size, self.img_size))
        orig_img_tensor = torch.from_numpy(orig_img_resized).permute(2, 0, 1).float() / 255.0

        # 未触发拼接：返回原始单张图像
        if random.random() > self.prob:
            label = self.labels[index]
            soft_label = self._make_soft_label(label)

            if self.transform:
                aug = self.transform(image=img)
                img_tensor = aug['image']

            # 单图样本天然属于"类内组合"（4 个图块均为同一类），
            # 供 D 组（same-class mask）判定语义一致性条件是否成立。
            patch_labels = torch.full((4,), int(label), dtype=torch.long)
            return img_tensor, soft_label, orig_img_tensor, torch.zeros(4), patch_labels

        # 触发拼接增强
        current_label = self.labels[index]
        indices = self._get_indices(index, current_label)

        w = h = self.img_size
        if self.grid_type == 'random':
            cx = random.randint(int(0.3 * w), int(0.7 * w))
            cy = random.randint(int(0.3 * h), int(0.7 * h))
            regions = [
                (0, 0, cx, cy),
                (cx, 0, w - cx, cy),
                (0, cy, cx, h - cy),
                (cx, cy, w - cx, h - cy)
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
        patches_info = []
        patch_cls = []          # 4 个图块各自的原始阶段类别（供 D 组掩码使用）

        # n_global / random_position 逻辑（与 n_global 版完全一致）
        if self.random_position and 0 < self.n_global < 4:
            is_global_mask = [True] * self.n_global + [False] * (4 - self.n_global)
            random.shuffle(is_global_mask)
        else:
            is_global_mask = [i < self.n_global for i in range(4)]

        for i, idx in enumerate(indices):
            path = self.img_paths[idx]
            label = self.labels[idx]
            img_i = cv2_imread_safe(path)
            if img_i is None:
                patches_info.append((self._make_soft_label(label), 0, 0))
                patch_cls.append(int(label))
                continue

            x, y, rw, rh = regions[i]
            patch, energy = self._extract_patch(img_i, rw, rh, is_global=is_global_mask[i])

            final_img[y:y + rh, x:x + rw] = patch
            # 消融核心：使用 _make_soft_label 替代硬编码的 gaussian_ordinal_label
            label_soft = self._make_soft_label(label)
            patches_info.append((label_soft, rw * rh, energy))
            patch_cls.append(int(label))

        # 权重计算（与原版完全一致）
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
        patch_labels = torch.tensor(patch_cls, dtype=torch.long)
        return transformed['image'], final_label, orig_img_tensor, weights, patch_labels
