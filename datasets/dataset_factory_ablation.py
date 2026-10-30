"""
消融实验版 Dataset Factory
"""
import random
from collections import defaultdict
from .cls_dataset import ClassificationDataset
from .mixing_dataset_ablation import MixingDataset
import sys
import io

# Windows UTF-8 修复（幂等）。切勿无条件重复包装：第二次包装会让第一个
# TextIOWrapper 被析构并关闭共享缓冲区，使后续 print 全部失败。
if sys.platform == 'win32':
    try:
        if getattr(sys.stdout, 'encoding', '').lower().replace('-', '') != 'utf8':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                          errors='replace', line_buffering=True)
        except Exception:
            pass


def get_stratified_subset(img_paths, labels, ratio, seed):
    if ratio >= 1.0:
        return img_paths, labels

    print(f"Stratified Sampling: Keeping {ratio * 100}% of data per class...")

    class_map = defaultdict(list)
    for path, label in zip(img_paths, labels):
        class_map[label].append(path)

    final_paths = []
    final_labels = []

    rng = random.Random(seed)
    sorted_classes = sorted(class_map.keys())

    total_kept = 0
    for cls in sorted_classes:
        paths = class_map[cls]
        total = len(paths)
        keep_count = int(total * ratio)
        if keep_count < 1 and total > 0:
            keep_count = 1
        rng.shuffle(paths)
        selected_paths = paths[:keep_count]
        final_paths.extend(selected_paths)
        final_labels.extend([cls] * len(selected_paths))
        total_kept += len(selected_paths)

    combined = list(zip(final_paths, final_labels))
    rng.shuffle(combined)
    final_paths, final_labels = zip(*combined)

    print(f"   Subset Created: {len(img_paths)} -> {total_kept} images.")
    return list(final_paths), list(final_labels)


def create_dataset(cfg, img_paths, labels, transform, is_train=True, num_classes=None):
    """
    Dataset 工厂函数（消融实验版）
    """
    if not is_train:
        return ClassificationDataset(img_paths, labels, transform=transform)

    subset_ratio = cfg.dataset.subset_ratio

    if subset_ratio < 1.0:
        img_paths, labels = get_stratified_subset(
            img_paths, labels, subset_ratio, seed=cfg.train.seed
        )

    strategy = cfg.augment.mix_strategy
    weighting_mode = getattr(cfg.augment, 'weighting_mode', 'area')

    if strategy in ['mosaic', 'ricap']:
        ricap_mode = getattr(cfg.augment, 'ricap_mode', 'curriculum')
        prob = getattr(cfg.augment, 'prob', 0.5)
        saliency_dir = getattr(cfg.dataset, 'saliency_dir', None)

        if strategy == 'mosaic':
            dual_stream = False
            grid_type = 'uniform'
        elif strategy == 'ricap':
            dual_stream = getattr(cfg.augment, 'dual_stream', False)
            grid_type = getattr(cfg.augment, 'grid_type', 'random')

        # 读取所有消融参数
        n_global = getattr(cfg.augment, 'n_global', 2)
        random_position = getattr(cfg.augment, 'random_position', False)
        label_type = getattr(cfg.augment, 'label_type', 'gaussian')
        switch_epoch = getattr(cfg.augment, 'switch_epoch', 100)
        label_smooth_sigma = getattr(cfg.augment, 'label_smooth_sigma', 0.5)
        # [packaged] Table 5 的 Random+Area / Random+Energy 依赖此开关
        local_crop_mode = str(getattr(cfg.augment, 'local_crop_mode', 'guided'))

        print(f"[Ablation Factory] Strategy={strategy}, "
              f"Sampling={ricap_mode}, Weighting={weighting_mode}, "
              f"Dual={dual_stream}, Grid={grid_type}, "
              f"n_global={n_global}, random_position={random_position}, "
              f"label_type={label_type}, switch_epoch={switch_epoch}, "
              f"local_crop_mode={local_crop_mode}")

        return MixingDataset(
            img_paths=img_paths,
            labels=labels,
            img_size=cfg.train.img_size,
            num_classes=num_classes,
            ricap_mode=ricap_mode,
            prob=getattr(cfg.augment, 'prob', 0.5),
            subset_ratio=1.0,
            dual_stream=dual_stream,
            grid_type=grid_type,
            seed=cfg.train.seed,
            transform=transform,
            weighting_mode=weighting_mode,
            saliency_dir=saliency_dir,
            n_global=n_global,
            random_position=random_position,
            label_type=label_type,                  # <-- 消融：软标签类型
            switch_epoch=switch_epoch,              # <-- 消融：课程切换时机
            label_smooth_sigma=label_smooth_sigma,
            local_crop_mode=local_crop_mode,        # [packaged] guided / random
        )
    else:
        print(f"Dataset Factory: Using Standard Dataset")
        return ClassificationDataset(
            img_paths,
            labels,
            transform=transform
        )
