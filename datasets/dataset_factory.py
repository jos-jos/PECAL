# -*- coding: utf-8 -*-
"""
dataset_factory.py 


可用 --dataset-impl 或配置里的 dataset_impl 显式覆盖：
    1. label_type 存在                      -> ablation（只有它支持软标签类型）
    2. 需要 patch_labels 做同类掩码         -> ablation
    3. patch_mode / local_crop_mode 存在    -> dualstream2_ablation
    4. n_global / random_position 存在      -> n_global
    5. 否则                                 -> dualstream2

⚠️ 注意 1 优先于 3：Table 5 的 random_area / random_energy 同时含
   local_crop_mode，但它们走的是 ablation 链路（配置了 label_type），
   因此 local_crop_mode 必须在 mixing_dataset_ablation 里实现才生效
   —— 该参数已同步补进 ablation 链路（见该文件 _extract_patch）。
"""
from .cls_dataset import ClassificationDataset
from .dataset_factory_ablation import get_stratified_subset

# 实现标识
IMPL_ABLATION = 'ablation'
IMPL_DUALSTREAM2_ABLATION = 'dualstream2_ablation'
IMPL_N_GLOBAL = 'n_global'
IMPL_DUALSTREAM2 = 'dualstream2'

_IMPL_CHOICES = (IMPL_ABLATION, IMPL_DUALSTREAM2_ABLATION, IMPL_N_GLOBAL, IMPL_DUALSTREAM2)

# 延迟导入表：只有被选中的实现才会 import，避免四份重量级模块全部加载
_IMPL_MODULES = {
    IMPL_ABLATION: ('mixing_dataset_ablation', 'MixingDataset'),
    IMPL_DUALSTREAM2_ABLATION: ('mixing_dataset3_dualstream2_ablation', 'MixingDataset'),
    IMPL_N_GLOBAL: ('mixing_dataset_n_global', 'MixingDataset'),
    IMPL_DUALSTREAM2: ('mixing_dataset3_dualstream2', 'MixingDataset'),
}


def _load_impl(impl):
    import importlib
    mod_name, cls_name = _IMPL_MODULES[impl]
    mod = importlib.import_module('datasets.' + mod_name)
    return getattr(mod, cls_name)


def resolve_impl(cfg, override=None):
    """根据配置推断应使用哪个 MixingDataset 实现。"""
    if override:
        override = str(override)
        if override not in _IMPL_CHOICES:
            raise ValueError(f"Unknown dataset_impl: {override}，可选 {_IMPL_CHOICES}")
        return override

    aug = cfg.augment
    has = lambda k: getattr(aug, k, None) is not None

    # 1. 软标签类型 —— 仅 ablation 实现支持
    if has('label_type'):
        return IMPL_ABLATION
    # 2. 需要 patch_labels 做同类掩码 —— 仅 ablation 返回 5 元组
    if has('same_class_mask_mode') or str(getattr(aug, 'consistency_schedule', '')) == 'same_class_mask':
        return IMPL_ABLATION
    # 3. 图块构造方式
    if has('patch_mode') or has('local_crop_mode'):
        return IMPL_DUALSTREAM2_ABLATION
    # 4. 全局/局部比例
    if has('n_global') or has('random_position'):
        return IMPL_N_GLOBAL
    # 5. 基础
    return IMPL_DUALSTREAM2


def create_dataset(cfg, img_paths, labels, transform, is_train=True,
                   num_classes=None, impl=None):
    """
    统一 create_dataset —— 五个训练脚本共用。

    impl: 显式指定实现（对应 --dataset-impl），None 时自动推断。
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

    if strategy not in ('mosaic', 'ricap'):
        print(f"Dataset Factory: Using Standard Dataset (strategy={strategy})")
        return ClassificationDataset(img_paths, labels, transform=transform)

    ricap_mode = getattr(cfg.augment, 'ricap_mode', 'curriculum')
    prob = getattr(cfg.augment, 'prob', 0.5)
    saliency_dir = getattr(cfg.dataset, 'saliency_dir', None)

    if strategy == 'mosaic':
        dual_stream = False
        grid_type = 'uniform'
    else:
        dual_stream = getattr(cfg.augment, 'dual_stream', False)
        grid_type = getattr(cfg.augment, 'grid_type', 'random')

    chosen = resolve_impl(cfg, override=impl)
    MixingDataset = _load_impl(chosen)

    n_global = getattr(cfg.augment, 'n_global', 2)
    random_position = getattr(cfg.augment, 'random_position', False)
    label_type = getattr(cfg.augment, 'label_type', 'gaussian')
    switch_epoch = getattr(cfg.augment, 'switch_epoch', 100)
    label_smooth_sigma = float(getattr(cfg.augment, 'label_smooth_sigma', 0.5))
    patch_mode = getattr(cfg.augment, 'patch_mode', 'dual_stream')
    local_crop_mode = str(getattr(cfg.augment, 'local_crop_mode', 'guided'))

    print(f"[Unified Factory] impl={chosen} Strategy={strategy}, "
          f"Sampling={ricap_mode}, Weighting={weighting_mode}, "
          f"Dual={dual_stream}, Grid={grid_type}")

    common = dict(
        img_paths=img_paths,
        labels=labels,
        img_size=cfg.train.img_size,
        num_classes=num_classes,
        ricap_mode=ricap_mode,
        prob=prob,
        subset_ratio=1.0,
        dual_stream=dual_stream,
        grid_type=grid_type,
        seed=cfg.train.seed,
        transform=transform,
        weighting_mode=weighting_mode,
        saliency_dir=saliency_dir,
    )

    # 只传目标实现真正接受的参数，避免 TypeError
    if chosen == IMPL_ABLATION:
        extra = dict(label_smooth_sigma=label_smooth_sigma,
                     n_global=n_global,
                     random_position=random_position,
                     label_type=label_type,
                     switch_epoch=switch_epoch,
                     local_crop_mode=local_crop_mode)
    elif chosen == IMPL_DUALSTREAM2_ABLATION:
        # ⚠️ 该实现的 __init__ 不接受 switch_epoch（旧 factory 曾传它，只是碰巧没走到这条分支）
        extra = dict(label_smooth_sigma=label_smooth_sigma,
                     patch_mode=patch_mode,
                     local_crop_mode=local_crop_mode)
    elif chosen == IMPL_N_GLOBAL:
        extra = dict(label_smooth_sigma=label_smooth_sigma,
                     n_global=n_global,
                     random_position=random_position)
    else:
        extra = dict(label_smooth_sigma=label_smooth_sigma)

    return MixingDataset(**common, **extra)
