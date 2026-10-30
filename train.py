# -*- coding: utf-8 -*-
"""
train.py —— PECAL 统一训练入口

用法：
    python train.py --config-augment configs/augment/ablation_label_gaussian.yaml \
                    --exp-dir experiments/ablation_full/label_gaussian

    python train.py --config-augment configs/experiments/full_pecal.yaml \
                    --exp-dir experiments/table5/full_pecal --seed 42

    python train.py --config-augment configs/wheat.yaml \
                    --exp-dir wheat_experiments/resnet18_100pct_Wheat_PECAL

新增参数：
    --dataset-impl {ablation,dualstream2_ablation,n_global,dualstream2}
            显式指定 MixingDataset 实现；不传时按配置自动推断。
    --exp-root experiments|wheat_experiments   输出根目录（默认 experiments）
    --no-resume   从头训，不读取目录里的 last.pth
"""
import os
import sys
import io
import argparse

if sys.platform == 'win32':
    for _name in ('stdout', 'stderr'):
        _stream = getattr(sys, _name)
        try:
            if getattr(_stream, 'encoding', '').lower().replace('-', '') != 'utf8':
                _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            try:
                setattr(sys, _name, io.TextIOWrapper(_stream.buffer,
                                                     encoding='utf-8',
                                                     errors='replace',
                                                     line_buffering=True))
            except Exception:
                pass

from utils.config_parser import parse_config
from utils.train_common import run_training
from datasets.dataset_factory import create_dataset, resolve_impl, _IMPL_CHOICES


def _make_dataset_fn(impl):
    def _create(cfg, img_paths, labels, transform, is_train=True, num_classes=None):
        return create_dataset(cfg, img_paths, labels, transform,
                              is_train=is_train, num_classes=num_classes, impl=impl)
    return _create


def main():
    ap = argparse.ArgumentParser(description="PECAL 统一训练入口")
    ap.add_argument('--exp-dir', type=str, default=None)
    ap.add_argument('--dataset-impl', type=str, default=None, choices=list(_IMPL_CHOICES),
                    help="显式指定 MixingDataset 实现，缺省自动推断")
    ap.add_argument('--exp-root', type=str, default=None,
                    help="输出根目录，默认 experiments（小麦用 wheat_experiments）")
    ap.add_argument('--no-resume', action='store_true',
                    help="不读取 exp-dir 里的 last.pth，从头训练")
    cli, _ = ap.parse_known_args()

    cfg = parse_config()

    # 输出目录：--exp-dir > 配置里的 exp_dir > exp_root/exp_name
    if cli.exp_dir:
        exp_dir = cli.exp_dir
    elif hasattr(cfg, 'exp_dir') and cfg.exp_dir:
        exp_dir = cfg.exp_dir
    else:
        exp_root = cli.exp_root or getattr(cfg.train, 'exp_root', None) or 'experiments'
        exp_dir = os.path.join(exp_root, cfg.train.exp_name)

    resume = (not cli.no_resume) and bool(getattr(cfg.train, 'resume', True))

    impl = cli.dataset_impl or resolve_impl(cfg)
    print(f"[train.py] dataset_impl = {impl} | exp_dir = {exp_dir} | resume = {resume}")

    run_training(cfg, exp_dir, _make_dataset_fn(impl), resume=resume)


if __name__ == "__main__":
    main()
