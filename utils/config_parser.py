import yaml
import argparse
from types import SimpleNamespace


def merge_dicts(dict1, dict2):
    """递归合并字典"""
    for k, v in dict2.items():
        if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
            merge_dicts(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def parse_config():
    parser = argparse.ArgumentParser()
    # 允许通过命令行覆盖具体的 config 文件
    parser.add_argument('--config-train', default='configs/resume.yaml')
    parser.add_argument('--config-dataset', default='configs/dataset/subset_100.yaml')
    parser.add_argument('--config-model', default='configs/model/resnet18.yaml')
    parser.add_argument('--config-augment', default='configs/augment/base.yaml')

    # 允许命令行临时覆盖参数 (如 --seed 42)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--exp-dir', type=str, default=None)
    # 合并后新增（老脚本不传时行为不变）
    parser.add_argument('--dataset-impl', type=str, default=None)
    parser.add_argument('--exp-root', type=str, default=None)
    parser.add_argument('--no-resume', action='store_true')

    # [packaged] 原为 parse_args()：一旦调用方多传一个未知参数就直接报错退出。
    # 改为 parse_known_args() 后，各脚本可自由追加自己的开关，行为对已知参数完全一致。
    args, _ = parser.parse_known_args()

    # 读取并合并 YAML
    final_config = {}
    for cfg_path in [args.config_train, args.config_dataset, args.config_model, args.config_augment]:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            final_config = merge_dicts(final_config, cfg)

    # 命令行参数覆盖 YAML
    if args.seed is not None: final_config['train']['seed'] = args.seed

    # 转为 Object 访问 (cfg.train.lr)
    def dict_to_obj(d):
        if not isinstance(d, dict): return d
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})

    return dict_to_obj(final_config)