# utils/reproducibility.py
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    全方位的随机种子设置，确保实验可复现。

    Args:
        seed (int): 随机种子数值
    """
    if seed is None:
        return

    # 1. Python 内置 random
    random.seed(seed)

    # 2. 环境变量 (影响 Hash 算法等)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 3. Numpy
    np.random.seed(seed)

    # 4. PyTorch CPU
    torch.manual_seed(seed)

    # 5. PyTorch GPU (单卡 & 多卡)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 6. CuDNN 确定性设置 (关键!)
        # benchmark = True 会让 cudnn 自动寻找最快的算法，导致结果波动
        # benchmark = False 配合 deterministic = True 才能保证结果一致
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f" [Reproducibility] Random seed set to {seed}. (CuDNN Deterministic Mode: ON)")


def worker_init_fn(worker_id):
    """
    用于 DataLoader 的 worker_init_fn
    防止多进程加载数据时，不同 worker 生成相同的随机数序列
    """
    # 获取主进程当前的 seed
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)