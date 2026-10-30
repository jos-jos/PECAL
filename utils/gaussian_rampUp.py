import numpy as np

def get_consistency_weight(epoch, rampup_epochs=50, max_weight=0.5):
    """
    使用高斯 S 型曲线进行 Ramp-up
    随着 Epoch 增加，一致性损失的权重从 0 慢慢增长到 max_weight
    rampup_epochs: 生长周期 (比如前 50 轮)
    max_weight: 一致性损失的最大权重（建议设为 0.2 ~ 0.5，防御主导 Loss）
    """
    if epoch < 1:  # 防止 0 轮次数学报错
        return 0.0
    
    if epoch < rampup_epochs:
        p = max(0.0, float(epoch)) / float(rampup_epochs)
        res = np.exp(-5.0 * (1.0 - p) ** 2)
        return float(res) * max_weight
    else:
        return max_weight