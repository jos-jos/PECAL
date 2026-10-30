import torch.nn.functional as F

def consistency_kl_loss(p_teacher, p_student, T=2.0):
    """
    p_teacher: 已经是 Teacher 预测加权融合后的概率分布 (Shape: [B, num_classes])
    s_logits: Student 对混合大图的直接输出 logits (Shape: [B, num_classes])
    KL(P_teacher || P_student)
    """
    # p_t = F.softmax(p_teacher / T, dim=1)
    # 1. 教师概率本身已经 detach 过了，安全
    # 2. 学生输出转为 Log 空间 (PyTorch kl_div 的要求)
    p_s = F.log_softmax(p_student / T, dim=1)

    loss = F.kl_div(p_s, p_teacher, reduction='batchmean') * (T * T)
    return loss