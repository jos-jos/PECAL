# utils/model_factory.py
import os

# 设置 Hugging Face 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import timm


def create_model(model_name, num_classes, pretrained=False, checkpoint_path=None, device=torch.device("cpu")):
    """
    统一模型工厂，基于 timm 构建，支持自动修改分类头和加载本地权重。

    Args:
        model_name (str): 模型名称，支持 timm 所有模型 (如 'resnet18', 'efficientnet_b0', 'swin_tiny_patch4_window7_224')
        num_classes (int): 目标分类数
        pretrained (bool): 是否使用 ImageNet 预训练权重
        checkpoint_path (str, optional): 本地 .pth 权重路径 (用于断点续训或加载微调过的模型)

    Returns:
        model (nn.Module): 准备好的 PyTorch 模型
        checkpoint (dict or None): 加载的 checkpoint（如果存在）
    """
    checkpoint = None  # 初始化，防止 checkpoint_path 为 None 时 NameError
    try:
        # 1. 创建模型架构
        # 如果提供了本地 checkpoint_path，这里 pretrained 建议设为 False，避免重复下载
        use_timm_pretrained = pretrained and (checkpoint_path is None)
        model = timm.create_model(
            model_name,
            pretrained=use_timm_pretrained,
            num_classes=num_classes
        )
    except Exception as e:
        print(f"Error: timm 无法加载模型 '{model_name}'。")
        raise e

        # 2. 加载本地自定义权重 (或者本地的 ImageNet 权重)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
        # print(f"Resuming from epoch {start_epoch}...")

        # 兼容性处理：检查是否包含 'state_dict' 键
        state_dict = checkpoint.get('state_dict', checkpoint)

        # A. 移除 DataParallel 的 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v

        # B. [关键修复] 智能过滤：只加载形状匹配的权重
        model_dict = model.state_dict()
        filtered_dict = {}
        mismatch_keys = []

        for k, v in new_state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    mismatch_keys.append(k)

        if len(mismatch_keys) > 0:
            print(
                f"[Shape Mismatch] Ignoring {len(mismatch_keys)} layers (e.g., fc/head) because shapes don't match.")
            print(f"    Ignored keys: {mismatch_keys[:3]} ...")

        # C. 加载过滤后的权重
        missing, unexpected = model.load_state_dict(filtered_dict, strict=False)

        if missing:
            # 过滤掉我们故意忽略的层，只显示真正丢失的
            real_missing = [k for k in missing if k not in mismatch_keys]
            if len(real_missing) > 0:
                print(f"   Missing keys: {len(real_missing)}")

    return model, checkpoint


if __name__ == "__main__":
    # 测试代码
    try:
        # 测试 CNN
        m1 = create_model('resnet18', 10)
        print("ResNet18 created.")

        # 测试 ViT (需要固定输入大小)
        m2 = create_model('swin_tiny_patch4_window7_224', 10)
        print("Swin Transformer created.")

        # 测试 EfficientNet
        m3 = create_model('efficientnet_b0', 10)
        print("EfficientNet created.")

    except Exception as e:
        print(f"Test failed: {e}")