"""
测试训练好的模型在三个站点上的性能，计算多种指标，并保存结果到CSV文件。
统计指标包括：
- Accuracy
- Macro F1-score
- Macro Recall  
- Macro Precision
- Quadratic Weighted Kappa
- Adjacent F1 (mean of F1 scores for adjacent class pairs)
- MAE (Mean Absolute Error)
- Hard F1 (F1-score calculated only on the "hard" transitional classes 21,31,41,61,71,81)
每个站点的样本级预测结果（原始标签、预测标签、真实标签）也会保存到一个 .npz 文件中，便于后续分析。把每个样本的完整路径也保存下来。
使用方法示例：
python test3.py --model resnet18 --exp_dir ./experiments --strategies Energy_curriculum_dualstream_strongOrdinaryMix --test_root /path/to/test --output_csv test_results.csv --exclude_11_99
其中：
- --model: 模型名称（如 resnet18, vgg, mobilenetv2, efficientnet_b0, vit_b16）
- --exp_dir: 基础实验目录，模型检查点应位于 exp_dir/model_strategy/best.pth
- --strategies: 要测试的策略列表（默认为 ["Energy_curriculum_dualstream_strongOrdinaryMix"]）
- --test_root: 包含站点文件夹的测试数据根目录
- --output_csv: 保存聚合结果的CSV文件路径
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, cohen_kappa_score
from PIL import Image
import glob
from tqdm import tqdm
import argparse
from collections import OrderedDict

# ================= 配置区域 =================
TEST_ROOT = "PECAL/data/test" 
CLASS_MAPPING = { 21: 0, 31: 1, 41: 2, 61: 3, 71: 4, 81: 5, 91: 6}

SITES = [
    "O3543-2022",
    "O8562-2022",
    "O8563-2022",
    "O6965-2021",
    "O9113-2018"
]

IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
NUM_CLASSES = len(CLASS_MAPPING)

IMG_SIZE = 224
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SiteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for class_name in CLASS_MAPPING.keys():
            class_dir = os.path.join(root_dir, str(class_name))
            if not os.path.isdir(class_dir):
                continue
            # 获取所有图片文件，确保顺序一致
            img_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG')
            img_paths = []
            for ext in img_extensions:
                img_paths.extend(glob.glob(os.path.join(class_dir, ext)))
            img_paths.sort()  # 排序以确保每次加载顺序一致
            
            for p in img_paths:
                self.samples.append((p, class_name))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir} with class subfolders.")
        print(f"Loaded {len(self.samples)} images from {root_dir}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir} with class subfolders.")
        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, raw_label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = CLASS_MAPPING[raw_label]
        return image, label, raw_label, path  # ← 新增：返回完整路径

def load_model(model_name, checkpoint_path, num_classes):
    if model_name == "resnet18":
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg":
        from torchvision.models import vgg16_bn
        model = vgg16_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "mobilenetv2":
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif model_name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "vit_b16":
        from torchvision.models import vit_b_16
        model = vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "swin_tiny" or model_name == "swin_tiny_patch4_window7_224":
        # 添加 Swin Tiny 支持
        try:
            from timm import create_model
        except ImportError:
            raise ImportError("timm 库未安装，请运行: pip install timm")
        
        # 创建 Swin Tiny 模型
        model = create_model(
            "swin_tiny_patch4_window7_224", 
            pretrained=False,  # 设置为 False，因为我们有自己的预训练权重
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model

def calculate_adjacent_f1(filtered_labels, filtered_preds, excluded_classes=None):
    """
    计算相邻类别的 F1 分数
    
    Args:
        filtered_labels: 过滤后的真实标签
        filtered_preds: 过滤后的预测标签
        excluded_classes: 被排除的类别索引列表
    """
    if excluded_classes is None:
        excluded_classes = []
    
    # 获取过滤后剩余的类别索引
    all_classes = np.unique(np.concatenate([filtered_labels, filtered_preds]))
    # 排除指定的类别
    remaining_classes = np.array([c for c in all_classes if c not in excluded_classes])
    remaining_classes.sort()
    
    adjacent_f1s = []
    
    # 只计算剩余的相邻类别对
    for i in range(len(remaining_classes) - 1):
        cls_i = remaining_classes[i]
        cls_j = remaining_classes[i + 1]
        
        # 检查这两个类别是否是原始定义中的相邻类别
        # 类别顺序: 11(0), 21(1), 31(2), 41(3), 61(4), 71(5), 81(6), 91(7), 99(8)
        original_adjacent_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)
        ]
        
        # 只有原始定义中的相邻类别对才计算
        if (cls_i, cls_j) in original_adjacent_pairs:
            # 创建只包含这两个类别的样本子集
            mask_pair = (filtered_labels == cls_i) | (filtered_labels == cls_j) | \
                       (filtered_preds == cls_i) | (filtered_preds == cls_j)
            
            if np.sum(mask_pair) == 0:
                continue
                
            sub_preds = filtered_preds[mask_pair]
            sub_labels = filtered_labels[mask_pair]
            
            # 将这两类重新映射为0和1
            sub_preds_bin = np.where((sub_preds == cls_i) | (sub_preds == cls_j), sub_preds, -1)
            sub_labels_bin = np.where((sub_labels == cls_i) | (sub_labels == cls_j), sub_labels, -1)
            
            # 只保留属于这两类的样本
            keep = (sub_labels_bin != -1) & (sub_preds_bin != -1)
            if np.sum(keep) == 0:
                continue
                
            # 将 cls_j 映射为 1，cls_i 映射为 0
            pred_bin = (sub_preds_bin[keep] == cls_j).astype(int)
            true_bin = (sub_labels_bin[keep] == cls_j).astype(int)
            
            f1_adj = f1_score(true_bin, pred_bin, zero_division=0)
            adjacent_f1s.append(f1_adj)
    
    return np.mean(adjacent_f1s) if adjacent_f1s else np.nan

def evaluate_site(model, site_dir, site_name, exclude_11_99=True):
    """
    评估单个站点的模型性能，并可选择排除类别 11 和 99。
    Args:
        exclude_11_99: 如果为 True，则计算指标时排除类别 11 和 99 的样本。
    """
    dataset = SiteDataset(site_dir, transform=test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []
    all_raw_labels = []  # 存储原始标签 (11, 21, ..., 99)
    all_image_paths = []  # ← 新增：存储图片路径

    with torch.no_grad():
        for images, labels, raw_labels, image_paths in tqdm(loader, desc=f"Evaluating {site_name}"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_raw_labels.extend(raw_labels)  # raw_labels 是原始类别号
            all_image_paths.extend(list(image_paths))  # ← 新增：保存路径

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_raw_labels = np.array(all_raw_labels)  # 转换为 numpy 数组方便过滤
    all_image_paths = np.array(all_image_paths)  # ← 新增


    # ----------------- 关键修改：创建过滤掩码 -----------------
    if exclude_11_99:
        # 保留原始标签为 21, 31, 41, 61, 71, 81, 91 的样本
        mask = np.isin(all_raw_labels, [21, 31, 41, 61, 71, 81, 91])
        print(f"  [信息] 评估已排除类别 11 和 99。有效样本数: {np.sum(mask)} / {len(mask)}")
    else:
        # 使用所有样本
        mask = np.ones_like(all_raw_labels, dtype=bool)
        print(f"  [信息] 评估使用全部类别。总样本数: {len(mask)}")

    # 应用过滤
    filtered_preds = all_preds[mask]
    filtered_labels = all_labels[mask]
    filtered_raw_labels = all_raw_labels[mask]

    # 后续所有计算均基于 filtered_preds 和 filtered_labels
    calc_labels = filtered_labels
    calc_preds = filtered_preds

    # 1. 整体宏观指标 (基于过滤后数据)
    if len(calc_labels) == 0:
        print(f"  [警告] 站点 {site_name} 在过滤后无有效样本用于计算指标。")
        # 返回NaN或默认值
        overall = {k: np.nan for k in ['accuracy', 'f1_macro', 'recall_macro', 'precision_macro', 'quadratic_kappa', 'adjacent_f1_mean', 'mae', 'hard_f1_macro']}
        per_class_metrics = {}
        return overall, per_class_metrics, all_raw_labels, all_preds, all_labels

    acc = accuracy_score(calc_labels, calc_preds)
    f1_macro = f1_score(calc_labels, calc_preds, average='macro')
    recall_macro = recall_score(calc_labels, calc_preds, average='macro')
    precision_macro = precision_score(calc_labels, calc_preds, average='macro')


    # 2. 有序一致性 (Quadratic Weighted Kappa)
    try:
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    except ValueError:
        kappa = np.nan

    # 3. 边界混淆: 相邻类别对的 F1 平均值
    if exclude_11_99:
        # 排除类别 0 和 8（对应 11 和 99）
        excluded_indices = [0, 8]
    else:
        excluded_indices = []
    adj_f1_mean = calculate_adjacent_f1(calc_labels, calc_preds, excluded_indices)
    

    # 4. 误差距离 (MAE)
    mae = np.mean(np.abs(calc_preds - calc_labels))

    # 5. 困难子集: 过渡期类别 (索引 1~6, 即原始标签 21,31,41,61,71,81)
    hard_classes = [1,2,3,4,5,6]   # 索引
    # 提取这些类别的样本
    mask_hard = np.isin(all_labels, hard_classes)
    if np.sum(mask_hard) > 0:
        hard_preds = all_preds[mask_hard]
        hard_labels = all_labels[mask_hard]

        # 注意：计算宏平均F1时，只考虑这些类别的出现，未出现的类别会报错，需要手动计算
        f1_per_class = []
        for cls in hard_classes:
            tp = np.sum((hard_preds == cls) & (hard_labels == cls))
            fp = np.sum((hard_preds == cls) & (hard_labels != cls))
            fn = np.sum((hard_preds != cls) & (hard_labels == cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_per_class.append(f1)
        hard_f1_macro = np.mean(f1_per_class)
    else:
        hard_f1_macro = np.nan

    # 6. 各类别详细指标
    # 获取数据中实际存在的类别
    unique_labels = np.unique(calc_labels)
    target_names = [str(IDX_TO_CLASS[i]) for i in unique_labels]

    # 使用 labels 参数指定要报告的类别
    report = classification_report(
        calc_labels, 
        calc_preds, 
        labels=unique_labels,
        target_names=target_names, 
        output_dict=True, 
        zero_division=0
    )
    per_class_metrics = {}
    for class_idx in np.unique(calc_labels):
        class_name = str(IDX_TO_CLASS[class_idx])
        per_class_metrics[class_name] = {
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1-score': report[class_name]['f1-score'],
            'support': report[class_name]['support']
        }

    overall = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'recall_macro': recall_macro,
        'precision_macro': precision_macro,
        'quadratic_kappa': kappa,
        'adjacent_f1_mean': adj_f1_mean,
        'mae': mae,
        'hard_f1_macro': hard_f1_macro,
    }
    # 返回所有样本的原始信息，以便保存
    return overall, per_class_metrics, all_raw_labels, all_preds, all_labels, all_image_paths

def main():
    parser = argparse.ArgumentParser(description="Test trained models on three sites")
    parser.add_argument("--model", type=str, default="resnet18", help="Model name (e.g., resnet18, vgg)")
    parser.add_argument("--exp_dir", type=str, default="./experiments/ablation_mixing_methods",  help="Base experiment directory")
    parser.add_argument("--strategies", nargs="+", default=None, help="List of strategy names to test")
    parser.add_argument("--test_root", type=str, default=TEST_ROOT, help="Root directory containing site folders")
    parser.add_argument("--output_csv", type=str, default="test_results_periods_mixing.csv", help="Output CSV file for aggregated results")
    parser.add_argument("--exclude_11_99", action='store_true', help="Exclude classes 11 and 99 from evaluation metrics.")
    args = parser.parse_args()

    if args.strategies is None:
        STRATEGIES = [
            # "Baseline", "Strong", "CutMix", "Mixup", "Mosaic", "RICAP", "Energy_random", "Energy_neighbor", "Energy_intra", "Area_random", "Area_neighbor",
            # "Area_intra", "Area_neighbor", "Fixed_random", "Fixed_neighbor", "Fixed_intra"
            # "Energy_intra_smooth", "Energy_curriculum", "Area_intra_smooth", "Fixed_intra_smooth",
            # "Energy_curriculum_dualstream_strongOnMix", "Energy_curriculum_dualstream_patience_strongOnMix",
            # "Energy_curriculum_consistency",
            # "Energy_curriculum_dualstream_strongOriginal",
            # "Energy_curriculum_dualstream_strongOrdinaryMix",
            # "Energy_curriculum_consistency_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOnMix_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginalMix_exclude_11_99",
            # "Baseline_exclude_11_99",
            # "Strong_exclude_11_99",
            # "Mixup_exclude_11_99",
            # "CutMix_exclude_11_99",
            # "Mosaic_exclude_11_99",
            # "RICAP_exclude_11_99",
            # "Energy_random_exclude_11_99",
            # "Energy_intra_exclude_11_99",
            # "Energy_neighbor_exclude_11_99",
            # "Area_random_exclude_11_99",
            # "Fixed_neighbor_exclude_11_99",
            # "Area_intra_exclude_11_99",
            # "Area_neighbor_exclude_11_99",
            # "Fixed_random_exclude_11_99",
            # "Fixed_intra_exclude_11_99",
            # "Energy_curriculum_consistency_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOnMix_aug2_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_exclude_11_99",
            # "RICAP_ordinal_dual_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug2_exclude_11_99",
            # "atest_Energy_curriculum_dualstream_strongOriginal_aug2_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug3_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug2_synced_geom_noCurriculum_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug4_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug3_noOrdinal_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug2_0.3Mix_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug2_meanTeach_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug2_consis_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug5_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug5_meanTeach_exclude_11_99",
            # "Adaptive_curriculum_dualstream_strongOriginal_aug2_meanTch_exclude_11_99",
            # "Adaptive2_curriculum_dualstream_strongOriginal_aug2_meanTch_exclude_11_99",
            # "Adaptive2_intra_dualstream_strongOriginal_aug2_meanTch_exclude_11_99",
            # "Adaptive2_intra_dualstream_strongOriginal_aug2_exclude_11_99",
            # "Adaptive2_curriculum_dualstream_strongOriginal_aug2_exclude_11_99",
            # "Adaptive2_curriculum_dualstream_strongOriginal2_aug2_exclude_11_99",
            # "Adaptive2_intra_dualstream_strongOriginal_klVers_aug2_exclude_11_99",
            # "Adaptive2_curriculum_dualstream_stageaware_strongOriginal2_aug2_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug2&4_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug2_visNorm2_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug2_shuffle_exclude_11_99"
            # "Energy_curriculum_dualstream_strongOriginal2_aug2_shuffle_exclude_11_99"
            # "atest_Energy_curriculum_dualstream_strongOriginal_aug2_exclude_11_99",
            # "Energy_curriculum_dualstream_strongOriginal_aug2_lossCons_exclude_11_99"，
            "ablation_resize_all",
            "ablation_random_crop_all"

        ]
    else:
        STRATEGIES = args.strategies

    results = []

    for strategy in STRATEGIES:
        exp_name = f"{strategy}"
        # exp_name = f"{args.model}_100pct_{strategy}"
        exp_dir = os.path.join(args.exp_dir, exp_name)
        # checkpoint_path = os.path.join(exp_dir, "Ep90.pth")
        checkpoint_path = os.path.join(exp_dir, "best.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}, skipping {strategy}")
            continue

        print(f"\n========== Testing strategy: {strategy} ==========")
        print(f"Loading model from {checkpoint_path}")
        model = load_model(args.model, checkpoint_path, NUM_CLASSES)

        site_results = {}
        all_samples_info = {}  # 新增：保存每个站点的样本信息
        for site in SITES:
            site_dir = os.path.join(args.test_root, site)
            if not os.path.isdir(site_dir):
                print(f"Site directory not found: {site_dir}, skipping")
                continue

            # 调用时传入参数
            overall, per_class, raw_labels_arr, preds_idx_arr, true_labels_idx_arr, image_paths_arr = evaluate_site(
                model, site_dir, site, exclude_11_99=args.exclude_11_99
            )

            site_results[site] = (overall, per_class)
            all_samples_info[site] = (raw_labels_arr, preds_idx_arr, true_labels_idx_arr, image_paths_arr)

            print(f"\n--- {site} ---")
            print(f"Accuracy: {overall['accuracy']:.4f}")
            print(f"Macro F1: {overall['f1_macro']:.4f}")
            print(f"Macro Recall: {overall['recall_macro']:.4f}")
            print(f"Quadratic Kappa: {overall['quadratic_kappa']:.4f}")
            print(f"Adjacent F1 (mean): {overall['adjacent_f1_mean']:.4f}")
            print(f"MAE: {overall['mae']:.4f}")
            print(f"Hard (transitional) F1: {overall['hard_f1_macro']:.4f}")
            print("Per-class metrics:")
            if per_class:
                print("Per-class metrics:")
                for class_name, metrics in per_class.items():
                    print(f"  {class_name}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f} (n={metrics['support']})")

        # 保存样本级预测结果
        if all_samples_info:
            sample_save_path = os.path.join(exp_dir, f"sample_predictions_{strategy}_nd3.npz")
            
            # 创建保存字典
            save_dict = {}
            for site in all_samples_info:
                raw_arr, pred_arr, true_arr, path_arr = all_samples_info[site]
                save_dict[f'{site}_raw'] = raw_arr
                save_dict[f'{site}_pred'] = pred_arr
                save_dict[f'{site}_true'] = true_arr
                save_dict[f'{site}_path'] = path_arr  # ← 新增：保存路径
            
            np.savez(sample_save_path, **save_dict)
            print(f"\n样本级预测结果已保存至: {sample_save_path}")

        for site, (overall, per_class) in site_results.items():
            row = {
                'strategy': strategy,
                'site': site,
                'accuracy': overall['accuracy'],
                'macro_f1': overall['f1_macro'],
                'macro_recall': overall['recall_macro'],
                'macro_precision': overall['precision_macro'],
                'quadratic_kappa': overall['quadratic_kappa'],
                'adjacent_f1_mean': overall['adjacent_f1_mean'],
                'mae': overall['mae'],
                'hard_f1_macro': overall['hard_f1_macro'],
            }
            for class_name, m in per_class.items():
                row[f'{class_name}_precision'] = m['precision']
                row[f'{class_name}_recall'] = m['recall']
                row[f'{class_name}_f1'] = m['f1-score']
                row[f'{class_name}_support'] = m['support']
            results.append(row)

    df = pd.DataFrame(results)

    # 根据是否排除 11/99 修改输出文件名
    if args.exclude_11_99:
        output_file = args.output_csv.replace('.csv', '_exclude_11_99.csv')
    else:
        output_file = args.output_csv

    # 追加写入 CSV：如果文件不存在/为空，则写入表头；如果已存在，则追加到末尾，不覆盖原结果
    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)

    if file_exists:
        print(f"\nAll results appended to existing CSV: {output_file}")
    else:
        print(f"\nAll results saved to new CSV: {output_file}")

if __name__ == "__main__":
    main()