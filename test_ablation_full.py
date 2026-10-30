"""
批量测试 ablation_full 下的消融实验权重。
在 5 个站点上评估每个实验的 best.pth，计算多指标并保存结果。

站点: O3543-2022, O8562-2022, O8563-2022, O6965-2021, O9113-2018
指标: Accuracy, Macro F1, Macro Recall, Macro Precision, QWK, Adjacent F1, MAE, Hard F1

用法:
    python test_ablation_full.py
    python test_ablation_full.py --exp_dir experiments/ablation_full --model resnet18
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, cohen_kappa_score
)
from PIL import Image
import glob
from tqdm import tqdm
import argparse
from collections import OrderedDict

# ================= 配置区域 =================
TEST_ROOT = "PECAL/corn-cls/test"
CLASS_MAPPING = {21: 0, 31: 1, 41: 2, 61: 3, 71: 4, 81: 5, 91: 6}
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
NUM_CLASSES = len(CLASS_MAPPING)

# [packaged] 本包已放开全部 5 个测试子集（原件只放开 O6965/O9113 两个）
SITES = [
    "O3543-2022",
    "O8562-2022",
    "O8563-2022",
    "O6965-2021",
    "O9113-2018",
]

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
            img_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG')
            img_paths = []
            for ext in img_extensions:
                img_paths.extend(glob.glob(os.path.join(class_dir, ext)))
            img_paths.sort()
            for p in img_paths:
                self.samples.append((p, class_name))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir} with class subfolders.")
        print(f"  Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, raw_label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            import random
            new_idx = random.randint(0, len(self.samples) - 1)
            path, raw_label = self.samples[new_idx]
            image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = CLASS_MAPPING[raw_label]
        return image, label, raw_label, path


def load_model(model_name, checkpoint_path, num_classes):
    if model_name == "resnet18":
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg":
        from torchvision.models import vgg16_bn
        model = vgg16_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "mobilenetv2":
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
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
    if excluded_classes is None:
        excluded_classes = []
    all_classes = np.unique(np.concatenate([filtered_labels, filtered_preds]))
    remaining_classes = np.array([c for c in all_classes if c not in excluded_classes])
    remaining_classes.sort()
    adjacent_f1s = []
    for i in range(len(remaining_classes) - 1):
        cls_i = remaining_classes[i]
        cls_j = remaining_classes[i + 1]
        original_adjacent_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)
        ]
        if (cls_i, cls_j) in original_adjacent_pairs:
            mask_pair = (filtered_labels == cls_i) | (filtered_labels == cls_j) | \
                        (filtered_preds == cls_i) | (filtered_preds == cls_j)
            if np.sum(mask_pair) == 0:
                continue
            sub_preds = filtered_preds[mask_pair]
            sub_labels = filtered_labels[mask_pair]
            sub_preds_bin = np.where((sub_preds == cls_i) | (sub_preds == cls_j), sub_preds, -1)
            sub_labels_bin = np.where((sub_labels == cls_i) | (sub_labels == cls_j), sub_labels, -1)
            keep = (sub_labels_bin != -1) & (sub_preds_bin != -1)
            if np.sum(keep) == 0:
                continue
            pred_bin = (sub_preds_bin[keep] == cls_j).astype(int)
            true_bin = (sub_labels_bin[keep] == cls_j).astype(int)
            f1_adj = f1_score(true_bin, pred_bin, zero_division=0)
            adjacent_f1s.append(f1_adj)
    return np.mean(adjacent_f1s) if adjacent_f1s else np.nan


def evaluate_site(model, site_dir, site_name):
    dataset = SiteDataset(site_dir, transform=test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    all_raw_labels = []
    all_image_paths = []

    with torch.no_grad():
        for images, labels, raw_labels, image_paths in tqdm(loader, desc=f"  {site_name}"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_raw_labels.extend(raw_labels)
            all_image_paths.extend(list(image_paths))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_raw_labels = np.array(all_raw_labels)
    all_image_paths = np.array(all_image_paths)

    mask = np.isin(all_raw_labels, [21, 31, 41, 61, 71, 81, 91])
    print(f"    [info] excluded 11/99. valid: {np.sum(mask)} / {len(mask)}")

    filtered_preds = all_preds[mask]
    filtered_labels = all_labels[mask]

    if len(filtered_labels) == 0:
        print(f"    [WARN] No valid samples for {site_name}")
        overall = {k: np.nan for k in ['accuracy', 'f1_macro', 'recall_macro', 'precision_macro',
                                        'quadratic_kappa', 'adjacent_f1_mean', 'mae', 'hard_f1_macro']}
        return overall, {}, all_raw_labels, all_preds, all_labels, all_image_paths

    acc = accuracy_score(filtered_labels, filtered_preds)
    f1_macro = f1_score(filtered_labels, filtered_preds, average='macro')
    recall_macro = recall_score(filtered_labels, filtered_preds, average='macro')
    precision_macro = precision_score(filtered_labels, filtered_preds, average='macro')

    try:
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    except ValueError:
        kappa = np.nan

    adj_f1_mean = calculate_adjacent_f1(filtered_labels, filtered_preds, excluded_classes=[0, 8])
    mae = np.mean(np.abs(filtered_preds - filtered_labels))

    # Hard F1 (transitional classes: indices 1-6, i.e. 21,31,41,61,71,81)
    hard_classes = [1, 2, 3, 4, 5, 6]
    mask_hard = np.isin(all_labels, hard_classes)
    if np.sum(mask_hard) > 0:
        hard_preds = all_preds[mask_hard]
        hard_labels = all_labels[mask_hard]
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

    # Per-class
    unique_labels = np.unique(filtered_labels)
    target_names = [str(IDX_TO_CLASS[i]) for i in unique_labels]
    report = classification_report(
        filtered_labels, filtered_preds,
        labels=unique_labels, target_names=target_names,
        output_dict=True, zero_division=0
    )
    per_class_metrics = {}
    for class_idx in unique_labels:
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
    return overall, per_class_metrics, all_raw_labels, all_preds, all_labels, all_image_paths


def find_experiments(exp_dir):
    """扫描 exp_dir 下所有含 best.pth 的子目录（支持二级嵌套）"""
    experiments = []
    
    # 如果传入的本身就是一个实验目录，直接加入
    best_path_root = os.path.join(exp_dir, "best.pth")
    if os.path.isdir(exp_dir) and os.path.exists(best_path_root):
        # 取文件夹名
        exp_name = os.path.basename(exp_dir.rstrip(os.sep))
        experiments.append((exp_name, exp_dir))
        return experiments

    for entry in sorted(os.listdir(exp_dir)):
        entry_path = os.path.join(exp_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        # 一级目录直接含 best.pth
        best_path = os.path.join(entry_path, "best.pth")
        if os.path.exists(best_path):
            experiments.append((entry, entry_path))
            continue
        # 二级嵌套（如 n_global_ablation/n_global_0）
        for sub in sorted(os.listdir(entry_path)):
            sub_path = os.path.join(entry_path, sub)
            if not os.path.isdir(sub_path):
                continue
            best_path = os.path.join(sub_path, "best.pth")
            if os.path.exists(best_path):
                experiments.append((f"{entry}/{sub}", sub_path))
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Batch test ablation_full experiments on 5 sites")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--exp_dir", type=str, default="experiments/ablation_full") 
    parser.add_argument("--test_root", type=str, default=TEST_ROOT)
    parser.add_argument("--output_csv", type=str, default="test_results/ablation_full_5sites.csv")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only test experiments matching this substring (e.g. 'curriculum')")
    parser.add_argument("--exact", type=str, default=None,
                        help="Only test the experiment whose name exactly matches (e.g. 'best_combo')")
    args = parser.parse_args()

    experiments = find_experiments(args.exp_dir)
    if args.exact:
        experiments = [(name, path) for name, path in experiments if name == args.exact]
    elif args.filter:
        experiments = [(name, path) for name, path in experiments if args.filter in name]

    if not experiments:
        print(f"No experiments with best.pth found in {args.exp_dir}")
        sys.exit(1)

    print(f"Found {len(experiments)} experiments to test:")
    for name, path in experiments:
        print(f"  - {name}")

    print(f"\nTest sites: {SITES}")
    print(f"Device: {DEVICE}\n")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    results = []

    # 检查已有结果，跳过已测试的 (实验, 站点) 组合（站点级粒度，防止中断后整实验被误跳过）
    tested_pairs = set()
    if os.path.exists(args.output_csv):
        try:
            existing_df = pd.read_csv(args.output_csv)
            tested_pairs = set(zip(existing_df['experiment'], existing_df['site']))
            results = existing_df.to_dict('records')
            print(f"Resuming: {len(tested_pairs)} (exp, site) pairs already tested")
        except Exception:
            pass

    for exp_name, exp_dir in experiments:
        checkpoint_path = os.path.join(exp_dir, "best.pth")
        print(f"\n{'='*60}")
        print(f"Testing: {exp_name}")
        print(f"  Checkpoint: {checkpoint_path}")

        try:
            model = load_model(args.model, checkpoint_path, NUM_CLASSES)
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            continue

        site_results = {}
        all_samples_info = {}

        for site in SITES:
            # 站点级断点续传：跳过已测试的 (实验, 站点) 组合
            if (exp_name, site) in tested_pairs:
                print(f"  [SKIP] {site} (already tested)")
                continue
            site_dir = os.path.join(args.test_root, site)
            if not os.path.isdir(site_dir):
                print(f"  Site directory not found: {site_dir}, skipping")
                continue

            overall, per_class, raw_arr, pred_arr, true_arr, path_arr = evaluate_site(
                model, site_dir, site
            )

            site_results[site] = (overall, per_class)
            all_samples_info[site] = (raw_arr, pred_arr, true_arr, path_arr)

            print(f"  --- {site} ---")
            print(f"    Acc={overall['accuracy']:.4f}  F1={overall['f1_macro']:.4f}  "
                  f"Kappa={overall['quadratic_kappa']:.4f}  MAE={overall['mae']:.4f}  "
                  f"HardF1={overall['hard_f1_macro']:.4f}")

            # 逐站点构建行并增量保存（防止中断丢失已测站点数据）
            row = {
                'experiment': exp_name,
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
            df.to_csv(args.output_csv, index=False)
            print(f"  [增量保存] {site} -> {args.output_csv} (共 {len(results)} rows)")

        # 保存样本级预测结果
        if all_samples_info:
            safe_name = exp_name.replace("/", "_")
            sample_save_path = os.path.join(exp_dir, f"sample_predictions_{safe_name}_5sites.npz")
            save_dict = {}
            for site in all_samples_info:
                raw_arr, pred_arr, true_arr, path_arr = all_samples_info[site]
                save_dict[f'{site}_raw'] = raw_arr
                save_dict[f'{site}_pred'] = pred_arr
                save_dict[f'{site}_true'] = true_arr
                save_dict[f'{site}_path'] = path_arr
            np.savez(sample_save_path, **save_dict)
            print(f"  Sample predictions saved: {sample_save_path}")

        # 释放模型显存
        del model
        torch.cuda.empty_cache()

    # 最终保存 CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\n{'='*60}")
    print(f"All results saved to: {args.output_csv}")
    print(f"Total rows: {len(df)} ({len(experiments)} experiments x {len(SITES)} sites)")

    # 打印汇总表
    print(f"\n{'='*60}")
    print("Summary (Accuracy / Macro F1 / Hard F1):")
    print(f"{'Experiment':<30} {'Site':<16} {'Acc':>8} {'F1':>8} {'HardF1':>8}")
    print("-" * 72)
    for _, row in df.iterrows():
        print(f"{row['experiment']:<30} {row['site']:<16} "
              f"{row['accuracy']:>8.4f} {row['macro_f1']:>8.4f} {row['hard_f1_macro']:>8.4f}")


if __name__ == "__main__":
    main()
