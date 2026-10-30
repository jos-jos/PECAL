# -*- coding: utf-8 -*-
"""
train_common.py — PECAL 五个训练脚本的公共实现（2026-10-30 合并）

合并来源（均在 PECAL_code_package/ 根）：
    train_ablation_full.py        520 行  A/B/C 组敏感性 + Table 5 组件消融
    train_ablation_patch_mode.py  361 行  patch-mode 补充实验
    train_baseline.py             339 行  基础对比（旧管路）→ Table 4
    train_n_global_ablation.py    350 行  D 组全局/局部比例
    train_wheat.py                347 行  跨作物（小麦）

合并前：1570 行里 848 行是跨文件复制（54%），6 个函数在 4~5 个脚本里各存一份。
本模块收纳全部公共部分，行为差异改为配置驱动，见 TrainOptions。

对外主入口：run_training(cfg, exp_dir)  —— 由 train.py 调用。
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# 强增强统一常量（四个原脚本里三份一致，train_baseline.py 一份缺失）
# --------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CONSISTENCY_SCHEDULES = ('none', 'ramp_const', 'ramp_decay', 'same_class_mask')
KL_DIRECTIONS = ('mix_to_strong', 'strong_to_mix', 'bidirectional')


# --------------------------------------------------------------------------
# 一、四个脚本中完全一致的纯函数（逐字搬移）
# --------------------------------------------------------------------------
class SoftTargetCrossEntropy(nn.Module):
    """支持软标签的交叉熵，用于 mixup/cutmix/mosaic/ricap。"""

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        loss = torch.sum(-target * log_probs, dim=-1)
        return loss.mean()


def ordinal_shift_loss(logits, target_soft, num_classes):
    """序数位移损失：预测类别期望 vs 软标签类别期望，Smooth-L1。"""
    pred_probs = F.softmax(logits, dim=1)
    pred_rank = torch.sum(pred_probs * torch.arange(num_classes, device=logits.device), dim=1)
    target_rank = torch.sum(target_soft * torch.arange(num_classes, device=logits.device), dim=1)
    return F.smooth_l1_loss(pred_rank, target_rank)


def save_checkpoint(state, is_best, exp_dir, epoch=None):
    """保存中间检查点 / last.pth / best.pth。"""
    if epoch is not None:
        torch.save(state, os.path.join(exp_dir, f"Ep{epoch}.pth"))
    torch.save(state, os.path.join(exp_dir, "last.pth"))
    if is_best:
        torch.save(state, os.path.join(exp_dir, "best.pth"))


def read_csv(path):
    """读取 img_path,label 两列的 CSV（无表头亦可）。"""
    import csv
    imgs, labels = [], []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for line in reader:
            if len(line) < 2:
                continue
            im = line[0]
            la = line[1]
            imgs.append(im)
            labels.append(la)
    return imgs, labels


# --------------------------------------------------------------------------
# 二、强增强管线（统一版，修掉 train_baseline.py 缺失 Normalize 的缺陷）
# --------------------------------------------------------------------------
def get_strong_augmentation_torch():
    """
    返回 Albumentations 强增强组合（aug2 策略）

    统一要点：
      * 三个消融脚本原为 ToGray→BrightnessContrast→HueSaturation→Normalize→ToTensorV2
      * train_baseline.py 原为 BrightnessContrast→HueSaturation→ToGray→ToFloat，
        **末尾没有 A.Normalize**，导致路径 A 输入值域为 [0,1] 而路径 B 已归一化。
      本函数统一为前者（含 Normalize），输入要求 uint8 [0,255] 的 HWC 数组。
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.ToGray(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# --------------------------------------------------------------------------
# 三、一致性约束（原仅 train_ablation_full.py 有配置版本，其余三处硬编码）
# --------------------------------------------------------------------------
def build_consistency_scheduler(cfg_augment, total_epochs):
    """
    返回 (w, schedule_name)，w(epoch) -> float。

    cfg.augment 读取项（均有默认值）：
        consistency_schedule      : none | ramp_const | ramp_decay | same_class_mask
        consistency_rampup_epochs : ramp-up 长度（默认 80）
        consistency_peak_weight   : 峰值权重（默认 1.0）
        consistency_decay_start   : 衰减起点（默认取 switch_epoch=100）
        consistency_final_weight  : 结束时权重（默认 0.2）
    """
    schedule = str(getattr(cfg_augment, 'consistency_schedule', 'ramp_const'))
    if schedule not in CONSISTENCY_SCHEDULES:
        raise ValueError(f"Unknown consistency_schedule: {schedule}, "
                         f"expected one of {CONSISTENCY_SCHEDULES}")

    rampup = max(float(getattr(cfg_augment, 'consistency_rampup_epochs', 80)), 1e-6)
    peak = float(getattr(cfg_augment, 'consistency_peak_weight', 1.0))
    final_w = float(getattr(cfg_augment, 'consistency_final_weight', 0.2))
    decay_start = float(getattr(cfg_augment, 'consistency_decay_start',
                                getattr(cfg_augment, 'switch_epoch', 100)))

    def w(epoch):
        e = float(epoch)
        if schedule == 'none':
            return 0.0
        base = peak * min(1.0, e / rampup)
        if schedule in ('ramp_const', 'same_class_mask'):
            return base
        if e <= decay_start:
            return base
        span = max(float(total_epochs) - decay_start, 1e-6)
        frac = min(1.0, (e - decay_start) / span)
        return peak + (final_w - peak) * frac

    return w, schedule


def make_same_class_mask(patch_labels, mode='hard'):
    """
    判定每个混合样本是否为"语义一致的类内组合"。

    patch_labels: LongTensor [B, 4]
    mode='hard' : 4 个图块全部同类才为 1.0
    mode='soft' : 与首图块同类的占比（1.0 / 0.75 / 0.5 / 0.25）
    """
    anchor = patch_labels[:, :1]
    same = (patch_labels == anchor).float()
    if mode == 'soft':
        return same.mean(dim=1)
    return (same.sum(dim=1) == patch_labels.size(1)).float()


def compute_consistency_loss(outputs_mixed, outputs_strong, kl_direction, reduction='batchmean'):
    """
    F.kl_div(input, target) = KL(target || input)，input 传 log-probs。

    'mix_to_strong' : KL(p_mix || p_strong) —— 论文式(15) D_KL(p_m ‖ p_s)，默认
    'strong_to_mix' : KL(p_strong || p_mix) —— train_wheat.py 的历史写法
    'bidirectional' : 两者之和

    reduction='none' 时返回逐样本 [B]，供 same_class_mask 加权。
    """
    if kl_direction not in KL_DIRECTIONS:
        raise ValueError(f"Unknown kl_direction: {kl_direction}, expected one of {KL_DIRECTIONS}")

    prob_mixed = F.softmax(outputs_mixed, dim=1)
    prob_strong = F.softmax(outputs_strong, dim=1)
    log_prob_mixed = F.log_softmax(outputs_mixed, dim=1)
    log_prob_strong = F.log_softmax(outputs_strong, dim=1)

    if kl_direction == 'strong_to_mix':
        loss = F.kl_div(log_prob_mixed, prob_strong, reduction='none').sum(dim=1)
    elif kl_direction == 'bidirectional':
        loss = (F.kl_div(log_prob_strong, prob_mixed, reduction='none').sum(dim=1) +
                F.kl_div(log_prob_mixed, prob_strong, reduction='none').sum(dim=1))
    else:
        loss = F.kl_div(log_prob_strong, prob_mixed, reduction='none').sum(dim=1)

    if reduction == 'none':
        return loss
    return loss.mean()      # 等价于 reduction='batchmean'（sum(所有元素)/B）


# --------------------------------------------------------------------------
# 四、批次解包与强增强批次构造
# --------------------------------------------------------------------------
def unpack_batch(batch):
    """
    兼容两种 __getitem__ 返回结构：
        4 元组: (img_mixed, label_mixed, orig_imgs, weights)
                —— mixing_dataset3_dualstream2 / _ablation / _n_global
        5 元组: (img_mixed, label_mixed, orig_imgs, weights, patch_labels)
                —— mixing_dataset_ablation（多出四个图块的类别，供同类掩码用）
    """
    if len(batch) == 5:
        img_mixed, label_mixed, orig_imgs, weights, patch_labels = batch
        return img_mixed, label_mixed, orig_imgs, weights, patch_labels
    if len(batch) == 4:
        img_mixed, label_mixed, orig_imgs, weights = batch
        return img_mixed, label_mixed, orig_imgs, weights, None
    raise ValueError(f"Unexpected batch length {len(batch)}，期望 4 或 5")


def to_uint8_hwc(img_tensor):
    """[C,H,W] float [0,1] -> [H,W,C] uint8 [0,255]，供 Albumentations。"""
    if img_tensor.ndim == 3:
        arr = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    elif img_tensor.ndim == 2:
        arr = np.expand_dims(img_tensor.detach().cpu().numpy(), axis=-1)
    elif img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
        arr = img_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    else:
        raise ValueError(f"Unexpected tensor shape: {tuple(img_tensor.shape)}")
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def build_strong_batch(orig_imgs, strong_aug):
    """对原始图像逐个施加强增强，返回 [B,C,H,W] 已归一化的 tensor。"""
    imgs = [strong_aug(image=to_uint8_hwc(t))['image'] for t in orig_imgs]
    return torch.stack(imgs).to(orig_imgs.device)


# --------------------------------------------------------------------------
# 五、配置读取（默认值 = 各原脚本的历史行为，除已确认的缺陷外）
# --------------------------------------------------------------------------
class TrainOptions(object):
    """
    从 cfg.augment / cfg.train 读取全部训练开关。

    默认值说明（合并时逐个核对过）：
      ordinal_weight    默认 0.5   = ablation_full / patch_mode / n_global / wheat
                                     注：train_baseline.py 历史为 1.0，复现需在配置里显式写 1.0
      kl_direction      默认 mix_to_strong = 论文式(15)，与四个玉米脚本默认一致
                                     注：train_wheat.py 历史为 strong_to_mix
      consistency_*     默认 ramp_const + ramp 80，等价于原 get_consistency_weight(80)
      lr_scheduler      默认 cosine = 四个脚本；train_baseline.py 历史为 cosine_warm_restarts
    """

    def __init__(self, cfg):
        aug = cfg.augment
        self.consistency_schedule = str(getattr(aug, 'consistency_schedule', 'ramp_const'))
        self.kl_direction = str(getattr(aug, 'kl_direction', 'mix_to_strong'))
        self.ordinal_weight = float(getattr(aug, 'ordinal_weight', 0.5))
        self.same_class_mask_mode = str(getattr(aug, 'same_class_mask_mode', 'hard'))
        self.use_consistency = self.consistency_schedule != 'none'
        self.lr_scheduler = str(getattr(cfg.train, 'lr_scheduler', 'cosine'))
        self.warm_restart_t0 = int(getattr(cfg.train, 'warm_restart_t0', 100))
        self.device_ids = list(getattr(cfg.train, 'device_ids', [0, 1, 2, 3]))
        self.patience = int(getattr(cfg.train, 'patience', 15))
        self.amp = bool(getattr(cfg.train, 'amp', False))

        if self.consistency_schedule not in CONSISTENCY_SCHEDULES:
            raise ValueError(f"Unknown consistency_schedule: {self.consistency_schedule}")
        if self.same_class_mask_mode not in ('hard', 'soft'):
            raise ValueError(f"Unknown same_class_mask_mode: {self.same_class_mask_mode}")


# --------------------------------------------------------------------------
# 六、训练 / 验证
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, opts,
                    epoch, total_epochs, strong_aug, consistency_w, writer=None):
    """跑一个 epoch，返回 (avg_loss, acc, mask_hit, mask_tot)。"""
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    mask_hit, mask_tot = 0, 0
    use_mask = (opts.consistency_schedule == 'same_class_mask')

    for batch in loader:
        img_mixed, label_mixed, orig_imgs, _, patch_labels = unpack_batch(batch)
        img_mixed = img_mixed.to(device)
        label_mixed = label_mixed.to(device)
        orig_imgs = orig_imgs.to(device)
        if patch_labels is not None:
            patch_labels = patch_labels.to(device)

        # 路径 A：原始图 -> 强增强（统一为 uint8 -> Normalize -> ToTensorV2）
        img_strong = build_strong_batch(orig_imgs, strong_aug)
        outputs_strong = model(img_strong)
        # 路径 B：混合图
        outputs_mixed = model(img_mixed)

        loss_cls = criterion(outputs_mixed, label_mixed)

        if opts.use_consistency:
            if use_mask and patch_labels is not None:
                kl_per_sample = compute_consistency_loss(
                    outputs_mixed, outputs_strong, opts.kl_direction, reduction='none')
                with torch.no_grad():
                    w_mask = make_same_class_mask(patch_labels, opts.same_class_mask_mode)
                    mask_hit += int(w_mask.sum().item())
                    mask_tot += int(w_mask.numel())
                denom = w_mask.sum().clamp(min=1.0)
                loss_cons = (kl_per_sample * w_mask).sum() / denom
            else:
                loss_cons = compute_consistency_loss(
                    outputs_mixed, outputs_strong, opts.kl_direction, reduction='batchmean')
        else:
            loss_cons = torch.zeros((), device=device)

        if opts.ordinal_weight > 0:
            loss_ord = ordinal_shift_loss(outputs_mixed, label_mixed, label_mixed.size(1))
        else:
            loss_ord = torch.zeros((), device=device)

        cons_w = consistency_w(epoch)
        loss = loss_cls + cons_w * loss_cons + opts.ordinal_weight * loss_ord

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * img_mixed.size(0)
        preds = outputs_mixed.argmax(dim=1)
        targets = label_mixed.argmax(dim=1)
        train_correct += (preds == targets).sum().item()
        train_total += img_mixed.size(0)

    return (train_loss / max(train_total, 1),
            train_correct / max(train_total, 1),
            mask_hit, mask_tot)


@torch.no_grad()
def evaluate(model, val_loader, val_ds, device):
    """验证集评估，返回 (val_loss, val_acc, precision, recall, f1)。"""
    from sklearn.metrics import precision_recall_fscore_support

    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    val_criterion = nn.CrossEntropyLoss()
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        val_loss += val_criterion(outputs, labels).item() * imgs.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    epoch_val_loss = val_loss / len(val_ds)
    val_acc = (np.array(all_preds) == np.array(all_targets)).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0)
    return epoch_val_loss, val_acc, precision, recall, f1


# --------------------------------------------------------------------------
# 七、主训练流程
# --------------------------------------------------------------------------
def run_training(cfg, exp_dir, create_dataset, resume=True):
    """
    完整训练流程。create_dataset 由调用方注入（train.py 用统一分派工厂）。
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torch.utils.tensorboard import SummaryWriter
    from timm.data import Mixup
    from tqdm import tqdm

    from utils.augmentations import get_transforms
    from utils.logger import CSVLogger
    from utils.model_factory import create_model
    from utils.reproducibility import set_seed

    opts = TrainOptions(cfg)
    set_seed(cfg.train.seed)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Results will be saved to: {exp_dir}")

    logger = CSVLogger(os.path.join(exp_dir, "log.csv"), resume=resume)
    writer = SummaryWriter(log_dir=exp_dir)

    # ---------------- 数据 ----------------
    train_transform = get_transforms(cfg.augment, is_train=True)
    val_transform = get_transforms(cfg.augment, is_train=False)

    train_imgs, train_raw_labels = read_csv(cfg.train.train_csv)
    val_imgs, val_raw_labels = read_csv(cfg.train.val_csv)
    all_labels = sorted(list(set(train_raw_labels)))
    num_classes = len(all_labels)
    label_map = {label: idx for idx, label in enumerate(all_labels)}

    train_labels = [label_map[l] for l in train_raw_labels]
    val_labels = [label_map[l] for l in val_raw_labels]

    print(f"   Classes: {num_classes} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")

    train_ds = create_dataset(cfg, train_imgs, train_labels, train_transform,
                              is_train=True, num_classes=num_classes)
    val_ds = create_dataset(cfg, val_imgs, val_labels, val_transform,
                            is_train=False, num_classes=num_classes)

    # 类别均衡采样（逆频加权）
    sampler = None
    if str(getattr(cfg.dataset, 'sampler', 'weighted')) == 'weighted':
        print("Using WeightedRandomSampler for class balance...")
        targets = np.array(train_ds.labels) if hasattr(train_ds, 'labels') else np.array(train_labels)
        class_counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
        class_counts[class_counts == 0] = 1.0
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(),
                                        num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=cfg.train.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.workers, pin_memory=True)

    # ---------------- 模型 ----------------
    print(f"Building Model: {cfg.model.name}...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if resume:
        ckpt_path = os.path.join(exp_dir, "last.pth")
    else:
        ckpt_path = getattr(cfg.model, 'checkpoint_path', None)
    model, checkpoint = create_model(cfg.model.name, num_classes,
                                     pretrained=cfg.model.pretrained,
                                     checkpoint_path=ckpt_path,
                                     device=device)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=opts.device_ids)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=5e-4)

    if opts.lr_scheduler == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=opts.warm_restart_t0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs)

    # 断点续训
    start_epoch = 0
    if resume and checkpoint:
        start_epoch = checkpoint.get('epoch', 0)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state restored")
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Scheduler state restored")

    mix_strategy = cfg.augment.mix_strategy
    using_soft_label = mix_strategy in ['mixup', 'cutmix', 'mosaic', 'ricap']
    criterion = SoftTargetCrossEntropy() if using_soft_label else nn.CrossEntropyLoss()

    mixup_fn = None
    if mix_strategy in ['mixup', 'cutmix']:
        mixup_fn = Mixup(
            mixup_alpha=getattr(cfg.augment, 'mixup_alpha', 1.0) if mix_strategy == 'mixup' else 0.,
            cutmix_alpha=1.0 if mix_strategy == 'cutmix' else 0.,
            prob=getattr(cfg.augment, 'prob', 1.0), switch_prob=0.0, mode='batch',
            label_smoothing=0.1, num_classes=num_classes)

    strong_aug = get_strong_augmentation_torch()
    consistency_w, schedule_name = build_consistency_scheduler(cfg.augment, cfg.train.epochs)

    print(f"\nStart Training: {cfg.train.epochs} Epochs")
    print(f"   Mix: {mix_strategy} | Strong: aug2 | Devices: {opts.device_ids}")
    print(f"   KL direction: {opts.kl_direction}")
    print(f"   Ordinal weight: {opts.ordinal_weight}")
    print(f"   Consistency schedule: {schedule_name}"
          + (f" (mask_mode={opts.same_class_mask_mode})" if schedule_name == 'same_class_mask' else ""))
    print(f"   LR scheduler: {opts.lr_scheduler}")

    best_f1 = 0.0
    no_improve_epochs = 0
    total_epochs = cfg.train.epochs

    for epoch in range(start_epoch, start_epoch + total_epochs):
        start_time = time.time()
        if hasattr(train_ds, 'set_epoch'):
            train_ds.set_epoch(epoch)

        avg_loss, acc, mask_hit, mask_tot = train_one_epoch(
            model, train_loader, criterion, optimizer, device, opts,
            epoch, total_epochs, strong_aug, consistency_w, writer)

        scheduler.step()

        val_loss, val_acc, precision, recall, f1 = evaluate(model, val_loader, val_ds, device)
        duration = time.time() - start_time

        cons_w = consistency_w(epoch)
        writer.add_scalar('train/train_loss', avg_loss, epoch)
        writer.add_scalar('train/train_acc', acc, epoch)
        writer.add_scalar('val/val_loss', val_loss, epoch)
        writer.add_scalar('val/val_acc', val_acc, epoch)
        writer.add_scalar('val/f1', f1, epoch)
        writer.add_scalar('train/consistency_weight', cons_w, epoch)

        cons_rate = (mask_hit / mask_tot) if mask_tot > 0 else float('nan')
        if schedule_name == 'same_class_mask':
            writer.add_scalar('train/same_class_rate', cons_rate, epoch)

        print(f"Ep {epoch+1}/{start_epoch + total_epochs} | {duration:.0f}s | "
              f"TrLoss:{avg_loss:.3f} TrAcc:{acc:.3f} | "
              f"ValLoss:{val_loss:.3f} ValAcc:{val_acc:.3f} F1:{f1:.3f} | consW:{cons_w:.3f}"
              + (f" sameCls:{cons_rate:.3f}" if schedule_name == 'same_class_mask' else ""))

        logger.log(epoch=epoch + 1, train_loss=avg_loss, train_acc=acc,
                   val_loss=val_loss, val_acc=val_acc, val_f1=f1, val_recall=recall,
                   val_precision=precision, time_sec=duration,
                   consistency_weight=cons_w, same_class_rate=cons_rate,
                   consistency_schedule=schedule_name)

        is_best = f1 > best_f1
        if is_best:
            best_f1 = f1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        state = {'epoch': epoch + 1, 'state_dict': model_state,
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'best_f1': best_f1}
        if (epoch + 1) % 10 == 0:
            save_checkpoint(state, is_best, exp_dir, epoch=epoch + 1)
        save_checkpoint(state, is_best, exp_dir)

        if no_improve_epochs >= opts.patience:
            print(f"Early stopping at epoch {epoch+1} (patience={opts.patience})")
            break

    writer.close()
    print(f"Training Finished. Best F1: {best_f1:.4f}")
    return best_f1
