# train / run 合并说明（2026-10-30）

对象目录：`E:\桌面\workbuddy\PECAL_code_package\`

---

## 一、直接回答

| 问题 | 结论 |
|---|---|
| 三个 `run_*.py` 能否直接合并 | **能**，且已合并进 `run.py`。三者本就是同一模式的三份拷贝（实验列表 → 拼命令 → subprocess → 落日志 → 汇总） |
| 五个 `train_*.py` 能否直接合并 | **能**，且已合并进 `train.py` + `utils/train_common.py`。但**不能**把四个 MixingDataset 也揉成一个——它们语义不同，强行合并会改变行为 |

合并后入口从 8 个降到 2 个：

```
train.py                    ← 取代 5 个 train_*.py
run.py                      ← 取代 3 个 run_*.py
utils/train_common.py       ← 公共实现（5 份里 848 行重复全部收在这里）
datasets/dataset_factory.py ← 四选一的分派器
```

旧的 8 个文件**原样保留、未改动**，可随时回退。

---

## 二、合并了什么

### 2.1 `utils/train_common.py`（新增）

| 收纳内容 | 原分布 |
|---|---|
| `SoftTargetCrossEntropy` / `ordinal_shift_loss` / `save_checkpoint` / `read_csv` | 5 份逐字相同 |
| `get_strong_augmentation_torch` | 4 个变体 → 统一为**带 Normalize** 的正确版本 |
| `build_consistency_scheduler` | 原仅 `train_ablation_full` 有；另 3 处是函数内硬编码的 `get_consistency_weight(80)` |
| `compute_consistency_loss` / `make_same_class_mask` | 原仅 `train_ablation_full` 有 |
| `train_one_epoch` / `evaluate` / `run_training` | 5 份，重叠 71%–80% |

### 2.2 `datasets/dataset_factory.py`（新增）

四个工厂骨架逐字相同，差异只在传给 `MixingDataset` 的额外参数，因此**只做分派、不重写语义**：

| impl | 对应实现 | 返回元组 | 支持的关键开关 |
|---|---|---|---|
| `ablation` | `mixing_dataset_ablation` | **5 元组**（多 `patch_labels`） | `label_type` / `switch_epoch` / `n_global` / `random_position` / `local_crop_mode` |
| `dualstream2_ablation` | `mixing_dataset3_dualstream2_ablation` | 4 元组 | `patch_mode` / `local_crop_mode` |
| `n_global` | `mixing_dataset_n_global` | 4 元组 | `n_global` / `random_position` |
| `dualstream2` | `mixing_dataset3_dualstream2` | 4 元组 | —（基础，baseline / wheat 用） |

**impl 必须显式指定，不能只靠自动推断。** 例如组 A 的 `ablation_curriculum_ep50.yaml` 只写了 `switch_epoch`，而基础对比的 `ricap.yaml` 什么都不写——两者都缺可判别的关键字，但历史上前者必须走 `ablation`、后者必须走 `dualstream2`。因此 `run.py` 的 25 条实验逐条写死了 impl。自动推断只作为直接调用 `train.py` 时的 fallback。

### 2.3 `run.py`（新增）

把 6 + 8 + 11 = **25 个实验**收进一张注册表，共用一套循环：

| suite | 数量 | 来源 | impl |
|---|---|---|---|
| `base` | 6 | 《Table 4》六个对比方法 | `dualstream2` |
| `table5` | 8 | 《Table 5》八行组件消融 | `ablation` |
| `sens` | 11 | 《Fig 5》四组敏感性（A/B/C/D） | A/B/C→`ablation`，D→`n_global` |

---

## 三、合并过程中修掉的四个既有问题

### 【P0】Table 5 的 `local_crop_mode` 在主链路静默失效

该参数此前只加在 `mixing_dataset3_dualstream2_ablation.py`（patch-mode 链路），而 **Table 5 走的是 `mixing_dataset_ablation`（ablation 链路），根本不接受这个参数**。

后果：**Random+Area 与 Random+Energy 会训出完全相同的模型**，Table 5 这两行会失去意义。

修复：给 `mixing_dataset_ablation` 加 `local_crop_mode`，并在 `dataset_factory_ablation.py:100,131` 透传。

验证（同种子、同 index）：

```
guided vs random  图像平均像素差 = 0.3198   -> 生效
Random+Area 权重 (0.3394,0.2365,0.2499,0.1742)
Random+Energy 权重 (0.0764,0.0705,0.5262,0.3269)   -> 两者已区分
```

> 2×2 四格里 `guided+area` 与 `random+area` 的权重相同是**数学必然**（面积加权只取决于图块面积，与取窗无关），这两行的差异体现在取窗位置（即上面的 0.3198 像素差），不影响可区分性。

### 【P1】`switch_epoch` 传给不接受它的类

`dataset_factory2_curriculum_ablation.py:111` 把 `switch_epoch` 传给 `mixing_dataset3_dualstream2_ablation.MixingDataset`，但该类的 `__init__` 没有这个形参 —— 之前碰巧没走到这条分支才没暴露。已移除该透传。

### 【P1】`run_ablation_A.py` 配置路径写错

8 个配置实际在 `configs/experiments/`，脚本里却指向 `configs/augment/` → 直接 FileNotFound。合并时已改为正确路径。

### 【P0】`train_ablation_full.py` 曾从包内消失

本次开工时发现该文件不在包内（三个 run 脚本全部依赖它，会直接失效）。已从 20:0x 打的 `PECAL_code_package.zip` 中恢复最完整版本（522 行，含 `ordinal_weight` / `local_crop_mode` / `[packaged]` 全部补丁）。

同类问题：`configs/resume.yaml`（训练脚本默认读取）也丢失，已重建。

> ⚠️ 该目录在 19:52–20:24 之间被持续改动过。动手前已做快照：`E:\桌面\workbuddy\_snapshot_20261030_2024\`

---

## 四、验证结果

| 项目 | 结果 |
|---|---|
| 语法检查（7 个文件） | 通过 |
| `run.py --list` | 25 条全部列出 |
| 25 条实验的配置文件存在性 | 25/25 存在 |
| 评测脚本 `test.py` / `test_ablation_full.py` | 均在包内 |
| `run.py --suite table5 --dry-run` | 8 条命令正确 |
| `run.py --suite sens --group D --dry-run` | 5 条命令正确 |
| **端到端冒烟**（合成数据 112/28，1 epoch，随机初始化 resnet18） | **4/4 impl 全部产出 best.pth** |
| `local_crop_mode` 生效性 | 像素差 0.3198，权重已区分 |

冒烟脚本：`G:\AgriAug-ubuntu\tmp_tasks\merge_check\smoke_train.py`
验证脚本：`G:\AgriAug-ubuntu\tmp_tasks\merge_check\check_localcrop.py`

---

## 五、合并后的行为差异（复现历史需改配置）

统一为正确行为后，复现旧结果需要在配置里显式写回：

| 项 | 合并后默认 | 历史值 | 复现写法 |
|---|---|---|---|
| 强增强路径 A | 做 ImageNet Normalize | `train_baseline.py` **缺失** | 无法复现（这是 bug 修复） |
| 序数损失系数 | 0.5 | `train_baseline.py` 为 **1.0** | 配置写 `ordinal_weight: 1.0` |
| KL 方向 | `mix_to_strong`（=论文式15） | `train_wheat.py` 为 `strong_to_mix` | 配置写 `kl_direction: strong_to_mix` |
| LR 调度 | `cosine` | `train_baseline.py` 为 `CosineAnnealingWarmRestarts(T_0=100)` | 配置写 `lr_scheduler: cosine_warm_restarts` |
| 一致性与序数损失 | 可用 `consistency_schedule: none` / `ordinal_weight: 0` 关闭 | `train_baseline.py` **无条件加上、关不掉** | 无需处理 |

> **结论：旧的 `train_baseline.py` 权重无法 bit-wise 复现**（因为它缺归一化的路径 A 已修）。修这个是必要的，代价是《Table 4》若引用现有数字，需在论文中如实说明或重跑。

---

## 六、迁移方式（二选一，旧文件暂未改动）

### 方式 A：直接改命令（推荐，零风险）

```bash
# 原来
python train_ablation_full.py --config-augment X --exp-dir Y
# 现在
python train.py --config-augment X --exp-dir Y

# 原来
python run_ablation_full.py --group D
# 现在
python run.py --suite sens --group D
```

### 方式 B：把旧脚本换成薄壳（服务器上的既有命令不用改）

```bash
# 以 train_ablation_full.py 为例，内容替换为 4 行：
```

```python
# 已合并进 train.py，本文件仅作兼容入口
import os, sys, runpy
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv[0] = os.path.join(_HERE, 'train.py')
runpy.run_path(sys.argv[0], run_name='__main__')
```

`run_ablation_A.py` / `run_ablation_B.py` 同理指向 `run.py`（注意 `run.py` 用 `--suite` 参数，薄壳里可默认追加 `--suite table5` / `--suite sens`）。

---

## 七、仍需注意的坑（与合并无关，但会咬人）

1. **历史权重目录名不一致**：敏感性 A/B/C 组的权重实际在 `experiments/ablation_full1/`，而脚本里写的是 `experiments/ablation_full/`。要续跑已有权重需手动指定目录。
2. **`train_dual_piModelVar_strongOriginal.py`（旧管路训练器）和 `test4.py` 不在本包内**，但 `run_base.py` 仍在引用它们。合并后 `base` 套件改由 `train.py` + `test.py` 承担。
3. **四个训练脚本都写死了 4 卡**：`train_common.py` 里已改为读 `device_ids`（默认仍是 `[0,1,2,3]`），单卡或卡数不同需在配置里改 `train.device_ids`。
4. **预训练权重缺失会静默随机初始化**：跑之前务必确认 `/home/zyai/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` 存在。
5. **cv2 先于 torch 导入会段错误**（libiomp5 冲突）；Git Bash 跑 conda python 会静默 139，改用 PowerShell。
