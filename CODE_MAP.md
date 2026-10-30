## 一、基础对比实验

**玉米 7 类，5 个独立测试子集等权平均。**

| 方法 | 配置文件 | 备注 |
|---|---|---|
| Baseline | `configs/augment/base.yaml` | 无混合、无一致性、无序数列损失 |
| Strong Augmentation | `configs/augment/full.yaml` | RandAugment N=2, M=9 |
| Mixup | `configs/augment/mixup.yaml` | α=0.8 |
| CutMix | `configs/augment/cutmix.yaml` | α=1.0 |
| Mosaic | `configs/augment/mosaic.yaml` | 拼接概率 0.5，面积加权 |
| RICAP | `configs/augment/ricap.yaml` | 随机模式，面积加权 |
| **PECAL（本文）** | `configs/augment/ablation_label_gaussian.yaml`<br> | 二者语义等价：Guided+Energy、curriculum ep100、KL(mix‖strong)、Gaussian σ=0.5、2 全局 + 2 引导局部 |

### 训练

```bash
# 批量：在 resume_ablation.py 的 STRATEGIES 里放开需要的键（当前默认全部注释），然后
nohup python resume_ablation.py > comparison_train.log 2>&1 &

# 单个：
python train_baseline.py \
    --config-train   configs/resume.yaml \
    --config-model   configs/model/resnet18.yaml \
    --config-dataset configs/dataset/subset_100.yaml \
    --config-augment configs/augment/base.yaml \
    --exp-dir        experiments/resnet18_100pct_Baseline_exclude_11_99
```

### 评测

```bash
python test4.py --model resnet18 \
    --exp_dir ./experiments \
    --strategies Baseline_exclude_11_99 Strong_exclude_11_99 Mixup_exclude_11_99 \
                 CutMix_exclude_11_99 Mosaic_exclude_11_99 RICAP_exclude_11_99 \
    --output_csv test_results/test_results_periods.csv \
    --exclude_11_99
```

> 带 `--exclude_11_99` 时，输出文件名会被自动改写为 `test_results_periods_exclude_11_99.csv`（脚本内部 `.replace('.csv','_exclude_11_99.csv')`），不要按原文件名去找。
> 支持 `--exclude_11_99` 的还有 `test_wheat.py`；`test_ablation_full.py` 没有该开关（它的 CLASS_MAPPING 本来就只有 7 个生育期）。

> ⚠️ **跨管路**：本批量用旧管路训练器，而 PECAL 的消融曲线是新管路。历史结果属于两批不同的 checkpoint 与测试快照，跨批比较建议只用 O3543 / O8562 / O8563 三站（README §3-⑤）。

---

## 二、消融与敏感性实验（论文 Table 5、Fig 5）

### 2.1 四组敏感性（全部 ✅ 配置驱动）

```bash
# 一键跑 15 个条件（A/B/C 组用 train_ablation_full.py，D 组用 train_n_global_ablation.py）
nohup python run_ablation_full.py > ablation.log 2>&1 &

# 只跑某一组
python run_ablation_full.py --group D      # A=课程 B=标签 C=KL D=全局比
python run_ablation_full.py --only 10      # 只跑第 10 个实验
python run_ablation_full.py --skip-existing

# 评测（增量写入，续传粒度 = 实验×站点）
python test_ablation_full.py --exp_dir experiments/ablation_full --model resnet18
```

| 论文 Fig 5 | 论文内容 | 实验名 | 配置 | 状态 |
|---|---|---|---|---|
| 5(a) | 目标形式：Gaussian / One-hot / Uniform | `label_gaussian` / `label_onehot` / `label_uniform` | `configs/augment/ablation_label_{gaussian,onehot,uniform}.yaml` | ✅ |
| 5(b) | 课程切换 epoch：50 / 100 / 150 | `curriculum_ep50` / `ep100` / `ep150` | `configs/augment/ablation_curriculum_ep{50,100,150}.yaml` | ✅ |
| 5(c) | 全局/局部：0…4 全局 + 位置随机 | `n_global_{0,1,2,3,4}` / `n_global_2_random` | `configs/augment/n_global_{0,1,2,3,4}.yaml`、`n_global_2_random.yaml` | ✅ |
| 5(d) | KL 方向：mix→strong / strong→mix / 双向 | `kl_mix_to_strong` / `kl_strong_to_mix` / `kl_bidirectional` | `configs/augment/ablation_kl_{mix_to_strong,strong_to_mix,bidirectional}.yaml` | ✅ |

> D 组用 `train_n_global_ablation.py` + `datasets/{dataset_factory_n_global,mixing_dataset_n_global}.py`，已在包内。

### 2.2 组件消融（Table 5）——8 行全部 ✅ 配置驱动

8 行统一走 `train_ablation_full.py`，唯一差异在 `configs/experiments/*.yaml`。细节见 `TABLE5_EXPERIMENTS.md`。

| 论文 Table 5 行 | 配置 | 唯一变量 |
|---|---|---|
| Full PECAL | `configs/experiments/full_pecal.yaml` | —（guided + energy，基准） |
| w/o Ordinal Loss | `configs/experiments/wo_ordinal.yaml` | `ordinal_weight: 0.0` |
| Random + Area | `configs/experiments/random_area.yaml` | `local_crop_mode: random` + `weighting_mode: area` |
| Guided + Area | `configs/experiments/guided_area.yaml` | `weighting_mode: area` |
| Random + Energy | `configs/experiments/random_energy.yaml` | `local_crop_mode: random` |
| Intra-only | `configs/experiments/intra_only.yaml` | `ricap_mode: intra` |
| Neighbor-only | `configs/experiments/neighbor_only.yaml` | `ricap_mode: neighbor` |
| w/o KL | `configs/experiments/wo_kl.yaml` | `consistency_schedule: none` |

```bash
python run_table5.py --dry-run                    # 先看命令
nohup python run_table5.py --skip-existing > table5.log 2>&1 &
python run_table5.py --only random_area wo_kl     # 只跑某几个
python run_table5.py --test-only                  # 五站点评测 → test_results/table5_5sites.csv
```

> 为支持这 8 行，包内新增/打通的三个开关（均标 `# [packaged]`，默认值=原行为）：
> `local_crop_mode`（guided/random，只作用于 dual_stream 的后 2 个局部块）、
> `ordinal_weight`（原本硬编码 0.5）、`label_smooth_sigma`/`switch_epoch` 的配置透传。
>
> ⚠️ Table 5 的 Random/Guided **不是** `ablation_random_crop_all.yaml`（那个是 4 块全部随机，属 Fig 5 的 patch-mode 补充实验）。
>
> ⚠️ 代码齐 ≠ 数字有出处：Random+Area、Guided+Area、Neighbor-only、w/o KL 至今没有权重与结果文件；
> Intra-only、Random+Energy、w/o Ordinal 只有旧管路的 3 站结果。详见 `TABLE5_EXPERIMENTS.md` §6。

---

## 三、跨作物实验（冬小麦，论文 Fig 6）

**10 类（22 也算一类），两个独立生长季：O3543-2022–2023、O8562-2021–2022。**

### 训练

```bash
# 1) resume_ablation_wheat.py 的 WHEAT_STRATEGIES 当前只放开 PECAL 一行，
#    放开 Baseline/Strong/Mixup 需要取消对应注释：
#      "Wheat_Baseline": "base.yaml",  "Wheat_Strong": "full.yaml",  "Wheat_Mixup": "mixup.yaml",
# 2) 在它的 cmd 列表里加 --config-train configs/wheat.yaml（否则会拿玉米数据训练）
nohup python resume_ablation_wheat.py > wheat_train.log 2>&1 &

# 单个训练（推荐，路径显式）
python train_dual_piModelVar_strongOriginal2.py \
    --config-train   configs/wheat.yaml \
    --config-model   configs/model/resnet18.yaml \
    --config-dataset configs/dataset/subset_100.yaml \
    --config-augment configs/augment/energy_based_curriculum_consistency.yaml \
    --exp-dir        wheat_experiments/resnet18_100pct_Wheat_PECAL
```

### 评测

```bash
# TEST_ROOT 默认 /media/zyai/数据/lyj/wheat，SITES = O3543-22-23_test, O8562-21-22_test
python test_wheat.py --model resnet18 --exp_dir ./wheat_experiments \
    --strategies resnet18_100pct_Wheat_PECAL --exclude_11_99
```

> ⚠️ **β 不一致**：小麦训练器里序数列损失系数是 **1.0**（`… + 1 * loss_ord`），而论文声明 β=0.5。投稿前须统一，否则小麦结果是"另一个 β"下取得的（README §3-④）。

小麦数据集划分（按生长季切分，见 §2-A）可由 `utils/generate_wheat_csv.py` 重放生成。

---

## 四、与论文数字的对应关系 ⚠️ 上传前必须确认

代码与论文**方法描述**逐项吻合（README §4 已核对 13 项），但代码包与**论文中报告的数字**之间存在一处必须交代的缺口：

| 论文 Table 4 中 PECAL | acc 0.7760 / Macro-F1 0.7621 / MAE 0.2482 / QWK 0.9404 |
|---|---|

在上述两份代码备份所能找到的结果文件中，**没有与该组数字对应的记录**。现有可查的最新 PECAL 等价记录为：

| 来源 | 实验名 | acc | Macro-F1 | MAE | QWK |
|---|---|---|---|---|---|
| `test_results/ablation_full_5sites_ubuntu.csv` | `label_gaussian`（五站等权均值） | 0.7566 | 0.7419 | 0.2673 | 0.9365 |
| `five_site_comparison/ranking_all_methods_5sites.csv` | `EDCL:strongOnMix_aug2` | — | 0.7449 | — | — |

差异远大于复现波动噪声（同配置重复实测 σ≈0.027），不是舍入或口径微调能解释的。

**可能原因（需你确认）**：1028 稿的数字来自服务器上一次尚未备份回本机的最新运行（很可能是 10-21 之后用新版 `train_ablation_full.py` 重跑的一致性 B 组）。

**上传前请做三件事**：
1. 在服务器上定位产出 0.7760 / 0.7621 的那次运行目录（checkpoint + `log.csv` + `test_results/*.csv`）；
2. 把该次结果 CSV 一并归入 uploaded artifact，让数字可追溯；
3. 若无法定位，则该组数字应标注为"未随代码公开"，或在论文/补充说明中给出出处说明。

> 同质问题已在历史分析中确认过：Strong@O6965 的 0.7845 属**人工修正值**（`strong_corrected_full_metrics.csv` 记为 `user-corrected F1 + regression-imputed`），论文中须标注来源，不能当作实测值呈现。
