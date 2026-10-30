# PECAL 

**论文**：Phenology-Aware Energy-Guided Consistency and Ordinal Learning for Crop Growth Stage Recognition from Field RGB Images
**收录范围**：基础对比实验、消融与敏感性实验、跨作物（冬小麦）实验

---

##  统一入口（2026-10-30 起，推荐）

原来 5 个训练脚本 + 3 个批量脚本已合并为两个入口，**旧文件原样保留、未改动**，可随时回退：

```bash
# 训练（取代 train_ablation_full / _patch_mode / _baseline / _n_global_ablation / _wheat）
python train.py --config-augment configs/experiments/full_pecal.yaml \
                --exp-dir experiments/table5/full_pecal

# 批量（取代 run_base / run_ablation_A / run_ablation_B，共 25 个实验）
python run.py --list                 # 查看 25 个实验
python run.py --suite base           # Table 4 的 6 个
python run.py --suite table5         # Table 5 的 8 个
python run.py --suite sens --group D # Fig 5 的 D 组
```

支撑文件：

| 文件 | 作用 |
|---|---|
| `utils/train_common.py` | 五个训练脚本的公共实现（848 行重复全部收在这里） |
| `datasets/dataset_factory.py` | 四个 MixingDataset 的四选一分派器 |

> ⚠️ impl 必须显式指定（`--dataset-impl`）。`run.py` 的 25 条实验已逐条写死；单独调 `train.py` 时的自动推断只是 fallback，面对组 A 这类只写 `switch_epoch` 的配置会判错。
>
> 合并细节、修掉的四个既有问题、行为差异与迁移方式见 **`代码合并说明_train与run.md`**（在 `E:\桌面\workbuddy\`）。

---

##  环境

论文 §II-F 声明的环境：

| 项 | 值 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| CPU / 内存 | 2×AMD EPYC 9554 / 125 GiB |
| GPU | 4×NVIDIA RTX 4090（24 GB） |
| Python / CUDA / cuDNN | 3.10.19 / 12.1 / 9.1.0 |
| PyTorch / torchvision | 2.5.1 / 0.20.1 |
| 分布式 | DataParallel（**4 卡**） |

第三方依赖见 `requirements.txt`。注意：`torch.utils.tensorboard` 是硬依赖（训练脚本无条件 `import`），离线机器需一并安装或注释掉对应行。

---

## 3. 必须先知道的 6 个坑

> 这几条每一条都会让"看起来跑起来了，但结果不对"。上传前请务必确认。

**① ImageNet 预训练权重可能静默失效**
`utils/model_factory.py`：`use_timm_pretrained = pretrained and (checkpoint_path is None)`。当 `pretrained: true` **且** `checkpoint_path` 同时给出时，timm 的预训练被关掉，改从本地 `.pth` 加载；而若该 `.pth` 不存在，代码**不报错、不警告**，直接随机初始化。
→ 训练前先确认 /torch/hub/checkpoints/resnet18-f37072fd.pth` 存在。

**② 4 卡**
训练脚本里都有 `torch.nn.DataParallel(model, device_ids=[0,1,2,3])`。单卡或卡数不同必须改这一行，且 `configs/resume.yaml` 的 `batch_size: 256` 是**总**批量（非每卡）。

③ 小麦训练要换 trainer_csv**
`configs/resume.yaml` 默认是玉米的 train/val CSV，小麦路径在文件里被注释着。本包给出 `configs/wheat.yaml`；注意 `resume_ablation_wheat.py` **没有**传 `--config-train`，批量跑之前要在它的 cmd 列表里补上，否则会拿玉米数据训小麦。

---

## 4. 方法描述与代码的对应关系（已逐项核对）

| 论文公式 | 代码位置 |
|---|---|
| (2) `V = max(0, 3G − 2.4R − B)` | `datasets/mixing_dataset_ablation.py:45-46` |
| (3) Otsu + 形态学 + 高斯 | 同上，5×5 椭圆核 |
| (4) 归一化采样概率 | 同上 |
| (5) 0.3W–0.7W / 0.3H–0.7H 随机切分 | `:249-250` |
| (6)(7) 图块响应求和，`ε=1e-6` | `:202, :305` |
| (9) 课程切换 `switch_epoch=100` | 配置 `ablation_curriculum_ep100.yaml` |
| (10) 高斯序数软标签 `σ=0.5` | `:70`（默认参数 0.5） |
| (12)(13)(14)(15) `L = Lcls + λ(t)Lcons + 0.5·Lord`，`Tr=80`，`KL(pm‖ps)` | `train_ablation_full.py:441` + `build_consistency_scheduler()` |
| 混合触发概率 0.5、2 全局 + 2 引导局部 | 配置 `prob: 0.5` / `n_global: 2` |
| 类别均衡采样（逆频加权） | `train_ablation_full.py:299-307` WeightedRandomSampler |
| (16)(17) 排序MAE、五站点等权平均 | `test_ablation_full.py` CLASS_MAPPING / DataFrame 分组 |

---

## 5. 数据路径清单

上传前应把下面这些路径替换成公开/可申请的形式，或明确标注"需向作者申请"：

| 位置 | 当前值 |
|---|---|
| `configs/train.yaml`（`configs/resume.yaml` 为同内容的默认别名） | `PECAL/corn-cls/train/{train,val}.csv`（相对占位；服务器原件为 `/media/zyai/程序/lyj/AgriAug/corn-cls/train/{train,val}_ubuntu_exclude_11_99.csv`） |
| `configs/wheat.yaml` | `/media/zyai/数据/lyj/wheat/trainval/{train,val}.csv` |
| `test_ablation_full.py` TEST_ROOT | `/media/zyai/程序/lyj/AgriAug/corn-cls/test` |
| `test4.py` TEST_ROOT | `/media/zyai/程序/lyj/AgriAug/corn-cls/test` |
| `test_wheat.py` TEST_ROOT | `/media/zyai/数据/lyj/wheat` |
| `configs/dataset/subset_100.yaml` | `saliency_dir` / `probmap_pkl_path`（显著图与 ExG 概率图缓存，**体积很大，未收录**） |

关于-prop缓存：`datasets/precompute_exg_probmaps_exact.py` 与 `precompute_saliency*.py` 是生成这两个缓存的脚本，**未收录**（生成耗时很长且依赖全量数据）。可以在 README 里说明缓存可由脚本重建，也可在线计算。

---

## 6. 目录结构

```
PECAL_code_package/
├── README.md              本文档
├── CODE_MAP.md            论文表/图 → 脚本+配置+命令 的逐条映射
├── TABLE5_EXPERIMENTS.md  Table 5 组件消融 8 行的配置/开关/运行/验证
├── requirements.txt
│
├── train_ablation_full.py             ★ PECAL 主训练（新管路）
├── run_ablation_full.py               四组敏感性消融批量运行器（Fig 5）
├── run_table5.py                      ★ Table 5 组件消融 8 配置一键运行/评测
├── test_ablation_full.py              五站点评测（已放开 5 站）
├── train_n_global_ablation.py         消融：全局/局部图块比例（Fig 5c）
├── train_ablation_patch_mode.py       消融：图块构造方式（patch-mode 补充实验）
│
├── resume_ablation.py                 基础对比批量运行器
├── train_baseline.py                  对比方法训练器（旧管路）
├── test.py                            对比方法五站点评测
│
├── resume_ablation_wheat.py           小麦批量运行器
├── train_wheat.py                     小麦训练器
├── test_wheat.py                      小麦两季测试
│
├── datasets/  (8 个数据相关文件)
├── utils/     (7 个公共模块)
└── configs/
    ├── train.yaml / resume.yaml / wheat.yaml
    ├── dataset/  model/
    ├── augment/       (33 个敏感性/对比配置)
    └── experiments/   ★ Table 5 的 8 个组件消融配置
```

详细命令见 `CODE_MAP.md`；Table 5 的逐行对应与验证记录见 `TABLE5_EXPERIMENTS.md`。
