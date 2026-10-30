# PECAL

**Phenology-Aware Energy-Guided Consistency and Ordinal Learning for Crop Growth Stage Recognition from Field RGB Images**

Official implementation of the training-time augmentation framework **PECAL**, covering the baseline comparison, the ablation / sensitivity study, and the cross-crop (winter wheat) extension reported in the paper.

---

## 1. Overview

Crop growth stage recognition from field RGB images must cope with small inter-stage appearance differences, large intra-stage variation, and distribution shifts across sites and years. Generic mixing augmentations (Mixup, CutMix, RICAP) expand sample diversity but assign label weights from a preset coefficient or geometric **area**, which does not necessarily reflect how much stage-relevant content a retained patch actually carries; they also ignore the **ordinal** structure of growth stages.

PECAL addresses this at three coordinated levels, and is applied **only during training** — inference is a standard single-forward pass:

| Level | Mechanism |
|---|---|
| **Image content** | A four-region mosaic built from **globally resized** patches and **vegetation-guided local crops**. The sampling map comes from a colour vegetation index (ExGR, `3G − 2.4R − B`) computed on the fly, so no segmentation labels or auxiliary network are needed. Mixing weights are estimated from the **vegetation response energy** of each region instead of its area. |
| **Stage supervision** | A **curriculum** that switches the candidate set from the anchor's own class to its own class plus adjacent stages at a preset epoch, combined with **Gaussian ordinal soft targets** and an **ordinal shift loss** that penalises predicted-vs-target rank distance. |
| **Training constraint** | A **dual-view consistency** term: KL divergence between the mixed view and a conventional strong-augmentation view of the same anchor, weighted by a linear ramp-up schedule. |

---

## 2. Repository structure

```
PECAL_code_package/
├── train.py                    # unified training entry point (replaces 5 legacy train_*.py)
├── run.py                      # batch runner over 25 registered experiments
├── test.py                     # 5-site evaluation for baseline / comparison methods
├── test_ablation_full.py       # 5-site evaluation for ablation / sensitivity experiments
│
├── datasets/
│   ├── dataset_factory.py              # dispatcher: selects one of the 4 MixingDataset impls
│   ├── dataset_factory_ablation.py     # stratified subset helper
│   ├── cls_dataset.py                  # plain classification dataset (val / non-mixing)
│   ├── mixing_dataset_ablation.py      # impl `ablation`  — soft-label types, curriculum, 5-tuple output
│
├── utils/
│   ├── train_common.py         # shared training / validation loop, losses, schedulers
│   ├── config_parser.py        # YAML merging + argparse
│   ├── augmentations.py        # standard + strong augmentation pipelines
│   ├── model_factory.py        # model construction and checkpoint loading
│   ├── logger.py, reproducibility.py, consistency_loss.py, gaussian_rampUp.py
│   └── generate_wheat_csv.py   # builds wheat train/val/test CSVs
│
├── configs/
│   ├── train.yaml / resume.yaml        # training hyper-parameters
│   ├── wheat.yaml                      # cross-crop experiment
│   ├── model/                          # resnet18 (default), vgg, mobilenetv2, efficientnet_b0, vit_b16
│   ├── dataset/                        # subset_10 / 50 / 100
│   ├── augment/                        # 26 configs: base, strong, mixup, cutmix, mosaic, ricap,
│   │                                   #   energy/area variants, n_global_*, sensitivity ablations
│   └── experiments/                    # 8 single-factor configs for the Table 5 component ablation
│
├── checkpoint/                 # released model weights (25 experiment directories)
├── data/                       # format samples only — NOT the full dataset (see §4)
│   ├── trainval                # contained train.csv and val.csv
│   └── test
└── requirements.txt
```

---

## 3. Setup

Tested with Python 3.10.19 / CUDA 12.1 / cuDNN 9.1.0.

```bash
conda create -n pecal python=3.10 -y
conda activate pecal

# install torch matching your CUDA, e.g. cu121
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**ImageNet pre-trained backbone.** `utils/model_factory.py` sets

```python
use_timm_pretrained = pretrained and (checkpoint_path is None)
```

so if the weight file is missing the model falls back to **random initialisation without raising an error**. Confirm the file exists before a long run:

```bash
ls ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

**Multi-GPU.** The training scripts are hard-coded to 4 GPUs (`device_ids=[0,1,2,3]`). Edit that line for a different layout.

**Windows.** Use `--num_workers 0` for evaluation; the default (`16`) assumes a Linux host.

---

## 4. Data

The training and validation partitions are provided as CSV files containing image identifiers and growth-stage labels. The maize test subsets are defined by the following held-out site–year combinations: O3543-2022, O8562-2022, O8563-2022, O6965-2021, and O9113-2018. The winter wheat test subsets correspond to O3543-2022–2023 and O8562-2021–2022 growing seasons. Test images were excluded from model training, validation, and checkpoint selection. The complete image dataset is available from the corresponding author upon reasonable request.

### 4.1 Maize (main experiment)

Seven ordered stages coded as `21, 31, 41, 61, 71, 81, 91` (emergence, three-leaf, seven-leaf, jointing, tasseling, milk, maturity), mapped to consecutive rank indices in developmental order. The numeric gaps between codes do **not** represent temporal intervals.

| Split | Images |
|---|---|
| Training | 16,450 |
| Validation | 4,115 |

Five **independent** test subsets, each holding out either a year at a seen site or an entire unseen site:

| Test subset | Images | Held-out factor |
|---|---|---|
| O3543-2022 | 932 | unseen year |
| O8562-2022 | 476 | unseen year |
| O8563-2022 | 537 | unseen year |
| O6965-2021 | 695 | unseen year |
| O9113-2018 | 474 | **unseen site** (absent from train & val) |
| **Total** | **3,114** | |

Reported scores are **equal-weight means over the five subsets**, alongside per-subset values.

### 4.2 Winter wheat (cross-crop extension)

Ten stages (`21, 22, 31, 41, 51, 61, 71, 72, 81, 91`); training 13,321 / validation 5,573 images. Two independent test subsets defined by site **and growing season** (winter wheat spans two calendar years): O3543-2022–2023 (1,719) and O8562-2021–2022 (1,284), **3,003 images in total**.

### 4.3 What is in `data/`

`data/` holds **format samples only** (14 test + 92 trainval images) so that the expected layout is unambiguous. It is not the dataset. Filenames encode site, plot and acquisition time:

```
O3543_010405_20220729140000_01.jpg
│     │      │                └─ index
│     │      └─ YYYYMMDDHHMMSS
│     └─ plot id
└─ site id
```

Training and validation splits are driven by CSV files (`configs/train.yaml` -> `train.train_csv` / `train.val_csv`) with `image_path,label` rows. Point these at your own copies.

The evaluation scripts expect images grouped **by site-year** under `--test_root` (default `PECAL/data/test`). Note that the public release contains the full images only for the sample subset.

---

## 5. Training

Single run:

```bash
# Full PECAL (Table 5 reference row)
python train.py --config-augment configs/experiments/full_pecal.yaml \
                --exp-dir experiments/table5/full_pecal

# Baseline / classical augmentation (Table 4)
python train.py --config-augment configs/augment/strong.yaml \
                --exp-dir experiments/resnet18_100pct_Strong_exclude_11_99

# Cross-crop (winter wheat)
python train.py --config-augment configs/wheat.yaml \
                --exp-root wheat_experiments \
                --exp-dir wheat_experiments/resnet18_100pct_Wheat_PECAL
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--config-train / --config-dataset / --config-model / --config-augment` | override individual YAML files |
| `--seed N` | override `train.seed` |
| `--dataset-impl {ablation,dualstream2_ablation,n_global,dualstream2}` | **explicitly** select the `MixingDataset` implementation |
| `--exp-root` | output root (`experiments` by default; use `wheat_experiments` for wheat) |
| `--no-resume` | ignore `last.pth` in the experiment directory and start from scratch |

> **Always pass `--dataset-impl` when reproducing a specific experiment.** Automatic inference is unreliable: e.g. `configs/augment/ablation_curriculum_ep50.yaml` only sets `switch_epoch`, and `configs/augment/ricap.yaml` sets nothing discriminative, yet historically they must run through `ablation` and `dualstream2` respectively. `run.py` therefore pins the implementation for every registered experiment.

Batch runner (25 registered experiments: 6 baseline + 8 Table 5 + 11 sensitivity):

```bash
python run.py --list                    # show the registry
python run.py --suite base              # Table 4 comparison methods
python run.py --suite table5            # Table 5 component ablation
python run.py --suite sens --group D    # one sensitivity group
python run.py --suite all --dry-run     # print commands without executing
```

Runner options:

| Flag | Meaning |
|---|---|
| `--list` | print the registry and exit |
| `--suite {base,table5,sens,all}` | select an experiment suite |
| `--group {A,B,C,D}` | restrict to one sensitivity group |
| `--only <name|id> ...` | run selected entries, e.g. `--only full_pecal 21` |
| `--out-root PATH` | override the output root of registered `exp_dir`s |
| `--seeds N ...` | repeat each experiment with several seeds (default `42`) |
| `--skip-existing` | skip directories that already contain `best.pth` |
| `--dry-run` | print commands without executing |
| `--test-after` | evaluate automatically after each training run |
| `--test-only` | skip training, only evaluate |
| `--test-csv PATH` | aggregated result file (default `test_results/results.csv`) |
| `--config-train / --config-model / --config-dataset` | override config files for all entries |
| `--no-summary` | suppress the final summary table |

> Note that `run.py` invokes `train.py` as a subprocess; `--dataset-impl` is pinned per experiment inside the registry, so you never need to pass it manually through the runner.

Other options: `--only <id...>`, `--skip-existing`, `--seeds 0 1 2`, `--test-only`.

Default hyper-parameters (`configs/train.yaml`): ResNet-18, 224×224, 200 epochs, AdamW, lr 1e-4, batch size 256, 5 warm-up epochs, cosine schedule, seed 42.

---

## 6. Evaluation

```bash
# Ablation / sensitivity experiments (per-experiment best.pth over the 5 maize test subsets)
python test_ablation_full.py --exp_dir experiments/ablation_full \
                             --output_csv test_results/ablation_full_5sites.csv

# Baseline / comparison methods
python test.py --exp_dir experiments \
               --strategies resnet18_100pct_Baseline_exclude_11_99 \
                            resnet18_100pct_Strong_exclude_11_99 \
               --output_csv test_results/baseline_5sites.csv

```

Metrics: Accuracy, Macro-F1, Macro Recall/Precision, Quadratic Weighted Kappa (QWK), Adjacent-stage F1, MAE over stage ranks, and Hard-F1. Stage-wise precision / recall / F1 / support are written per class.

## 7. Results


PECAL reaches Macro-F1 **0.685** on O3543-2022–2023 and **0.641** on O8562-2021–2022 (equal-weight mean **0.6630**, +6.85 pp over Baseline); mean MAE falls from **0.5045** to **0.3870** (−23.3% relative). Gains are not uniform across stages: three-leaf and heading vary by season, and maturity declines in both seasons (−3.75 pp on average).

## 8. Model weights

`checkpoint/` contains 25 directories, one per experiment, each with `best.pth` (ResNet-18, ≈134 MB;).

| Directory | Corresponds to |
|---|---|
| `Baseline/` | Table 4 — Baseline |
| `Strong/` | Table 4 — Strong Augmentation |
| `Mixup/` | Table 4 — Mixup |
| `CutMix/` | Table 4 — CutMix |
| `Mosaic/` | Table 4 — Mosaic |
| `RICAP/` | Table 4 — RICAP |
| `wo_ordinal/` | Table 5 — w/o Ordinal Loss |
| `wo_kl/` | Table 5 — w/o KL |
| `random-energy/`, `random_area/`, `guided_area/` | Table 5 — 2×2 crop × weighting grid |
| `intra_only/`, `neighbor_only/` | Table 5 — sampling-curriculum variants |
| `PECAL/` | Full PECAL — **default** configuration (Gaussian targets, switch epoch 100, `n_global=2`, KL mix→strong) |
| `curriculum_ep50/`, `curriculum_ep150/` | Sensitivity — switch epoch (the 100 default is `PECAL/`) |
| `label_onehot/`, `label_uniform/` | Sensitivity — target encoding |
| `kl_strong_to_mix/`, `kl_bidirectional/` | Sensitivity — KL direction |
| `n_global_0/`, `n_global_1/`, `n_global_3/`, `n_global_4/`, `n_global_2_random/` | Sensitivity — global/local ratio |


