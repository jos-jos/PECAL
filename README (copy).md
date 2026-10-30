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
│   ├── mixing_dataset3_dualstream2_ablation.py  # impl `dualstream2_ablation` — patch_mode / local_crop_mode
│   ├── mixing_dataset_n_global.py      # impl `n_global`  — global/local patch ratio
│   └── mixing_dataset3_dualstream2.py  # impl `dualstream2` — base RICAP-style mixing
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
├── test_results/               # reference measurements shipped with the release (see §7)
│   ├── ablation_full_5sites.csv
│   └── five_site_comparison/five_site_exploratory_complete_corrected.csv
└── wheat_results/test_results_wheat.csv   # corresponds to EARLIER weights, see §7.4
│
├── CODE_MAP.md                 # paper table/figure -> script + config + command (incl. Table 5 mapping)
├── MERGE_NOTES.md              # notes on consolidating 8 legacy entry points into train.py / run.py
├── README_zh.md                # Chinese counterpart of this file
└── requirements.txt
```

> **Why four `MixingDataset` classes instead of one?** They are not stylistic variants — they differ in semantics (one returns a 5-tuple with `patch_labels` used by the same-class mask; others accept different switches such as `n_global` vs. `patch_mode`). Merging them would silently change behaviour, so `dataset_factory.py` **dispatches** rather than unifies.

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

> **`test.py` does not prepend the `resnet18_100pct_` prefix** — pass full directory names to `--strategies`. `test_ablation_full.py` discovers experiment directories instead. The two behave differently; this is a known inconsistency.
>
> **Winter wheat evaluation is not included in this release.** `test.py` hard-codes the maize configuration (`CLASS_MAPPING` = 7 stages, `SITES` = the five maize subsets), so `--test_root` alone is not enough for the 10-class wheat task. Build the wheat splits with `utils/generate_wheat_csv.py` and supply a wheat-configured evaluation script, or adapt `test.py` by overriding `CLASS_MAPPING` and `SITES`:

```bash
python utils/generate_wheat_csv.py   # writes wheat train / val / test CSVs
```

---

## 7. Results

All values below are as reported in the paper: equal-weight means over the five maize test subsets (↑ higher is better; ↓ lower is better).

**Reference measurements are shipped with this release** so that the traceable numbers can be checked without re-running: `test_results/ablation_full_5sites.csv` (sensitivity study, §7.2), `test_results/five_site_comparison/five_site_exploratory_complete_corrected.csv` (Table 4 comparison methods, §7.1), and `wheat_results/test_results_wheat.csv` (wheat, §7.4 — note the caveat there).

### 7.1 Table 4 — comparison with baselines and classical augmentations

| Method | Accuracy ↑ | Macro-F1 ↑ | MAE ↓ | QWK ↑ | Traceable to a result file |
|---|---|---|---|---|---|
| Baseline | 0.6957 | 0.6601 | 0.3523 | 0.9102 | ✅ exact |
| Strong Augmentation | 0.7413 | 0.7025 | 0.2737 | 0.9429 | ⚠️ see note |
| Mixup | 0.7359 | 0.6938 | 0.3472 | 0.8756 | ✅ exact |
| CutMix | 0.7272 | 0.6954 | 0.2945 | 0.9321 | ✅ exact |
| Mosaic | 0.7352 | 0.6937 | 0.3142 | 0.9053 | ✅ exact |
| RICAP | 0.6014 | 0.5600 | 0.6539 | 0.7239 | ✅ exact |
| **PECAL** | **0.7760** | **0.7621** | **0.2482** | **0.9404** | ⚠️ see note |

### 7.2 Figure 5 — sensitivity study (measured, fully reproducible)

Equal-weight means over the five maize test subsets, from `test_results/ablation_full_5sites.csv`:

| Configuration | Accuracy ↑ | Macro-F1 ↑ | MAE ↓ | QWK ↑ |
|---|---|---|---|---|
| `label_gaussian` (default) | 0.7620 | 0.7445 | 0.2614 | 0.9386 |
| `label_uniform` | 0.7462 | 0.7176 | 0.2808 | 0.9347 |
| `label_onehot` | 0.7120 | 0.6657 | 0.3218 | 0.9181 |
| `kl_strong_to_mix` | 0.7519 | 0.7074 | 0.2715 | 0.9353 |
| `kl_mix_to_strong` (default) | 0.7324 | 0.7038 | 0.3017 | 0.9204 |
| `kl_bidirectional` | 0.7103 | 0.6486 | 0.3285 | 0.9140 |
| `curriculum_ep50` | 0.7683 | 0.7226 | 0.2585 | 0.9275 |
| `curriculum_ep100` (default) | 0.7111 | 0.6797 | 0.3377 | 0.9093 |
| `curriculum_ep150` | 0.7601 | 0.7092 | 0.2750 | 0.9256 |
| `n_global_0` | 0.7250 | 0.6807 | 0.3420 | 0.8916 |
| `n_global_1` | 0.7223 | 0.6960 | 0.3054 | 0.9236 |
| `n_global_2` (default) | 0.7357 | 0.6946 | 0.3105 | 0.9132 |
| `n_global_2_random` | 0.6964 | 0.6510 | 0.3649 | 0.8945 |
| `n_global_3` | 0.7142 | 0.6848 | 0.3463 | 0.8961 |
| `n_global_4` | 0.7681 | 0.7194 | 0.2580 | 0.9359 |

⚠️ These are **single runs**. Repeated runs of the same configuration vary by about **σ ≈ 0.027 Macro-F1**; treat differences below 1σ as noise. Four of the rows above — `label_gaussian`, `curriculum_ep100`, `n_global_2` and `kl_mix_to_strong` — are in fact independent re-runs of the *same* default configuration, and they range from **0.6797 to 0.7445 Macro-F1 (a spread of 0.065, ≈ 2.4σ)**. The ranking within this table should therefore not be over-interpreted.

### 7.3 Table 5 — component ablation (as reported in the paper)

| Configuration | Accuracy ↑ | Macro-F1 ↑ | MAE ↓ | QWK ↑ |
|---|---|---|---|---|
| Full PECAL | 0.7760 | 0.7621 | 0.2482 | 0.9404 |
| w/o Ordinal Loss | 0.7812 | 0.7685 | 0.2782 | 0.9266 |
| Random + Area | 0.6815 | 0.6723 | 0.3315 | 0.8741 |
| Guided + Area | 0.7524 | 0.7445 | 0.2664 | 0.9188 |
| Random + Energy | 0.7336 | 0.7260 | 0.3037 | 0.9002 |
| Intra-only | 0.7472 | 0.7398 | 0.2711 | 0.9305 |
| Neighbor-only | 0.6748 | 0.6682 | 0.3406 | 0.8674 |
| w/o KL | 0.7006 | 0.6920 | 0.3173 | 0.8856 |

⚠️ **None of these eight rows matches any record in the released result files.** The eight configurations are individually valid — `configs/experiments/` provides them as strict single-factor contrasts against `full_pecal.yaml` — but the reported numbers come from a historical batch whose raw outputs were not retained. Re-running them with `run.py --suite table5` is the supported way to regenerate this table.

### 7.4 Figure 6 — winter wheat (as reported in the paper)

PECAL reaches Macro-F1 **0.685** on O3543-2022–2023 and **0.641** on O8562-2021–2022 (equal-weight mean **0.6630**, +6.85 pp over Baseline); mean MAE falls from **0.5045** to **0.3870** (−23.3% relative). Gains are not uniform across stages: three-leaf and heading vary by season, and maturity declines in both seasons (−3.75 pp on average).

⚠️ `wheat_results/test_results_wheat.csv` in this repository corresponds to an **earlier** set of weights (2026-07-07) and does **not** reproduce the figures above. The wheat numbers in the paper come from a later run whose outputs were not retained.

### 7.5 Reproducibility status

Please read this before citing any number:

| Reported in | Status |
|---|---|
| Table 4 — Baseline, Mixup, CutMix, Mosaic, RICAP | ✅ Reproduced exactly from `test_results/five_site_comparison/five_site_exploratory_complete_corrected.csv` (scenario `O6965-2021_v3__O9113_n950`) |
| Figure 5 sensitivity table | ✅ Reproduced from `test_results/ablation_full_5sites.csv` (§7.2) |
| Table 4 — PECAL row | ⚠️ No matching record. Four independent re-runs of the same configuration give 0.7445 / 0.6797 / 0.6946 / 0.7038 Macro-F1 (spread 0.065, §7.2); the reported 0.7621 exceeds all of them |
| Table 4 — Strong Augmentation row | ⚠️ The MAE / QWK / Adjacent-F1 / Hard-F1 values were author-corrected from imputed data; the nearest recorded variant gives 0.7606 / 0.7244 / 0.2546 / 0.9468 |
| Table 5 — all 8 rows | ⚠️ No matching record in any of 62 result files searched |
| Figure 6 — all wheat numbers | ⚠️ Result file corresponds to earlier weights |

---

## 8. Model weights

`checkpoint/` contains 25 directories, one per experiment, each with `best.pth` (ResNet-18, ≈134 MB; `resnet18_100pct_*` directories additionally retain `Ep10`–`Ep200` intermediate checkpoints).

| Directory | Corresponds to | Trained |
|---|---|---|
| `resnet18_100pct_Baseline_exclude_11_99/` | Table 4 — Baseline | 2026-04-30 |
| `resnet18_100pct_Strong_exclude_11_99/` | Table 4 — Strong Augmentation | 2026-04-30 |
| `resnet18_100pct_Mixup_exclude_11_99/` | Table 4 — Mixup | 2026-04-30 |
| `resnet18_100pct_CutMix_exclude_11_99/` | Table 4 — CutMix | 2026-05-01 |
| `resnet18_100pct_Mosaic_exclude_11_99/` | Table 4 — Mosaic | 2026-05-02 |
| `resnet18_100pct_RICAP_exclude_11_99/` | Table 4 — RICAP | 2026-05-02 |
| `wo_ordinal/` | Table 5 — w/o Ordinal Loss | 2026-05-17 |
| `wo_kl/` | Table 5 — w/o KL | 2026-05-23 |
| `random-energy/`, `random_area/`, `guided_area/` | Table 5 — 2×2 crop × weighting grid | 2026-03-30 – 03-31 |
| `intra_only/`, `neighbor_only/` | Table 5 — sampling-curriculum variants | 2026-03-31 |
| `PECAL/` | Full PECAL — **default** configuration (Gaussian targets, switch epoch 100, `n_global=2`, KL mix→strong) | 2026-08-03 |
| `curriculum_ep50/`, `curriculum_ep150/` | Sensitivity — switch epoch (the 100 default is `PECAL/`) | 2026-07-31 – 08-02 |
| `label_onehot/`, `label_uniform/` | Sensitivity — target encoding | 2026-08-04 – 08-05 |
| `kl_strong_to_mix/`, `kl_bidirectional/` | Sensitivity — KL direction | 2026-08-07 – 08-08 |
| `n_global_0/`, `n_global_1/`, `n_global_3/`, `n_global_4/`, `n_global_2_random/` | Sensitivity — global/local ratio | 2026-08-12 – 08-19 |

There is no separate `n_global_2/` directory: `n_global=2` is the default setting and is covered by `PECAL/`. Likewise `curriculum_ep100` shares the default configuration with `PECAL/`.

> **Distribution.** `checkpoint/` totals about **3.3 GB** and is therefore **not** tracked in this repository (see `.gitignore`); the packaged `PECAL_code_package.zip` excludes it as well. Obtain the weights from the release asset / hosted link accompanying the submission and place them under `checkpoint/`, then verify with:
>
> ```bash
> find checkpoint -name best.pth | wc -l    # expect 25
> ```

Note that Table 5's **Guided + Energy** cell is the same configuration as Full PECAL; the released grid covers `random-energy`, `random_area` and `guided_area`.

> **Weight ≠ traceable number.** Weights exist for all eight Table 5 configurations, but that does **not** make the reported numbers traceable — see §7.5. The `checkpoint/` directories were gathered from the historical runs and are provided so that the released models can be inspected and re-evaluated; use `test_ablation_full.py` / `test.py` to recompute metrics from them directly.

---

## 9. Known issues

Things worth knowing before you run anything long:

1. **Silent random initialisation.** Missing backbone weights cause a fallback to random init with no error (§3).
2. **`support` columns are doubled.** `test_ablation_full.py` globs eight extensions and extends the list per extension, so per-class `support` values in the result CSVs are exactly 2× the true counts (the true per-subset totals are listed in §4.1). Scores are unaffected.
3. **Batch-dependent test snapshots.** The baseline batch and the ablation batch were evaluated against different O6965 / O9113 test snapshots. Cross-batch comparisons are only valid on the first three subsets; "5-site mean" mixes evaluation protocols.
4. **`test.py` vs `test_ablation_full.py` argument conventions differ** (§6): the former takes explicit `--strategies` directory names, the latter discovers them.
5. **Legacy `train_baseline.py` weights cannot be reproduced bit-for-bit.** (`train_baseline.py` itself is not shipped here — it was superseded by `train.py`; see `MERGE_NOTES.md` for the migration.) The consolidated `train.py` fixed a missing ImageNet normalisation on the strong-augmentation branch (the legacy script fed `[0,1]` tensors to one KL branch while the other was already normalised). The legacy behaviour also used an ordinal coefficient of `1.0` and `CosineAnnealingWarmRestarts(T_0=100)`. To approximate it, set the corresponding keys in the augment config — `ordinal_weight: 1.0`, `lr_scheduler: cosine_warm_restarts`, `kl_direction: strong_to_mix` — rather than passing them on the command line; these are YAML keys, not CLI flags. See `MERGE_NOTES.md`.
6. **Winter wheat evaluation script is not shipped** (§6).
7. **Four GPUs hard-coded** in the training scripts (§3).

---

## 10. Citation

The paper is under review. Please cite this repository in the meantime:

```bibtex
@article{pecal2026,
  title   = {Phenology-Aware Energy-Guided Consistency and Ordinal Learning for
             Crop Growth Stage Recognition from Field RGB Images},
  author  = {},
  journal = {},
  year    = {},
  note    = {Code available at this repository}
}
```

---

## 11. License

Released for peer review and reproduction of the reported experiments. A licence will be added upon publication.

---

## 12. Additional documentation

| File | Contents |
|---|---|
| `CODE_MAP.md` | Paper table/figure → script + config + exact command, entry by entry, including the Table 5 line-by-line mapping and remaining gaps (§2.2) |
| `MERGE_NOTES.md` | How 8 legacy entry points were consolidated into `train.py` / `run.py`, and which behaviours changed |
| `README_zh.md` | Chinese counterpart of this file |
