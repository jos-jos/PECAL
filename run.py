# -*- coding: utf-8 -*-
"""
run.py —— PECAL 统一批量运行器


本文件把 6+8+11 = 25 个实验收进一张注册表，共用一套循环。

⚠️ 每条实验都显式指定 impl（MixingDataset 实现）。

用法：
    python run.py --list                          # 列出 25 个实验
    python run.py --suite base                    # 基础对比 6 个（Table 4）
    python run.py --suite table5                  # 组件消融 8 个（Table 5）
    python run.py --suite sens                    # 四组敏感性 11 个（Fig 5）
    python run.py --suite sens --group D          # 只跑 D 组
    python run.py --only full_pecal n_global_4    # 按名字挑
    python run.py --skip-existing                 # 已有 best.pth 的跳过
    python run.py --dry-run                       # 只打印命令
    python run.py --seeds 42 123 2024             # 多种子（目录加 seed_ 层）
    python run.py --test-only --suite table5      # 只评测

"""
import argparse
import os
import subprocess
import sys
import time

PYTHON = sys.executable
TRAIN_SCRIPT = "train.py"

# impl 取值与 datasets/dataset_factory.py 一致
IMPL_ABLATION = "ablation"
IMPL_N_GLOBAL = "n_global"
IMPL_DUALSTREAM2 = "dualstream2"
IMPL_DUALSTREAM2_ABLATION = "dualstream2_ablation"


def _e(eid, suite, name, desc, config, exp_dir, impl, group=None, test_script=None):
    return {"id": eid, "suite": suite, "group": group, "name": name, "desc": desc,
            "config": config, "exp_dir": exp_dir, "impl": impl,
            "test_script": test_script or "test_ablation_full.py"}


# ==========================================================================
# 注册表：6（base）+ 8（table5）+ 11（sens）= 25
# ==========================================================================
AUG = "configs/augment/"
EXP_CFG = "configs/experiments/"

EXPERIMENTS = [
    # ---------- base：基础对比（Table 4）----------
    # 由 train.py + dualstream2 实现训练。
    _e(1, "base", "Baseline_exclude_11_99", "无混合、无一致性、无序数损失",
       AUG + "base.yaml", "experiments/resnet18_100pct_Baseline_exclude_11_99",
       IMPL_DUALSTREAM2, test_script="test.py"),
    _e(2, "base", "Strong_exclude_11_99", "RandAugment N=2, M=9",
       AUG + "strong.yaml", "experiments/resnet18_100pct_Strong_exclude_11_99",
       IMPL_DUALSTREAM2, test_script="test.py"),
    _e(3, "base", "Mixup_exclude_11_99", "alpha=0.8",
       AUG + "mixup.yaml", "experiments/resnet18_100pct_Mixup_exclude_11_99",
       IMPL_DUALSTREAM2, test_script="test.py"),
    _e(4, "base", "CutMix_exclude_11_99", "alpha=1.0",
       AUG + "cutmix.yaml", "experiments/resnet18_100pct_CutMix_exclude_11_99",
       IMPL_DUALSTREAM2, test_script="test.py"),
    _e(5, "base", "Mosaic_exclude_11_99", "拼接概率 0.5，面积加权",
       AUG + "mosaic.yaml", "experiments/resnet18_100pct_Mosaic_exclude_11_99",
       IMPL_DUALSTREAM2, test_script="test.py"),
    _e(6, "base", "RICAP_exclude_11_99", "随机模式，面积加权",
       AUG + "ricap.yaml", "experiments/resnet18_100pct_RICAP_exclude_11_99",
       IMPL_DUALSTREAM2, test_script="test.py"),

    # ---------- table5：组件消融（Table 5 的 8 行）----------
    _e(7, "table5", "full_pecal", "Full-PECAL（基准）",
       EXP_CFG + "full_pecal.yaml", "experiments/table5/full_pecal", IMPL_ABLATION),
    _e(8, "table5", "wo_ordinal", "w/o Ordinal Loss（beta=0）",
       EXP_CFG + "wo_ordinal.yaml", "experiments/table5/wo_ordinal", IMPL_ABLATION),
    _e(9, "table5", "random_area", "Random + Area",
       EXP_CFG + "random_area.yaml", "experiments/table5/random_area", IMPL_ABLATION),
    _e(10, "table5", "guided_area", "Guided + Area",
       EXP_CFG + "guided_area.yaml", "experiments/table5/guided_area", IMPL_ABLATION),
    _e(11, "table5", "random_energy", "Random + Energy",
       EXP_CFG + "random_energy.yaml", "experiments/table5/random_energy", IMPL_ABLATION),
    _e(12, "table5", "intra_only", "Intra-only（固定类内）",
       EXP_CFG + "intra_only.yaml", "experiments/table5/intra_only", IMPL_ABLATION),
    _e(13, "table5", "neighbor_only", "Neighbor-only（固定相邻）",
       EXP_CFG + "neighbor_only.yaml", "experiments/table5/neighbor_only", IMPL_ABLATION),
    _e(14, "table5", "wo_kl", "w/o KL（一致性关闭）",
       EXP_CFG + "wo_kl.yaml", "experiments/table5/wo_kl", IMPL_ABLATION),

    # ---------- sens：四组敏感性（Fig 5）----------
    _e(15, "sens", "curriculum_ep50", "switch_epoch=50", AUG + "ablation_curriculum_ep50.yaml",
       "experiments/ablation_full/curriculum_ep50", IMPL_ABLATION, group="A"),
    _e(16, "sens", "curriculum_ep150", "switch_epoch=150", AUG + "ablation_curriculum_ep150.yaml",
       "experiments/ablation_full/curriculum_ep150", IMPL_ABLATION, group="A"),
    _e(17, "sens", "label_onehot", "one-hot encoding", AUG + "ablation_label_onehot.yaml",
       "experiments/ablation_full/label_onehot", IMPL_ABLATION, group="B"),
    _e(18, "sens", "label_uniform", "uniform smoothing (eps=0.1)", AUG + "ablation_label_uniform.yaml",
       "experiments/ablation_full/label_uniform", IMPL_ABLATION, group="B"),
    _e(19, "sens", "kl_strong_to_mix", "KL(strong || mix)", AUG + "ablation_kl_strong_to_mix.yaml",
       "experiments/ablation_full/kl_strong_to_mix", IMPL_ABLATION, group="C"),
    _e(20, "sens", "kl_bidirectional", "KL(mix||strong) + KL(strong||mix)",
       AUG + "ablation_kl_bidirectional.yaml", "experiments/ablation_full/kl_bidirectional",
       IMPL_ABLATION, group="C"),
    _e(21, "sens", "n_global_4", "resize x4（全部全局缩放）", AUG + "n_global_4.yaml",
       "experiments/n_global_ablation/n_global_4", IMPL_N_GLOBAL, group="D"),
    _e(22, "sens", "n_global_3", "resize x3 + saliency x1", AUG + "n_global_3.yaml",
       "experiments/n_global_ablation/n_global_3", IMPL_N_GLOBAL, group="D"),
    _e(23, "sens", "n_global_1", "resize x1 + saliency x3", AUG + "n_global_1.yaml",
       "experiments/n_global_ablation/n_global_1", IMPL_N_GLOBAL, group="D"),
    _e(24, "sens", "n_global_0", "saliency x4（全部局部裁剪）", AUG + "n_global_0.yaml",
       "experiments/n_global_ablation/n_global_0", IMPL_N_GLOBAL, group="D"),
    _e(25, "sens", "n_global_2_random", "resize x2 + saliency x2（随机位置）",
       AUG + "n_global_2_random.yaml", "experiments/n_global_ablation/n_global_2_random",
       IMPL_N_GLOBAL, group="D"),
]


def exp_dir_for(exp, out_root, seed, n_seeds):
    """多种子时在 out_root 下加一层 seed_<n>。"""
    base = os.path.join(out_root, f"seed_{seed}") if n_seeds > 1 else out_root
    return os.path.join(base, exp["name"])


def run_stream(cmd, log_path=None):
    """实时回显 + 落盘，返回返回码。"""
    print(f"  $ {subprocess.list2cmdline(cmd)}", flush=True)
    if log_path is None:
        return subprocess.call(cmd)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
            lf.flush()
        proc.wait()
        return proc.returncode


def collect_results(exps):
    """读取各实验 log.csv 的 val_f1 峰值，打印汇总表。"""
    import csv
    print("\n" + "=" * 96)
    print("  Results Summary (best val_f1 from log.csv)")
    print("=" * 96)
    suites = {}
    for e in exps:
        suites.setdefault(e["suite"], []).append(e)

    for suite, items in suites.items():
        print(f"\n  --- suite: {suite} ---")
        print(f"  {'ID':<4}{'Name':<26}{'Description':<34}{'Best F1':<10}{'Status':<10}")
        print("  " + "-" * 84)
        for e in items:
            log_csv = os.path.join(e["exp_dir"], "log.csv")
            f1, status = "N/A", "missing"
            if os.path.exists(log_csv):
                try:
                    with open(log_csv, encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        col = next((c for c in ("val_f1", "f1") if c in rows[0]), None)
                        if col:
                            vals = [float(r[col]) for r in rows if r.get(col)]
                            if vals:
                                f1 = f"{max(vals):.4f}"
                        status = "done"
                except Exception:
                    status = "error"
            print(f"  {e['id']:<4}{e['name']:<26}{e['desc']:<34}{f1:<10}{status:<10}")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser(description="PECAL 统一批量运行器（25 个实验）")
    ap.add_argument("--list", action="store_true", help="列出全部实验后退出")
    ap.add_argument("--suite", default=None,
                    choices=["base", "table5", "sens", "all"], help="选择实验套件")
    ap.add_argument("--group", default=None, choices=["A", "B", "C", "D"],
                    help="只跑敏感性实验的某一组")
    ap.add_argument("--only", nargs="+", default=None,
                    help="按名字或序号挑选，如 --only full_pecal 21")
    ap.add_argument("--out-root", default=None,
                    help="覆盖输出根目录（默认沿用注册表里的 exp_dir）")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config-train", default=None)
    ap.add_argument("--config-model", default=None)
    ap.add_argument("--config-dataset", default=None)
    ap.add_argument("--test-after", action="store_true", help="训练完自动评测")
    ap.add_argument("--test-only", action="store_true", help="跳过训练只评测")
    ap.add_argument("--test-csv", default="test_results/results.csv")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'ID':<4}{'SUITE':<8}{'GRP':<5}{'NAME':<26}{'IMPL':<22}{'EXP DIR'}")
        print("-" * 118)
        for e in EXPERIMENTS:
            print(f"{e['id']:<4}{e['suite']:<8}{str(e['group'] or '-'):<5}{e['name']:<26}"
                  f"{e['impl']:<22}{e['exp_dir']}")
        return

    # ---------- 筛选 ----------
    exps = EXPERIMENTS
    if args.suite and args.suite != "all":
        exps = [e for e in exps if e["suite"] == args.suite]
    if args.group:
        exps = [e for e in exps if e["group"] == args.group]
    if args.only:
        wanted = {str(x) for x in args.only}
        sel = [e for e in EXPERIMENTS if e["name"] in wanted or str(e["id"]) in wanted]
        if not sel:
            ap.error(f"没有匹配的实验: {args.only}（用 --list 查看可选）")
        exps = sel
    if not exps:
        ap.error("没有选中的实验")

    extra = []
    for flag, val in (("--config-train", args.config_train),
                      ("--config-model", args.config_model),
                      ("--config-dataset", args.config_dataset)):
        if val:
            extra += [flag, val]

    n_jobs = len(exps) * len(args.seeds)
    print("=" * 78)
    print(f"  PECAL Batch Runner · {len(exps)} exp(s) × {len(args.seeds)} seed(s) = {n_jobs} job(s)")
    print(f"  train script: {TRAIN_SCRIPT}    skip-existing: {args.skip_existing}")
    print("=" * 78)

    # ---------- 训练 ----------
    if not args.test_only:
        for seed in args.seeds:
            for i, exp in enumerate(exps, 1):
                d = exp_dir_for(exp, args.out_root, seed, len(args.seeds)) \
                    if args.out_root else exp["exp_dir"]
                cmd = [PYTHON, TRAIN_SCRIPT,
                       "--config-augment", exp["config"],
                       "--exp-dir", d,
                       "--seed", str(seed),
                       "--dataset-impl", exp["impl"]] + extra

                print(f"\n[{i}/{len(exps)}] {exp['name']} — {exp['desc']}  (seed={seed}, impl={exp['impl']})")
                print(f"  -> {d}")
                if args.skip_existing and os.path.exists(os.path.join(d, "best.pth")):
                    print("  [SKIP] best.pth 已存在")
                    continue
                if args.dry_run:
                    continue
                rc = run_stream(cmd, log_path=os.path.join(d, "run.log"))
                if rc != 0:
                    print(f"  [FAIL] {exp['name']} 返回码 {rc}，日志 {d}/run.log")
                    print("  继续下一个；如需中止请 Ctrl-C。")

    # ---------- 评测 ----------
    if args.test_after or args.test_only:
        for seed in args.seeds:
            csv_out = args.test_csv
            if len(args.seeds) > 1:
                stem, ext = os.path.splitext(args.test_csv)
                csv_out = f"{stem}_seed{seed}{ext}"
            # 按 test_script 分组，逐个目录评测
            for e in exps:
                d = exp_dir_for(e, args.out_root, seed, len(args.seeds)) \
                    if args.out_root else e["exp_dir"]
                cmd = [PYTHON, e["test_script"], "--exp_dir", d,
                       "--output_csv", csv_out]
                print(f"\n=== 评测: {d} -> {csv_out}  ({e['test_script']})")
                if args.dry_run:
                    continue
                run_stream(cmd)

    if args.dry_run:
        print("\n[dry-run] 未执行任何训练/评测。")
    elif not args.no_summary:
        collect_results(exps)


if __name__ == "__main__":
    main()
