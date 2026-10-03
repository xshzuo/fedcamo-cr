#!/usr/bin/env python3
"""
FedCAMO-CR 实验聚合脚本：把 results/baseline/**/log.json 聚合成
  - Table 5 同构表（CSV + LaTeX）
  - Fig 5 同构 Pareto frontier 图
  - Fig 10 同构（accuracy / trust / λ / μ）时间序列图

用法：python3 experiments/aggregate.py [--scenario baseline] [--out paper/tables]
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
import statistics

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES_OUT = ROOT / "paper" / "tables"
FIGS_OUT = ROOT / "paper" / "figures"
TABLES_OUT.mkdir(parents=True, exist_ok=True)
FIGS_OUT.mkdir(parents=True, exist_ok=True)


def load_runs(scenario: str) -> pd.DataFrame:
    """把 results/<scenario>/<config>/seed=*/log.json 加载成 DataFrame。"""
    base = RESULTS / scenario
    rows = []
    if not base.exists():
        return pd.DataFrame()
    for cfg_dir in sorted(base.iterdir()):
        if not cfg_dir.is_dir():
            continue
        for seed_dir in sorted(cfg_dir.glob("seed=*")):
            log_path = seed_dir / "log.json"
            if not log_path.exists():
                continue
            with open(log_path) as f:
                d = json.load(f)
            meta = d["meta"]
            rounds = d["rounds"]
            algo = meta.get("algo")
            attack = meta.get("attack")
            attack_ratio = meta.get("attack_ratio")
            scenario = meta.get("scenario", scenario)
            agg = meta.get("agg")
            seed_v = meta.get("seed", int(seed_dir.name.split("=")[1]))
            dataset_v = meta.get("dataset")
            cfg_dir_name = seed_dir.parent.name
            # 从目录名 robust 解析（algo_agg_attack_rX.X）
            parts = cfg_dir_name.split("_")
            # 找 r 结尾
            for i, p in enumerate(parts):
                if p.startswith("r") and i > 0:
                    try:
                        ratio_val = float(p[1:])
                        # 之前是 attack 名
                        attack_from_dir = "_".join(parts[i-1:i])
                        # agg 是 algo 之后到 attack 之前
                        if algo is None:
                            algo = "_".join(parts[:i-2]) if i >= 2 else cfg_dir_name
                        if agg is None:
                            agg = "_".join(parts[len(algo.split("_")):i-1])
                        if attack is None:
                            attack = attack_from_dir
                        if attack_ratio is None:
                            attack_ratio = ratio_val
                        break
                    except ValueError:
                        continue
            last10_acc = float(np.mean([r["accuracy"] for r in rounds[-10:]]))
            rows.append(dict(
                algo=algo,
                agg=agg or "?",
                attack=attack or "?",
                attack_ratio=attack_ratio if attack_ratio is not None else 0.0,
                scenario=scenario,
                seed=int(seed_v),
                dataset=dataset_v,
                budget_g=meta["budget_g"],
                final_acc=float(meta["final_acc"]),
                final_acc_last10=last10_acc,
                cum_carbon_g=float(meta["cum_carbon_g"]),
                final_lambda=float(meta.get("final_lambda", 0.0)),
                final_mu=float(meta.get("final_mu", 0.0)),
                avg_robustness_gap=float(meta.get("avg_robustness_gap", 0.0)),
            ))
    return pd.DataFrame(rows)


def make_table5(df: pd.DataFrame, scenario: str, out_csv: Path, out_tex: Path):
    """
    Table 5 同构：rows=algo×agg，cols=final_acc_last10 ± CI, cum_carbon_g
    """
    df = df.copy()
    df["label"] = df["algo"] + "+" + df["agg"]
    grp = df.groupby(["label", "attack"]).agg(
        acc_mean=("final_acc_last10", "mean"),
        acc_std=("final_acc_last10", "std"),
        acc_n=("final_acc_last10", "count"),
        carb_mean=("cum_carbon_g", "mean"),
        carb_std=("cum_carbon_g", "std"),
    ).reset_index()

    # 95% CI = 1.96 * std / sqrt(n)
    grp["acc_ci"] = 1.96 * grp["acc_std"] / np.sqrt(grp["acc_n"].clip(lower=1))
    grp["carb_ci"] = 1.96 * grp["carb_std"] / np.sqrt(grp["acc_n"].clip(lower=1))

    grp = grp.sort_values(["attack", "carb_mean"])
    grp.to_csv(out_csv, index=False)

    # LaTeX
    with open(out_tex, "w") as f:
        f.write(f"% Auto-generated Table 5 (scenario={scenario})\n")
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\caption{FedCAMO-CR results on " + scenario + r" scenario. "
                r"Accuracy is mean of last 10 rounds over $N$ seeds; "
                r"CO$_2$e is cumulative grams; CI is 95\%.}" + "\n")
        f.write(r"\label{tab:res-" + scenario + r"}" + "\n")
        f.write(r"\begin{tabular}{llrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Attack & Algo+Aggregator & Acc.\ (\%) & Acc. CI & "
                r"CO$_2$e (g) & CO$_2$e CI \\" + "\n")
        f.write(r"\midrule" + "\n")
        for _, r in grp.iterrows():
            f.write(f"{r['attack']} & {r['label']} & "
                    f"{r['acc_mean']*100:.2f} & $\\pm${r['acc_ci']*100:.2f} & "
                    f"{r['carb_mean']:.1f} & $\\pm${r['carb_ci']:.1f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"[table5] {out_csv}  {out_tex}")


def make_pareto(df: pd.DataFrame, scenario: str, out_path: Path):
    """Fig 5 同构：accuracy (x) vs CO2e (y)，Pareto frontier。"""
    df = df.copy()
    df["label"] = df["algo"] + "+" + df["agg"]
    grp = df.groupby(["label", "attack"]).agg(
        acc=("final_acc_last10", "mean"),
        carb=("cum_carbon_g", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    markers = {"none": "o", "lf": "s", "mr": "^", "aa": "D"}
    colors = {"none": "#888888", "lf": "#1f77b4",
              "mr": "#d62728", "aa": "#2ca02c"}

    # 关键算法优先高亮
    hl = {"fedcamo_cr_pr+trust_weighted", "fedcamo_cr_pr+fedavg",
          "fedcamo_pr+fedavg", "fedcamo_cb+fedavg", "fedavg+fedavg",
          "fedavg+krum", "fedavg+trimmed_mean", "fedavg+foolsgold"}

    # 过滤掉 carbon_only（CO₂ 极低但 Acc 极低，会拉爆 x 轴）
    grp = grp[~grp["label"].str.contains("carbon_only")]
    # 过滤 Acc < 5% 的退化点
    grp = grp[grp["acc"] > 0.05]

    plotted = []
    for attack in ["none", "lf", "mr", "aa"]:
        sub = grp[grp["attack"] == attack].sort_values("carb")
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            kw = dict(marker=markers[attack], color=colors[attack],
                      s=90, alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.scatter(r["carb"], r["acc"]*100, **kw, zorder=3)
            plotted.append((attack, r["label"], r["carb"], r["acc"]*100))

    # 用稍紧凑的标签布局：每个 attack 子集只标最高 acc 的 2 个
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker=markers[a], color="w",
                      markerfacecolor=colors[a], markersize=10,
                      markeredgecolor="black", label=a.upper())
               for a in ["none","lf","mr","aa"]]
    ax.legend(handles=handles, loc="lower right", title="Attack", fontsize=9)

    # 给每个 attack 的 top-2 acc 点贴标签，错开避免重叠
    for attack in ["none", "lf", "mr", "aa"]:
        sub = [p for p in plotted if p[0] == attack]
        sub = sorted(sub, key=lambda x: -x[3])[:3]
        for i, (at, lbl, x, y) in enumerate(sub):
            short = lbl.replace("fedcamo_cr_pr", "F-CR") \
                       .replace("fedcamo_pr", "F-PR") \
                       .replace("fedcamo_cb", "F-CB") \
                       .replace("trust_weighted", "TW") \
                       .replace("trimmed_mean", "TM") \
                       .replace("fedavg", "FA") \
                       .replace("foolsgold", "FG") \
                       .replace("krum", "Kr") \
                       .replace("+", "+")
            # 错开 y 偏移
            dy = 8 + (i % 3) * 4
            ax.annotate(short, (x, y), fontsize=6.5, ha="left",
                        xytext=(4, dy), textcoords="offset points")

    ax.set_xlabel(r"Cumulative CO$_2$e (g)")
    ax.set_ylabel("Final Test Accuracy (\\%)")
    ax.set_title(f"Pareto Frontier — {scenario}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[pareto] {out_path}")


def make_dynamics(df: pd.DataFrame, scenario: str, config_label: str,
                  out_path: Path):
    """Fig 10/11 同构：单个 (algo, agg, attack) 的 acc / λ / μ / trust 时间序列。"""
    base = RESULTS / scenario
    # 找到对应目录
    parts = config_label.split("+")
    if len(parts) != 2:
        return
    algo, agg = parts
    # 默认 attack=none seed=0
    pattern = f"{algo}_{agg}_*"
    candidates = list(base.glob(pattern))
    if not candidates:
        print(f"[dynamics] no match for {config_label}")
        return

    fig, axes = plt.subplots(4, 1, figsize=(7, 9), sharex=True)
    for c in candidates:
        attack = c.name.split("_")[-2]
        ratio = c.name.split("_")[-1]
        for seed_dir in c.glob("seed=*"):
            with open(seed_dir / "log.json") as f:
                d = json.load(f)
            rounds = d["rounds"]
            rs = [r["round"] for r in rounds]
            acc = [r["accuracy"]*100 for r in rounds]
            lam = [r["lambda_t"] for r in rounds]
            mu  = [r["mu_t"] for r in rounds]
            tr  = [r["trust_mean"] for r in rounds]
            axes[0].plot(rs, acc, alpha=0.4, lw=0.6)
            axes[1].plot(rs, lam, alpha=0.4, lw=0.6)
            axes[2].plot(rs, mu,  alpha=0.4, lw=0.6)
            axes[3].plot(rs, tr,  alpha=0.4, lw=0.6)

    axes[0].set_ylabel("Accuracy (\\%)")
    axes[1].set_ylabel(r"$\lambda$ (carbon price)")
    axes[2].set_ylabel(r"$\mu$ (robustness price)")
    axes[3].set_ylabel("Trust mean")
    axes[3].set_xlabel("Round")
    axes[0].set_title(f"Dynamics — {config_label} @ {scenario}")
    axes[0].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[dynamics] {out_path}")


def welch_ttest(df: pd.DataFrame, baseline_label: str, out_path: Path):
    """对 (algo, agg) vs FedAvg+FedAvg 做 Welch t-test，写成 LaTeX 表。"""
    base = df[(df["algo"] == "fedavg") & (df["agg"] == "fedavg")]
    rows = []
    for (algo, agg), sub in df.groupby(["algo", "agg"]):
        if algo == "fedavg" and agg == "fedavg":
            continue
        # 对每个 attack 单独测
        for attack in ["none", "lf", "mr", "aa"]:
            base_sub = base[base["attack"] == attack]["final_acc_last10"]
            targ_sub = sub[sub["attack"] == attack]["final_acc_last10"]
            if len(base_sub) < 2 or len(targ_sub) < 2:
                continue
            try:
                t, p = stats.ttest_ind(base_sub.values, targ_sub.values,
                                        equal_var=False)
            except Exception:
                p = float("nan")
            rows.append(dict(
                algo=algo, agg=agg, attack=attack,
                base_mean=base_sub.mean(), targ_mean=targ_sub.mean(),
                delta=base_sub.mean()-targ_sub.mean(),
                pvalue=p,
            ))
    pdf = pd.DataFrame(rows)
    pdf.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"[welch] {out_path.with_suffix('.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--out", default=str(TABLES_OUT))
    ap.add_argument("--cross-dataset", action="store_true",
                    help="when set, merge scenario/baseline, baseline_svhn, "
                         "baseline_gsc into a single CSV with dataset column")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.cross_dataset:
        # Merge three datasets
        scenarios = ["baseline", "baseline_svhn", "baseline_gsc"]
        frames = []
        for s in scenarios:
            df_s = load_runs(s)
            if df_s is not None and len(df_s) > 0:
                df_s = df_s.copy()
                # The dataset name may already be in meta; fall back to scenario suffix
                if "dataset" in df_s.columns:
                    pass
                else:
                    df_s["dataset"] = s.replace("baseline_", "") if "_" in s else "cifar10"
                frames.append(df_s)
        if not frames:
            print("no results found across datasets")
            return
        df = pd.concat(frames, ignore_index=True)
        # 'dataset' might not be in df.columns; infer if missing
        if "dataset" not in df.columns:
            df["dataset"] = "cifar10"
        # Backfill: dataset may be None if meta didn't carry it (old logs)
        df["dataset"] = df.apply(
            lambda r: r["dataset"] if r["dataset"]
                      else (r["scenario"].replace("baseline_", "") if r["scenario"].startswith("baseline_") else "cifar10"),
            axis=1,
        )
        out_csv = out / f"table5_cross_dataset.csv"
        out_tex = out / f"table5_cross_dataset.tex"
        make_table5_cross(df, out_csv, out_tex)
        make_pareto_cross(df, FIGS_OUT / "pareto_cross_dataset.png")
        return

    print(f"loading scenario={args.scenario} ...")
    df = load_runs(args.scenario)
    if df is None or len(df) == 0:
        print(f"no results for scenario={args.scenario}")
        return
    print(f"  rows={len(df)}  configs={df.groupby(['algo','agg','attack']).ngroups}")

    make_table5(df, args.scenario,
                out / f"table5_{args.scenario}.csv",
                out / f"table5_{args.scenario}.tex")
    make_pareto(df, args.scenario, FIGS_OUT / f"pareto_{args.scenario}.png")
    make_dynamics(df, args.scenario, "fedcamo_cr_pr+trust_weighted",
                  FIGS_OUT / f"dynamics_cr_{args.scenario}.png")
    make_dynamics(df, args.scenario, "fedcamo_pr+fedavg",
                  FIGS_OUT / f"dynamics_pr_{args.scenario}.png")
    welch_ttest(df, "fedavg+fedavg", out / f"welch_{args.scenario}")


def make_table5_cross(df, out_csv, out_tex):
    """Table 5 同构：rows=dataset×algo×agg，cols=acc, CO2e per attack."""
    df = df.copy()
    df["label"] = df["algo"] + "+" + df["agg"]
    grp = df.groupby(["dataset", "label", "attack"]).agg(
        acc_mean=("final_acc_last10", "mean"),
        acc_std=("final_acc_last10", "std"),
        acc_n=("final_acc_last10", "count"),
        carb_mean=("cum_carbon_g", "mean"),
        carb_std=("cum_carbon_g", "std"),
    ).reset_index()
    grp["acc_ci"] = 1.96 * grp["acc_std"] / np.sqrt(grp["acc_n"].clip(lower=1))
    grp["carb_ci"] = 1.96 * grp["carb_std"] / np.sqrt(grp["acc_n"].clip(lower=1))
    grp = grp.sort_values(["dataset", "attack", "carb_mean"])
    grp.to_csv(out_csv, index=False)

    with open(out_tex, "w") as f:
        f.write(r"% Auto-generated Table 5 (cross-dataset)" + "\n")
        f.write(r"\begin{table*}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\caption{FedCAMO-CR results across synthetic {CIFAR-10, SVHN, GSC} "
                r"regimes. Accuracy is mean of last 10 rounds over $N$ seeds; "
                r"CO$_2$e is cumulative grams; CI is 95\%.}" + "\n")
        f.write(r"\label{tab:res-cross-dataset}" + "\n")
        f.write(r"\begin{tabular}{lllrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Dataset & Attack & Algo+Aggregator & Acc.\ (\%) & Acc. CI & "
                r"CO$_2$e (g) & CO$_2$e CI \\" + "\n")
        f.write(r"\midrule" + "\n")
        for _, r in grp.iterrows():
            f.write(f"{r['dataset']} & {r['attack']} & {r['label']} & "
                    f"{r['acc_mean']*100:.2f} & $\\pm${r['acc_ci']*100:.2f} & "
                    f"{r['carb_mean']:.1f} & $\\pm${r['carb_ci']:.1f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table*}" + "\n")
    print(f"[table5-cross] {out_csv}  {out_tex}")


def make_pareto_cross(df, out_path):
    """3 个 dataset × 4 attack 共享一个 Pareto 图。"""
    df = df.copy()
    df["label"] = df["algo"] + "+" + df["agg"]
    grp = df.groupby(["dataset", "label", "attack"]).agg(
        acc=("final_acc_last10", "mean"),
        carb=("cum_carbon_g", "mean"),
    ).reset_index()
    # 过滤极端点
    grp = grp[~grp["label"].str.contains("carbon_only")]
    grp = grp[grp["acc"] > 0.05]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    markers = {"none": "o", "lf": "s", "mr": "^", "aa": "D"}
    colors = {"none": "#888888", "lf": "#1f77b4",
              "mr": "#d62728", "aa": "#2ca02c"}

    for ax, ds in zip(axes, ["cifar10", "svhn", "gsc"]):
        sub = grp[grp["dataset"] == ds]
        for attack in ["none", "lf", "mr", "aa"]:
            s = sub[sub["attack"] == attack]
            if s.empty: continue
            ax.scatter(s["carb"], s["acc"]*100,
                       marker=markers[attack], color=colors[attack],
                       s=60, alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_xlabel(r"Cumulative CO$_2$e (g)")
        ax.set_title(ds.upper())
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Final Test Accuracy (\\%)")

    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker=markers[a], color="w",
                      markerfacecolor=colors[a], markersize=10,
                      markeredgecolor="black", label=a.upper())
               for a in ["none","lf","mr","aa"]]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), title="Attack")
    fig.suptitle("Pareto Frontier — Cross-dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[pareto-cross] {out_path}")


if __name__ == "__main__":
    main()
