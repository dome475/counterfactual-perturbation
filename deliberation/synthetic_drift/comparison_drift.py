"""
normalized drift response profiles: overlays pid synergy, non-syn mi,
br gap and nbs distance per game config, all normalized to alpha=0.
control synergy is flat at ~1.0 by construction and serves as reference.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extract_features import compute_ideal
from pid_numpy import bootstrap_ci, N_BOOT
from synthetic_drift import (GAME_CONFIGS, invert_scores,
                             build_drift_trajectory, build_o_information_trajectory)
from game_theory_drift import run_sweep

FIGURES_DIR  = Path(__file__).parent / "figures"
RESULTS_DIR  = Path(__file__).parent / "results"


# cache helpers

def _cache_path(cfg_name, kind):
    return RESULTS_DIR / f"comparison_{kind}_{cfg_name}.json"


def _save(data, path):
    # convert float keys -> str, tuples -> lists for JSON
    serializable = {str(k): {ik: list(iv) for ik, iv in v.items()}
                    for k, v in data.items()}
    with open(path, "w") as f:
        json.dump(serializable, f)


def _load(path):
    with open(path) as f:
        raw = json.load(f)
    return {float(k): {ik: tuple(iv) for ik, iv in v.items()}
            for k, v in raw.items()}


# pid series

def get_pid_series(cfg):
    drift_agent = cfg["drift_agent"]
    agent_keys  = cfg["agent_keys"]
    issue_sizes = cfg["issue_sizes"]

    scores_base, _ = cfg["load_scores"]()
    scores_adv     = invert_scores(scores_base, drift_agent)

    score_ranges = {}
    for a in agent_keys:
        _, ideal_sc, worst_sc = compute_ideal(scores_base, a, issue_sizes)
        score_ranges[a] = max(ideal_sc - worst_sc, 1)

    traj = build_drift_trajectory(scores_base, scores_adv, cfg["condition"],
                                   agent_keys, drift_agent, score_ranges)

    for data in traj.values():
        for lst in data.values():
            for atom in lst:
                atom["NonSyn"] = atom["Red"] + atom["Un1"] + atom["Un2"]

    result = {}
    for alpha, data in traj.items():
        d_lst = data["drift"]
        c_lst = data["ctrl"]
        result[float(alpha)] = {
            "syn_drift":    bootstrap_ci(d_lst, "Syn",    n_boot=N_BOOT) if d_lst else (np.nan,)*3,
            "nonsyn_drift": bootstrap_ci(d_lst, "NonSyn", n_boot=N_BOOT) if d_lst else (np.nan,)*3,
            "syn_ctrl":     bootstrap_ci(c_lst, "Syn",    n_boot=N_BOOT) if c_lst  else (np.nan,)*3,
        }
    return result


# o-information series

def get_o_info_series(cfg):
    drift_agent = cfg["drift_agent"]
    agent_keys  = cfg["agent_keys"]
    issue_sizes = cfg["issue_sizes"]

    scores_base, _ = cfg["load_scores"]()
    scores_adv     = invert_scores(scores_base, drift_agent)

    score_ranges = {}
    for a in agent_keys:
        _, ideal_sc, worst_sc = compute_ideal(scores_base, a, issue_sizes)
        score_ranges[a] = max(ideal_sc - worst_sc, 1)

    traj = build_o_information_trajectory(scores_base, scores_adv, cfg["condition"],
                                          agent_keys, drift_agent, score_ranges)
    return {alpha: {"o_info": v} for alpha, v in traj.items()}


# plot

LINES = [
    # (source, key,           label,                          color,     ls,   lw)
    ("pid", "syn_drift",    "PID synergy (drift pairs)",      "#d62728", "-",  2.5),
    ("pid", "nonsyn_drift", "PID non-syn MI (drift pairs)",   "#ff7f0e", "-",  2.0),
    ("gt",  "br_drift",    "best-response gap",               "#2ca02c", "-",  2.0),
    ("gt",  "nbs_interp",  "NBS distance",                    "#9467bd", "-",  2.0),
    ("pid", "syn_ctrl",    "PID synergy ctrl (ref, ~1.0)",    "#aaaaaa", "--", 1.5),
    ("oi",  "o_info",      "O-information (whole system)",    "#17becf", "-.", 2.0),
]


def plot_comparison(all_results):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cfg_names = list(GAME_CONFIGS.keys())
    n_cols = 2
    n_rows = (len(cfg_names) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows), sharey=True)
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    # hide unused panels
    for ax in axes_flat[len(cfg_names):]:
        ax.set_visible(False)

    legend_handles = []

    for idx, (ax, cfg_name) in enumerate(zip(axes_flat, cfg_names)):
        cfg   = GAME_CONFIGS[cfg_name]
        pid_s = all_results[cfg_name]["pid"]
        gt_s  = all_results[cfg_name]["gt"]

        alphas = sorted(pid_s.keys())
        a0     = alphas[0]

        oi_s = all_results[cfg_name]["oi"]
        baselines = {
            "syn_drift":    pid_s[a0]["syn_drift"][0],
            "nonsyn_drift": pid_s[a0]["nonsyn_drift"][0],
            "syn_ctrl":     pid_s[a0]["syn_ctrl"][0],
            "br_drift":     gt_s[a0]["br_drift"][0],
            "nbs_interp":   gt_s[a0]["nbs_interp"][0],
            "o_info":       oi_s[a0]["o_info"][0],
        }

        print(f"\n  {cfg_name} baselines at alpha=0:")
        for k, v in baselines.items():
            print(f"    {k:<20} {v:.4f}")

        for src, key, label, color, ls, lw in LINES:
            series = pid_s if src == "pid" else (gt_s if src == "gt" else oi_s)
            b = baselines[key]
            if abs(b) < 1e-9:
                print(f"  skip {cfg_name}/{key}: baseline ~ 0")
                continue

            xs, ms, los, his = [], [], [], []
            for alpha in alphas:
                m, lo, hi = series[alpha][key]
                if np.isnan(m):
                    continue
                xs.append(alpha)
                ms.append(m / b)
                los.append(lo / b)
                his.append(hi / b)

            h, = ax.plot(xs, ms, color=color, ls=ls, lw=lw,
                         marker="o", ms=4, label=label)
            if cfg_name == cfg_names[0]:
                legend_handles.append(h)

        ax.axhline(1.0, color="black", lw=0.7, ls=":", alpha=0.4)
        ax.set_title(f"configuration {idx + 1}", fontsize=10)
        ax.set_xlabel("alpha  (0 = base, 1 = adversarial)", fontsize=10)
        if ax is axes_flat[0] or axes_flat.tolist().index(ax) % n_cols == 0:
            ax.set_ylabel("fraction of baseline  (alpha = 0)", fontsize=10)
        ax.set_ylim(0, 3.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc="lower center", ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Normalized drift response: PID vs game-theoretic quantities", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out = FIGURES_DIR / "comparison_drift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")

    # absolute magnitudes, for figure caption; normalization can visually compress
    # quantities with large baselines (e.g. NBS ~66 -> ~121) relative to quantities
    # with small baselines (e.g. PID Syn ~0.38 -> ~0.06)
    print("\nABSOLUTE VALUES  (for caption, normalization compresses large-baseline quantities)")
    print(f"  {'game':<14}  {'quantity':<20}  {'alpha=0':>8}  {'alpha=1':>8}  {'ratio':>7}")
    print("-" * 66)
    for cfg_name in cfg_names:
        pid_s = all_results[cfg_name]["pid"]
        gt_s  = all_results[cfg_name]["gt"]
        a0    = min(pid_s.keys())
        a1    = max(pid_s.keys())
        oi_s  = all_results[cfg_name]["oi"]
        rows  = [
            ("PID Syn (drift)",    "pid", "syn_drift"),
            ("PID NonSyn (drift)", "pid", "nonsyn_drift"),
            ("PID Syn (ctrl)",     "pid", "syn_ctrl"),
            ("BR gap",             "gt",  "br_drift"),
            ("NBS dist",           "gt",  "nbs_interp"),
            ("O-information",      "oi",  "o_info"),
        ]
        for lbl, src, key in rows:
            series = pid_s if src == "pid" else (gt_s if src == "gt" else oi_s)
            v0 = series[a0][key][0]
            v1 = series[a1][key][0]
            ratio = v1 / v0 if abs(v0) > 1e-9 else float("nan")
            print(f"  {cfg_name:<14}  {lbl:<20}  {v0:>8.4f}  {v1:>8.4f}  {ratio:>7.3f}")


# main

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {}
    for cfg_name, cfg in GAME_CONFIGS.items():
        pid_path = _cache_path(cfg_name, "pid")
        gt_path  = _cache_path(cfg_name, "gt")
        oi_path  = _cache_path(cfg_name, "oi")

        if pid_path.exists():
            print(f"  {cfg_name}: loading cached PID from {pid_path.name}")
            pid_s = _load(pid_path)
        else:
            print(f"\n=== {cfg_name}, PID sweep ===")
            pid_s = get_pid_series(cfg)
            _save(pid_s, pid_path)
            print(f"  saved -> {pid_path.name}")

        if gt_path.exists():
            print(f"  {cfg_name}: loading cached GT from {gt_path.name}")
            gt_s = _load(gt_path)
        else:
            print(f"\n=== {cfg_name}, GT sweep ===")
            gt_s = run_sweep(cfg_name, cfg)
            _save(gt_s, gt_path)
            print(f"  saved -> {gt_path.name}")

        if oi_path.exists():
            print(f"  {cfg_name}: loading cached O-info from {oi_path.name}")
            oi_s = _load(oi_path)
        else:
            print(f"\n=== {cfg_name}, O-information sweep ===")
            oi_s = get_o_info_series(cfg)
            _save(oi_s, oi_path)
            print(f"  saved -> {oi_path.name}")

        all_results[cfg_name] = {"pid": pid_s, "gt": gt_s, "oi": oi_s}

    plot_comparison(all_results)


if __name__ == "__main__":
    main()
