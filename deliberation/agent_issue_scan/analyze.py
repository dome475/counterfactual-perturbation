"""
agent x issue joint scan.
for each of 6 agents x 5 issues x 2 conditions, runs alpha-perturbation at endpoints only.
cluster bootstrap only on cells with |gap| >= CI_THRESHOLD.
outputs: scan.json, scan_heatmaps.png, scan_gap.png.
"""
import copy
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "synthetic_drift"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from extract_features import load_scores, AGENT_KEYS
from pairwise_pid import fit_quantile_bins, apply_bins, N_BINS, N_TARGET
from pid_numpy import imin_numpy
from synthetic_drift import compute_states_at_alpha

FEAT_DIR    = Path(__file__).parent.parent / "collusion" / "features"
RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR     = Path(__file__).parent / "figures"

MIN_SAMPLES  = 10
SCORE_RANGES = {a: 100 for a in AGENT_KEYS}
CONDITIONS   = ["control", "collusion"]
ISSUE_NAMES  = ["A", "B", "C", "D", "E"]
N_ISSUES     = 5
CI_THRESHOLD = 0.10   # only bootstrap cells with |gap| >= this


# data

def load_games(condition):
    games = []
    for p in sorted(FEAT_DIR.glob(f"{condition}_*.json")):
        g = json.load(open(p))
        if g["n_rounds_parsed"] >= 10:
            games.append(g)
        else:
            print(f"  skip {g['game_id']}: {g['n_rounds_parsed']} rounds")
    return games


def build_adv_scores(scores_base, drift_agent, issue_idx):
    s = copy.deepcopy(scores_base)
    s[drift_agent][issue_idx] = list(reversed(s[drift_agent][issue_idx]))
    return s


# edges / triplets

def fit_edges(games, scores_base, scores_adv, drift_agent):
    all_states, all_deltas = [], []
    for g in games:
        ss = compute_states_at_alpha(g, scores_base, scores_adv, 0.0,
                                     AGENT_KEYS, drift_agent, SCORE_RANGES)
        all_states.extend(ss)
        for idx in range(len(ss) - 1):
            s, sn = ss[idx], ss[idx + 1]
            for (j, k) in combinations(AGENT_KEYS, 2):
                noncoal = [a for a in AGENT_KEYS if a not in (j, k)]
                all_deltas.append(sum(sn["scores"][a] - s["scores"][a] for a in noncoal))
    return {
        "self":  {a: fit_quantile_bins([s["self_scores"][a] for s in all_states])
                  for a in AGENT_KEYS},
        "sens":  {a: fit_quantile_bins([s["sensitivity"][a] for s in all_states])
                  for a in AGENT_KEYS},
        "delta": fit_quantile_bins(all_deltas, n_bins=N_TARGET),
    }


def collect_game_triplets(games, scores_base, scores_adv, edges, drift_agent):
    self_e  = edges["self"]
    sens_e  = edges["sens"]
    delta_e = edges["delta"]

    def jbin(s, a):
        return apply_bins(s["self_scores"][a], self_e[a]) * N_BINS + \
               apply_bins(s["sensitivity"][a], sens_e[a])

    per_game = []
    for g in games:
        gdata = {}
        for alpha in [0.0, 1.0]:
            ss = compute_states_at_alpha(g, scores_base, scores_adv, alpha,
                                         AGENT_KEYS, drift_agent, SCORE_RANGES)
            pt = defaultdict(list)
            for idx in range(len(ss) - 1):
                s, sn = ss[idx], ss[idx + 1]
                for (j, k) in combinations(AGENT_KEYS, 2):
                    noncoal = [a for a in AGENT_KEYS if a not in (j, k)]
                    delta = sum(sn["scores"][a] - s["scores"][a] for a in noncoal)
                    pt[(j, k)].append((jbin(s, j), jbin(s, k),
                                       apply_bins(delta, delta_e)))
            gdata[alpha] = dict(pt)
        per_game.append(gdata)
    return per_game


# metrics

def _syn_from_pool(per_game_subset, alpha, drift_agent):
    pair_triplets = defaultdict(list)
    for gd in per_game_subset:
        for pair, trips in gd[alpha].items():
            pair_triplets[pair].extend(trips)
    drift_atoms = []
    for (j, k), triplets in pair_triplets.items():
        if len(triplets) < MIN_SAMPLES or drift_agent not in (j, k):
            continue
        drift_atoms.append(imin_numpy(triplets))
    return float(np.mean([a["Syn"] for a in drift_atoms])) if drift_atoms else float("nan")


def point_estimate(per_game, drift_agent):
    s0 = _syn_from_pool(per_game, 0.0, drift_agent)
    s1 = _syn_from_pool(per_game, 1.0, drift_agent)
    ratio = s1 / s0 if not np.isnan(s0) and s0 > 1e-9 else float("nan")
    return s0, s1, ratio


def cluster_bootstrap_ratio(per_game, drift_agent, n_boot=1000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n   = len(per_game)
    boot_ratios = []
    for _ in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        subset = [per_game[i] for i in idx]
        s0 = _syn_from_pool(subset, 0.0, drift_agent)
        s1 = _syn_from_pool(subset, 1.0, drift_agent)
        if not np.isnan(s0) and s0 > 1e-9:
            boot_ratios.append(s1 / s0)
    if len(boot_ratios) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(boot_ratios, 2.5)), float(np.percentile(boot_ratios, 97.5))


# plots

def plot_heatmaps(grid, gap):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, cond in zip(axes, CONDITIONS):
        data = np.array([[grid[ag][i][cond]["ratio"] for i in range(N_ISSUES)]
                         for ag in AGENT_KEYS], dtype=float)
        im = ax.imshow(data, vmin=0.4, vmax=1.3, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(N_ISSUES)); ax.set_xticklabels(ISSUE_NAMES, fontsize=8)
        ax.set_yticks(range(len(AGENT_KEYS))); ax.set_yticklabels(AGENT_KEYS, fontsize=8)
        ax.set_title(f"{cond}, synergy ratio (syn@1/syn@0)", fontsize=9)
        for ai, ag in enumerate(AGENT_KEYS):
            for ii in range(N_ISSUES):
                r = grid[ag][ii][cond]["ratio"]
                txt = f"{r:.2f}" if not np.isnan(r) else ","
                ax.text(ii, ai, txt, ha="center", va="center", fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = FIG_DIR / "scan_heatmaps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")

    # gap heatmap
    gap_data = np.array([[gap[ag][i] for i in range(N_ISSUES)]
                          for ag in AGENT_KEYS], dtype=float)
    vabs = max(0.30, np.nanmax(np.abs(gap_data)))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(gap_data, norm=norm, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(N_ISSUES)); ax.set_xticklabels(ISSUE_NAMES, fontsize=8)
    ax.set_yticks(range(len(AGENT_KEYS))); ax.set_yticklabels(AGENT_KEYS, fontsize=8)
    ax.set_title("collusion − control gap  (target: SportCo x D)", fontsize=9)
    for ai, ag in enumerate(AGENT_KEYS):
        for ii in range(N_ISSUES):
            g = gap_data[ai, ii]
            txt = f"{g:+.2f}" if not np.isnan(g) else ","
            weight = "bold" if (ag in ("SportCo", "union") and ii == 3) else "normal"
            ax.text(ii, ai, txt, ha="center", va="center", fontsize=6, fontweight=weight)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = FIG_DIR / "scan_gap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")


# main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scores_base, _ = load_scores()
    games_by_cond  = {c: load_games(c) for c in CONDITIONS}
    for c, gs in games_by_cond.items():
        print(f"{c}: {len(gs)} games")

    # phase 1: point estimates for all 30x2 cells
    print("\n=== phase 1: point estimates (60 cells) ===")
    per_game_cache = {}
    grid = {ag: {i: {} for i in range(N_ISSUES)} for ag in AGENT_KEYS}

    total = len(AGENT_KEYS) * N_ISSUES * len(CONDITIONS)
    done  = 0
    for drift_agent in AGENT_KEYS:
        for issue_idx in range(N_ISSUES):
            orig       = scores_base[drift_agent][issue_idx]
            scores_adv = build_adv_scores(scores_base, drift_agent, issue_idx)
            for condition in CONDITIONS:
                games    = games_by_cond[condition]
                edges    = fit_edges(games, scores_base, scores_adv, drift_agent)
                per_game = collect_game_triplets(games, scores_base, scores_adv,
                                                 edges, drift_agent)
                s0, s1, ratio = point_estimate(per_game, drift_agent)
                per_game_cache[(drift_agent, issue_idx, condition)] = per_game
                grid[drift_agent][issue_idx][condition] = {
                    "syn_a0": s0, "syn_a1": s1, "ratio": ratio,
                    "ratio_ci_low": float("nan"), "ratio_ci_high": float("nan"),
                }
                done += 1
                print(f"  [{done}/{total}] {drift_agent} Issue {ISSUE_NAMES[issue_idx]} "
                      f"{condition}: ratio={ratio:.3f}")

    # compute gaps
    gap = {}
    for ag in AGENT_KEYS:
        gap[ag] = {}
        for i in range(N_ISSUES):
            rc = grid[ag][i]["control"]["ratio"]
            rl = grid[ag][i]["collusion"]["ratio"]
            gap[ag][i] = rl - rc if not (np.isnan(rc) or np.isnan(rl)) else float("nan")

    # phase 2: bootstrap only cells with |gap| >= threshold
    print(f"\n=== phase 2: cluster bootstrap (|gap| >= {CI_THRESHOLD}) ===")
    n_boot_cells = 0
    for ag in AGENT_KEYS:
        for i in range(N_ISSUES):
            g = gap[ag][i]
            if np.isnan(g) or abs(g) < CI_THRESHOLD:
                continue
            print(f"  {ag} Issue {ISSUE_NAMES[i]}  gap={g:+.3f}")
            for condition in CONDITIONS:
                per_game = per_game_cache[(ag, i, condition)]
                lo, hi = cluster_bootstrap_ratio(per_game, ag)
                grid[ag][i][condition]["ratio_ci_low"]  = lo
                grid[ag][i][condition]["ratio_ci_high"] = hi
            n_boot_cells += 1
    print(f"  bootstrapped {n_boot_cells} cells")

    # summary table
    print("\n=== gap summary (collusion − control) ===")
    header = "  ".join(f"Iss{n}" for n in ISSUE_NAMES)
    print(f"  {'agent':<12}  {header}")
    for ag in AGENT_KEYS:
        row = "  ".join(f"{gap[ag][i]:+.3f}" if not np.isnan(gap[ag][i]) else "   nan"
                        for i in range(N_ISSUES))
        print(f"  {ag:<12}  {row}")

    plot_heatmaps(grid, gap)

    def _clean(obj):
        if isinstance(obj, float) and np.isnan(obj): return None
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        return obj

    out_data = {
        "grid": {
            ag: {
                str(i): {cond: grid[ag][i][cond] for cond in CONDITIONS}
                for i in range(N_ISSUES)
            } for ag in AGENT_KEYS
        },
        "gap":  {ag: {str(i): gap[ag][i] for i in range(N_ISSUES)} for ag in AGENT_KEYS},
        "ci_threshold": CI_THRESHOLD,
        "n_boot": 1000,
        "issue_names": ISSUE_NAMES,
        "agent_keys": AGENT_KEYS,
    }
    out_path = RESULTS_DIR / "scan.json"
    with open(out_path, "w") as f:
        json.dump(_clean(out_data), f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
