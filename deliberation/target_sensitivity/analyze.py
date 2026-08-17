"""
target variable sensitivity: tests four target encodings on SportCoxIssue D.
non_pair (baseline), pair, total, drift_only.
source fixed: self_score + payoff_sensitivity. cluster bootstrap n_boot=1000.
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

from extract_features import load_scores, AGENT_KEYS
from pairwise_pid import fit_quantile_bins, apply_bins, N_BINS, N_TARGET
from pid_numpy import imin_numpy
from synthetic_drift import compute_states_at_alpha

FEAT_DIR    = Path(__file__).parent.parent / "collusion" / "features"
RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR     = Path(__file__).parent / "figures"

DRIFT_AGENT  = "SportCo"
MOD_ISSUE    = 3
MOD_MODIFIED = [0, 20, 29, 35]
SCORE_RANGES = {a: 100 for a in AGENT_KEYS}
CONDITIONS   = ["control", "collusion"]
MIN_SAMPLES  = 10

TARGET_CONFIGS = {
    "non_pair":   lambda s, sn, pair: sum(sn["scores"][a] - s["scores"][a]
                                          for a in AGENT_KEYS if a not in pair),
    "pair":       lambda s, sn, pair: sum(sn["scores"][a] - s["scores"][a]
                                          for a in pair),
    "total":      lambda s, sn, pair: sum(sn["scores"][a] - s["scores"][a]
                                          for a in AGENT_KEYS),
    "drift_only": lambda s, sn, pair: sn["scores"][DRIFT_AGENT] - s["scores"][DRIFT_AGENT],
}


# data

def load_games(condition):
    games = []
    for p in sorted(FEAT_DIR.glob(f"{condition}_*.json")):
        g = json.load(open(p))
        if g["n_rounds_parsed"] >= 10:
            games.append(g)
    return games


def build_adv_scores(scores_base):
    s = copy.deepcopy(scores_base)
    s[DRIFT_AGENT][MOD_ISSUE] = list(MOD_MODIFIED)
    return s


# edges

def fit_edges(games, scores_base, scores_adv, delta_fn):
    """fit bin edges for self_sens source + given delta_fn at alpha=0"""
    all_states, all_deltas = [], []
    for g in games:
        ss = compute_states_at_alpha(g, scores_base, scores_adv, 0.0,
                                     AGENT_KEYS, DRIFT_AGENT, SCORE_RANGES)
        all_states.extend(ss)
        for idx in range(len(ss) - 1):
            s, sn = ss[idx], ss[idx + 1]
            for pair in combinations(AGENT_KEYS, 2):
                all_deltas.append(delta_fn(s, sn, pair))
    return {
        "self": {a: fit_quantile_bins([s["self_scores"][a] for s in all_states])
                 for a in AGENT_KEYS},
        "sens": {a: fit_quantile_bins([s["sensitivity"][a] for s in all_states])
                 for a in AGENT_KEYS},
        "delta": fit_quantile_bins(all_deltas, n_bins=N_TARGET),
    }


# triplet collection

def collect_game_triplets(games, scores_base, scores_adv, edges, delta_fn):
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
                                         AGENT_KEYS, DRIFT_AGENT, SCORE_RANGES)
            pt = defaultdict(list)
            for idx in range(len(ss) - 1):
                s, sn = ss[idx], ss[idx + 1]
                for pair in combinations(AGENT_KEYS, 2):
                    delta = delta_fn(s, sn, pair)
                    pt[pair].append((jbin(s, pair[0]), jbin(s, pair[1]),
                                     apply_bins(delta, delta_e)))
            gdata[alpha] = dict(pt)
        per_game.append(gdata)
    return per_game


# metrics

def _syn_from_pool(per_game_subset, alpha):
    pair_triplets = defaultdict(list)
    for gd in per_game_subset:
        for pair, trips in gd[alpha].items():
            pair_triplets[pair].extend(trips)
    drift_atoms = []
    for pair, triplets in pair_triplets.items():
        if len(triplets) < MIN_SAMPLES or DRIFT_AGENT not in pair:
            continue
        drift_atoms.append(imin_numpy(triplets))
    return float(np.mean([a["Syn"] for a in drift_atoms])) if drift_atoms else float("nan")


def cluster_bootstrap_ratio(per_game, n_boot=1000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n   = len(per_game)
    boot_ratios = []
    for _ in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        subset = [per_game[i] for i in idx]
        s0 = _syn_from_pool(subset, 0.0)
        s1 = _syn_from_pool(subset, 1.0)
        if not np.isnan(s0) and s0 > 1e-9:
            boot_ratios.append(s1 / s0)
    if len(boot_ratios) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(boot_ratios, 2.5)), float(np.percentile(boot_ratios, 97.5))


# plot

def plot_comparison(results):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configs = list(TARGET_CONFIGS.keys())
    x       = np.arange(len(configs))
    width   = 0.35
    colors  = {"control": "#2ca02c", "collusion": "#d62728"}

    fig, ax = plt.subplots(figsize=(8, 4))
    for ci, cond in enumerate(CONDITIONS):
        ratios = [results[cfg][cond]["ratio"]   for cfg in configs]
        los    = [results[cfg][cond]["ci_low"]  for cfg in configs]
        his    = [results[cfg][cond]["ci_high"] for cfg in configs]
        errs   = [[r - lo for r, lo in zip(ratios, los)],
                  [hi - r  for r, hi in zip(ratios, his)]]
        errs        = [[0 if np.isnan(e) else e for e in row] for row in errs]
        ratios_plot = [0 if np.isnan(r) else r for r in ratios]
        ax.bar(x + ci * width, ratios_plot, width, label=cond,
               color=colors[cond], alpha=0.85,
               yerr=errs, capsize=4, error_kw={"linewidth": 1.2})

    ax.axhline(1.0, color="k",    linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.8, color="gray", linewidth=0.8, linestyle=":",  alpha=0.5, label="0.80 threshold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel("synergy ratio  (syn@1 / syn@0)", fontsize=9)
    ax.set_ylim(0, 1.4)
    ax.set_title("Target variable sensitivity, SportCo x Issue D", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "target_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


# main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scores_base, _ = load_scores()
    scores_adv     = build_adv_scores(scores_base)
    print(f"SportCo Issue {MOD_ISSUE}: {scores_base[DRIFT_AGENT][MOD_ISSUE]} -> {MOD_MODIFIED}")

    games_by_cond = {c: load_games(c) for c in CONDITIONS}
    for c, gs in games_by_cond.items():
        print(f"{c}: {len(gs)} games")

    results = {}

    for cfg_name, delta_fn in TARGET_CONFIGS.items():
        print(f"\n=== {cfg_name} ===")
        results[cfg_name] = {}

        for condition in CONDITIONS:
            games    = games_by_cond[condition]
            edges    = fit_edges(games, scores_base, scores_adv, delta_fn)
            per_game = collect_game_triplets(games, scores_base, scores_adv,
                                             edges, delta_fn)
            s0 = _syn_from_pool(per_game, 0.0)
            s1 = _syn_from_pool(per_game, 1.0)
            ratio = s1 / s0 if not np.isnan(s0) and s0 > 1e-9 else float("nan")
            lo, hi = cluster_bootstrap_ratio(per_game)
            results[cfg_name][condition] = {
                "score_a0": s0, "score_a1": s1, "ratio": ratio,
                "ci_low": lo, "ci_high": hi,
            }
            excl = not np.isnan(hi) and hi < 1.0
            print(f"  {condition}: score@0={s0:.4f}  score@1={s1:.4f}  "
                  f"ratio={ratio:.3f}  [{lo:.3f}, {hi:.3f}]"
                  f"{'  CI excl 1.0' if excl else ''}")

    # summary table
    print("\n=== summary ===")
    print(f"  {'config':<12}  {'ctrl ratio':>10}  {'ctrl CI':>20}  "
          f"{'coll ratio':>10}  {'coll CI':>20}  {'gap':>7}  pass")
    for cfg in TARGET_CONFIGS:
        ctrl = results[cfg]["control"]
        coll = results[cfg]["collusion"]
        gap  = coll["ratio"] - ctrl["ratio"]
        passed = (not np.isnan(ctrl["ratio"]) and ctrl["ratio"] < 0.80
                  and not np.isnan(coll["ratio"]) and coll["ratio"] > 0.80
                  and gap >= 0.20)
        ctrl_ci = f"[{ctrl['ci_low']:.3f}, {ctrl['ci_high']:.3f}]"
        coll_ci = f"[{coll['ci_low']:.3f}, {coll['ci_high']:.3f}]"
        print(f"  {cfg:<12}  {ctrl['ratio']:>10.3f}  {ctrl_ci:>20}  "
              f"{coll['ratio']:>10.3f}  {coll_ci:>20}  {gap:>7.3f}  {passed}")

    plot_comparison(results)

    def _clean(obj):
        if isinstance(obj, float) and np.isnan(obj): return None
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        return obj

    out_data = {
        "drift_agent": DRIFT_AGENT,
        "mod_issue": MOD_ISSUE,
        "mod_values": MOD_MODIFIED,
        "n_boot": 1000,
        "target_configs": list(TARGET_CONFIGS.keys()),
        "results": results,
    }
    out_path = RESULTS_DIR / "target_configs.json"
    with open(out_path, "w") as f:
        json.dump(_clean(out_data), f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
