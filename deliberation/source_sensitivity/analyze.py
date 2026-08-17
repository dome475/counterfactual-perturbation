"""
source feature sensitivity: tests four source encodings on SportCoxIssue D.
self_sens (baseline), self_only (MI), self_rank, self_dist.
cluster bootstrap n_boot=1000 for all configs.
"""
import copy
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "synthetic_drift"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extract_features import load_scores, AGENT_KEYS, ISSUE_SIZES
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
MAX_DIST     = sum(n - 1 for n in ISSUE_SIZES)   # max L1 deal distance

SOURCE_CONFIGS = {
    "self_sens": ("self_score", "payoff_sensitivity"),
    "self_only": ("self_score", None),
    "self_rank": ("self_score", "score_rank"),
    "self_dist": ("self_score", "deal_distance"),
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


# state augmentation

def augment_states(states, game):
    """add score_rank and deal_distance to each state in-place"""
    ideal_opts = game["ideal_opts"]
    rounds     = game["rounds"]
    for s in states:
        t    = s["t"]
        deal = rounds[t]["deal"]
        # score_rank: normalized rank (1/n_agents = best)
        sorted_agents = sorted(AGENT_KEYS, key=lambda a: s["scores"][a], reverse=True)
        s["score_rank"] = {a: (sorted_agents.index(a) + 1) / len(AGENT_KEYS)
                           for a in AGENT_KEYS}
        # deal_distance: normalized L1 to reference ideal
        s["deal_distance"] = {
            a: sum(abs(deal[i] - ideal_opts[a][i]) for i in range(len(deal))) / MAX_DIST
            for a in AGENT_KEYS
        }
    return states


def get_feature(s, feat, agent):
    if feat == "self_score":        return s["self_scores"][agent]
    if feat == "payoff_sensitivity": return s["sensitivity"][agent]
    if feat == "score_rank":        return s["score_rank"][agent]
    if feat == "deal_distance":     return s["deal_distance"][agent]  # L1 option-index dist from reference ideal, not score-space
    raise ValueError(feat)


# edges

def fit_edges(games, scores_base, scores_adv, feat1, feat2):
    """fit bin edges for (feat1, feat2) at alpha=0; returns None if f2 bins are degenerate"""
    all_states, all_deltas = [], []
    for g in games:
        ss = compute_states_at_alpha(g, scores_base, scores_adv, 0.0,
                                     AGENT_KEYS, DRIFT_AGENT, SCORE_RANGES)
        augment_states(ss, g)
        all_states.extend(ss)
        for idx in range(len(ss) - 1):
            s, sn = ss[idx], ss[idx + 1]
            for (j, k) in combinations(AGENT_KEYS, 2):
                noncoal = [a for a in AGENT_KEYS if a not in (j, k)]
                all_deltas.append(sum(sn["scores"][a] - s["scores"][a] for a in noncoal))
    edges = {
        "f1":    {a: fit_quantile_bins([get_feature(s, feat1, a) for s in all_states])
                  for a in AGENT_KEYS},
        "delta": fit_quantile_bins(all_deltas, n_bins=N_TARGET),
    }
    if feat2 is not None:
        f2_edges = {a: fit_quantile_bins([get_feature(s, feat2, a) for s in all_states])
                    for a in AGENT_KEYS}
        # degeneracy check: score_rank has only 6 distinct values; warn if bins collapse
        n_unique = len(set(apply_bins(get_feature(s, feat2, DRIFT_AGENT), f2_edges[DRIFT_AGENT])
                           for s in all_states))
        if n_unique < 2:
            print(f"  WARNING: {feat2} bins degenerate ({n_unique} unique bin), skipping config")
            return None
        edges["f2"] = f2_edges
    return edges


# triplet collection

def collect_game_triplets(games, scores_base, scores_adv, edges, feat1, feat2):
    """collect per-game (bin_x1[, bin_x2], bin_y) at alpha=0 and alpha=1"""
    f1_e    = edges["f1"]
    f2_e    = edges.get("f2")
    delta_e = edges["delta"]

    def xbin(s, a):
        b1 = apply_bins(get_feature(s, feat1, a), f1_e[a])
        if feat2 is None:
            return b1
        b2 = apply_bins(get_feature(s, feat2, a), f2_e[a])
        return b1 * N_BINS + b2

    per_game = []
    for g in games:
        gdata = {}
        for alpha in [0.0, 1.0]:
            ss = compute_states_at_alpha(g, scores_base, scores_adv, alpha,
                                         AGENT_KEYS, DRIFT_AGENT, SCORE_RANGES)
            augment_states(ss, g)
            pt = defaultdict(list)
            for idx in range(len(ss) - 1):
                s, sn = ss[idx], ss[idx + 1]
                for (j, k) in combinations(AGENT_KEYS, 2):
                    noncoal = [a for a in AGENT_KEYS if a not in (j, k)]
                    delta = sum(sn["scores"][a] - s["scores"][a] for a in noncoal)
                    pt[(j, k)].append((xbin(s, j), xbin(s, k),
                                       apply_bins(delta, delta_e)))
            gdata[alpha] = dict(pt)
        per_game.append(gdata)
    return per_game


# metrics

def mutual_information(pairs):
    """I(X;Y) in bits from (bin_x, bin_y) pairs"""
    n = len(pairs)
    if n == 0:
        return 0.0
    joint = Counter(pairs)
    px    = Counter(x for x, y in pairs)
    py    = Counter(y for x, y in pairs)
    mi    = 0.0
    for (x, y), cnt in joint.items():
        pxy = cnt / n
        mi += pxy * np.log2(pxy / (px[x] / n * py[y] / n))
    return float(mi)


def _score_from_pool(per_game_subset, alpha, feat2):
    """pool triplets from resampled games and compute syn (2D) or MI (1D)"""
    pair_triplets = defaultdict(list)
    for gd in per_game_subset:
        for pair, trips in gd[alpha].items():
            pair_triplets[pair].extend(trips)
    scores = []
    for (j, k), triplets in pair_triplets.items():
        if len(triplets) < MIN_SAMPLES or DRIFT_AGENT not in (j, k):
            continue
        if feat2 is None:
            # 1D: computes I(SportCo_feature; delta), single-agent MI, not a pair measure.
            # this answers "does PID decomposition matter?" not "what is pair synergy?"
            da = DRIFT_AGENT
            pairs = [(t[0], t[2]) for t in triplets] if j == da else [(t[1], t[2]) for t in triplets]
            scores.append(mutual_information(pairs))
        else:
            scores.append(imin_numpy(triplets)["Syn"])
    return float(np.mean(scores)) if scores else float("nan")


def cluster_bootstrap_ratio(per_game, feat2, n_boot=1000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n   = len(per_game)
    boot_ratios = []
    for _ in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        subset = [per_game[i] for i in idx]
        s0 = _score_from_pool(subset, 0.0, feat2)
        s1 = _score_from_pool(subset, 1.0, feat2)
        if not np.isnan(s0) and s0 > 1e-9:
            boot_ratios.append(s1 / s0)
    if len(boot_ratios) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(boot_ratios, 2.5)), float(np.percentile(boot_ratios, 97.5))


# plot

def plot_comparison(results):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configs = list(SOURCE_CONFIGS.keys())
    x       = np.arange(len(configs))
    width   = 0.35
    colors  = {"control": "#2ca02c", "collusion": "#d62728"}

    fig, ax = plt.subplots(figsize=(8, 4))
    for ci, cond in enumerate(CONDITIONS):
        ratios = [results[cfg][cond]["ratio"]     for cfg in configs]
        los    = [results[cfg][cond]["ci_low"]    for cfg in configs]
        his    = [results[cfg][cond]["ci_high"]   for cfg in configs]
        errs   = [[r - lo for r, lo in zip(ratios, los)],
                  [hi - r  for r, hi in zip(ratios, his)]]
        # replace nan errors with 0 for plotting
        errs = [[0 if np.isnan(e) else e for e in row] for row in errs]
        ratios_plot = [0 if np.isnan(r) else r for r in ratios]
        ax.bar(x + ci * width, ratios_plot, width, label=cond,
               color=colors[cond], alpha=0.85,
               yerr=errs, capsize=4, error_kw={"linewidth": 1.2})

    ax.axhline(1.0, color="k",  linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.8, color="gray", linewidth=0.8, linestyle=":",  alpha=0.5, label="0.80 threshold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel("synergy ratio  (syn@1 / syn@0)", fontsize=9)
    ax.set_ylim(0, 1.4)
    ax.set_title("Source feature sensitivity, SportCo x Issue D", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "source_comparison.png"
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

    for cfg_name, (feat1, feat2) in SOURCE_CONFIGS.items():
        print(f"\n=== {cfg_name}  ({feat1}, {feat2}) ===")
        results[cfg_name] = {}

        for condition in CONDITIONS:
            games  = games_by_cond[condition]
            edges  = fit_edges(games, scores_base, scores_adv, feat1, feat2)
            if edges is None:
                results[cfg_name][condition] = {"score_a0": None, "score_a1": None,
                                                "ratio": None, "ci_low": None, "ci_high": None}
                continue
            per_game = collect_game_triplets(games, scores_base, scores_adv,
                                             edges, feat1, feat2)
            s0 = _score_from_pool(per_game, 0.0, feat2)
            s1 = _score_from_pool(per_game, 1.0, feat2)
            ratio = s1 / s0 if not np.isnan(s0) and s0 > 1e-9 else float("nan")
            lo, hi = cluster_bootstrap_ratio(per_game, feat2)
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
    for cfg in SOURCE_CONFIGS:
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
        "source_configs": {k: list(v) for k, v in SOURCE_CONFIGS.items()},
        "results": results,
    }
    out_path = RESULTS_DIR / "source_configs.json"
    with open(out_path, "w") as f:
        json.dump(_clean(out_data), f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
