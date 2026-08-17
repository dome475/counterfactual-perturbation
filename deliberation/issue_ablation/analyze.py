"""
per-issue ablation on existing collusion data.

for each of 5 issues, perturb only SportCo's score sheet for that issue
and run the alpha sweep on both conditions (control, collusion).
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
from pid_numpy import imin_numpy, bootstrap_ci, N_BOOT
from synthetic_drift import compute_states_at_alpha

FEAT_DIR    = Path(__file__).parent.parent / "collusion" / "features"
RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR     = Path(__file__).parent / "figures"

N_ALPHA     = 11
MIN_SAMPLES = 10
DRIFT_AGENT = "SportCo"
SCORE_RANGES = {a: 100 for a in AGENT_KEYS}

ISSUE_NAMES   = ["A_Infrastructure", "B_Eco", "C_Employment", "D_FederalLoan", "E_Compensation"]
CONDITIONS    = ["control", "collusion"]
COND_LABELS   = {"control": "control", "collusion": "mismatch"}


def load_games(condition):
    games = []
    for p in sorted(FEAT_DIR.glob(f"{condition}_*.json")):
        g = json.load(open(p))
        if g["n_rounds_parsed"] >= 10:
            games.append(g)
        else:
            print(f"  skip {g['game_id']}: {g['n_rounds_parsed']} rounds")
    return games


def build_adv_scores(scores_base, issue_idx):
    """reverse SportCo's options for issue_idx only"""
    s = copy.deepcopy(scores_base)
    s[DRIFT_AGENT][issue_idx] = list(reversed(scores_base[DRIFT_AGENT][issue_idx]))
    return s


def fit_edges(games, scores_base, scores_adv):
    """fit bin edges at alpha=0 from this cell's games"""
    all_states, all_deltas = [], []
    for g in games:
        ss = compute_states_at_alpha(g, scores_base, scores_adv, 0.0,
                                     AGENT_KEYS, DRIFT_AGENT, SCORE_RANGES)
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


def collect_game_triplets_at_endpoints(games, scores_base, scores_adv, edges):
    """
    precompute per-game pair->triplet lists at alpha=0 and alpha=1 only.
    returns list[{alpha: {(j,k): [triplets]}}], one dict per game.
    used by cluster bootstrap so compute_states_at_alpha is called once
    per game, not once per bootstrap iteration.
    """
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
                for (j, k) in combinations(AGENT_KEYS, 2):
                    noncoal = [a for a in AGENT_KEYS if a not in (j, k)]
                    delta = sum(sn["scores"][a] - s["scores"][a] for a in noncoal)
                    pt[(j, k)].append((jbin(s, j), jbin(s, k), apply_bins(delta, delta_e)))
            gdata[alpha] = dict(pt)
        per_game.append(gdata)
    return per_game


def _syn_from_pool(per_game_subset, alpha):
    """aggregate triplets from resampled games, return mean drift-pair Syn"""
    pair_triplets = defaultdict(list)
    for gd in per_game_subset:
        for pair, trips in gd[alpha].items():
            pair_triplets[pair].extend(trips)
    drift_atoms = []
    for (j, k), triplets in pair_triplets.items():
        if len(triplets) < MIN_SAMPLES or DRIFT_AGENT not in (j, k):
            continue
        drift_atoms.append(imin_numpy(triplets))
    return float(np.mean([a["Syn"] for a in drift_atoms])) if drift_atoms else float("nan")


def cluster_bootstrap_ratio(per_game, n_boot=1000, rng_seed=42):
    """
    cluster bootstrap over games (not pairs).
    pairs within a game share the same trajectory, so resampling at the
    game level avoids an anti-conservative pair-level bootstrap.
    """
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


def run_sweep(games, scores_base, scores_adv, edges, alphas):
    self_e  = edges["self"]
    sens_e  = edges["sens"]
    delta_e = edges["delta"]

    def jbin(s, a):
        return apply_bins(s["self_scores"][a], self_e[a]) * N_BINS + \
               apply_bins(s["sensitivity"][a], sens_e[a])

    result = {}
    for alpha in alphas:
        pair_triplets = defaultdict(list)
        for g in games:
            ss = compute_states_at_alpha(g, scores_base, scores_adv, alpha,
                                         AGENT_KEYS, DRIFT_AGENT, SCORE_RANGES)
            for idx in range(len(ss) - 1):
                s, sn = ss[idx], ss[idx + 1]
                for (j, k) in combinations(AGENT_KEYS, 2):
                    noncoal = [a for a in AGENT_KEYS if a not in (j, k)]
                    delta = sum(sn["scores"][a] - s["scores"][a] for a in noncoal)
                    pair_triplets[(j, k)].append((jbin(s, j), jbin(s, k),
                                                  apply_bins(delta, delta_e)))
        drift_atoms, ctrl_atoms = [], []
        for (j, k), triplets in pair_triplets.items():
            if len(triplets) < MIN_SAMPLES:
                continue
            atoms = imin_numpy(triplets)
            target = drift_atoms if DRIFT_AGENT in (j, k) else ctrl_atoms
            target.append({**atoms, "pair": f"{j}|{k}", "n": len(triplets)})
        result[float(alpha)] = {"drift": drift_atoms, "ctrl": ctrl_atoms}
    return result


def compute_metrics(traj, per_game):
    """point estimates from traj; cluster-bootstrap CIs from per_game data"""
    alphas  = sorted(traj.keys())
    atoms_0 = traj[alphas[0]]["drift"]
    atoms_1 = traj[alphas[-1]]["drift"]
    s0 = float(np.mean([a["Syn"] for a in atoms_0])) if atoms_0 else float("nan")
    s1 = float(np.mean([a["Syn"] for a in atoms_1])) if atoms_1 else float("nan")
    ratio = s1 / s0 if not np.isnan(s0) and s0 > 1e-9 else float("nan")
    ci_lo, ci_hi = cluster_bootstrap_ratio(per_game)
    return {"syn_a0": s0, "syn_a1": s1, "ratio": ratio,
            "ratio_ci_low": ci_lo, "ratio_ci_high": ci_hi}


def plot_heatmap(grid_ratios, grid_metrics):
    """5x2 heatmap: rows=condition, cols=issue"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.5))

    data = np.array([[grid_ratios[cond][i] for i in range(5)]
                     for cond in CONDITIONS])

    # diverging colormap centered at 1.0
    vmin, vmax = 0.4, 1.2
    im = ax.imshow(data, cmap="RdBu", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(5))
    ax.set_xticklabels([n.split("_")[0] for n in ISSUE_NAMES], fontsize=10)
    ax.set_yticks(range(2))
    ax.set_yticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=10)

    for r, cond in enumerate(CONDITIONS):
        for c in range(5):
            m = grid_metrics[cond][c]
            v  = m["ratio"]
            lo = m["ratio_ci_low"]
            hi = m["ratio_ci_high"]
            txt_color = "white" if v < 0.65 or v > 1.10 else "black"
            ratio_str = f"{v:.3f}" if not np.isnan(v) else "nan"
            ci_str    = (f"[{lo:.2f},{hi:.2f}]" if not (np.isnan(lo) or np.isnan(hi))
                         else "[?,?]")
            ax.text(c, r - 0.12, ratio_str, ha="center", va="center",
                    fontsize=9, color=txt_color, fontweight="bold")
            ax.text(c, r + 0.22, ci_str, ha="center", va="center",
                    fontsize=6.5, color=txt_color)

    plt.colorbar(im, ax=ax, label="synergy ratio  (syn@1 / syn@0)", shrink=0.8)
    ax.set_title("Per-issue ablation, SportCo  (red=drop, blue=flat/rise)", fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / "issue_ablation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def plot_curves(all_trajs):
    """2x5 grid of alpha-sweep curves"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=False)

    colors = {"control": "#2ca02c", "collusion": "#d62728"}

    for r, cond in enumerate(CONDITIONS):
        for c in range(5):
            ax = axes[r, c]
            traj = all_trajs[cond][c]
            alphas = sorted(traj.keys())
            xs, ms, los, his = [], [], [], []
            for alpha in alphas:
                lst = traj[alpha]["drift"]
                if not lst:
                    continue
                m, lo, hi = bootstrap_ci(lst, "Syn", n_boot=N_BOOT)
                xs.append(alpha); ms.append(m); los.append(lo); his.append(hi)
            if xs:
                ax.plot(xs, ms, "o-", color=colors[cond], lw=1.5, ms=3)
                ax.fill_between(xs, los, his, color=colors[cond], alpha=0.15)
            ax.set_ylim(bottom=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if r == 0:
                ax.set_title(ISSUE_NAMES[c].split("_")[0], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{COND_LABELS[cond]}\nsynergy (bits)", fontsize=8)
            if r == 1:
                ax.set_xlabel("alpha", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("Per-issue alpha-sweep  (SportCo drift agent)", fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / "issue_ablation_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scores_base, _ = load_scores()
    alphas = np.linspace(0.0, 1.0, N_ALPHA)

    print("SportCo original scores:")
    for i, row in enumerate(scores_base[DRIFT_AGENT]):
        rev = list(reversed(row))
        print(f"  issue {i} ({ISSUE_NAMES[i]}): {row}  -> reversed: {rev}")

    games_by_cond = {}
    for cond in CONDITIONS:
        games = load_games(cond)
        games_by_cond[cond] = games
        print(f"loaded {len(games)} {cond} games")

    grid_metrics = {cond: {} for cond in CONDITIONS}
    all_trajs    = {cond: {} for cond in CONDITIONS}

    for cond in CONDITIONS:
        games = games_by_cond[cond]
        for issue_idx in range(5):
            label = ISSUE_NAMES[issue_idx]
            print(f"\n=== {cond} / {label} ===")
            scores_adv = build_adv_scores(scores_base, issue_idx)
            edges      = fit_edges(games, scores_base, scores_adv)
            traj       = run_sweep(games, scores_base, scores_adv, edges, alphas)
            per_game   = collect_game_triplets_at_endpoints(games, scores_base, scores_adv, edges)
            m          = compute_metrics(traj, per_game)
            grid_metrics[cond][issue_idx] = m
            all_trajs[cond][issue_idx]    = traj
            print(f"  syn@0={m['syn_a0']:.4f}  syn@1={m['syn_a1']:.4f}  "
                  f"ratio={m['ratio']:.3f} [{m['ratio_ci_low']:.3f},{m['ratio_ci_high']:.3f}]")

    # summary table
    print("\n=== 5x2 ratio grid (ratio [ci_lo, ci_hi]) ===")
    print(f"{'issue':<20}  {'control':>22}  {'collusion':>22}  {'gap':>7}")
    for i in range(5):
        mc   = grid_metrics["control"][i]
        mm   = grid_metrics["collusion"][i]
        gap  = mm["ratio"] - mc["ratio"]
        ctrl_str = f"{mc['ratio']:.3f} [{mc['ratio_ci_low']:.3f},{mc['ratio_ci_high']:.3f}]"
        coll_str = f"{mm['ratio']:.3f} [{mm['ratio_ci_low']:.3f},{mm['ratio_ci_high']:.3f}]"
        print(f"  {ISSUE_NAMES[i]:<18}  {ctrl_str:>22}  {coll_str:>22}  {gap:>7.3f}")

    # plot
    grid_ratios = {cond: [grid_metrics[cond][i]["ratio"] for i in range(5)]
                   for cond in CONDITIONS}
    plot_heatmap(grid_ratios, grid_metrics)
    plot_curves(all_trajs)

    # save results
    out_data = {
        "drift_agent":  DRIFT_AGENT,
        "issue_names":  ISSUE_NAMES,
        "conditions":   CONDITIONS,
        "n_games":      {cond: len(games_by_cond[cond]) for cond in CONDITIONS},
        "grid_metrics": {
            cond: {str(i): grid_metrics[cond][i] for i in range(5)}
            for cond in CONDITIONS
        },
        "trajectory": {
            cond: {
                str(i): {
                    str(alpha): {
                        "drift": [{k: v for k, v in a.items() if k not in ("pair", "n")}
                                  for a in all_trajs[cond][i][float(alpha)]["drift"]],
                    }
                    for alpha in sorted(all_trajs[cond][i].keys())
                }
                for i in range(5)
            }
            for cond in CONDITIONS
        },
    }
    out_path = RESULTS_DIR / "issue_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
