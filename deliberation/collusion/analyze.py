"""
perturbation sweep on existing collusion data.
drift agent: SportCo, Issue D (idx=3) reversed.
conditions: control (25 games), collusion (25 games), single (15 games).
reports cluster-bootstrap CIs (game-level resampling).
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

FEAT_DIR    = Path(__file__).parent / "features"
RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR     = Path(__file__).parent / "figures"

N_ALPHA      = 11
MIN_SAMPLES  = 10
DRIFT_AGENT  = "SportCo"
SCORE_RANGES = {a: 100 for a in AGENT_KEYS}
MOD_ISSUE    = 3
MOD_MODIFIED = [0, 20, 29, 35]   # SportCo Issue D reversed

CONDITIONS  = ["control", "collusion", "single"]
COLORS      = {"control": "#2ca02c", "collusion": "#d62728", "single": "#1f77b4"}


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


def build_adv_scores(scores_base):
    s = copy.deepcopy(scores_base)
    s[DRIFT_AGENT][MOD_ISSUE] = list(MOD_MODIFIED)
    return s


# edges / sweep

def fit_edges(games, scores_base, scores_adv):
    """fit bin edges at alpha=0 from this condition's games"""
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


# cluster bootstrap

def collect_game_triplets_at_endpoints(games, scores_base, scores_adv, edges):
    """precompute per-game pair->triplet lists at alpha=0 and alpha=1"""
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


ATOMS = ["Red", "Un1", "Un2", "Syn"]


def _atoms_mean_from_pool(per_game_subset, alpha):
    """mean of each PID atom across drift pairs from pooled per-game data."""
    pair_triplets = defaultdict(list)
    for gd in per_game_subset:
        for pair, trips in gd[alpha].items():
            pair_triplets[pair].extend(trips)
    drift_atoms = []
    for (j, k), triplets in pair_triplets.items():
        if len(triplets) < MIN_SAMPLES or DRIFT_AGENT not in (j, k):
            continue
        drift_atoms.append(imin_numpy(triplets))
    if not drift_atoms:
        return None
    return {k: float(np.mean([a[k] for a in drift_atoms])) for k in ATOMS}


def compute_atom_bootstrap(per_game, n_boot=1000, rng_seed=42):
    """cluster bootstrap CIs on absolute delta (bits) of each PID atom."""
    rng  = np.random.default_rng(rng_seed)
    n    = len(per_game)
    boot = {k: [] for k in ATOMS}
    for _ in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        subset = [per_game[i] for i in idx]
        m0 = _atoms_mean_from_pool(subset, 0.0)
        m1 = _atoms_mean_from_pool(subset, 1.0)
        if m0 is None or m1 is None:
            continue
        for k in ATOMS:
            boot[k].append(m1[k] - m0[k])
    m0_full = _atoms_mean_from_pool(per_game, 0.0)
    m1_full = _atoms_mean_from_pool(per_game, 1.0)
    result  = {}
    for k in ATOMS:
        vals  = np.array(boot[k])
        delta = m1_full[k] - m0_full[k] if (m0_full and m1_full) else float("nan")
        result[k] = {
            "delta":   delta,
            "a0":      m0_full[k] if m0_full else float("nan"),
            "a1":      m1_full[k] if m1_full else float("nan"),
            "ci_low":  float(np.percentile(vals, 2.5))  if len(vals) >= 10 else float("nan"),
            "ci_high": float(np.percentile(vals, 97.5)) if len(vals) >= 10 else float("nan"),
        }
    return result


def cluster_bootstrap_ratio(per_game, n_boot=1000, rng_seed=42):
    """cluster bootstrap over games, honest CI for ratio, preserves within-game dep."""
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


# metrics

def compute_metrics(traj, per_game):
    alphas  = sorted(traj.keys())
    atoms_0 = traj[alphas[0]]["drift"]
    atoms_1 = traj[alphas[-1]]["drift"]
    s0 = float(np.mean([a["Syn"] for a in atoms_0])) if atoms_0 else float("nan")
    s1 = float(np.mean([a["Syn"] for a in atoms_1])) if atoms_1 else float("nan")
    ratio = s1 / s0 if not np.isnan(s0) and s0 > 1e-9 else float("nan")
    ci_lo, ci_hi = cluster_bootstrap_ratio(per_game)
    return {"syn_a0": s0, "syn_a1": s1, "ratio": ratio,
            "ratio_ci_low": ci_lo, "ratio_ci_high": ci_hi}


# plot

def plot_curves(trajs):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for cond, traj in trajs.items():
        alphas = sorted(traj.keys())
        xs, ms, los, his = [], [], [], []
        for alpha in alphas:
            lst = traj[alpha]["drift"]
            if not lst:
                continue
            m, lo, hi = bootstrap_ci(lst, "Syn", n_boot=N_BOOT)
            xs.append(alpha); ms.append(m); los.append(lo); his.append(hi)
        if xs:
            ax.plot(xs, ms, "o-", color=COLORS[cond], lw=2, ms=4, label=cond)
            ax.fill_between(xs, los, his, color=COLORS[cond], alpha=0.12)
    ax.set_xlabel("alpha  (0=original prefs, 1=Issue D reversed)", fontsize=9)
    ax.set_ylabel("synergy  (bits)", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)
    ax.set_title("Collusion perturbation sweep, SportCo Issue D", fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / "collusion_perturbation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


# main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scores_base, _ = load_scores()
    scores_adv     = build_adv_scores(scores_base)
    alphas         = np.linspace(0.0, 1.0, N_ALPHA)

    print(f"SportCo Issue {MOD_ISSUE}: {scores_base[DRIFT_AGENT][MOD_ISSUE]} -> {MOD_MODIFIED}")

    metrics_all = {}
    trajs       = {}

    for cond in CONDITIONS:
        games = load_games(cond)
        print(f"\n=== {cond}: {len(games)} games ===")
        if not games:
            print("  no games, skip")
            continue
        edges    = fit_edges(games, scores_base, scores_adv)
        traj     = run_sweep(games, scores_base, scores_adv, edges, alphas)
        per_game = collect_game_triplets_at_endpoints(games, scores_base, scores_adv, edges)
        m        = compute_metrics(traj, per_game)
        metrics_all[cond] = m
        trajs[cond]       = traj
        ci_str = f"[{m['ratio_ci_low']:.3f}, {m['ratio_ci_high']:.3f}]"
        excludes = (not np.isnan(m['ratio_ci_high']) and m['ratio_ci_high'] < 1.0)
        overlaps = (not np.isnan(m['ratio_ci_low'])  and m['ratio_ci_low']  < 1.0 <
                    (m['ratio_ci_high'] if not np.isnan(m['ratio_ci_high']) else 2.0))
        flag = "CI excludes 1.0" if excludes else ("CI overlaps 1.0" if overlaps else "")
        print(f"  syn@0={m['syn_a0']:.4f}  syn@1={m['syn_a1']:.4f}  "
              f"ratio={m['ratio']:.3f} {ci_str}  {flag}")

    print("\n=== summary (cluster-bootstrap CIs, n=1000) ===")
    print(f"  {'condition':<12}  {'ratio':>7}  {'CI [2.5%, 97.5%]':>20}  note")
    for cond in CONDITIONS:
        if cond not in metrics_all:
            continue
        m = metrics_all[cond]
        ci_str = f"[{m['ratio_ci_low']:.3f}, {m['ratio_ci_high']:.3f}]"
        excludes_1 = (not np.isnan(m['ratio_ci_high']) and m['ratio_ci_high'] < 1.0)
        overlaps_1 = (not np.isnan(m['ratio_ci_low'])  and m['ratio_ci_low'] < 1.0)
        note = "drops, CI excl 1.0" if excludes_1 else ("flat, CI overlaps 1.0" if overlaps_1 else "")
        print(f"  {cond:<12}  {m['ratio']:>7.3f}  {ci_str:>20}  {note}")

    if trajs:
        plot_curves(trajs)

    # atom decomposition for collusion condition (paired mismatch)
    atom_decomp = None
    if "collusion" in trajs:
        coll_per_game = collect_game_triplets_at_endpoints(
            load_games("collusion"), scores_base, scores_adv,
            fit_edges(load_games("collusion"), scores_base, scores_adv),
        )
        atom_decomp = compute_atom_bootstrap(coll_per_game)
        print("\natom decomposition (collusion):")
        for k, v in atom_decomp.items():
            print(f"  {k}: delta={v['delta']:+.4f}  CI=[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")

    # per-alpha mean-Syn trajectory (same estimator as metrics/bootstrap)
    trajectory = {}
    for cond in CONDITIONS:
        if cond not in trajs:
            continue
        traj = trajs[cond]
        trajectory[cond] = {
            str(alpha): float(np.mean([a["Syn"] for a in traj[float(alpha)]["drift"]]))
            if traj[float(alpha)]["drift"] else None
            for alpha in sorted(traj.keys())
        }

    out_data = {
        "drift_agent": DRIFT_AGENT,
        "mod_issue":   MOD_ISSUE,
        "mod_values":  MOD_MODIFIED,
        "n_boot":      1000,
        "metrics":     {c: metrics_all[c] for c in CONDITIONS if c in metrics_all},
        "trajectory":  trajectory,
        "atom_decomposition": atom_decomp,
    }
    out_path = RESULTS_DIR / "cluster_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
