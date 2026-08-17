"""
binning sensitivity: rerun the drift sweep at n_bins in {2,3,4,5}
(target bins fixed at 3, edges refit per resolution at alpha=0).
checks the synergy drop is not a discretization artifact; n_bins=4
must reproduce the baseline run.
"""

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

from extract_features import (
    load_scores, compute_ideal,
    AGENT_KEYS, ISSUE_SIZES,
)
from pairwise_pid import load_condition, fit_quantile_bins, apply_bins, N_TARGET
from pid_numpy import imin_numpy, bootstrap_ci, N_BOOT
from synthetic_drift import compute_states_at_alpha, invert_scores

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR     = Path(__file__).parent / "figures"

CONDITION   = "cooperative"
DRIFT_AGENT = "SportCo"
N_BINS_LIST = [2, 3, 4, 5]
N_ALPHA     = 11
MIN_SAMPLES = 5

# baseline reference for internal consistency check
F1_SYN_A0    = 0.3766
F1_DROP_RATIO = 0.4178


def fit_edges(n_source_bins, baseline_states, baseline_deltas):
    """
    fit source and delta edges for a given source bin count.
    source edges: n_source_bins quantile bins (re-fitted, not shared across runs).
    delta edges: always N_TARGET=3, not varied.
    """
    self_edges  = {a: fit_quantile_bins([s["self_scores"][a] for s in baseline_states],
                                         n_bins=n_source_bins)
                   for a in AGENT_KEYS}
    sens_edges  = {a: fit_quantile_bins([s["sensitivity"][a] for s in baseline_states],
                                         n_bins=n_source_bins)
                   for a in AGENT_KEYS}
    delta_edges = fit_quantile_bins(baseline_deltas, n_bins=N_TARGET)
    return {"self": self_edges, "sens": sens_edges, "delta": delta_edges}


def joint_bin(s, agent, n_source_bins, self_edges, sens_edges):
    """2D source bin: (self_score_bin * n_source_bins + sensitivity_bin)"""
    b1 = apply_bins(s["self_scores"][agent], self_edges[agent])
    b2 = apply_bins(s["sensitivity"][agent], sens_edges[agent])
    return b1 * n_source_bins + b2


def run_sweep(n_source_bins, scores_base, scores_adv, score_ranges):
    """
    full alpha sweep for one source bin count.
    edges fitted once from cooperative games at alpha=0.
    returns {alpha: {"drift": [atom_dict,...], "ctrl": [atom_dict,...]}}
    """
    games  = load_condition(CONDITION)
    alphas = np.linspace(0.0, 1.0, N_ALPHA)

    # fit edges once at alpha=0; fixed across entire sweep
    print(f"  n_source_bins={n_source_bins}: fitting edges from {len(games)} games at alpha=0...")
    baseline_states, baseline_game_states = [], []
    for g in games:
        ss = compute_states_at_alpha(g, scores_base, scores_adv, 0.0,
                                     AGENT_KEYS, DRIFT_AGENT, score_ranges)
        baseline_game_states.append(ss)
        baseline_states.extend(ss)

    baseline_deltas = []
    for ss in baseline_game_states:
        for idx in range(len(ss) - 1):
            s, s_next = ss[idx], ss[idx + 1]
            for (j, k) in combinations(AGENT_KEYS, 2):
                noncoal = [a for a in AGENT_KEYS if a not in (j, k)]
                baseline_deltas.append(
                    sum(s_next["scores"][a] - s["scores"][a] for a in noncoal)
                )

    edges = fit_edges(n_source_bins, baseline_states, baseline_deltas)
    self_edges  = edges["self"]
    sens_edges  = edges["sens"]
    delta_edges = edges["delta"]

    n_joint_states = n_source_bins ** 2
    print(f"    joint states per source: {n_joint_states}  "
          f"(full table: {n_joint_states}x{n_joint_states}x{N_TARGET} = "
          f"{n_joint_states**2 * N_TARGET} cells)")

    result = {}
    for alpha in alphas:
        game_states = [
            compute_states_at_alpha(g, scores_base, scores_adv, alpha,
                                    AGENT_KEYS, DRIFT_AGENT, score_ranges)
            for g in games
        ]

        pair_triplets = defaultdict(list)
        for ss in game_states:
            for idx in range(len(ss) - 1):
                s, s_next = ss[idx], ss[idx + 1]
                for (j, k) in combinations(AGENT_KEYS, 2):
                    noncoal = [a for a in AGENT_KEYS if a not in (j, k)]
                    delta = sum(s_next["scores"][a] - s["scores"][a] for a in noncoal)
                    pair_triplets[(j, k)].append((
                        joint_bin(s, j, n_source_bins, self_edges, sens_edges),
                        joint_bin(s, k, n_source_bins, self_edges, sens_edges),
                        apply_bins(delta, delta_edges),
                    ))

        drift_atoms, ctrl_atoms = [], []
        for (j, k), triplets in pair_triplets.items():
            if len(triplets) < MIN_SAMPLES:
                continue
            atoms = imin_numpy(triplets)
            (drift_atoms if DRIFT_AGENT in (j, k) else ctrl_atoms).append(atoms)

        # sparsity note at N_BINS=5: ~600 triplets / 625 joint cells, expected
        if n_source_bins >= 5 and alpha == 0.0:
            for (j, k), triplets in pair_triplets.items():
                n_unique = len(set(triplets))
                coverage = n_unique / (n_joint_states ** 2 * N_TARGET)
                print(f"    sparsity {j}|{k}: {len(triplets)} triplets, "
                      f"{n_unique} unique cells / {n_joint_states**2 * N_TARGET} "
                      f"({100*coverage:.1f}% coverage)")

        result[float(alpha)] = {"drift": drift_atoms, "ctrl": ctrl_atoms}

    return result


def summarize(traj, n_source_bins):
    alphas = sorted(traj.keys())

    drift_means = []
    for alpha in alphas:
        lst = traj[alpha]["drift"]
        drift_means.append(np.mean([a["Syn"] for a in lst]) if lst else float("nan"))

    s0 = drift_means[0]
    s1 = drift_means[-1]
    drop_ratio = s1 / s0 if (not np.isnan(s0) and s0 > 1e-9) else float("nan")

    ctrl_means = []
    for alpha in alphas:
        lst = traj[alpha]["ctrl"]
        ctrl_means.append(np.mean([a["Syn"] for a in lst]) if lst else float("nan"))

    return {
        "n_source_bins":  n_source_bins,
        "syn_a0":         float(s0),
        "syn_a1":         float(s1),
        "drop_ratio":     float(drop_ratio),
        "ctrl_syn_a0":    float(ctrl_means[0]) if ctrl_means else float("nan"),
        "drift_curve":    [float(v) for v in drift_means],
        "alphas":         [float(a) for a in alphas],
    }


def check_consistency(summary):
    """flag if n_bins=4 does not reproduce the baseline"""
    if summary["n_source_bins"] != 4:
        return
    tol = 0.005
    ok_s0    = abs(summary["syn_a0"]    - F1_SYN_A0)    < tol
    ok_drop  = abs(summary["drop_ratio"] - F1_DROP_RATIO) < tol
    if ok_s0 and ok_drop:
        print(f"  ok N_BINS=4 consistency check PASSED  "
              f"(syn@0={summary['syn_a0']:.4f}, drop={summary['drop_ratio']:.4f})")
    else:
        print(f"  N_BINS=4 consistency check FAILED:")
        if not ok_s0:
            print(f"    syn@0:      got {summary['syn_a0']:.4f}, expected ~{F1_SYN_A0}")
        if not ok_drop:
            print(f"    drop_ratio: got {summary['drop_ratio']:.4f}, expected ~{F1_DROP_RATIO}")


def plot(all_summaries, all_trajs):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    colors = {2: "#e377c2", 3: "#ff7f0e", 4: "#1f77b4", 5: "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 5))

    for s in all_summaries:
        n  = s["n_source_bins"]
        dr = s["drop_ratio"]
        alphas = s["alphas"]
        curve  = s["drift_curve"]

        a0 = s["syn_a0"]
        if np.isnan(a0) or a0 < 1e-9:
            continue
        norm = [v / a0 for v in curve]

        label = f"N_BINS={n}  (drop={dr:.3f})"

        if n == 4:
            # CI band for reference case only
            traj = all_trajs[n]
            xs_ci, los, his = [], [], []
            for alpha in sorted(traj.keys()):
                lst = traj[alpha]["drift"]
                if not lst:
                    continue
                m, lo, hi = bootstrap_ci(lst, "Syn", n_boot=N_BOOT)
                xs_ci.append(alpha)
                los.append(lo / a0)
                his.append(hi / a0)
            ax.fill_between(xs_ci, los, his, color=colors[n], alpha=0.15)

        ax.plot(alphas, norm, "o-", color=colors[n], lw=2, ms=5, label=label)

    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("alpha  (0 = base SportCo scores, 1 = adversarial)", fontsize=11)
    ax.set_ylabel("normalized synergy  (÷ own alpha=0)", fontsize=11)
    ax.set_title("Binning sensitivity: SportCo drift on cooperative\n"
                 "N_TARGET=3 fixed; shaded band = N_BINS=4 bootstrap CI",
                 fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "binning_sensitivity.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scores_base, _ = load_scores()
    scores_adv     = invert_scores(scores_base, DRIFT_AGENT)

    score_ranges = {}
    for a in AGENT_KEYS:
        _, ideal_sc, worst_sc = compute_ideal(scores_base, a, ISSUE_SIZES)
        score_ranges[a] = max(ideal_sc - worst_sc, 1)

    all_summaries = []
    all_trajs     = {}

    print(f"\n{'='*60}")
    print(f"binning sensitivity  condition={CONDITION}  drift_agent={DRIFT_AGENT}")
    print(f"N_TARGET={N_TARGET} fixed  |  N_BINS  in  {N_BINS_LIST}")
    print(f"{'='*60}")

    for n_source_bins in N_BINS_LIST:
        print(f"\n--- N_BINS={n_source_bins} ---")
        traj    = run_sweep(n_source_bins, scores_base, scores_adv, score_ranges)
        summary = summarize(traj, n_source_bins)
        check_consistency(summary)

        print(f"  syn@0={summary['syn_a0']:.4f}  syn@1={summary['syn_a1']:.4f}  "
              f"drop_ratio={summary['drop_ratio']:.4f}  ctrl_syn@0={summary['ctrl_syn_a0']:.4f}")

        all_summaries.append(summary)
        all_trajs[n_source_bins] = traj

    # summary table
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'N_BINS':>7}  {'syn@0':>8}  {'syn@1':>8}  {'drop_ratio':>11}  {'ctrl@0':>8}")
    print("-" * 50)
    for s in all_summaries:
        marker = "  <- baseline" if s["n_source_bins"] == 4 else ""
        print(f"{s['n_source_bins']:>7}  {s['syn_a0']:>8.4f}  {s['syn_a1']:>8.4f}  "
              f"{s['drop_ratio']:>11.4f}  {s['ctrl_syn_a0']:>8.4f}{marker}")

    # save
    out = RESULTS_DIR / "binning_sensitivity.json"
    with open(out, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nsaved -> {out}")

    plot(all_summaries, all_trajs)


if __name__ == "__main__":
    main()
