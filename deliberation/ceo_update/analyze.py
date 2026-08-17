"""
two-window perturbation analysis for the mid-game preference swap.

window a (rounds 1-12): behaviour under original prefs.
window b (rounds 13-24): behaviour under swapped prefs.
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

N_ALPHA     = 11
MIN_SAMPLES = 10
COLORS      = {"A": "#2ca02c", "B": "#d62728"}

# CEO modification: SportCo Issue D (Federal Loan), 0-indexed=3, reversed
# original [35, 29, 20, 0] -> reversed [0, 20, 29, 35]
# same issue as collusion experiment, required for methodology comparison
DRIFT_AGENT  = "SportCo"
CEO_MOD      = {"issue": 3, "modified": [0, 20, 29, 35]}
SCORE_RANGES = {a: 100 for a in AGENT_KEYS}  # ideal=100 worst=0 for all agents in base game


def load_games():
    games = []
    for p in sorted(FEAT_DIR.glob("ceo_*.json")):
        g = json.load(open(p))
        if g["n_rounds_total"] >= 18:
            games.append(g)
        else:
            print(f"  skipping {g['game_id']}: only {g['n_rounds_total']} rounds")
    return games


def build_adv_scores(scores_base):
    s = copy.deepcopy(scores_base)
    s[DRIFT_AGENT][CEO_MOD["issue"]] = list(CEO_MOD["modified"])
    return s


def fit_edges(all_games, scores_base, scores_adv):
    """fit per-window bin edges at alpha=0; returns {win_key: edges}

    global edge fitting is incorrect here: the experimental design produces a
    distributional shift between windows (Window B proposals reflect inverted
    prefs), so aggregate quantiles do not represent within-window distributions.
    """
    result = {}
    for win_key in ("window_a", "window_b"):
        all_states, all_deltas = [], []
        for g in all_games:
            sub = g[win_key]
            ss = compute_states_at_alpha(sub, scores_base, scores_adv, 0.0,
                                         AGENT_KEYS, DRIFT_AGENT, SCORE_RANGES)
            all_states.extend(ss)
            for idx in range(len(ss) - 1):
                s, sn = ss[idx], ss[idx + 1]
                for (j, k) in combinations(AGENT_KEYS, 2):
                    noncoal = [a for a in AGENT_KEYS if a not in (j, k)]
                    all_deltas.append(sum(sn["scores"][a] - s["scores"][a] for a in noncoal))
        print(f"  edge fitting {win_key}: {len(all_states)} states, {len(all_deltas)} deltas")
        result[win_key] = {
            "self":  {a: fit_quantile_bins([s["self_scores"][a] for s in all_states])
                      for a in AGENT_KEYS},
            "sens":  {a: fit_quantile_bins([s["sensitivity"][a] for s in all_states])
                      for a in AGENT_KEYS},
            "delta": fit_quantile_bins(all_deltas, n_bins=N_TARGET),
        }
    return result


def run_sweep(games, win_key, scores_base, scores_adv, edges, alphas):
    """pool triplets across games in window, sweep alpha"""
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
            sub = g[win_key]
            ss  = compute_states_at_alpha(sub, scores_base, scores_adv, alpha,
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
            atoms  = imin_numpy(triplets)
            target = drift_atoms if DRIFT_AGENT in (j, k) else ctrl_atoms
            target.append({**atoms, "pair": f"{j}|{k}", "n": len(triplets)})

        result[float(alpha)] = {"drift": drift_atoms, "ctrl": ctrl_atoms}

    return result


def compute_metrics(traj):
    alphas = sorted(traj.keys())
    vals   = [np.mean([a["Syn"] for a in traj[alpha]["drift"]]) if traj[alpha]["drift"]
              else float("nan") for alpha in alphas]
    s0 = vals[0] if vals and not np.isnan(vals[0]) else float("nan")
    s1 = vals[-1] if vals and not np.isnan(vals[-1]) else float("nan")
    ratio = s1 / s0 if not np.isnan(s0) and s0 > 1e-9 else float("nan")
    return {"syn_a0": s0, "syn_a1": s1, "ratio": ratio}


def collect_game_triplets_both_windows(games, scores_base, scores_adv, edges_by_win):
    """
    precompute per-game pair->triplet lists at alpha=0 and alpha=1 for both windows.
    returns list of {"window_a": {0.0: {pair: trips}, 1.0: ...},
                     "window_b": {0.0: {pair: trips}, 1.0: ...}}
    cluster bootstrap resamples game indices and uses whichever window is needed,
    preserving the within-game dependency between window_a and window_b.
    """
    per_game = []
    for g in games:
        gdata = {}
        for win_key in ("window_a", "window_b"):
            edges   = edges_by_win[win_key]
            self_e  = edges["self"]
            sens_e  = edges["sens"]
            delta_e = edges["delta"]

            def jbin(s, a, se=self_e, se2=sens_e):
                return apply_bins(s["self_scores"][a], se[a]) * N_BINS + \
                       apply_bins(s["sensitivity"][a], se2[a])

            win_data = {}
            for alpha in [0.0, 1.0]:
                sub = g[win_key]
                ss  = compute_states_at_alpha(sub, scores_base, scores_adv, alpha,
                                              AGENT_KEYS, DRIFT_AGENT, SCORE_RANGES)
                pt = defaultdict(list)
                for idx in range(len(ss) - 1):
                    s, sn = ss[idx], ss[idx + 1]
                    for (j, k) in combinations(AGENT_KEYS, 2):
                        noncoal = [a for a in AGENT_KEYS if a not in (j, k)]
                        delta = sum(sn["scores"][a] - s["scores"][a] for a in noncoal)
                        pt[(j, k)].append((jbin(s, j), jbin(s, k),
                                           apply_bins(delta, delta_e)))
                win_data[alpha] = dict(pt)
            gdata[win_key] = win_data
        per_game.append(gdata)
    return per_game


def _ceo_syn_from_pool(win_data_subset, alpha):
    """pool resampled per-game window data, return mean drift-pair Syn"""
    pair_triplets = defaultdict(list)
    for wd in win_data_subset:
        for pair, trips in wd[alpha].items():
            pair_triplets[pair].extend(trips)
    drift_atoms = []
    for (j, k), triplets in pair_triplets.items():
        if len(triplets) < MIN_SAMPLES or DRIFT_AGENT not in (j, k):
            continue
        drift_atoms.append(imin_numpy(triplets))
    return float(np.mean([a["Syn"] for a in drift_atoms])) if drift_atoms else float("nan")


def cluster_bootstrap_ratio_window(per_game, win_key, n_boot=1000, rng_seed=42):
    """cluster bootstrap over games for a single window's ratio CI"""
    rng = np.random.default_rng(rng_seed)
    n   = len(per_game)
    boot_ratios = []
    for _ in range(n_boot):
        idx      = rng.integers(0, n, size=n)
        win_data = [per_game[i][win_key] for i in idx]
        s0 = _ceo_syn_from_pool(win_data, 0.0)
        s1 = _ceo_syn_from_pool(win_data, 1.0)
        if not np.isnan(s0) and s0 > 1e-9:
            boot_ratios.append(s1 / s0)
    if len(boot_ratios) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(boot_ratios, 2.5)), float(np.percentile(boot_ratios, 97.5))


def plot(trajs, metrics):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_n, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))

    for win, (label, traj) in trajs.items():
        color  = COLORS[win]
        alphas = sorted(traj.keys())
        a0_val = metrics[win]["syn_a0"]

        xs_r, ms_r, lo_r, hi_r = [], [], [], []
        xs_n, ms_n, lo_n, hi_n = [], [], [], []

        for alpha in alphas:
            lst = traj[alpha]["drift"]
            if not lst:
                continue
            m, lo, hi = bootstrap_ci(lst, "Syn", n_boot=N_BOOT)
            xs_r.append(alpha); ms_r.append(m); lo_r.append(lo); hi_r.append(hi)
            if not np.isnan(a0_val) and a0_val > 1e-9:
                xs_n.append(alpha)
                ms_n.append(m / a0_val); lo_n.append(lo / a0_val); hi_n.append(hi / a0_val)

        lbl = f"window {win}  (ratio={metrics[win]['ratio']:.3f})"
        ax_r.plot(xs_r, ms_r, "o-", color=color, lw=2, ms=4, label=lbl)
        ax_r.fill_between(xs_r, lo_r, hi_r, color=color, alpha=0.12)
        if xs_n:
            ax_n.plot(xs_n, ms_n, "o-", color=color, lw=2, ms=4, label=lbl)
            ax_n.fill_between(xs_n, lo_n, hi_n, color=color, alpha=0.12)

    for ax, title in [(ax_n, "normalized synergy  (÷ own alpha=0)"),
                      (ax_r, "raw synergy  (bits)")]:
        ax.set_xlabel("alpha  (0=original prefs, 1=CEO/inverted prefs)", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)
    ax_n.axhline(1.0, color="gray", ls=":", alpha=0.4, lw=1)

    fig.suptitle(f"CEO Update, SportCo Issue 3 reversed at round 13\n"
                 f"Window A (pre-CEO): ratio={metrics['A']['ratio']:.3f}  "
                 f"Window B (post-CEO): ratio={metrics['B']['ratio']:.3f}", fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / "ceo_update.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scores_base, _ = load_scores()
    scores_adv     = build_adv_scores(scores_base)
    alphas = np.linspace(0.0, 1.0, N_ALPHA)

    games = load_games()
    print(f"loaded {len(games)} games")
    if not games:
        print("no games found, run extract_features.py first")
        return

    print("fitting per-window bin edges at alpha=0...")
    edges_by_win = fit_edges(games, scores_base, scores_adv)
    for wk in ("window_a", "window_b"):
        print(f"  {wk} delta edges: {edges_by_win[wk]['delta']}")

    metrics  = {}
    trajs    = {}
    for win_key, win_label, label in [("window_a", "A", "pre-CEO (rounds 1-12)"),
                                       ("window_b", "B", "post-CEO (rounds 13-24)")]:
        print(f"\n=== Window {win_label}: {label} ===")
        traj = run_sweep(games, win_key, scores_base, scores_adv, edges_by_win[win_key], alphas)
        m    = compute_metrics(traj)
        metrics[win_label] = m
        trajs[win_label]   = (label, traj)
        print(f"  syn@0={m['syn_a0']:.4f}  syn@1={m['syn_a1']:.4f}  ratio={m['ratio']:.3f}")

    print("\nprecomputing per-game triplets for cluster bootstrap...")
    per_game = collect_game_triplets_both_windows(games, scores_base, scores_adv, edges_by_win)

    print("running cluster bootstrap (n=1000) per window...")
    for win_key, win_label in [("window_a", "A"), ("window_b", "B")]:
        ci_lo, ci_hi = cluster_bootstrap_ratio_window(per_game, win_key)
        metrics[win_label]["ratio_ci_low"]  = ci_lo
        metrics[win_label]["ratio_ci_high"] = ci_hi
        excludes_1 = not np.isnan(ci_hi) and ci_hi < 1.0
        overlaps_1 = not np.isnan(ci_lo) and ci_lo < 1.0
        note = "CI excl 1.0" if excludes_1 else ("CI overlaps 1.0" if overlaps_1 else "")
        print(f"  Window {win_label}: ratio={metrics[win_label]['ratio']:.3f} "
              f"[{ci_lo:.3f}, {ci_hi:.3f}]  {note}")

    gap = metrics["B"]["ratio"] - metrics["A"]["ratio"]
    print(f"\n=== summary (cluster-bootstrap CIs, n=1000) ===")
    print(f"  ratio_A = {metrics['A']['ratio']:.3f} "
          f"[{metrics['A']['ratio_ci_low']:.3f}, {metrics['A']['ratio_ci_high']:.3f}]")
    print(f"  ratio_B = {metrics['B']['ratio']:.3f} "
          f"[{metrics['B']['ratio_ci_low']:.3f}, {metrics['B']['ratio_ci_high']:.3f}]")
    print(f"  gap     = {gap:.3f}")

    plot(trajs, metrics)

    out_data = {
        "n_games":       len(games),
        "drift_agent":   DRIFT_AGENT,
        "ceo_mod":       CEO_MOD,
        "split_t":       13,
        "metrics":       metrics,
        "gap":           gap,
        "n_boot":        1000,
        "trajectory": {
            win_label: {
                str(alpha): {
                    "drift": [{k: v for k, v in a.items() if k not in ("pair", "n")}
                              for a in traj[float(alpha)]["drift"]],
                    "ctrl":  [{k: v for k, v in a.items() if k not in ("pair", "n")}
                              for a in traj[float(alpha)]["ctrl"]],
                }
                for alpha in sorted(traj.keys())
            }
            for win_label, (_, traj) in trajs.items()
        },
    }
    out_path = RESULTS_DIR / "ceo_update.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"saved -> {out_path}")

    cb_data = {
        "n_games": len(games), "n_boot": 1000,
        "window_A": {k: metrics["A"][k] for k in ("ratio", "ratio_ci_low", "ratio_ci_high")},
        "window_B": {k: metrics["B"][k] for k in ("ratio", "ratio_ci_low", "ratio_ci_high")},
    }
    cb_path = RESULTS_DIR / "cluster_bootstrap.json"
    with open(cb_path, "w") as f:
        json.dump(cb_data, f, indent=2)
    print(f"saved -> {cb_path}")


if __name__ == "__main__":
    main()
