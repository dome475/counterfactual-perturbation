"""
synthetic drift validation.
for each fixed alpha, apply interpolated score sheet for drift_agent uniformly across all rounds.
target always uses base scores; control pairs are identical at every alpha.
supports multiple game configs; overlaid on same figure.
"""

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extract_features import (
    load_scores, load_scores_game1, load_scores_game2, load_scores_game3,
    score_deal, compute_ideal,
    AGENT_KEYS,  ISSUE_SIZES,
    AGENT_KEYS_GAME1, ISSUE_SIZES_GAME1,
    AGENT_KEYS_GAME2, ISSUE_SIZES_GAME2,
    AGENT_KEYS_GAME3, ISSUE_SIZES_GAME3,
)
from pairwise_pid import load_condition, fit_quantile_bins, apply_bins, N_BINS, N_TARGET
from pid_numpy import imin_numpy, bootstrap_ci, compute_o_information, N_BOOT

FIGURES_DIR = Path(__file__).parent / "figures"
MIN_SAMPLES = 5
N_ALPHA     = 11


GAME_CONFIGS = {
    "cooperative": {
        "condition":   "cooperative",
        "drift_agent": "SportCo",
        "load_scores": load_scores,
        "agent_keys":  AGENT_KEYS,
        "issue_sizes": ISSUE_SIZES,
    },
    "adv_targeted": {
        "condition":   "adv_targeted",
        "drift_agent": "union",
        "load_scores": load_scores,
        "agent_keys":  AGENT_KEYS,
        "issue_sizes": ISSUE_SIZES,
    },
    "adv_untargeted": {
        "condition":   "adv_untargeted",
        "drift_agent": "enviroment",
        "load_scores": load_scores,
        "agent_keys":  AGENT_KEYS,
        "issue_sizes": ISSUE_SIZES,
    },
    "game1_coop": {
        "condition":   "game1_coop",
        "drift_agent": "proposing",
        "load_scores": load_scores_game1,
        "agent_keys":  AGENT_KEYS_GAME1,
        "issue_sizes": ISSUE_SIZES_GAME1,
    },
    "game2_coop": {
        "condition":   "game2_coop",
        "drift_agent": "project_manager",
        "load_scores": load_scores_game2,
        "agent_keys":  AGENT_KEYS_GAME2,
        "issue_sizes": ISSUE_SIZES_GAME2,
    },
    "game3_coop": {
        "condition":   "game3_coop",
        "drift_agent": "private_investors",
        "load_scores": load_scores_game3,
        "agent_keys":  AGENT_KEYS_GAME3,
        "issue_sizes": ISSUE_SIZES_GAME3,
    },
}


# score sheet manipulation

def invert_scores(scores, drift_agent):
    """reverse option rankings per issue for drift_agent only"""
    inv = {a: [row[:] for row in scores[a]] for a in scores}
    for i in range(len(scores[drift_agent])):
        inv[drift_agent][i] = list(reversed(scores[drift_agent][i]))
    return inv


def interpolate_scores(scores_a, scores_b, alpha):
    """alpha=0 -> scores_a, alpha=1 -> scores_b"""
    return {a: [
        [int(round((1 - alpha) * scores_a[a][i][j] + alpha * scores_b[a][i][j]))
         for j in range(len(scores_a[a][i]))]
        for i in range(len(scores_a[a]))]
        for a in scores_a}


# state computation at fixed alpha

def compute_states_at_alpha(game, scores_base, scores_adv, alpha,
                            agent_keys, drift_agent, score_ranges):
    rounds = game["rounds"]
    if len(rounds) < 2:
        return []

    sc_interp = interpolate_scores(scores_base, scores_adv, alpha)

    last_self_base  = {a: None for a in agent_keys}
    last_self_drift = None
    prev_sc_base    = None
    prev_sc_drift   = None
    states = []

    for idx, r in enumerate(rounds):
        sc_base    = {a: score_deal(scores_base, a, r["deal"]) for a in agent_keys}
        sc_drift_t = score_deal(sc_interp, drift_agent, r["deal"])
        spk        = r["speaker"]

        if idx == 0:
            last_self_base[spk] = sc_base[spk]
            if spk == drift_agent:
                last_self_drift = sc_drift_t
            prev_sc_base  = sc_base
            prev_sc_drift = sc_drift_t
            continue

        last_self_base[spk] = sc_base[spk]
        if spk == drift_agent:
            last_self_drift = sc_drift_t

        if any(v is None for v in last_self_base.values()) or last_self_drift is None:
            prev_sc_base  = sc_base
            prev_sc_drift = sc_drift_t
            continue

        self_scores = dict(last_self_base)
        self_scores[drift_agent] = last_self_drift

        # normalize sensitivity by base score range for consistent bins across alpha
        sensitivity = {a: abs(sc_base[a] - prev_sc_base[a]) / score_ranges[a]
                       for a in agent_keys}
        sensitivity[drift_agent] = abs(sc_drift_t - prev_sc_drift) / score_ranges[drift_agent]

        states.append({
            "t":           idx,
            "speaker":     spk,
            "scores":      sc_base,
            "self_scores": self_scores,
            "sensitivity": sensitivity,
        })
        prev_sc_base  = sc_base
        prev_sc_drift = sc_drift_t

    return states


# pid per alpha

def build_drift_trajectory(scores_base, scores_adv, condition,
                           agent_keys, drift_agent, score_ranges):
    games  = load_condition(condition)
    alphas = np.linspace(0, 1, N_ALPHA)
    result = {}

    # fit bin edges once at alpha=0 so measurement scale is fixed across alpha
    print("  fitting baseline bin edges at alpha=0...")
    baseline_states, baseline_game_states = [], []
    for g in games:
        ss = compute_states_at_alpha(g, scores_base, scores_adv, 0.0,
                                     agent_keys, drift_agent, score_ranges)
        baseline_game_states.append(ss)
        baseline_states.extend(ss)

    self_edges = {a: fit_quantile_bins([s["self_scores"][a] for s in baseline_states])
                  for a in agent_keys}
    sens_edges = {a: fit_quantile_bins([s["sensitivity"][a] for s in baseline_states])
                  for a in agent_keys}

    baseline_deltas = []
    for ss in baseline_game_states:
        for idx in range(len(ss) - 1):
            s, s_next = ss[idx], ss[idx + 1]
            for (j, k) in combinations(agent_keys, 2):
                noncoal = [a for a in agent_keys if a not in (j, k)]
                baseline_deltas.append(sum(s_next["scores"][a] - s["scores"][a] for a in noncoal))
    delta_edges = fit_quantile_bins(baseline_deltas, n_bins=N_TARGET)

    def joint_bin(s, agent):
        b1 = apply_bins(s["self_scores"][agent], self_edges[agent])
        b2 = apply_bins(s["sensitivity"][agent], sens_edges[agent])
        return b1 * N_BINS + b2

    for alpha in alphas:
        print(f"  alpha={alpha:.2f}")
        game_states = [
            compute_states_at_alpha(g, scores_base, scores_adv, alpha,
                                    agent_keys, drift_agent, score_ranges)
            for g in games
        ]

        if not any(game_states):
            result[float(alpha)] = {"drift": [], "ctrl": []}
            continue

        pair_triplets = defaultdict(list)
        for ss in game_states:
            for idx in range(len(ss) - 1):
                s, s_next = ss[idx], ss[idx + 1]
                for (j, k) in combinations(agent_keys, 2):
                    noncoal = [a for a in agent_keys if a not in (j, k)]
                    delta = sum(s_next["scores"][a] - s["scores"][a] for a in noncoal)
                    pair_triplets[(j, k)].append((
                        joint_bin(s, j),
                        joint_bin(s, k),
                        apply_bins(delta, delta_edges),
                    ))

        drift_atoms, ctrl_atoms = [], []
        for (j, k), triplets in pair_triplets.items():
            if len(triplets) < MIN_SAMPLES:
                continue
            atoms = imin_numpy(triplets)
            (drift_atoms if drift_agent in (j, k) else ctrl_atoms).append(atoms)

        result[float(alpha)] = {"drift": drift_atoms, "ctrl": ctrl_atoms}

    return result


# o-information sweep

def build_o_information_trajectory(scores_base, scores_adv, condition,
                                   agent_keys, drift_agent, score_ranges,
                                   n_boot=200):
    """
    alpha sweep of O-information (binary self_score encoding, median split at alpha=0).
    returns {alpha: (omega, ci_lo, ci_hi)}.
    bootstrap resamples over games.
    """
    games  = load_condition(condition)
    alphas = np.linspace(0, 1, N_ALPHA)
    n_agents = len(agent_keys)

    # fit median splits at alpha=0
    baseline_states = []
    baseline_game_states = []
    for g in games:
        ss = compute_states_at_alpha(g, scores_base, scores_adv, 0.0,
                                     agent_keys, drift_agent, score_ranges)
        baseline_game_states.append(ss)
        baseline_states.extend(ss)

    medians = {a: np.median([s["self_scores"][a] for s in baseline_states])
               for a in agent_keys}

    def encode(game_states_list):
        rows = []
        for ss in game_states_list:
            for s in ss:
                rows.append([1 if s["self_scores"][agent_keys[i]] >= medians[agent_keys[i]] else 0
                              for i in range(n_agents)])
        return np.array(rows, dtype=int) if rows else None

    result = {}
    rng = np.random.default_rng(42)

    for alpha in alphas:
        print(f"  O-info alpha={alpha:.2f}")
        game_states = [
            compute_states_at_alpha(g, scores_base, scores_adv, alpha,
                                    agent_keys, drift_agent, score_ranges)
            for g in games
        ]

        X = encode(game_states)
        if X is None or len(X) < 10:
            result[float(alpha)] = (float("nan"),) * 3
            continue

        omega, _, _ = compute_o_information(X, n_agents)

        boot_omegas = []
        n_games = len(game_states)
        for _ in range(n_boot):
            idx = rng.integers(0, n_games, size=n_games)
            Xb  = encode([game_states[i] for i in idx])
            if Xb is None or len(Xb) < 5:
                continue
            o, _, _ = compute_o_information(Xb, n_agents)
            boot_omegas.append(o)

        if len(boot_omegas) > 2:
            lo = float(np.percentile(boot_omegas, 2.5))
            hi = float(np.percentile(boot_omegas, 97.5))
        else:
            lo = hi = float(omega)

        result[float(alpha)] = (float(omega), lo, hi)

    return result


# plot

def plot_drift():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    titles = {"Syn": "Synergy", "Red": "Redundancy", "NonSyn": "Non-synergistic MI\n(Red + Un1 + Un2)"}
    line_styles = {"cooperative": "-", "adv_targeted": "--", "adv_untargeted": ":",
                   "game1_coop": "-.", "game2_coop": (0, (3, 1, 1, 1)), "game3_coop": (0, (5, 2))}
    drift_color = "#d62728"
    ctrl_color  = "#1f77b4"
    legend_handles = []

    for cfg_name, cfg in GAME_CONFIGS.items():
        condition   = cfg["condition"]
        drift_agent = cfg["drift_agent"]
        agent_keys  = cfg["agent_keys"]
        issue_sizes = cfg["issue_sizes"]
        ls          = line_styles[cfg_name]

        print(f"\n=== {cfg_name} (drift_agent={drift_agent}) ===")
        scores_base, _ = cfg["load_scores"]()
        scores_adv     = invert_scores(scores_base, drift_agent)

        score_ranges = {}
        for a in agent_keys:
            _, ideal_sc, worst_sc = compute_ideal(scores_base, a, issue_sizes)
            score_ranges[a] = max(ideal_sc - worst_sc, 1)
        print(f"  score ranges: { {a: score_ranges[a] for a in agent_keys} }")

        print(f"  {drift_agent} issue scores (base -> adversarial):")
        for i, (rb, ra) in enumerate(zip(scores_base[drift_agent], scores_adv[drift_agent])):
            print(f"    issue {i}: {rb}  ->  {ra}")

        print("  running PID per alpha...")
        traj       = build_drift_trajectory(scores_base, scores_adv, condition,
                                            agent_keys, drift_agent, score_ranges)
        alphas_all = sorted(traj.keys())

        for alpha_data in traj.values():
            for atom_list in alpha_data.values():
                for a in atom_list:
                    a["NonSyn"] = a["Red"] + a["Un1"] + a["Un2"]

        for ax, atom in zip(axes, ["Syn", "Red", "NonSyn"]):
            for group, color in [("drift", drift_color), ("ctrl", ctrl_color)]:
                label = (f"{cfg_name}: pairs w/ {drift_agent}"
                         if group == "drift" else f"{cfg_name}: ctrl pairs")
                xs, means, los, his = [], [], [], []
                for alpha in alphas_all:
                    atom_list = traj[alpha][group]
                    if not atom_list:
                        continue
                    m, lo, hi = bootstrap_ci(atom_list, atom, n_boot=N_BOOT)
                    xs.append(alpha); means.append(m); los.append(lo); his.append(hi)
                if not xs:
                    continue
                h, = ax.plot(xs, means, color=color, lw=2, marker="o", ms=4,
                             linestyle=ls, label=label)
                ax.fill_between(xs, los, his, color=color, alpha=0.10)
                if atom == "Syn":
                    legend_handles.append(h)

        print(f"\n  === atom breakdown ({drift_agent} drift pairs) ===")
        print(f"  {'alpha':>6}  {'Red':>7}  {'Un1':>7}  {'Un2':>7}  {'Syn':>7}  {'MI':>7}  {'Syn/MI':>7}")
        for alpha in alphas_all:
            atom_list = traj[alpha]["drift"]
            if not atom_list:
                continue
            ms = {k: np.mean([a[k] for a in atom_list])
                  for k in ["Red", "Un1", "Un2", "Syn", "MI", "NonSyn"]}
            ratio = ms["Syn"] / ms["MI"] if ms["MI"] > 1e-9 else float("nan")
            print(f"  {alpha:>6.2f}  {ms['Red']:>7.4f}  {ms['Un1']:>7.4f}  "
                  f"{ms['Un2']:>7.4f}  {ms['Syn']:>7.4f}  {ms['MI']:>7.4f}  {ratio:>7.3f}")

    for ax, atom in zip(axes, ["Syn", "Red", "NonSyn"]):
        ax.set_title(titles[atom], fontsize=13, fontweight="bold")
        ax.set_xlabel("alpha  (0 = base, 1 = adversarial)", fontsize=11)
        ax.set_ylabel("bits", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("PID under preference drift, all game configurations", fontsize=12)
    fig.tight_layout(rect=[0, 0.10, 1, 1])

    out = FIGURES_DIR / "synthetic_drift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    plot_drift()
