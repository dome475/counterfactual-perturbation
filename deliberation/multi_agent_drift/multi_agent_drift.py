"""
alpha sweep over several drift agents per game config, producing
agent-specific sensitivity profiles. asks whether high-stakes agents
show steeper synergy drops than low-stakes agents.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "synthetic_drift"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extract_features import (
    load_scores, load_scores_game1, load_scores_game2, load_scores_game3,
    compute_ideal,
    AGENT_KEYS,  ISSUE_SIZES,
    AGENT_KEYS_GAME1, ISSUE_SIZES_GAME1,
    AGENT_KEYS_GAME2, ISSUE_SIZES_GAME2,
    AGENT_KEYS_GAME3, ISSUE_SIZES_GAME3,
)
from synthetic_drift import build_drift_trajectory, invert_scores

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR     = Path(__file__).parent / "figures"
N_ALPHA     = 11

# all drift agents to test per game config; key must match synthetic_drift GAME_CONFIGS
MULTI_AGENT_CONFIGS = {
    "cooperative": {
        "condition":   "cooperative",
        "load_scores": load_scores,
        "agent_keys":  AGENT_KEYS,
        "issue_sizes": ISSUE_SIZES,
        "drift_agents": ["SportCo", "union", "mayor", "DoT"],
    },
    "adv_targeted": {
        "condition":   "adv_targeted",
        "load_scores": load_scores,
        "agent_keys":  AGENT_KEYS,
        "issue_sizes": ISSUE_SIZES,
        "drift_agents": ["union", "SportCo", "mayor"],
    },
    "adv_untargeted": {
        "condition":   "adv_untargeted",
        "load_scores": load_scores,
        "agent_keys":  AGENT_KEYS,
        "issue_sizes": ISSUE_SIZES,
        "drift_agents": ["enviroment", "SportCo", "mayor"],
    },
    "game1_coop": {
        "condition":   "game1_coop",
        "load_scores": load_scores_game1,
        "agent_keys":  AGENT_KEYS_GAME1,
        "issue_sizes": ISSUE_SIZES_GAME1,
        "drift_agents": ["proposing", "bank", "construction"],
    },
    "game2_coop": {
        "condition":   "game2_coop",
        "load_scores": load_scores_game2,
        "agent_keys":  AGENT_KEYS_GAME2,
        "issue_sizes": ISSUE_SIZES_GAME2,
        "drift_agents": ["project_manager", "landowners", "government"],
    },
    "game3_coop": {
        "condition":   "game3_coop",
        "load_scores": load_scores_game3,
        "agent_keys":  AGENT_KEYS_GAME3,
        "issue_sizes": ISSUE_SIZES_GAME3,
        "drift_agents": ["private_investors", "community", "federal"],
    },
}


def compute_agent_stakes(scores, agent, issue_sizes):
    """sum of (max-min) per issue, higher = more at stake"""
    return sum(max(scores[agent][i]) - min(scores[agent][i])
               for i in range(len(scores[agent])))


def extract_mean_syn(traj, group):
    """extract mean synergy per alpha from build_drift_trajectory output"""
    alphas = sorted(traj.keys())
    out = {}
    for alpha in alphas:
        atom_list = traj[alpha][group]
        if not atom_list:
            out[alpha] = float("nan")
        else:
            out[alpha] = float(np.mean([a["Syn"] for a in atom_list]))
    return out


def run_agent(cfg, drift_agent):
    """run full alpha sweep for one drift agent; returns per-alpha syn dict"""
    condition   = cfg["condition"]
    agent_keys  = cfg["agent_keys"]
    issue_sizes = cfg["issue_sizes"]

    scores_base, _ = cfg["load_scores"]()
    scores_adv     = invert_scores(scores_base, drift_agent)

    score_ranges = {}
    for a in agent_keys:
        _, ideal_sc, worst_sc = compute_ideal(scores_base, a, issue_sizes)
        score_ranges[a] = max(ideal_sc - worst_sc, 1)

    traj = build_drift_trajectory(
        scores_base, scores_adv, condition,
        agent_keys, drift_agent, score_ranges,
    )

    drift_syn = extract_mean_syn(traj, "drift")
    ctrl_syn  = extract_mean_syn(traj, "ctrl")
    stakes    = compute_agent_stakes(scores_base, drift_agent, issue_sizes)

    # drop ratio: syn at alpha=1 relative to alpha=0 (lower = steeper drop)
    s0 = drift_syn.get(0.0, float("nan"))
    s1 = drift_syn.get(1.0, float("nan"))
    drop_ratio = (s1 / s0) if (s0 and not np.isnan(s0) and s0 > 1e-9) else float("nan")

    return {
        "drift_syn":   drift_syn,
        "ctrl_syn":    ctrl_syn,
        "stakes":      stakes,
        "drop_ratio":  drop_ratio,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for game_name, cfg in MULTI_AGENT_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"game: {game_name}  condition: {cfg['condition']}")
        print(f"drift agents: {cfg['drift_agents']}")
        print(f"{'='*60}")

        game_results = {}
        for drift_agent in cfg["drift_agents"]:
            print(f"\n  drift_agent={drift_agent}")
            try:
                result = run_agent(cfg, drift_agent)
                game_results[drift_agent] = result
                print(f"    stakes={result['stakes']}  drop_ratio={result['drop_ratio']:.3f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                game_results[drift_agent] = {"error": str(e)}

        all_results[game_name] = game_results

    out = RESULTS_DIR / "multi_agent_drift.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved -> {out}")

    plot_results(all_results)


def plot_results(all_results):
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # figure 1: per-game panels, normalized synergy vs alpha
    n_games = len(all_results)
    ncols   = 3
    nrows   = (n_games + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for idx, (game_name, game_data) in enumerate(all_results.items()):
        ax = axes[idx]
        for agent_idx, (agent, data) in enumerate(game_data.items()):
            if "error" in data:
                continue
            drift_syn = data["drift_syn"]
            alphas    = sorted(drift_syn.keys(), key=float)  # keys are str via --replot
            xs        = [float(a) for a in alphas]
            syns      = [drift_syn[a] for a in alphas]

            s0 = syns[0] if (syns and not np.isnan(syns[0]) and syns[0] > 1e-9) else None
            if s0 is None:
                continue
            syns_norm = [s / s0 for s in syns]

            ax.plot(xs, syns_norm, "o-", color=colors[agent_idx],
                    label=f"agent {agent_idx + 1}", markersize=4, lw=1.8)

        ax.set_title(f"configuration {idx + 1}", fontsize=10)
        ax.set_ylim(0, 1.5)
        ax.set_xticks([0.0, 0.5, 1.0])  # thin ticks; full sweep is 11 alphas
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        if idx >= ncols * (nrows - 1):
            ax.set_xlabel("alpha (preference misalignment)", fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel("normalized synergy", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")

    # hide unused panels
    for idx in range(n_games, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("agent-specific perturbation sensitivity profiles "
                 "(each panel: one game configuration, aggregated over its games)",
                 fontsize=13)
    plt.tight_layout()
    out1 = FIG_DIR / "multi_agent_drift_profiles.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"saved figure -> {out1}")

    # figure 2: stakes vs drop_ratio scatter
    stakes_all    = []
    drop_all      = []
    labels_all    = []
    game_idx_all  = []

    for g_idx, (game_name, game_data) in enumerate(all_results.items()):
        for agent, data in game_data.items():
            if "error" in data:
                continue
            s = data.get("stakes")
            d = data.get("drop_ratio")
            if s is None or d is None or np.isnan(d):
                continue
            stakes_all.append(s)
            drop_all.append(d)
            labels_all.append(f"{game_name}\n{agent}")
            game_idx_all.append(g_idx)

    if stakes_all:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sc = ax2.scatter(stakes_all, drop_all,
                         c=game_idx_all, cmap="tab10", s=60, zorder=3)
        for x, y, lbl in zip(stakes_all, drop_all, labels_all):
            ax2.annotate(lbl, (x, y), fontsize=6, ha="left",
                         xytext=(4, 2), textcoords="offset points")

        # fit line
        if len(stakes_all) >= 3:
            m, b = np.polyfit(stakes_all, drop_all, 1)
            xs = np.linspace(min(stakes_all), max(stakes_all), 100)
            ax2.plot(xs, m * xs + b, "k--", lw=1.2, alpha=0.6,
                     label=f"slope={m:.4f}")
            ax2.legend(fontsize=8)

        ax2.set_xlabel("agent stakes (total score spread)", fontsize=10)
        ax2.set_ylabel("syn drop ratio  (syn@alpha=1 / syn@alpha=0)", fontsize=10)
        ax2.set_title("high-stakes agents -> steeper synergy drop?", fontsize=11)
        ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        out2 = FIG_DIR / "multi_agent_drift_stakes.png"
        plt.savefig(out2, dpi=150)
        plt.close()
        print(f"saved figure -> {out2}")


def replot():
    # regenerate figure from saved results, no recompute
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = json.load(open(RESULTS_DIR / "multi_agent_drift.json"))
    plot_results(all_results)


if __name__ == "__main__":
    replot() if "--replot" in sys.argv else main()
