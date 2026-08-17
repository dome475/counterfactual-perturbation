"""
game-theoretic baselines under preference drift.
same alpha sweep as synthetic_drift.py; measures BR gap, Pareto dist, SW, NBS dist
instead of PID atoms. tests whether GT tools show same selective response as PID synergy.
"""

import sys
from itertools import product as iproduct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extract_features import score_deal
from pairwise_pid import load_condition
from synthetic_drift import GAME_CONFIGS, invert_scores, interpolate_scores

FIGURES_DIR = Path(__file__).parent / "figures"
N_ALPHA     = 11
N_BOOT      = 500


# gt primitives

def payoff_matrix(sc, agent_keys, all_deals):
    """(N_deals, N_agents) score array"""
    return np.array(
        [[score_deal(sc, a, list(d)) for a in agent_keys] for d in all_deals],
        dtype=float,
    )


def pareto_frontier(pmat):
    """non-dominated rows of pmat"""
    N   = pmat.shape[0]
    dom = np.zeros(N, dtype=bool)
    for i in range(N):
        beats = np.all(pmat >= pmat[i], axis=1) & np.any(pmat > pmat[i], axis=1)
        beats[i] = False
        if beats.any():
            dom[i] = True
    return pmat[~dom]


def nbs_payoff(pmat, mins_arr):
    """Nash bargaining solution payoff vector (max log-surplus product)"""
    surplus = pmat - mins_arr
    valid   = np.all(surplus > 0, axis=1)
    if valid.any():
        log_s  = np.where(surplus > 0, np.log(np.clip(surplus, 1e-12, None)), -1e9)
        scores = np.where(valid[:, None], log_s, -1e9).sum(axis=1)
    else:
        scores = surplus.sum(axis=1)  # fallback: no feasible deal
    return pmat[int(np.argmax(scores))]


def min_l1(vec, frontier):
    """min L1 distance from vec to any row of frontier"""
    return float(np.min(np.sum(np.abs(frontier - vec), axis=1)))


def bootstrap_ci(values, n_boot=N_BOOT, seed=42):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        m = float(v.mean()) if len(v) else float("nan")
        return m, m, m
    rng  = np.random.default_rng(seed)
    idx  = rng.integers(0, len(v), size=(n_boot, len(v)))
    boot = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# per-round data

def round_rows(game, scores_base, sc_interp, drift_agent, agent_keys):
    """
    per usable round (all agents spoken once):
      sc_b[a]  base score of current proposal
      sc_i[a]  interp score of current proposal (only drift_agent differs)
      cf_b[a]  base score of carry-forward deal
      cf_i[a]  interp score of carry-forward (only drift_agent differs)
    """
    last_cf = {a: None for a in agent_keys}
    rows    = []
    for r in game["rounds"]:
        spk  = r["speaker"]
        deal = r["deal"]
        if spk not in last_cf:
            continue
        last_cf[spk] = deal
        if any(v is None for v in last_cf.values()):
            continue
        sc_b = r["scores"]
        sc_i = {**sc_b, drift_agent: score_deal(sc_interp, drift_agent, deal)}
        cf_b = {a: score_deal(scores_base, a, last_cf[a]) for a in agent_keys}
        cf_i = {**cf_b, drift_agent: score_deal(sc_interp, drift_agent, last_cf[drift_agent])}
        rows.append({"sc_b": sc_b, "sc_i": sc_i, "cf_b": cf_b, "cf_i": cf_i})
    return rows


# main sweep

def run_sweep(cfg_name, cfg):
    condition   = cfg["condition"]
    drift_agent = cfg["drift_agent"]
    agent_keys  = cfg["agent_keys"]
    issue_sizes = cfg["issue_sizes"]

    scores_base, mins_dict = cfg["load_scores"]()
    scores_adv  = invert_scores(scores_base, drift_agent)
    mins_arr    = np.array([mins_dict[a] for a in agent_keys])
    ctrl_agents = [a for a in agent_keys if a != drift_agent]
    dai         = agent_keys.index(drift_agent)

    all_deals = list(iproduct(*[range(s) for s in issue_sizes]))
    alphas    = np.linspace(0, 1, N_ALPHA)
    games     = load_condition(condition)

    # base quantities, constant across alpha
    pmat_b  = payoff_matrix(scores_base, agent_keys, all_deals)
    front_b = pareto_frontier(pmat_b)
    nbs_b   = nbs_payoff(pmat_b, mins_arr)
    br_ctrl = {a: float(pmat_b[:, agent_keys.index(a)].max()) for a in ctrl_agents}

    print("  note: br_ctrl, pd_base, nbs_base are flat by design (base scores fixed)")
    result = {}

    for alpha in alphas:
        print(f"  alpha={alpha:.2f}")
        sc_interp = interpolate_scores(scores_base, scores_adv, alpha)
        pmat_i    = payoff_matrix(sc_interp, agent_keys, all_deals)
        front_i   = pareto_frontier(pmat_i)
        nbs_i     = nbs_payoff(pmat_i, mins_arr)
        br_drift  = float(pmat_i[:, dai].max())

        keys = ["br_drift", "br_ctrl", "pd_interp", "pd_base",
                "sw_base",  "sw_interp", "nbs_interp", "nbs_base"]
        g_vals = {k: [] for k in keys}

        for g in games:
            rr = round_rows(g, scores_base, sc_interp, drift_agent, agent_keys)
            if not rr:
                continue
            per = {k: [] for k in keys}
            for row in rr:
                sc_b, sc_i, cf_b, cf_i = row["sc_b"], row["sc_i"], row["cf_b"], row["cf_i"]
                pv_i = np.array([sc_i[a] for a in agent_keys])
                pv_b = np.array([sc_b[a] for a in agent_keys])

                per["br_drift"].append(br_drift - cf_i[drift_agent])
                per["br_ctrl"].append(np.mean([br_ctrl[a] - cf_b[a] for a in ctrl_agents]))
                per["pd_interp"].append(min_l1(pv_i, front_i))
                per["pd_base"].append(min_l1(pv_b, front_b))
                per["sw_base"].append(float(sum(sc_b[a] for a in agent_keys)))
                per["sw_interp"].append(float(sum(sc_i[a] for a in agent_keys)))
                per["nbs_interp"].append(float(np.sum(np.abs(pv_i - nbs_i))))
                per["nbs_base"].append(float(np.sum(np.abs(pv_b - nbs_b))))

            for k in keys:
                if per[k]:
                    g_vals[k].append(float(np.mean(per[k])))

        result[float(alpha)] = {k: bootstrap_ci(g_vals[k]) for k in keys}

    return result


# plot

def plot_results(results):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    panels = [
        ("br_drift",   "br_ctrl",
         "Best-Response Gap",    "score units",
         "drift agent (interp prefs)", "control agents (base prefs)"),
        ("pd_interp",  "pd_base",
         "Pareto Distance (L1)", "L1 score units",
         "under interp frontier", "under base frontier (constant)"),
        ("sw_interp",  "sw_base",
         "Social Welfare",       "total score",
         "interp scores", "base scores (sanity check)"),
        ("nbs_interp", "nbs_base",
         "NBS Distance (L1)",    "L1 score units",
         "from interp NBS", "from base NBS (constant)"),
    ]

    ls_map    = {"cooperative": "-", "adv_targeted": "--", "adv_untargeted": ":",
                 "game1_coop": "-.", "game2_coop": (0, (3, 1, 1, 1)), "game3_coop": (0, (5, 2))}
    col_drift = "#d62728"
    col_ctrl  = "#1f77b4"
    handles   = []

    for ax, (dk, ck, title, ylabel, dl, cl) in zip(axes, panels):
        for cfg_name, res in results.items():
            ls       = ls_map[cfg_name]
            alphas_s = sorted(res.keys())
            for key, color, lbl in [(dk, col_drift, f"{cfg_name}: {dl}"),
                                     (ck, col_ctrl,  f"{cfg_name}: {cl}")]:
                xs, ms, los, his = [], [], [], []
                for a in alphas_s:
                    m, lo, hi = res[a][key]
                    xs.append(a); ms.append(m); los.append(lo); his.append(hi)
                h, = ax.plot(xs, ms, color=color, lw=2, ls=ls,
                             marker="o", ms=4, label=lbl)
                ax.fill_between(xs, los, his, color=color, alpha=0.12)
                if title == "Best-Response Gap":
                    handles.append(h)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("alpha  (0 = base, 1 = adversarial)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(handles, [h.get_label() for h in handles],
               loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Game-theoretic quantities under preference drift", fontsize=13)
    fig.tight_layout(rect=[0, 0.09, 1, 1])

    out = FIGURES_DIR / "game_theory_drift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


# comparison table

def print_comparison_table(results):
    coop = results.get("cooperative", {})
    if not coop:
        return

    a0, a1 = min(coop.keys()), max(coop.keys())
    W = 95

    # (label, drift_key, ctrl_key, ctrl_flat_by_design)
    rows = [
        ("best-response gap",       "br_drift",   "br_ctrl",    True),
        ("pareto dist (interp)",    "pd_interp",  "pd_base",    True),
        ("social welfare (base)",   "sw_base",    "sw_base",    True),
        ("social welfare (interp)", "sw_interp",  "sw_interp",  True),
        ("NBS dist (interp)",       "nbs_interp", "nbs_base",   True),
    ]

    print("\n" + "=" * W)
    print("comparison table (cooperative, alpha = 0 -> 1; ctrl held flat by design)")
    print("=" * W)
    print(f"  {'quantity':<26}  {'drift alpha=0':>9}  {'drift alpha=1':>9}  {'deltadrift%':>8}  "
          f"{'ctrl alpha=0':>9}  {'ctrl alpha=1':>9}")
    print("-" * W)

    for lbl, dk, ck, by_design in rows:
        d0, _, _ = coop[a0][dk];  d1, _, _ = coop[a1][dk]
        c0, _, _ = coop[a0][ck];  c1, _, _ = coop[a1][ck]
        d_pct = 100.0 * (d1 - d0) / max(abs(d0), 1e-9)
        print(f"  {lbl:<26}  {d0:>9.2f}  {d1:>9.2f}  {d_pct:>7.1f}%  "
              f"{c0:>9.2f}  {c1:>9.2f}")
    print("=" * W)


# main

def main():
    results = {}
    for cfg_name, cfg in GAME_CONFIGS.items():
        print(f"\n=== {cfg_name} (drift_agent={cfg['drift_agent']}) ===")
        results[cfg_name] = run_sweep(cfg_name, cfg)

    plot_results(results)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
