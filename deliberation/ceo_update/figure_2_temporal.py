"""
figure 2: temporal split, mid term swap update experiment.
two panels: window A (pre-update rounds) and window B (post-update rounds).
sweep curves with CI bands + endpoint dot with cluster-bootstrap CI.
data: ceo_update.json (sweep), cluster_bootstrap.json (endpoints)
output: paper/figures/figure_2_temporal.{pdf,png}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE  = Path(__file__).parent
PAPER = HERE.parent.parent / "paper" / "figures"
PAPER.mkdir(parents=True, exist_ok=True)

cu    = json.load(open(HERE / "results/ceo_update.json"))
cu_bs = json.load(open(HERE / "results/cluster_bootstrap.json"))

traj = cu["trajectory"]

COLOR = "#2166ac"   # single window color

WINDOW_TITLES = {"A": "Window A  (pre-update)", "B": "Window B  (post-update)"}
BS_KEYS       = {"A": "window_A", "B": "window_B"}


def sweep_mean(win_traj):
    """mean normalized synergy per alpha across drift pairs."""
    alphas = sorted(win_traj.keys(), key=float)
    s0     = np.mean([p["Syn"] for p in win_traj["0.0"]["drift"]])
    xs, ys = [], []
    for a in alphas:
        sv = np.array([p["Syn"] for p in win_traj[a]["drift"]]) / s0
        xs.append(float(a))
        ys.append(sv.mean())
    return np.array(xs), np.array(ys)


plt.rcParams.update({
    "font.family":       "serif",
    "mathtext.fontset":  "cm",
    "font.size":         10,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True,
                         layout="constrained",
                         gridspec_kw={"wspace": 0.12})

for ax, win in zip(axes, ["A", "B"]):
    xs, ys = sweep_mean(traj[win])
    ax.plot(xs, ys, color=COLOR, lw=1.8)

    ep = cu_bs[BS_KEYS[win]]
    r, ci_lo, ci_hi = ep["ratio"], ep["ratio_ci_low"], ep["ratio_ci_high"]
    ax.errorbar(1.0, r, yerr=[[r - ci_lo], [ci_hi - r]],
                fmt="o", color=COLOR, ms=5, lw=1.2, capsize=3)

    ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.set_xlabel(r"$\alpha$  preference perturbation")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(0.3, 1.3)
    ax.set_title(WINDOW_TITLES[win], fontsize=10, pad=4)

axes[0].set_ylabel("synergy ratio")

for ext in ["pdf", "png"]:
    out = PAPER / f"figure_2_temporal.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved -> {out}")
plt.close(fig)
