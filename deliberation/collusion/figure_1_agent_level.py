"""
figure 1: perturbation sweep curves per condition plus endpoint dot plot
with cluster-bootstrap cis. reads collusion/results/cluster_bootstrap.json,
writes paper/figures/figure_1_agent_level.{pdf,png}.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE  = Path(__file__).parent
PAPER = HERE.parent.parent / "paper" / "figures"
PAPER.mkdir(parents=True, exist_ok=True)

bs   = json.load(open(HERE / "results/cluster_bootstrap.json"))
traj = bs["trajectory"]   # keyed by cond -> alpha_str -> mean_syn

COLORS = {"control": "#666666", "single": "#2166ac", "collusion": "#d6604d"}
LABELS = {"control": "control", "single": "single-mismatch", "collusion": "paired-mismatch"}
CONDS  = ["control", "collusion", "single"]


def sweep_mean(cond):
    """normalized mean-Syn per alpha, same estimator as bootstrap ratio."""
    t      = traj[cond]
    alphas = sorted(t.keys(), key=float)
    s0     = t["0.0"]
    xs, ys = [], []
    for a in alphas:
        xs.append(float(a))
        ys.append(t[a] / s0 if s0 and s0 > 1e-9 else float("nan"))
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

fig, (ax_l, ax_r) = plt.subplots(
    1, 2, figsize=(4.2, 2.5), layout="constrained", sharey=True,
    gridspec_kw={"width_ratios": [3, 0.7], "wspace": 0.05},
)

# left: sweep curves
for cond in CONDS:
    xs, ys = sweep_mean(cond)
    ax_l.plot(xs, ys, color=COLORS[cond], lw=1.8, label=LABELS[cond])

ax_l.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
ax_l.set_xlabel(r"$\alpha$  preference perturbation")
ax_l.set_ylabel("synergy ratio")
ax_l.set_xlim(-0.02, 1.02)
ax_l.set_ylim(0.4, 1.15)
ax_l.legend(frameon=False, loc="lower left")

# right: endpoint cluster cis, same y-axis
x_pos = {"control": 0, "collusion": 1, "single": 2}
for cond in CONDS:
    m = bs["metrics"][cond]
    r, lo, hi = m["ratio"], m["ratio_ci_low"], m["ratio_ci_high"]
    ax_r.errorbar(x_pos[cond], r, yerr=[[r - lo], [hi - r]],
                  fmt="o", color=COLORS[cond], ms=5, lw=1.4, capsize=4)

ax_r.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
ax_r.set_xticks([])
ax_r.set_xlim(-0.5, 2.5)
ax_r.tick_params(axis="y", left=False, labelleft=False)
ax_r.spines["left"].set_visible(False)
ax_r.set_xlabel("endpoint\n95% CI", fontsize=9)


for ext in ["pdf", "png"]:
    out = PAPER / f"figure_1_agent_level.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved -> {out}")
plt.close(fig)
