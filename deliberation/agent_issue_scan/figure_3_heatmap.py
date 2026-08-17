"""
figure 3: composite cross-column layout.
left: 6x5 agent-by-issue gap heatmap (diverging RdBu_r).
right: 4-bar PID atom fractional response for paired-mismatch condition.
data: scan.json (heatmap), collusion/results/cluster_bootstrap.json (atoms)
output: paper/figures/figure_3_heatmap.{pdf,png}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE  = Path(__file__).parent
BASE  = HERE.parent
PAPER = HERE.parent.parent / "paper" / "figures"
PAPER.mkdir(parents=True, exist_ok=True)

scan = json.load(open(HERE / "results/scan.json"))
bs   = json.load(open(BASE / "collusion/results/cluster_bootstrap.json"))

# heatmap data
AGENT_ORDER  = ["SportCo", "union", "mayor", "enviroment", "other_cities", "DoT"]
AGENT_LABELS = [str(i + 1) for i in range(len(AGENT_ORDER))]  # generic agent ids
ISSUE_LABELS = scan["issue_names"]
N_AGENTS, N_ISSUES = len(AGENT_ORDER), len(ISSUE_LABELS)

gap = scan["gap"]
Z   = np.zeros((N_AGENTS, N_ISSUES))
for i, agent in enumerate(AGENT_ORDER):
    for j in range(N_ISSUES):
        val = gap.get(agent, {}).get(str(j), 0.0)
        Z[i, j] = val if val is not None else 0.0

D_IDX        = ISSUE_LABELS.index("D")
TRUE_ROWS    = [AGENT_ORDER.index("union"), AGENT_ORDER.index("SportCo")]  # ground-truth misaligned

# atom data
atom_data  = bs["atom_decomposition"]
ATOM_KEYS  = ["Red", "Un1", "Un2", "Syn"]
ATOM_LABELS = ["Redundancy", "Unique$_1$", "Unique$_2$", "Synergy"]
ATOM_COLORS = ["#666666", "#666666", "#666666", "#d6604d"]

fracs  = [atom_data[k]["delta"] for k in ATOM_KEYS]

plt.rcParams.update({
    "font.family":       "serif",
    "mathtext.fontset":  "cm",
    "font.size":         10,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, (ax_hm, ax_at) = plt.subplots(
    1, 2, figsize=(10, 3.5), layout="constrained",
    gridspec_kw={"width_ratios": [3, 2], "wspace": 0.15},
)

# left: heatmap
vmax = np.abs(Z).max()
im   = ax_hm.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

ax_hm.set_xticks(range(N_ISSUES))
ax_hm.set_xticklabels([str(j + 1) for j in range(N_ISSUES)])  # generic issue ids
ax_hm.set_yticks(range(N_AGENTS))
ax_hm.set_yticklabels(AGENT_LABELS)
ax_hm.set_xlabel("Issue")
ax_hm.set_ylabel("Agent")
ax_hm.set_title("Agent x issue perturbation gap", fontsize=10, pad=4)
ax_hm.set_box_aspect(1)

for row in TRUE_ROWS:
    ax_hm.text(D_IDX, row, "*", ha="center", va="center",
               fontsize=12, fontweight="bold", color="white")

cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
cbar.set_label(r"$\Delta r$ (mismatch $-$ control)", fontsize=8)
cbar.ax.tick_params(labelsize=8)

# right: atom bar chart
xs = np.arange(len(ATOM_KEYS))
ax_at.bar(xs, fracs, color=ATOM_COLORS, width=0.55, zorder=3)

ax_at.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.7)
ax_at.set_xticks(xs)
ax_at.set_xticklabels(ATOM_LABELS, fontsize=8)
ax_at.set_ylabel("delta atom value (bits)")
ax_at.set_ylim(-0.06, 0.015)
ax_at.set_title("PID atom decomposition\n(paired mismatch)", fontsize=10, pad=4)

for ext in ["pdf", "png"]:
    out = PAPER / f"figure_3_heatmap.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved -> {out}")
plt.close(fig)
