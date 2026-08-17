"""
figure 4: GovSim perturbation sweep, two-panel (control / mismatch).
5 persona lines per panel; persona_4 (outsider) in red, others in grey.
data: govsim_pid/results/perturbation_ratios.json
output: paper/figures/figure_4_govsim.{pdf,png}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE  = Path(__file__).parent
PAPER = HERE.parent.parent / "paper" / "figures"
PAPER.mkdir(parents=True, exist_ok=True)

pr = json.load(open(HERE / "results/perturbation_ratios.json"))

PERSONAS  = ["persona_0", "persona_1", "persona_2", "persona_3", "persona_4"]
OUTSIDER  = "persona_4"
C_OUT     = "#d6604d"    # red, outsider
C_ALN     = "#666666"    # grey, aligned
ALPHAS    = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COND_TITLES = {"control": "Control", "mismatch": "Mismatch"}


def norm_sweep(data, persona):
    """normalized syn_sweep for persona; returns (alphas, ratios)."""
    sw  = data["syn_sweep"]
    s0  = sw["0.0"]
    if s0 is None or s0 < 1e-9:
        return None, None
    xs  = sorted(sw.keys(), key=float)
    ys  = [sw[a] / s0 if sw[a] is not None else np.nan for a in xs]
    return np.array([float(a) for a in xs]), np.array(ys)


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

for ax, cond in zip(axes, ["control", "mismatch"]):
    cond_data = pr.get(cond, {})
    for persona in PERSONAS:
        d = cond_data.get(persona)
        if d is None:
            continue
        xs, ys = norm_sweep(d, persona)
        if xs is None:
            continue
        is_p4  = persona == OUTSIDER
        color  = C_OUT if is_p4 else C_ALN
        zorder = 3     if is_p4 else 2
        lw     = 2.0   if is_p4 else 1.2
        ax.plot(xs, ys, color=color, lw=lw, zorder=zorder)

        # CI error bar at alpha=1 for persona_4 only
        if is_p4:
            d   = cond_data[persona]
            r, lo, hi = d["ratio"], d["ci_lo"], d["ci_hi"]
            ax.errorbar(1.0, r, yerr=[[r - lo], [hi - r]],
                        fmt="none", color=C_OUT, lw=1.4, capsize=4, zorder=4)

    ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.set_xlabel(r"$\alpha$  preference perturbation")
    ax.set_xlim(-0.02, 1.08)
    ax.set_title(COND_TITLES[cond], fontsize=10, pad=4)

axes[0].set_ylabel("synergy ratio")

# y-axis: accommodate persona_4 mismatch ci_hi (~5.51)
ci_hi_max = pr["mismatch"][OUTSIDER]["ci_hi"]
axes[0].set_ylim(bottom=0, top=ci_hi_max * 1.05)

# legend on mismatch panel where persona_4 visually stands out
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=C_OUT, lw=2.0, label="outsider"),
    Line2D([0], [0], color=C_ALN, lw=1.2, label="residents"),
]
axes[1].legend(handles=handles, frameon=False, loc="upper left")

for ext in ["pdf", "png"]:
    out = PAPER / f"figure_4_govsim.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved -> {out}")
plt.close(fig)
