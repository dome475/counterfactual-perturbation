# Third-party benchmarks

The experiments build on two external benchmarks that are **not vendored** in this
repository. All analysis in this repository runs offline from the included processed
data and does not require them; they are only needed to regenerate raw game data
from scratch.

| Benchmark | Used for | Upstream | Pinned commit |
|---|---|---|---|
| LLM-Deliberation (Abdelnabi et al., NeurIPS 2024) | multi-issue negotiation experiments (`deliberation/`) | https://github.com/S-Abdelnabi/LLM-Deliberation | `78e11e0` |
| GovSim (Piatti et al., 2024) | common-pool resource probe (`govsim/`) | https://github.com/giorgiopiatti/GovSim | `1d11adf` |

Notes:
- `deliberation/collusion/run_with_corruption.py` is a modified game runner meant to be
  placed inside the LLM-Deliberation repository; it injects the score-sheet
  modifications for the mismatch conditions.
- The GovSim games were run on a SLURM cluster against the pinned upstream commit;
  `govsim/govsim_pid/analyze.py` processes the resulting episode logs.
