"""
perturbation analysis on the govsim fishing commons. sources per agent:
harvest over an interpolated sustainable-to-extractive target, and its
round-over-round change. only the drift agent's features depend on alpha;
the pair target (non-pair harvest delta) does not. cluster bootstrap on
the alpha=0 vs alpha=1 synergy ratio.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "deliberation"))

from pid_numpy import imin_numpy
from pairwise_pid import fit_quantile_bins, apply_bins, N_BINS, N_TARGET

RESULTS_DIR  = Path(__file__).parent / "results"
_LOG_BASE    = Path(__file__).parent / "logs"
LOG_GLOB     = {
    "control":  str(_LOG_BASE / "stage3_control_*/log_env.json"),
    "mismatch": str(_LOG_BASE / "stage3_mismatch_*/log_env.json"),
}

AGENTS            = ["persona_0", "persona_1", "persona_2", "persona_3", "persona_4"]
N_BOOT            = 1000
MIN_TRIPS         = 50
EXTRACTIVE_TARGET = 30.0    # 3x sustainable_share (N=5, pool=100 -> sus=10)
ALPHAS            = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# data loading

def load_games(condition):
    import glob
    games = []
    for path in sorted(glob.glob(LOG_GLOB[condition])):
        rows = json.load(open(path))
        harvest = [r for r in rows if r.get("action") == "harvesting"]
        if not harvest:
            continue
        games.append({"path": path, "rows": harvest})
    return games


# per-round states

def compute_states(game):
    """
    returns list of per-round state dicts, one per round where all present
    agents have harvested. rounds 0-2: N=4 (no persona_4). rounds 3+: N=5.
    """
    by_round = {}
    for r in game["rows"]:
        rnd = r["round"]
        if rnd not in by_round:
            by_round[rnd] = {}
        by_round[rnd][r["agent_id"]] = {
            "harvest":    r["resource_collected"],
            "pool_before": r["resource_in_pool_before_harvesting"],
        }

    states = []
    for rnd in sorted(by_round):
        agents_present = sorted(by_round[rnd].keys())
        n = len(agents_present)
        sustainable_share = {
            a: max((by_round[rnd][a]["pool_before"] / 2.0) / n, 1.0)
            for a in agents_present
        }
        states.append({
            "round":   rnd,
            "agents":  agents_present,
            "harvest": {a: by_round[rnd][a]["harvest"] for a in agents_present},
            "pool_before": {a: by_round[rnd][a]["pool_before"] for a in agents_present},
            "sustainable_share": sustainable_share,
        })
    return states


# source features

def agent_features(state, agent, prev_state, alpha):
    """
    feat1: harvest / interpolated_target. ~1 when on target.
           aligned at alpha=0: ~0.5-0.8. outsider at alpha=0: ~2-3.
           aligned at alpha=1: ~0.2. outsider at alpha=1: ~1.0.
    feat2: |deltafeat1| round-over-round (sensitivity). first round: 0.
    """
    h   = state["harvest"][agent]
    sus = state["sustainable_share"][agent]
    tgt = (1 - alpha) * sus + alpha * EXTRACTIVE_TARGET
    score = h / max(tgt, 1e-9)

    if prev_state is not None and agent in prev_state["agents"]:
        h_prev   = prev_state["harvest"][agent]
        sus_prev = prev_state["sustainable_share"][agent]
        tgt_prev = (1 - alpha) * sus_prev + alpha * EXTRACTIVE_TARGET
        sensitivity = abs(score - h_prev / max(tgt_prev, 1e-9))
    else:
        sensitivity = 0.0

    return score, sensitivity


# edge fitting

def fit_edges_across_alphas(games, drift_agent, alphas=ALPHAS):
    """
    fit bin edges spanning the full alpha range.
    sources: collected at each alpha for drift_agent, at alpha=0 for others.
    target: non-pair harvest delta, alpha-independent, collected once.
    """
    hf_vals, sd_vals, delta_vals = [], [], []
    for g in games:
        states = compute_states(g)
        for idx in range(len(states) - 1):
            s, sn  = states[idx], states[idx + 1]
            prev_s = states[idx - 1] if idx > 0 else None
            common = [a for a in s["agents"] if a in sn["agents"]]
            for alpha in alphas:
                for a in common:
                    eff_alpha = alpha if a == drift_agent else 0.0
                    hf, sd = agent_features(s, a, prev_s, eff_alpha)
                    hf_vals.append(hf)
                    sd_vals.append(sd)
            total_h = sum(s["harvest"][a] for a in common)
            pool_t  = max(s["pool_before"].values())
            pool_t1 = max(sn["pool_before"].values())
            growth  = pool_t1 - (pool_t - total_h)
            for (j, k) in combinations(common, 2):
                non_pair_h = total_h - s["harvest"][j] - s["harvest"][k]
                delta_vals.append(-non_pair_h + growth)

    return {
        "hf":    fit_quantile_bins(hf_vals),
        "sd":    fit_quantile_bins(sd_vals),
        "delta": fit_quantile_bins(delta_vals, n_bins=N_TARGET),
    }


# triplet collection

def joint_bin(hf, sd, hf_edges, sd_edges):
    return apply_bins(hf, hf_edges) * N_BINS + apply_bins(sd, sd_edges)


def collect_triplets(games, edges, drift_agent, alpha):
    """
    pool all drift-pair triplets across games for a given drift_agent at alpha.
    only drift_agent's features use alpha; partners use alpha=0.
    target: non-pair harvest delta (alpha-independent).
    """
    hf_e, sd_e, delta_e = edges["hf"], edges["sd"], edges["delta"]
    triplets = []
    for g in games:
        states = compute_states(g)
        for idx in range(len(states) - 1):
            s, sn  = states[idx], states[idx + 1]
            prev_s = states[idx - 1] if idx > 0 else None
            common = [a for a in s["agents"] if a in sn["agents"]]
            if drift_agent not in common:
                continue
            for (j, k) in combinations(common, 2):
                if drift_agent not in (j, k):
                    continue
                total_h    = sum(s["harvest"][a] for a in common)
                pool_t     = max(s["pool_before"].values())
                pool_t1    = max(sn["pool_before"].values())
                growth     = pool_t1 - (pool_t - total_h)
                non_pair_h = total_h - s["harvest"][j] - s["harvest"][k]
                delta      = -non_pair_h + growth

                hfj, sdj = agent_features(s, j, prev_s, alpha if j == drift_agent else 0.0)
                hfk, sdk = agent_features(s, k, prev_s, alpha if k == drift_agent else 0.0)

                bj = joint_bin(hfj, sdj, hf_e, sd_e)
                bk = joint_bin(hfk, sdk, hf_e, sd_e)
                bd = apply_bins(delta, delta_e)
                triplets.append((bj, bk, bd))
    return triplets


# bootstrap

def per_game_triplets(games, edges, drift_agent, alpha):
    return [collect_triplets([g], edges, drift_agent, alpha) for g in games]


def syn_from_pool(trip_lists):
    pool = [t for tl in trip_lists for t in tl]
    if len(pool) < MIN_TRIPS:
        return float("nan")
    return imin_numpy(pool)["Syn"]


def bootstrap_ratio(per_game_a0, per_game_a1, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(per_game_a0)
    ratios, n_dropped = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s0  = syn_from_pool([per_game_a0[i] for i in idx])
        s1  = syn_from_pool([per_game_a1[i] for i in idx])
        if s0 > 1e-9 and not np.isnan(s0) and not np.isnan(s1):
            ratios.append(s1 / s0)
        else:
            n_dropped += 1
    if n_dropped > 0.2 * n_boot:
        print(f"    warning: {n_dropped}/{n_boot} bootstrap iterations dropped (unstable)")
    if not ratios:
        return float("nan"), float("nan"), float("nan")
    ratios = np.array(ratios)
    return float(np.mean(ratios)), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


# main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("loading games...")
    games_by_cond = {}
    for c in ["control", "mismatch"]:
        games_by_cond[c] = load_games(c)
        print(f"  {c}: {len(games_by_cond[c])} games")

    results = {}
    print(f"\n{'condition':<12}  {'agent':<12}  {'n_trips_a0':>10}  {'syn_a0':>8}  {'syn_a1':>8}  {'ratio':>7}  {'ci_lo':>7}  {'ci_hi':>7}")

    for condition in ["control", "mismatch"]:
        games = games_by_cond[condition]
        results[condition] = {}

        for agent in AGENTS:
            edges = fit_edges_across_alphas(games, agent, alphas=ALPHAS)

            pg_a0 = per_game_triplets(games, edges, agent, 0.0)
            n_trips = sum(len(t) for t in pg_a0)
            if n_trips < MIN_TRIPS:
                print(f"  {condition:<12}  {agent:<12}  {n_trips:>10}  insufficient data")
                results[condition][agent] = None
                continue

            sweep = {a: syn_from_pool(per_game_triplets(games, edges, agent, a)) for a in ALPHAS}

            pg_a1 = per_game_triplets(games, edges, agent, 1.0)
            point_ratio = sweep[1.0] / sweep[0.0] if sweep[0.0] > 1e-9 else float("nan")
            _, ci_lo, ci_hi = bootstrap_ratio(pg_a0, pg_a1)
            n_games = sum(1 for t in pg_a0 if len(t) > 0)

            sweep_str = "  ".join(f"a{a:.1f}={sweep[a]:.3f}" for a in ALPHAS)
            print(f"  {condition:<12}  {agent:<12}  {n_trips:>10}  ratio={point_ratio:.3f} [{ci_lo:.3f},{ci_hi:.3f}]  {sweep_str}")
            results[condition][agent] = {
                "ratio": point_ratio, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "n_games": n_games, "n_trips_a0": n_trips,
                "syn_sweep": {str(a): sweep[a] for a in ALPHAS},
            }

    def _clean(obj):
        if isinstance(obj, float) and np.isnan(obj): return None
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        return obj

    out = RESULTS_DIR / "perturbation_ratios.json"
    with open(out, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
