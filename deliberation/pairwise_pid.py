"""
shared binning helpers for the pairwise pid pipeline.

sources per agent: self score of the current proposal, and payoff sensitivity
(absolute round-over-round score change / score range). target for a pair:
summed score change of the agents outside the pair, quantile-binned.

fit bin edges once at alpha=0 and reuse them across the sweep; refitting
changes bin meanings and invalidates comparisons.
"""

import json
from pathlib import Path

import numpy as np

FEAT_DIR = Path(__file__).parent / "features"

N_BINS   = 4   # quantile bins per source feature dimension
N_TARGET = 3   # terciles for target delta


def load_condition(condition):
    games = []
    for p in sorted((FEAT_DIR / condition).glob("*.json")):
        with open(p) as f:
            g = json.load(f)
        if g["n_rounds_parsed"] < 3:
            continue
        games.append(g)
    return games


def fit_quantile_bins(values, n_bins=N_BINS):
    values = np.array(values)
    edges  = np.unique(np.percentile(values, np.linspace(0, 100, n_bins + 1)))
    return edges


def apply_bins(value, edges):
    idx = np.searchsorted(edges[1:-1], value, side='right')
    return int(np.clip(idx, 0, len(edges) - 2))
