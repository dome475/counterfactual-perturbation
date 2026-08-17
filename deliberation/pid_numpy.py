"""fast Williams-Beer I_min in numpy + bootstrap CI."""

import numpy as np

N_BOOT = 500


def imin_numpy(triplets):
    """
    Williams-Beer I_min, pure numpy.
    triplets: list/array of (s1_bin, s2_bin, t_bin).
    returns dict {Red, Un1, Un2, Syn, MI}.
    """
    arr  = np.array(triplets, dtype=int)
    s1, s2, t = arr[:, 0], arr[:, 1], arr[:, 2]
    n    = len(arr)
    eps  = 1e-300

    n_s1 = int(s1.max()) + 1
    n_s2 = int(s2.max()) + 1
    n_t  = int(t.max()) + 1

    idx    = s1 * n_s2 * n_t + s2 * n_t + t
    counts = np.bincount(idx, minlength=n_s1 * n_s2 * n_t).reshape(n_s1, n_s2, n_t)
    p      = counts / n

    p_s1   = p.sum(axis=(1, 2))
    p_s2   = p.sum(axis=(0, 2))
    p_tm   = p.sum(axis=(0, 1))
    p_s1s2 = p.sum(axis=2)
    p_s1t  = p.sum(axis=1)
    p_s2t  = p.sum(axis=0)

    p_s1_gt = p_s1t / (p_tm[None, :] + eps)
    p_s2_gt = p_s2t / (p_tm[None, :] + eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        lr1 = np.where(p_s1_gt > eps, np.log2(p_s1_gt / (p_s1[:, None] + eps)), 0.0)
        lr2 = np.where(p_s2_gt > eps, np.log2(p_s2_gt / (p_s2[:, None] + eps)), 0.0)

    i_spec_s1 = (p_s1_gt * lr1).sum(axis=0)
    i_spec_s2 = (p_s2_gt * lr2).sum(axis=0)
    red = float(np.sum(p_tm * np.minimum(i_spec_s1, i_spec_s2)))

    with np.errstate(divide="ignore", invalid="ignore"):
        mi1  = np.where(p_s1t > eps,
                        p_s1t * np.log2(p_s1t / (p_s1[:, None] * p_tm[None, :] + eps)), 0.0).sum()
        mi2  = np.where(p_s2t > eps,
                        p_s2t * np.log2(p_s2t / (p_s2[:, None] * p_tm[None, :] + eps)), 0.0).sum()
        mi12 = np.where(p > eps,
                        p * np.log2(p / (p_s1s2[:, :, None] * p_tm[None, None, :] + eps)), 0.0).sum()

    un1 = float(mi1) - red
    un2 = float(mi2) - red
    syn = float(mi12) - float(mi1) - float(mi2) + red

    if syn < -1e-9:
        print(f"  [warn] syn={syn:.4f} < 0 before clamp "
              f"(mi12={float(mi12):.4f} mi1={float(mi1):.4f} mi2={float(mi2):.4f} red={red:.4f})")

    return {
        "Red": max(0.0, red),
        "Un1": max(0.0, un1),
        "Un2": max(0.0, un2),
        "Syn": max(0.0, syn),
        "MI":  max(0.0, float(mi12)),   # joint I(S1,S2;T), not sum of clamped atoms
    }


def compute_o_information(X, n_agents):
    """
    O-information with Miller-Madow bias correction.
    X: (n_samples, n_agents) binary int array
    returns omega, TC, DTC
    """
    n_samples = X.shape[0]

    def H_mm(cols):
        if len(cols) == 1:
            vals = X[:, cols[0]]
            _, counts = np.unique(vals, return_counts=True)
        else:
            sub = X[:, cols]
            idx = sum(sub[:, j] * (2 ** j) for j in range(len(cols)))
            _, counts = np.unique(idx, return_counts=True)
        p = counts / counts.sum()
        h = -np.sum(p * np.log2(p + 1e-300))
        return h + (len(counts) - 1) / (2 * n_samples * np.log(2))

    all_cols  = list(range(n_agents))
    H_joint   = H_mm(all_cols)
    H_i       = [H_mm([i]) for i in all_cols]
    H_minus_i = [H_mm([j for j in all_cols if j != i]) for i in all_cols]

    TC    = sum(H_i) - H_joint
    DTC   = sum(H_minus_i) - (n_agents - 1) * H_joint
    return TC - DTC, TC, DTC


def bootstrap_ci(atom_list, key, n_boot=N_BOOT, seed=42):
    """bootstrap 95% CI for one atom key across a list of atom dicts."""
    arr = np.array([a[key] for a in atom_list])
    if len(arr) < 2:
        m = float(arr.mean())
        return m, m, m
    rng  = np.random.default_rng(seed)
    boot = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
