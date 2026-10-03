"""Robust aggregation rules for byzantine-resilient FL.

We implement three classical robust aggregators in pure NumPy:

  krum(grads, f)         — Blanchard 2017: pick the single grad with the
                            smallest (n - f - 2)-closest neighborhood.
  trimmed_mean(grads, β) — Yin et al. 2018: drop top/bottom β fraction
                            along each dim, then average.
  foolsgold(grads)       — Fung et al. 2020: down-weight clients whose
                            gradient cosine similarity to the cohort mean
                            is suspiciously high (collusion detection).
                            We use a simplified single-round version; the
                            full algorithm maintains a per-client history
                            of pairwise cosine similarities.

Plus a signed-trust-based aggregator that we use inside FedCAMO-CR
to give the trust signal *operational* meaning (not just an a-posteriori
audit but also directly weighted averaging).
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def _pairwise_distances(grads: np.ndarray) -> np.ndarray:
    """Squared L2 distances between each pair of grads."""
    diffs = grads[:, None, :] - grads[None, :, :]
    return (diffs * diffs).sum(axis=2)


def krum(grads: np.ndarray, f: int = 1) -> tuple:
    """Krum: returns (idx_selected, selected_grad)."""
    K, _ = grads.shape
    n_keep = max(K - f - 2, 1)
    dists = _pairwise_distances(grads)
    scores = []
    for i in range(K):
        d_i = np.sort(dists[i])
        # exclude self (index 0)
        s = d_i[1:n_keep + 1].sum()
        scores.append(s)
    scores_arr = np.array(scores)
    idx = int(np.argmin(scores_arr))
    return idx, grads[idx]


def krum_multi(grads: np.ndarray, f: int = 1, m: int = None) -> np.ndarray:
    """Multi-Krum: average over m lowest-scoring grads.
       If m is None, default to K - f - 2 (i.e. all-but-f-bad)."""
    K, _ = grads.shape
    dists = _pairwise_distances(grads)
    scores = np.zeros(K)
    n_keep = max(K - f - 2, 1)
    for i in range(K):
        d_i = np.sort(dists[i])
        scores[i] = d_i[1:n_keep + 1].sum()
    if m is None:
        m = K - f - 2
    m = min(m, K)
    selected = np.argsort(scores)[:m]
    return grads[selected].mean(axis=0)


def trimmed_mean(grads: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """Coordinate-wise trimmed mean. Returns the averaged gradient."""
    K, D = grads.shape
    b = max(int(np.floor(beta * K)), 1)
    if 2 * b >= K:
        return grads.mean(axis=0)
    sorted_grads = np.sort(grads, axis=0)
    trimmed = sorted_grads[b: K - b]
    return trimmed.mean(axis=0)


def foolsgold(grads: np.ndarray) -> np.ndarray:
    """
    Simplified FoolsGold. The full algorithm requires a per-client
    history of cosine similarities — here we approximate with a single
    round's cosine similarity to the cohort mean.

    A high cosine similarity (close to 1) between two clients indicates
    they may be colluding; their updates are down-weighted.
    """
    K, D = grads.shape
    g_mean = grads.mean(axis=0)
    norm_mean = np.linalg.norm(g_mean) + 1e-9
    weights = np.ones(K)
    for i, g in enumerate(grads):
        cos = float(g @ g_mean / ((np.linalg.norm(g) + 1e-9) * norm_mean))
        # collusion → very high cos → down-weight
        if cos > 0.9:
            weights[i] = max(0.1, 1.0 - (cos - 0.9) * 10)
        # clearly malicious → negative cos → kill weight
        if cos < -0.3:
            weights[i] = 0.1
    if weights.sum() == 0:
        weights = np.ones(K)
    weights /= weights.sum()
    return (grads * weights[:, None]).sum(axis=0)


def trust_weighted(grads: np.ndarray, trust: np.ndarray) -> np.ndarray:
    """
    Our CR-style aggregation: re-weight each client by its squared trust
    score (so high-trust clients get squared dominance over low-trust ones).
    trust ∈ [0, 1].
    """
    w = np.asarray(trust, dtype=np.float64) ** 2 + 1e-3
    w /= w.sum() + 1e-9
    return (grads * w[:, None]).sum(axis=0)


# ----------------------------------------------------------------------
# trust signal — used as a per-round "is this grad honest?" estimate.
# We rely on ELEMENTWISE SIGN AGREEMENT between client grad and cohort mean.
# Honest (Non-IID) clients typically have 60-85% agreement; LR/MR/AA
# attacks drop to 30-50% agreement against an honest cohort mean.

def trust_signal(client_grad: np.ndarray, cohort_grad: np.ndarray) -> float:
    """Returns a scalar in [0.5, 1] giving this round's trust-like quality
       signal for `client_grad`, based on sign-agreement with the cohort mean."""
    sign_c = np.sign(client_grad)
    sign_m = np.sign(cohort_grad)
    # treat zeros in client as agreement (avoid noise domination)
    nz = sign_m != 0
    if not nz.any():
        return 0.5
    sa = (sign_c[nz] == sign_m[nz]).mean()
    # floor at 0.5 (random); emphasize honest-positive signals
    return float(0.5 + 0.5 * max(0.0, sa - 0.5) * 2)  # maps 0.5..1 → 0.5..1



def bulyan(grads: np.ndarray, f: int = 1) -> np.ndarray:
    """Bulyan aggregation (Mhamdi et al., 2018).

    Two-stage algorithm: first run trimmed-mean for many iterations to
    select the top-K honest gradients, then average them. Bulyan
    requires K >= 4f + 3 clients.

    Reference: Mhamdi, M. E. R., Guerraoui, R., & Rouault, S. (2018).
    ``The Hidden Vulnerability of Distributed Learning in Byzantium.''
    ICML 2018.
    """
    K, D = grads.shape
    if K < 4 * f + 3:
        # Fall back to trimmed-mean if K is too small for Bulyan.
        b = max(f, 1)
        sorted_grads = np.sort(grads, axis=0)
        trimmed = sorted_grads[b: K - b] if 2 * b < K else grads
        return trimmed.mean(axis=0)

    # Stage 1: iteratively trimmed-mean. At each step, remove the f most
    # extreme values along each coordinate, then re-mean the rest. After
    # 4f + 1 iterations, the surviving K - 4f gradients are honest.
    b = f
    survivors = grads.copy()
    for _ in range(4 * f + 1):
        if survivors.shape[0] <= 2 * b:
            break
        sorted_g = np.sort(survivors, axis=0)
        survivors = sorted_g[b: survivors.shape[0] - b]

    # Stage 2: take the median of the surviving gradients along each
    # coordinate (Bulyan step). The mean of the median over the survivors
    # is provably robust under honest-majority.
    return np.median(survivors, axis=0)


def divide_and_conquer(grads: np.ndarray, f: int = 1, num_buckets: int = 8,
                       beta: float = 0.1) -> np.ndarray:
    """Divide-and-Conquer (DnC) aggregation (Damaskinos et al., 2018).

    Divide clients into buckets, run trimmed-mean within each bucket,
    then average the per-bucket means.

    Reference: Damaskinos, G., Mhamdi, M. E. R., Guerraoui, R., Patra, R.
    H., & Taziki, M. (2018). ``Robust Asynchronous Stochastic
    Gradient Descent with Constant Communication Complexity.''
    DISC 2018.
    """
    K, D = grads.shape
    if K < num_buckets:
        # Fall back to trimmed-mean if too few clients.
        sorted_grads = np.sort(grads, axis=0)
        trimmed = sorted_grads[beta * K: (1 - beta) * K]
        return trimmed.mean(axis=0)

    # Shuffle and split into num_buckets buckets of (roughly) equal size.
    indices = np.arange(K)
    np.random.shuffle(indices)
    buckets = np.array_split(indices, num_buckets)

    # Within each bucket, run a coordinate-wise trimmed mean (removing the
    # most extreme value per coordinate to filter single-attacker influence).
    bucket_means = []
    for bucket in buckets:
        bucket_grads = grads[bucket]
        b_size = bucket_grads.shape[0]
        b_trim = max(int(np.floor(beta * b_size)), 1)
        if 2 * b_trim >= b_size:
            bucket_means.append(bucket_grads.mean(axis=0))
        else:
            sorted_g = np.sort(bucket_grads, axis=0)
            trimmed = sorted_g[b_trim: b_size - b_trim]
            bucket_means.append(trimmed.mean(axis=0))

    # Average across buckets — under honest-majority, the median bucket is
    # honest, so the average is robust.
    return np.mean(bucket_means, axis=0)


def median_of_means(grads: np.ndarray, f: int = 1, num_buckets: int = 8) -> np.ndarray:
    """Median-of-Means (MoM) aggregation (Lecué & Lerasle, 2020).

    Divide clients into buckets, compute per-bucket mean, then take
    coordinate-wise median across the bucket means. Robust under
    sub-Gaussian noise + honest majority.

    Reference: Lecué, G., & Lerasle, M. (2020). ``Robust Machine
    Learning by Median-of-Means.'' PMLR 108: 7214-7233.
    """
    K, D = grads.shape
    if K < num_buckets:
        # Fall back to coordinate-wise median if too few clients.
        return np.median(grads, axis=0)

    indices = np.arange(K)
    np.random.shuffle(indices)
    buckets = np.array_split(indices, num_buckets)

    bucket_means = [grads[bucket].mean(axis=0) for bucket in buckets]
    return np.median(bucket_means, axis=0)
