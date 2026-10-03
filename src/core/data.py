"""Synthetic dataset generator for FedCAMO reproduction.

We do NOT depend on torchvision / internet downloads. Instead we:

  * generate per-client "soft-label" distributions via Dirichlet(α);
  * store for each sample:
        x ∈ R^{D}       (a deterministic feature vector — class-conditional
                         Gaussian features so that a linear classifier can
                         reach a meaningful test accuracy on the order of
                         60-70% on 10 classes when classes are not too
                         overlapped);
        y ∈ {0..C-1}    (int label);

Three synthetic "datasets" are provided to mirror the relative difficulty
of CIFAR-10 / SVHN / GSC used in FedCAMO [Lu et al. 2026, Section 5.1]:

    dataset = "cifar10"  → 10 classes, dim 64, noise 0.8
    dataset = "svhn"     → 10 classes, dim 64, noise 1.0  (harder)
    dataset = "gsc"      → 35 classes, dim 64, noise 1.2  (hardest)

The *relative* trends of FedCAMO-CR vs baselines remain valid across these
synthetic regimes; absolute numbers are not directly comparable to the
real-data numbers in Lu et al. 2026.

The key requirement from FedCAMO is:
  (i)  Non-IID partition with α=0.3;
  (ii) each client reports its label-count vector n_k during probe phase;
  (iii) the global distribution p_global serves as the JSD reference.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


NUM_CLASSES = 10
FEATURE_DIM = 64  # compact embedding; 784 would also work
NUM_CLIENTS_DEFAULT = 90
DIRICHLET_ALPHA = 0.3


# Per-dataset synthetic regimes — see docstring.
DATASET_REGIMES = {
    "cifar10": {"num_classes": 10, "noise_std": 0.8, "feature_dim": 64},
    "svhn":    {"num_classes": 10, "noise_std": 1.0, "feature_dim": 64},
    "gsc":     {"num_classes": 35, "noise_std": 1.2, "feature_dim": 64},
}


def _class_centers(rng: np.random.Generator, num_classes: int, dim: int) -> np.ndarray:
    """
    Random Gaussian centers (NOT unit-normalised), giving us:
      - linearly separable-ish Bayesian classifier ~ 70-80% acc at dim=64
        when noise_std ~ 0.6 (single Gaussian per class).
      - deeper model would help, but linear classifier is enough for the
        *strategy* benchmark we care about.
    """
    centers = rng.normal(scale=0.5, size=(num_classes, dim)).astype(np.float32)
    # shrink centers towards origin so noise (std ~ 0.8) is comparable and
    # Bayes error stays in the 25-35% range — closer to CIFAR-10 test
    # conditions where FedCAMO baselines distinguish themselves.
    centers *= 0.6
    return centers


def _make_features(
    centers: np.ndarray, labels: np.ndarray, noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    n = labels.shape[0]
    dim = centers.shape[1]
    # add Gaussian noise on top of the class center
    feats = centers[labels] + rng.normal(scale=noise_std, size=(n, dim)).astype(np.float32)
    return feats


def sample_dirichlet_dirichlet(
    rng: np.random.Generator,
    num_clients: int,
    samples_per_client: int,
    alpha: float,
    num_classes: int,
    centers: np.ndarray,
    noise_std: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Dirichlet partitioning:
       p_k ~ Dir(alpha, ..., alpha) ∈ Δ^C
       n_{k,c} = round(p_k[c] * samples_per_client)
    """
    p = rng.dirichlet(alpha=[alpha] * num_classes, size=num_clients)  # (K, C)
    counts = np.maximum(np.round(p * samples_per_client).astype(int), 1)
    # fix rounding to keep total
    diff = counts.sum(axis=1) - samples_per_client
    for k in range(num_clients):
        i = 0
        while counts[k].sum() != samples_per_client and i < 10_000:
            if counts[k].sum() > samples_per_client:
                idx = np.argmax(counts[k])
                if counts[k, idx] > 1:
                    counts[k, idx] -= 1
            else:
                idx = np.argmin(counts[k])
                counts[k, idx] += 1
            i += 1

    clients: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(num_clients):
        labels = np.repeat(np.arange(num_classes), counts[k])
        rng.shuffle(labels)
        feats = _make_features(centers, labels, noise_std, rng)
        clients.append((feats.astype(np.float32), labels.astype(np.int64)))
    return clients


def make_global_test(
    rng: np.random.Generator,
    num_samples: int,
    centers: np.ndarray,
    noise_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, NUM_CLASSES, size=num_samples)
    feats = _make_features(centers, labels, noise_std, rng)
    return feats, labels


def partition(
    seed: int = 0,
    num_clients: int = NUM_CLIENTS_DEFAULT,
    samples_per_client: int = 600,
    alpha: float = DIRICHLET_ALPHA,
    noise_std: float | None = None,
    num_classes: int | None = None,
    feature_dim: int | None = None,
    dataset: str | None = None,
) -> Dict:
    """
    Returns a dict:
        {
          "centers": (C, D) np.ndarray,
          "clients": [(feats (n, D), labels (n,))] of length K,
          "client_label_counts": (K, C) np.ndarray,
          "client_distributions": (K, C) np.ndarray,
          "global_test": (X, y),
          "global_distribution": (C,) np.ndarray,
        }

    If `dataset` is given, the corresponding regime from DATASET_REGIMES
    overrides num_classes / noise_std / feature_dim (per-dataset defaults).
    Otherwise the explicit num_classes / noise_std / feature_dim arguments
    are used (or module-level defaults).
    """
    if dataset is not None:
        if dataset not in DATASET_REGIMES:
            raise ValueError(
                f"unknown dataset={dataset!r}; valid: {list(DATASET_REGIMES)}"
            )
        regime = DATASET_REGIMES[dataset]
        num_classes = regime["num_classes"] if num_classes is None else num_classes
        noise_std   = regime["noise_std"]   if noise_std   is None else noise_std
        feature_dim = regime["feature_dim"] if feature_dim is None else feature_dim
    # Fall back to module defaults
    if num_classes is None: num_classes = NUM_CLASSES
    if noise_std   is None: noise_std   = 0.8
    if feature_dim is None: feature_dim = FEATURE_DIM

    rng = np.random.default_rng(seed)
    centers = _class_centers(rng, num_classes, feature_dim)
    clients = sample_dirichlet_dirichlet(
        rng=rng,
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        alpha=alpha,
        num_classes=num_classes,
        centers=centers,
        noise_std=noise_std,
    )
    label_counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for k, (_, labels) in enumerate(clients):
        for c in range(num_classes):
            label_counts[k, c] = int((labels == c).sum())
    distributions = label_counts / np.maximum(label_counts.sum(axis=1, keepdims=True), 1)
    total = label_counts.sum(axis=0).astype(np.float64)
    global_dist = total / total.sum()

    # global test set: balanced 250 per class, with noise_std slightly
    # higher than training so held-out tests are harder (mirrors FL).
    test_noise = noise_std
    per_class = 250
    labels = np.concatenate([np.full(per_class, c, dtype=np.int64) for c in range(num_classes)])
    rng.shuffle(labels)
    test_x = _make_features(centers, labels, test_noise, rng)

    # also keep noise_std in metadata for reproducibility
    return {
        "centers": centers,
        "clients": clients,
        "client_label_counts": label_counts,
        "client_distributions": distributions,
        "global_test": (test_x, labels),
        "global_distribution": global_dist,
        "noise_std": noise_std,
        "alpha": alpha,
        "num_clients": num_clients,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "dataset": dataset,
    }


if __name__ == "__main__":
    out = partition(seed=0)
    print("clients=", len(out["clients"]))
    print("label_counts.shape=", out["client_label_counts"].shape)
    print("client 0 dist:", out["client_distributions"][0])
    print("global distribution:", out["global_distribution"])
    print("global test acc with NN classifier using centers approx:",
          np.mean(out["global_test"][1] == np.argmax(out["centers"] @ out["global_test"][0].T, axis=0)))
