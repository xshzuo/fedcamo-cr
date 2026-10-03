"""Simple selection strategies other than FedCAMO/FedCAMO-CR:

- random           (FedAvg-style)
- loss_only        (top-m by EMA(loss-drop); no carbon or diversity)
- coverage_only    (top-m by JSD coverage)
- carbon_only      (top-m by cheapest marginal carbon)
- poc              (Power-of-Choice: prob. proportional to recent loss-drop)
- oort_proxy       (combined utility/latency score)
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from src.fedcamo.selector import (
    diversity_utility, normal_entropy, coverage_gain, jsd, ClientHistory,
)
from src.core.carbon import marginal_carbon_g


def _ensure_hist(state: dict, idx: int) -> ClientHistory:
    if "histories" not in state:
        state["histories"] = {}
    if idx not in state["histories"]:
        state["histories"][idx] = ClientHistory()
    return state["histories"][idx]


def select_random(candidates: List[Dict], m: int, rng: np.random.Generator) -> List[Dict]:
    idx = rng.choice(len(candidates), size=min(m, len(candidates)), replace=False)
    return [candidates[int(i)] for i in idx]


def select_loss_only(
    candidates: List[Dict], m: int, state: dict, rng: np.random.Generator
) -> List[Dict]:
    scored = []
    for c in candidates:
        h = _ensure_hist(state, c["idx"])
        n_eff = max(h.n_recent, 1e-3)
        score = h.sigma_acc + 0.5 * np.sqrt(np.log1p(sum(x.n_selections for x in state["histories"].values())) / n_eff)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:m]]


def select_coverage_only(
    candidates: List[Dict], m: int, p_global: np.ndarray, candidates_dists: List[np.ndarray]
) -> List[Dict]:
    scored = []
    for c, p in zip(candidates, candidates_dists):
        scored.append((coverage_gain(p, p_global), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:m]]


def select_carbon_only(
    candidates: List[Dict], m: int,
    *, tau: int = 3, z_bits: float = 1e6, PUE: float = 1.5,
) -> List[Dict]:
    scored = []
    for c in candidates:
        mc = marginal_carbon_g(c, [], tau=tau, z_bits=z_bits, PUE=PUE)
        scored.append((mc, c))
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:m]]


def select_poc(
    candidates: List[Dict], m: int, state: dict, rng: np.random.Generator, d: float = 4.0
) -> List[Dict]:
    pool_size = max(m, int(d * m))
    pool_idx = rng.choice(len(candidates), size=min(pool_size, len(candidates)), replace=False)
    pool = [candidates[int(i)] for i in pool_idx]
    scored = []
    for c in pool:
        h = _ensure_hist(state, c["idx"])
        scored.append((h.sigma_acc + 1e-6, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:m]]
