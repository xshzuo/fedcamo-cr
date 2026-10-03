"""FedCAMO client selection (PR + CB) — Algorithm 1 of Lu et al. 2026.

Reproduces the per-round score-based greedy cohort construction
using Lagrangian dual-update on λ.

We rely on:
  src/core/carbon.py — marginal_carbon_g, total_round_carbon_g
  src/core/data.py   — synthetic partition; client distributions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


# ----------------------------------------------------------------------
# Signal computation: utility
# ----------------------------------------------------------------------

def normal_entropy(p: np.ndarray, eps: float = 1e-9) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / max(p.sum(), 1.0)
    C = p.shape[0]
    if C <= 1:
        return 0.0
    nz = p > eps
    h = -float((p[nz] * np.log(p[nz])).sum())
    h_max = float(np.log(C))
    return float(h / h_max) if h_max > 0 else 0.0


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return float((a * np.log(a / b)).sum())
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def coverage_gain(p_k: np.ndarray, p_global: np.ndarray) -> float:
    p_unif = np.full_like(p_global, 1.0 / len(p_global))
    num = jsd(p_k, p_global)
    den = jsd(p_unif, p_global)
    return max(0.0, 1.0 - num / max(den, 1e-9))


def diversity_utility(p_k: np.ndarray, p_global: np.ndarray, omega: float = 0.5) -> float:
    return float(omega * normal_entropy(p_k) + (1 - omega) * coverage_gain(p_k, p_global))


# ----------------------------------------------------------------------
# EWMA-tracked accuracy utility, kept server-side per client
# ----------------------------------------------------------------------

@dataclass
class ClientHistory:
    n_selections: int = 0
    n_recent: float = 0.0          # EWMA of selection counts
    sigma_acc: float = 0.0         # smoothed loss-drop estimate
    last_loss_drop: float = 0.0
    trust: float = 1.0             # honest init (overridden by CR)


# ----------------------------------------------------------------------
# Selection state
# ----------------------------------------------------------------------

@dataclass
class FedCAMOState:
    histories: Dict[int, ClientHistory] = field(default_factory=dict)
    lambda_t: float = 0.0
    bank: float = 0.0
    cohort_size: int = 9
    alpha_ucb: float = 0.5
    gamma_ewma: float = 0.7
    gamma_ucb_smooth: float = 0.95
    eta_dual: float = 0.5
    budget_gate_slack: float = 0.05
    m_min: int = 8
    m_max: int = 12


def _add_bank(bank: float, delta: float, cap: float = 10.0) -> float:
    new = bank + delta
    return float(np.clip(new, -cap, cap))


# ----------------------------------------------------------------------
# Public entry
# ----------------------------------------------------------------------

def select_cohort(
    candidate_attrs: List[Dict],
    candidate_dists: List[np.ndarray],
    p_global: np.ndarray,
    state: FedCAMOState,
    *,
    budget_g: float,
    mode: str = "PR",   # "PR" or "CB"
    tau: int = 3, z_bits: float = 1e6, PUE: float = 1.5,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """Returns (cohort, predicted_total_carbon)."""
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(p_global, np.ndarray):
        p_global = np.asarray(p_global, dtype=np.float64)
    m_target = state.cohort_size
    m_max = state.m_max if mode == "CB" else m_target
    m_min = state.m_min if mode == "CB" else m_target

    bank = state.bank if mode == "CB" else 0.0
    B_gate = (1.0 + state.budget_gate_slack) * budget_g + (bank if mode == "CB" else 0.0)

    S: List[Dict] = []
    indices_chosen: List[int] = []
    rem = list(range(len(candidate_attrs)))
    projected_carbon = 0.0

    while len(S) < m_max and rem:
        scores = []
        per_client_marginal = []
        for i in rem:
            c = candidate_attrs[i]
            hist = state.histories.setdefault(c["idx"], ClientHistory())

            # optimistic accuracy utility (UCB over smoothed loss-drop)
            n_eff = max(hist.n_recent, 1e-3)
            acc_ucb = hist.sigma_acc + state.alpha_ucb * np.sqrt(np.log1p(sum(h.n_selections for h in state.histories.values())) / n_eff)
            # diversity utility (deterministic given metadata)
            div = diversity_utility(candidate_dists[i], p_global)
            total_util = acc_ucb + div

            # contextual marginal carbon cost
            from src.core.carbon import marginal_carbon_g
            mc = marginal_carbon_g(c, S, tau=tau, z_bits=z_bits, PUE=PUE)
            per_client_marginal.append(mc)
            score = total_util - state.lambda_t * mc
            scores.append(score)

        i_star = int(np.argmax(scores))
        cand = candidate_attrs[rem[i_star]]
        cand_mc = per_client_marginal[i_star]
        projected_carbon += cand_mc

        # feasibility gate (with hard cap in CB mode for round spikes)
        if mode == "CB":
            hard_cap = 2.0 * (1.0 + state.budget_gate_slack) * budget_g
            if projected_carbon > B_gate or projected_carbon > hard_cap:
                projected_carbon -= cand_mc
                break
            if len(S) + 1 < m_min:
                pass  # ensure at least m_min when budget allows
        if projected_carbon > B_gate:
            projected_carbon -= cand_mc
            break

        S.append(cand)
        indices_chosen.append(rem[i_star])
        rem.pop(i_star)
        if len(S) >= m_target and mode == "PR":
            break

    # If we *underspent* in PR mode but need at least m_min, don't force top-up.
    return S, indices_chosen, projected_carbon


# ----------------------------------------------------------------------
# Post-round dual update
# ----------------------------------------------------------------------

def update_duals(
    state: FedCAMOState,
    carbon_realized_g: float,
    budget_g: float,
    *,
    mode: str = "PR",
    target_g: Optional[float] = None,    # for CB bank update
):
    """Online subgradient ascent on λ as in Eq.(37)."""
    violation = carbon_realized_g - budget_g
    new_lambda = max(0.0, state.lambda_t + state.eta_dual * violation)
    state.lambda_t = float(new_lambda)
    if mode == "CB":
        tgt = target_g if target_g is not None else budget_g
        delta = budget_g - tgt
        state.bank = _add_bank(state.bank, delta)
    # EWMA-smooth UCB counts
    for h in state.histories.values():
        h.n_recent = state.gamma_ucb_smooth * h.n_recent + (1 - state.gamma_ucb_smooth) * 1.0


# ----------------------------------------------------------------------
# Logging snapshot for later analysis
# ----------------------------------------------------------------------

def update_utility_signals(
    state: FedCAMOState,
    cohort: List[Dict],
    loss_drops: Dict[int, float],
):
    """Update EWMA-tracked accuracy utility after observing the round outcome."""
    g = state.gamma_ewma
    by_idx = {c["idx"]: c for c in cohort}
    for c in cohort:
        h = state.histories.setdefault(c["idx"], ClientHistory())
        h.n_selections += 1
        if c["idx"] in loss_drops:
            drop = float(loss_drops[c["idx"]])
            h.last_loss_drop = drop
            h.sigma_acc = g * h.sigma_acc + (1 - g) * drop


def fedcamo_pr_select(
    candidate_attrs, candidate_dists, p_global, state, *, budget_g,
    tau=3, z_bits=1e6, PUE=1.5, rng=None,
):
    return select_cohort(candidate_attrs, candidate_dists, p_global, state,
                         budget_g=budget_g, mode="PR",
                         tau=tau, z_bits=z_bits, PUE=PUE, rng=rng)


def fedcamo_cb_select(
    candidate_attrs, candidate_dists, p_global, state, *, budget_g,
    tau=3, z_bits=1e6, PUE=1.5, rng=None,
):
    return select_cohort(candidate_attrs, candidate_dists, p_global, state,
                         budget_g=budget_g, mode="CB",
                         tau=tau, z_bits=z_bits, PUE=PUE, rng=rng)
