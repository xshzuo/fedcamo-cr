"""FedCAMO-CR: stage-aware trust-aware carbon-budgeted client selection.

This is the main novelty of the A1 paper. The full algorithm is documented
in the LaTeX draft; here we implement the surface area:

  * Per-client trust EWMA
  * Risk_k derived from trust via soft-floor function
  * Stage-aware weighting (early/mid/late) of accuracy vs robustness vs carbon
  * Score = U_acc + U_div − λ·ΔC − μ·Risk
  * Two dual-variables (λ for carbon, μ for robustness), both updated
    online through sub-gradient ascent — see Eq.(37) of FedCAMO for λ,
    analogous rule for μ.

Differences vs the vanilla FedCAMO selector:
  (a) Selection score has an extra penalty term −μ·Risk_k
  (b) μ has its own dual update rule driven by the cohort's *robustness gap*
  (c) Stage schedule swaps the relative weight of accuracy-utility (U_acc)
      and robustness-risk (Risk_k) across training rounds
  (d) We integrate with `aggregation` (in `experiments/run_robust.py`)
      through trust_weighted() so trust scores are *operational*, not
      merely diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.carbon import marginal_carbon_g
from src.fedcamo.selector import (
    ClientHistory, FedCAMOState, diversity_utility,
)


def stage_weight(t: int, T: int) -> Tuple[float, float]:
    """
    Returns (w_acc, w_risk): weights for accuracy-utility and risk-penalty.
    Stage I  (early, t/T < 0.30): high w_acc (≈0.7), low w_risk  (≈0.3)
    Stage II (mid,   0.30–0.65): balanced w_acc=w_risk=0.5
    Stage III(late, t/T > 0.65): low w_acc (≈0.4), high w_risk (≈0.6)

    Tanh-style smooth interpolation so the stage transitions don't
    produce kinks in μ dynamics.
    """
    p = t / max(T - 1, 1)
    # ease-in-out via tanh around stage boundaries
    s1 = 1.0 / (1.0 + np.exp(-12 * (p - 0.30)))
    s2 = 1.0 / (1.0 + np.exp(-12 * (p - 0.65)))
    w_risk = 0.3 + (0.6 - 0.3) * (s1 + s2) / 2.0
    w_acc = 1.0 - w_risk
    # make sure the early stay near 0.70/0.30 split
    w_acc = max(0.40, min(0.70, w_acc))
    w_risk = 1.0 - w_acc
    return float(w_acc), float(w_risk)


@dataclass
class CRHistory(ClientHistory):
    trust: float = 1.0              # 1.0 = honest by default
    ema_loss_drop: float = 0.0
    last_grad: np.ndarray = field(default=None, repr=False)


@dataclass
class FedCAMOCRState(FedCAMOState):
    mu_t: float = 0.0               # dual variable for robustness gap
    eta_mu: float = 0.5             # learning rate for μ
    rho_target: float = 0.10        # acceptable adversarial success threshold
    mu_min: float = 0.0
    mu_max: float = 50.0


def risk_from_trust(trust: float, *, floor: float = 0.10) -> float:
    """Risk: 0 when trust=1, 1 when trust≈floor.
       Uses soft-floor; guarantees 0 < floor ≤ 1 to keep some uncertainty."""
    floor = max(0.05, min(0.95, floor))
    return float(1.0 - (trust - floor) / (1.0 - floor))


# ----------------------------------------------------------------------


def select_cohort_cr(
    candidate_attrs: List[Dict],
    candidate_dists: List[np.ndarray],
    p_global: np.ndarray,
    state: FedCAMOCRState,
    *,
    budget_g: float,
    mode: str = "PR",
    tau: int = 3,
    z_bits: float = 1e6,
    PUE: float = 1.5,
    round_t: int = 0,
    total_rounds: int = 200,
    rng: Optional[np.random.Generator] = None,
):
    """Returns (cohort, indices, projected_carbon). Mirrors FedCAMO's select_cohort but
       adds the −μ·Risk_k term."""
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(p_global, np.ndarray):
        p_global = np.asarray(p_global, dtype=np.float64)

    m_target = state.cohort_size
    m_max = state.m_max if mode == "CB" else m_target
    m_min = state.m_min if mode == "CB" else m_target

    bank = state.bank if mode == "CB" else 0.0
    B_gate = (1.0 + state.budget_gate_slack) * budget_g + (bank if mode == "CB" else 0.0)
    hard_cap = (2.0 * (1.0 + state.budget_gate_slack) * budget_g) if mode == "CB" else None

    w_acc, w_risk = stage_weight(round_t, total_rounds)

    S: List[Dict] = []
    rem = list(range(len(candidate_attrs)))
    projected_carbon = 0.0
    indices_chosen: List[int] = []

    while len(S) < m_max and rem:
        scores = []
        mcs = []
        n_total_sel = sum(h.n_selections for h in state.histories.values())
        for i in rem:
            c = candidate_attrs[i]
            hist = state.histories.setdefault(c["idx"], CRHistory())
            n_eff = max(hist.n_recent, 1e-3)
            acc_ucb = hist.sigma_acc + state.alpha_ucb * np.sqrt(np.log1p(n_total_sel) / n_eff)
            div = diversity_utility(candidate_dists[i], p_global)
            util_score = w_acc * acc_ucb + div   # robustness handled separately

            mc = marginal_carbon_g(c, S, tau=tau, z_bits=z_bits, PUE=PUE)
            mcs.append(mc)
            risk = risk_from_trust(hist.trust)
            score = util_score - state.lambda_t * mc - state.mu_t * w_risk * risk
            scores.append(score)
        idx_local = int(np.argmax(scores))
        cand = candidate_attrs[rem[idx_local]]
        cand_mc = mcs[idx_local]
        projected_carbon += cand_mc

        if mode == "CB" and (projected_carbon > B_gate or projected_carbon > hard_cap):
            projected_carbon -= cand_mc
            break
        if projected_carbon > B_gate:
            projected_carbon -= cand_mc
            break
        S.append(cand); indices_chosen.append(rem[idx_local]); rem.pop(idx_local)
        if len(S) >= m_target and mode == "PR":
            break

    return S, indices_chosen, projected_carbon


def update_mu(state: FedCAMOCRState, *, robustness_gap: float):
    """Sub-gradient ascent on μ with target = rho_target."""
    violation = max(0.0, robustness_gap - state.rho_target)
    new_mu = state.mu_t + state.eta_mu * violation
    state.mu_t = float(np.clip(new_mu, state.mu_min, state.mu_max))


def update_signals_cr(
    state: FedCAMOCRState,
    cohort: List[Dict],
    loss_drops: Dict[int, float],
    trust_signals: Dict[int, float],
    *,
    gamma_trust: float = 0.7,
    gamma_ema_acc: float = 0.95,
):
    """Update EWMA-tracked accuracy, loss-drop and trust score per-client."""
    g = state.gamma_ewma
    by_idx = {c["idx"]: c for c in cohort}
    for c in cohort:
        h = state.histories.setdefault(c["idx"], CRHistory())
        h.n_selections += 1
        if c["idx"] in loss_drops:
            drop = float(loss_drops[c["idx"]])
            h.last_loss_drop = drop
            h.sigma_acc = g * h.sigma_acc + (1 - g) * drop
        if c["idx"] in trust_signals:
            t_now = float(trust_signals[c["idx"]])
            h.trust = gamma_trust * h.trust + (1 - gamma_trust) * t_now
        # EWMA-smooth UCB counts
        h.n_recent = gamma_ema_acc * h.n_recent + (1 - gamma_ema_acc) * 1.0
