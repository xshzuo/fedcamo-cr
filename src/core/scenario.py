"""Scenario definition: 3 regions x 3 hardware tiers + per-cell client counts.

Reproduces the 7 scenarios of FedCAMO Table A.8. Each scenario is just
a dict {region: {tier: count}}.

Region → baseline CI ≈ real 2024 annual-average:
  DE (high) ~ 380 gCO2e/kWh
  ES (mid)  ~ 170
  SE (low)  ~ 50
Tier → (frequency GHz, idle power W, kappa):
  Server Xeon: 3.5, 50.8, 2.3e-27  (NB: server is not the unit here — it's
          consumed by PUE × T_round of aggregation; included for completeness)
  Strong (Jetson Orin): 2.2, 13.0, 6.0e-27
  Mid    (Jetson Xavier NX): 1.9, 6.0, 8.0e-27
  Weak   (RPi 4): 1.5, 2.6, 1.0e-26
"""

from __future__ import annotations

from typing import Dict, Tuple

REGION_CI = {"DE": 380.0, "ES": 170.0, "SE": 50.0}  # gCO2e/kWh (annual avg)

HARDWARE = {
    "Strong": {"f": 2.2e9, "P_idle": 13.0, "kappa": 6.0e-27},
    "Mid":    {"f": 1.9e9, "P_idle": 6.0,  "kappa": 8.0e-27},
    "Weak":   {"f": 1.5e9, "P_idle": 2.6,  "kappa": 1.0e-26},
}
SERVER = {"f": 3.5e9, "P_idle": 50.8, "kappa": 2.3e-27, "PUE": 1.5}
SERVER_REGION = "DE"  # authors placed aggregation server in DE datacenter


SCENARIOS: Dict[str, Dict[str, Dict[str, int]]] = {
    "baseline": {  # 30 per region × 30 per tier
        "DE": {"Strong": 10, "Mid": 10, "Weak": 10},
        "ES": {"Strong": 10, "Mid": 10, "Weak": 10},
        "SE": {"Strong": 10, "Mid": 10, "Weak": 10},
    },
    "coupling_a": {  # DE → Weak (low-grade hardware in dirty grid)
        "DE": {"Strong": 5, "Mid": 5, "Weak": 20},
        "ES": {"Strong": 10, "Mid": 10, "Weak": 10},
        "SE": {"Strong": 15, "Mid": 15, "Weak": 0},
    },
    "coupling_b": {  # DE → Strong
        "DE": {"Strong": 20, "Mid": 5, "Weak": 5},
        "ES": {"Strong": 10, "Mid": 10, "Weak": 10},
        "SE": {"Strong": 0,  "Mid": 15, "Weak": 15},
    },
    "geo_high": {  # 80% in DE
        "DE": {"Strong": 24, "Mid": 24, "Weak": 24},
        "ES": {"Strong": 3,  "Mid": 3,  "Weak": 3},
        "SE": {"Strong": 3,  "Mid": 3,  "Weak": 3},
    },
    "geo_low": {  # 80% in SE
        "DE": {"Strong": 3,  "Mid": 3,  "Weak": 3},
        "ES": {"Strong": 3,  "Mid": 3,  "Weak": 3},
        "SE": {"Strong": 24, "Mid": 24, "Weak": 24},
    },
    "tier_strong": {  # 80% Strong
        "DE": {"Strong": 24, "Mid": 3, "Weak": 3},
        "ES": {"Strong": 24, "Mid": 3, "Weak": 3},
        "SE": {"Strong": 24, "Mid": 3, "Weak": 3},
    },
    "tier_weak": {  # 80% Weak
        "DE": {"Strong": 3, "Mid": 3, "Weak": 24},
        "ES": {"Strong": 3, "Mid": 3, "Weak": 24},
        "SE": {"Strong": 3, "Mid": 3, "Weak": 24},
    },
}


def enumerate_clients(scenario: str) -> Tuple[list, list]:
    """
    Returns:
        client_attrs: list of dicts, length K; each dict has keys
                       region, tier, hw (HARDWARE subdict), CI
        assignment: list of original Dirichlet client indices to take;
                    since we generated K=90 clients with same labels per
                    scenario, each scenario just permutes indices per
                    (region, tier) cell. The order doesn't matter for
                    global stats, only CI/hardware assignment matters.
    """
    s = SCENARIOS[scenario]
    client_attrs = []
    idx = 0
    for region, tiers in s.items():
        for tier, count in tiers.items():
            for _ in range(count):
                client_attrs.append({
                    "region": region,
                    "tier": tier,
                    "hw": HARDWARE[tier],
                    "CI": REGION_CI[region],
                    "idx": idx,
                })
                idx += 1
    return client_attrs, [a["idx"] for a in client_attrs]
