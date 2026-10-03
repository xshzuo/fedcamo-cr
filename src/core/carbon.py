"""Holistic carbon model from FedCAMO Section 3.4.

Reproduces equations (5),(7),(9)-(17) in pure NumPy. Computation, communication
and idle energies are computed for a chosen cohort S round-by-round.

To keep CPU cost down and avoid simulating the long-tail channel/SNR
distributions, we follow the paper's *projected* values:
  T_k = τ·φ_k·D_k / f_k + 2·Z / R_k
where R_k = W·log2(1+SINR_k).
We fix W=20kHz (typical NB-IoT), Z = 1e6 bits (1 MB, small stand-in for a
CNN/transformer update, FedCAMO used "Z" generic), and assign per-client
SINR_k based on a tier-aware lookup: Strong=12dB, Mid=6dB, Weak=2dB.
Energy efficiency EE follows the paper's "Optimized IoT (50 kbit/J)" baseline
(which is the implicit benchmark for our reproduction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

# Default channel params (NB-IoT regime per FedCAMO Table 4 "Optimized IoT")
W_HZ = 20e3
SINR_DB = {"Strong": 12.0, "Mid": 6.0, "Weak": 2.0}
EE_KBIT_J = {"uplink": 50.0, "downlink": 50.0}  # Optimized-IoT profile

# Workload intensity in cycles/sample (synthetic stand-in).
PHI_CYCLES_PER_SAMPLE = 1.0e6
Z_BITS_DEFAULT = 1e6   # 1 MB payload
TAU_DEFAULT = 3         # local epochs per round
F_SERVER_HZ = 3.5e9
GAMMA_AGG = 1e6         # cycles per bit during aggregation


def _data_rate(sinr_db: float) -> float:
    snr = 10 ** (sinr_db / 10)
    return float(W_HZ * np.log2(1.0 + snr))


@dataclass
class CarbonParams:
    tau: int = TAU_DEFAULT
    z_bits: float = Z_BITS_DEFAULT
    PUE: float = 1.5
    phi: float = PHI_CYCLES_PER_SAMPLE


def round_latency_s(
    selected: List[Dict],
    cohort_clients: List[Dict],
    *,
    tau: int = TAU_DEFAULT,
    z_bits: float = Z_BITS_DEFAULT,
) -> float:
    """Max over cohort of (compute + round-trip comm) + server agg time."""
    latencies = []
    for c in cohort_clients:
        T_comp = tau * PHI_CYCLES_PER_SAMPLE * c["n_samples"] / c["hw"]["f"]
        R = _data_rate(SINR_DB[c["tier"]])
        T_comm = 2.0 * z_bits / max(R, 1.0)
        latencies.append(T_comp + T_comm)
    T_round = max(latencies) if latencies else 0.0
    # Server aggregation time (a single Xeon core is plenty for K<=90)
    server_size_bits = z_bits * len(cohort_clients)
    T_agg = GAMMA_AGG * server_size_bits / F_SERVER_HZ
    return float(T_round + T_agg)


def client_energy_j(
    c: Dict,
    T_round_s: float,
    *,
    tau: int = TAU_DEFAULT,
    z_bits: float = Z_BITS_DEFAULT,
) -> float:
    """Per-client total energy (J) for the round."""
    T_comp = tau * PHI_CYCLES_PER_SAMPLE * c["n_samples"] / c["hw"]["f"]
    E_comp = c["hw"]["kappa"] * tau * PHI_CYCLES_PER_SAMPLE * c["n_samples"] * (c["hw"]["f"] ** 2)
    R = _data_rate(SINR_DB[c["tier"]])
    E_comm = z_bits * (1.0 / (EE_KBIT_J["uplink"] * 1e3) + 1.0 / (EE_KBIT_J["downlink"] * 1e3))
    E_idle = c["hw"]["P_idle"] * T_round_s
    return float(E_comp + E_comm + E_idle)


def server_energy_j(
    cohort: List[Dict],
    T_round_s: float,
    *,
    z_bits: float = Z_BITS_DEFAULT,
    PUE: float = 1.5,
) -> float:
    """Server-side total energy for the round."""
    server_size_bits = z_bits * len(cohort)
    E_agg = 2.3e-27 * GAMMA_AGG * server_size_bits * (F_SERVER_HZ ** 2)
    E_comm = server_size_bits * (2.0 / (50e3))  # 1/EE  symmetric
    E_idle = 50.8 * T_round_s
    return float(PUE * (E_agg + E_comm + E_idle))


def marginal_carbon_g(
    c: Dict,
    current_cohort: List[Dict],
    *,
    tau: int = TAU_DEFAULT,
    z_bits: float = Z_BITS_DEFAULT,
    PUE: float = 1.5,
) -> float:
    """Eq.(30): direct local emissions + induced server penalty."""
    # Projected latency if c is added
    proj = current_cohort + [c]
    T_new = round_latency_s([], proj, tau=tau, z_bits=z_bits)
    E_c = client_energy_j(c, T_new, tau=tau, z_bits=z_bits)
    C_local = E_c * c["CI"] / 3.6e6  # J·(g/kWh) → g; 1 kWh = 3.6e6 J
    # Server penalty: difference between server cost with and without c
    T_old = round_latency_s([], current_cohort, tau=tau, z_bits=z_bits)
    E_new = server_energy_j(proj, T_new, z_bits=z_bits, PUE=PUE)
    E_old = server_energy_j(current_cohort, T_old, z_bits=z_bits, PUE=PUE) if current_cohort else 0.0
    C_server = (E_new - E_old) * REGION_CI_FOR_SERVER(SERVER_REGION_DEFAULT()) / 3.6e6
    return float(C_local + C_server)


def REGION_CI_FOR_SERVER(region: str) -> float:
    from src.core.scenario import REGION_CI
    return REGION_CI[region]


def SERVER_REGION_DEFAULT() -> str:
    return "DE"


def total_round_carbon_g(
    cohort: List[Dict],
    *,
    tau: int = TAU_DEFAULT,
    z_bits: float = Z_BITS_DEFAULT,
    PUE: float = 1.5,
) -> float:
    """Eq.(16): sum of per-client carbon + server carbon."""
    if not cohort:
        return 0.0
    T = round_latency_s([], cohort, tau=tau, z_bits=z_bits)
    total = 0.0
    for c in cohort:
        E_c = client_energy_j(c, T, tau=tau, z_bits=z_bits)
        total += E_c * c["CI"] / 3.6e6
    E_server = server_energy_j(cohort, T, z_bits=z_bits, PUE=PUE)
    total += E_server * REGION_CI_FOR_SERVER(SERVER_REGION_DEFAULT()) / 3.6e6
    return float(total)
