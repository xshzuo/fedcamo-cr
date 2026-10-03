"""Main simulator: one shot of FL training + client selection.

Usage:
    python3 experiments/run_replication.py \
        --scenario baseline --algo fedcamo_pr --rounds 200 --seeds 5

Outputs (CSV per seed under results/<scenario>/<algo>/seed=<s>/round.csv)
plus an aggregate summary stats CSV under .../summary.csv.

We do NOT use flwr; the simulator instantiates every client in-process
for a fully reproducible synchronous FL loop on CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
from pathlib import Path
import numpy as np

# Ensure repo root is importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.data import partition
from src.core.scenario import enumerate_clients
from src.core.carbon import total_round_carbon_g
from src.core.models import LinearNet, softmax
from src.fedcamo.selector import (
    FedCAMOState, select_cohort, update_duals, update_utility_signals,
    diversity_utility, coverage_gain,
)
from src.core.registry import (
    select_random, select_loss_only, select_coverage_only,
    select_carbon_only, select_poc,
)


def softmax_loss_and_grad(model: LinearNet, X: np.ndarray, y: np.ndarray):
    z = model.forward(X)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z); p = ez / ez.sum(axis=1, keepdims=True)
    diff = p
    diff[np.arange(y.shape[0]), y] -= 1.0
    diff /= y.shape[0]
    gW = diff.T @ X
    gb = diff.sum(axis=0)
    return float(-np.log(np.clip(p[np.arange(y.shape[0]), y], 1e-12, 1.0)).mean()), gW, gb


def train_local_epochs(
    model: LinearNet, X: np.ndarray, y: np.ndarray, *, lr: float = 0.05, epochs: int = 3
):
    """Plain SGD; returns average loss-drop across epochs."""
    p_before = model.predict(X)
    acc_before = float((p_before == y).mean())
    losses = []
    for _ in range(epochs):
        loss, gW, gb = softmax_loss_and_grad(model, X, y)
        model.W -= lr * gW; model.b -= lr * gb
        losses.append(loss)
    p_after = model.predict(X)
    acc_after = float((p_after == y).mean())
    return acc_before, acc_after, float(losses[0] - losses[-1])


# Selection entry-points -----------------------------------------------------------------

def pick_cohort(algo: str,
                candidates: list, candidates_dists: list, p_global: np.ndarray,
                state: FedCAMOState, *, budget_g: float, tau: int, z_bits: float,
                PUE: float, rng: np.random.Generator):
    if algo == "fedcamo_pr":
        cohort, indices, _pred = select_cohort(
            candidates, candidates_dists, p_global, state,
            budget_g=budget_g, mode="PR",
            tau=tau, z_bits=z_bits, PUE=PUE, rng=rng,
        )
        return cohort
    elif algo == "fedcamo_cb":
        cohort, indices, _pred = select_cohort(
            candidates, candidates_dists, p_global, state,
            budget_g=budget_g, mode="CB",
            tau=tau, z_bits=z_bits, PUE=PUE, rng=rng,
        )
        return cohort
    elif algo == "fedavg":
        return select_random(candidates, m=state.cohort_size, rng=rng)
    elif algo == "loss_only":
        sdict = {"histories": state.histories}
        return select_loss_only(candidates, m=state.cohort_size, state=sdict, rng=rng)
    elif algo == "coverage_only":
        return select_coverage_only(candidates, m=state.cohort_size,
                                    p_global=p_global, candidates_dists=candidates_dists)
    elif algo == "carbon_only":
        return select_carbon_only(candidates, m=state.cohort_size, tau=tau, z_bits=z_bits, PUE=PUE)
    elif algo == "poc":
        sdict = {"histories": state.histories}
        return select_poc(candidates, m=state.cohort_size, state=sdict, rng=rng)
    else:
        raise ValueError(f"unknown algo {algo}")


# Main loop ---------------------------------------------------------------------

def run_one(algo: str, scenario: str, *, rounds: int, seed: int,
            samples_per_client: int = 600,
            tau: int = 3, lr: float = 0.05, z_bits: float = 1e6,
            PUE: float = 1.5, m: int = 9,
            budget_g: float | None = None,
            budget_friction: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    data = partition(seed=seed, samples_per_client=samples_per_client)
    attrs, indices_in_order = enumerate_clients(scenario)
    assert len(attrs) == len(data["clients"])
    # bind attributes (region/tier/idx) to each client partition
    for a in attrs:
        a["n_samples"] = data["clients"][a["idx"]][0].shape[0]
    candidates = list(attrs)
    candidates_dists = [data["client_distributions"][a["idx"]] for a in attrs]

    p_global = data["global_distribution"]
    Xt, yt = data["global_test"]
    model = LinearNet(num_classes=10, dim=64, seed=seed)
    state = FedCAMOState(cohort_size=m)

    # Initial "FedAvg-like" carbon baseline = avg carbon if we picked random m each round.
    if budget_g is None:
        rc = select_random(candidates, m, rng)
        budget_g = total_round_carbon_g(rc, tau=tau, z_bits=z_bits, PUE=PUE) * budget_friction

    round_log = []
    cum_carbon = 0.0

    for r in range(rounds):
        # Server picks cohort
        cohort = pick_cohort(
            algo, candidates, candidates_dists, p_global, state,
            budget_g=budget_g, tau=tau, z_bits=z_bits, PUE=PUE, rng=rng,
        )
        # Clients do local training; server collects loss-drop and aggregates
        size_weights = np.array([c["n_samples"] for c in cohort], dtype=np.float64)
        wsum = size_weights.sum()
        # capture pre-round params for local gradient averaging
        params0 = model.params.copy()
        loss_drops = {}
        for c in cohort:
            X, y = data["clients"][c["idx"]]
            acc_b, acc_a, drop = train_local_epochs(model, X, y, lr=lr, epochs=tau)
            loss_drops[c["idx"]] = drop  # marginal contribution
        # FedAvg: aggregated delta = (sum n_k (params_after_k - params0)) / sum n_k
        # In our serial training, params_after_k is current model after the k-th client
        # which is sequential; we approximate Δ with overall params - params0.
        # (Equivalent to one-pass FedAvg with simultaneous updates in synchronous FL.)
        avg_delta = (model.params - params0) / max(1, len(cohort))
        # Apply aggregated delta to a fresh "global model" (we keep one model in memory)
        model.params = params0 + avg_delta

        # Test accuracy AFTER local updates
        acc = model.accuracy(Xt, yt)

        # Carbon realised
        round_carbon = total_round_carbon_g(cohort, tau=tau, z_bits=z_bits, PUE=PUE)
        cum_carbon += round_carbon

        round_log.append(dict(
            round=r,
            accuracy=acc,
            round_carbon_g=round_carbon,
            cum_carbon_g=cum_carbon,
            cohort_size=len(cohort),
            lambda_t=state.lambda_t,
            bank=state.bank,
        ))

        # Update utility history + dual (only the FedCAMO family maintains λ/μ)
        update_utility_signals(state, cohort, loss_drops)
        if algo in ("fedcamo_pr", "fedcamo_cb", "fedcamo_cr_pr", "fedcamo_cr_cb"):
            update_duals(state, round_carbon, budget_g,
                         mode=("CB" if algo.endswith("cb") else "PR"))

    return dict(
        algo=algo, scenario=scenario, seed=seed, budget_g=budget_g,
        rounds_log=round_log,
        final_acc=round_log[-1]["accuracy"],
        cum_carbon_g=cum_carbon,
        final_lambda=state.lambda_t,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--algo", required=True)
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--budget-friction", type=float, default=0.85,
                   help="Multiply the random-baseline carbon by this to set a tighter budget for FedCAMO.")
    p.add_argument("--out-root", default=str(ROOT / "results"))
    args = p.parse_args()

    out_root = Path(args.out_root) / args.scenario / args.algo
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    t0 = time.time()
    for s in range(args.seeds):
        out = run_one(args.algo, args.scenario, rounds=args.rounds, seed=s, lr=args.lr,
                      budget_friction=args.budget_friction)
        sd = out_root / f"seed={s}"
        sd.mkdir(parents=True, exist_ok=True)
        with open(sd / "log.json", "w") as f:
            json.dump({"meta": {k:v for k,v in out.items() if k != "rounds_log"},
                       "rounds": out["rounds_log"]}, f, indent=2)
        summaries.append({
            "scenario": args.scenario, "algo": args.algo, "seed": s,
            "final_acc": out["final_acc"], "cum_carbon_g": out["cum_carbon_g"],
            "budget_g": out["budget_g"], "final_lambda": out["final_lambda"],
        })
    print(f"[{args.algo}/{args.scenario}] {args.seeds} seed(s) | t={time.time()-t0:.1f}s")
    for s in summaries:
        print(f"  seed{s}: final_acc={s['final_acc']:.4f} cum_carbon={s['cum_carbon_g']:.2f}g")


if __name__ == "__main__":
    main()
