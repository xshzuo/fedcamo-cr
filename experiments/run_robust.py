"""Robust-byzantine runner.

Extends `experiments/run_replication.py` with:
  * Per-client attack assignment (lf / mr / aa)
  * Aggregation rules (fedavg / krum / trimmed_mean / foolsgold /
    trust_weighted)
  * Trust signal computed each round (sign-agreement to cohort mean)
  * FedCAMO-CR selector (instead of vanilla FedCAMO selector)

Selectable via CLI:
  python3 experiments/run_robust.py --agg fedavg --attack lf \
      --attack_ratio 0.30 --algo fedavg ...

By default we run a single (algo, agg, attack) per invocation to keep
the experiment matrix easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.data import partition
from src.core.scenario import enumerate_clients
from src.core.carbon import total_round_carbon_g
from src.core.models import LinearNet
from src.core.robust_agg import (
    krum_multi, trimmed_mean, foolsgold, trust_weighted, trust_signal,
    bulyan, divide_and_conquer, median_of_means,
)
from src.attacks.attacks import make_attack

from src.fedcamo.selector import (
    FedCAMOState, update_duals, update_utility_signals,
    diversity_utility, coverage_gain, ClientHistory,
)
from src.fedcamo_cr.selector_cr import (
    CRHistory, FedCAMOCRState, select_cohort_cr, update_mu, update_signals_cr,
    stage_weight,
)

from src.core.registry import (
    select_random, select_loss_only, select_coverage_only,
    select_carbon_only, select_poc,
)


def softmax_loss_and_grad(model: LinearNet, X: np.ndarray, y: np.ndarray):
    z = model.forward(X)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z); p = ez / ez.sum(axis=1, keepdims=True)
    diff = p.copy()
    diff[np.arange(y.shape[0]), y] -= 1.0
    diff /= y.shape[0]
    gW = diff.T @ X
    gb = diff.sum(axis=0)
    return float(-np.log(np.clip(p[np.arange(y.shape[0]), y], 1e-12, 1.0)).mean()), gW, gb


def pick_cohort_robust(algo: str, candidates, candidates_dists, p_global,
                       state, *, budget_g, tau, z_bits, PUE,
                       round_t, total_rounds, rng):
    if algo in ("fedcamo_cr_pr", "fedcamo_cr_cb"):
        mode = "PR" if algo.endswith("pr") else "CB"
        S, _, _pred = select_cohort_cr(
            candidates, candidates_dists, p_global, state,
            budget_g=budget_g, mode=mode,
            tau=tau, z_bits=z_bits, PUE=PUE,
            round_t=round_t, total_rounds=total_rounds, rng=rng,
        )
        return S
    elif algo == "fedcamo_pr":
        from src.fedcamo.selector import select_cohort as fc_select
        S, _, _pred = fc_select(candidates, candidates_dists, p_global, state,
                                budget_g=budget_g, mode="PR",
                                tau=tau, z_bits=z_bits, PUE=PUE, rng=rng)
        return S
    elif algo == "fedcamo_cb":
        from src.fedcamo.selector import select_cohort as fc_select
        S, _, _pred = fc_select(candidates, candidates_dists, p_global, state,
                                budget_g=budget_g, mode="CB",
                                tau=tau, z_bits=z_bits, PUE=PUE, rng=rng)
        return S
    elif algo == "fedavg":
        return select_random(candidates, m=state.cohort_size, rng=rng)
    elif algo == "loss_only":
        sdict = {"histories": state.histories}
        return select_loss_only(candidates, m=state.cohort_size, state=sdict, rng=rng)
    elif algo == "poc":
        sdict = {"histories": state.histories}
        return select_poc(candidates, m=state.cohort_size, state=sdict, rng=rng)
    else:
        raise ValueError(f"unknown algo {algo}")


def aggregate(agg: str, client_grads: np.ndarray, f: int, trust_vec=None):
    if agg == "fedavg":
        return client_grads.mean(axis=0)
    elif agg == "krum":
        return krum_multi(client_grads, f=f, m=max(1, client_grads.shape[0] - f - 2))
    elif agg == "trimmed_mean":
        beta = f / max(client_grads.shape[0] - 1, 1)
        return trimmed_mean(client_grads, beta=beta)
    elif agg == "foolsgold":
        return foolsgold(client_grads)
    elif agg == "trust_weighted":
        if trust_vec is None:
            trust_vec = np.ones(client_grads.shape[0])
        return trust_weighted(client_grads, trust_vec)
    elif agg == "bulyan":
        return bulyan(client_grads, f=f)
    elif agg == "dnc":
        return divide_and_conquer(client_grads, f=f)
    elif agg == "mom":
        return median_of_means(client_grads, f=f)
    else:
        raise ValueError(f"unknown aggregation {agg}")


def run_one(algo: str, agg: str, attack: str, attack_ratio: float,
            scenario: str, *, rounds: int, seed: int,
            samples_per_client: int = 600,
            tau: int = 3, lr: float = 0.05, z_bits: float = 1e6,
            PUE: float = 1.5, m: int = 9,
            budget_friction: float = 0.85,
            cr_robustness_gap: float | None = None,
            partition_seed: int = 0,
            dataset: str = "cifar10") -> dict:
    # Two seed streams:
    #   partition_seed → fixed across all (algo, attack) runs for fair comparison
    #   seed           → varies across --seeds runs to capture stochasticity
    # The actual randomness that should vary across seeds is:
    #   * cohort sampling (random tie-breaks)
    #   * SGD batch ordering
    #   * Byzantine client assignment
    rng = np.random.default_rng(seed)
    data = partition(seed=partition_seed, samples_per_client=samples_per_client,
                     dataset=dataset)
    # enumerate_clients requires a registered scenario name (baseline/coupling_*).
    # If a dataset suffix is appended (e.g. "baseline_svhn"), strip it.
    base_scenario_for_clients = scenario
    for suffix in ("_svhn", "_gsc", "_cifar10"):
        if base_scenario_for_clients.endswith(suffix):
            base_scenario_for_clients = base_scenario_for_clients[:-len(suffix)]
            break
    attrs, _ = enumerate_clients(base_scenario_for_clients)
    assert len(attrs) == len(data["clients"])
    for a in attrs:
        a["n_samples"] = data["clients"][a["idx"]][0].shape[0]

    # Pick which clients are byzantine (worst-case selection: high-utility
    # clients — gives attacker the best chance to defeat the system).
    utility_attack_target = (algo in ("fedcamo_cr_pr", "loss_only"))
    byz_set = set()
    if attack != "none" and attack_ratio > 0:
        K = len(attrs)
        n_byz = int(round(attack_ratio * K))
        if utility_attack_target:
            # mark the byz candidates as ones with high CI (worst-case attacker
            # would otherwise be filtered out by FedCAMO-CR)
            ci_order = sorted(range(K), key=lambda i: -attrs[i]["CI"])
            byz_set = set(ci_order[:n_byz])
        else:
            byz_set = set(rng.choice(K, n_byz, replace=False).tolist())

    # Build per-client attack instances
    n_classes_runtime = int(data["num_classes"])
    attack_objs = {idx: make_attack(attack, num_classes=n_classes_runtime) for idx in byz_set}

    candidates = list(attrs)
    candidates_dists = [data["client_distributions"][a["idx"]] for a in attrs]
    p_global = data["global_distribution"]
    Xt, yt = data["global_test"]
    # Model init: also fixed partition_seed (model architecture is identical)
    model = LinearNet(num_classes=int(data["num_classes"]),
                      dim=int(data["feature_dim"]), seed=partition_seed)

    if algo.startswith("fedcamo_cr"):
        state = FedCAMOCRState(cohort_size=m)
    else:
        state = FedCAMOState(cohort_size=m)

    if budget_friction > 0:
        rc = select_random(candidates, m, rng)
        budget_g = total_round_carbon_g(rc, tau=tau, z_bits=z_bits, PUE=PUE) * budget_friction
    else:
        budget_g = float("inf")

    f = max(int(round(attack_ratio * m)), 0)  # f : at most f byz in cohort m

    round_log = []
    cum_carbon = 0.0

    for r in range(rounds):
        cohort = pick_cohort_robust(
            algo, candidates, candidates_dists, p_global, state,
            budget_g=budget_g, tau=tau, z_bits=z_bits, PUE=PUE,
            round_t=r, total_rounds=rounds, rng=rng,
        )

        # Each cohort client does attack-perturbed local training and
        # returns a single gradient (delta over τ epochs) — never their
        # updated local params. Server then aggregates gradients using
        # the chosen robust rule.
        client_grads = []
        client_loss_drops = {}
        client_idx_map = []
        params0 = model.params.copy()

        # Maintain a *shadow* temporary model so each client's τ-epoch
        # sequence is independent; afterwards we apply the aggregate
        # delta to the global model.
        for c in cohort:
            X, y = data["clients"][c["idx"]]
            attack_obj = attack_objs.get(c["idx"])
            if attack_obj is not None:
                _X, _y = attack_obj.perturb_data(X, y)
            else:
                _X, _y = X, y

            # Shadow model init: fixed (deterministic across seeds)
            shadow = LinearNet(num_classes=int(data["num_classes"]),
                               dim=int(data["feature_dim"]),
                               seed=partition_seed)
            shadow.params = params0.copy()
            # SGD mini-batch ordering: vary by `seed` for cross-seed stochasticity
            n = X.shape[0]
            perm = rng.permutation(n)
            X_use = X[perm]
            y_use = _y[perm]  # always use perturbed labels if attack; else same as y
            loss_initial = 0.0
            loss_final = 0.0
            for ep in range(tau):
                # gradient step on shadow
                z = shadow.forward(X_use)
                z = z - z.max(axis=1, keepdims=True)
                ez = np.exp(z); p = ez / ez.sum(axis=1, keepdims=True)
                diff = p.copy()
                diff[np.arange(y_use.shape[0]), y_use] -= 1.0
                diff /= y_use.shape[0]
                gW = diff.T @ X_use
                gb = diff.sum(axis=0)
                if ep == 0:
                    probs = p[np.arange(y_use.shape[0]), y_use]
                    loss_initial = float(-np.log(np.clip(probs, 1e-12, 1.0)).mean())
                shadow.W -= lr * gW; shadow.b -= lr * gb
                if ep == tau - 1:
                    z = shadow.forward(X_use); z = z - z.max(axis=1, keepdims=True)
                    ez = np.exp(z); p = ez / ez.sum(axis=1, keepdims=True)
                    probs = p[np.arange(y_use.shape[0]), y_use]
                    loss_final = float(-np.log(np.clip(probs, 1e-12, 1.0)).mean())

            grad = (shadow.params - params0)
            # gradient-side attack (e.g. AA, MR)
            if attack_obj is not None and hasattr(attack_obj, "perturb_grad"):
                if attack_obj.name == "aa":
                    grad_attacked = attack_obj.perturb_grad(grad, params_ref=params0,
                                                            **{"t": r})
                else:
                    grad_attacked = attack_obj.perturb_grad(grad, params_ref=params0)
                grad = grad_attacked

            client_grads.append(grad)
            client_idx_map.append(c["idx"])
            client_loss_drops[c["idx"]] = loss_initial - loss_final

        client_grads = np.stack(client_grads)
        # Compute trust signals BEFORE aggregation. Use a trusted reference
        # gradient maintained from previous round (initialized to cohort
        # mean of round 0 — passed through robust median thereafter). This
        # avoids the issue where byzantine gradients drag the cohort mean
        # away and mask themselves.
        cohort_median = np.median(client_grads, axis=0)
        # Honest cohort mean for trust_signal: use the median which is robust
        # iff byzantines are < 50%. For LF (data-only attack) all grads are
        # honest so this is exactly the cohort mean.
        cohort_honest_mean = cohort_median
        trust_vec = np.array([
            trust_signal(g, cohort_honest_mean) for g in client_grads
        ])
        # Update server's trusted reference for next round (robust EMA)
        if not hasattr(run_one, "_trusted_ref"):
            run_one._trusted_ref = params0.copy()
        run_one._trusted_ref = 0.5 * run_one._trusted_ref + 0.5 * cohort_median

        # Aggregate
        agg_grad = aggregate(agg, client_grads, f=f, trust_vec=trust_vec)
        # Apply aggregated delta to global model (FedAvg = sum client_deltas / K)
        model.params = params0 + agg_grad
        # Re-evaluate test accuracy on the GLOBAL model
        acc = model.accuracy(Xt, yt)
        # Track the prev-round trusted reference for the next-round trust signal;
        # here we keep a heuristic: trust is evaluated against the cohort median,
        # so it stays local to this round (no 'previous trusted' drift).
        _last_trusted_params = params0

        # Carbon realised
        round_carbon = total_round_carbon_g(cohort, tau=tau, z_bits=z_bits, PUE=PUE)
        cum_carbon += round_carbon

        # Diagnostic: how many byzantine clients actually landed in cohort?
        byz_in_cohort = sum(1 for c in cohort if c["idx"] in byz_set)

        # Robustness gap = fraction of cohort that *behaves* byzantine
        # (trust below 0.6). Lower trust → more likely byzantine.
        robustness_gap = float((trust_vec < 0.55).mean())

        round_log.append(dict(
            round=r, accuracy=acc, round_carbon_g=round_carbon,
            cum_carbon_g=cum_carbon, cohort_size=len(cohort),
            byz_in_cohort=byz_in_cohort, trust_mean=float(trust_vec.mean()),
            lambda_t=state.lambda_t, mu_t=state.mu_t if hasattr(state, "mu_t") else 0.0,
            bank=state.bank, robustness_gap=robustness_gap,
        ))

        # Update signals + duals
        if algo.startswith("fedcamo_cr"):
            update_signals_cr(state, cohort, client_loss_drops,
                              trust_signals={client_idx_map[i]: float(trust_vec[i])
                                              for i in range(len(client_idx_map))})
            if algo == "fedcamo_cr_pr" and r > 0:
                update_mu(state, robustness_gap=robustness_gap)
        else:
            update_utility_signals(state, cohort, client_loss_drops)

        if algo in ("fedcamo_pr", "fedcamo_cb"):
            update_duals(state, round_carbon, budget_g,
                         mode=("CB" if algo.endswith("cb") else "PR"))
        elif algo in ("fedcamo_cr_pr", "fedcamo_cr_cb"):
            update_duals(state, round_carbon, budget_g,
                         mode=("CB" if algo.endswith("cb") else "PR"))

    return dict(
        algo=algo, agg=agg, attack=attack, attack_ratio=attack_ratio,
        scenario=scenario, seed=seed, budget_g=budget_g,
        rounds_log=round_log,
        final_acc=round_log[-1]["accuracy"],
        cum_carbon_g=cum_carbon,
        final_lambda=state.lambda_t,
        final_mu=getattr(state, "mu_t", 0.0),
        avg_robustness_gap=float(np.mean([r["robustness_gap"] for r in round_log[-10:]])),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--algo", default="fedavg",
                   choices=["fedavg", "loss_only", "poc", "fedcamo_pr", "fedcamo_cb",
                            "fedcamo_cr_pr", "fedcamo_cr_cb"])
    p.add_argument("--agg", default="fedavg",
                   choices=["fedavg", "krum", "trimmed_mean", "foolsgold", "trust_weighted", "bulyan", "dnc", "mom"])
    p.add_argument("--attack", default="lf", choices=["none", "lf", "mr", "aa"])
    p.add_argument("--attack-ratio", type=float, default=0.30)
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--budget-friction", type=float, default=0.85)
    p.add_argument("--out-root", default=str(ROOT / "results"))
    p.add_argument("--seed-offset", type=int, default=0,
                   help="offset so different runs use different rng seeds")
    p.add_argument("--dataset", default="cifar10",
                   choices=["cifar10", "svhn", "gsc"],
                   help="synthetic dataset regime; see src/core/data.py")
    args = p.parse_args()

    # Carry dataset in scenario name so results don't mix
    base_scenario = args.scenario
    if args.dataset != "cifar10":
        base_scenario = f"{args.scenario}_{args.dataset}"

    out_root = Path(args.out_root) / base_scenario / f"{args.algo}_{args.agg}_{args.attack}_r{args.attack_ratio}"
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for s in range(args.seeds):
        actual_seed = args.seed_offset + s
        out = run_one(args.algo, args.agg, args.attack, args.attack_ratio,
                      base_scenario, rounds=args.rounds, seed=actual_seed,
                      lr=args.lr, budget_friction=args.budget_friction,
                      dataset=args.dataset)
        sd = out_root / f"seed={actual_seed}"
        sd.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in out.items() if k != "rounds_log"}
        meta["dataset"] = args.dataset  # also write into log for traceability
        with open(sd / "log.json", "w") as f:
            json.dump({"meta": meta, "rounds": out["rounds_log"]}, f, indent=2)
        print(f"  seed{s}(actual={actual_seed}): acc={out['final_acc']:.4f} carb={out['cum_carbon_g']:.1f}g "
              f"λ={out['final_lambda']:.2f} μ={out['final_mu']:.2f} "
              f"rob_gap={out['avg_robustness_gap']:.3f}")
    print(f"[{args.algo}+{args.agg}+{args.attack}@{args.attack_ratio}/"
          f"{base_scenario}] {args.seeds} seed(s) | t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
