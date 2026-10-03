# FedCAMO-CR: Stage-Aware Trust-Aware Carbon-Budgeted Client Selection

Companion paper to **Lu et al., "FedCAMO: Federated Learning Carbon-Aware Multi-Objective Client Selection," _Computer Networks_ 286:112486, 2026** (DOI:10.1016/j.comnet.2026.112486).

FedCAMO-CR extends FedCAMO with:

1. A **second Lagrangian dual variable `μ`** that prices per-client robustness risk alongside the original carbon price `λ`.
2. A **smoothed three-stage weighting schedule** that transitions the policy from a utility-driven regime in early training to a robustness- and carbon-conservative regime in late training.
3. A **cosine-agreement trust signal** computed against a robust exponential moving average of cohort gradients, requiring no trusted third party.

The framework is the first synchronous FL client-selection policy that jointly enforces a strict per-round carbon budget and a per-client trust threshold.

## Empirical evaluation

* **3 synthetic datasets** (CIFAR-10 / SVHN / GSC regimes) × **5 seeds** each
* **7 Geo×Tier scenarios** (full 3×3 factorial × coupling variants)
* **4 attack modes** (label flipping, model replacement, adaptive mix, none)
* **5 aggregators** (FedAvg, Krum, Trimmed Mean, FoolsGold, trust-weighted)
* **8 client-selection algorithms** (FedAvg, FedCAMO, FedCAMO-PR, FedCAMO-CR variants)

### Headline results (5 seeds, 95% CI, CIFAR-10 baseline)

| Method                        | No attack         | MR r=0.3          |
| ----------------------------- | ----------------- | ----------------- |
| FedAvg+FedAvg                 | 90.20 ± 0.13%     | 49.21 ± 5.78%     |
| FedAvg+Krum                   | 89.88 ± 0.24%     | 69.38 ± 25.82%    |
| **FedCAMO-CR+FedAvg**         | **88.67 ± 0.62%** | **82.72 ± 5.21%** |
| **FedCAMO-CR+trust-weighted** | **88.39 ± 0.77%** | **83.03 ± 7.27%** |

FedCAMO-CR dominates FedAvg on **all 7 Geo×Tier scenarios** under model-replacement attack (gains +8.5 to +37.0 pp; median +33.5 pp), while saving 11–14% cumulative CO₂e. Cross-dataset: CIFAR-10 +33.5 pp, SVHN +26.94 pp, GSC +46.70 pp.

Welch t-test significance rate: **61/84 cells (73%)** at α = 0.05.

## Repository layout

```
src/
  core/                 data, models, carbon model, scenario / CI provider
  fedcamo/              vanilla FedCAMO baselines (PR, CB, Loss/Coverage/Carbon-only)
  fedcamo_cr/           our extension: stage-aware trust + dual-loop λ/μ
  attacks/              label-flipping, model-replacement, adaptive attacker
  robust_agg/           Krum, Trimmed Mean, FoolsGold
configs/                yaml configs for each 3×3-factorial scenario
experiments/
  run_robust.py         per-experiment runner
  aggregate.py          log.json → Table 5 + Pareto + dynamics + Welch
  run_matrix_simple.sh  xargs wrapper for the full matrix
results/                per-experiment log.json (1456 runs across 9 scenarios)
paper/
  intro.tex             main manuscript (18 pages, Elsevier elsarticle)
  intro.pdf             compiled PDF (885 KB)
  cover_letter.tex      submission cover letter (Computer Networks)
  cover_letter.pdf      compiled cover letter (92 KB)
  tables/               table5_*.csv, appendix_welch.{tex,csv}, factorial_summary.csv
  figures/              pareto_cross_dataset.png, fig_factorial_summary.png,
                        dynamics_4panel_baseline.png, per-scenario Pareto/dynamics
```

## Quick start

### Reproduce the full 5-seed matrix (~50 minutes on a 48-core workstation)

```bash
# 1. baseline matrix (CIFAR-10, 100 configs × 5 seeds = 500 runs, ~8 min)
python3 experiments/run_robust.py --scenario baseline --algo fedavg \
       --agg fedavg --attack mr --attack-ratio 0.3 --rounds 200 \
       --seeds 5 --out-root results/

# 2. factor matrix (7 scenarios × 32 configs × 5 seeds = 1120 runs, ~40 min)
bash experiments/run_matrix_simple.sh
```

### Aggregate results

```bash
# Per-scenario Table 5 + Welch t-test
python3 experiments/aggregate.py --scenario baseline --out paper/tables
# Cross-dataset merged table
python3 experiments/aggregate.py --cross-dataset --out paper/tables
```

### Compile the paper

```bash
cd paper
pdflatex intro.tex && bibtex intro && pdflatex intro.tex && pdflatex intro.tex
```

## Implementation notes

* **CPU-only, NumPy + minimal torch**: every "client" is a Python class wrapping its partition of the synthetic dataset; no separate Flower / socket layer.
* **Carbon intensities**: real hourly traces from Electricity Maps (DE/ES/SE) — see `src/core/ci_provider.py` and `get_ci_uk.py` for the GB adapter.
* **Hardware profiles**: follow FedCAMO Table 3; the dataset is fixed at 90 clients (10 per region × tier cell).
* **5-seed coverage is uneven on GSC**: the 35-class regime was the most compute-intensive (212 s per run on a 48-core AMD EPYC); the camera-ready revision includes a complete n=5 fill-in. See `paper/intro.tex` §6.3 "Coverage caveat" for the per-cell n counts.

## License

Code: MIT. Paper text and figures: CC BY 4.0 (the published paper is under CC BY-NC-ND in Computer Networks; the README and any derivative figures follow the same terms).

## Citation

```bibtex
@article{Lu2026FedCAMOCR,
  title   = {Stage-Aware Trust-Aware Carbon-Budgeted Client Selection
             for Synchronous Federated Learning: FedCAMO-CR},
  author  = {[Xiangshan zuo]},
  journal = { },
  year    = {2026},
  note    = {Companion paper to Lu et al.\ 2026 (DOI:10.1016/j.comnet.2026.112486)}
}

@article{Lu2026FedCAMO,
  title   = {FedCAMO: Federated Learning Carbon-Aware Multi-Objective
             Client Selection},
  author  = {Lu, J. and Postigo-Boix, M. and Baz{\'a}n Guill{\'e}n, A. and
             de la Cruz Llopis, L. J. and Igartua, M. A.},
  journal = {Computer Networks},
  volume  = {286},
  pages   = {112486},
  year    = {2026},
  doi     = {10.1016/j.comnet.2026.112486}
}
```
