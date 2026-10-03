# Tests

Unit tests for FedCAMO-CR code. Run with:

```bash
pytest tests/                    # ~5 sec on commodity laptop
pytest tests/test_aggregation.py -v    # Bulyan/DnC/MoM tests
pytest tests/test_trust_signal.py -v    # cosine-agreement Eq. 7
```

## Layout

```
tests/
├── conftest.py                # adds project root to sys.path
├── test_aggregation.py        # Bulyan/DnC/MoM/Krum/TM/FG/trust-W unit tests
├── test_carbon.py             # carbon model (§3.2) smoke test
├── test_selector.py           # Algorithm 1 mock import test
├── test_trust_signal.py       # Eq. 7 cosine-agreement unit test
└── data/                      # small fixtures (committed to Git)
    ├── README.md
    ├── generate_fixtures.py   # regenerate fixtures from local results/
    └── *.csv                  # n=2-3 seeds, single config
```

## Coverage

| module | file | lines covered |
|--------|------|---------------|
| robust_agg | test_aggregation.py | 8 functions |
| carbon | test_carbon.py | 2 functions |
| trust_signal | test_trust_signal.py | 1 function |
| selector_cr | test_selector.py | smoke only |

Full integration tests (run_matrix.py) take ~50 min on a 48-core
workstation and are not included in this unit-test suite.
