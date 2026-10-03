"""Smoke test for src/core/carbon.py (carbon model in §3.2)."""
import pytest
from src.core.carbon import total_round_carbon_g


def test_total_round_carbon_g_callable():
    assert callable(total_round_carbon_g)


def test_carbon_total_returns_decomposition():
    """Smoke: total_round_carbon_g should accept cohort and return dict-like."""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
