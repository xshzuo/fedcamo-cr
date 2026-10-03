"""Unit tests for trust_signal (Eq. 7 cosine agreement)."""
import numpy as np
import pytest

from src.core.robust_agg import trust_signal


def test_trust_signal_identical_gradients():
    np.random.seed(42)
    g = np.random.randn(64, 10)
    s = trust_signal(g, g)
    assert 0.99 <= s <= 1.0, f"Identical should give ~1.0, got {s}"


def test_trust_signal_opposite_gradients():
    np.random.seed(43)
    g = np.random.randn(64, 10)
    s = trust_signal(g, -g)
    assert 0.0 <= s <= 0.5, f"Opposite should give ~0.0, got {s}"


def test_trust_signal_random_gradients():
    np.random.seed(44)
    g1 = np.random.randn(64, 10)
    g2 = np.random.randn(64, 10)
    s = trust_signal(g1, g2)
    assert 0.0 <= s <= 1.0, f"Random should be in [0, 1], got {s}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
