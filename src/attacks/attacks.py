"""Byzantine attack implementations used by FedCAMO-CR.

We keep these as pure-Python drop-in modules:

* LabelFlipAttack — static label permutation (LF)
* ModelReplaceAttack — sign-flipped, amplified gradient (MR)
* AdaptiveAttack — sleep-then-strike variant (AA)

The attack module never needs to "see" the model — the wrapper
`apply_attack` in `experiments/run_robust.py` glues them to a
per-client local-SGD loop.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class AttackSpec:
    """Base interface. Concrete attacks override one or both of:
      perturb_data(X, y)         → X', y'  (data-side attacks: LF)
      perturb_grad(grad, params_ref)  → grad'  (gradient-side attacks: MR, AA)

    An attack that does nothing returns the inputs unchanged.
    """
    name: str = "none"

    def perturb_data(self, X: np.ndarray, y: np.ndarray):
        return X, y

    def perturb_grad(self, grad: np.ndarray, *, params_ref: np.ndarray):
        return grad


# ----------------------------------------------------------------------


class LabelFlipAttack(AttackSpec):
    """LF attack: flip labels to (c+1) mod C. Operates on raw labels only;
       the gradient is then computed (by the honest path) on the flipped set.
       This is the canonical attack of Bhagoji et al. and Blanchard et al."""

    name = "lf"

    def __init__(self, num_classes: int = 10, mode: str = "next"):
        self.C = num_classes
        self.mode = mode

    def perturb_data(self, X, y):
        if self.mode == "next":
            return X, (y + 1) % self.C
        elif self.mode == "swap_pair":
            # Swap a few pairs of semantically close labels
            pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
            y2 = y.copy()
            rng = np.random.default_rng(7)
            for a, b in pairs:
                mask = (y == a)
                if mask.any():
                    if rng.random() < 0.5:
                        y2[mask] = b
                    else:
                        y2[mask] = a
                    mask2 = (y == b)
                    y2[mask2] = a if y2[mask].any() else b
            return X, y2
        else:
            return X, y


# ----------------------------------------------------------------------


class ModelReplaceAttack(AttackSpec):
    """MR attack: replace local grad with a goal gradient that maximises the
       global loss. Following Bhagoji et al. NeurIPS 2019, we set
       g_adv = -scale * params_ref  (push weights in opposite direction).

       Intuition: the malicious client submits a "move far from current model"
       update; even after FedAvg aggregation, this has visible influence.
    """

    name = "mr"

    def __init__(self, scale: float = 5.0):
        self.scale = scale

    def perturb_grad(self, grad: np.ndarray, *, params_ref: np.ndarray):
        return -self.scale * params_ref


# ----------------------------------------------------------------------


class AdaptiveAttack(AttackSpec):
    """AA attack: behave honestly until t > t_attack, then switch to full
       sign-flip attack. Models an attacker that waits until the server's
       trust signal has built up before striking.

       With FedCAMO-CR, this stress-tests whether the dual μ-loop is fast
       enough to recover after the attack phase starts.
    """

    name = "aa"

    def __init__(self, t_attack: int = 30, scale: float = 5.0):
        self.t_attack = t_attack
        self.scale = scale
        self.t = 0

    def perturb_grad(self, grad: np.ndarray, *, params_ref: np.ndarray, **kwargs):
        self.t += 1
        if self.t < self.t_attack:
            return grad
        # sign-flip with scaling
        return -self.scale * params_ref + 0.02 * np.random.randn(*params_ref.shape)


# ----------------------------------------------------------------------


def make_attack(name: str, **kwargs) -> AttackSpec:
    table = {
        "lf":   LabelFlipAttack,
        "mr":   ModelReplaceAttack,
        "aa":   AdaptiveAttack,
        "none": AttackSpec,
    }
    if name not in table:
        raise ValueError(f"unknown attack {name}")
    # Only LF consumes num_classes — strip the key for the others.
    if name in ("mr", "aa"):
        kwargs.pop("num_classes", None)
    return table[name](**kwargs)
