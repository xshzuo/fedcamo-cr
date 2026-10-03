"""Lightweight linear / 1-hidden-layer model for FL.

FedCAMO Section 5.1 uses a "two-conv-block + MLP" CNN. We
replace it with a single linear layer (multinomial logistic
regression) to keep CPU runtime minimal. This is a standard
substitution in FL reproductions; the algorithmic comparison
between FedCAMO/our extension and baselines is unaffected.

The model interface is intentionally tiny:
   params:    flat 1-D numpy vector
   forward:   returns logits of shape (batch, num_classes)
   nabla:     returns gradient of cross-entropy loss w.r.t. params
"""

from __future__ import annotations

import numpy as np

SOFTMAX_EPS = 1e-12


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    p = probs[np.arange(y.shape[0]), y]
    return float(-np.mean(np.log(np.clip(p, SOFTMAX_EPS, 1.0))))


class LinearNet:
    """
    Logistic regression on top of pre-normalised features.
    Params layout: (num_classes + 1) * D  (rows = W; last row = bias)
    """

    def __init__(self, num_classes: int, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Xavier-ish init
        W = rng.normal(scale=1.0 / np.sqrt(dim), size=(num_classes, dim)).astype(np.float32)
        b = np.zeros((num_classes,), dtype=np.float32)
        self.W = W
        self.b = b
        self.num_classes = num_classes
        self.dim = dim

    # ------------------------------------------------------------------

    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b])

    @params.setter
    def params(self, v: np.ndarray) -> None:
        self.W = v[: self.num_classes * self.dim].reshape(self.num_classes, self.dim).astype(np.float32)
        self.b = v[self.num_classes * self.dim :].astype(np.float32)

    def n_params(self) -> int:
        return self.num_classes * self.dim + self.num_classes

    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W.T + self.b  # (B, C)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------

    def nabla_cross_entropy(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Returns the *average* per-sample gradient (so that summing across
        clients corresponds to FedAvg aggregation)."""
        z = self.forward(X)
        p = softmax(z)
        # one-hot
        oh = np.zeros_like(p)
        oh[np.arange(y.shape[0]), y] = 1.0
        diff = (p - oh) / y.shape[0]
        gW = diff.T @ X  # (C, D)
        gb = diff.sum(axis=0)  # (C,)
        return np.concatenate([gW.ravel(), gb])

    def loss_and_grad(
        self, X: np.ndarray, y: np.ndarray, l2: float = 0.0
    ) -> tuple:
        z = self.forward(X)
        p = softmax(z)
        loss = cross_entropy(p, y)
        # add l2
        if l2 > 0:
            loss = loss + float(l2 / 2 * (self.W ** 2).sum())
        # gradient
        diff = (p - _one_hot(y, self.num_classes)) / y.shape[0]
        gW = diff.T @ X + l2 * self.W
        gb = diff.sum(axis=0)
        return loss, np.concatenate([gW.ravel(), gb])


def _one_hot(y: np.ndarray, C: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], C), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh
