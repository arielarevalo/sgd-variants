"""Test problems: closed-form 2D surfaces and a real logistic regression task.

Every problem exposes:
    loss(theta)           full objective
    grad(theta)           full gradient
    minibatch_grad(theta, rng, batch)   sampled gradient (finite-sum problems)

SyntheticLS reproduces the presentation widget's dataset bit for bit by
using the same mulberry32 PRNG and Box-Muller transform as playground.js.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class Ravine:
    """f(x, y) = 1/2 (x^2 + kappa * y^2), the ill-conditioned quadratic."""

    def __init__(self, kappa=25.0):
        self.kappa = kappa
        self.start = np.array([-2.0, 1.0])
        self.optimum = np.array([0.0, 0.0])

    def loss(self, theta):
        x, y = theta
        return 0.5 * (x * x + self.kappa * y * y)

    def grad(self, theta):
        x, y = theta
        return np.array([x, self.kappa * y])


class Rosenbrock:
    """f(x, y) = (1-x)^2 + 100 (y - x^2)^2, the curved valley."""

    start = np.array([-1.4, 1.8])
    optimum = np.array([1.0, 1.0])

    def loss(self, theta):
        x, y = theta
        return (1 - x) ** 2 + 100 * (y - x * x) ** 2

    def grad(self, theta):
        x, y = theta
        return np.array([
            -2 * (1 - x) - 400 * x * (y - x * x),
            200 * (y - x * x),
        ])


def _mulberry32(seed):
    """The exact PRNG used by playground.js, for bit-identical data."""
    a = seed & 0xFFFFFFFF

    def rand():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) ^ t
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rand


def _gauss_pair(rand):
    u = max(rand(), 1e-12)
    v = rand()
    r = np.sqrt(-2 * np.log(u))
    return r * np.cos(2 * np.pi * v), r * np.sin(2 * np.pi * v)


class SyntheticLS:
    """Least squares over n synthetic points: y ~ -0.4 + 1.2 x + noise.

    f(theta) = 1/(2n) sum_i (theta_0 + theta_1 x_i - y_i)^2

    Same generating process (and PRNG) as the 'empirical' surface in the
    presentation's playground.js.
    """

    def __init__(self, n=80, seed=20260701):
        rand = _mulberry32(seed)
        A, B = [], []
        for _ in range(n):
            a = -2 + 4 * rand()
            noise = _gauss_pair(rand)[0] * 0.8
            A.append(a)
            B.append(-0.4 + 1.2 * a + noise)
        self.A = np.array(A)
        self.B = np.array(B)
        self.n = n
        # closed-form optimum via the normal equations
        X = np.column_stack([np.ones(n), self.A])
        self.optimum, *_ = np.linalg.lstsq(X, self.B, rcond=None)
        self.start = self.optimum + np.array([-2.7, 1.55])

    def _residuals(self, theta, idx=slice(None)):
        return theta[0] + theta[1] * self.A[idx] - self.B[idx]

    def loss(self, theta):
        r = self._residuals(theta)
        return float(r @ r) / (2 * self.n)

    def grad(self, theta):
        r = self._residuals(theta)
        return np.array([r.mean(), (r * self.A).mean()])

    def minibatch_grad(self, theta, rng, batch):
        idx = rng.integers(0, self.n, size=batch)
        r = self._residuals(theta, idx)
        return np.array([r.mean(), (r * self.A[idx]).mean()])


class LogisticRegression:
    """L2-regularized logistic regression on the UCI banknote dataset.

    f(theta) = 1/n sum_i log(1 + exp(-y_i x_i^T theta)) + lam/2 ||theta||^2

    with y in {-1, +1}, standardized features, and a bias column. Loads
    data/banknote.csv with np.loadtxt: no dependencies beyond NumPy.
    """

    def __init__(self, lam=1e-4, csv_path=None):
        raw = np.loadtxt(csv_path or DATA_DIR / "banknote.csv", delimiter=",")
        X = raw[:, :4]
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        self.X = np.column_stack([np.ones(len(X)), X])
        self.y = np.where(raw[:, 4] > 0.5, 1.0, -1.0)
        self.n, self.dim = self.X.shape
        self.lam = lam
        self.start = np.zeros(self.dim)

    def _margins(self, theta, idx=slice(None)):
        return self.y[idx] * (self.X[idx] @ theta)

    def loss(self, theta):
        z = self._margins(theta)
        # log(1 + exp(-z)) computed stably
        return float(np.logaddexp(0.0, -z).mean() + 0.5 * self.lam * theta @ theta)

    def _grad_at(self, theta, idx):
        z = self._margins(theta, idx)
        w = -self.y[idx] / (1.0 + np.exp(z))  # d/dz log(1+e^{-z}) * y
        return self.X[idx].T @ w / len(w) + self.lam * theta

    def grad(self, theta):
        return self._grad_at(theta, slice(None))

    def minibatch_grad(self, theta, rng, batch):
        idx = rng.integers(0, self.n, size=batch)
        return self._grad_at(theta, idx)

    def accuracy(self, theta):
        return float(((self.X @ theta > 0) == (self.y > 0)).mean())
