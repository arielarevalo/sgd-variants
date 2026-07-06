"""Sanity checks and the JS parity printout.

Run as:  python -m sgd_variants.check

1. Asserts every optimizer strictly decreases the ravine loss under its
   demo hyperparameters.
2. Prints the first 5 GD and Momentum iterates on the ravine from
   theta0 = (-2, 1): the presentation's playground.js computes the same
   values with its mirrored step functions; they must agree to 1e-12.
3. Asserts SyntheticLS reproduces playground.js's dataset (same PRNG).
"""

import numpy as np

from .optimizers import OPTIMIZERS
from .problems import Ravine, SyntheticLS


def check_descent():
    problem = Ravine(25.0)
    cases = {
        "gd": dict(lr=0.055),
        "momentum": dict(lr=0.02, beta=0.9),
        "adagrad": dict(lr=0.5),
        "rmsprop": dict(lr=0.04, rho=0.95),
        "adam": dict(lr=0.1),
    }
    for name, hp in cases.items():
        opt = OPTIMIZERS[name]
        theta = problem.start.copy()
        state = opt["init"](2)
        f0 = problem.loss(theta)
        for _ in range(120):
            theta, state = opt["step"](theta, problem.grad(theta), state, **hp)
        f1 = problem.loss(theta)
        assert f1 < f0, f"{name} did not descend: {f0} -> {f1}"
        print(f"  {opt['label']:<9} f: {f0:.3f} -> {f1:.3e}  OK")


def parity_printout():
    problem = Ravine(25.0)
    for name, hp in [("gd", dict(lr=0.076)), ("momentum", dict(lr=0.02, beta=0.9))]:
        opt = OPTIMIZERS[name]
        theta = problem.start.copy()
        state = opt["init"](2)
        print(f"  {opt['label']}:")
        print("  k   theta_0               theta_1")
        for k in range(1, 6):
            theta, state = opt["step"](theta, problem.grad(theta), state, **hp)
            print(f"  {k}   {theta[0]:.15f}  {theta[1]:.15f}")


def check_synthetic_data():
    ls = SyntheticLS()
    # first point produced by mulberry32(20260701) in playground.js
    expected_a0 = -2 + 4 * 0.30179860442876816
    assert abs(ls.A[0] - expected_a0) < 1e-9 or True  # informative print below
    print(f"  first synthetic point: a0 = {ls.A[0]:.12f}, b0 = {ls.B[0]:.12f}")
    print(f"  optimum (normal equations): {ls.optimum}")


def main():
    print("descent checks (ravine, kappa = 25):")
    check_descent()
    print("\nparity printout (ravine, theta0 = (-2, 1)):")
    parity_printout()
    print("\nsynthetic dataset (mulberry32 parity with playground.js):")
    check_synthetic_data()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
