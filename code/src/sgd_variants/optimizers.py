"""Optimizer update rules as pure step functions over explicit state.

Each optimizer is a pair (init, step):
    init(dim) -> state
    step(theta, g, state, **hyperparams) -> (theta_next, state_next)

The bodies are kept under 10 lines on purpose: they appear verbatim on
the presentation's code slide and in the report, and playground.js
mirrors them line for line (verified by check.py / the deck's parity log).
"""

import numpy as np


def gd_init(dim):
    return ()


def gd_step(theta, g, state, lr=0.01):
    theta = theta - lr * g
    return theta, state


# SGD is gd_step fed a sampled (mini-batch) gradient: the update rule is
# identical; only the source of g changes. See runner.run(batch=...).


def momentum_init(dim):
    return np.zeros(dim)


def momentum_step(theta, g, v, lr=0.01, beta=0.9):
    v = beta * v + g
    theta = theta - lr * v
    return theta, v


def adagrad_init(dim):
    return np.zeros(dim)


def adagrad_step(theta, g, G, lr=0.01, eps=1e-8):
    G = G + g * g
    theta = theta - lr * g / (np.sqrt(G) + eps)
    return theta, G


def rmsprop_init(dim):
    return np.zeros(dim)


def rmsprop_step(theta, g, s, lr=0.01, rho=0.9, eps=1e-8):
    s = rho * s + (1 - rho) * g * g
    theta = theta - lr * g / (np.sqrt(s) + eps)
    return theta, s


def adam_init(dim):
    return (np.zeros(dim), np.zeros(dim), 0)


def adam_step(theta, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t += 1
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, (m, v, t)


OPTIMIZERS = {
    "gd": {"init": gd_init, "step": gd_step, "label": "GD"},
    "sgd": {"init": gd_init, "step": gd_step, "label": "SGD"},
    "momentum": {"init": momentum_init, "step": momentum_step, "label": "Momentum"},
    "adagrad": {"init": adagrad_init, "step": adagrad_step, "label": "AdaGrad"},
    "rmsprop": {"init": rmsprop_init, "step": rmsprop_step, "label": "RMSProp"},
    "adam": {"init": adam_init, "step": adam_step, "label": "Adam"},
}
