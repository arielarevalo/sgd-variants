"""Experiment runner with honest cost accounting.

History records the full-data loss against *epochs* (gradient
evaluations divided by n), so GD and SGD comparisons are per unit of
work, not per iteration.
"""

from dataclasses import dataclass, field

import numpy as np

from .optimizers import OPTIMIZERS


@dataclass
class History:
    name: str
    thetas: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    epochs: list = field(default_factory=list)

    def final(self):
        return self.losses[-1]


def run(problem, opt_name, *, iters, hp=None, batch=None, seed=0, record_every=1):
    """Run one optimizer on one problem.

    batch=None uses the full gradient (GD); batch=k samples k terms per
    step via problem.minibatch_grad. Epoch cost per step is batch/n for
    finite-sum problems and 1 for closed-form surfaces.
    """
    opt = OPTIMIZERS[opt_name]
    hp = hp or {}
    rng = np.random.default_rng(seed)
    theta = np.array(problem.start, dtype=float)
    state = opt["init"](len(theta))

    n = getattr(problem, "n", None)
    epoch_per_step = (batch / n) if (batch and n) else 1.0

    hist = History(name=opt["label"])
    hist.thetas.append(theta.copy())
    hist.losses.append(problem.loss(theta))
    hist.epochs.append(0.0)

    for k in range(1, iters + 1):
        if batch:
            g = problem.minibatch_grad(theta, rng, batch)
        else:
            g = problem.grad(theta)
        theta, state = opt["step"](theta, g, state, **hp)
        if k % record_every == 0 or k == iters:
            hist.thetas.append(theta.copy())
            hist.losses.append(problem.loss(theta))
            hist.epochs.append(k * epoch_per_step)

    return hist
