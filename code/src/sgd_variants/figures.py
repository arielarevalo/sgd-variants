"""Report and deck figures. Every figure has a light (report, 300 dpi)
and a dark (deck palette, 2x) variant. Axis labels and legends are in
Spanish because they are audience-facing; the palette hex values mirror
the deck's CSS custom properties exactly.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .problems import Ravine, Rosenbrock, LogisticRegression, _mulberry32, _gauss_pair
from .runner import run

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"

COLORS = {
    "GD": "#9FB2D3",
    "SGD": "#38BDF8",
    "SGD16": "#7DD3FC",
    "Momentum": "#FFB454",
    "good": "#3EE08F",
}

THEMES = {
    "dark": {
        "bg": "#0A0E1A", "panel": "#0E1526", "text": "#EDF2FF",
        "muted": "#9DAAC8", "grid": "#263455", "dpi": 220,
    },
    "light": {
        "bg": "#FFFFFF", "panel": "#FFFFFF", "text": "#1A2233",
        "muted": "#4A5568", "grid": "#D5DBE8", "dpi": 300,
    },
}


def _style(ax, th):
    ax.set_facecolor(th["panel"])
    # which="both": on log axes the visible labels (e.g. 6x10^-1, 2x10^-1) are
    # MINOR ticks and would otherwise stay matplotlib-default black, invisible
    # against the dark card. Colour both tick marks and their labels.
    ax.tick_params(axis="both", which="both", colors=th["muted"], labelsize=9)
    for s in ax.spines.values():
        s.set_color(th["grid"])
    ax.grid(True, color=th["grid"], alpha=0.5, linewidth=0.6)
    ax.xaxis.label.set_color(th["text"])
    ax.yaxis.label.set_color(th["text"])
    # the axis offset / exponent text is a separate artist from tick labels.
    ax.xaxis.get_offset_text().set_color(th["muted"])
    ax.yaxis.get_offset_text().set_color(th["muted"])
    ax.title.set_color(th["text"])


def _save(fig, name, theme, transparent=False):
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"{name}-{theme}.png"
    if transparent:
        # Figure and axes patches render fully transparent so the deck's own
        # card background shows through uniformly, with no seam between the
        # figure margin and the axes rectangle regardless of card color.
        fig.savefig(out, dpi=THEMES[theme]["dpi"], transparent=True, bbox_inches="tight")
    else:
        fig.savefig(out, dpi=THEMES[theme]["dpi"], facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
    plt.close(fig)
    return out


def _contour(ax, problem, th, xlim, ylim, log=False):
    xs = np.linspace(*xlim, 240)
    ys = np.linspace(*ylim, 160)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.array([[problem.loss(np.array([x, y])) for x in xs] for y in ys])
    Z = np.log10(ZZ - ZZ.min() + 1e-6) if log else ZZ
    ax.contour(XX, YY, Z, levels=12, colors=th["grid"], linewidths=0.7, alpha=0.9)


def fig_ravine(theme="light"):
    """F1: GD zigzag vs Momentum on the kappa=25 ravine."""
    th = THEMES[theme]
    problem = Ravine(25.0)
    runs = [
        ("GD", "gd", dict(lr=0.076), 90),
        ("Momentum", "momentum", dict(lr=0.02, beta=0.9), 150),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fig.patch.set_facecolor(th["bg"])
    _style(ax, th)
    _contour(ax, problem, th, (-2.4, 2.4), (-1.35, 1.35))
    for label, name, hp, iters in runs:
        h = run(problem, name, iters=iters, hp=hp)
        T = np.array(h.thetas)
        ax.plot(T[:, 0], T[:, 1], "o-", color=COLORS[label], lw=1.4, ms=2.2, label=label)
    ax.plot(0, 0, "+", color=th["text"], ms=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(facecolor=th["panel"], edgecolor=th["grid"], labelcolor=th["text"], fontsize=9)
    ax.set_title("Zigzag de GD contra la bola pesada (κ = 25)", fontsize=11)
    return _save(fig, "fig-ravine", theme)


def fig_rosenbrock(theme="light"):
    """F2: the race on Rosenbrock, same runs as the deck."""
    th = THEMES[theme]
    problem = Rosenbrock()
    runs = [
        ("GD", "gd", dict(lr=0.0016), 0.0),
        ("SGD", "sgd", dict(lr=0.0016), 0.15),
        ("Momentum", "momentum", dict(lr=0.0016, beta=0.9), 0.0),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.patch.set_facecolor(th["bg"])
    _style(ax, th)
    _contour(ax, problem, th, (-2.0, 2.0), (-0.9, 2.7), log=True)
    rng = np.random.default_rng(32)
    for label, name, hp, noise in runs:
        if noise:
            # gradient noise, mirroring the deck's synthetic SGD on Rosenbrock
            theta = problem.start.copy()
            T = [theta.copy()]
            for _ in range(600):
                g = problem.grad(theta)
                g = g + noise * np.linalg.norm(g) * rng.standard_normal(2)
                theta = theta - hp["lr"] * g
                T.append(theta.copy())
            T = np.array(T)
        else:
            h = run(problem, name, iters=600, hp=hp)
            T = np.array(h.thetas)
        ax.plot(T[:, 0], T[:, 1], color=COLORS[label], lw=1.5, label=label)
    ax.plot(1, 1, "+", color=th["text"], ms=10)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.9, 2.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(facecolor=th["panel"], edgecolor=th["grid"], labelcolor=th["text"], fontsize=9)
    ax.set_title("La carrera sobre Rosenbrock (600 iteraciones, α = 0.0016 para los tres)", fontsize=11)
    return _save(fig, "fig-rosenbrock", theme)


def fig_logistic(theme="light", epochs=30, batch=16):
    """F3: loss vs epochs on the banknote logistic regression (deck slide)."""
    th = THEMES[theme]
    problem = LogisticRegression()
    n = problem.n
    runs = [
        ("GD", "gd", dict(lr=0.5), None, epochs, 1),
        ("SGD", "sgd", dict(lr=0.05), 1, int(epochs * n), 200),
        ("SGD16", "sgd", dict(lr=0.2), 16, int(epochs * n / 16), 20),
        ("Momentum", "momentum", dict(lr=0.02, beta=0.9), 16, int(epochs * n / 16), 20),
    ]
    labels = {"GD": "GD", "SGD": "SGD · |B| = 1", "SGD16": "SGD · |B| = 16",
              "Momentum": "Momentum · |B| = 16"}
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    fig.patch.set_facecolor(th["bg"])
    _style(ax, th)
    accs = {}
    for key, name, hp, b, iters, rec in runs:
        h = run(problem, name, iters=iters, hp=hp, batch=b, seed=7, record_every=rec)
        ax.plot(h.epochs, h.losses, color=COLORS[key], lw=1.7, label=labels[key])
        accs[labels[key]] = problem.accuracy(h.thetas[-1])
    ax.set_yscale("log")
    ax.set_xlabel("épocas (pasadas por los datos)")
    ax.set_ylabel("riesgo empírico  f(θ)")
    ax.set_title(f"Regresión logística · UCI banknote (n = {n})", fontsize=11)
    ax.legend(facecolor=th["panel"], edgecolor=th["grid"], labelcolor=th["text"], fontsize=9)
    # Deck variant only: transparent so it blends into the .figimg card with
    # no rectangular seam (see fig_logistic docstring / _save).
    out = _save(fig, "fig-logistic", theme, transparent=(theme == "dark"))
    return out, accs


def fig_lr_sensitivity(theme="light", epochs=10, batch=16):
    """F4: final loss vs alpha for SGD and SGD+Momentum on banknote."""
    th = THEMES[theme]
    problem = LogisticRegression()
    n = problem.n
    iters = int(epochs * n / batch)
    alphas = np.logspace(-4, 0.5, 14)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fig.patch.set_facecolor(th["bg"])
    _style(ax, th)
    for key, name, extra in [("SGD16", "sgd", {}), ("Momentum", "momentum", {"beta": 0.9})]:
        finals = []
        for a in alphas:
            h = run(problem, name, iters=iters, hp=dict(lr=float(a), **extra), batch=batch,
                    seed=11, record_every=iters)
            f = h.final()
            finals.append(f if np.isfinite(f) else np.nan)
        label = "SGD · |B| = 16" if key == "SGD16" else "SGD + Momentum"
        ax.plot(alphas, finals, "o-", color=COLORS[key], lw=1.5, ms=4, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tasa de aprendizaje  α")
    ax.set_ylabel(f"riesgo tras {epochs} épocas")
    ax.set_title("Sensibilidad a α", fontsize=11)
    ax.legend(facecolor=th["panel"], edgecolor=th["grid"], labelcolor=th["text"], fontsize=9)
    return _save(fig, "fig-lr-sensitivity", theme)


def fig_early_stopping(theme="light"):
    """F5: train/validation curves for the degree-9 polynomial, with the
    early-stopping point. Same seeded experiment as the deck (playground.js)."""
    th = THEMES[theme]
    rng = _mulberry32(21)
    n_tr, n_va, deg = 14, 20, 9

    def target(x):
        return np.sin(2.4 * np.pi * x) * 0.75 + 0.4 * x

    def make_set(n):
        X, Y = [], []
        for _ in range(n):
            x = -1 + 2 * rng()
            X.append(x)
            Y.append(target(x) + _gauss_pair(rng)[0] * 0.35)
        return np.array(X), np.array(Y)

    xt, yt = make_set(n_tr)
    xv, yv = make_set(n_va)
    dim = deg + 1
    FT = np.vander(xt, dim, increasing=True)
    FV = np.vander(xv, dim, increasing=True)
    mu, sd = FT.mean(axis=0), FT.std(axis=0)
    mu[0], sd[0] = 0.0, 1.0
    sd[sd == 0] = 1.0
    XT, XV = (FT - mu) / sd, (FV - mu) / sd

    w = np.zeros(dim)
    lr, iters, rec = 0.06, 800, 2
    curve_t, curve_v = [], []
    for k in range(iters + 1):
        if k % rec == 0:
            curve_t.append(float(((XT @ w - yt) ** 2).mean()))
            curve_v.append(float(((XV @ w - yv) ** 2).mean()))
        g = 2 * XT.T @ (XT @ w - yt) / n_tr
        w = w - lr * g
    best = int(np.argmin(curve_v))
    its = np.arange(len(curve_t)) * rec

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    fig.patch.set_facecolor(th["bg"])
    _style(ax, th)
    ax.plot(its, curve_t, color=COLORS["SGD"], lw=1.7, label="entrenamiento")
    ax.plot(its, curve_v, color=COLORS["Momentum"], lw=1.7, label="validación")
    ax.axvline(best * rec, color=COLORS["good"], lw=1.2, ls="--")
    ax.plot(best * rec, curve_v[best], "o", color=COLORS["good"], ms=6)
    ax.annotate("detenerse aquí", (best * rec, curve_v[best]),
                textcoords="offset points", xytext=(10, 12),
                color=COLORS["good"], fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("iteraciones de entrenamiento")
    ax.set_ylabel("error cuadrático medio")
    ax.set_title("Parada temprana: polinomio grado 9 sobre 14 datos", fontsize=11)
    ax.legend(facecolor=th["panel"], edgecolor=th["grid"], labelcolor=th["text"], fontsize=9)
    return _save(fig, "fig-early-stopping", theme)


def generate_all():
    outputs = []
    accs = {}
    for theme in ("light", "dark"):
        outputs.append(fig_ravine(theme))
        outputs.append(fig_rosenbrock(theme))
        out, accs = fig_logistic(theme)
        outputs.append(out)
        outputs.append(fig_lr_sensitivity(theme))
        outputs.append(fig_early_stopping(theme))
    return outputs, accs
