# SGD and Variants (MA-1006 research project)

Implementation and presentation of stochastic gradient descent and its
variants (Momentum, AdaGrad, RMSProp, Adam) for MA-1006 Introducción al
Análisis Numérico, Universidad de Costa Rica, I Ciclo 2026.

## Layout

- `presentation/` — self-contained HTML slide deck (Spanish). Open
  `index.html` by double-clicking it; everything (KaTeX, fonts, demo
  engine) is vendored locally, so it works with no network at all.
- `code/` — the NumPy package: optimizer step functions, test problems,
  experiment runner, and figure generation. This is the code deliverable
  linked from the report.
- `report/` — the written report (Spanish, IEEEtran conference format
  with a standalone cover page). Build with `report/build.sh`
  (pdflatex + bibtex); output is `report/main.pdf`. Figures are pulled
  directly from `code/figures/`, so run `code/scripts/run_all.py`
  first if they are missing.

## Presentation

Navigation: scroll or use the keys; both drive the same state.

- Scrolling snaps to slides AND walks each slide's scenes: a slide with
  N scenes spans N+1 scroll stops, its content pinned while you scroll
  through them. Scrolling back up rewinds the scenes.
- `→` / `Space` / `PageDown`: next scene of the current slide, then next slide
- `←` / `PageUp`: previous scene of the current slide, then previous slide
  (lands on the previous slide's last scene, not its first)
- `Home` / `End`: first / last slide
- `F`: fullscreen

Demo slides advance through choreographed, seeded scenes: every run is
deterministic, so rehearsal equals performance. Pressing `→` while a
scene is animating skips to its end state.

PDF export (backup for projector failures): Chrome → Print → destination
PDF, enable "Background graphics". Every demo canvas prints its final
composite frame.

## Code

Requires [uv](https://docs.astral.sh/uv/) (or any Python >= 3.10 with
numpy and matplotlib).

```sh
cd code
uv sync
uv run scripts/run_all.py   # sanity checks + regenerate all figures
```

`run_all.py` runs the descent checks, prints the Adam parity iterates,
regenerates every figure (light variants for the report, dark variants
for the deck), and copies the logistic-regression figure into
`presentation/assets/`.

### JS/NumPy parity

The demo engine (`presentation/js/playground.js`) mirrors
`code/src/sgd_variants/optimizers.py` line for line. To verify: compare
the printout of

```sh
uv run python -m sgd_variants.check
```

against the same five GD and Momentum iterates computed by the JS step
functions (they agree to 15 decimal places; last verified 2026-07-01).
The package additionally implements AdaGrad, RMSProp, and Adam for the
report's reference, though they are outside the talk's scope.

## Data

`code/data/banknote.csv` is the UCI Banknote Authentication dataset
(Lohweg, 2013), vendored so the repository runs offline: 1372 rows, 4
wavelet features, binary genuine/forged label.
