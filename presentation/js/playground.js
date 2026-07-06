/* playground.js
 * Scene-driven optimizer demos on 3D loss surfaces (Plotly gl3d).
 * No fetch, no ES modules: runs from file://.
 * Optimizer step functions mirror code/src/sgd_variants/optimizers.py
 * line for line; that parity is claimed on the code slide.
 * The only Spanish strings here are on-screen captions and labels.
 */
(function () {
  'use strict';

  // ---------- seeded PRNG (mulberry32) ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussPair(rng) { // Box-Muller
    const u = Math.max(rng(), 1e-12), v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)];
  }

  // ---------- optimizer step functions (mirrors of optimizers.py) ----------
  function gdStep(theta, g, state, hp) {
    theta = theta.map((th, j) => th - hp.lr * g[j]);
    return [theta, state];
  }
  // SGD is gdStep fed a sampled gradient; same update rule.
  function momentumStep(theta, g, state, hp) {
    let { v } = state;
    v = v.map((vi, j) => hp.beta * vi + g[j]);
    theta = theta.map((th, j) => th - hp.lr * v[j]);
    return [theta, { v }];
  }

  const METHODS = {
    gd:       { label: 'GD',       init: d => ({}),                          step: gdStep },
    sgd:      { label: 'SGD',      init: d => ({}),                          step: gdStep },
    momentum: { label: 'Momentum', init: d => ({ v: new Array(d).fill(0) }), step: momentumStep },
  };

  // ---------- test problems ----------
  // Synthetic least squares: y ~ -0.4 + 1.2 x + noise over 80 seeded points.
  // f(theta) = 1/(2n) sum (t0 + t1 x_i - y_i)^2  (mirrors problems.py SyntheticLS)
  const bowl = (function () {
    const n = 80, rng = mulberry32(20260701);
    const A = [], B = [];
    for (let i = 0; i < n; i++) {
      const a = -2 + 4 * rng();
      const noise = gaussPair(rng)[0] * 0.8;
      A.push(a); B.push(-0.4 + 1.2 * a + noise);
    }
    let sa = 0, sb = 0, saa = 0, sab = 0;
    for (let i = 0; i < n; i++) { sa += A[i]; sb += B[i]; saa += A[i] * A[i]; sab += A[i] * B[i]; }
    const det = n * saa - sa * sa;
    const opt = [(sb * saa - sa * sab) / det, (n * sab - sa * sb) / det];
    const res = (t0, t1, i) => t0 + t1 * A[i] - B[i];
    return {
      key: 'bowl',
      f: (x, y) => {
        let s = 0;
        for (let i = 0; i < n; i++) { const r = res(x, y, i); s += r * r; }
        return s / (2 * n);
      },
      grad: (x, y) => {
        let g0 = 0, g1 = 0;
        for (let i = 0; i < n; i++) { const r = res(x, y, i); g0 += r; g1 += r * A[i]; }
        return [g0 / n, g1 / n];
      },
      stochGrad: (x, y, batch, rng2) => {
        let g0 = 0, g1 = 0;
        for (let k = 0; k < batch; k++) {
          const i = Math.floor(rng2() * n);
          const r = res(x, y, i);
          g0 += r; g1 += r * A[i];
        }
        return [g0 / batch, g1 / batch];
      },
      n, A, B,
      domain: [opt[0] - 3.0, opt[0] + 3.0, opt[1] - 2.2, opt[1] + 2.2],
      start: [opt[0] - 2.5, opt[1] + 1.8],
      optimum: opt,
      zT: v => v,                       // no transform
      axes: ['θ₀ (sesgo)', 'θ₁ (pendiente)', 'f(θ)'],
      camera: { eye: { x: -1.55, y: -1.45, z: 0.9 } },
    };
  })();

  // Ill-conditioned quadratic: least squares with features on very
  // different scales. kappa = 25.
  const ravine = {
    key: 'ravine',
    f: (x, y) => 0.5 * (x * x + 25 * y * y),
    grad: (x, y) => [x, 25 * y],
    domain: [-2.4, 2.4, -1.2, 1.2],
    start: [-2.0, 1.0],
    optimum: [0, 0],
    zT: v => v,
    axes: ['θ₀', 'θ₁', 'f(θ)'],
    camera: { eye: { x: -0.9, y: -2.1, z: 1.25 } },
    aspect: { x: 1.5, y: 1, z: 0.5 },
  };

  const rosenbrock = {
    key: 'rosenbrock',
    f: (x, y) => (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x),
    grad: (x, y) => [
      -2 * (1 - x) - 400 * x * (y - x * x),
      200 * (y - x * x),
    ],
    domain: [-2.0, 2.0, -0.9, 2.6],
    start: [-1.4, 1.8],
    optimum: [1, 1],
    showStart: true,                    // static "salida" dot at scene 0
    zT: v => Math.log10(1 + v),         // huge range: log height, labeled as such
    axes: ['x', 'y', 'log₁₀(1 + f)'],
    // View from the open low-y side, elevated: the curved valley floor and
    // both arms read as one canyon, no wall in front of the camera.
    camera: { eye: { x: 0, y: -1.45, z: 1.2 }, center: { x: 0, y: -0.05, z: -0.06 } },
    // The optimum disk is a plain circle by default (see diskTrace), but
    // Rosenbrock's curvature at (1,1) is ~2500:1 anisotropic: a circle
    // there drapes into a visible kink. This ellipse's long axis follows
    // the valley tangent (d/dx of y = x^2 at x = 1 is 2, so direction
    // (1, 2)) and its short axis is capped tight across the steep wall.
    marker: { dir: [1, 2], a: 0.04, b: 0.01 },
  };

  const PROBLEMS = { bowl, ravine, rosenbrock };

  // ---------- palette ----------
  const css = getComputedStyle(document.documentElement);
  const color = name => css.getPropertyValue(name).trim();
  let C = {};
  function loadColors() {
    C = {
      gd: color('--c-gd'), sgd: color('--c-sgd'), sgdb: color('--c-sgdb'),
      mom: color('--c-mom'), good: color('--c-good'), bad: color('--c-bad'),
      accent: color('--accent'), line: color('--line'),
      muted: color('--muted'), muted2: color('--muted-2'), text: color('--text'),
      surface: color('--surface'),
    };
  }

  function fmt(x) {
    if (!isFinite(x)) return '∞';
    const ax = Math.abs(x);
    if (ax !== 0 && (ax < 1e-3 || ax >= 1e4)) return x.toExponential(2).replace('e', '·10^');
    if (ax >= 100) return String(Math.round(x));
    return (Math.round(x * 1000) / 1000).toString();
  }

  // Round tick values (0, then nice multiples) covering [0, max]: gives the
  // loss panel a readable axis at any budget (12, 90, 350, 600, ...).
  function niceTicks(max, count) {
    if (!(max > 0)) return [0];
    const raw = max / (count || 4);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const ticks = [];
    for (let t = 0; t <= max + 1e-9; t += step) ticks.push(Math.round(t));
    return ticks;
  }

  // Smallest "nice" axis maximum that sits comfortably ABOVE dataMax: a
  // multiple of a 1-2-5 tick step plus ~10% headroom, so niceTicks() lands
  // clean ticks and the curve's leading edge never touches the right border.
  // Called every animation frame with the x actually revealed so far, so the
  // axis grows with the data in discrete, rock-stable steps (a clean zoom-out
  // each time the curve crosses a threshold) instead of easing or rescaling
  // continuously. `floor` lifts the first few revealed frames off zero so one
  // or two points never render as a degenerate full-width vertical drop.
  function axisBudget(dataMax, floor) {
    const m = Math.max(dataMax || 0, floor || 0);
    if (!(m > 0)) return 1;
    const rawStep = m / 4;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    return Math.ceil((m * 1.1) / step) * step;
  }

  // ---------- runners ----------
  function makeRunner(problem, methodKey, hp, opts) {
    const method = METHODS[methodKey];
    const theta0 = (opts && opts.start) || problem.start;
    return {
      methodKey, method, hp,
      label: (opts && opts.label) || method.label,
      color: (opts && opts.color) || C[{ gd: 'gd', sgd: 'sgd', momentum: 'mom' }[methodKey]],
      theta: theta0.slice(),
      state: method.init(2),
      traj: [theta0.slice()],
      losses: [{ x: 0, f: problem.f(theta0[0], theta0[1]) }],
      stepsDone: 0,
      epochPerStep: (opts && opts.epochPerStep) || 1,
      batch: (opts && opts.batch) || 0,
      noise: (opts && opts.noise) || 0,
      rng: mulberry32((opts && opts.seed) || 7),
      dead: false,
      chip: null,
    };
  }

  function runnerStep(runner, problem) {
    if (runner.dead) return;
    const [x, y] = runner.theta;
    let g;
    if (runner.batch > 0 && problem.stochGrad) {
      g = problem.stochGrad(x, y, runner.batch, runner.rng);
    } else {
      g = problem.grad(x, y);
      if (runner.noise > 0) {
        const [n1, n2] = gaussPair(runner.rng);
        const mag = Math.hypot(g[0], g[1]);
        g = [g[0] + runner.noise * mag * n1, g[1] + runner.noise * mag * n2];
      }
    }
    const [theta, state] = runner.method.step(runner.theta, g, runner.state, runner.hp);
    runner.theta = theta; runner.state = state;
    runner.stepsDone += 1;
    if (runner.traj.length < 4000) runner.traj.push(theta.slice());
    const f = problem.f(theta[0], theta[1]);
    runner.losses.push({ x: runner.stepsDone * runner.epochPerStep, f });
    const [x0, x1, y0, y1] = problem.domain;
    if (!isFinite(f) || Math.abs(theta[0]) > 8 * (x1 - x0) || Math.abs(theta[1]) > 8 * (y1 - y0)) {
      runner.dead = true;
    }
  }

  // Run a whole scene's trajectory up front (deterministic), then reveal it.
  function computeTrajectory(problem, spec) {
    const r = makeRunner(problem, spec.method, spec.hp, spec);
    for (let k = 0; k < spec.steps && !r.dead; k++) runnerStep(r, problem);
    return r;
  }

  // ---------- 3D surface pane ----------
  const GRID = 76;

  function surfaceData(problem) {
    const [x0, x1, y0, y1] = problem.domain;
    const xs = [], ys = [], zs = [];
    for (let i = 0; i < GRID; i++) xs.push(x0 + (x1 - x0) * i / (GRID - 1));
    for (let j = 0; j < GRID; j++) ys.push(y0 + (y1 - y0) * j / (GRID - 1));
    for (let j = 0; j < GRID; j++) {
      const row = [];
      for (let i = 0; i < GRID; i++) row.push(problem.zT(problem.f(xs[i], ys[j])));
      zs.push(row);
    }
    return { xs, ys, zs };
  }

  class SurfacePane {
    constructor(div, problem) {
      this.div = div;
      this.problem = problem;
      this.built = false;
      this.dynCount = 0; // dynamic traces beyond the static ones
      this.staticCount = 2; // surface + optimum marker (build() may add more)
      const { xs, ys, zs } = surfaceData(problem);
      this.zmin = Math.min(...zs.map(r => Math.min(...r)));
      this.zmax = Math.max(...zs.map(r => Math.max(...r)));
      // deliberate visual cheat: everything drawn ON the surface floats a
      // little above the true values so lines and dots never clip into it.
      // One uniform amount applied via zOf() everywhere (paths, endpoints,
      // markers) so nothing floats higher than anything else.
      this.lift = (this.zmax - this.zmin) * 0.03;
      this.surface = { xs, ys, zs };
    }

    zOf(x, y) { return this.problem.zT(this.problem.f(x, y)) + this.lift; }

    // A filled disk draped flat on the surface around (cx, cy) in parameter
    // space: a fan of triangles from a center vertex to a rim of points,
    // every vertex at zOf(x, y), so the disk hugs the surface's own
    // curvature instead of floating above it like a symbol. Plain circle by
    // default; a problem may instead carry a `marker` {dir, a, b} to orient
    // and cap an ellipse for anisotropic curvature (see the rosenbrock
    // problem def) so the drape stays round-looking.
    diskTrace(cx, cy) {
      const p = this.problem;
      const [x0, x1, y0, y1] = p.domain;
      const N = 40;
      let ux, uy, vx, vy, ra, rb;
      if (p.marker) {
        const [dx, dy] = p.marker.dir;
        const len = Math.hypot(dx, dy);
        ux = dx / len; uy = dy / len; // long-axis unit vector
        vx = -uy; vy = ux;            // perpendicular (short-axis) unit vector
        ra = p.marker.a; rb = p.marker.b;
      } else {
        ra = rb = Math.max(x1 - x0, y1 - y0) * 0.012;
        ux = 1; uy = 0; vx = 0; vy = 1;
      }
      const x = [cx], y = [cy], z = [this.zOf(cx, cy)]; // vertex 0: center
      for (let i = 0; i < N; i++) { // vertices 1..N: the rim, not closed
        const t = (i / N) * 2 * Math.PI;
        const ca = ra * Math.cos(t), sb = rb * Math.sin(t);
        const px = cx + ca * ux + sb * vx, py = cy + ca * uy + sb * vy;
        x.push(px); y.push(py); z.push(this.zOf(px, py));
      }
      const fi = [], fj = [], fk = []; // explicit fan triangulation
      for (let f = 0; f < N; f++) {
        fi.push(0);
        fj.push(f + 1);
        fk.push(((f + 1) % N) + 1);
      }
      return {
        type: 'mesh3d', x, y, z, i: fi, j: fj, k: fk,
        color: C.text, flatshading: true,
        lighting: { ambient: 1, diffuse: 0, specular: 0 },
        hoverinfo: 'skip', name: '',
      };
    }

    build() {
      if (this.built) return;
      const p = this.problem;
      const { xs, ys, zs } = this.surface;
      const axis = title => ({
        title: { text: title, font: { size: 11, color: C.muted, family: 'IBM Plex Sans' } },
        tickfont: { size: 9, color: C.muted2, family: 'JetBrains Mono' },
        gridcolor: C.line, zerolinecolor: C.line, showbackground: false, nticks: 5,
        showspikes: false,
      });
      const traces = [
        {
          type: 'surface', x: xs, y: ys, z: zs,
          colorscale: [[0, '#101A33'], [0.45, '#1B2A52'], [1, '#33477E']],
          showscale: false, opacity: 0.96,
          // highlight: false on every axis kills the hover-following
          // highlight contour lines; the projected z-contours (the fixed
          // ring styling below) are a separate, always-on visual and are
          // untouched by this.
          contours: {
            x: { highlight: false },
            y: { highlight: false },
            z: { show: true, highlight: false, usecolormap: false, color: '#4A5F94', width: 1,
                 project: { z: true }, start: this.zmin, end: this.zmax,
                 size: (this.zmax - this.zmin) / 12 },
          },
          lighting: { ambient: 0.55, diffuse: 0.6, specular: 0.12, roughness: 0.85 },
          hoverinfo: 'skip',
        },
        // optimum marker: a filled disk draped flat on the surface, not a
        // floating symbol, so it reads as lying ON the target from any angle
        this.diskTrace(p.optimum[0], p.optimum[1]),
      ];
      if (p.showStart) {
        // static "salida" dot so start and goal both read at scene 0
        traces.push({
          type: 'scatter3d', mode: 'markers',
          x: [p.start[0]], y: [p.start[1]],
          z: [this.zOf(p.start[0], p.start[1])],
          marker: { size: 5.5, color: C.accent }, hoverinfo: 'skip', name: '',
        });
      }
      this.staticCount = traces.length;
      const layout = {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        showlegend: false,
        hovermode: false,
        scene: {
          xaxis: axis(p.axes[0]), yaxis: axis(p.axes[1]), zaxis: axis(p.axes[2]),
          bgcolor: 'rgba(0,0,0,0)',
          camera: p.camera,
          aspectmode: 'manual', aspectratio: p.aspect || { x: 1.25, y: 1, z: 0.62 },
        },
      };
      Plotly.newPlot(this.div, traces, layout, { displayModeBar: false, scrollZoom: false, responsive: true });
      this.built = true;
      this.bindWheelForward();
    }

    // Plotly's gl3d canvas grabs wheel events for its own use even with
    // scrollZoom off, which would otherwise swallow the deck's wheel-scroll
    // whenever the cursor sits over a plot. Forward the raw deltaY onto the
    // deck's scrollTop so wheel-driven navigation keeps working.
    bindWheelForward() {
      if (this.div.dataset.wheelBound) return;
      this.div.dataset.wheelBound = '1';
      const deck = document.getElementById('deck');
      if (!deck) return;
      this.div.addEventListener('wheel', e => { deck.scrollTop += e.deltaY; }, { passive: true });
    }

    // clip a trajectory for display so a divergent path exits gracefully
    clipped(traj) {
      const [x0, x1, y0, y1] = this.problem.domain;
      const mx = (x1 - x0) * 0.06, my = (y1 - y0) * 0.06;
      const out = { x: [], y: [], z: [] };
      for (const p of traj) {
        if (p[0] < x0 - mx || p[0] > x1 + mx || p[1] < y0 - my || p[1] > y1 + my) break;
        out.x.push(p[0]); out.y.push(p[1]); out.z.push(this.zOf(p[0], p[1]));
      }
      return out;
    }

    addPath(colorStr, ghost, withMarkers) {
      const idx = this.staticCount + this.dynCount;
      this.dynCount += 1;
      Plotly.addTraces(this.div, [
        {
          type: 'scatter3d', mode: withMarkers ? 'lines+markers' : 'lines',
          x: [], y: [], z: [],
          line: { color: colorStr, width: ghost ? 4 : 7 },
          marker: withMarkers ? { size: 2.6, color: colorStr } : undefined,
          opacity: ghost ? 0.45 : 1,
          hoverinfo: 'skip', name: '',
        },
        {
          type: 'scatter3d', mode: 'markers',
          x: [], y: [], z: [],
          marker: { size: 5, color: colorStr },
          opacity: ghost ? 0.45 : 1,
          hoverinfo: 'skip', name: '',
        },
      ]);
      this.dynCount += 1;
      return idx; // line trace index; marker is idx+1
    }

    setPath(idx, pts, upTo) {
      const k = upTo === undefined ? pts.x.length : Math.min(upTo, pts.x.length);
      const xs = pts.x.slice(0, k), ys = pts.y.slice(0, k), zs = pts.z.slice(0, k);
      const last = k > 0 ? k - 1 : 0;
      Plotly.restyle(this.div, { x: [xs], y: [ys], z: [zs] }, [idx]);
      Plotly.restyle(this.div, {
        // endpoint dot sits at the same height as the line: pts.z already
        // carries the pane's uniform lift via zOf()
        x: [k ? [pts.x[last]] : []], y: [k ? [pts.y[last]] : []],
        z: [k ? [pts.z[last]] : []],
      }, [idx + 1]);
    }

    dimPath(idx) {
      Plotly.restyle(this.div, { opacity: 0.45, 'line.width': 4 }, [idx]);
      Plotly.restyle(this.div, { opacity: 0.45 }, [idx + 1]);
    }

    clearPaths() {
      if (!this.built || this.dynCount === 0) return;
      const idxs = [];
      for (let k = 0; k < this.dynCount; k++) idxs.push(this.staticCount + k);
      Plotly.deleteTraces(this.div, idxs);
      this.dynCount = 0;
    }
  }

  // ---------- 2D loss panel (canvas, labeled axes) ----------
  class LossPanel {
    constructor(host, opts) {
      this.host = host;
      this.opts = opts || {};
      this.canvas = document.createElement('canvas');
      host.appendChild(this.canvas);
      this.series = []; // {color, pts:[{x,f}], ghost}
    }
    dpr() { return Math.min(window.devicePixelRatio || 1, 2); }
    resize() {
      const w = this.host.clientWidth, h = this.host.clientHeight;
      if (w < 10) return;
      this.canvas.width = Math.round(w * this.dpr());
      this.canvas.height = Math.round(h * this.dpr());
      this.draw();
    }
    setSeries(series, budget) {
      this.series = series;
      this.budget = budget;
      this.draw();
    }
    draw() {
      const W = this.canvas.width, H = this.canvas.height;
      if (W < 10) return;
      const ctx = this.canvas.getContext('2d');
      const d = this.dpr();
      ctx.clearRect(0, 0, W, H);
      const padL = 34 * d, padR = 8 * d, padT = 8 * d, padB = 34 * d;
      let fmin = Infinity, fmax = -Infinity;
      for (const s of this.series) for (const p of s.pts) {
        if (isFinite(p.f) && p.f > 0) { fmin = Math.min(fmin, p.f); fmax = Math.max(fmax, p.f); }
      }
      ctx.strokeStyle = 'rgba(38,52,85,0.95)'; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
      ctx.stroke();
      ctx.fillStyle = 'rgba(157,170,200,0.85)';
      ctx.font = (9.5 * d) + 'px "JetBrains Mono", monospace';
      // x-axis unit label, centered below the tick numbers
      ctx.textAlign = 'center';
      ctx.fillText(this.opts.xLabel || 'iteraciones', (padL + (W - padR)) / 2, H - 6 * d);
      // y label rotated
      ctx.save();
      ctx.translate(11 * d, H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText(this.opts.yLabel || 'f(θ)  · log', 0, 0);
      ctx.restore();
      ctx.textAlign = 'left';
      if (!this.series.length || !isFinite(fmin) || fmin === fmax) return;
      // x-budget supplied by the caller (refreshPanels): a "nice" value sized
      // to the data revealed so far with headroom, stepping up in discrete
      // jumps as the curve grows. The fallback keeps the panel drawable if a
      // budget ever fails to arrive.
      const budget = this.budget || Math.max(...this.series.map(s => s.pts[s.pts.length - 1].x), 1);
      const lmin = Math.log10(fmin), lmax = Math.log10(fmax);
      const X = x => padL + (W - padL - padR) * Math.min(1, x / budget);
      const Y = f => padT + (H - padT - padB) * (1 - (Math.log10(Math.max(f, fmin)) - lmin) / (lmax - lmin));
      // y extremes
      ctx.fillStyle = 'rgba(102,113,141,0.95)';
      ctx.textAlign = 'left';
      ctx.fillText(fmt(fmax), padL + 7 * d, padT + 9 * d);
      ctx.fillText(fmt(fmin), padL + 7 * d, H - padB - 4 * d);
      // x ticks: several nice round values with faint gridlines, so the
      // axis reads as a real scale rather than one lone number at the edge.
      ctx.textAlign = 'center';
      for (const t of niceTicks(budget, 4)) {
        const xp = X(t);
        ctx.strokeStyle = 'rgba(38,52,85,0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(xp, padT); ctx.lineTo(xp, H - padB); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(xp, H - padB); ctx.lineTo(xp, H - padB + 4 * d); ctx.stroke();
        ctx.fillText(String(t), xp, H - padB + 14 * d);
      }
      ctx.textAlign = 'left';
      for (const s of this.series) {
        ctx.strokeStyle = s.color;
        ctx.globalAlpha = s.ghost ? 0.4 : 1;
        ctx.lineWidth = 1.6 * d;
        ctx.beginPath();
        let started = false;
        const step = Math.max(1, Math.floor(s.pts.length / 400));
        for (let k = 0; k < s.pts.length; k += step) {
          const p = s.pts[k];
          if (!isFinite(p.f) || p.f <= 0) continue;
          if (!started) { ctx.moveTo(X(p.x), Y(p.f)); started = true; }
          else ctx.lineTo(X(p.x), Y(p.f));
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }
  }

  // ---------- early stopping chart (train vs validation) ----------
  // Degree-9 polynomial fit to 14 noisy points, full-batch GD: the classic
  // U-shaped validation curve. Deterministic (seeded; seed/lr/iters chosen so
  // the validation minimum lands visibly inside the chart).
  const earlyStop = (function () {
    const rng = mulberry32(21);
    const nTr = 14, nVa = 20, deg = 9;
    const target = x => Math.sin(2.4 * Math.PI * x) * 0.75 + 0.4 * x;
    function makeSet(n) {
      const X = [], Y = [];
      for (let i = 0; i < n; i++) {
        const x = -1 + 2 * rng();
        X.push(x);
        Y.push(target(x) + gaussPair(rng)[0] * 0.35);
      }
      return { X, Y };
    }
    const tr = makeSet(nTr), va = makeSet(nVa);
    const feats = x => { const row = []; for (let p = 0; p <= deg; p++) row.push(Math.pow(x, p)); return row; };
    const FT = tr.X.map(feats), FV = va.X.map(feats);
    // standardize features (fit on train)
    const dim = deg + 1;
    const mu = new Array(dim).fill(0), sd = new Array(dim).fill(0);
    for (let j = 1; j < dim; j++) {
      for (const r of FT) mu[j] += r[j] / nTr;
      for (const r of FT) sd[j] += (r[j] - mu[j]) ** 2 / nTr;
      sd[j] = Math.sqrt(sd[j]) || 1;
    }
    const norm = rows => rows.map(r => r.map((v, j) => j === 0 ? 1 : (v - mu[j]) / sd[j]));
    const XT = norm(FT), XV = norm(FV);
    const mse = (Xs, Ys, w) => {
      let s = 0;
      for (let i = 0; i < Xs.length; i++) {
        let p = 0;
        for (let j = 0; j < dim; j++) p += Xs[i][j] * w[j];
        s += (p - Ys[i]) ** 2;
      }
      return s / Xs.length;
    };
    let w = new Array(dim).fill(0);
    const lr = 0.06, iters = 800, rec = 2;
    const curveT = [], curveV = [];
    for (let k = 0; k <= iters; k++) {
      if (k % rec === 0) { curveT.push(mse(XT, tr.Y, w)); curveV.push(mse(XV, va.Y, w)); }
      const g = new Array(dim).fill(0);
      for (let i = 0; i < nTr; i++) {
        let p = 0;
        for (let j = 0; j < dim; j++) p += XT[i][j] * w[j];
        const e = 2 * (p - tr.Y[i]) / nTr;
        for (let j = 0; j < dim; j++) g[j] += e * XT[i][j];
      }
      w = w.map((wj, j) => wj - lr * g[j]);
    }
    let best = 0;
    for (let k = 1; k < curveV.length; k++) if (curveV[k] < curveV[best]) best = k;
    return { curveT, curveV, best, rec };
  })();

  function drawEarlyStop(canvas, progress) {
    const d = Math.min(window.devicePixelRatio || 1, 2);
    const host = canvas.parentElement;
    if (canvas.width !== Math.round(host.clientWidth * d)) {
      canvas.width = Math.round(host.clientWidth * d);
      canvas.height = Math.round(host.clientHeight * d);
    }
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    const { curveT, curveV, best, rec } = earlyStop;
    const padL = 52 * d, padR = 22 * d, padT = 22 * d, padB = 44 * d;
    const n = curveT.length;
    const upto = Math.max(2, Math.floor(n * progress));
    const all = curveT.concat(curveV).filter(v => isFinite(v) && v > 0);
    const fmin = Math.min(...all), fmax = Math.max(...all);
    const lmin = Math.log10(fmin), lmax = Math.log10(fmax);
    const X = k => padL + (W - padL - padR) * (k / (n - 1));
    const Y = f => padT + (H - padT - padB) * (1 - (Math.log10(f) - lmin) / (lmax - lmin));
    // axes
    ctx.strokeStyle = 'rgba(38,52,85,1)'; ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
    ctx.stroke();
    ctx.fillStyle = 'rgba(157,170,200,0.95)';
    ctx.font = (11 * d) + 'px "JetBrains Mono", monospace';
    ctx.fillText('iteraciones de entrenamiento', padL, H - 14 * d);
    ctx.save();
    ctx.translate(16 * d, H / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center';
    ctx.fillText('error cuadrático medio · log', 0, 0);
    ctx.restore();
    // curves
    const drawCurve = (curve, colorStr) => {
      ctx.strokeStyle = colorStr; ctx.lineWidth = 2 * d;
      ctx.beginPath();
      for (let k = 0; k < upto; k++) {
        const px = X(k), py = Y(curve[k]);
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    };
    drawCurve(curveT, C.sgd);
    drawCurve(curveV, C.mom);
    // legend with backdrop so it stays legible over the curves
    ctx.font = (11.5 * d) + 'px "IBM Plex Sans", sans-serif';
    ctx.fillStyle = 'rgba(14,21,38,0.88)';
    ctx.strokeStyle = 'rgba(38,52,85,1)';
    ctx.beginPath();
    ctx.roundRect(W - 205 * d, padT + 2 * d, 130 * d, 44 * d, 8 * d);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = C.sgd; ctx.fillText('entrenamiento', W - 190 * d, padT + 20 * d);
    ctx.fillStyle = C.mom; ctx.fillText('validación', W - 190 * d, padT + 38 * d);
    // early stop marker once revealed past it
    if (upto > best) {
      const bx = X(best), by = Y(curveV[best]);
      ctx.strokeStyle = C.good; ctx.setLineDash([5 * d, 5 * d]); ctx.lineWidth = 1.4 * d;
      ctx.beginPath(); ctx.moveTo(bx, padT); ctx.lineTo(bx, H - padB); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.good;
      ctx.beginPath(); ctx.arc(bx, by, 5 * d, 0, 2 * Math.PI); ctx.fill();
      ctx.font = 'bold ' + (12 * d) + 'px "IBM Plex Sans", sans-serif';
      ctx.fillText('detenerse aquí', bx + 10 * d, padT + 16 * d);
      ctx.font = (10.5 * d) + 'px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(157,170,200,0.9)';
      ctx.fillText('iteración ' + best * rec, bx + 10 * d, padT + 34 * d);
    }
  }

  // ---------- Playground (scene runner over a surface pane) ----------
  const REVEAL_FRAMES = 50; // target frames for any trajectory's reveal animation

  class Playground {
    constructor(root, preset) {
      this.root = root;
      this.preset = preset;
      this.problem = PROBLEMS[preset.problem];
      this.slide = root.closest('.slide');
      this.sceneIdx = 0;
      this.animating = false;
      this.raf = 0;
      this.active = []; // {runner, pts, traceIdx, revealed}
      this.printImg = null;
      this.buildDom();
      this.pane = new SurfacePane(this.plotDiv, this.problem);
      document.addEventListener('slidechange', e => this.onSlideChange(e.detail));
      document.addEventListener('scenechange', e => this.onSceneChange(e.detail));
    }

    buildDom() {
      this.root.innerHTML =
        '<div class="pg-main"><div class="pg-plot"></div><canvas class="pg-overlay"></canvas><img class="pg-print" alt="" /></div>' +
        '<div class="pg-side">' +
        '<div class="pg-caption"></div>' +
        '<div class="pg-legend"></div>' +
        (this.preset.loss === false ? '' : '<div class="pg-loss"></div>') +
        '</div>';
      this.plotDiv = this.root.querySelector('.pg-plot');
      this.overlay = this.root.querySelector('.pg-overlay');
      this.printImgEl = this.root.querySelector('.pg-print');
      this.captionEl = this.root.querySelector('.pg-caption');
      this.legendEl = this.root.querySelector('.pg-legend');
      const lossHost = this.root.querySelector('.pg-loss');
      this.lossPanel = lossHost ? new LossPanel(lossHost, { xLabel: this.preset.lossX, yLabel: 'f(θ) · log' }) : null;
      this.captionEl.innerHTML = this.preset.intro || '';
    }

    ensureBuilt() {
      if (!this.pane.built) {
        this.pane.build();
        if (this.lossPanel) this.lossPanel.resize();
      }
    }

    onSlideChange(detail) {
      if (detail.slide === this.slide) {
        this.ensureBuilt();
      } else {
        this.stopAnim();
      }
    }

    onSceneChange(detail) {
      if (detail.slide !== this.slide) return;
      this.ensureBuilt();
      if (detail.scene === this.sceneIdx) return;
      if (detail.scene < this.sceneIdx) {
        this.resetAll();
        if (detail.scene === 0) return;
      }
      while (this.sceneIdx < detail.scene) {
        if (this.animating) this.finishScene();
        this.sceneIdx += 1;
        const script = this.preset.scenes[this.sceneIdx - 1];
        if (script) this.startScene(script, this.sceneIdx < detail.scene);
      }
    }

    resetAll() {
      this.stopAnim();
      this.sceneIdx = 0;
      this.active = [];
      this.pane.clearPaths();
      this.legendEl.innerHTML = '';
      this.captionEl.innerHTML = this.preset.intro || '';
      this.overlay.classList.remove('on');
      if (this.lossPanel) this.lossPanel.setSeries([], null);
    }

    startScene(script, instant) {
      if (script.caption) this.captionEl.innerHTML = script.caption;
      if (script.chart) {
        // early-stopping chart scene: reveal the 2D overlay
        this.overlay.classList.add('on');
        this.chartProgress = 0;
        if (instant) { this.chartProgress = 1; drawEarlyStop(this.overlay, 1); this.schedulePrintImage(); return; }
        this.animating = true;
        const tick = () => {
          if (!this.animating) return;
          this.chartProgress = Math.min(1, this.chartProgress + 0.012);
          drawEarlyStop(this.overlay, this.chartProgress);
          if (this.chartProgress >= 1) { this.animating = false; this.schedulePrintImage(); return; }
          this.raf = requestAnimationFrame(tick);
        };
        tick();
        return;
      }
      if (script.run) {
        for (const a of this.active) { this.pane.dimPath(a.traceIdx); a.ghost = true; }
        for (const spec of script.run) {
          const runner = computeTrajectory(this.problem, spec);
          const pts = this.pane.clipped(runner.traj);
          const traceIdx = this.pane.addPath(runner.color, false, !!this.preset.markers);
          this.active.push({ runner, pts, traceIdx, revealed: 0, ghost: false });
          this.addChip(runner);
        }
      }
      if (script.focusLoss && this.lossPanel) {
        this.lossPanel.host.style.borderColor = C.accent;
      }
      if (instant) { this.finishScene(); return; }
      if (script.run) {
        this.animating = true;
        this.loop();
      } else {
        this.refreshPanels();
        this.onSceneDone();
      }
    }

    currentScene() { return this.preset.scenes[this.sceneIdx - 1]; }

    loop() {
      if (!this.animating) return;
      const fresh = this.active.filter(a => !a.ghost);
      let allDone = true;
      for (const a of fresh) {
        if (a.revealed < a.pts.x.length) {
          // Reveal over roughly REVEAL_FRAMES frames regardless of how many
          // points the trajectory has: short, fast-converging runs and
          // long, noisy ones take about the same wall-clock time to draw,
          // so pacing feels consistent instead of short runs "popping" in.
          const per = Math.max(a.pts.x.length / REVEAL_FRAMES, 0.34);
          a.revealed = Math.min(a.pts.x.length, a.revealed + per);
          this.pane.setPath(a.traceIdx, a.pts, a.revealed);
          if (a.revealed < a.pts.x.length) allDone = false;
        }
      }
      this.refreshPanels(true);
      if (allDone) { this.animating = false; this.onSceneDone(); return; }
      this.raf = requestAnimationFrame(() => this.loop());
    }

    finishScene() {
      const script = this.currentScene();
      if (script && script.chart) {
        this.animating = false;
        cancelAnimationFrame(this.raf);
        this.chartProgress = 1;
        drawEarlyStop(this.overlay, 1);
        this.schedulePrintImage();
        return;
      }
      for (const a of this.active) {
        if (!a.ghost && a.revealed < a.pts.x.length) {
          a.revealed = a.pts.x.length;
          this.pane.setPath(a.traceIdx, a.pts, a.revealed);
        }
      }
      this.animating = false;
      cancelAnimationFrame(this.raf);
      this.refreshPanels();
      this.onSceneDone();
    }

    onSceneDone() {
      this.refreshPanels();
      const script = this.currentScene();
      if (script && script.after === 'ranking') this.showRanking();
      this.schedulePrintImage();
    }

    showRanking() {
      const rs = this.active.slice().sort((a, b) => {
        const fa = a.runner.dead ? Infinity : a.runner.losses[a.runner.losses.length - 1].f;
        const fb = b.runner.dead ? Infinity : b.runner.losses[b.runner.losses.length - 1].f;
        return fa - fb;
      });
      this.legendEl.innerHTML = '';
      rs.forEach((a, k) => {
        this.addChip(a.runner);
        a.runner.chip.querySelector('.name').textContent = (k + 1) + '. ' + a.runner.label;
      });
      this.refreshPanels();
    }

    addChip(runner) {
      const chip = document.createElement('div');
      chip.className = 'pg-chip';
      chip.innerHTML = '<span class="swatch" style="background:' + runner.color + '"></span>' +
        '<span class="name">' + runner.label + '</span><span class="val">·</span>';
      this.legendEl.appendChild(chip);
      runner.chip = chip;
    }

    refreshPanels(cheap) {
      for (const a of this.active) {
        const r = a.runner;
        if (!r.chip) continue;
        const frac = a.pts.x.length ? a.revealed / a.pts.x.length : 1;
        const li = Math.max(0, Math.min(r.losses.length - 1, Math.floor(frac * (r.losses.length - 1))));
        const last = r.losses[li];
        const finished = a.revealed >= a.pts.x.length;
        r.chip.classList.toggle('dead', r.dead && finished);
        r.chip.querySelector('.val').textContent =
          (r.dead && finished) ? '✗ divergió' : 'f = ' + fmt(last.f);
      }
      if (!this.lossPanel) return;
      // Final x of the widest series in the scene. The runners are computed to
      // completion at scene start, so this is known up front; ghost/complete
      // series from earlier scenes count in FULL (a later scene may legitimately
      // open with a wide axis because a ghost already spans it). preset.lossBudget
      // is intentionally ignored: honoring it reproduced the "full-size axis from
      // frame one" look the user rejected. The axis grows from the data instead.
      const finalMax = this.active.length
        ? Math.max(1, ...this.active.map(a => a.runner.losses[a.runner.losses.length - 1].x)) : 1;
      if (!cheap) {
        // Settled (scene done, or a non-animated / rewind-instant scene): size
        // the axis to the full data so it snaps to its final nice value.
        this.lossPanel.setSeries(
          this.active.map(a => ({ color: a.runner.color, pts: a.runner.losses, ghost: a.ghost })),
          axisBudget(finalMax, 0)
        );
        return;
      }
      // Animating: size the axis to the x actually revealed so far, floored at a
      // fraction of the scene's final width so the opening points don't span the
      // whole panel. axisBudget() turns that into a stable nice value that steps
      // up as the curve grows.
      let revealedMax = 0;
      const series = this.active.map(a => {
        const frac = a.pts.x.length ? a.revealed / a.pts.x.length : 1;
        const upto = Math.max(1, Math.floor(frac * a.runner.losses.length));
        const slice = a.runner.losses.slice(0, upto);
        revealedMax = Math.max(revealedMax, slice[slice.length - 1].x);
        return { color: a.runner.color, pts: slice, ghost: a.ghost };
      });
      this.lossPanel.setSeries(series, axisBudget(revealedMax, finalMax * 0.2));
    }

    stopAnim() {
      if (this.animating) {
        this.animating = false;
        cancelAnimationFrame(this.raf);
      }
    }

    // Keep a static PNG of the current plot for reliable PDF export.
    schedulePrintImage() {
      if (!this.pane.built || !window.Plotly || !Plotly.toImage) return;
      clearTimeout(this.printT);
      this.printT = setTimeout(() => {
        Plotly.toImage(this.plotDiv, { format: 'png', width: 1100, height: 500 })
          .then(url => { this.printImgEl.src = url; })
          .catch(() => {});
      }, 500);
    }

    consumePress() {
      if (this.animating) { this.finishScene(); return true; }
      return false;
    }
  }

  // ---------- anchor demo (slide "el paisaje de pérdida") ----------
  // Left: the 80 data points and the candidate line for theta.
  // Right: the loss bowl with a marker at theta. One story, two views.
  class AnchorDemo {
    constructor() {
      this.root = document.getElementById('anchor');
      if (!this.root) return;
      this.slide = this.root.closest('.slide');
      this.plotDiv = this.root.querySelector('.anchor-plot');
      this.fitCanvas = this.root.querySelector('.anchor-fit');
      this.captionEl = this.slide.querySelector('.anchor-caption');
      this.pane = new SurfacePane(this.plotDiv, bowl);
      this.printImgEl = this.root.querySelector('.pg-print');
      this.markerIdx = null;
      this.lastScene = 0;
      this.raf = 0;
      this.thetas = [
        [bowl.optimum[0] + 2.5, bowl.optimum[1] + 1.5],
        [bowl.optimum[0] + 0.9, bowl.optimum[1] + 0.75],
        bowl.optimum.slice(),
      ];
      this.captions = [
        'Cada punto del plano de abajo es UNA recta candidata. Esta exagera la pendiente y corre el sesgo: errores enormes (segmentos rosados), así que su punto vive en la ladera alta del tazón.',
        'Otra candidata, más razonable: menos error, y su punto está más abajo en la superficie. Bajar por el tazón ES mejorar la recta.',
        'La mejor recta posible para estos datos: el fondo del tazón. Entrenar un modelo ES buscar este punto, sin poder ver el tazón completo.',
      ];
      this.intro = 'Un modelo de dos parámetros: recta y = θ₀ + θ₁·x.';
      document.addEventListener('slidechange', e => {
        if (e.detail.slide === this.slide) this.ensureBuilt();
      });
      document.addEventListener('scenechange', e => {
        if (e.detail.slide !== this.slide) return;
        this.ensureBuilt();
        this.setScene(e.detail.scene);
      });
    }

    ensureBuilt() {
      if (this.pane.built) return;
      this.pane.build();
      Plotly.addTraces(this.plotDiv, [
        { // 2: path along the surface connecting visited candidates
          type: 'scatter3d', mode: 'lines',
          x: [], y: [], z: [],
          line: { color: C.good, width: 6 },
          hoverinfo: 'skip', name: '',
        },
        { // 3: the candidate dot, riding the surface at zOf() like any
          // trajectory endpoint dot elsewhere; no elevation
          type: 'scatter3d', mode: 'markers',
          x: [], y: [], z: [],
          marker: { size: 6, color: C.good },
          hoverinfo: 'skip', name: '',
        },
      ]);
      this.pathIdx = 2;
      this.markerIdx = 3;
      // full surface-following polyline through the three candidates
      this.pathPts = this.samplePath();
      this.drawFit(null);
      this.captionEl.innerHTML = this.intro;
      this.schedulePrintImage();
    }

    samplePath() {
      const pts = { x: [], y: [], z: [] };
      const lift = 0; // zOf already carries the pane's visual lift
      for (let s = 0; s < this.thetas.length - 1; s++) {
        const a = this.thetas[s], b = this.thetas[s + 1];
        const K = 36;
        for (let k = s === 0 ? 0 : 1; k <= K; k++) {
          const t = k / K;
          const x = a[0] + (b[0] - a[0]) * t;
          const y = a[1] + (b[1] - a[1]) * t;
          pts.x.push(x); pts.y.push(y);
          pts.z.push(this.pane.zOf(x, y) + lift);
        }
      }
      return pts; // 37 points for segment 1, 36 more for segment 2
    }

    setPin(th) {
      if (!th) {
        Plotly.restyle(this.plotDiv, { x: [[]], y: [[]], z: [[]] }, [this.markerIdx]);
        return;
      }
      this.setPinXYZ(th[0], th[1], this.pane.zOf(th[0], th[1]));
    }

    setPinXYZ(x, y, z) {
      Plotly.restyle(this.plotDiv, { x: [[x]], y: [[y]], z: [[z]] }, [this.markerIdx]);
    }

    // Fit line + caption f(θ) both describe whatever theta the dot is
    // CURRENTLY at, so they stay in lockstep with it: during an animated
    // reveal that's the leading edge of the path, not the destination.
    updateReadout(th) {
      this.drawFit(th);
      this.captionEl.innerHTML = this.captionPrefix +
        ' <span class="mono">f(θ) = ' + fmt(bowl.f(th[0], th[1])) + '</span>';
    }

    // reveal the surface path up to candidate k (1-based), animating the
    // newly visible stretch. The pin and the fit/caption readout ride the
    // leading edge of the reveal every frame, so nothing arrives ahead of
    // the line; instant reveals (rewind, multi-scene skip) jump everything
    // together in one step.
    revealPath(k, instant) {
      cancelAnimationFrame(this.raf);
      const total = k <= 1 ? 0 : (k === 2 ? 37 : this.pathPts.x.length);
      const setLen = len => {
        Plotly.restyle(this.plotDiv, {
          x: [this.pathPts.x.slice(0, len)],
          y: [this.pathPts.y.slice(0, len)],
          z: [this.pathPts.z.slice(0, len)],
        }, [this.pathIdx]);
        if (len > 0) {
          const last = len - 1;
          const x = this.pathPts.x[last], y = this.pathPts.y[last];
          this.setPinXYZ(x, y, this.pathPts.z[last]);
          this.updateReadout([x, y]);
        }
      };
      if (instant || total === 0) { setLen(total); return; }
      const from = k === 2 ? 0 : 37;
      let len = from;
      const tick = () => {
        len = Math.min(total, len + 2);
        setLen(len);
        if (len < total) this.raf = requestAnimationFrame(tick);
      };
      tick();
    }

    schedulePrintImage() {
      if (!this.pane.built || !window.Plotly || !Plotly.toImage || !this.printImgEl) return;
      clearTimeout(this.printT);
      this.printT = setTimeout(() => {
        Plotly.toImage(this.plotDiv, { format: 'png', width: 900, height: 520 })
          .then(url => { this.printImgEl.src = url; })
          .catch(() => {});
      }, 500);
    }

    setScene(k) {
      const prev = this.lastScene;
      this.lastScene = k;
      if (k === 0) {
        this.setPin(null);
        this.revealPath(0, true);
        this.drawFit(null);
        this.captionEl.innerHTML = this.intro;
        return;
      }
      const th = this.thetas[k - 1];
      this.captionPrefix = this.captions[k - 1];
      // Sets the pin/readout to the destination immediately; revealPath's
      // first animated frame (or its instant jump) supersedes this before
      // the browser paints, except at scene 1 where there is no path yet
      // to derive a leading edge from, so this is the only positioning.
      this.setPin(th);
      this.updateReadout(th);
      this.revealPath(k, k < prev || k - prev > 1);
      this.schedulePrintImage();
    }

    drawFit(theta) {
      const d = Math.min(window.devicePixelRatio || 1, 2);
      const host = this.fitCanvas.parentElement;
      this.fitCanvas.width = Math.round(host.clientWidth * d);
      this.fitCanvas.height = Math.round(host.clientHeight * d);
      const W = this.fitCanvas.width, H = this.fitCanvas.height;
      const ctx = this.fitCanvas.getContext('2d');
      ctx.clearRect(0, 0, W, H);
      const pad = 26 * d;
      const xr = [-2.2, 2.2];
      let ymin = Math.min(...bowl.B), ymax = Math.max(...bowl.B);
      const yPad = (ymax - ymin) * 0.12;
      ymin -= yPad; ymax += yPad;
      const X = a => pad + (W - 2 * pad) * (a - xr[0]) / (xr[1] - xr[0]);
      const Y = b => pad + (H - 2 * pad) * (1 - (b - ymin) / (ymax - ymin));
      // axes
      ctx.strokeStyle = 'rgba(38,52,85,1)'; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad, H - pad);
      ctx.stroke();
      ctx.fillStyle = 'rgba(157,170,200,0.9)';
      ctx.font = (10 * d) + 'px "JetBrains Mono", monospace';
      ctx.fillText('x (entrada)', pad + 4 * d, H - 8 * d);
      ctx.save();
      ctx.translate(11 * d, H / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center';
      ctx.fillText('y (salida)', 0, 0);
      ctx.restore();
      // residuals then data then line, so the line reads on top
      if (theta) {
        ctx.strokeStyle = 'rgba(248,109,160,0.55)';
        ctx.lineWidth = 1 * d;
        for (let i = 0; i < bowl.n; i++) {
          const pred = theta[0] + theta[1] * bowl.A[i];
          ctx.beginPath();
          ctx.moveTo(X(bowl.A[i]), Y(bowl.B[i]));
          ctx.lineTo(X(bowl.A[i]), Y(Math.max(ymin, Math.min(ymax, pred))));
          ctx.stroke();
        }
      }
      ctx.fillStyle = 'rgba(157,178,211,0.9)';
      for (let i = 0; i < bowl.n; i++) {
        ctx.beginPath();
        ctx.arc(X(bowl.A[i]), Y(bowl.B[i]), 2.4 * d, 0, 2 * Math.PI);
        ctx.fill();
      }
      if (theta) {
        ctx.strokeStyle = C.good;
        ctx.lineWidth = 2.2 * d;
        ctx.beginPath();
        ctx.moveTo(X(xr[0]), Y(theta[0] + theta[1] * xr[0]));
        ctx.lineTo(X(xr[1]), Y(theta[0] + theta[1] * xr[1]));
        ctx.stroke();
      }
    }
  }

  // ---------- presets ----------
  function buildPresets() {
    return {
      'gd-alpha': {
        problem: 'bowl',
        lossX: 'iteraciones',
        intro: 'Cuatro corridas desde el mismo punto sobre el mismo tazón. Lo único que cambia es α.',
        scenes: [
          { caption: '<span class="mono">α = 0.015</span> · demasiado tímida: 80 iteraciones y sigue en la ladera. Correcta pero <b>carísima</b>.',
            run: [{ method: 'gd', hp: { lr: 0.015 }, steps: 80, label: 'α = 0.015', seed: 1, color: '#6E82A6' }] },
          { caption: '<span class="mono">α = 0.8</span> · bien calibrada: baja directo al fondo en unas decenas de iteraciones.',
            run: [{ method: 'gd', hp: { lr: 0.8 }, steps: 50, label: 'α = 0.8', seed: 2, color: '#C7D6EF' }] },
          { caption: '<span class="mono">α = 1.25</span> · cerca del límite de estabilidad <span style="white-space:nowrap">2/λ<sub>max</sub> ≈ 1.375</span>: <b>rebota</b> de ladera a ladera, y aun así converge.',
            run: [{ method: 'gd', hp: { lr: 1.25 }, steps: 70, label: 'α = 1.25', seed: 3, color: '#FFB454' }] },
          { caption: '<span class="mono">α = 1.381</span> · apenas sobre el límite <span style="white-space:nowrap">2/λ<sub>max</sub> ≈ 1.375</span>: tras acomodarse, cada rebote sube más que el anterior hasta escapar del tazón. El método <b>diverge</b>.',
            run: [{ method: 'gd', hp: { lr: 1.381 }, steps: 350, label: 'α = 1.381', seed: 4, color: '#F86DA0' }] },
        ],
      },

      'sgd-vs': {
        problem: 'bowl',
        lossX: 'épocas',
        lossBudget: 12,
        intro: 'El mismo tazón, ahora contado por épocas: una época es una pasada completa por los 80 datos.',
        scenes: [
          { caption: '<b>GD exacto</b> · cada iteración recorre los 80 datos: una época entera por iteración. En 12 épocas, solo <span class="mono">12 iteraciones</span>, perfectas y carísimas.',
            run: [{ method: 'gd', hp: { lr: 0.8 }, steps: 12, label: 'GD · todos los datos', seed: 5 }] },
          { caption: '<b>SGD</b> · un dato por iteración: <span class="mono">80 iteraciones</span> por época. En 12 épocas, <span class="mono">960 iteraciones</span>, ruidosas y baratas.',
            run: [{ method: 'sgd', hp: { lr: 0.05 }, steps: 960, batch: 1, epochPerStep: 1 / 80, label: 'SGD · un dato', seed: 6 }] },
          { caption: '<b>La lectura</b> · el ruido hace temblar la trayectoria, pero el estimador apunta bien en promedio: por época consumida, SGD llegó mucho antes.',
            focusLoss: true },
        ],
      },

      'batches': {
        problem: 'bowl',
        lossX: 'épocas',
        lossBudget: 12,
        intro: 'Dos perillas prácticas sobre el mismo SGD: el tamaño del lote y cuándo detenerse.',
        scenes: [
          { caption: '<b>|B| = 1</b> · el caso extremo: máximo ahorro, máximo ruido. Cerca del fondo <b>no se asienta</b>: sigue temblando.',
            run: [{ method: 'sgd', hp: { lr: 0.05 }, steps: 960, batch: 1, epochPerStep: 1 / 80, label: '|B| = 1', seed: 6 }] },
          { caption: '<b>|B| = 16</b> · promediar 16 muestras corta la varianza (∝ 1/|B|): casi la misma velocidad por época, trayectoria mucho más limpia.',
            run: [{ method: 'sgd', hp: { lr: 0.25 }, steps: 60, batch: 16, epochPerStep: 16 / 80, label: '|B| = 16', seed: 7, color: null }] },
          { caption: '<b>Parada temprana</b> · en un modelo con más capacidad (polinomio grado 9, 14 datos), entrenar de más <b>memoriza el ruido</b>: el error de validación baja… y vuelve a subir. Detenerse en su mínimo es regularización gratuita.',
            chart: true },
        ],
      },

      'zigzag': {
        problem: 'ravine',
        markers: true,
        lossX: 'iteraciones',
        intro: 'Un tazón estirado: κ = 25. El método arranca en la pared alta del cañón.',
        scenes: [
          { caption: 'El gradiente apunta casi perpendicular al valle: <b>90 iteraciones</b> cruzando de pared a pared para avanzar lo que una diagonal haría en diez. Es geometría, no ruido: lo hereda cualquier (S)GD.',
            run: [{ method: 'gd', hp: { lr: 0.076 }, steps: 90, label: 'GD · α = 0.076', seed: 8 }] },
        ],
      },

      'momentum-vs': {
        problem: 'ravine',
        markers: true,
        lossX: 'iteraciones',
        intro: 'Misma α para ambos. Lo único que cambia es la memoria β.',
        scenes: [
          { caption: '<span class="mono">β = 0</span> · sin memoria, la actualización es <b>exactamente GD</b>: el zigzag de siempre.',
            run: [{ method: 'momentum', hp: { lr: 0.076, beta: 0 }, steps: 90, label: 'β = 0 (GD)', color: null, seed: 9 }] },
          { caption: '<span class="mono">β = 0.9</span> · la velocidad promedia los últimos ~10 gradientes: lo transversal <b>se cancela</b>, lo longitudinal <b>se acumula</b>.',
            run: [{ method: 'momentum', hp: { lr: 0.076, beta: 0.9 }, steps: 90, label: 'β = 0.9', seed: 10 }] },
          { caption: '<b>La lectura</b> · misma tasa, mismas 90 iteraciones: sin memoria, GD cruza el cañón a rebotes; con memoria, la bola pesada amortigua el rebote y baja por el valle.',
            focusLoss: true },
        ],
      },

      // Rosenbrock intro + three-way race on one slide: scene 0 presents the
      // benchmark (start and goal markers are static on the surface), scene 1
      // runs the race, scene 2 shows the ranking.
      'rosen-race': {
        problem: 'rosenbrock',
        lossX: 'iteraciones',
        intro: '<b>La pista</b> · la salida (punto morado) está en (-1.4, 1.8), sobre el brazo izquierdo del valle; la meta (disco blanco) en (1, 1). Los tres métodos parten del mismo punto, con 600 iteraciones y α afinada. Un buen optimizador debe bajar al valle y saber doblar. Ojo: la altura está en escala log₁₀.',
        scenes: [
          { caption: '<b>En vivo</b> · el panel muestra f(θ) en escala logarítmica. SGD corre sobre el gradiente con ruido inyectado (aquí no hay datos que muestrear).',
            run: [
              { method: 'gd',       hp: { lr: 0.0016 },            steps: 600, seed: 21 },
              { method: 'sgd',      hp: { lr: 0.0016 },            steps: 600, seed: 32, noise: 0.15 },
              { method: 'momentum', hp: { lr: 0.0016, beta: 0.9 }, steps: 600, seed: 23 },
            ] },
          { caption: '<b>Foto final</b> · la memoria de Momentum dobló la curva que a GD le costó todo el presupuesto. El orden de hoy es reproducible: misma semilla, mismas iteraciones.',
            after: 'ranking' },
        ],
      },
    };
  }

  // ---------- namespace ----------
  const instances = [];
  let anchor = null;

  window.PG = {
    boot() {
      loadColors();
      const presets = buildPresets();
      // colors that depend on loadColors()
      presets['batches'].scenes[1].run[0].color = C.sgdb;
      presets['momentum-vs'].scenes[0].run[0].color = C.gd;
      document.querySelectorAll('.pg[data-cfg]').forEach(root => {
        const preset = presets[root.dataset.cfg];
        if (preset) instances.push(new Playground(root, preset));
      });
      anchor = new AnchorDemo();
      // Build the panes of the first slides eagerly so nothing is blank.
      setTimeout(() => {
        if (anchor && anchor.root) anchor.ensureBuilt();
        instances.forEach(p => p.ensureBuilt());
        if (window.dispatchEvent) window.dispatchEvent(new Event('resize'));
      }, 60);
      let rzT = 0;
      window.addEventListener('resize', () => {
        clearTimeout(rzT);
        rzT = setTimeout(() => {
          instances.forEach(p => { if (p.lossPanel) p.lossPanel.resize(); });
          if (anchor && anchor.root && anchor.pane.built) {
            anchor.drawFit(anchor.lastScene ? anchor.thetas[anchor.lastScene - 1] : null);
          }
        }, 150);
      });
    },
    consumeForwardPress(slideEl) {
      for (const p of instances) {
        if (slideEl.contains(p.root) && p.consumePress()) return true;
      }
      return false;
    },
    pauseAll() {
      instances.forEach(p => p.stopAnim());
    },
  };
})();
