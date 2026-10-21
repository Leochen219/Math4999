# Bounded Rollout Lab

A dependency-free, static teaching demo for observed rollout-error geometry, a conditional lower bound, and a separate synthetic scalar controller. It does not run inference and has no backend.

## Run it locally

The page loads its reviewed JSON fixture with `fetch`, so browsers require it to be served over localhost rather than opened as a `file://` URL. From this directory, run:

```powershell
python -m http.server 8765 --bind 127.0.0.1
```

Then open <http://127.0.0.1:8765>. No packages or network access are needed.
On Windows, `Launch-Demo.cmd` starts the same local server and opens the page.

## Observed scalar evidence

`data/task11-scalars.json` contains only the public Task 11 scalar decomposition: 16 rows from one Bridge episode (record index 15), two seed schedules, endpoints h2–h5, and two separate metrics: RGB and condition-masked latent. Each row keeps its horizon and actual source seed. The source SHA-256, public source revision, record index, and source-analysis/run hashes are included in `provenance`.

To reproduce the fixture from the repository root, run:

```powershell
python bounded-rollout-lab/tools/export_task11_scalars.py
```

The exporter accepts `--source PATH` and `--output PATH` when used from another checkout. It verifies the source SHA-256 and exports only the listed scalars and limited scope metadata; it does not copy images, vectors, raw arrays, or machine paths. Precision metadata was not available and is marked `unknown`.

## Math shown in the page

For normalized RMS magnitudes, `A = sqrt(AR MSE)`, `B = sqrt(TF MSE)`, and `P = sqrt(feedback-change MSE)`. The UI displays the norm triangle `|B − P| ≤ A ≤ B + P` and the recorded identity `AR² − TF² = 2 dot + P²`, where `dot = (b·p)/n`. The triangle is a standard norm bound; the identity describes this observed decomposition and does not validate how teacher-forced and autoregressive samples were paired.

The selectable `ρ` bound is `sqrt(max(0, B² + P² − 2ρBP))`. It is displayed as applicable only when the selected observed aggregate meets `dot ≥ −ρBP`, with an explicit `1e-12` scale-aware comparison tolerance. Zero-norm cosine values are `N/A`. A failed premise remains visible and the bound is withheld.

The final panel is a separate synthetic recurrence, `e[t+1] = (a−k)e[t] + w[t]`, with `|w[t]| ≤ ε` and `E[t+1] = |a−k|E[t] + ε`. Its deterministic disturbance sequence is configurable. The conditional steady bound `ε/(1−q)` is shown only for `q = |a−k| < 1`; marginal and unstable cases have no steady bound.

## Tests

Run the pure math and fixture tests with Node.js 20 or later:

```powershell
node --test bounded-rollout-lab/tests/math.test.mjs
```

The test suite uses Node's built-in test runner and has no installed dependencies.

## Scope boundaries

- One episode and two seed schedules are limited cases, not independent episodes or population evidence.
- Future observed poses enter the motion conditions. This does not establish control-only open-loop accuracy.
- The data covers observed endpoints only; the horizon lines are visual guides and support no fitted growth law.
- RGB and condition-masked latent are separate metrics.
- There is no global Lipschitz or stability certification. Shared teacher forcing and identity closure do not validate pairing or the proposed correction mechanism.
- The standard norm triangle and conditional correlation bound are textbook inequalities, not original results.
- The synthetic controller is not applied to Cosmos or a physical system and makes no safety claim.

## Files

- `index.html`, `styles.css`, `app.mjs`: responsive interactive page and SVG plots.
- `Launch-Demo.cmd`: Windows helper for the documented local server command.
- `math.mjs`: pure scalar math and export helpers.
- `data/task11-scalars.json`: reviewed public scalar fixture with provenance.
- `tools/export_task11_scalars.py`: standard-library fixture exporter.
- `tests/math.test.mjs`: known-answer, premise, fixture, envelope, unstable-regime, and JSON export tests.
