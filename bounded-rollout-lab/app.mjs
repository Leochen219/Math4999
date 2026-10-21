import { decomposeErrors, conditionalLowerBound, simulateScalarControl, buildResultsExport } from './math.mjs';

const $ = (selector) => document.querySelector(selector);
const datasetResponse = await fetch('./data/task11-scalars.json');
if (!datasetResponse.ok) throw new Error(`Could not load scalar fixture (${datasetResponse.status}). Start the local server from the demo folder.`);
const dataset = await datasetResponse.json();
const state = { metric: 'rgb', schedule: 0, horizon: 2, rho: 0.5, selectedRow: null, rhoResult: null, synthetic: null };
const observedRows = () => dataset.rows.filter((row) => row.metric === state.metric && row.scheduleIndex === state.schedule);
const findRow = () => observedRows().find((row) => row.horizonChunks === state.horizon);
const number = (value, digits = 4) => value === null || value === undefined || !Number.isFinite(value)
  ? 'N/A'
  : new Intl.NumberFormat('en', { maximumSignificantDigits: digits, minimumSignificantDigits: 1 }).format(value);
const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

function renderGeometry(row, decomposition) {
  const svg = $('#geometry-svg');
  const { A, B, P, cosine } = decomposition;
  const angleCos = cosine === null ? 0 : Math.max(-1, Math.min(1, cosine));
  const angleSin = Math.sqrt(Math.max(0, 1 - angleCos * angleCos));
  const bx = B;
  const by = 0;
  const px = B + P * angleCos;
  const py = -P * angleSin;
  const xMin = Math.min(0, bx, px);
  const xMax = Math.max(0, bx, px);
  const yMin = Math.min(0, py);
  const yMax = Math.max(0, py);
  const rangeX = Math.max(xMax - xMin, 0.001);
  const rangeY = Math.max(yMax - yMin, 0.001);
  const scale = Math.min(410 / rangeX, 190 / rangeY);
  const xOffset = 290 - ((xMin + xMax) * scale) / 2;
  const yOffset = 168 + ((yMin + yMax) * scale) / 2;
  const point = (x, y) => [xOffset + x * scale, yOffset - y * scale];
  const [ox, oy] = point(0, 0);
  const [bxp, byp] = point(bx, by);
  const [pxp, pyp] = point(px, py);
  const label = (x, y, text, fill) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="${fill}" class="vector-label">${esc(text)}</text>`;
  svg.innerHTML = `
    <defs>
      <marker id="arrow-tf" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#e6b875" stroke-width="1.5"/></marker>
      <marker id="arrow-feedback" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#84b8e8" stroke-width="1.5"/></marker>
      <marker id="arrow-ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#7ee0c0" stroke-width="1.5"/></marker>
    </defs>
    <path d="M40 168H540" stroke="#253238" stroke-width="1" stroke-dasharray="2 7"/>
    <path d="M${ox.toFixed(1)} ${oy.toFixed(1)}L${bxp.toFixed(1)} ${byp.toFixed(1)}" fill="none" stroke="#e6b875" stroke-width="2.4" marker-end="url(#arrow-tf)"/>
    <path d="M${bxp.toFixed(1)} ${byp.toFixed(1)}L${pxp.toFixed(1)} ${pyp.toFixed(1)}" fill="none" stroke="#84b8e8" stroke-width="2.4" marker-end="url(#arrow-feedback)"/>
    <path d="M${ox.toFixed(1)} ${oy.toFixed(1)}L${pxp.toFixed(1)} ${pyp.toFixed(1)}" fill="none" stroke="#7ee0c0" stroke-width="1.4" stroke-dasharray="4 5" marker-end="url(#arrow-ar)"/>
    <circle cx="${ox.toFixed(1)}" cy="${oy.toFixed(1)}" r="4" fill="#0c1116" stroke="#dce7e2" stroke-width="1.5"/>
    <circle cx="${pxp.toFixed(1)}" cy="${pyp.toFixed(1)}" r="4" fill="#7ee0c0"/>
    ${label((ox + bxp) / 2, oy + 20, `B = ${number(B, 3)}`, '#e6b875')}
    ${label((bxp + pxp) / 2 + 7, (byp + pyp) / 2 - 7, `P = ${number(P, 3)}`, '#84b8e8')}
    ${label((ox + pxp) / 2 - 12, (oy + pyp) / 2 - 10, `A = ${number(A, 3)}`, '#7ee0c0')}
    <text x="${Math.max(50, Math.min(465, ox + 8)).toFixed(1)}" y="${Math.max(24, oy - 11).toFixed(1)}" fill="#7f8d89" class="vector-small">origin</text>
    <text x="438" y="26" fill="#788782" class="vector-small">cos θ = ${cosine === null ? 'N/A' : number(cosine, 3)}</text>
  `;
  svg.setAttribute('aria-label', `Observed normalized RMS error vectors. A is ${number(A)}, B is ${number(B)}, P is ${number(P)}. TF plus feedback change equals AR error.`);
}

function renderHorizon() {
  const svg = $('#horizon-svg');
  const rows = observedRows().sort((a, b) => a.horizonChunks - b.horizonChunks);
  const series = [
    { key: 'tfSquaredError', label: 'TF error B', color: '#e6b875', val: (r) => Math.sqrt(r.tfSquaredError) },
    { key: 'feedbackChangeSquared', label: 'Feedback change P', color: '#84b8e8', val: (r) => Math.sqrt(r.feedbackChangeSquared) },
    { key: 'arSquaredError', label: 'AR error A', color: '#7ee0c0', val: (r) => Math.sqrt(r.arSquaredError) },
  ];
  const values = rows.flatMap((row) => series.map((item) => item.val(row)));
  const maxValue = Math.max(...values, 0.001) * 1.15;
  const left = 62; const right = 865; const top = 24; const bottom = 242;
  const xAt = (i) => left + i * ((right - left) / Math.max(1, rows.length - 1));
  const yAt = (value) => bottom - (value / maxValue) * (bottom - top);
  let markup = '';
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = maxValue * tick / 4; const y = yAt(value);
    markup += `<path d="M${left} ${y.toFixed(1)}H${right}" stroke="#27343a" stroke-width="1"/>`;
    markup += `<text x="${left - 12}" y="${(y + 3).toFixed(1)}" text-anchor="end">${esc(number(value, 2))}</text>`;
  }
  rows.forEach((row, index) => {
    const x = xAt(index);
    markup += `<path d="M${x} ${top}V${bottom}" stroke="#202c31" stroke-width="1" stroke-dasharray="2 6"/>`;
    markup += `<text x="${x}" y="${bottom + 22}" text-anchor="middle">h${row.horizonChunks} · seed ${row.seed}</text>`;
  });
  for (const item of series) {
    const coords = rows.map((row, index) => [xAt(index), yAt(item.val(row))]);
    const path = coords.map(([x, y], index) => `${index ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
    markup += `<path d="${path}" fill="none" stroke="${item.color}" stroke-width="2"/>`;
    coords.forEach(([x, y], index) => {
      markup += `<circle cx="${x}" cy="${y}" r="4" fill="#11191e" stroke="${item.color}" stroke-width="2"><title>${item.label} h${rows[index].horizonChunks}: ${number(item.val(rows[index]))}</title></circle>`;
    });
  }
  markup += `<text x="${left}" y="289">NORMALIZED RMS SCALE</text><text x="${right}" y="289" text-anchor="end">OBSERVED ENDPOINTS ONLY</text>`;
  svg.innerHTML = markup;
}

function renderObserved() {
  const row = findRow();
  if (!row) return;
  state.selectedRow = row;
  const decomposition = decomposeErrors(row);
  const metricLabel = state.metric === 'rgb' ? 'RGB' : 'Masked latent';
  $('#case-name').textContent = `${metricLabel} · schedule ${state.schedule} · h${row.horizonChunks}`;
  $('#case-seed').textContent = `Observed seed ${row.seed} · one episode`;
  $('#decomp-id').textContent = `h${row.horizonChunks} / s${state.schedule}`;
  $('#metric-a').textContent = number(decomposition.A);
  $('#metric-b').textContent = number(decomposition.B);
  $('#metric-p').textContent = number(decomposition.P);
  $('#cross-value').textContent = `cross term ${number(row.crossTerm2DotOverN)}`;
  $('#p2-value').textContent = `P² ${number(row.feedbackChangeSquared)}`;
  $('#tf-squared').textContent = number(row.tfSquaredError);
  $('#ar-squared').textContent = number(row.arSquaredError);
  $('#identity-value').textContent = number(row.identityResidual);

  const triangleStatus = $('#triangle-status');
  triangleStatus.classList.toggle('invalid', !decomposition.triangle.valid);
  triangleStatus.querySelector('span:last-child').textContent = decomposition.triangle.valid
    ? `Triangle holds · ${number(decomposition.triangle.lower, 3)} ≤ A ≤ ${number(decomposition.triangle.upper, 3)}`
    : `Recorded values outside triangle interval · ${number(decomposition.triangle.lower, 3)} ≤ A ≤ ${number(decomposition.triangle.upper, 3)}`;
  renderGeometry(row, decomposition);
  renderHorizon();
  renderConditional();
}

function renderConditional() {
  const row = state.selectedRow;
  const B = Math.sqrt(row.tfSquaredError);
  const P = Math.sqrt(row.feedbackChangeSquared);
  const dot = row.crossTerm2DotOverN / 2;
  const result = conditionalLowerBound({ B, P, rho: state.rho, dot });
  state.rhoResult = result;
  $('#rho-readout').textContent = Number(state.rho).toFixed(2);
  $('#dot-readout').textContent = number(dot, 5);
  $('#rho-case').textContent = `h${row.horizonChunks} / schedule ${state.schedule}`;
  const status = $('#premise-status');
  status.classList.toggle('failed', !result.conditionSatisfied);
  status.querySelector('strong').textContent = result.conditionSatisfied
    ? 'Premise satisfied for this observed aggregate'
    : 'Premise violated for this observed aggregate';
  status.querySelector('small').textContent = result.conditionSatisfied
    ? `dot + ρBP = ${number(result.conditionSlack, 5)} · tolerance ${number(result.tolerance, 2)}`
    : `dot + ρBP = ${number(result.conditionSlack, 5)} · tolerance ${number(result.tolerance, 2)}`;
  $('#bound-value').textContent = result.value === null ? 'Not applicable' : number(result.value, 5);
  $('#bound-subnote').textContent = result.value === null ? 'The displayed premise is not met' : 'Conditional lower bound for this scalar case';
  $('#slack-value').textContent = number(result.conditionSlack, 5);
  $('#required-rho').textContent = result.requiredRho === null ? 'N/A (zero BP)' : number(result.requiredRho, 5);
}

function ensureEnvelopeFloor() {
  const e0Input = $('#control-e0');
  const E0Input = $('#control-E0');
  const e0 = Number(e0Input.value);
  const envelope = Number(E0Input.value);
  if (!e0Input.value.trim() || !E0Input.value.trim() || !Number.isFinite(e0) || !Number.isFinite(envelope)) {
    $('#envelope-hint').textContent = 'Enter finite values for e₀ and E₀; the envelope must start above the initial error.';
    return;
  }
  const magnitude = Math.abs(e0);
  const adjusted = envelope < magnitude;
  if (adjusted) E0Input.value = magnitude.toFixed(2);
  $('#envelope-hint').textContent = adjusted
    ? `E₀ was adjusted to |e₀| = ${number(magnitude, 3)} so the initial state starts inside the envelope.`
    : `E₀ = ${number(envelope, 3)} ≥ |e₀| = ${number(magnitude, 3)}; the initial state is inside the envelope.`;
}

function renderControlPlot(result) {
  const svg = $('#control-svg');
  const width = 800; const left = 55; const right = 780; const top = 21; const bottom = 252;
  const maxAbs = Math.max(...result.points.map((point) => point.E), ...result.points.map((point) => Math.abs(point.e)), 0.05) * 1.14;
  const xAt = (step) => left + (step / Math.max(1, result.steps)) * (right - left);
  const yAt = (value) => top + ((maxAbs - value) / (2 * maxAbs)) * (bottom - top);
  const upper = result.points.map((point) => [xAt(point.step), yAt(point.E)]);
  const lower = [...result.points].reverse().map((point) => [xAt(point.step), yAt(-point.E)]);
  const band = `${upper.map(([x, y], index) => `${index ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ')} ${lower.map(([x, y]) => `L${x.toFixed(1)} ${y.toFixed(1)}`).join(' ')} Z`;
  const trajectory = result.points.map((point, index) => `${index ? 'L' : 'M'}${xAt(point.step).toFixed(1)} ${yAt(point.e).toFixed(1)}`).join(' ');
  let grid = '';
  for (let i = 0; i <= 4; i += 1) {
    const value = maxAbs - (2 * maxAbs * i) / 4; const y = yAt(value);
    grid += `<path d="M${left} ${y.toFixed(1)}H${right}" stroke="${Math.abs(value) < maxAbs / 50 ? '#46534f' : '#253238'}" stroke-width="1"/>`;
    grid += `<text x="${left - 10}" y="${(y + 3).toFixed(1)}" text-anchor="end">${esc(number(value, 2))}</text>`;
  }
  const xTicks = [...new Set([0, Math.round(result.steps / 2), result.steps])];
  xTicks.forEach((step) => { const x = xAt(step); grid += `<text x="${x.toFixed(1)}" y="${bottom + 20}" text-anchor="middle">${step}</text>`; });
  const dots = result.points.map((point) => `<circle cx="${xAt(point.step).toFixed(1)}" cy="${yAt(point.e).toFixed(1)}" r="2.7" fill="#7ee0c0"><title>t=${point.step}, e=${number(point.e)}, E=${number(point.E)}</title></circle>`).join('');
  svg.innerHTML = `${grid}<path d="${band}" fill="#84b8e8" fill-opacity=".09" stroke="none"/><path d="${upper.map(([x,y],i)=>`${i?'L':'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ')}" fill="none" stroke="#84b8e8" stroke-opacity=".75" stroke-width="1.2" stroke-dasharray="4 5"/><path d="${result.points.map((point,i)=>`${i?'L':'M'}${xAt(point.step).toFixed(1)} ${yAt(-point.E).toFixed(1)}`).join(' ')}" fill="none" stroke="#84b8e8" stroke-opacity=".75" stroke-width="1.2" stroke-dasharray="4 5"/><path d="${trajectory}" fill="none" stroke="#7ee0c0" stroke-width="2.2"/>${dots}<text x="${left}" y="292">STEP t</text><text x="${right}" y="292" text-anchor="end">w[t] DETERMINISTIC · |w[t]| ≤ ε</text>`;
  svg.setAttribute('aria-label', `Synthetic scalar trajectory for ${result.steps} steps. q is ${number(result.q)} and regime is ${result.regime}. The signed trajectory is shown with plus and minus envelope.`);
}

function currentSyntheticConfig() {
  const read = (id) => {
    const input = $(id);
    const label = input.previousElementSibling?.textContent ?? input.id;
    if (!input.value.trim()) throw new TypeError(`Enter a value for ${label}`);
    const value = Number(input.value);
    if (!Number.isFinite(value)) throw new TypeError(`${label} must be finite`);
    if (input.min !== '' && value < Number(input.min)) throw new RangeError(`${label} must be at least ${input.min}`);
    if (input.max !== '' && value > Number(input.max)) throw new RangeError(`${label} must be at most ${input.max}`);
    return value;
  };
  return {
    a: read('#control-a'), k: read('#control-k'), epsilon: read('#control-epsilon'),
    e0: read('#control-e0'), E0: read('#control-E0'), steps: read('#control-steps'),
    disturbance: $('#control-disturbance').value,
  };
}

function renderSynthetic() {
  ensureEnvelopeFloor();
  try {
    const config = currentSyntheticConfig();
    const result = simulateScalarControl(config);
    state.synthetic = result;
    $('#q-value').textContent = number(result.q, 4);
    $('#steady-value').textContent = result.steadyBound === null ? 'Unavailable' : number(result.steadyBound, 4);
    const pill = $('#regime-pill');
    pill.textContent = result.regime.toUpperCase();
    pill.classList.toggle('warning', result.regime === 'unstable');
    pill.classList.toggle('marginal', result.regime === 'marginal');
    $('#regime-note').textContent = result.regime === 'stable'
      ? `Since q < 1, the conditional steady envelope is ε/(1−q) = ${number(result.steadyBound, 4)}.`
      : result.regime === 'marginal'
        ? 'Here q = 1. The envelope is not contractive and no steady bound is reported.'
        : 'Here q > 1. The envelope can grow and no steady bound is reported.';
    renderControlPlot(result);
    $('#download-config').disabled = false;
    $('#download-results').disabled = false;
  } catch (error) {
    state.synthetic = null;
    $('#q-value').textContent = '—';
    $('#steady-value').textContent = 'Unavailable';
    $('#regime-note').textContent = error.message;
    $('#regime-pill').textContent = 'INVALID CONFIG';
    $('#regime-pill').className = 'regime-pill warning';
    $('#control-svg').innerHTML = `<text x="400" y="150" text-anchor="middle">${esc(error.message)}</text>`;
    $('#control-svg').setAttribute('aria-label', 'No synthetic trajectory: invalid configuration');
    $('#download-config').disabled = true;
    $('#download-results').disabled = true;
  }
}

function download(filename, object) {
  const blob = new Blob([JSON.stringify(object, null, 2) + '\n'], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadConfig() {
  download('bounded-rollout-lab-config.json', {
    schema: 'bounded-rollout-lab-config-v1',
    labels: { observed: 'Task 11 observed scalar case', synthetic: 'Synthetic scalar control illustration' },
    provenance: dataset.provenance,
    observedSelection: { metric: state.metric, scheduleIndex: state.schedule, horizonChunks: state.horizon, seed: state.selectedRow?.seed, rho: state.rho },
    syntheticConfiguration: currentSyntheticConfig(),
  });
}

function downloadResults() {
  if (!state.synthetic) return;
  const result = buildResultsExport({ dataset, selectedRow: state.selectedRow, rho: state.rho, rhoResult: state.rhoResult, synthetic: state.synthetic });
  download('bounded-rollout-lab-results.json', result);
}

$('#metric-select').addEventListener('change', (event) => { state.metric = event.target.value; renderObserved(); });
$('#schedule-select').addEventListener('change', (event) => { state.schedule = Number(event.target.value); renderObserved(); });
$('#horizon-select').addEventListener('change', (event) => { state.horizon = Number(event.target.value); renderObserved(); });
$('#rho-slider').addEventListener('input', (event) => { state.rho = Number(event.target.value); renderConditional(); });
for (const id of ['#control-a', '#control-k', '#control-epsilon', '#control-e0', '#control-E0', '#control-steps', '#control-disturbance']) {
  $(id).addEventListener('input', renderSynthetic);
  $(id).addEventListener('change', renderSynthetic);
}
$('#download-config').addEventListener('click', downloadConfig);
$('#download-results').addEventListener('click', downloadResults);
$('#case-count').textContent = `${dataset.rows.length} PUBLIC SCALARS`;
$('#source-link').href = dataset.provenance.source;
renderObserved();
renderSynthetic();
