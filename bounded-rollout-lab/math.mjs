export const NUMERIC_TOLERANCE = 1e-12;
export const MAX_SIMULATION_STEPS = 40;

function requireFinite(name, value) {
  if (!Number.isFinite(value)) throw new TypeError(`${name} must be finite`);
}

export function decomposeErrors({ tfSquaredError, arSquaredError, feedbackChangeSquared, crossTerm2DotOverN }) {
  for (const [name, value] of Object.entries({ tfSquaredError, arSquaredError, feedbackChangeSquared, crossTerm2DotOverN })) {
    requireFinite(name, value);
  }
  if (tfSquaredError < 0 || arSquaredError < 0 || feedbackChangeSquared < 0) {
    throw new RangeError('squared errors must be non-negative');
  }

  const A = Math.sqrt(arSquaredError);
  const B = Math.sqrt(tfSquaredError);
  const P = Math.sqrt(feedbackChangeSquared);
  const lower = Math.abs(B - P);
  const upper = B + P;
  const scale = Math.max(1, A, B, P);
  const triangle = {
    lower,
    upper,
    valid: A >= lower - NUMERIC_TOLERANCE * scale && A <= upper + NUMERIC_TOLERANCE * scale,
  };
  const dot = crossTerm2DotOverN / 2;
  const identityResidual = arSquaredError - tfSquaredError - crossTerm2DotOverN - feedbackChangeSquared;
  const cosine = B * P === 0 ? null : dot / (B * P);
  return { A, B, P, dot, cosine, triangle, identityResidual };
}

export function conditionalLowerBound({ B, P, rho, dot }) {
  for (const [name, value] of Object.entries({ B, P, rho })) requireFinite(name, value);
  if (B < 0 || P < 0 || rho < 0 || rho > 1) throw new RangeError('B and P must be non-negative; rho must be between 0 and 1');
  const product = B * P;
  if (dot === undefined && product !== 0) throw new TypeError('dot is required when B and P are nonzero');
  if (dot !== undefined) requireFinite('dot', dot);
  const observedDot = dot ?? 0;
  const rhs = -rho * product;
  const conditionSlack = observedDot - rhs;
  const tolerance = NUMERIC_TOLERANCE * Math.max(1, Math.abs(observedDot), Math.abs(rhs));
  const conditionSatisfied = conditionSlack >= -tolerance;
  const radicand = B * B + P * P + 2 * rhs;
  return {
    conditionSatisfied,
    conditionSlack,
    tolerance,
    requiredRho: product === 0 ? null : Math.max(0, -observedDot / product),
    value: conditionSatisfied ? Math.sqrt(Math.max(0, radicand)) : null,
  };
}

export function stableSteadyBound(q, epsilon) {
  requireFinite('q', q);
  requireFinite('epsilon', epsilon);
  if (q < 0 || epsilon < 0) throw new RangeError('q and epsilon must be non-negative');
  return q < 1 ? epsilon / (1 - q) : null;
}

function disturbanceAt(mode, step, epsilon) {
  if (epsilon === 0 || mode === 'zero') return 0;
  if (mode === 'alternating') return (step % 2 === 0 ? 1 : -1) * epsilon;
  if (mode === 'positive') return epsilon;
  if (mode === 'negative') return -epsilon;
  throw new RangeError(`unknown disturbance mode: ${mode}`);
}

export function simulateScalarControl({ a, k, epsilon, e0, E0, steps, disturbance = 'alternating' }) {
  for (const [name, value] of Object.entries({ a, k, epsilon, e0, E0, steps })) requireFinite(name, value);
  if (epsilon < 0 || E0 < 0 || !Number.isInteger(steps) || steps < 0 || steps > MAX_SIMULATION_STEPS) {
    throw new RangeError(`epsilon and E0 must be non-negative; steps must be an integer from 0 to ${MAX_SIMULATION_STEPS}`);
  }
  if (E0 + NUMERIC_TOLERANCE < Math.abs(e0)) throw new RangeError('E0 must be at least |e0|');
  const c = a - k;
  const q = Math.abs(c);
  const points = [{ step: 0, e: e0, E: E0, w: null }];
  for (let t = 0; t < steps; t += 1) {
    const w = disturbanceAt(disturbance, t, epsilon);
    const previous = points[t];
    points.push({ step: t + 1, e: c * previous.e + w, E: q * previous.E + epsilon, w });
  }
  const regime = q < 1 ? 'stable' : q === 1 ? 'marginal' : 'unstable';
  return {
    label: 'SYNTHETIC scalar control demonstration',
    equation: 'e[t+1] = (a-k)e[t] + w[t], |w[t]| <= epsilon',
    envelopeEquation: 'E[t+1] = |a-k|E[t] + epsilon',
    a, k, epsilon, e0, E0, steps, disturbance, c, q, regime,
    steadyBound: stableSteadyBound(q, epsilon),
    points,
  };
}

export function buildResultsExport({ dataset, selectedRow, rho, rhoResult, synthetic }) {
  return {
    schema: 'bounded-rollout-lab-results-v1',
    labels: {
      observed: 'Task 11 observed scalar evidence; one episode and two seed schedules',
      synthetic: 'Synthetic scalar control illustration; not applied to Cosmos or a physical system',
    },
    provenance: dataset.provenance,
    observed: {
      row: selectedRow,
      rho,
      conditionalBound: rhoResult,
      decomposition: decomposeErrors({
        tfSquaredError: selectedRow.tfSquaredError,
        arSquaredError: selectedRow.arSquaredError,
        feedbackChangeSquared: selectedRow.feedbackChangeSquared,
        crossTerm2DotOverN: selectedRow.crossTerm2DotOverN,
      }),
    },
    synthetic,
  };
}
