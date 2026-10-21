import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { decomposeErrors, conditionalLowerBound, simulateScalarControl, buildResultsExport, stableSteadyBound } from '../math.mjs';

const fixture = JSON.parse(readFileSync(new URL('../data/task11-scalars.json', import.meta.url), 'utf8'));
const close = (actual, expected, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

test('decomposes a known feedback example and satisfies the triangle interval', () => {
  const result = decomposeErrors({ tfSquaredError: 1, arSquaredError: 0.25, feedbackChangeSquared: 0.25, crossTerm2DotOverN: -1 });
  assert.equal(result.A, 0.5); assert.equal(result.B, 1); assert.equal(result.P, 0.5);
  assert.deepEqual(result.triangle, { lower: 0.5, upper: 1.5, valid: true }); close(result.identityResidual, 0);
});

test('handles zero errors and reports cosine as unavailable', () => {
  const result = decomposeErrors({ tfSquaredError: 0, arSquaredError: 0, feedbackChangeSquared: 0, crossTerm2DotOverN: 0 });
  assert.equal(result.cosine, null); assert.equal(result.triangle.valid, true);
  const bound = conditionalLowerBound({ B: 0, P: 0, rho: 0 }); assert.equal(bound.value, 0); assert.equal(bound.conditionSatisfied, true);
});

test('conditional lower bound is unavailable when its premise fails', () => {
  const result = conditionalLowerBound({ B: 2, P: 1, rho: 0.25, dot: -1 });
  assert.equal(result.conditionSatisfied, false); assert.equal(result.value, null); assert.equal(result.requiredRho, 0.5);
});

test('conditional lower bound computes the stated value when its premise holds', () => {
  const result = conditionalLowerBound({ B: 2, P: 1, rho: 0.5, dot: -1 });
  assert.equal(result.conditionSatisfied, true); close(result.value, Math.sqrt(3));
});

test('conditional lower bound requires an observed dot for nonzero norms', () => {
  assert.throws(() => conditionalLowerBound({ B: 1, P: 1, rho: 0.5 }), /dot is required/);
});

test('public scalar fixture preserves provenance and audited row parity', () => {
  assert.equal(fixture.provenance.sourceSha256, '9290a21548f9de2a6131a12b3b521d56ef90ba536b91dd72653ec6e762350602');
  assert.equal(fixture.provenance.sourceRevision, '8f41d976ae699eae5d10392823d3e0f060196968');
  assert.equal(fixture.provenance.sourceRecordIndex, 15);
  assert.equal(fixture.scope.episodeCount, 1); assert.equal(fixture.scope.seedScheduleCount, 2);
  assert.deepEqual(Object.keys(fixture.rows.reduce((acc, row) => (acc[row.metric] = true, acc), {})).sort(), ['condition_masked_latent', 'rgb']);
  assert.equal(fixture.rows.length, 16);
  for (const metric of ['rgb', 'condition_masked_latent']) {
    const rows = fixture.rows.filter(row => row.metric === metric);
    assert.equal(rows.length, 8); assert.deepEqual([...new Set(rows.map(row => row.scheduleIndex))].sort(), [0, 1]);
    assert.deepEqual([...new Set(rows.map(row => row.horizonChunks))].sort(), [2, 3, 4, 5]);
    for (const row of rows) {
      assert.equal(row.seed, row.horizonChunks + 5 * row.scheduleIndex - 1);
      close(row.feedbackChangeSquared, row.feedbackChangeRmse ** 2, 1e-12);
      close(row.squaredErrorDifference, row.crossTerm2DotOverN + row.feedbackChangeSquared, 1e-12);
      close(row.identityResidual, row.squaredErrorDifference - row.crossTerm2DotOverN - row.feedbackChangeSquared, 1e-12);
      const derived = decomposeErrors(row);
      close(derived.identityResidual, row.identityResidual, 1e-12);
      assert.equal(derived.triangle.valid, true);
      if (derived.cosine !== null) close(derived.cosine, row.errorFeedbackCosine, 1e-12);
      if (metric === 'condition_masked_latent') assert.equal(row.maskedElementCount, 12288);
      else assert.equal('maskedElementCount' in row, false);
    }
  }
  assert.equal(fixture.scope.precisionMetadata, 'unknown');
});

test('synthetic stable trajectory stays inside its signed error envelope', () => {
  const result = simulateScalarControl({ a: 1.2, k: 0.5, epsilon: 0.1, e0: -0.4, E0: 0.4, steps: 12, disturbance: 'alternating' });
  assert.equal(result.q, 0.7); assert.equal(result.regime, 'stable');
  for (const point of result.points) { assert.ok(Math.abs(point.e) <= point.E + 1e-12); assert.ok(point.E >= 0); }
  close(result.points[1].e, -0.18); close(result.points[1].E, 0.38); close(result.steadyBound, 0.1 / 0.3);
});

test('unstable or marginal synthetic regimes have no steady bound', () => {
  assert.equal(stableSteadyBound(1, 0.1), null);
  const result = simulateScalarControl({ a: 2, k: 0.5, epsilon: 0, e0: 1, E0: 1, steps: 4, disturbance: 'alternating' });
  assert.equal(result.regime, 'unstable'); assert.equal(result.steadyBound, null);
  assert.deepEqual(result.points.map(point => point.e), [1, 1.5, 2.25, 3.375, 5.0625]);
});

test('synthetic simulation rejects a step count above the bounded UI limit', () => {
  assert.throws(() => simulateScalarControl({ a: 1, k: 0, epsilon: 0, e0: 0, E0: 0, steps: 41 }), /0 to 40/);
});

test('results export carries selected config, labels, and values as JSON', () => {
  const exportValue = buildResultsExport({
    dataset: fixture, selectedRow: fixture.rows[0], rhoResult: conditionalLowerBound({ B: 1, P: 0.5, rho: 0.5, dot: -0.1 }),
    synthetic: simulateScalarControl({ a: 0.8, k: 0.2, epsilon: 0, e0: 0.2, E0: 0.2, steps: 2, disturbance: 'alternating' }), rho: 0.5,
  });
  const serialized = JSON.stringify(exportValue); assert.deepEqual(JSON.parse(serialized), exportValue);
  assert.match(serialized, /synthetic/); assert.match(serialized, /provenance/);
});
