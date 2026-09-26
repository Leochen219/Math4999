"""Compare independently computed raw-tensor metrics to production analysis."""
import json
import math
import sys
from pathlib import Path

independent = json.loads(Path(sys.argv[1]).read_text())
production = json.loads(Path(sys.argv[2]).read_text())
assert independent['status'] == production['status'] == 'PASS'
assert independent['verified_samples'] == 20
count = 0
maximum = 0.0

def check(a,b):
    global count, maximum
    count += 1
    if a is None or b is None:
        assert a is b
        return
    assert math.isfinite(a) and math.isfinite(b)
    maximum = max(maximum, abs(a-b))
    assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-13), (a,b)

actual = {(r['schedule_index'],r['horizon_chunks'],r['mode']):r for r in production['endpoint_metrics_per_seed']}
assert len(actual)==18
for row in independent['endpoints']:
    match = actual[row['schedule_index'],row['horizon'],row['mode']]
    for key in ('rmse','mae','psnr','condition_latent_rms','condition_latent_cosine'):
        check(row[key],match[key])
actual = {(r['sample_id'],r['source_frame_index']):r for r in production['per_frame_metrics']}
assert len(actual)==288
for row in independent['frame_metrics']:
    for key in ('rmse','mae','psnr'):
        check(row[key], actual[row['sample_id'],row['source_frame_index']][key])
for space, collection in [('rgb','tf_ar_error_geometry_rgb'),('latent','tf_ar_error_geometry_condition_masked_latent')]:
    actual = {(r['schedule_index'],r['horizon_chunks']):r for r in production[collection]}
    assert len(actual)==8
    for row in independent['geometry']:
        if row['space']!=space: continue
        match = actual[row['schedule_index'],row['horizon']]
        for a,b in [('mse_delta','squared_error_difference'),('cross','cross_term_2dot_over_n'),('pure','feedback_change_squared'),('identity_residual','identity_residual')]:
            check(row[a],match[b])
        deltas = [r for r in production['tf_ar_endpoint_error_deltas'] if (r['schedule_index'],r['horizon_chunks'])==(row['schedule_index'],row['horizon'])]
        assert len(deltas)==1
        check(row['delta_rmse'],deltas[0]['rgb_rmse_ar_minus_tf' if space=='rgb' else 'condition_latent_rms_ar_minus_tf'])
actual={(r['schedule_index'],r['horizon_chunks']):r for r in production['adjacent_ar_endpoint_changes']}
assert len(actual)==8
for row in independent['adjacent']:
    match=actual[row['schedule_index'],row['horizon']]
    for key in ('endpoint_prediction_change_rms','endpoint_error_change_rms','previous_endpoint_rmse','current_endpoint_rmse','endpoint_rmse_delta'):
        check(row[key],match[key])
print(json.dumps({'status':'PASS','comparisons':count,'max_absolute_difference':maximum},indent=2))
