"""Independent Task11 raw-array audit. No production experiment imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_sample(root, name):
    folder = root / 'samples' / name
    status = json.loads((folder / 'status.json').read_text())
    assert status['status'] == 'success'
    for rel, digest in status['artifact_sha256'].items():
        assert Path(rel).name == rel
        assert sha(folder / rel) == digest, str(folder / rel)
    record = json.loads((folder / 'record.json').read_text())
    def array(key):
        descriptor = record
        for part in key.split('.'):
            descriptor = descriptor[part]
        path = folder / descriptor['artifact']
        assert path.name == descriptor['artifact']
        value = np.load(path, allow_pickle=False)
        assert str(value.dtype) == descriptor['dtype']
        assert list(value.shape) == descriptor['shape']
        assert np.isfinite(value).all()
        return value
    return record, array


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x*x)))


def metrics(pred, truth):
    assert pred.dtype == truth.dtype == np.float32
    d = pred.astype(np.float64) - truth.astype(np.float64)
    return {'rmse': rms(d), 'mae': float(np.mean(np.abs(d))),
            'psnr': None if not np.any(d) else float(-20*np.log10(rms(d)))}


def audit(run_dir, input_npz):
    root = Path(run_dir)
    status = json.loads((root / 'run_status.json').read_text())
    assert status['status'] == 'COMPLETE'
    with np.load(input_npz, allow_pickle=False) as data:
        rgb = data['rgb_float32']
        actions = data['actions_normalized']
    with np.load(root / 'truth_conditions.npz', allow_pickle=False) as data:
        truth = {i: data[f'condition_x{i}'] for i in (0,16,32,48,64,80)}
        mask = data['full_condition_mask']
    # Derive the condition-only selection from stored geometry, not a hardcoded time index.
    differing = [i for i, (a,b) in enumerate(zip(mask.shape, truth[0].shape)) if a != b]
    assert len(differing) == 1
    axis = differing[0]
    indexes = np.flatnonzero(np.any(mask, axis=tuple(i for i in range(mask.ndim) if i != axis)))
    score_mask = np.take(mask, indexes, axis=axis)
    assert score_mask.shape == truth[0].shape
    rows, geometries, frames, adjacent = [], [], [], []
    verified = 0
    for schedule in (0,1):
        offset = 5*schedule
        previous_rgb = previous_latent = None
        g0_arrays = None
        first_noise = None
        for h in range(1,6):
            paired = {}
            for call in (('G0','G0_repeat') if h == 1 else ('TF','AR')):
                seed = offset+h-1
                name = f'schedule{schedule}_{call}_h{h}_seed{seed}'
                record, read = load_sample(root, name)
                assert record['seed'] == seed and record['horizon'] == h
                assert record['schedule_index'] == schedule and record['call'] == call
                np.testing.assert_array_equal(read('action'), actions[h-1])
                input_latent = read('condition_input_fp32')
                wanted = previous_latent if call == 'AR' else truth[(h-1)*16]
                np.testing.assert_array_equal(input_latent, wanted)
                wanted_rgb = previous_rgb if call == 'AR' else rgb[(h-1)*16]
                np.testing.assert_array_equal(read('condition_rgb'), wanted_rgb)
                steps = read('condition_steps_fp32')
                for step in steps:
                    np.testing.assert_array_equal(step[mask], input_latent[score_mask])
                assert len(steps) == 30
                generated = read('generated_rgb')
                full = read('decoded_rgb_full')
                latent = read('encoded_condition')
                np.testing.assert_array_equal(full[:,1:], generated)
                np.testing.assert_array_equal(generated[:,-1], read('decoded_last_rgb'))
                assert generated.shape == (3,16,256,256)
                assert float(generated.min()) >= 0 and float(generated.max()) <= 1
                assert record['generation']['sampler_generator_seeds'] == [seed]*30
                assert record['generation']['noise_evidence']['seed'] == seed
                assert record['generation']['noise_evidence']['prepare_seed'] == seed
                np.testing.assert_array_equal(read('generation.mask'), mask)
                consumed_noise = read('generation.consumed_initial_state')[~mask].copy()
                assert record['generation']['cache']['installed'] is False
                assert record['generation']['cache']['requested'] is False
                execution = record['generation']['execution']
                assert not execution['autocast'] and not execution['tf32_cudnn'] and not execution['tf32_matmul']
                assert set(execution['operation_dtypes']) == {'float32'}
                for precision in ('G','D','E'):
                    assert record['precision'][precision] == 'float32'
                for evidence in (record['precision']['encoder_input'],record['encoder'],record['decoder']):
                    assert evidence['inner_input_dtype'] == evidence['inner_output_dtype'] == 'float32'
                    assert evidence['autocast_disabled'] and evidence['tf32_disabled']
                    assert set(evidence['operation_dtypes']) == {'float32'}
                verified += 1
                if call == 'G0':
                    g0_arrays = (read('output_full'), generated.copy(), latent.copy(), input_latent.copy(), consumed_noise.copy())
                    first_noise = record['prediction_noise_hash']
                elif call == 'G0_repeat':
                    for a,b in zip(g0_arrays,(read('output_full'),generated,latent,input_latent,consumed_noise)):
                        np.testing.assert_array_equal(a,b)
                    assert first_noise == record['prediction_noise_hash']
                    g0_arrays = None
                    continue
                paired[call] = (generated[:,-1].copy(), latent.copy(), record, consumed_noise)
                endpoint = metrics(generated[:,-1], rgb[h*16])
                latent_rms = rms(latent[score_mask].astype(np.float64)-truth[h*16][score_mask].astype(np.float64))
                lx, ly = latent[score_mask].astype(np.float64), truth[h*16][score_mask].astype(np.float64)
                denominator = np.linalg.norm(lx) * np.linalg.norm(ly)
                latent_cosine = None if denominator == 0 else float(np.dot(lx,ly)/denominator)
                rows.append({'schedule_index':schedule,'horizon':h,'mode':call,'seed':seed,
                             **endpoint,'condition_latent_rms':latent_rms,'condition_latent_cosine':latent_cosine})
                for t in range(16):
                    frames.append({'sample_id':name,'source_frame_index':(h-1)*16+t+1,
                                   **metrics(generated[:,t],rgb[(h-1)*16+t+1])})
                if call in ('G0','AR'):
                    if h > 1:
                        old = previous_rgb.astype(np.float64)
                        new = generated[:,-1].astype(np.float64)
                        old_error = old - rgb[(h-1)*16].astype(np.float64)
                        new_error = new - rgb[h*16].astype(np.float64)
                        adjacent.append({'schedule_index':schedule,'horizon':h,
                            'endpoint_prediction_change_rms':rms(new-old),
                            'endpoint_error_change_rms':rms(new_error-old_error),
                            'previous_endpoint_rmse':rms(old_error),
                            'current_endpoint_rmse':rms(new_error),
                            'endpoint_rmse_delta':rms(new_error)-rms(old_error)})
                    previous_rgb, previous_latent = generated[:,-1].copy(), latent.copy()
            if h > 1:
                tf, ar = paired['TF'], paired['AR']
                assert tf[2]['prediction_noise_hash'] == ar[2]['prediction_noise_hash']
                assert tf[2]['packed_action_token_hashes'] == ar[2]['packed_action_token_hashes']
                np.testing.assert_array_equal(tf[3], ar[3])
                for space, tv, av, target in [('rgb',tf[0],ar[0],rgb[h*16]),
                        ('latent',tf[1][score_mask],ar[1][score_mask],truth[h*16][score_mask])]:
                    b = tv.astype(np.float64)-target.astype(np.float64)
                    p = av.astype(np.float64)-tv.astype(np.float64)
                    ea = av.astype(np.float64)-target.astype(np.float64)
                    lhs = float(np.mean(ea*ea)-np.mean(b*b))
                    cross, pure = float(2*np.mean(b*p)),float(np.mean(p*p))
                    geometries.append({'schedule_index':schedule,'horizon':h,'space':space,
                        'mse_delta':lhs,'cross':cross,'pure':pure,'identity_residual':lhs-cross-pure,
                        'delta_rmse':rms(ea)-rms(b)})
    assert verified == 20
    return {'status':'PASS','verified_samples':verified,'endpoints':rows,
            'frame_metrics':frames,'geometry':geometries,'adjacent':adjacent,
            'condition_axis':axis,'condition_indexes':indexes.tolist()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--formal-dir',required=True)
    parser.add_argument('--input-npz',required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.formal_dir,args.input_npz),indent=2))
