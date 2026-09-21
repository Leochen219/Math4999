"""Main-review-only read-only audit, independent of Task7 runtime code."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp/cosmos-experiments/2026-09-16/umi_task6_cross_context_linearity_run07')
DEC = Path(str(ROOT) + '_decoder_cleanup01')
IDS = ['baseline_pre'] + [f'v0_alpha_{i:02d}_{sign}' for i in range(3) for sign in ('plus', 'minus')] + ['baseline_post']

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def array(path):
    mapped = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        assert np.isfinite(mapped).all(), str(path)
        return np.array(mapped, copy=True)
    finally:
        mapped._mmap.close()

def rms(value):
    data = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(data * data)))

rows = []
frozen = None
baseline_encoded = None
direction = array(ROOT / 'samples/bridge_0__seed_0__v0_alpha_00_plus/direction.npy')
for name in IDS:
    sample = ROOT / 'samples' / ('bridge_0__seed_0__' + name)
    replay = DEC / ('bridge_0__seed_0__' + name + '__temporary_fp32')
    status = json.loads((sample / 'status.json').read_text())
    record = json.loads((replay / 'record.json').read_text())
    assert status['status'] == 'success', name
    for base, metadata in ((sample, status), (replay, record)):
        for filename, expected in metadata['artifact_sha256'].items():
            assert digest(base / filename) == expected, (name, filename)
    assert status['artifact_sha256']['output_full.npy'] == record['artifact_sha256']['decoder_input_full_latent.npy'], name
    z0, mask = array(sample / 'z_bar.npy'), array(sample / 'mask.npy').astype(bool)
    identity = (status['artifact_sha256']['z_bar.npy'], status['artifact_sha256']['mask.npy'])
    if frozen is None:
        frozen = identity
    assert identity == frozen, name
    x = array(sample / 'consumed_input_fp32.npy')
    delta = np.asarray(x - z0, dtype=np.float32)
    assert delta.shape == mask.shape == direction.shape, name
    assert np.all(delta[~mask] == 0), name
    image = array(replay / 'decoded_final_float32.npy')
    encoded = array(replay / 'direct_condition_latent_float32.npy')
    assert image.dtype == np.float32 and image.shape == (3, 256, 256), name
    assert image.min() >= 0 and image.max() <= 1, name
    assert encoded.dtype == np.float32 and encoded.size == mask.sum(), name
    if baseline_encoded is None:
        baseline_encoded = encoded.copy()
    response = rms(np.subtract(encoded, baseline_encoded, dtype=np.float32))
    rows.append({'sample': name, 'input_rms': rms(delta[mask]), 'historical_native_feedback_rms': response, 'frame_sha256': record['artifact_sha256']['decoded_final_float32.npy'], 'native_encoder_sha256': record['artifact_sha256']['direct_condition_latent_float32.npy'], 'condition_shape': list(encoded.shape)})
fits = {}
for sign in ('plus', 'minus'):
    selected = [row for row in rows if row['sample'].endswith(sign)]
    x = np.log([row['input_rms'] for row in selected])
    y = np.log([row['historical_native_feedback_rms'] for row in selected])
    slope, intercept = np.polyfit(x, y, 1)
    fits[sign] = {'slope': float(slope), 'r2': float(1 - np.sum((y - slope*x - intercept)**2) / np.sum((y-y.mean())**2))}
print(json.dumps({'status': 'PASS', 'samples': len(rows), 'source_roots': [str(ROOT), str(DEC)], 'z0_mask_sha256': frozen, 'v0_mask_rms': rms(direction[mask]), 'z0_mask_rms': rms(z0[mask]), 'historical_native_feedback_fits': fits, 'rows': rows}, indent=2))
