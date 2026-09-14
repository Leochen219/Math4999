# Reproducibility Notes

## Required external components

This package is not standalone model software. A full run needs:

1. A separately installed, compatible official Cosmos 3 Edge framework.
2. An authorised model checkpoint and VAE at explicit filesystem paths.
3. A compatible CUDA/PyTorch environment. The verified experiment used CUDA
   13.0 and Torch 2.10.0+cu130.
4. A GPU with enough memory for a batch-one Cosmos inference workload.
5. An explicit empty output directory with sufficient disk space for raw
   float tensors.

No credentials, hostnames, model downloads, checkpoints or absolute server
paths are stored here.

For development tests, install requirements-dev.txt. Matplotlib is required
only when regenerating the checked-in CPU calibration plots.

## Inference controls that must be fixed

- scenario input, prompt, action chunk and action representation;
- model/VAE/checkpoint hashes;
- model seed and paired prediction-region initial noise;
- sampler, 30 denoising steps, guidance, shift and FPS;
- GPU index and precision settings;
- condition/prediction mask, direction-bank hash and amplitude list;
- cache state (the precision and Task 5 comparisons use cache-off).

For the Task 5 reference run, autocast and TF32 were disabled on the FP32
condition path. Decoder replays record the consumed dtype and clear the decoder
cache between calls.

## Run lifecycle

1. Run the CPU unit tests first.
2. Perform the bridge smoke checks before any large scan.
3. Write each sample atomically with status, provenance, tensors and hashes.
4. Resume only when the code, model, VAE, directions, base latent, mask and
   noise-strategy identities match exactly.
5. Analyse from saved float tensors, not encoded MP4 or PNG frames.
6. Build and verify a SHA256 manifest before archive or review.

## Output handling

Raw float tensors and visual videos are intentionally excluded from Git. Keep
them in a dated experiment archive with its manifest and a smaller review
bundle containing code, configuration, metrics, reports and selected evidence.
