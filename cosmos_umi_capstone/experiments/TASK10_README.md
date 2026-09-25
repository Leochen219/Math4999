# Task 10 reproducibility notes

Task 10 runs the fixed Bridge record indices `1,2,3,6,7,14`, seed pairs
`(0,1),(2,3)`, and four calls per pair. It uses the reviewed Task 8 snapshot
under `task8_frozen/` so later repository changes do not silently alter the
generation pathway. Large float32 outputs and model weights are not in Git.

Use the CUDA 13.0 / Torch `2.10.0+cu130` environment from the experiment.
The CLI requires paths to the verified Bridge subset, fixed framework commit,
checkpoint, VAE, and official parity artifact. Run `--stage preflight` first,
then `--stage smoke --resume --release`. After checking the smoke resource gate,
run `--stage formal --record-index N --resume --release` once per index, in
preregistered order, **one OS process at a time**. A new process is required
between records because the CUDA context alone may exceed the ≤1 GiB preload
gate after releasing a model inside the same process. Do not relax the gate.

After `run_status.json` says `GENERATION_COMPLETE`, run
`analyze_umi_task10_real_scenes.py --run-root RUN_ROOT`, then
`independently_verify_umi_task10.py --run-root RUN_ROOT --output
RUN_ROOT/analysis/independent_review.json`, and finally
`finalize_umi_task10_bundle.py --run-root RUN_ROOT`.

The 2026-09-25 analysis is in `../reports/task10_real_scenes_2026-09-25.md`.
Quantitative comparisons use float32 saved frames/latents, not MP4 or PNG.
Actions contain future observed poses, so the result concerns feedback under
real-motion conditioning, not control-only open-loop prediction.
