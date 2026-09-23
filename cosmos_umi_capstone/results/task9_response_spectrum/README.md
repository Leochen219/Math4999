# Task 9 — Bridge response-spectrum pilot (2026-09-23)

**Outcome:** All six pre-registered scene/seed strata completed (972 formal model calls). The tested finite-amplitude responses do not support a useful low-rank approximation that transfers from 32 training perturbation directions to eight independent held-out directions. This is a local, finite-amplitude finding—not a proof that the full Jacobian is high rank or that no low-rank regime exists elsewhere.

The three Bridge episodes were selected in original eligible order (TFRecord record indices 1, 2, 3); each was run with model seeds 0 and 1. Within each stratum, the initial state, scene-specific official action chunk, prompt, and diffusion noise were fixed. The FP32 post-VAE condition was perturbed along 32 masked training directions and eight independently sampled held-out directions, with paired ± responses at relative amplitude `alpha=0.003` and a pre-registered half-step check at `alpha=0.0015`. All scans used the verified Task 8 FP32 generation/decoding/re-encoding path. The action adapter derives motion from future observed poses, so this is not a control-command-only open-loop test.

## Main numbers

Each SVD was computed *within* one scene, seed, and output space. `k95` is the number of sampled training response columns needed for 95% of their squared-singular-value energy. The held-out number is the mean relative residual after projecting each unseen response onto the full 32-column training response span; **lower is better**.

| Bridge record | Seed | Predicted-latent k95 / 32 | Entropy effective rank | Full-span held-out residual | Half-step pass |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 25 | 6.47 | 0.631 | 35 / 40 |
| 1 | 1 | 28 | 12.65 | 0.801 | 40 / 40 |
| 2 | 0 | 29 | 9.69 | 0.811 | 40 / 40 |
| 2 | 1 | 31 | 30.63 | 0.962 | 40 / 40 |
| 3 | 0 | 30 | 30.27 | 0.981 | 40 / 40 |
| 3 | 1 | 31 | 31.42 | 0.988 | 40 / 40 |

The same pattern appears in the FP32 re-encoded feedback-condition latent (`k95=27–30`, full-span held-out residual `0.697–0.976`) and final float RGB (`k95=23–30`, residual `0.553–0.960`). Truncating at `k95` changes the held-out residual very little: most failure is already present with all 32 training columns, not caused by overly aggressive truncation. Some entropy effective ranks are small because dominant modes coexist with a consequential long tail; they are not evidence for an accurate low-rank operator on unseen directions. Seed changes can materially alter the observed spectrum at a fixed scene.

The two no-perturbation baselines were exactly equal in float output in all six strata. The half-step diagnostic passed all 40 directions in five strata. Record 1 / seed 0 failed on five directions in predicted and feedback latent and three in RGB; its spectrum must be described strictly as a finite-amplitude response spectrum. Even a 40/40 half-step pass gives empirical consistency at these two amplitudes, not a proof of differentiability. No matrices were pooled across scenes or seeds, and the observed rank is capped at 32.

## Audit and provenance

All 162 formal calls per stratum finished successfully. Independent verification checked raw FP32 tensor shapes/finiteness, paired noise and 30-step condition/action-token evidence, each sample's artifact hashes, singular values and metrics recomputed from saved tensors, and each 1026-entry `scan/MANIFEST.sha256`. Peak monitored GPU use was 31.55 GiB; peak cgroup memory was 5.50 GiB. GPU returned to 0 MiB with no OOM or cgroup high/max events. The expanded 110 GiB data disk had more than 74 GiB free at the final check; no older data was deleted for this scan.

Full raw tensors are retained only on the AutoDL server under `/root/autodl-tmp/cosmos-experiments/2026-09-23/umi_task9_response_spectrum_run01` through `run06`. The local verified light archive is `C:\Users\hongy\Desktop\Semester\Capstone\2026-09-23\task9_response_spectrum_run01`. This directory contains aggregate metrics, per-stratum reports and figures, configuration, source-data preflight, status, and monitoring records; it does not contain the complete raw response tensors. `matrix_summary.json`, `stratum_metrics.csv`, and `matrix_report.md` here are the reproducible aggregate outputs. The implementation and mathematical definitions are in `docs/task9_response_spectrum.md`.
