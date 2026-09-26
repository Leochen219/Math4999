# Task 11 — independently reviewed results

The five-chunk real-trajectory experiment completed 20 formal calls plus one separately counted engineering smoke. Both exact G0 repeats and all matched noise/action/feedback gates passed. Independent raw-tensor analysis passed 1,074 comparisons (maximum absolute difference 3.55e-15). The separate true-prediction-error spectrum analysis passed 1,182 independent comparisons and reused the historical Task 10 tensors without regeneration.

## Main findings

- Five-chunk RGB errors are nonmonotone. At h=2,3,5, AR exceeds TF in both schedules; at h=4 it is lower in both. Mean RGB differences are +0.004404, +0.028064, -0.041859, +0.014237.
- All eight paired condition-latent RMS differences are positive. RGB and latent errors are different metrics and need not agree.
- A negative cross term in `MSE_AR - MSE_TF = mean(2bp) + mean(p²)`, with `b=TF-truth` and `p=AR-TF`, explains the negative h=4 RGB differences algebraically. This is not a causal decomposition into separately identified model and numerical error.
- The three actual condition-latent error ensembles each require 10/12 modes for 95% energy; full training-span leave-one-episode-out residual norms average 0.966–0.970. Current samples do not support a very small transferable linear error subspace.

## Scope

Horizon study: one fixed Bridge episode (record 15), first 81 observations, five consecutive motion chunks, two fixed seed schedules. Error-matrix study: six fixed episodes, two seeds each, separate E1/ETF2/EAR2 matrices. Future observed poses enter Bridge motion conditions. Neither study establishes control-only open-loop accuracy, a global Lipschitz constant, monotone/exponential growth, long-term stability, or the rank of the full Jacobian.

## Contents

- `offline_true_error/`: immutable offline figures, source data, manifests, source/review ZIPs and independent audits. Its math review records the earlier offline checkpoint.
- `five_chunk/`: five-chunk lightweight evidence and review bundle; full original tensors remain on the server.
- `horizon_figures/`: reviewed five-horizon PNG/SVG figures, exact source rows, captions and hashes.
- `../final_report/task11_results.tex`: verified results fragment for later integration; not standalone or a claim that the eight-page final report is complete.

The Notion theme page contains the reader-facing results and uploaded images:
https://app.notion.com/p/3e4ccd7d08e38166ab78c691da632ee7

Remote root: `/root/autodl-tmp/cosmos-experiments/2026-09-25/umi_task11_horizon_error_run01`.
Five-chunk identity: `bb60d97e1ca86ac1b4ed1df36a36112c7ed83cef653fb6806f51317e7db78436`.

No historical tensors were deleted, no model/environment upgrades were performed, and no additional model generations were run after the approved 20 formal calls.
