# UMI Task 9: Bridge response-spectrum stratum

Scene: `{"episode_id": 32, "file_path": "/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/bridge_data_v2/deepthought_folding_table/stack_blocks/21/train/out.npy", "language": "put the blue cube on the right side of the table on top of the rectangular block", "record_index": 1}`; diffusion seed: `0`.
32 independent training directions, eight independent held-out directions; all 40 directions have a paired half-step check.

| Output space | Half-step passes | k90/k95/k99 | Effective rank | Unresolved energy | Held-out residual at k95 (mean/max) |
|---|---:|---:|---:|---:|---:|
| feedback_condition | 35/40 | 23/27/31 | 12.5536 | 0 | 0.6997/0.9721 |
| float_rgb_final | 37/40 | 16/23/30 | 6.7824 | 0 | 0.5548/0.9444 |
| predicted_latent | 35/40 | 17/25/31 | 6.4708 | 0 | 0.6313/0.9652 |

## Interpretation

At least one output space fails the complete 40-direction half-step gate. Those spaces have finite-amplitude response spectra, not a validated Jacobian spectrum.
The training directions' squared-singular-value energy can be concentrated while the independent held-out directions have large projection residuals; training compression alone does not establish a general low-rank response operator.
Baseline repeat RMS is reported separately from a heuristic FP32-scale resolution floor. Neither floor is a rigorous bound for the entire inference chain.
The three output spaces are distinct: predicted latent, actual FP32 re-encoded feedback condition latent, and final float RGB. The 32-column observed rank cannot exceed 32. No scene/seed matrices are pooled.

## Failed half-step directions

- `feedback_condition`: train_13, train_14, train_27, holdout_00, holdout_01.
- `float_rgb_final`: train_13, train_14, holdout_01.
- `predicted_latent`: train_13, train_14, train_27, holdout_00, holdout_01.

Numerical generation and the summary/CSV were completed on the server. Figures and this report were rendered locally from those unchanged lightweight files because the CUDA runtime does not include matplotlib; no model generation was repeated.
