# UMI Task 9: Bridge response-spectrum stratum

Scene: `{"episode_id": 32, "file_path": "/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/bridge_data_v2/deepthought_folding_table/stack_blocks/21/train/out.npy", "language": "put the blue cube on the right side of the table on top of the rectangular block", "record_index": 1}`; diffusion seed: `1`.
32 independent training directions, eight independent held-out directions; all 40 directions have a paired half-step check.

| Output space | Half-step passes | k90/k95/k99 | Effective rank | Unresolved energy | Held-out residual at k95 (mean/max) |
|---|---:|---:|---:|---:|---:|
| feedback_condition | 40/40 | 24/28/32 | 13.4114 | 0 | 0.8372/0.9221 |
| float_rgb_final | 40/40 | 21/27/31 | 9.2624 | 0 | 0.7188/0.9000 |
| predicted_latent | 40/40 | 25/28/32 | 12.6465 | 0 | 0.8015/0.9143 |

## Interpretation

All sampled directions pass the paired half-step diagnostic; this remains empirical local-linearity evidence, not a proof of a Jacobian.
The training directions' squared-singular-value energy can be concentrated while the independent held-out directions have large projection residuals; training compression alone does not establish a general low-rank response operator.
Baseline repeat RMS is reported separately from a heuristic FP32-scale resolution floor. Neither floor is a rigorous bound for the entire inference chain.
The three output spaces are distinct: predicted latent, actual FP32 re-encoded feedback condition latent, and final float RGB. The 32-column observed rank cannot exceed 32. No scene/seed matrices are pooled.

## Failed half-step directions

- `feedback_condition`: none.
- `float_rgb_final`: none.
- `predicted_latent`: none.

Numerical generation and the summary/CSV were completed on the server. Figures and this report were rendered locally from those unchanged lightweight files because the CUDA runtime does not include matplotlib; no model generation was repeated.
