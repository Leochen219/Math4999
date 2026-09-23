# UMI Task 9: Bridge response-spectrum stratum

Scene: `{"episode_id": 15, "file_path": "/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/rss/toykitchen2/pnp_sweep/120/train/out.npy", "language": "put the red object into the pot", "record_index": 2}`; diffusion seed: `1`.
32 independent training directions, eight independent held-out directions; all 40 directions have a paired half-step check.

| Output space | Half-step passes | k90/k95/k99 | Effective rank | Unresolved energy | Held-out residual at k95 (mean/max) |
|---|---:|---:|---:|---:|---:|
| feedback_condition | 40/40 | 29/30/32 | 31.2749 | 0 | 0.9761/0.9955 |
| float_rgb_final | 40/40 | 28/30/32 | 30.7699 | 0 | 0.9606/0.9832 |
| predicted_latent | 40/40 | 29/31/32 | 30.6335 | 0 | 0.9619/0.9900 |

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
