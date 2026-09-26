# Task 11 five-horizon figure captions

Source record index: `15`. Run identity SHA-256: `bb60d97e1ca86ac1b4ed1df36a36112c7ed83cef653fb6806f51317e7db78436`. Source metrics SHA-256: `a4e1b3a3b0bb6dd12a5b61d2afde43b16340d4350bffba5baa73f7321c748dce`.

Scope is one episode with two seed schedules (two preregistered schedules). The plotted two-seed means are descriptive only; there are no population confidence intervals or population claims. Endpoint errors are relative to real trajectory frames under future-pose motion conditions. The five observed chunk endpoints are shown without extrapolation or smoothing; connecting segments are visual guides, not fitted growth laws. No exponential or population-level growth claim is made.

Figure 1. Truth-relative decoded RGB endpoint RMSE (normalized RGB units) and condition-mask latent RMS (latent units) versus generated horizon h1–h5, in separate panels. At h1 the single G0 endpoint for each schedule is plotted once; TF and AR endpoint traces begin at h2. At h2–h5, solid/circle denotes TF and dashed/square denotes AR. Dark-gray traces are descriptive two-schedule means. Latent values use the saved image-condition latent mask; future poses are action conditions.

Figure 2. Matched same-schedule AR minus TF endpoint RMSE differences at h2–h5, with a visible zero line. The latent panel is a difference in condition-mask RMS (not RMSE). Negative differences are retained; the dark-gray line is the descriptive mean of the two paired schedules.

Figure 3. Per-schedule squared-error decomposition at h2–h5, shown separately for decoded RGB and the condition-masked latent carrier. With `b = TF − truth` and `p = AR − TF`, the observed `MSE_AR − MSE_TF` is compared with `mean(2 b p)` and `mean(p²)`. The independently computed identity residual is retained in `figure_source_data.json`; it is not hidden by defining the observed difference from the right-hand side.
