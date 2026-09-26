# Task 11 true-error figure companion

These panels summarize the immutable Task 10 true-error matrices for six episodes and two seed schedules (12 columns per ensemble and space). E1 is the G0 prediction minus its aligned 16-frame-condition ground truth; ETF2 is the TF2 prediction minus its aligned 32-frame-condition ground truth; EAR2 is the AR2 prediction minus that same 32-frame-condition ground truth. Latent panels use the saved condition-mask carrier; RGB panels use the decoded endpoint.

Cumulative energy is the cumulative squared singular-value share of the uncentered actual-error matrix. Truncation is the relative Frobenius norm residual, not an energy fraction. The LOEO panels show each of six held-out episode means (each averages its two seed columns) and the descriptive six-episode mean at fixed ranks 1, 2, 4, 8, and 10. Raw residual denominators are the held-out error norms; centered sensitivity denominators are the norms after subtracting a training-only mean. No population interval is implied.

In every centered LOEO training fold, ten centered training columns have numerical rank at most 9; therefore requested rank 10 is numerically capped for centered sensitivity. Fold-specific adaptive training-k95 ranks are retained separately in `figure_source_data.json` and are not mixed into the fixed-rank curves. The plots describe these twelve-column ensembles only; they are not full-Jacobian spectra and do not combine horizons, feedback modes, or spaces.

Source Task 10 manifest SHA-256: `e083aad1f0d6822424e10f50920faaacaf98f5a14acd0b7fd34a671264adba45`. Source analysis JSON SHA-256: `7a2773b31a165fef3eb358160f60e04b9f43aa82d6673429508186083d0ebe27`.
