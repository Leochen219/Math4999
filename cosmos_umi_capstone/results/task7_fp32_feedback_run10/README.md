# Task 7 completed experiment

Start with [the independent review](independent_review.md). This directory contains only lightweight reports, metrics and figures; full raw tensors remain on the server.

- Run: 2026-09-21, `umi_task7_fp32_feedback_run10`.
- Production runtime: `58b06005982881aac420c4fc47e1cdc7ec190577`.
- Final offline analyzer: `ddc0370` (no generation rerun).
- Completed formal run10 calls: G54 / D54 / E70. Engineering attempts and A equivalence replay are disclosed separately in the review.
- A: native encoder FAIL, FP32 encoder PASS; cross-direction additivity not tested.
- B: two-step amplification 1.184–1.191; engineering/repeatability gates PASS.
- C: six local windows PASS; actual held-out two-step prediction 4 PASS / 2 FAIL. The largest original amplitude fails both signs (10.66%, 11.74%).
- Controlled repeated action, seeds [0,1]. Not a continuous ground-truth trajectory or a prediction-accuracy experiment.

Verified local archives under `Capstone/2026-09-21/task7_fp32_feedback_run10`:

| File | SHA256 |
| --- | --- |
| review_bundle.zip | c00cbf67b8ed1b704cca2caa1a036394e92f9278e2bc852d1be50671bf3c9523 |
| execution_evidence_bundle.zip | 3c287a93f6071ce7e08e15fafe2973ed684a14f275152ec82f5c5888a436b238 |
| MANIFEST.sha256 | 80c0ad724b8369501372b3d3d934c1bb6773e5f9c4771675bf7cff66f2b6da2a |

The root manifest inventories 3,231 pre-packaging files, excluding itself and the subsequently created evidence archive. Source archives and independent audit scripts are also retained locally. No model weights, credentials, videos or raw tensor arrays are included in Git.

Verification: runner 48/48 and final analyzer 29/29 passed. Local full discovery had 504 passes, 21 skips and two missing-matplotlib errors; those two tests passed on the existing remote analysis interpreter. This is split-environment verification, not a claim that the local suite was entirely green.
