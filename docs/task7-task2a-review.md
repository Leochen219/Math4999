# Task2a review at290c262

Main spec compliance APPROVED, code quality APPROVED for bounded full/deferred seam. Diff checked: default native decode block unchanged except conditional guard; module diagnostic excluded from new deferred option; full G validator retains scheduler/condition/noise/precision/cache rules; omitted decode only allowed with explicit deferred metadata and no dummy decoded_final.

Independent verification from project cwd, PYTHONPATH=experiments, bundled Python:
`python -m unittest test_umi_task7_official_deferred test_umi_precision_official test_umi_precision_runtime -v`
41 tests,25.384 s,OK. Tests verify30 G/scheduler steps, default/deferred identical generated latent, zero deferred decode, original net restored and working clone collectible, failure restore. Fake official-shaped CPU backend only, not GPU/real model execution.

Task2b must call full/deferred, never module. Real seeded carrier binding, FP32 D/E and stage runner remain unimplemented/unapproved. Full Task4-6 regression remains final integration gate.
