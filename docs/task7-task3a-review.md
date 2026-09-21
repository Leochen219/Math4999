# Main review of Task3a numerical API

Approved at 7784c39 for offline evidence integration. Main independently ran the focused NumPy suite:16 tests,0.004 seconds,OK. Earlier13-test checkpoint also passed independently. Scope-reviewed formulas against the independent affine oracle: FP32 differences/FP64 norms, actual signed step lengths, earlier-q relative-change denominator, condition-mask amplification, actual delta1 ray and fixed beta0.1 estimator.

Development corrections addressed multidimensional cosine reduction, the affine fixture's (1,5) response, per-ray reliability, strict boolean flags, zero-denominator reasons and distinction between reliable prediction FAIL and unreliable estimation. Three prescribed points remain mandatory; no adaptive beta/direction selection or additivity claim.

This certifies numerical fixtures only, not model evidence. Task3b must independently validate sources/conditions/noise/dtypes/counts, pass per-ray computed flags and confirm beta/source labels from records before using these functions. It must not treat a caller-provided true flag as evidence by itself.
