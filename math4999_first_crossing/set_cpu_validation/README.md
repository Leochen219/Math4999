# Exact CPU validation package

Python standard library only. The production algorithms use `Fraction`; the separate oracle expands path coefficients and applies strict rational Fourier–Motzkin elimination, or enumerates complete finite paths. A cap or timeout is a technical failure, never an infeasibility result. Rectangular minima are checked with box KKT and rational attaining vectors.

Run implementation tests from the repository root:

```
python -B -m unittest -v math4999_first_crossing.set_cpu_validation.test_validation
```

The formal run uses the frozen protocol and must target a new directory whose parent already exists:

```
python -B -m math4999_first_crossing.set_cpu_validation.campaign --protocol evidence/math4999_next_step_20261105_001/CPU_PROTOCOL_V1.md --output <new-directory> --max-seconds 600
```

Manifest precedes all cases; JSONL retains completed and failed cases. Summary and hashes are emitted at normal or handled early stop. An abrupt process kill or filesystem failure can prevent final files; such an incomplete directory is not a pass. Cases use exact rational strings; elapsed time is the only floating-point scientific-output field. The independent reviewer and formal run are the root agent's responsibility.
