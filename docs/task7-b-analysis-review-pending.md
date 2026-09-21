# Pending independent B analysis review

RESOLVED 2026-09-21: revision 71fc887 adds the own-trajectory feedback check. Main independently verified all 16 B calls, all eight feedback connections, 30-step condition consumption, FP32 evidence and raw paired noise. C held-out truth was subsequently corrected in fc8c074 to use B's actual second-step difference; ddc0370 preserves original admission-gate provenance during offline reanalysis. Final analyzer tests: 29/29 pass. Historical review request follows.

This does not block the engineering smoke or stage A. B/C analysis is not yet released.

At analyzer revision 6c9f05f, `_b_stage_analysis` validates within-call condition readbacks and explicitly checks baseline step-1 input against baseline step-0 encoded output. The six perturbed trajectory loops compute deltas but do not independently compare each perturbed step-1 `condition_input_fp32` against its own step-0 `encoded_condition`. Add this byte-exact check and a negative fixture before approving B evidence. Do not substitute baseline feedback for the perturbed trajectory.

Also review actual G/D/E dtype evidence, raw noise readbacks, and Task6 FP32 Decoder frame parity against real saved B records; the current summary-only checks do not by themselves independently establish all of those requirements. Preserve the existing runner gates and fixed scientific thresholds.
