# Task1 main review, 35064b3

Spec: not yet approved. Quality: changes required. Main ran real Torch2.10 CPU tests via isolated remote directory: 6 tests PASS (0.067s), warning from fixture requires_grad scalar conversion. These supersede local all-skipped evidence but do not supply missing preimplementation RED evidence. No GPU/model calls.

Important findings (fix round1):

1. Real Wan _dec_cache is a fixed-length list of None after clear_decoder_cache(), not an empty list. _clear_cache currently .clear() empties it and doesn't call clear_decoder_cache. Use official clear methods (including this one), recognize recursively empty/None slots, preserve cache structure. Test actual-style nonzero-length None list and failure. Avoid claiming exact restoration while changing cache shape.
2. _PrecisionSnapshot.restore catches AttributeError/RuntimeError and silently passes; FeedbackEncoder.finally may raise from hook restoration or precision restoration before remaining cleanup runs. Attempt ALL cleanup components, aggregate/annotate failures and stop publication. Test a restore failure still clears cache/restores remaining flags/hooks and cannot return success. Record after identity; verify params/buffer/constants/metadata round-trip against before; cache normalized empty rather than raw cached equality.
3. _call_encoder catches any TypeError/ValueError from inside encoding and invokes it again with device. This can double an actual failing inference and violate no retry/formal accounting. Resolve signature before call or restrict to binding error without execution; supported Wan method needs no retry. Test one-call counter when body raises ValueError.
4. If TorchDispatchMode import fails, observer silently yields and no operation_count guard exists. Strict FP32 must fail closed absent observed operations. Add positive-count/observed flag and per-call observed FP32 weight/buffer/constant dtype metadata, not just conversion assertions. Test hidden cast, missing observer, backend flags initially true restored afterward. Fix fixture .detach warning and test real plain wrapper -> plain Wan -> nn.Module chain (matching installed source).

No requirement to rerun entire 3-minute missing-matplotlib local suite each fix: focused covering tests now, full remote suite later once branch integration ready. Do not install dependencies. User requested main independent review; main repeats scoped actual CPU verification after patch.

## Round1 re-review b55c6ce

Spec compliance APPROVED for encoder seam; code quality APPROVED for this scope. All four findings addressed: official fixed-slot clear, aggregate cleanup and explicit round-trip verification, pre-bind/no body retry, fail-closed dispatch/state dtype evidence. Main independently ran 12 real Torch CPU fixture tests (0.046 s, all PASS) and matched both remote/local SHA256:

- encoder f8ebbc10b9e10d08868f51f6cc792b4d69d2f47ea63a86514c1c50e705d5b128
- tests d1d485fd1b1f6f462e34a4d8d562599f457d4fbdf7d5489ec26fc7ed6bbbc825

This is not real VAE GPU proof. Actual native parity and operation coverage remain Task4 live engineering gates. Round-trip checks preserve exact object/storage metadata with compact value probes, not full per-call weight hashes; immutable source checkpoint hashes and live parity provide separate evidence. Initial TDD RED transcript was not supplied, a process limitation; current covering tests execute, not skip.
