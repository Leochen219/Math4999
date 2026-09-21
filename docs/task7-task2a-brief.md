# Task2a: full official generation with explicitly deferred decode

Main interrupts Task2 draft before approval: current _official_module_step incorrectly substitutes scope=module, which is only step0/no scheduler. No live model calls occurred. First solve this bounded dependency, then return to staged runner.

Scope: modify existing umi_precision_official.py and umi_precision_runtime.py with explicit optional decode policy; all existing behavior defaults unchanged. Add focused tests in new test_umi_task7_official_deferred.py, using existing Task4/6 full official fake model fixtures where possible. Do not edit Task7 runner/runtime draft during Task2a except remove its incorrect module-path claim or leave clearly unapproved until Task2b. No GPU/model/upload calls. No subagents.

Required API: OfficialPrecisionRuntime.execute(spec, inputs, *, scope, decode_policy='native'). Accept native (current behavior) or deferred only for scope='full'. Full scope ALWAYS performs all configured30 steps, actual scheduler updates, full output_full, predicted slice and actual evidence coverage. Deferred means no model.decode invocation. It does NOT mean scope=module, synthetic decoded image, dummy frame, step0 exception or skip evidence. Unknown/invalid policy fails before inference.

validate_capture(..., require_decoded=True) retains default exact legacy rules. With explicit require_decoded=False and scope full, verify every existing G/input/noise/cache/dtype/scheduler rule, but require absence of decoded_final and explicit record decode_policy='deferred'. Still save predicted_latent/slicing and full condition evidence. No weakened physics/precision validation. Decoder later executed once by Task7 adapter and checked there.

At execute return/exception, all patched net/hooks/methods/metadata and request cache restored with same original semantics. Ensure deferred return holds no GPU working-net references/closures through record or model. Tests show original net identity restored and working copy collectable after call. No simultaneous VAE phase inside this deferred call. Exception path restores too.

Tests first genuine RED, then implementation: default full native path emits decoded frame and one decode; deferred full path completes30 denoise and30 scheduler observations, output_full exactly matches default native generation tensor, zero decode; validator refuses missing G proof or dummy decoded field; invalid policy early rejection; scope module never accepted for deferred; cleanup success/failure. Use old fake official fixtures, not new imaginary step API. Existing full regression same defaults. Bundled Python lacks Torch but existing fakeOps tests run NumPy; read their fixtures.

Commit only official sources/new test and root docs/task7-task2a-report.md. Report exact API, test commands and outputs, limitations. Main review will approve before finishing dynamic feedback runtime. Do not claim original Task2 complete.
