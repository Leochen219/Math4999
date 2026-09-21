# SDD ledger — plan: docs/task7-plan.md

2026-09-21 user explicitly approved Task7 implementation and Luna xhigh execution/main review.
Base 439e845. New native worktree tool cannot address nested Math4999 repo (invalid ref); manual worktree created in authorized Plan/.worktrees/task7-fp32-feedback, branch codex/task7-fp32-feedback. No originals changed.

| Cross-task interface | Check / resolution |
|---|---|
|1->2 encoder returns actual latent/evidence|No terminal float32 cast accepted as compute proof; API finalized by Task1 review.|
|2->3 records and consumption|Runtime saves actual mask, condition and noise; analysis may not infer hidden state.|
|3->4 scientific vs engineering gates|A science FAIL permits B finite propagation, never C; interface/resource failure stops.|
|1 internal|Native parity preserved; FP32 wrapper conversion and exceptional restore required.|
|2 internal|54 G ceiling; optional C38 plus B16. A16 E; total E70.|
|3 internal|v0 cannot test additivity; NOT_TESTED. C six actual directions need own noise-floor validation.|
|4 internal|Only reviewed new code; no model downloads, no deletes; stage authorization main review.|

Task1 pending implementation. Task2/3/4 pending. User plan is authoritative spec.
Review roles: explicit user requests main as independent reviewer; main performs task-scoped and final review (not additional unsolicited reviewer seats).

Task1 implementer /root/luna_task7_encoder (Luna xhigh). Main authorized ONLY small-code CPU-test upload to fresh task7-cpu-check directory with CUDA_VISIBLE_DEVICES empty, no model loading. GPU experiments not released.
Preliminary review while tests developed: guard torch=None class definitions; repair invalid test RGB >1 and missing fixture train state; snapshot actual parameter/buffer collection slots because Module.to replaces buffers; inspect converted slot not old reference; control cuda.matmul.allow_tf32; never silently swallow restore failures. Implementer notified; final review pending.
Read-only environment: CUDA torch2.10.0+cu130, GPUused0, cgroup110GiB, disk12.1GiB. Data source8 frames and raw hash confirmed. docs/task7-source-review.md and task7-mathematical-interpretation.md completed by main; docs/task7-task2-brief.md prepared but not dispatched.

Task 1: fix round 1/5 (4 addressed, 0 open; code commits 35064b3..b55c6ce). Main scoped review docs/task7-task1-review.md; local/remote hashes equal and 12 Torch CPU tests PASS. No model/GPU calls.
Task 1: complete (commits 439e845..b55c6ce, encoder-seam review clean). Real native parity/FP32 compute remain live stage gates, not certified by fixtures.
Task 2: released for code implementation only. Task3 brief ready; no live stage release.
Task2 implementer /root/luna_task7_runtime (Luna xhigh), base0313c9c. Main independent source audit PASS across all recorded artifacts for 8 raw and8 FP32 decoder samples; details docs/task7-source-review.md. No GPU calls yet.
