# Task 7 FP32 feedback implementation plan

User-approved 2026-09-21. Source: explicit Task 7 implementation request in current conversation. Luna gpt-5.6-luna xhigh implements and executes; main independently reviews and releases gates. No model call until main release.

## Global constraints

Reuse Task6 run07 bridge_0/seed0 original z0, mask, v0, prompt/action. Do not reencode initial frame to replace z0. Alphas [0.001,0.003,0.01], +/- alpha * RMS(z0[M]) * v0. Same existing (16,10) actions repeated at both steps: controlled repeated-action sensitivity, NOT real continuous trajectory. Seeds [0,1], prediction noise identical within each step. CUDA13.0, torch2.10.0+cu130, GPU0, UniPC30, guidance1 shift10; no autocast/TF32/diffusion cache. No other Task6 groups/SVD/ground-truth error studies. No environment installation/upgrades. Preserve Tasks4-6 default behavior and data. No automatic deletion or retries after OOM/resource-monitor failure.

Define F_t(z)=E_FP32(P(D_FP32(G_t(z))_last)). Same condition coordinates/normalization. Actual network FP32, decoder/encoder weights/buffers/constants/inputs/critical compute outputs verified FP32 (not projection casts). Loaded BF16 weights cast to FP32 do not recover lost information. Official necessary float preprocessing only, normalized [0,1] images to [-1,1] encoder input; no redundant resize/pad at already valid shape, no uint8/PNG/MP4 quantitative feedback. Record every clamp/resize/normalization, actual tensor hashes. Fresh condition carrier/reference/initial condition state per call, prediction region paired per-step noise, no stale z0. Encoder output must exactly equal next step consumed condition; save preparation, first/last denoise condition and per-step hashes. eval/inference_mode, exception-safe dtype/hooks/cache restoration.

## Stage A

Reuse 8 saved temporary FP32 Decoder frames from Task6 decoder_cleanup01: baseline_pre, v0 alpha00/01/02 +/- and baseline_post. Native and temporary FP32 encode each SAME frame: 16 encodes, 0 G/D. Validate manifests and source hashes. Native outputs exactly equal historical direct_condition_latent_float32. Compare input/output deltas and precision evidence; reference independent stage baseline floor.
Gates: input direction cosine>=.99, +/- cosine<=-.99, no mask-out delta; response>10*floor or >0 if zero; plus/minus slopes [.8,1.2] R2>=.98; adjacent actual-step central secants cosine>=.95 and relative RMS change<=.25. Only v0, additivity NOT_TESTED. Interface mismatch is engineering stop, scientific FAIL is valid.

## Stage B

After interface/resource approval, run regardless of A scientific PASS/FAIL. 8 trajectories: baseline_pre, 6 perturbations, baseline_post; each 2 F calls =16 G/16D/16E. Initial perturbation only; each trajectory feeds own frame encoding, no re-injection or rescaling. First-step G latent exact match Task6 corresponding source.
delta_t=tilde_z_t-z_t at same condition mask, norm RMS. A_t=RMS(delta_t)/RMS(delta0), t1,t2; incremental A2/A1; zero denominator N/A, floor threshold flags. Two baseline trajectories repeat, separate floor at each step. No accuracy/Lipschitz claim.

## Stage C

Only if A scientific PASS and B wiring/repeatability gates pass. Six actual delta1 directions independently; w=delta1/RMS(delta1), no projection to original basis. Around baseline z1, h=beta*RMS(delta1), beta [.1,.2,.4], +/-, step2 action and seed1. 36 calls plus 2 shared baseline calls=38 F. Estimation amplitudes distinct from evaluated beta1. Same local gates as A. Fixed estimator beta.1 actual-step central secant times RMS(delta1), Eprop=RMS(delta2-predicted_delta2)/RMS(delta2), pass<=.10 only if local/floor gates pass; otherwise unreliable/N/A. Do not choose successful beta/directions post hoc. Total formal ceiling54G/54D/70E; engineering attempts separately counted.

## Resource and storage

One process/runtime, batch1, sequential generation/VAE, no concurrent FP32 copies. Reuse Task6 monitor incl cgroup: preload GPUused<=1GiB; smoke peakallocated<=35GiB NVML<=45GiB. Runtime stops GPUused>75/free<20/reserved>65/OOM/cleanup growth; cgroupfree<20 warning/<10 stop; preserve host RAM/RSS/swap gates. Disk>=10GiB preload, forecast 1.3*measured sample size*remaining +5GiB, stop launching if forecast fails, never delete originals. Sample atomic temp+rename/hash/gc/cache clear, close mmaps between phases. Strict resume match code/model/VAE/framework/config/directions/z0/action/noise policy. Successful samples immutable; incomplete attempts numbered.

## Tasks

### Task 1: verified FP32 VAE encoder seam
Implement isolated encoder precision adapter and tests. Main review before next task. See docs/task7-task1-brief.md.

### Task 2: feedback runtime and staged operations
After adapter review, add dynamic condition/per-step seed binding; CLI with explicit stage/smoke/resume, source roots and run-dir, fixed spec defaults. Build A/B/C plans and monitored atomic stages, fail-closed engineering gates. Preserve official preparation/noise semantics, no hardcoded mask channels. Exact source/consumption hashes; source Task6 immutable. TDD and regression Tasks4-6.

### Task 3: offline scientific analysis and artifacts
Tensor-only FP32 subtraction float64 reductions. A native-vs-FP32 metrics/local gate, B A1/A2 + floors, C all-six derivative gate/Eprop, zero handling, no inappropriate rank claim. Synthetic linear/nonlinear/high-floor/quantization/shape cases. Config/provenance/callplan/data/diagnostic CSV/PNG+SVG/math/report/manifest/light bundle, raw tensors remote. Resource logs and independent recomputation.

### Task 4: guarded live execution
Read-only preflight actual interpreter/runtime/source hashes/resource quota; baseline smoke main approval; A then main review; B then main review; conditional C. No mutations of installed framework. Upload reviewed code only. New lowest available runNN under /root/autodl-tmp/cosmos-experiments/<date>/umi_task7_fp32_feedback_runNN. Local Capstone/<date>/task7_fp32_feedback_runNN; verify copies. Main reviews real evidence and scientific conclusions; Notion update. Core/light results under cosmos_umi_capstone, no model/videos/keys/large tensors in git.
