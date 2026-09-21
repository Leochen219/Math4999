# Task7 main source review (pre-implementation)

## Verified live source 2026-09-21

Framework /root/autodl-tmp/cosmos-framework-task6-clean (do not edit). Wan source cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py:
- Wan wrapper encode lines1221-1239 records input dtype, casts videos to self.dtype, self.model.encode(videos,self.scale), casts returned latent back to input dtype. Thus returned FP32 is NOT internal FP32 evidence.
- scale created as tuple(mean,1/std) using native dtype. Preserve values/precision lineage. Temporary FP32 should cast existing constants, not reconstruct supposed unquantized values.
- interface encode line1508 delegates model.encode. Inner encoder caches call-local; decode cache must clear across independent samples. Compiled encoder/decoder paths, if installed, must be detected/rejected unless actual full FP32 proof obtained.

Official preparation: cosmos_framework/model/generator/omni_mot_model.py:2474 builds sequence plan, gets clean data, packs mask, then misc.arch_invariant_rand(tuple(x0_token.shape),tensor_kwargs_fp32 dtype/device,seed) and blends mask*x0+(1-mask)*noise. Prepared eight-tuple and flattened noise/reference/mask layout already verified by Task6. Dynamic feedback should preserve this construction, not change algorithm based on random.normal assumptions.

Existing OfficialPrecisionRuntime __init__ encodes once and freezes prepared with model_seed; execute rejects spec seed mismatch and reuses prepared, with a single _paired_noise_hash. Task7 cannot merely alter request seed. Use isolated bound request state with same resident model, rebuild per-step paired noise and baseline identity, never instantiate second resident model. No historical Task4-6 semantics change.

Existing execute always calls model.decode after G and stores decoded_final. Task7 needs ONE actual FP32 decode per F call. Avoid executing native decode then a second replay while claiming one decode. Request-local decode seam or explicit backward-compatible decode option may preserve existing defaults. Count actual method invocations in tests and live evidence.

Source validation review concerns:
- raw z0 must equal historical frozen carrier; don't silently replace with newly loaded encoder output.
- shape normalizations must preserve mask coordinate ordering (no reshape that scrambles channels/time).
- decode frame is last genuinely generated frame; full latent includes original conditioned frame.
- StageA exact historical native reencode comparison uses temporary_fp32 decoded_final frame and its direct_condition_latent_float32, NOT native Decoder frame.
- G FP32 context must end/restore before VAE context to avoid unintended dtype coupling. Precision evidence includes floating buffers/constants, not only first parameter dtype.
- Full tensor operation evidence may be summarized rather than storing every activation; retain key boundary arrays and per-step condition hashes.

Preflight measured GPU0 used0 MiB/free97251 MiB; cgroup limit118111600640 bytes/current359919616; disk available12995616768 bytes; torch2.10.0+cu130. Snapshot only; must recheck before live execution.
