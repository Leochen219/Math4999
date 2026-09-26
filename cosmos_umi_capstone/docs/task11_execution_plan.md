# Task 11: Five-chunk real-trajectory error and true-error spectra
User-approved plan, 2026-09-25. Primary user requires gpt-6-luna xhigh implementation/execution and root independent review.
## Global constraints
Preserve all Task 4-10 behavior/results. No deleting historical data, no replacing trajectory/seeds/thresholds based on results, no model/environment downloads or upgrades. GPU generation ONLY after root code/CPU/data review then smoke approval then formal approval. Offline CPU remote analysis can run after root review. One model/process; no overlapping decoder/generator runs; strict atomic outputs and resume identities. SSH alias seetacloud-umi, port 21175. Server verified online with GPU 0MiB and disk70GiB.
## Task 1: Implement and test Task 11, prepare immutable upload and preflight
Work in this isolated worktree on codex/task11-horizon-error. Existing base08700b2; cosmos_umi_capstone contains Task10 and experiments/task8_frozen verified runtime. DO NOT SPAWN SUBAGENTS. Read applicable instructions and test-driven-development skill before implementing. Run existing focused regression baseline first; preserve default semantics.

A. Offline true prediction error analysis:
Source remote /root/autodl-tmp/cosmos-experiments/2026-09-25/umi_task10_multi_episode_error_run03. Six records [1,2,3,6,7,14], schedules [0,1],[2,3]. Validate provenance and hashes, actual output dtype/mask and truth encoding. Use saved final decoded RGB and encoded_condition, ground_truth_conditions.npz keys gt_condition_x16,gt_condition_x32,condition_mask. No regeneration. Repetition is not an extra column.
Separate error ensembles E1, ETF2, EAR2, each12 columns (6 episodes x2 seeds), in condition-mask latent primary and float final RGB auxiliary. Subtract in float64, reductions float64, raw sources immutable. Each column prediction minus aligned truth. Do not combine horizons/modes/spaces.
Main uncentered SVD retain amplitude. Report singular values, cumulative squared-energy, k90/k95/k99, exp entropy of squared singular value fractions, truncation relative Frobenius residual. Zero matrix and zero column handling N/A.
Six leave-one-episode-out folds: both seeds held out, other10 fit. Ranks1,2,4,8,10 and training-only k95; cap to numerical rank explicitly. Projection residual is optimal representation of known error, not trained prediction. Centered sensitivity uses TRAIN mean only for heldout; label residual relative to centered target and expose raw reconstruction residual too to avoid denominator confusion. Report means per episode before six-episode summary; no posthoc PASS.
Compute b=TF2-truth,p=AR2-TF2 separately RGB/latent; save mean squared b,p,cross term 2dot(b,p)/n, squared-error difference, identity residual, cosine/N/A.
Bound memory via sequential tensor loading, close NPZ/mmap, thinSVD or small Gram with validated numerical handling; no all frames/model load in offline path. Source metric reproduction tolerance explicit (float64 should match Task10).

B. Long horizon:
Fixed candidate record15 in /root/autodl-tmp/datasets/bridge_v2_subset_20260921;52record shard. Local preflight inventory confirms89frames,language,sequence flags; revalidate COMPLETE, hashes, camera, decoded images, temporal and pose-action semantics across FIRST81frames,80aligned transitions. Five (16,10) motions via exact official backward_framewise/rot6d/quantile/gripper conversion (future observed poses); no padding,repeated actions,episode crossing,replacement. Use Task9 official preflight code/Task10 frozen paths; old33frame assertions require a separate extension without changing defaults.
FP32G/D/E Torch2.10.0+cu130 CUDA13 GPU0, UniPC30 guidance1 shift10, no cache/autocast/TF32. Quantitative float before video.
Schedules [0,1,2,3,4] and [5,6,7,8,9]. Per schedule: G0 + exact repeat2; AR h2-5 four calls feeding own previous FP32 encoded float endpoint; TF h2-5 four calls conditioned on real x16,x32,x48,x64. Total20 formal generation,1 separate smoke; encode setup/truth separately counted. First chunk commonTF/AR not duplicated statistics. Actual actions, noise hashes and consumed condition must be verified every step. First repeat input/action/noise/predlatent/float exact else STOP.
Save generated float full16frames/final,condition,truth endpoints,noise/action hashes,metadata,resource logs,atomic status. Reencode real endpoints FP32; shared true endpoint conditions across schedules identical.
Metrics eachh1..5 TF/AR endpointRGB RMSE MAE PSNR, condition latent RMS/cosine,deltaAR-TF,AR adjacenth changes,error geometry. Time-aligned perframe only. Display bothseeds and descriptive mean, no populationCI/exponential-growth claim.

Resources reuse Task10 proven gates including real cgroup quota,swap,RSS,GPUgrowth,OOM. Inspect source exact limits. PreloadGPU<=1GiB,smoke peakalloc<=35GiB,NVML<=45GiB; runtime used>75GiB/free<20GiB/reserved>65GiB hardstop; cgroupfree<10GiB hardstop; diskforecast samplebytes*remaining*1.3 leaves>=5GiB. No auto OOMretry/no deletes. Disk70GiB presently, oldsample~82MB not substitute smoke. Explicit identity-bound resume,successful outputs never overwrite.

C. Interfaces/test/delivery:
Add separate runner/analyzer/preflight helpers as necessary with explicit stage,input-run,record,horizon,seed schedules,run-dir,resume CLI. Keep files cohesive; avoid unrelated refactor.
Tests: fivechunk slicing/no leakage,seed/action routing,ownARfeedbackvsTFtruth,repeat gate,exacterrorgeometry,knownrank matrices,centeredLOEOtrainingonlymean,rankzero,zeroerrors/PSNR,resources,resume. Focused Task8/9/10 regressions then full CPU suite once. Never use mocks as evidence real model ran.
Prepare read-only remote preflight after CPU test success allowed (no GPU model). Upload small code archive only to unique versioned path, no overwrite verified server source. Report codehash,preflightresults,tests,exactsmokecommand to root then WAIT for review. No generation before explicit root release.
Remote output lowestfree /root/autodl-tmp/cosmos-experiments/<actual-date>/umi_task11_horizon_error_runNN. Local C:/Users/hongy/Desktop/Semester/Capstone/<actual-date>/task11_horizon_error_runNN. Large tensors remote, lightbundle local SHA256. Later after review root updates Notion/LaTeX. No GitHub push/merge unless root authorizes.
## Task 2: Reviewed execution and independent audit
After Task1 root review, Luna runs offlineanalysis and preflight, submits evidence; root releases1smoke then20formal. Luna monitors sequentially and reports meaningful progress. Root independently recomputes spectra,folds,errorgeometry,metrics from saved tensors; freezesmanifest and local lightarchive. User already authorized entire scoped experiment; no unnecessary additional permission asks.
## Source locations
LocalTask10bundle C:/Users/hongy/Desktop/Semester/Capstone/2026-09-25/task10_real_scenes_run03/review_bundle.
LocalPy C:/Users/hongy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe (numpy/pytest availability verify), alternative user's miniconda; do not install unnecessarily.
Proposal final_report C:/Users/hongy/Desktop/Semester/Capstone/2026-09-25/final_report.
## Scientific limits
Real motion contains futureposes,notcontrol-only. Task11long horizon oneepisode,twoseeds,5chunks;trueerrorensembles6episodes,2seeds. No generalization/populationstability claim. Ranks<=12byconstruction; spectra are trueerror ensembles not Jacobians. Never conflate this with Task9response spectra.

