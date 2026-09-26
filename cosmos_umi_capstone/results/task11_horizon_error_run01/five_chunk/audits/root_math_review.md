# Independent mathematical review — Task 11

Status: formulas reviewed; historical raw-tensor calculation independently cross-checked against the successful production offline analysis (1,182 scalar/vector comparisons pass). Five-chunk raw-tensor audit also passes: 20 samples and 1,074 metric comparisons, maximum absolute difference 3.552713678800501e-15.

## 1. Exact feedback-error identity (no differentiability assumption)

For a fixed episode, horizon, action and paired sampling noise, flatten the aligned float RGB frame or masked condition latent into a vector in R^n. Let t be truth, f teacher-forced prediction and a autoregressive prediction. Define b=f-t and p=a-f. Then a-t=b+p exactly, and

    MSE(a,t) - MSE(f,t) = 2 <b,p>/n + ||p||²/n.

Thus nonzero feedback displacement does not imply an increase in truth-relative error. A sufficiently negative cross term can reduce the error. This is an algebraic decomposition, not identification of independent dynamical, numerical or model-error causes.

For nonzero combined denominator,

    RMSE(a,t) - RMSE(f,t)
      = (2 <b,p>/n + ||p||²/n) / (RMSE(a,t)+RMSE(f,t)).

If both errors vanish the RMSE difference is exactly zero; do not evaluate the quotient. PSNR reverses the preference direction and is not an additive norm. Compute this identity separately in each space without mixing RGB and latent units.

## 2. Multi-step bound is conditional, not an empirical growth law

Fix the step noise as part of each abstract map F_t. Let q_t be the encoded real condition and let predicted conditions obey qhat_(t+1)=F_t(qhat_t). Let e_t=qhat_t-q_t and epsilon_t=F_t(q_t)-q_(t+1). Then

    e_(t+1) = F_t(qhat_t)-F_t(q_t)+epsilon_t.

If F_t is Lipschitz with upper bound L_t on a set containing the two arguments and relevant connecting region, the triangle inequality gives

    ||e_(t+1)|| <= L_t ||e_t|| + ||epsilon_t||.

Induction yields

    ||e_H|| <= (product_(t=0..H-1) L_t)||e_0||
             + sum_(j=0..H-1) (product_(t=j+1..H-1) L_t)||epsilon_j||,

where the empty product equals 1. The induction step multiplies the bound for H by L_H and adds ||epsilon_H||. This is not evidence that errors increase monotonically or exponentially. Cancellation is discarded by the triangle inequality. Five horizons and two seeds on one trajectory can display measured error curves but cannot estimate a proven L_t or establish a population growth law. Future observed poses enter the action conditions, so this is not control-only open-loop prediction.

For the final report's more explicit ideal/implemented distinction, apply the regularity assumption to ideal real-arithmetic F_t and write Fhat_t(qhat_t)=F_t(qhat_t)+eta_t. Then the recurrence gains ||eta_t|| and the unrolled sum uses (epsilon_j+nu_j) when these are norm bounds. Do not claim the quantized implementation is differentiable merely because an ideal map is. Measured teacher-forced residuals of Fhat_t combine model and numerical effects; they are not separately identified ideal-model b_t. The exact observed b/p squared-error identity in Section 1 needs no such regularity assumptions.

## 3. Error-matrix rank and out-of-episode representation

For each separately defined space/mode/horizon, columns are errors E=[e_1,...,e_12]. The uncentered thin SVD E=U Sigma V^T yields energy p_i=sigma_i²/sum_j sigma_j². Effective rank is exp(-sum p_i log p_i), not the number of directions or the entropy of unsquared singular values. The best rank-k Frobenius approximation has relative residual sqrt(sum_(i>k) sigma_i²/sum_i sigma_i²).

Rank <=12 is a construction constraint, not a scientific finding. A small k95 relative to 12 describes this finite ensemble only; shared cross-scene structure additionally requires successful out-of-episode representation. The full model Jacobian and the distribution of all possible errors are not being estimated by these matrices.

For each leave-one-episode-out fold, both seeds of one episode are held out; train on the other ten columns. Compute basis and k95 only from those ten columns. A held-out residual ||e-U_k U_k^T e||/||e|| measures representability of an already known error, not an error predictor available before observing truth. Average the two seed results inside each episode before aggregating the six episodes.

Centered sensitivity subtracts only the ten-column training mean mu. Projection residual is r=(e-mu)-U_k U_k^T(e-mu). Report ||r||/||e-mu|| and the raw reconstruction-relative ||r||/||e|| separately. Centered training rank <=9. Zero denominators are N/A, not zero. Centering can worsen raw reconstruction compared with the uncentered basis; do not suppress it.

## 4. Independently verified offline audit

Source Task10 run03; 156 selected files verified against the run manifest and sample receipts. Source arrays are float32; subtraction and reduction use float64. Separate root code uses no production analysis imports. In condition-latent space, E1/ETF2/EAR2 each require k95=10 out of 12. Their energy effective ranks are 9.267825620850838, 9.163035390455864, and 9.539638314232391. Full ten-column training-basis held-out residuals remain about 0.95–0.98. These are evidence against a very small transferable linear error subspace for this sample; not a theorem excluding other representations, larger datasets, alignment procedures or nonlinear manifolds.

The root calculation's maximum absolute squared-error decomposition residual across the 24 episode/seed/space cases is 4.163336342344337e-17. Production spectra, effective ranks, k90/k95/k99, all six folds (both centerings and rank policies), and the 24 decompositions pass the independent comparisons. Source audit verified 1,996 manifest-bound files and all 48 formal sample records. Successful production result: `offline_true_error_retry02/attempt_01/analysis.json`, SHA256 `7a2773b31a165fef3eb358160f60e04b9f43aa82d6673429508186083d0ebe27`. Two prior failed CPU analysis attempts remain preserved; neither ran a model nor altered historical samples. They exposed a NumPy-private-API incompatibility and historical CSV field-name mismatch, corrected with regression tests and a real one-stratum integration check before the successful run.

Full ten-column uncentered training-basis held-out norm-residual means (first average seeds inside each episode, then six episodes): E1 0.9656887501717772, ETF2 0.9698221507757273, EAR2 0.9668126650223821. These percentages are residual norms, not residual squared energy. RGB auxiliary k95 values are 10, 9, 10 with corresponding effective ranks 8.888924882968322, 8.087235387426672, 8.73956693152548. The results do not prove an intrinsic high dimension or exclude a nonlinear/aligned representation.

## 5. Primary teaching/research references checked 25 September 2026

- Asadi, Misra and Littman (ICML 2018), https://proceedings.mlr.press/v80/asadi18a.html : background on multi-step Lipschitz-model error bounds in Wasserstein distance. Do not call our fixed-noise pointwise latent recursion the exact theorem from this paper; our elementary induction above is explicitly adapted.
- Cornell CS3220 SVD notes, https://www.cs.cornell.edu/courses/cs3220/2020fa/SVD.pdf : Eckart–Young–Mirsky and the squared singular-value tail formula for Frobenius truncation. Use it as a standard mathematical result, not a project contribution.
- UMD AMSC466 numerical differentiation notes, https://math.umd.edu/~mariakc/AMSC466/LectureNotes/differentiation.pdf : background on truncation versus floating-point evaluation error. The report's vector central-difference bound requires its explicitly stated third-derivative and bounded evaluation-error assumptions; repeatability noise cannot be substituted for that deterministic error bound.

## 6. Five-chunk real-trajectory results independently verified

Fixed Bridge record 15, first 81 observations, five consecutive official motion chunks, schedules [0,1,2,3,4] and [5,6,7,8,9]. This is one episode, not ten independent scenes. All 20 formal calls completed. Direct tensor checks cover both exact G0 repeats, own-trajectory AR feedback, all 30 condition steps per call, actual initial prediction-region noise pairing, actions, float precision evidence, and artifact hashes. Independent endpoint/per-frame/geometry/adjacent-change calculations pass 1,074 comparisons. No extra model runs were performed for this audit.

Descriptive means across the two seed schedules:

| Chunk | TF RGB RMSE | AR RGB RMSE | AR minus TF | TF condition RMS | AR condition RMS |
|---|---:|---:|---:|---:|---:|
| 1 (shared G0) | 0.150713 | 0.150713 | 0 | 0.345743 | 0.345743 |
| 2 | 0.141335 | 0.145739 | +0.004404 | 0.395109 | 0.426857 |
| 3 | 0.171226 | 0.199290 | +0.028064 | 0.384676 | 0.451707 |
| 4 | 0.244579 | 0.202720 | -0.041859 | 0.490345 | 0.517850 |
| 5 | 0.168763 | 0.182999 | +0.014237 | 0.415005 | 0.491864 |

Both schedules have positive RGB feedback differences at chunks 2, 3 and 5, negative at chunk 4; all eight condition-latent feedback differences are positive. These are repeated within-episode comparisons, not eight independent replicates. AR RGB error is nonmonotone and drops from chunk 4 to 5. Thus the sampled longer rollout does not support monotone or exponential error growth. This is compatible with a conditional Lipschitz upper bound: the inequality is an upper bound and discards cancellation, not a growth law.

At chunk 4, a negative cross term in the exact RGB squared-error decomposition outweighs the nonnegative feedback-change term. The latent result differs because encoding changes the metric geometry; lower RGB RMSE does not imply a smaller condition-latent distance. This algebraic explanation does not identify a specific causal model mechanism or prove that AR is generally better.

All action conditions contain future observed poses, so this remains motion-conditioned generation, not control-only open-loop forecasting. The horizon axis is chunks, not measured wall-clock trajectory time; 5 Hz is only nominally inferred. No population confidence intervals or Lipschitz constants are inferred from this single episode.
