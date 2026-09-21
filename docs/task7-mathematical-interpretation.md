# Task7: interpretation of actual feedback propagation

All quantities below use one fixed condition mask M with n selected scalar
coordinates. Define ||a||_RMS = sqrt(sum(a[M]^2)/n). The constant n is the
same at every step, so amplification ratios equal those obtained with the
Euclidean norm on these coordinates. Images and predicted-block latents
have different coordinates and are not substituted into this norm.

## The measured maps

For fixed repeated action and seed s_t (s_0=0, s_1=1), let
Fhat_t=E_FP32 o P o D_FP32 o G_t. These are the implemented finite-precision
maps with the fixed loaded weights, not an assertion about exact real
arithmetic. z0 is Task6's saved starting condition; we do not retrospectively
replace its native initial VAE encoding. All feedback encoding thereafter
uses the tested FP32 path.

Baseline: z1=Fhat_0(z0), z2=Fhat_1(z1).
Perturbed: z1'=Fhat_0(z0+delta0), z2'=Fhat_1(z1').
delta1=z1'-z1; delta2=z2'-z2, computed in FP32 and reduced in FP64.
A1=||delta1||/||delta0|| and A2=||delta2||/||delta0|| measure finite-amplitude
trajectory sensitivity. If delta1 is nonzero, A2/A1=||delta2||/||delta1||.
These values are neither ground-truth forecast errors nor proven operator
norm/Lipschitz bounds. Two steps with repeated actions are not a claim of
matching a real physical trajectory.

## Conditional linear prediction

For theory only, reserve F1 for an ideal real-arithmetic reference map with
the same fixed represented weights and specified pipeline. FP32 is not this
exact reference. Define eta(z)=Fhat_1(z)-F1(z). If F1 is Frechet differentiable
at the measured baseline z1, its ideal response delta2_star satisfies
delta2_star=F1(z1+delta1)-F1(z1)=J1 delta1+R1(delta1), where
||R1(delta)||/||delta|| tends to zero as
delta tends to zero. This is an assumption on an idealized local map; a
finite set of observed PASS results does not prove it.

For each independently observed nonzero delta1, define w=delta1/||delta1||.
For requested h, centered response is
q_h=[Fhat_1(z1+h*w)-Fhat_1(z1-h*w)]/(2h).
Actual FP32 endpoints yield measured positive and negative lengths h+ and
h-. We use q_actual=(Fplus-Fminus)/(h+ + h-), record directional cosine and
asymmetry, and reject vanished/changed input directions; recording actual
steps does not remove all effects of rounding or nonsymmetry.

Estimate at beta=.1, verify consistency at beta=.2,.4, and evaluate the
independent finite response at beta=1. The evaluated delta2 is never used
to choose a beta or optimize a fit. Prediction is
delta2_hat=q_actual(beta=.1)*||delta1||.
For delta2 nonzero, Eprop=||delta2-delta2_hat||/||delta2||.
The six rays are derived separately; even if two nearly coincide we do not
silently merge them or project to v0,v1,v2.

## Error decomposition and noise floors

Write q_actual=J1*w+e_FD. The measured evaluation response is
delta2=delta2_star+eta(z1+delta1)-eta(z1). Consequently,
||delta2-delta2_hat|| <= ||R1(delta1)|| + ||delta1||*||e_FD||
                         + ||eta(z1+delta1)-eta(z1)||.
This separates local finite-amplitude remainder, directional-estimator error
and numerical error in the evaluated response. The final term cannot be
silently omitted for measured FP32 outputs. It vanishes only for ideal exact
evaluation, not merely because repeated baseline tensors are identical.
If F1 is C3 along w with
bounded third derivative M3 and exact symmetric endpoints,
the exact-reference central quotient q_h_star obeying
||q_h_star-J1*w|| <= M3*h^2/6.
If each endpoint evaluation has absolute numerical error at most eta,
its contribution to quotient error is at most eta/h. Only after assuming
eta<=C*u at a fixed output scale can this be abbreviated O(h^2)+O(u/h).
C and M3 are not established by this experiment. Clamp and finite dtype
operations can violate smoothness; the conditional estimate is not a
universal theorem about this implementation.

For asymmetric but collinear exact endpoints, Taylor expansion instead gives
q_actual = J1*w + (h+ - h-)*D2F1[w,w]/2 + O(max(h+,h-)^2),
before adding endpoint-evaluation error. Dividing by the measured total step
does not cancel this asymmetric second-order term. If endpoint directions
also differ from +/-w, the linear term itself is the Jacobian applied to
their length-weighted direction. The stored actual direction cosines and
positive/negative step lengths therefore matter independently of the output
secant-consistency gate. We do not claim pure O(h^2) convergence without
the symmetry and smoothness assumptions above.

Repeated baselines measure nondeterministic/repeatability differences in
the same space and phase. Exact zero floor does not rule out deterministic
quantization error. Native-vs-FP32 path differences are not this noise floor.
No statistically calibrated confidence interval is inferred from two
baseline repeats. Nonzero but near-floor differences receive reliability
flags; zero denominators are N/A, not zeros or fabricated infinite ratios.

## What this study does not identify

No true future trajectory is used here. Neither delta_t nor e_FD is an
isolated learned-model bias term b_t. Failure of a local gate does not
identify one nonlinear operator as its unique cause. One starting state,
one initial direction, six finite perturbations and two paired noise draws
cannot establish global stability, cross-seed generalization or low rank.
