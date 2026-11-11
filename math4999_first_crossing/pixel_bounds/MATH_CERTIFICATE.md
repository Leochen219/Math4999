# Conservative arithmetic certificate for the authorized implementation

These are symbolic implementation conditions, independently checked by Astra before execution. No test or dataset computation was performed to derive this note.

## Coordinates and box

Stored float32 RGB widens exactly to binary64, but 1-x need not be exactly representable (for example x=2^-100). Use the relaxed box Kplus=[-x, nextafter(fl(1-x),+infinity)], which contains the exact box [-x,1-x]. The unary minus is exact. Enclose exact w=x-x0-stored_mu with outward binary64 subtractions. Every basic arithmetic result is bracketed by nextafter under correctly rounded binary64, gradual underflow, no overflow and no fused/fast-math change of the prescribed operations.

For each interval endpoint v of w, fixed stored lambda>0, and finite exact 2*lambda, bracket v/(2*lambda), then clip both interval endpoints to the same Kplus. This interval contains the exact maximizer of v*z-lambda*z² on Kplus. Interval evaluation of the objective therefore supplies a SUPREMUM upper bound, not merely a primal value. At lambda=0 the objective is linear and endpoints suffice. The coordinate supremum is convex in v, so the maximum of the two endpoint upper bounds covers all w in its interval. For -w use [-w_hi,-w_lo]. Zero belongs to Kplus, hence each certified coordinate upper bound must be nonnegative; a negative computed upper bound is a certification error, not an opportunity for silent repair.

## Nonnegative reduction

Let p_i be finite nonnegative stored upper bounds, S=sum_exact p_i, S_hat their binary64 sum, u=2^-53, ell=2^-1074, gamma_D=D*u/(1-D*u), and D*u<1/2. Each term must be used exactly once with at most D-1 effective additions. For any reduction tree under fl(a+b)=(a+b)(1+delta)+eta, |delta|<=u, |eta|<=ell,

S <= (S_hat + 4*D*ell)/(1-gamma_D).

Proof: each original nonnegative term has a path multiplier at least 1-gamma_D. Each local additive error is amplified by at most 1/(1-D*u), and there are fewer than D such errors. Thus S_hat >= (1-gamma_D)*S - D*ell/(1-D*u); the latter correction is <2*D*ell, so 4*D*ell is conservative. Exact addition of an initializing zero does not change this count. All scalar constants and final arithmetic use Fraction; never first round the underflow correction to float.

This remains conditional on NumPy performing the specified ordinary float64 reduction without overflow, dropped/duplicated terms, FTZ/DAZ or unsafe algebraic changes. Runtime platform and finiteness checks are necessary, not a formal verification of NumPy.

## Scores

Divide the certified reduction upper bound by D and add exact stored_lambda*(E_hi)². Take the minimum across the frozen valid parameter candidates and the valid radial cap R_hi*E_hi, separately for both signs. Use the frozen theory's squared a_plus upper and a_minus lower formulas with exact Fractions. For square roots, a floating initial estimate may be moved outward with nextafter until its square, tested exactly with Fraction, brackets the rational radicand. Exact zero is separate; an interval containing zero is not an exact-zero certificate.

The prior 1e-7 RMS guard continues only for its stated R/E/A norm inputs. It does not certify this new coordinate/dual reduction. Scores are rational, candidate LOWER scores determine acceptance with threshold UPPER values, and candidate UPPER scores are used only for impossibility-of-deletion certificates. Stored old upper bounds and lower floors cap the new conservative values in the mathematically justified directions. All failures stop the single authorized attempt.
