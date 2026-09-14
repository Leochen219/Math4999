# Conditional finite-difference error bound

This note is the mathematical calibration layer for Task 2. It is separate
from the CPU mechanism implementation and makes no claim about a Cosmos
runtime. The exact ASCII form of the bound proved below is:

```text
||dhat_h-J_F(z_bar)v|| <= M*h/2 + L*||e_h||/h + 2*eta/h
```

Plain-text notation required by the specification: `Q(z_bar)=z_bar`,
`e_h=Q(z_bar+h*v)-(z_bar+h*v)`, and `unit v` means `||v||=1`.

## Definitions and assumptions

Let Z and Y be finite-dimensional normed vector spaces. The same notation
`||.||` denotes the chosen vector norm and its induced operator norm. Let
`D` be a convex region containing all line segments used below. Assume
`F:D -> Y` is twice continuously differentiable and, throughout this region,

```text
||D^2 F(z)[a,b]|| <= M*||a||*||b||
||F(a)-F(b)|| <= L*||a-b||
```

The second inequality is the local Lipschitz assumption. The first supplies a
uniform second-derivative bound M. The proof only needs the relevant segments
to stay in D, so these constants may be local rather than global.

Choose a base point `z_bar` in D, a direction `v` with unit norm, and a step
`h > 0`. Let Q be an input quantizer or representation map. The base point is
represented exactly:

```text
Q(z_bar)=z_bar
```

At the perturbed point define the quantization error:

```text
e_h = Q(z_bar+h*v) - (z_bar+h*v)
```

Thus `Q(z_bar+h*v)=z_bar+h*v+e_h`. Let the measured downstream response at
either evaluation be `F(q)+rho_q`, where the response error is bounded at the
relevant boundary by `||rho_q|| <= eta`. Write `rho_plus` and `rho_0` for the
perturbed and base response errors. The one-sided measured difference is

```text
dhat_h = [F(Q(z_bar+h*v)) + rho_plus
           - F(Q(z_bar)) - rho_0] / h
```

The point `Q(z_bar+h*v)`, together with the segment joining it to
`z_bar+h*v`, is also in the region where the L bound holds. No assumption that
`e_h/h` converges is made.

## Step-by-step proof

1. Add and subtract the exact one-sided difference and the unquantized
   perturbed response. Since `Q(z_bar)=z_bar`,

```text
dhat_h - J_F(z_bar)v
 = [F(z_bar+h*v)-F(z_bar)]/h - J_F(z_bar)v
   + [F(z_bar+h*v+e_h)-F(z_bar+h*v)]/h
   + [rho_plus-rho_0]/h.
```

   These are, respectively, Taylor truncation, input representation, and
   downstream response terms.

2. Apply the fundamental theorem of calculus to the Jacobian along
   `z_bar+t*v`:

```text
F(z_bar+h*v)-F(z_bar)
 = h*J_F(z_bar)v
   + integral(t=0..h) [J_F(z_bar+t*v)-J_F(z_bar)]v dt.
```

   The Hessian bound and unit v give

```text
||[J_F(z_bar+t*v)-J_F(z_bar)]v|| <= M*t.
```

   Integrating and dividing by h therefore gives

```text
||[F(z_bar+h*v)-F(z_bar)]/h - J_F(z_bar)v||
 <= (1/h) * integral(t=0..h) M*t dt
 = M*h/2.
```

3. Apply the Lipschitz assumption to the input-representation term:

```text
||[F(z_bar+h*v+e_h)-F(z_bar+h*v)]/h||
 <= L*||e_h||/h.
```

4. Apply the triangle inequality and the two downstream error bounds:

```text
||[rho_plus-rho_0]/h||
 <= (||rho_plus||+||rho_0||)/h
 <= 2*eta/h.
```

5. Combining the three estimates with the triangle inequality proves

```text
||dhat_h-J_F(z_bar)v|| <= M*h/2 + L*||e_h||/h + 2*eta/h.
```

The `M*h/2` term is the one-sided Taylor remainder under a bounded Hessian. A
centered difference is a different estimator; its sharper second-order
truncation statement requires an additional third-derivative (or
Jacobian-Lipschitz) assumption and is not silently substituted into this
one-sided proof.

## What this does and does not establish

- The argument is a direct application of ordinary numerical-analysis tools:
  Taylor's theorem/fundamental theorem of calculus, a local Lipschitz bound,
  the quantization error `e_h`, and the triangle inequality. It is a
  conditional bound, not an empirical estimate of its constants.
- Cosmos L, M, and eta are **not estimated** by the CPU calibration. The
  calibration function has known derivatives; that does not identify runtime
  regularity or downstream noise in Cosmos.
- **FP32 is not ground truth.** The artifact calls the FP64 calculation a
  numerical reference while keeping the represented FP32 matrix values fixed;
  wider arithmetic does not make the underlying model exact.
- A baseline zero response only says that the observed difference at that
  baseline is zero. **baseline zero does not imply eta=0**: the error bound is
  about an unobserved error budget at the response boundary.
- A single direction tests one directional derivative only. It does not prove
  the full Jacobian, a Jacobian rank statement, or low-rank behavior.
- A scan with no stable finite-difference window is an inconclusive numerical
  result. A **no-window result** does not prove that the underlying function is
  non-differentiable; quantization, cancellation, and downstream error can
  destroy a usable window.
- The upper bound contains terms that can increase as h decreases, but it does
  not require the observed curve to show a `1/h` regime or a U shape. The
  behavior of `||e_h||`, eta, rounding, and unmodeled effects can differ from
  the assumptions or dominate only over part of the scan.
- The CPU scan is method calibration. Its PNG/SVG/CSV/JSON artifacts are not
  Cosmos mechanism evidence and do not authorize loading Cosmos, SSH, GPU
  generation, or changing dependencies.
