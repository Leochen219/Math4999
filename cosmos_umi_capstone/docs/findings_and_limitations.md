# Findings and Limitations

## Supported findings

At the examined post-VAE condition-latent base point, the selected finite
directions show a stable first-order response in the predicted latent. Their
sum responses are close to the sum of individual responses, and the held-out
finite-amplitude predictions meet the planned error threshold.

The output image measurement is materially affected by decoder precision:
native BF16 and temporary FP32 decoder replays behave differently even when
the upstream latent experiment is matched. This is why RGB and latent metrics
are reported separately.

## Not supported

The measurements do not establish any of the following:

- a global Jacobian or global Lipschitz constant;
- a formal upper bound on multi-step rollout error;
- stability beyond the tested one-chunk setting;
- generalisation across seeds, actions, scenes, checkpoints, or hardware;
- a low-rank response or error subspace.

Five directions can span at most rank five, so they are insufficient for an
effective-rank claim. A future SVD study needs more directions, independent
seeds, and multiple initial scenes. A prediction-error study also needs
trajectories with aligned true future frames and action sequences.

## Theory connection

If each local transition map has a Lipschitz upper bound L_t and introduces a
one-step error ε_t, repeated application of the triangle inequality gives a
finite-horizon bound consisting of the initial error multiplied by products of
future L_t values plus each newly introduced error multiplied by the remaining
products. Empirical directional gains are observations in a local
neighbourhood; they must not be substituted for proven L_t upper bounds.
