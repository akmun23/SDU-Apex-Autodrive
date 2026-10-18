# v2.2.4 — Guarded IPOPT feasibility continuation (in-place)

## Actual failure diagnosed from user log

The v2.2.3 run did **not** find a feasible relaxation at dynamics tolerance
0.5, 0.2 or 0.07. The old implementation nevertheless passed each failed
iterate into a tighter problem. At 0.015 the nonlinear program produced NaNs,
and IPOPT's restoration phase failed. The log reported `max violation=None`
because diagnostics relied on reading the solver's debug state rather than
re-evaluating the numerical candidate and its bounds together.

## Changes

- Phase-I checks the initial seed against the current relaxed NLP before
  solving. If it is already feasible, it preserves it instead of running a
  potentially destructive local solve.
- Every stage checks the *current* NLP constraints evaluated at exactly its
  candidate variables. It advances **only after** that stage meets the
  existing strict feasibility tolerance of 1e-3.
- On any failed stage it **stops** rather than pushing a divergent/NaN point
  into the next, tighter stage. Printed and JSON diagnostics include the
  current stage, starting violation, resulting violation, worst constraint
  labels, and solver status.
- The final min-time solve remains strictly gated on reaching the *zero*
  dynamics-defect stage and satisfying original full-model constraints.
- The stage schedule is now 0.5, 0.35, 0.2, 0.10, 0.05, 0.02, 0.005, 0.0.
  This only changes the numerical initialization and does not weaken
  physical acceptance constraints.
- An EXACT v2.2.3 working-source cache can now be upgraded automatically.
  Restored historical source, `.venv`, PhysX binary, and results are kept.

## Limitations

This repairs the defect-continuation *logic*, not the underlying source of
physical infeasibility. A stage that cannot meet the relaxed tire/chassis
constraints will still fail, correctly and earlier. No full native PhysX
or Unity run has been performed for v2.2.4. A sub-4.9-second lap is **not**
verified. Do not deploy IPOPT outputs without native and Unity validation.
