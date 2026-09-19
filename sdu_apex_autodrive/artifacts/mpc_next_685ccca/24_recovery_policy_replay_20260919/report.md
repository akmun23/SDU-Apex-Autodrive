# Recovery-policy replay

This is the post-change replay of the clean ICRA PP event stream. It uses the
same legal runtime inputs as production and does not consume simulator truth.

## Result

- 13,030 synchronized/projected cycles were solved.
- 13,030 were accepted; 0 were rejected.
- The solver used N30 at 40 Hz (`dt=0.025 s`), adaptive rho, prefactorization,
  physical scaling, and a 100-iteration cap.
- Solve time was 2.605 ms median, 8.185 ms p95, 10.625 ms maximum.
- Residual p95 was 0.009978 primal and 0.009584 dual.
- Minimum predicted corridor clearance was 0.218062 m.
- Recovery was active on 0 cycles because this clean holdout did not contain a
  nominal seed corridor violation.

## Implemented recovery behavior

The normal corridor is unchanged. If the nominal bounded seed violates it, the
schedule can use a temporary directional envelope around the seed until the
first normal-feasible stage. The QP and exact nonlinear rollout consume that
same schedule, while steering-rate and speed-rate actuator bounds remain hard.
Heading-feedback and brake-heading-feedback seed policies are implemented and
selectable, but the runtime configuration remains `nominal` until a dedicated
violating-state comparison promotes one.

AMCL permits larger scan-supported corrections during the startup/first-lap
travel phase. After 51.7 m, each local correction is limited to 0.20 m and
0.12 rad; repeated accepted scans may still remove accumulated drift. MPC no
longer turns estimator/synchronizer uncertainty into a zero command: it
refreshes the last bounded driving command while waiting for the next legal
state. A missing trajectory remains a configuration failure and is the only
remaining controller `publish_stop()` path.

## Acceptance boundary

This is offline replay evidence, not a live MPC-authority lap. A controlled
batchmode authority run is still required to prove that Unity follows the MPC
commands for multiple laps. No Unity source, physics, scene, or sensor behavior
was changed for this work.
