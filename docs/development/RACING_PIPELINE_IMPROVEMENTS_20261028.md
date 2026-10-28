# Racing pipeline improvements — 2026-10-28

## Goal and starting point

Ten consecutive scored practice laps below 50 s, excluding warmup; repeated
success, not a reference integral or one fast lap. Best previously verified
ten scored laps: 52.239 s. Current warm-start gate fix is `61539ce98`; its
verification and failed weight/SQP experiments are documented in
`MPC_REPAIRABLE_WARM_START_20261028.md`. No physics or acceptance gate is
relaxed here. Existing IPOPT continuation remains running independently.

## Attribution before edits

The last slip-1 capture's 1,371 command-time predictions were compared with
interpolated later Unity truth, excluding samples after the first fault. This
is offline analysis only; truth is never an autonomy input.

| Command-time state | RMSE | Maximum absolute error |
| --- | ---: | ---: |
| XY position | .0759 m | .1485 m |
| Forward position component | .0706 m | .1483 m |
| Lateral position component | .0278 m | .0746 m |
| Yaw | .0036 rad | .0247 rad |
| Forward speed | .0508 m/s | .7503 m/s |
| Lateral speed | .0058 m/s | .0709 m/s |
| Yaw rate | .0702 rad/s | .8074 rad/s |

In the final 500 ms, lateral pose error was only .0236 m RMSE (max .0333 m),
while MPC's estimated cross-track error grew beyond .49 m. Localization is
not perfect, particularly along-track, but it does not explain away that
large tracking excursion. Tail velocity/yaw errors remain relevant and are
not declared solved.

## Concrete reference problem

The v343 tracking-reserve line's speed integral is 4.948605 s, but that is not
proof that its geometry, speed and actuator response are jointly feasible.
Its source OCP certificate does not transfer after geometric editing.

At the failed turn, recorded sampled curvature jumped between approximately
-.28, -.91, -.46, -.63, -.25, -.81 and -1.20 per metre. The active MPC's
steering-slew objective (weight 2000) is centered on changes in its
curvature-derived steering reference, not zero command change. This makes
reference jitter a direct source of demanded command changes.

Independent closed-loop geometry audit of 474 original nodes:

- Curvature-step p95 .4774 per metre.
- Curvature-derived geometric steering reference rate p95 10.11 rad/s,
  maximum 21.64 rad/s; 83 segments exceed source rack slew 3.2 rad/s.
- This rate is a reference diagnostic using `atan(.324*kappa)`, not a proof
  that the true high-speed inverse steering must equal a kinematic formula.
- The chatter is already in the original OCP-exported XY geometry. The
  tracking-reserve conversion did not create it; Menger curvature agrees
  with stored curvature to about 2.3e-7 RMSE.

## Implemented geometry correction

Extended the maintained `tools/make_tracking_clearance_target.py` with optional
periodic, arc-weighted XY regularization. It solves one sparse positive
definite system for both coordinates on the original nonuniform arc grid:
least displacement plus a second-derivative geometry penalty. It does not
fit tire forces, alter vehicle physics, change MPC weights or filter runtime
sensor data. No per-point clipping recreates curvature spikes. Original
geometry remains unchanged unless explicitly requested.

Candidate: `ipopt_v343_smooth_tracking_candidate.csv`. A .1 m regularization
length moves any point at most **.00194753 m**. Recomputed XY arc lengths,
heading, Menger curvature, acceleration and wall corridors; preserved all
474 speed samples. Steering-reference rate p95 becomes 1.19 rad/s, maximum
2.06 rad/s. Speed integral remains **4.94827949 s**.

The independent swept-footprint certificate retains the source-size body and
5 cm wall margin, with .03051 m additional certified clearance. This is not
a new vehicle-dynamics certificate. Seven unit tests cover the sparse solve
against an analytic circle eigenvalue, periodic seam, nonuniform spacing,
rigid transforms, noise reduction, zero/invalid smoothing and displacement
limits. The default no-smoothing path stays supported for reproduction.

## Live verification

`mpc_v343_smooth_geometry_20261028_01`: unchanged production MPC weights,
full candidate speed profile, eleven total laps requested. Full state,
commands, solver diagnostics and permitted-sensor odometry are recorded;
supervisor terminates on collision or controller fault. Completed eleven laps,
zero collisions: ten scored total **51.37400055 s**, mean 5.13740005 s,
population SD .05189 s, fastest 5.0548 s. Cross-track RMSE .06416 m versus
.0803 m for the prior ten-lap baseline. Twenty-eight prior-plan actions and
three candidate prefixes were used. This is a real one-run gain, not yet
repeatable: repeat `_02` failed after warmup on a corridor check.

Timing: gate median/p95 9.680/17.292 ms. One 20.423 ms cycle was deferred.
Two accepted commands passed the pre-publish gate but the DDS publish call
returned after 20 ms (maximum 21.489 ms). Do not claim strict full-cycle
maximum below 20 ms from the gate alone.

A coordinated ey/u/slew trial (8000/70/1000/20, other settings unchanged)
also failed a corridor check after one scored lap and is not promoted.
The candidate geometry remains separate from the installed production default.

## Next work

1. Score full-horizon acceptance, prefix-chain lengths, speed/path error,
   timing and laps versus the recorded baseline. Do not equate offline
   curvature reduction with a racing gain.
2. If the smooth candidate improves racing, repeat it before promotion and
   tune response against its consistent geometry instead of noisy feedforward.
3. Investigate remaining nonlinear corridor failures, including the QP's
   heading-frozen footprint constraint versus exact candidate geometry.
4. Keep only measured useful runtime changes; retain negative evidence in
   run reports rather than leaving rejected experimental code enabled.
