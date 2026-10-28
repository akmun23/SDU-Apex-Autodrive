# MPC repairable warm-start correction — 2026-10-28

## Goal and scope

Ten scored practice laps below 50 s total, excluding the first warmup lap.
IPOPT runs separately; runtime work must not stop or restart its service.
This correction does not change physics, weights, reference speed, hard state
bounds, corridor clearance or the 20 ms publication budget.

## Evidence and implementation

`mpc_v343_slip_cost_one_20261028_01` stopped in nominal construction with
`command_envelope / speed_below_zero / index 27` (675 ms ahead). The measured
initial forward speed was 5.24309015 m/s, not reverse motion. Rejecting the
future linearization seed prevented the QP from repairing its controls.

`f1tenth_mpc/src/mpc_rti.c` now checks the immutable initial condition against
the full command envelope before propagating a nominal. Future nominal
points may violate candidate bounds provided the source plant, tracking
transform and subsequent QP finite-value checks can evaluate them. No speed
is clipped and no surrogate state replaces the physically propagated seed.

Future QP state bounds and exact nonlinear candidate/prefix/retained-plan
validation remain unchanged. A warm-start trajectory is not a publication
certificate. Invalid measured speed/queues/steering still reject immediately.

## Verification

- Regression in `test_vehicle_model_source_rollout.cpp` starts from the
  recorded final body, wheel and suspension state. A deliberately sustained
  tight-turn control seed predicts negative future forward speed. It now
  reaches QP construction; QP speed bounds remain present; exact validation
  rejects that same uncorrected command sequence. Invalid measured forward
  speeds still fail at index zero.
- Complete local MPC CTest suite: **11/11 passed** after rebuilding.
- Supervised production-path test started:
  `live_runs/mpc_v343_repairable_nominal_20261028_01`, eleven total laps
  requested, unchanged production weights and tracking-reserve v343 target.
  The development container rebuilds mounted sources. Full diagnostics and
  commands are recorded; supervisor stops on collision/controller fault.
  Result: one warmup lap (5.8898 s), then corridor failure at stage 2. One
  collision event was recorded 85 ms after the unpublished controller fault;
  the supervisor did request termination, but this is not collision-free.
  Published cycle maximum 19.460464 ms. No nominal-build gate failure.
- The existing 600-state replay has identical status counts and control
  outcomes after the nominal-gate correction; it did not exercise the failed
  tail, so this is a non-regression result, not proof of improved racing.
- `mpc_v343_repairable_nominal_slip1_20261028_01`: prior replay-supported
  slip weight 1, otherwise production settings. Warmup 6.0147 s, five scored
  laps 5.2358 / 5.1698 / 5.2068 / 5.2698 / 5.3248 s, then corridor fault.
  Collision count eventually reached one. One 20.320495 ms cycle was deferred;
  do not call the whole observed cycle distribution <20 ms. Nominal rejection
  is no longer the terminal failure; the track boundary is still a problem.
- Fresh sequence replay: 1,373 full-state inputs from that capture. Slip-1
  baseline accepted 1,344 full horizons; halving steering damping and reducing
  throttle damping accepted 1,341; adding stronger path/speed weights accepted
  1,337. Neither weight bundle was promoted.
- The same fresh capture's recorded-command source-plant replay (not execution
  of a newly planned MPC sequence) scored 750 ms XY RMSE .02244 m, p95 .03273 m,
  maximum .17270 m; 49/1,351 windows exceed 5 cm. Recursive forward speed RMSE
  .02771 m/s, yaw-rate RMSE .03103 rad/s, yaw RMSE .00827 rad. Tail accuracy is
  not universally within 5 cm and must not be described as such.

## Budgeted nonlinear refinement experiment — removed, not promoted

An experimental solver permitted up to three total SQP passes, using one repeated
refinement path rather than another special candidate branch. Each pass is
at the same command epoch, is checked against the same exact objective and
physical limits, and can only replace an existing valid full-horizon candidate
when it improves that objective. Stop when no improvement is found or the
existing deadline reserve does not cover preparation/solve/validation. Normal
production cap remained two; only the separate tracking candidate selected three.

On the 1,373-state sequence replay, the rewritten two-pass path exactly
reproduces every status, ADMM status, iteration count, first steering/throttle,
nonlinear objective, line-search alpha and minimum slack from the original
two-pass path. Three passes accepted 1,343 horizons versus 1,344: no acceptance
gain. On 1,332 commonly accepted states median objective ratio was .92428;
807 improved by >1%, 444 worsened by >1% because the sequential solver memory
also changes. Mean ratio was 1.05559, so do not describe it as a uniform
objective improvement. Unbounded replay median cycle cost rose 10.827 ->
14.780 ms. This motivates an online experiment, not automatic promotion.

`mpc_v343_repairable_nominal_sqp3_20261028_01` tested that candidate online:
warmup 5.9728 s, one scored lap 5.1948 s, then corridor failure at stage 4.
Observed maximum cycle was 19.10086 ms; collision count eventually reached one.
This does not establish a racing benefit. Removed the three-pass implementation,
its parameter extension and the candidate overlay after testing. Production
keeps its original two-pass formulation, original weights and the independently
verified nominal-gate correction. Negative replay/live evidence remains with
the run. Rebuilt after removal to prevent a stale experimental evaluator.

## Remaining work

1. The nominal-gate fix is implemented, tested and locally committed as
   `61539ce98`. It does not by itself fix nonlinear corridor failures.
2. Investigate the discrepancy between the QP's heading-frozen footprint
   bounds (`set_state_bounds_with_corridor`) and exact candidate footprint /
   progress-dependent corridor checks. The QP currently bounds e_y using
   nominal heading, while nonlinear acceptance recomputes the full rectangle
   at candidate heading. This is a formulation discrepancy to investigate,
   not yet a demonstrated attribution for every failed run.
3. Preserve strict candidate validation while improving tracking. No tested
   weight/SQP bundle here established repeatable ten-lap racing improvement.
4. Verify repeatability and reduce the measured gap to 50 s. Best previously
   verified ten scored laps remain 52.239 s. IPOPT objective values do not count
   as achieved racing times. No simulator/controller is intentionally left
   running after these supervised tests; the IPOPT service remains active.
