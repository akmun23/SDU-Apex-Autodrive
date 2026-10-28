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
  Results pending; no racing improvement claimed yet.

## Remaining work

1. Score the live run: ten scored lap times, command acceptance, nominal/QP
   failures, tracking/speed errors and publication timing.
2. If it fails, localize the first actual candidate/solver/estimator failure
   from this recording rather than weakening candidate acceptance.
3. Verify repeatability and reduce the measured gap to 50 s. IPOPT objective
   values do not count as achieved racing times.
