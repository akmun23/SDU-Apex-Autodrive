# Hybrid raceline search — 2026-10-29

## Purpose and current status

Keep the command campaign's rapid large reductions. Add local minimum-time
search when fixed-duration attempts stall, and repair resampled actuator and
wheel histories using the production source equations. This does not replace
the runtime plant, vehicle parameters, acceptance tolerances or Unity tests.

The live, previously loaded competition campaign independently improved its
8.250 s witness to **7.425 s** during this work. That result belongs to the
existing fixed-duration search. Do not attribute it to the new hybrid code.
The running process keeps its loaded implementation; edited Python modules
are not hot-swapped into its current solve.

## Implemented

1. `command_ocp.py:command_problem` shares the physical bounds, winding,
   source maps, footprint checks and geometric bounds between formulations.
   The existing exact command solve retains its original objective/dynamics.
2. `solve_time_predictor` minimizes lap duration plus a small local change
   penalty. It uses a common continuous interpolation fraction between literal
   24 ms and 25 ms source maps, with local numerical trust bounds. This is an
   explicitly relaxed initializer, NOT a continuous-time PhysX implementation.
   The C ABI supplies derivatives only for fixed integer ticks; its zero
   duration tangent must never be misused as a free-time derivative.
3. Predictor checkpoints use `source-time-predictor/v1`; production witness
   loading and certification reject that schema. Improved shorter iterates
   remain available even when the relaxed NLP itself is not yet feasible.
4. Round proposals UP to whole 25 ms packets. Compare their exact source
   dynamic/periodic defects with ordinary resampling at the SAME duration.
   Prefer a better-conditioned initializer, not the fastest infeasible printout.
5. `recondition_command_seed` prescribes the proposed pose and u/v/r only
   while initializing wheels, rack, heave, attitude and jounce from source
   propagation. It settles their cyclic history and retains the better guess.
   This step has prescribed body motion; it is not a recursive prediction or
   certificate. All 21 NLP coordinates remain decisions. Final validation
   still replays the entire lap with NO resets.
6. The hybrid campaign invokes the predictor after a failed shortening,
   repairs the proposed seed with the exact 25 ms NLP and independently
   certifies it before publication. No proposal directly enters best.json or
   Unity. No-gain proposals leave the existing phase/duration search available.
7. Save/resume the predictor primal with source, seed and map fingerprints.
   The dashboard distinguishes this stage from exact command certification;
   null active context between solves is also handled.
8. The default scheduler backs off the proposed duration before revisiting
   every starting phase at the same failed aggressive duration. It tries the
   source-selected gap initializer first near a one-packet reduction.
9. Multiply the fixed-duration objective by `count/.001`. This changes no
   relative weight or mathematical minimizer. The old objective and gradients
   were unnecessarily tiny relative to the normalized constraint system.
10. `track_optimizer --command-seed` independently recertifies an immutable
    seed on the supplied map and continues the same automated search. This
    avoids rebuilding a slow geometric bootstrap merely to compare solvers.

The XY basin bound is **2 m per node**: CasADi `sum1` sums the two position
rows independently in each column. An earlier conversational claim that
this was one shared movement allowance for the whole lap was incorrect.

## Evidence collected so far

Artifacts: `live_runs/hybrid_time_pilot_20261029/`. Competition direction is
the Unity-verified direction, using the certified 7.425 s witness from
`live_runs/reverse_witness_transfer_20261029/direct_challenger/`.
Source plant fingerprint:
`8f4e47544d5e5205f31ec83a4e067f8c980c9cb1f588b127a46cfd559ae77747`.

- Unregularized time predictor: no usable time proposal after 80 iterations,
  approximately 95 solver seconds. Its feasible incumbent was preserved.
- Local predictor: did lower its time objective, but the first bounded trial
  did not produce a whole-packet feasible gain. Shorter infeasible states are
  now retained for exact-grid initialization rather than silently discarded.
- A raw 7.400 s proposal had initial exact scaled defect 0.9081 versus 0.9462
  for ordinary resampling. Its 160-iteration repair ended at defect 0.01217,
  recursive XY error 1.02 m and negative wall slack. This was rejected.
- The dominant initialization errors included rack state (about 0.33 rad)
  and all four wheel speeds. Source history initialization reduced the same
  guesses' maximum scaled defects **0.9462 -> 0.4030** (ordinary seed) and
  **0.9081 -> 0.3659** (time proposal), each in about 0.04 s and two settling
  laps. This measures seed consistency, not a new feasible raceline.
- Reconditioned-seed repair also failed at 160 iterations (residual 0.04573).
  It does not establish a convergence gain and remains optional.
- A single-gap initializer from the same certified 7.425 s seed starts at
  defect 0.06890. With the original objective scale, a 160-iteration solve
  failed (best residual 0.06890; recursive XY error 2.369 m).
- The IDENTICAL single-gap seed and constraints with the rescaled objective
  produced a **7.400 s source witness** after the same 160-iteration budget:
  residual **4.61896e-6**, no-reset whole-lap max XY disagreement
  **3.43372e-5 m**, positive extra footprint slack **0.000151798 m** beyond
  the 5 cm margin. Independent certification passed local dynamics and
  periodicity (zero periodic defect). This is a measured 25 ms source-lap
  improvement, not an achieved Unity lap. IPOPT still returned the iteration
  limit and dual residual 0.01594: do not claim KKT convergence.
- Finite-difference checks of the certified witness show source derivatives
  agree at sufficiently small perturbations, while tire/contact branches
  can change within 1e-7 to 1e-8 input changes. Example: at interval 134,
  jounce-to-yaw derivative approaches -15.4624, matching the source tangent;
  a 1e-6 perturbation yields -16.9798. Increasing iteration limits alone does
  not address this local sensitivity.

## Verification and next work

- Focused source/seed/campaign, map workflow, GUI, auto-validation and report
  tests passed; the production plant/derivative parity tests are included.
- The raw and reconditioned seed comparisons, scaled/unscaled identical-gap
  solves, and independent 7.400 s certification above have completed.
- Check proposal selection against reconditioned ordinary seeds, so a history
  fix is not mistakenly attributed to the time objective.
- Promote a hybrid result only after the unchanged independent source gates;
  use the existing automatic Unity/MPC queue for real driving evidence.
- Keep any unproven predictor optional if it does not improve validated search
  results. A finite local search or stalled predictor is not a vehicle limit.

## Running comparison and Unity trial

Updated campaign: `live_runs/optimizer_iros_conditioned_20261029/`.
It starts from the independently certified 7.400 s witness, uses the rescaled
fixed-duration cost and duration-first scheduler, and explicitly enables the
optional local time predictor as a separate challenger to the older process.
The older loaded campaign continues in its own workspace.

```bash
tools/run_raceline_campaign.sh f1tenth_planning/maps/autodrive_iros_2026_compete.yaml \
  --command-seed live_runs/hybrid_time_pilot_20261029/certified_7p400/witness.npz \
  --output live_runs/optimizer_iros_conditioned_20261029 --hybrid-time-search \
  --workers 4 --iterations 900 --stagnation-iterations 180 --phases 4
```

Dashboard: http://127.0.0.1:8767/ . Optimizer user service:
`sdu-track-optimizer-2a79aeaea99f.service`.
The independent Unity queue service is
`sdu-apex-mpc-validation-iros-conditioned.service`, watching this workspace.
Future certified exports automatically enter that serialized queue.

Actual Unity/MPC trial of 7.400 s candidate:
`live_runs/apex_mpc_f8c5b636134d8b64_a01/`.
It stopped without completing warmup, on
`rejected_nonlinear_rollout; nonlinear=corridor@24`. No racing lap-time claim.
The short, startup-inclusive recording has 180 diagnostic records, controller
cross-track RMSE 0.09162 m, heading RMSE 0.08516 rad, speed-error RMSE
1.25338 m/s. MPC publication median/p95/max is
10.351 / 16.159 / 18.514 ms, with no sampled cycle at/above its 20 ms budget.
These are short-run controller-estimate errors, not full-lap or truth parity.

This test found and fixed an automation failure: report generation used host
Python without ROS. Analysis now runs in the controller ROS image, using its
recorded immutable image ID for future runs. The existing bag was rescored
successfully without another simulation; JSON, cycle CSV and diagnostics JSONL
are present and its failure/metrics appear in the dashboard. Failed trials are
retained for manual retry, not automatically launched repeatedly.
