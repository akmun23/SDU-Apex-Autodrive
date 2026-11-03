# Source-SQP two-pass feasibility and timing audit (2026-11-03)

## Status

The two-pass nonlinear source-SQP is **not live-qualified**. It retains the full 26-state source plant, 30 stages, 60 steering/throttle decision variables, stage-local source Jacobians, a new full-source Gauss–Newton Hessian every relinearization, and a strict **22 ms end-to-end** publication budget. The artificial steering-command trust region remains removed; the source vehicle applies its real rack slew. Continuous positive-drive throttle is restricted to **0.001–1.0** to prevent entering Unity's distinct zero-throttle braking mode. No alternate controller or fallback command is used.

The previous one-pass 22 ms practice pilot completed six laps. The first live two-pass pilot (mpc_source_two_pass22_20261103_15) failed before finishing the first lap, after 64 source-certified published commands.

## Validated computational changes

- The second SQP pass relinearizes against the exact accepted source trajectory, rather than recalculating a full-horizon plant rollout to reconstruct that same trajectory. An exact accepted-nominal snapshot is protected from rejected line-search and second-order correction probes; input control bytes must match before reuse.
- Each relinearization uses the **fresh 60x60 source Gauss–Newton Hessian**, not leftover BFGS curvature. The full physical source model, objective and independent nonlinear feasibility gate are unchanged.
- The dense interior-point QP calculates Hx and transpose(A)*lambda once per IPM iteration for both the KKT residual and convergence scaling, and reuses a preallocated Newton RHS for Mehrotra predictor and corrector. Eight-case A/B parity retained identical commands/feasibility; timing gains were modest (~0–0.4 ms in many cases; scheduling-dependent).
- A new C++ regression explicitly verifies two full source linearizations, 120 source transition intervals, 240 tangent objective evaluations and exact source certification.
- A six-way split of the 60 scoring columns was **slower** by about 2 ms and reverted. A finite secant across steering-actuator saturation worsened physical feasibility and was reverted. A single quadratic soft Phase-I QP was slower/less robust than the existing lexicographic Phase-I and was reverted.

## Tests

- CMake C++ test suite: **22/22 CTest suites passed**, including the new two-pass regression.
- Source-only causal closed loop (six separate physical P-cores, no competing load): **60/60** complete source-valid epochs, two full SQP passes, hard 22 ms solver budget, median **9.86 ms**, maximum observed **19.46 ms**. One earlier run missed its deadline while a CPU-intensive viability probe was running concurrently. Offline timings do **not** establish the live 22 ms deadline.
- Captured previous live failures 08, 10, 12 and 14 were source repairable using two SQP passes offline, mostly within ~15–22 ms; failure 09 remained near the wall after two passes.

## Live failure and missed publication sequence

Run 15 failed at practice progress **28.0186 m**, lateral error **0.4862 m**, heading error **0.4481 rad**, speed **3.8898 m/s**, yaw rate **-2.731 rad/s**. The source-seeded nominal violates the exact physical corridor from prediction stage 9 (~225 ms) and has squared violation **4.5624 m²**.

Immediately before the rejection, the live ROS diagnostics recorded **nine consecutive waiting_for_solver_budget control cycles** (rows 122–130 of /tmp/mpc_twopass_live15_diagnostics.jsonl). Since the previous accepted update, lateral error grew from about **0.132 m to 0.486 m**. The controller continued to request solutions, but could not publish before its hard 22 ms callback deadline. The subsequent failure was a nonlinear horizon rejection, not a normal deadline deferral.

Repeated publication misses are a real robustness failure, not a benign timing metric. They can destroy shifted-plan recursive feasibility even when the previous plan passed exact source certification.

## Independent evidence a feasible source plan EXISTS

The new simulator-free diagnostic, tools/mpc_source_viability_probe.cpp, evaluates the complete original source plant with physical steering and **positive drive** commands. It is a diagnostic, **not** a publisher or fallback controller.

- An initial 800-command family contained 0 fully corridor-feasible trajectories.
- A wider deterministic 18,800-command piecewise-steering family contained **28** full-horizon corridor-feasible trajectories.
- The best has **14.097 mm minimum clearance** and passes the **independent production source validator** (MPC_RTI_ROLLOUT_OK, valid=true, failure stage=-1).
- Replaying the same recorded initial state with this valid source trajectory only as an **offline warm start** allows the normal source-SQP controller to return a source-certified plan in **10.35 ms** under a strict 22 ms solver budget.
- The original live warm start cannot produce an independently source-feasible plan in that time, and after more iterations remains outside the physical corridor.

Therefore the captured source state is **not inherently infeasible**. The failure is the SQP's inability to reach its narrow source-feasible basin from the warm start in real time. A feasible offline witness must **never** be injected as an alternate online controller output.

## Steering-actuator nonsmoothness

The checked-in exact steering actuator changes physical rack angle at a maximum **183.346 deg/s**. At the failing state the rack is near -0.2466 rad. Steering targets from roughly -0.35 to -0.5236 rad produce essentially the same first-stage physical response because rack motion is saturated. A correct local derivative can therefore be zero while a finite target change across the switching surface has a large effect on future clearance.

The exact-source single-axis sweep (/tmp/mpc15_branch_scan.txt) confirms this plateau. A tested cross-switching secant Jacobian harmed physical feasibility, so it was rejected. This requires a more principled nonsmooth SQP representation or feasible initialization inside **one normal MPC**, never hard trust regions or fallback steering.

## Remaining engineering gates

1. Remove consecutive 22 ms live publication misses: reduce real two-pass source derivative, dense Phase-I IPM and source-certification tails **under actual Unity/ROS load**, preserving the complete QP and nonlinear objective.
2. Improve Phase-I nonlinear merit and steering-saturation branch handling so it reaches known source-feasible trajectories quickly, with no alternative controller and strict source-certificate output.
3. Measure Unity-to-source one-step errors and command/state epoch synchronization in the exact corner before modifying the model or controller weights. Accurate source optimization is not automatically accurate Unity prediction.
4. Validate repeated 10-lap practice races then competition with zero collisions, no fallback outputs and zero deadline misses. Preserve the established practice configuration until qualified.

The recorded Unity failure and best independently source-certified offline witness are archived at live_runs/mpc_source_two_pass22_20261103_15/analysis (witness, nonlinear steering scan, JSON run report and full source diagnostics). The reconstructed solver replay fixture is at /tmp/mpc_live_failure_replay_20261103/15. These are machine-local evidence and require explicit preservation with the run archive for permanent reproducibility.

**No successful mandatory two-pass Unity race is claimed.**
