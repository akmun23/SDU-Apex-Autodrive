# Native source SQP: QP timing and active-path cleanup (2026-11-03)

The active opt-in source-derived SQP uses a dense primal-dual Mehrotra
interior-point QP solver, not the Riccati/ADMM backend. The old Riccati/ADMM
and R1/R2 code is bypassed when native_source_reduced_sqp is selected;
it is not yet safe to delete because it is still the default production
controller and the source solver does not meet the live deadline.

## Changes made

- The native IPM maximum was reduced from 90 to 40 iterations after
  comparing exact source results on 60 normal frozen states. All 60 had the
  same acceptance status, exact-source objective, wall violation and QP
  outcome for 40 and 90; a 25-iteration cap changed one case. No hard wall,
  actuator constraint or strict source-validation threshold was relaxed.
- The final redundant 30-stage source validation was removed *only* when
  the already strict-validated physical candidate has precisely matching
  command bytes. No unvalidated QP direction is publishable. On 60 saved
  states the certificate was reused in all 60 and the SQP result was identical.
- Trials of LLT replacing LDLT and 1e-6 (instead of 1e-7) QP termination
  changed cases or worsened timing: both were reverted. The strict QP
  stopping and independent feasibility certificate are unchanged.
- No wholesale legacy-code deletion: unused in the *selected* source path
  is not necessarily dead in the established production path.

## Measurements

60 saved normal source states (one SQP pass, 30 control decisions, 30 exact
source stages, 40 IPM cap): median 8.5 IPM iterations, P95 39; median QP
time approximately 1.0 ms, P95 12.3 ms, maximum 21.8 ms. Five QPs did not
reach the full KKT certificate under 40 iterations; their incumbents were
source-checked, and the solver cannot label incomplete QP solves optimal.

The 60-epoch uninterrupted source-only C++ closed loop after these changes
kept 60/60 independently full-source-valid horizons, identical terminal
objective (~21.0958), and approximately 5e-11 next-state agreement.
Unconstrained timing varies with the workstation's other running processes.

Strict 25 ms absolute-deadline replays of 60 saved epochs completed
54/60 on the first run, then 53/60, 53/60 and 50/60 across repeats.
The remaining deadline failures do NOT meet the real-time release gate.
Benchmarks were on a workstation with concurrent desktop/MATLAB processes.
No physical/Unity full race or hard real-time validation was performed.

## Remaining targeted work

1. Make the dense IPM latency predictable through stateful primal/dual
   warm-starts, structure-aware assembly of 300 constraint rows, allocation
   reuse and controlled CPU scheduling. Do not weaken QP certificates.
2. Reduce the dual source-plant sensitivity cost (typically ~12 ms on a
   difficult corner), while comparing the FULL objective, wall and rate
   derivatives against source finite differences in high-slip and branch
   transitions. Previously measured corner cost-gradient differences mean
   the old finite-difference oracle must remain available for validation.
3. Remove optional FD prefix/suffix and old R1/R2/recovery implementations
   only when full-source mathematical parity, deadline qualification and
   repeated safe closed-loop races justify replacement. Avoid parallel
   production implementations after that qualification.
4. Retain 30 optimized command stages and full 30-stage physical source
   prediction. The 20-stage frozen-tail mode can cause solver-inaccessible
   actuator violations at the terminal stage.
