# v2.3.7: Max-defect-monotone checkpoints and labeled residual breakdown

User-supplied v2.3.6 projected native seed failed after 50 Phase-I iterations.
Initial max raw physical defect was 4.0555896283; selected checkpoint max defect
was 12.3116275327, despite its weighted squared-slack objective decreasing
from 397.28963922 to 260.53762002. Its dominant violation was the symbolic
lateral force balance at node 7. These values are from user-run IPOPT, not
new native PhysX validation or model proof.

**Defect acceptance correction.** v2.3.5/6 checked hard track/actuator
constraints and decreasing weighted L2 energy, but did not check that the
largest absolute physical dynamics residual also decreased. A lower L2 can
come with a much worse localized physical equation. v2.3.7 requires BOTH
weighted merit reduction and non-increase of the actual max |F(z)| (numerical
1e-10 roundoff allowance) for every candidate. The final exact-zero slack
constraints, original IPOPT vehicle model, min-time objective, free geometry,
and native validation remain unchanged. This is a checkpoint improvement,
not a claim of convergence. The Phase-I objective remains weighted L2;
the safeguard only decides which diagnostic checkpoint is saved.

**Useful model evidence.** `initial_dynamic_defect_breakdown` and
`selected_dynamic_defect_breakdown` report max, RMS, and worst original equation
name for each major dynamics family. The report also counts
`rejected_lower_merit_higher_max_defect`. The breakdown uses the original
symbolic dynamics residual vector and matching labels, not an approximate
physics comparison. Because `unity_spring_damper_load` equations are scaled
by /10 in the original OCP, their reported magnitudes are the equations'
scaled residuals (not Newtons).

The v2.3.6 projected seed was demonstrably worse in raw max error than the
v2.3.5 unprojected native trajectory (4.05559 vs 1.064776 initially). Thus
`diagnose_phase1_v237.sh` uses `--no-seed-force-projection`, the proven lower-
max initial seed, with corrected forward-speed wheel integration. This is a
50-iteration, drive-only, no-native-replay diagnostic, not a final racing run.
Once the error groups are identified the next substantive work is reconciling
the historical symbolic tire/load/collocation model with the source vehicle.

Safe upgrade: the exact previous v2.3.6 generated cache is recognized in both
Hessian modes; unknown edits remain protected. All work remains inside the
same SDU_Apex_IPOPT_PhysX_MinTime_v2 directory; historical git restoration,
existing venv, compiled native binary and prior results are preserved.
