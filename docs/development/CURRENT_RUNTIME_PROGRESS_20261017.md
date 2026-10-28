# Current runtime and rewrite status — 2026-10-17

> **Current production cleanup / faster practice default — 2026-10-28:**
> See [RUNTIME_CLEANUP_AND_RACING_20261028.md](RUNTIME_CLEANUP_AND_RACING_20261028.md)
> before resuming work. It records the active source/scan chronology, ordered
> Riccati optimization, retired experiments, and three completed ten-lap
> comparisons on the new v343 tracking-reserve practice target. Older weight
> experiments below are historical evidence, not the current runtime profile.

> **2026-10-25 SQP/damping trials:** the exact 5.367 s nominal IPOPT
> reference was tested with a selective second-SQP pass, both at the original
> steering-slew weight and with the weight halved. Neither live trial improved
> post-warm lap pace; one ended at an MPC corridor rejection. No experimental
> weight or refinement trigger was promoted. See
> [MPC_SQP_STEERING_DAMPING_TRIALS_20261025.md](MPC_SQP_STEERING_DAMPING_TRIALS_20261025.md)
> for the controlled comparisons and artifacts.

> **Live MPC update — 2026-10-24:** practice production-config run completed six
> laps with zero collisions. Lap 1 was warmup; scored laps 2–6 were
> `6.044, 6.138, 6.101, 6.121, 6.180 s` (mean `6.117 s`). The publication
> gate peaked at `15.576 ms`, with no cycles at or above 20 ms. A bounded
> one-cycle deadline recovery was added after a prior run exceeded the gate
> and collided while retaining its last actuator command. The recovery branch
> did not trigger in this new run, so its online behavior remains unobserved.
> Details, metrics, bag, and evaluator output are in
> [MPC_THROTTLE_SLEW_DEADLINE_RECOVERY_20261024.md](MPC_THROTTLE_SLEW_DEADLINE_RECOVERY_20261024.md).

> **Latest MPC runtime update — 2026-10-21:** the supervised practice run,
> exact measurements, runtime input audit, bag-decoder correction, and current
> failure-policy behavior are documented in
> [MPC_RUNTIME_FAILURE_AND_BAG_DECODE_20261021.md](MPC_RUNTIME_FAILURE_AND_BAG_DECODE_20261021.md).
> Across the three capped practice runs in that report, the latest published
> exactly validated current/prior actions in 408/409 solve cycles, including
> 16 real prior-plan reuses; maximum end-to-end control time was 15.771 ms
> (<20 ms). It completed no lap and stopped after a genuine corridor rejection
> that the remaining prior plan also could not pass. This is not high-speed or
> racing validation.

> The dated progress entries below are historical snapshots. The 2026-10-21
> report linked above is the latest runtime status and supersedes older
> statements about schema versions, pending verification, and whether the MPC
> has been run.

> **Critical vehicle-model correction — 2026-10-22:** the exact `UnityPlayer.so`
> extracted from the pinned practice image calls `setSubStepCount(5.0f, 5, 1)`,
> not `setSubStepCount(128.0f, 5, 1)`. PhysX uses five substeps when the
> absolute body velocity projected onto vehicle forward is below 5 m/s, and
> one at or above 5 m/s. The earlier 2026-10-20 interpretation was wrong. The
> production plant and IPOPT helper had inherited that error. The shared plant
> is corrected and both MPC and odometry production targets compile against
> it. The active IPOPT generator uses an isolated `_working_solver_v249_*`
> cache with the corrected helper; older v244–v248 caches and their recorded
> outputs remain historical and are not valid practice-model evidence.
>
> A corrected 0.12 m/no-brake Phase-I run (1200 iterations, 2026-10-22) reduced
> maximum raw dynamic defect from `0.46576` to `0.44604` but did not reach
> feasibility. The worst term remains roll-rate collocation (`0.44604`); the
> next largest classes are lateral-force balance (`0.42486`) and pitch-rate
> collocation (`0.33041`). This shows the threshold correction alone does not
> resolve IPOPT feasibility. Full report:
> [`optimized_practice_v251_phase1_mesh12_no_brakes_correct5_20261022/ipopt_report_pass_00.json`](../../SDU_Apex_IPOPT_PhysX_MinTime_v2/optimized_practice_v251_phase1_mesh12_no_brakes_correct5_20261022/ipopt_report_pass_00.json).

## Maintained stack

> **Live status update — 2026-10-18:** this work is focused on production
> odometry and its source plant; no controller was run. The in-process plant
> remains a source-derived approximation, not a 1:1 port of Unity's private
> WheelCollider solver. Its low-speed constraint treatment now uses six
> sweeps, supported by two independent 1 kHz captures; the corrected one-step
> yaw-rate maximum fell from `0.0556/0.0669` to
> `0.0030/0.0040 rad/s`. This also removes the corresponding one-step forward
> speed error. The plant ABI has 18 states and `UnityPlantState` schema 9.
> Production odometry and targeted plant tests compile/pass. This is capture
> replay evidence, not yet a full-lap odometry claim.


### Time-scaled source-feasible seed and next continuation — 2026-10-23

The reference CSV speed profile was not enough by itself: replacing only the
speed state created a 1.813 maximum source-dynamics defect because yaw, roll,
pitch, and their rates remained inconsistent with the faster motion. A
time-scaled full-state seed reduced the raw defect to 0.493. Projecting the
steering-rate state back inside the source-derived bound kept the defect at
0.493 without relaxing a physical constraint. These are initialization
diagnostics, not accepted minimum-time solutions.

Run `optimized_practice_v290_time_scaled_source_phase1_20261023` then used
source-equality Phase I to restore the exact configured source constraints.
It exported a source-model-feasible incumbent at **5.889811 s**, maximum
constraint violation **0.0009348295** against the unchanged **0.001** gate.
The full checkpoint and witness are in
`SDU_Apex_IPOPT_PhysX_MinTime_v2/optimized_practice_v290_time_scaled_source_phase1_20261023/ipopt_candidate_pass_00/`.
Its target CSV is
`SDU_Apex_IPOPT_PhysX_MinTime_v2/optimized_practice_v290_time_scaled_source_phase1_20261023/optimizer_target_raceline_pass_00_not_mpc_validated.csv`.
The report identifies this as the captured-motion feasibility fallback; it
is **not** a converged minimum-time optimum and is **not** validated in Unity
or by MPC. It does show that the current source-equality transcription can
produce a strict-gate-feasible sub-6-second profile. The profile integral of
the supplied reference CSV is 5.97196 s, while the previously observed
working runtime line is about 5.7 s; therefore v290 is not a performance win.

The v290 minimum-time stage did try faster iterates (5.28049 s and 5.93429 s)
but they violated source constraints by 0.06876 and 0.18964 respectively.
Neither was accepted. This points to the remaining issue more specifically:
the time objective can find faster directions, but the local NLP is moving
out of the source-feasible basin, especially in roll-rate/yaw-rate
collocation. The objective value alone is not a raceline result.

Continuation `optimized_practice_v291_small_trust_lbfgs_20261023` has
completed from v290's saved feasible checkpoint. It used time trust factors
4, 8, and 16, limited-memory Hessians, 400 iterations per time stage, and the
same model, map, mesh, and 0.001 gate. It took **4,228.9 s**. All three time
stages hit the 400-iteration cap and were rejected. Their raw lap-time
iterates were 5.6078, 5.1504, and 5.6367 s, but their maximum *non-trust*
constraint violations were 1.4887, 1.4785, and 0.6300; they are not feasible
racelines. The stage trust-region constraints were themselves badly violated
by the unfinished iterates, so the apparent faster objectives are not
progress. The accepted output remains the same v290 Phase-I feasible profile:
**5.889811 s**, maximum source-constraint violation **0.0009348295** against
the unchanged **0.001** gate. This is below 6 s but not a speed improvement
over the reported ~5.7 s working runtime line, and it is not validated in
Unity/MPC. No Unity/MPC run has been started for v290/v291.

Interpretation: this does not establish that IPOPT is the wrong solver. It
shows that adding iterations to the current full-space time objective is not
working: IPOPT finds attractive objective values while leaving the local
feasible basin, especially in roll-rate and yaw-rate collocation. Do not
promote those iterates or repeat the same 4/8/16 continuation unchanged. The
next solver experiment should first preserve the feasible checkpoint and
diagnose a *single small, feasible descent step* with scaled variables and
constraints; only then widen the trust region. Compare that against the
current L-BFGS configuration on this exact checkpoint. Any candidate must
pass the existing 0.001 full constraint gate before its lap integral is
compared. Solver-family replacement is not yet supported by this result.

### Componentwise time continuation and finer-mesh refinement — 2026-10-23

The failed aggregate trust constraint was the immediate obstacle to useful
time descent: its scalar squared-distance inequality gave IPOPT poor local
steps. `solver_overlay_v251.py` now represents that same normalized trust
radius as componentwise bounds, with each bound set to
`sqrt(trust_bound / number_of_components)`. This preserves the aggregate
L2-equivalent radius when all normalized coordinates move equally, without
changing source dynamics or physical constraints. A focused unit test checks
the generated constraints/provenance; 10/10 focused tests and `py_compile`
passed. The previously ignored new implementation/test files are now
explicitly visible to Git; generated MinTime run outputs remain ignored.

Run `optimized_practice_v299_component_box_time_descent_20261023` started
from the v290 certified checkpoint and tested factors 4, 8, and 16 (100 time
iterations each). Factors 4 and 8 produced source-feasible checkpoints at
5.814198 s (maximum violation 0.0002889) and **5.784547 s** (0.00002921).
Factor 16 reached 5.748336 s but violated constraints by 0.008950, so it was
rejected. The exported v299 checkpoint is the feasible 5.784547 s point.

Run `optimized_practice_v300_component_box_refine_20261023` restarted from
v299 and used factors 4, 8, and 12 (250 iterations each). Factor 4 improved
to **5.719294 s** with maximum violation **0.00001133**. Factor 8's 5.695585 s
iterate violated source constraints by 0.04441; factor 12 reached 5.677401 s
but remained just outside the strict gate at 0.001862. Neither replaced the
factor-4 incumbent. The target therefore exports the feasible 5.719294 s
profile, not the faster invalid iterates.

Run `optimized_practice_v301_component_box_more_iterations_20261023` used
factors 8, 12, and 16 with 400 iterations each from v300's saved checkpoint.
Factor 8 reached 5.635705 s but violated source constraints by 0.004138.
Factor 12 then produced a valid improvement at **5.618784 s**, maximum
violation **0.00074681** (below the unchanged 0.001 gate). Factor 16 reached
5.606836 s but violated constraints by 0.007356. IPOPT hit its 400-iteration
cap at each stage; the certified v301 output is the factor-12 point. The run
took 920.84 s and exported
`optimized_practice_v301_component_box_more_iterations_20261023/optimizer_target_raceline_pass_00_not_mpc_validated.csv`.

This is genuine optimizer progress: a source-model-feasible predicted lap
time fell from v290's 5.889811 s to 5.618784 s through accepted continuation
steps. It is still not sub-5 s, not an IPOPT convergence proof, not a
continuous-time feasibility certificate (node/midpoint spacing is about
0.06 m), and not validated by Unity or MPC. v301's feasible witness reports
2.94–7.19 m/s, peak lateral acceleration 6.04 m/s², and minimum wheel load
1.48 N; this indicates the optimizer is already close to wheel unloading at
one tight-corner sample, so simply raising speed bounds would not be a
physics-grounded fix.

The first finer-mesh run, `optimized_practice_v302_mesh006_refinement_20261023`,
resampled v301 to 0.06 m nodes and Phase-I did recover a strict-feasible point
(max defect 0.0007843). However, the minimum-time warm-start immediately
violated its componentwise trust box by 0.1948: the box was centered on the
original interpolated seed, not on the Phase-I-restored feasible trajectory.
The two time stages then ended at 5.62135/5.62770 s with large trust and
physical residuals and no target export. This was a trust-region centering
bug, so it does **not** show that the fine mesh is infeasible.

The overlay now centers every time-stage trust box on the certified Phase-I
snapshot (including states, controls, phase lengths, loads, and acceleration),
while preserving the same normalized radius and all original physical
constraints. The generated solver source was inspected to confirm this exact
center is used; the focused suite still passes 10/10 and `py_compile` passes.
Run `optimized_practice_v303_mesh006_feasible_center_20261023` repeated the
fine-mesh experiment with this correction. On 474 nodes, factor 4 produced
**5.583335 s** with maximum violation **0.00001436**; factor 8 improved this
to **5.567649 s** with maximum violation **0.00062556**. Both satisfy the
unchanged 0.001 gate; the latter was exported as the best checkpoint and
target. The target remains theoretical and not validated in Unity/MPC.

Continuation `optimized_practice_v304_mesh006_time_descent_20261023` completed
from the certified v303 checkpoint. The best feasible profile reached
**5.502225 s**. Continuation `optimized_practice_v305_mesh006_longer_continuation_20261023`
then reached **5.432555 s** at trust factor 16, with maximum source-constraint
violation **0.00014874**. It used the 474-node, 0.06 m mesh and sampled the
source feasibility witness at nodes and midpoints (maximum sample gap
0.02997 m). The minimum clearance diagnostic is **0.00944 m after subtracting
both the 0.05 m requested wall margin and vehicle footprint**; thus the
candidate satisfies the configured margin at the sampled locations, with
about 9.4 mm extra. This is not continuous-space certification.

Continuation `optimized_practice_v306_mesh006_push_20261023` completed from
v305's exported source-feasible optimizer checkpoint on the same map,
centerline fingerprint, vehicle model, and 0.06 m mesh. Its factor-18 stage
reached 5.366848 s but exceeded the unchanged source-feasibility gate
(maximum violation 0.002241), so it was rejected. Factor 20 yielded the best
retained point, **5.363423 s** with maximum constraint violation **0.00008263**;
factor 22 ended at 5.363746 s with 0.00016935 violation. IPOPT hit the 800
iteration cap on all three time stages, so these are feasible retained
iterates, not convergence proofs. Compared with v305, v306 improves the
predicted optimizer time by 0.069132 s. Its sampled minimum extra clearance
after subtracting footprint and the configured 0.05 m wall margin is only
0.000947 m (nodes and segment midpoints); the requested margin is satisfied
at those samples but there is virtually no extra numerical reserve. It is not
a continuous-space or MPC/Unity validation. Native feasibility replay was
explicitly skipped.

Comparing exported checkpoints shows v305→v306 altered lateral geometry by at
most 0.0105 m while the speed state changed by up to 0.084 m/s. That is a
diagnostic sign that the 18–22 trust schedule is now too local to explore a
substantially different line. A larger-radius continuation is the next
optimizer experiment; it must keep the same source dynamics, collision
footprint, 0.05 m wall margin, and 0.001 maximum-violation gate. No Unity/MPC
validation has been run for v306.

### Plateau diagnosis and alternate geometry initialization — 2026-10-23

The v307/v308 wider-trust continuations did not produce accepted racelines.
They found lower objective iterates (roughly 5.29–5.32 s), but their dynamic
residuals remained above the fixed 0.001 source-feasibility gate; neither run
completed all planned trust stages. Those iterates are not candidates and are
not used as seeds. This rules out “just widen the same local trust region” as
the next search move.

The v306 feasible profile identifies a concrete geometric bottleneck: the
lowest-speed hairpin sample is about 2.78 m/s at s≈0.95 m, and the tight right
turn reaches about 1.03 m⁻¹ curvature while the optimized path remains on its
inside for most of the corner. I tested a geometric-only outside-shoulder seed
around that corner: two periodic +e_y Gaussian offsets (0.10 m amplitude,
0.85 m sigma, centered at s=28.45 and 31.35 m), with epsi rotated by the
corresponding path-tangent change. This lowers the preview path's maximum
curvature in that corner from 1.031 to 0.843 m⁻¹. Its path length rises from
27.365 to 27.606 m and its unchanged-state lap-time estimate is 0.067 s worse,
so this is a deliberate alternate-basin initializer, not an improvement claim.
The OCP node/midpoint corridor-slack minima are +0.0634/+0.0673 m after the
configured footprint and 5 cm margin; the map's swept-footprint preview also
passes. No physics equations, wall margins, or OCP constraints were changed.

The generated seed and recipe are under
`optimized_practice_v309_outside_in_seed_20261023/geometry_seed/`. The loader
records modified fields and the run explicitly reprojects the modified
checkpoint's algebraic/periodic suspension initialization before Phase I.
Run `optimized_practice_v309_outside_in_seed_20261023` is now in progress on
the same practice map, source model, 474-node / 0.06 m mesh, and strict 0.001
feasibility gate. It uses time trust factors 8, 14, 20, and 26 (600 iterations
per stage), with native replay skipped. A lower IPOPT objective alone will not
count: only a retained point under the strict source gate and passing the
existing swept-map/footprint export check is a feasible optimizer candidate.
Phase I first reached a hard-feasible source-model checkpoint at iteration 50
(maximum defect 0.0001462, zero hard-constraint violation); its final retained
checkpoint after 600 iterations has maximum defect 0.0001413 and zero
hard-constraint violation. The first minimum-time trust stage (factor 8)
reached 5.3431998 s after 600 iterations, with maximum non-trust constraint
defect 0.00093934 (inside, but close to, the fixed 0.001 gate). It did not
converge; IPOPT stopped at its iteration cap, and this point is only a feasible
restart point. This is 0.020223 s faster than v306 at the optimizer objective,
not yet a certified/export-checked line. Trust factor 14 is running from this
restart point. Trust factor 14 then found a lower raw objective (5.3265798 s)
but violated the source gate at 0.0085344, predominantly in periodic roll-rate
collocation; it did not update the feasible checkpoint. The optimizer retained
the factor-8 feasible point and continued with factor 20. This confirms that
the apparent speed gain outside the feasible basin is not usable. The retained
factor-8 point was selected for export. Factor 20 reached a 5.3095661 s raw
objective but had 0.0030934 maximum constraint violation, again dominated by
periodic roll-rate collocation, so it is not a candidate. Factor 26 reached
5.2948212 s but had 0.0192309 maximum violation and did not update the
restart point. Final v309 therefore remains the factor-8 point: 5.3431998 s,
maximum constraint violation 0.00093934 (feasible incumbent under the fixed
0.001 gate), 0.020223 s faster than v306. IPOPT itself did not converge its
KKT conditions within 600 iterations. Its export passed the configured dense
swept-map footprint and curvature-geometry checks; the dynamic source checks
are node/midpoint sampled (max gap 0.02997 m), not continuous certification.
No MPC or Unity driveability validation was run. v306 remains the best
previously validated line; v309 is a new, slightly faster optimizer-only
candidate.

The v309 source-state witness helps explain the plateau: the smallest sampled
wheel normal load is 0.389 N at s≈17.62 m (4.824 m/s, roll≈−0.109 rad,
steering≈−0.144 rad, curvature≈−0.421 m⁻¹), so that corner is already near
wheel unloading in the optimizer model. The peak sampled |roll rate| is
0.378 rad/s at s≈11.27 m (6.086 m/s, curvature≈−0.222 m⁻¹). These are node
samples, not a continuous-time or Unity validation; they indicate that roll
load transfer and transition dynamics, rather than a hidden centerline lock,
are among the active limits of the current source-model solution.

The v309 target retains another slow right hairpin at s≈14.2 m (peak
|curvature|≈0.818 m⁻¹ and local speed≈3.17 m/s). A preview of a second
outside-in-out seed, generated by `make_geometry_seed.py` from the v309
checkpoint, reduces that corner's peak curvature to ≈0.703 m⁻¹ with +0.523 m
path length. Holding the old speed profile fixed makes its time estimate
0.125 s slower, so this is only an alternate-basin initializer whose value
must come from re-optimizing speed and dynamics. Its minimum sampled corridor
slack after vehicle half-width and 5 cm wall margin is +0.120 m. The seed is
stored at `optimized_practice_v310_second_hairpin_seed_20261023/geometry_seed/`;
it is not itself a feasible trajectory. Its Phase-I solve has now recertified
the source-derived states, reducing the initial 0.06415 defect to a retained
0.0003597 maximum defect under the 0.001 gate. Time trust factor 6 returned a
5.3624476 s feasible incumbent (max defect 0.0008905) after 800 iterations,
but did not converge and is not faster than v306/v309. Time trust factor 8
returned 5.3489120 s after 800 iterations with maximum defect 0.0002520,
inside the unchanged 0.001 gate but still 0.005712 s slower than v309. Factor
10 ended at a raw 5.3748813 s with 0.01731 maximum defect (primarily roll-rate
collocation), so it did not replace the factor-8 feasible point. v310
finished by exporting the factor-8 5.3489120 s target. Its map/footprint
export check passed, sampled at nodes and segment midpoints (0.02997 m maximum
gap); minimum clearance after vehicle footprint is 0.05303 m, leaving only
about 3 mm beyond the configured 5 cm wall margin. Continuous dynamic
feasibility is not certified, and native MPC/Unity replay was skipped. v310
is therefore a valid optimizer-only incumbent for this run but slower and
with less wall reserve than v309; it is not promoted as the best line.

The v306→v309 speed-profile comparison shows the accepted 0.0202 s gain is
small and spread across short sectors; the 14–15 m bin actually lost about
0.009 s, while the first hairpin and approach sectors remain among the largest
time contributors. This supports testing a larger alternate geometry around
the first hairpin rather than continuing to make tiny trust-region changes to
the same local basin.

Prepared a separate, unsolved v311 first-hairpin geometry initializer from the
v309 feasible checkpoint: two periodic +e_y Gaussian shoulders centered at
28.45 m and 31.35 m, amplitude 0.20 m each, sigma 1.20 m. Its geometry-only
preview reduces peak absolute curvature from the v309 candidate's 0.968 m⁻¹ to
0.818 m⁻¹, with 0.1325 m minimum sampled corridor slack after footprint and
the 5 cm margin. The longer geometry raises the fixed-speed preview estimate
to 5.536 s, so this is not a time improvement by itself; its test is whether
IPOPT can re-plan speed and dynamics from that wider-radius basin. This is now
run `optimized_practice_v311_first_hairpin_wider_seed_20261023` with the same
0.06 m mesh, source-equalities formulation, unchanged 0.001 feasibility gate,
and 800 Phase-I / 800 per time stage iterations. Factor 6 produced a feasible
5.3856596 s point after 800 iterations, with 0.0001695 max defect (below the
unchanged 0.001 gate), but it is slower than v309; IPOPT did not reach KKT
convergence. Factor 8 completed at 5.3744791 s after 800 iterations, with
0.0012518 maximum constraint defect, dominated by periodic roll-rate
collocation. It is outside the unchanged 0.001 gate and is slower than v309, so
it is not a feasible/promotable result. Factor 10 ended at 5.3644293 s with
0.0011627 maximum defect, also outside the unchanged gate; it likewise did not
converge within 800 iterations. The selected/exported v311 checkpoint is
therefore factor 6 at 5.3856596 s, maximum defect 0.0001695. Its dense map
footprint/curvature export checks passed, with 0.08246 m minimum sampled
clearance after footprint and the configured 5 cm margin. The exported path
is 28.090 m long and has peak |curvature| 0.8483 m⁻¹, versus 27.539 m and
0.9679 m⁻¹ for v309; the wider turn did not recover its extra path length in
time. It remains 0.04246 s slower than v309 and is not promoted. No Unity/MPC
validation was run. The seed and preview are under
`optimized_practice_v311_first_hairpin_wider_seed_20261023/geometry_seed/`.

Also prepared a combined two-hairpin preview (v312) from the same v309
checkpoint, applying the same 0.20 m / 1.20 m outside offsets to both
hairpins. It lowers preview peak curvature to 0.770 m⁻¹ but extends path
length to 28.714 m and gives a fixed-speed estimate of 5.660 s, with 0.1325 m
minimum sampled corridor slack. This is a more aggressive alternate basin.
Although the v311 first-hairpin-only solve did not recover its path-length
penalty, the combined seed tests whether smoothing both principal hairpins
changes the speed profile jointly. Run
`optimized_practice_v312_two_hairpins_combined_seed_20261023` is now active
with the same 0.06 m mesh, source-equalities formulation, 0.001 feasibility
gate, and 800 Phase-I / 800 per time-stage iterations. Phase-I completed with
retained maximum defect 0.000795 (inside the unchanged gate). Time-stage
factor 6 completed at 5.4350909 s after 800 iterations with maximum source
constraint defect 0.00007405, comfortably inside the unchanged 0.001 gate;
it is 0.091891 s slower than v309. It did not converge KKT conditions.
Factor 8 completed at 5.4161741 s after 800 iterations, with maximum defect
0.0006786 (feasible) but still 0.072974 s slower than v309; it also did not
converge KKT conditions. Factor 10 completed at 5.4004101 s after 800
iterations with maximum defect 0.0006020, feasible under the unchanged gate
but still 0.057210 s slower than v309. IPOPT did not converge KKT within the
cap. Factor 10 is the selected v312 checkpoint; its swept-map footprint
export passed, with path length 28.549 m, peak |curvature| 0.8859 m⁻¹, and
0.10123 m sampled clearance after footprint and the 5 cm margin. This
confirms the wider geometry is feasible but loses time to path length; v309
remains the best objective. Native replay is skipped, so v312 remains
optimizer-only.

A focused roll-transition seed (v313) was also previewed at s=11.27 m, the
v309 node with maximum sampled |roll rate|: +0.15 m lateral offset, sigma
1.20 m. Its fixed-speed path-time preview is 5.37191 s, peak |curvature| is
0.9701 m⁻¹ (slightly above v309's 0.9679), and minimum sampled corridor slack
is 0.11986 m. Since it worsens preview time without lowering peak curvature,
it is stored for traceability but is not scheduled for a full IPOPT solve.

Prepared a distinct inside-apex initializer (v314) from the v309 feasible
checkpoint, shifting three slow-turn regions inward by 0.12 m (Gaussian
sigma 1.20 m at s=28.45, 31.35, and 14.20 m). Its unchanged-speed preview is
5.20253 s, 0.14067 s quicker than v309, through a shorter 26.980 m path, but
peak |curvature| rises to 1.1469 m⁻¹ and minimum sampled corridor slack after
footprint plus the fixed 5 cm margin is only 0.00296 m. This is the strongest
time-oriented initializer found so far, but the seed has very little geometric
reserve and is not a feasible trajectory. It must pass source-model dynamics,
steering-rate, dense map-footprint export checks, and the same 0.001 defect
gate before consideration. Its files are under
`optimized_practice_v314_inside_hairpin_seed_20261024/geometry_seed/`.

The v314 inside-apex preview failed the existing conservative swept-map
footprint check by 0.00157 m between knots 12 and 13, despite positive
node-only corridor slack. It is rejected before optimization. A less
aggressive v315 version at the same three corners uses 0.10 m inward offsets;
its fixed-speed preview is 5.22782 s, path length 27.072 m, peak curvature
1.1129 m⁻¹, and 0.02249 m minimum node slack. The exact existing dense map
footprint exporter precheck passes for this geometry at the unchanged 5 cm
margin. Run `optimized_practice_v315_inside_hairpin_margin_seed_20261024` is
now active with source-equalities, 0.06 m mesh, 600 Phase-I iterations (a
reduced cap based on observed Phase-I stagnation around iteration 510), 800
iterations per time stage, and the unchanged 0.001 feasibility gate. Phase-I
completed with retained maximum defect 0.0002569 (initial defect 0.064145),
with only ~3.2e-8 numerical steering-rate bound violation. Time-stage factor
6 reached 5.2349917 s after 800 iterations, 0.108208 s faster than v309, but
maximum defect was 0.0015525 (outside the unchanged gate), dominated by
periodic roll-rate collocation. The candidate is only a near-feasible restart,
not a promotable line; factor 8 is running to see whether it can repair the
defect while retaining the time gain. It must still pass the dense
swept-footprint export check. Prepared v315 files are under
`optimized_practice_v315_inside_hairpin_margin_seed_20261024/geometry_seed/`.

The prior v315 factor-8 continuation did not finish: its process was gone after
the interruption, and the output directory contains no time-stage report or
candidate. Do not treat its 5.235 s factor-6 point as feasible; its max defect
was 0.0015525, above the unchanged 0.001 gate. Restarted the inside-apex
continuation as v316 from the persisted v315 geometry seed. This is an
optimizer-only run on the practice map, with 0.06 m mesh, source-equalities,
600 Phase-I iterations, 1200 iterations per time stage, and trust factors
6, 8, 12, and 16. Unity/MPC replay is skipped. The exact command and live
output are preserved at
`optimized_practice_v316_inside_hairpin_margin_continuation_20261024/command.txt`
and `.../run.log`; process PID is in `.../pid`. No candidate is accepted
unless the existing 0.001 maximum-defect gate and dense swept-footprint check
pass. The v309 baseline diagnostics show the current local optimum is not at a
single uniform speed/steering cap: its maximum speed is 7.646 m/s on a mild
0.097 m⁻¹ segment, while the smallest modeled wheel load is 0.389 N at
s=17.621 m (4.824 m/s, κ=-0.421 m⁻¹). The inside-apex search targets the
slow-corner path geometry while preserving the current vehicle physics and
clearance constraints. This does not prove a global time floor or a 4.8 s
limit.

v316 has since completed its factor-6 and factor-8 time stages (1200 IPOPT
iterations each). Factor 6 returned 5.2349795 s with maximum constraint defect
0.00020555; factor 8 returned 5.2272762 s with maximum defect 0.00004735.
Both are feasible under the unchanged 0.001 gate and both hit their iteration
caps without KKT convergence. Factor 8 improved the v309 objective by
0.115924 s before factor 16 found the better feasible result. Factor 12
returned 5.2150388 s but violated the
unchanged defect gate at 0.0031023, so it was rejected. Factor 16 returned
5.2048364 s with maximum constraint defect 0.00032879, inside the same gate,
and is the selected v316 incumbent. It improves the optimizer objective by
0.138363 s (2.59%) versus v309. IPOPT still stopped at its 1200-iteration cap
without KKT convergence. The final target and witness were written, and the
existing dense swept-map export check passed. The source-model witness has
27.089 m path length, peak speed 7.780 m/s, peak absolute curvature
1.113 m⁻¹, minimum wheel load 0.403 N, and minimum node/midpoint corridor
slack after footprint and the configured 5 cm wall buffer of about
2.3e-7 m. Thus it uses nearly all allowed clearance; it is not robust to
additional map/model uncertainty. This is an IPOPT/source-model target only:
native replay was skipped, and neither MPC tracking nor Unity performance was
validated. It is still 5.205 s, not sub-5 or 4.8 s. Full artifacts are under
`optimized_practice_v316_inside_hairpin_margin_continuation_20261024/`.

Evaluated one alternate corner-shape basin for a possible next start: retain
the v309 outside shoulder in the first hairpin, add v310-style outside
shoulders at s=12.4/16.4 m, then cut inside at the two right-turn apexes
(s=1.2/14.2 m). A 0.15 m apex cut violated sampled corridor by 0.030 m.
Reducing it to 0.08 m gives positive 0.040 m node slack and passes the dense
swept-map check (map-only, curvature recomputed from preview XY), but the
fixed-speed preview is 5.414 s, slower than v309's 5.343 s. It has not yet
been tested by IPOPT; because it forms a different outside-entry/inside-apex/
outside-exit basin and lowers preview curvature, it remains a candidate for a
separate solve. That solve is now running as v318 with the same source model,
mesh, feasibility gate, and 600/1200 iteration budgets; Unity/native replay is
skipped. The run's command and output are persisted under
`optimized_practice_v318_outside_in_apex_search_20261024/`. The seed and
preview are under
`optimized_practice_v317_outside_in_apex_seed_20261024/geometry_seed/`.

The Codex app crashed while displaying command history, but the detached v318
solver survived; Phase-I completed at iteration 600, reducing the seed defect
from 0.06415 to 0.0008023 and passing the unchanged 0.001 strict gate with no
callback errors. It has entered time-continuation factor 6. No duplicate solver
was launched. A witness-based spatial
time decomposition of v309 versus v316 (1 m reference-station bins, using the
same sdot integral) attributes the largest gains to several distributed
sections rather than one isolated corner: about 0.012 s at s=3–4 m, 0.012 s
at 14–15 m, and 0.010 s at 2–3 m and 15–16 m. The integrated witness totals
are 5.3453 s and 5.2070 s respectively (close to, but not identical with, the
reported OCP objectives 5.3432 s and 5.2048 s). The result supports continuing
to search for coordinated geometry/speed changes around multiple turns; it
does not identify a single corner whose adjustment can recover the remaining
0.4 s, and it is not evidence of a global optimum.

At the latest check v318 remained CPU-active in time-continuation factor 6
(about 27 minutes total elapsed, ~7.4 GB RSS); IPOPT had not yet emitted the
stage summary. This is consistent with the prior same-size v316 time stages,
which each took roughly 20–26 minutes wall time for 1200 iterations. The
detached solve remains running after the UI crash.

Update: v318 factor 6 has completed at 5.3187656 s after the 1200-iteration
cap. Its maximum non-trust constraint violation was 0.00121149, just above
the strict 0.001 gate, so it is not an accepted feasible candidate. The
continuation saved a near-feasible restart point and proceeded to trust factor
8, which is active. IPOPT did not converge at factor 6; the incumbent remains
v316 until a later stage proves feasible and faster.
The largest factor-6 violation is the periodic roll-rate collocation defect
(0.0012115); the next largest rows are also roll-rate, with yaw-rate behind
them. This is a near-gate dynamics defect, not a map-corridor rejection.
Factor 8 has now completed at 5.3024359 s after 1200 iterations, with maximum
non-trust constraint violation 0.00031066, so it is feasible under the same
0.001 gate. It is still slower than v316's 5.2048364 s and IPOPT did not
converge. It saved a feasible restart and v318 is continuing at trust factor
12; v316 remains the selected best feasible result.
Factor 12 then returned a raw 5.2887944 s objective but its maximum non-trust
violation rose to 0.0204204. It failed the 0.001 gate and did not replace the
previous feasible restart. v318 has advanced to trust factor 16 from the
retained feasible point; the v316 candidate remains the best feasible time.

Following the user's concern that this basin was unlikely to improve v316,
v318 was intentionally terminated during factor 16 before that stage produced
a result. This is a stopped search branch, not a completed factor-16 rejection.
The feasible v318 factor-8 checkpoint and all logs remain preserved. A new
v319 IPOPT search has started from v316's feasible full-state checkpoint with
the separately generated first-hairpin outside-shoulder geometry seed. That
seed passes the exact dense swept-footprint map check with the 5 cm margin;
its unchanged-speed preview is 5.2589 s (slower than v316's 5.2182 s), but it
reduces peak Menger curvature from 1.1133 to 1.0570 m⁻¹, making it a distinct
speed/geometry tradeoff to test rather than a claimed improvement. The trust
schedule is 6, 8, 9, 10, using closer continuation increments after factor 12
in v318 produced a large roll-rate defect. v319 Phase I completed successfully,
reducing max defect from 0.0861828 to 0.00004666 (strict gate 0.001, no callback
errors). Trust factor 6 is running; no time-stage result exists yet. Native
replay is skipped.

Seed-preview alignment was then audited: `make_geometry_seed.py` had been
interpolating raw CSV XY/heading/curvature, whereas IPOPT constructs a
periodic cubic-spline track and resamples it on its 0.06 m mesh. The v317
0.08 m seed preview differed from the exact OCP-mesh path by up to 0.0430 m;
reconstructed on the exact mesh it still passed the 5 cm swept-footprint map
check with 0.0255 m conservative clearance beyond that margin. This did not
invalidate v318 because Phase I reprojects the seed and the final candidate
must pass exact export checks, but raw-CSV preview scores were not exact.
`run_ipopt.py` now saves `optimizer_reference_track_mesh.csv` and records its
hash in run provenance; `make_geometry_seed.py` now requires that mesh and
checks its stations and source-centerline fingerprint before constructing a
seed. The corrected generator was exercised on v316's checkpoint. A separate
v319 first-hairpin outside-shoulder seed (+0.12 m at s=26.8/4.1 m, sigma
1.05 m) passes the dense 5 cm swept-footprint map check. Its fixed-speed
preview is 5.2589 s versus 5.2182 s for v316 on the same exact mesh, while
path length grows 0.175 m and maximum Menger curvature falls from 1.1133 to
1.0570 m⁻¹. This is only an initialization tradeoff, not an optimized or
dynamically feasible result; keep it as a later basin candidate and do not
count its preview time as progress.

v319 time-stage updates: factor 6 returned 5.1761684 s with maximum
non-trust constraint violation 0.000452456; factor 8 returned 5.1687878 s
with violation 0.000322934; factor 9 returned 5.1656925 s with violation
0.000038579. All three pass the strict 0.001 feasibility gate and were kept
as feasible restart points, but none converged: each reached the configured
1200-iteration cap. At factor 9 the remaining dual infeasibility was 0.12924,
so this is a feasible descent iterate, not a KKT-stationary minimum. IPOPT
options in this run are tol=1e-6, acceptable_tol=1e-5, limited-memory Hessian.
Factor 10 is active. The best interim v319 time is 0.039144 s (0.75%) below
v316's 5.2048364 s, but it has not yet been exported or validated in Unity/MPC.

Final v319 update: factors 6, 8, 9, and 10 all reached the 1200-iteration cap
without IPOPT convergence. Their objectives were 5.1761684, 5.1687878,
5.1656925, and 5.1624293 s respectively. Factor 10's maximum non-trust
constraint violation was 0.00002555 (under the 0.001 feasibility gate), but
its dual infeasibility rose to 0.15185 from 0.03674 at factor 6. The last
continuation step gained only 0.003263 s over factor 9; total factor-6-to-10
gain was 0.013739 s. The v319 process completed and no solver process remains.
Its target remains unvalidated by MPC or Unity, and 5.1624 s is still above
the sub-5 s objective. This run shows feasible descent with diminishing lap-
time gains, not convergence to a local minimum. The increasing dual residual
means simply raising the iteration cap is not yet justified as the next move;
the next investigation should identify the stationarity bottleneck before
spending another multi-hour solve.

The user asked to push the search farther. A single continuation is now running
from v319's selected factor-10 source-feasible checkpoint, with the same map,
centerline, 0.06 m mesh, model, and wall clearance; it uses trust factor 11 and
up to 2400 time-stage iterations. The first detached launch ended during
symbolic construction before IPOPT and produced no stage result (no OOM/core
evidence); it is preserved separately as an incomplete launch. The active
attempt is `optimized_practice_v320_factor11_continuation_20261024_attempt02/`
and is attached to a persistent command session so its exit status and output
remain observable. It skips native replay; any result remains an optimizer
candidate until separately validated in MPC/Unity.

At the first v320 factor-11 progress check, IPOPT had reached iteration 267;
its lowest raw objective so far was 5.139646 s at iteration 120, while the
latest raw iterate was 5.141160 s with scaled primal infeasibility 0.00536.
These are intermediate, not yet accepted feasible checkpoints; the stage is
still running and its final gate decides whether the improvement survives.

To prepare a distinct follow-up basin without starting a concurrent solve, I
compared exact spatial time-density contributions from the v316 and v319
saved witness states. The largest remaining slow sections are reference
stations 0–2 m (about 0.25–0.26 s/m), 13–16 m (about 0.21–0.24 s/m), and
27–28 m (about 0.20 s/m). v319 has steering-rate saturation near 10.8 m and
20.8–23.8 m; its peak steering angle is only 0.340 rad, below the 0.524 rad
angle limit. A geometry-only seed was prepared from v319 with +0.12 m lateral
offsets at s=1.4 and 14.4 m (both turns have negative reference curvature, so
positive offset opens the outside); sigma is 0.9 m. Its unchanged-speed preview
is slower (5.2794 s) and peak curvature is 1.0796 m^-1, so this is only a
different-basin probe, not an improvement claim. It clears the exact OCP node
corridor screen by at least 0.0501 m after the active footprint and 5 cm
margin; midpoint/dense and dynamic feasibility still require IPOPT. The seed
is saved under
`optimized_practice_v321_apex_outside_seed_20261024/geometry_seed/`; no v321
optimization has started while v320 is active.

Final v320 factor-11 result: IPOPT reached the 2400-iteration cap, so it did
not converge (final dual infeasibility 0.14093). The saved incumbent passes
the exact non-trust constraint gate at 0.00037544 (<0.001), and the checkpoint
was updated. Its optimizer time-density integral is 5.1332403 s, a 0.029189 s
improvement over v319 factor 10's feasible 5.1624293 s. The target is exported
as `optimized_practice_v320_factor11_continuation_20261024_attempt02/optimizer_target_raceline_pass_00_not_mpc_validated.csv`.
It has 7.947 m/s peak speed, 0.3394 rad peak steering, and effectively zero
extra corridor slack beyond the configured car footprint plus 5 cm margin.
Its dynamic source checks sample OCP nodes and midpoints with maximum gap
0.02997 m; they are not continuous proof. Native witness replay was explicitly
skipped, and neither MPC nor Unity has validated this target. Treat it as a
feasible optimizer candidate, not a proven driveable lap.

Final v322 factor-12 result: the time stage reached its 2400-iteration cap and
did not converge. Its final raw objective was 5.111513 s, but its exact maximum
non-trust constraint violation was 0.00410073, above the 0.001 gate. IPOPT
retained the prior feasible incumbent. The exported target has the same SHA256
as v320's target and the same 5.1332403 s time; factor 12 produced no usable
improvement. The factor-12 feasible checkpoint remains v320 at 5.1332403 s,
0.029189 s faster than v319 factor 10. Neither target has MPC/Unity validation.

The v321 outside-apex seed is a separate geometry initialization, not a
candidate: it offsets lateral path states by +0.12 m around s=1.4 and 14.4 m,
has a slower unchanged-speed preview (5.27936 s), and requires algebraic and
dynamics reprojection. Since v322 failed to find a better feasible point in
the v320 basin, a v323 solve has now started from this alternative seed with
`--reproject-optimizer-seed`, trust factor 11, same source model/track/0.06 m
mesh, and 2400 time-stage iterations. Phase I reached iteration 500/600, with
18 accepted corrections; the best checkpoint defect is 8.99e-4 (<0.001), close
to the gate. It is
still refining feasibility; native replay is skipped. Only the exact OCP gate
can accept an optimizer checkpoint, and
even an accepted point is not a real simulator lap until MPC/Unity validation.

The Phase-I time observed in v322 suggests an efficiency opportunity, but no
code change has been made: a future fast path must re-evaluate a
provenance-bound checkpoint against current exact source constraints before
bypassing Phase I. A stored feasibility label alone is insufficient.

v323 Phase I completed its 600-iteration budget after 607.6 s of IPOPT wall
time. Its source-dynamics defect fell from 0.0947852 at seed to 0.000271592
at the best accepted point, below the strict 0.001 gate. IPOPT reported its
iteration limit, not convergence, but the wrapper accepted that feasible
checkpoint and entered factor-11 time continuation. At the latest status
check (process PID 742160, elapsed 12:07, about 475% CPU), the time stage was
still running; no final v323 objective or feasibility result exists yet. The
run uses `--skip-native`, so it cannot establish MPC/Unity driveability.

At the next v323 check, the time-continuation process remained active at
12:55 elapsed and about 470% CPU (roughly five cores); resident memory was
7.42 GB. No new iteration text or report/checkpoint file had appeared since
the time stage began, so its iteration number and objective are currently
unobservable. This is active computation, not evidence of convergence. No
other IPOPT or simulator run was active.

The time-stage output subsequently advanced through iteration 174. Its
objective oscillated around 5.15--5.17 s and the latest displayed primal
infeasibility was 0.00933, still above the 0.001 acceptance gate. At 16:40
elapsed the process remained compute-active at about 488% CPU and 7.42 GB RSS.
This is not yet a feasible improvement over v320; continue to the final
checkpoint/status before judging the alternative seed.

The same v323 time stage has now advanced through iteration 267. The displayed
objective remains around 5.16--5.17 s and the latest primal infeasibility is
0.0114; this is still above the gate and the raw objective remains slower than
v320's feasible 5.13324 s. At 18:29 elapsed, the solver was active at about
494% CPU and 7.42 GB RSS. No candidate from this stage is currently usable.

At iteration 359, v323's raw objective was about 5.159 s while primal
infeasibility had risen to 0.074. Thus the optimizer is exploring lower-time
points without maintaining feasibility; no usable improvement has been
demonstrated at this checkpoint. The process remains active and must be judged
by its retained feasible incumbent and exact final gate, not its best raw
objective.

At iteration 452, the lowest displayed primal violation near the gate was
0.000681 at objective 5.179 s; this is feasible by the numerical gate but
slower than v320's feasible 5.13324 s. Lower objective iterates remain
infeasible (for example, about 5.159 s at 0.020 primal violation). As of this
checkpoint no iterate combines feasibility with a better objective.

The v323 time stage has reached iteration 545/2400. Its objective is still
oscillating around 5.16--5.17 s; the latest primal infeasibility is 0.00644,
better than some earlier spikes but above the strict gate. There is still no
faster feasible point. The distinct-seed run remains active for a full-basin
result rather than being treated as a success from its raw objective.

At iteration 638, the objective remained about 5.167 s and primal
infeasibility was 0.0118. The run has not shown a downward trend toward the
0.001 gate at this checkpoint; continue this one configured solve to its
endpoint, then assess the retained feasible result before authorizing another
optimizer variant.

At iteration 730, primal infeasibility decreased to 0.00360, moving closer to
the 0.001 gate. The corresponding objective was 5.17062 s, still about 0.0374
s slower than v320's feasible incumbent. This is the first recent reduction
in violation; continue to the next checkpoint to see whether it reaches the
gate without losing time.

At iteration 823, the time-stage point is about 5.16216 s with primal
infeasibility 0.00881. The earlier near-gate iteration did not persist; the
candidate is currently outside feasibility and slower than v320. Let this
bounded attempt finish to inspect the best retained feasible checkpoint.

At iteration 916, the displayed point is about 5.17696 s with 0.00837 primal
violation. Some iterates have approached the feasibility gate, but only at
objectives materially slower than v320; no better feasible successor has
emerged so far. Continue this one bounded solve, then do not repeat this seed
and factor configuration if its final retained candidate does not improve.

At iteration 1009, the current point is 5.16542 s with primal violation
0.00401. This is closer to feasibility than the recent 0.008--0.02 range, but
still four times over the 0.001 gate and about 0.032 s slower than v320's
feasible candidate. The trial remains active within its 2400-iteration cap.

At iteration 1101, primal violation rose to 0.035 at objective 5.15908 s.
This confirms the current pattern: lower objective iterates repeatedly carry
large constraint defects, and feasible-near iterates are slower than v320.
The next checkpoint will determine whether to keep this solve running or stop
it for lack of any feasible incumbent improvement.

At iteration 1191, IPOPT briefly reported primal infeasibility 0.000935,
inside the 0.001 numerical gate, at objective 5.17209 s. This is about 0.03885
s slower than v320's 5.13324 s. The alternative seed can reach feasibility,
but has not yet reached a faster feasible objective; continue the bounded run
to check whether it improves this feasible basin.

At iteration 1287, the current point returned to 0.00652 primal violation at
5.16684 s. It is outside the gate, and the only observed gate-passing point
remains slower than v320. The outside-apex initialization has not yet yielded
a usable improvement.

At iteration 1379, the point is 5.16869 s with primal violation 0.00353. The
violation has decreased relative to earlier spikes, but this objective remains
about 0.035 s slower than v320's feasible candidate. The run still has not
found the faster-feasible combination sought.

At iteration 1472, primal violation is 0.00229 at objective 5.16974 s. It is
near but still above the gate, and the objective remains slower than v320.
The current run may recover a feasible line, but has not found a faster one.

## 2026-10-24 practice-line identity and closed-loop result

The earlier run directory `mpc_ipoptfast_practice_20261024_run1` did **not**
load the faster IPOPT-generated target. Its manifest points to
`f1tenth_planning/trajectories/autodrive_practice_20260924_b/ipopt_v1_baseline_raceline.csv`
(SHA-256 `35bed562a8b0611c48cbf2df8520aff091a3f212adf0eacd5f59e6b9f527fb73`).
That run completed no lap and ended at an unpublished MPC rejection after
5.612 s; collision was recorded at 5.724 s. The run label was misleading and
must not be used as evidence about the newer optimizer target.

The verified conservative reference is
`autodrive_practice_20260924_b_mintime_raceline.csv` (SHA-256
`09b0c3e5b36b781b8709eb08a98cd8c36eb7951391f269ed761409a64938a253`). It
completed one warmup and five scored laps on rebuilt current code: scored
times 6.2137, 6.1767, 6.1327, 6.2267, and 6.1557 s (mean 6.1811 s, best
6.1327 s). No collision occurred. This is the only listed trajectory with a
complete measured multi-lap result in these comparisons.

The actual v322 target (optimizer estimate 5.133 s; SHA-256
`8a5441063f90aaab5f196c00a96381b5f6199b5bd320f0e7e7cc3bf334b55239`) and
v323 target (optimizer estimate 5.231 s; SHA-256
`9827e5f2f8f4e4dd416fde19e28ed70f37d1516571cd0f6f2c334b62a923b8ad`) were
also explicitly loaded in practice. Neither completed a lap. v322 rejected
at 2.758 s and collided at 2.795 s; its recovery-overlay repeat rejected at
2.706 s and collided at 2.795 s. v323 rejected at 2.559 s and collided at
2.646 s. These are optimizer objective estimates, not observed lap times.

The v323 bag localizes the breakdown to its high-curvature section: at
progress 25.60 m the car was near the path (cross-track 4.5 cm), but from
26.07 m onward the MPC emitted a chain of 18 candidate prefixes. Through that
chain cross-track error grew to 1.20 m and heading error to 1.03 rad; actual
speed remained roughly 0.2--1.0 m/s above the falling reference. Steering
commands alternated and repeatedly approached the ±0.524 rad limit. The
publication gate was not the cause: max callback-to-gate time was 11.51 ms,
with no 20 ms violations. This is evidence of loss of closed-loop
trackability/feasibility in this section, not evidence that the faster line
has a measured lap time or that solve time caused the failure.

The v323 source-plant replay has 75 complete 750 ms windows before the failure;
all had under 5 cm endpoint error (RMSE 1.26 cm, p95 2.27 cm, max 2.47 cm).
However, none of those windows adequately samples the high-steering regime
at the failure, so the replay result does not certify the hairpin rollout.
Current container/process inspection found no simulator, controller, or IPOPT
run active.

Next work: keep the conservative measured line as the reproducible baseline;
do not call the 5.133/5.231 s objective a usable raceline. Use the v323
high-curvature failure window to revise the optimizer/runtime compatibility
there (especially speed-transition feasibility and command/prefix behavior),
then rerun the exact hashed fast target under immediate collision supervision.
Only a completed, collision-free practice lap can supply a measured time; the
first lap remains warmup when scoring a multi-lap result.

## 2026-10-24 saved-state solver conditioning and brake-mode review

The exact v323 failure state at MPC diagnostic index 108 was replayed with the
same saved state and nominal control sequence while changing only ADMM penalty
initialization. Every case reached its 200-iteration cap, failed nonlinear
validation at stage 5, and returned no first action. Best primal/dual residuals
were still `0.2542 / 0.3245` at uniform `rho=14`; the configured `rho=7` case
was `0.2815 / 0.3732`. Uniform `rho=2` was worse (`0.3787 / 0.7774`). A
mixed `rho=4, rho_steer=4, rho_throttle=7, rho_u=4` reached `0.2847 / 0.3345`.
This narrow sweep does not show a converged or publishable solution. Its
single-row host timings varied from 7.7 to 23.3 ms and are not a runtime
benchmark; penalty changes alone did not resolve the bad candidate.

The current MPC intentionally bounds every drive-mode decision to
`q >= 0.01`. In the source-derived plant, `q=0.01` applies about `1.07 N m` of
positive motor torque per wheel, whereas exactly `q=0` selects the separate
Unity brake branch and applies `428 N m` brake torque per wheel. The jump is
therefore a real actuator-mode discontinuity, not a smooth throttle endpoint.
The plant already models that branch; this question concerns whether the
optimizer may choose it. The existing forward-sensitivity code seeds a
throttle derivative only for `q>0`, so merely changing the QP lower bound to
zero would leave the optimizer without a valid local derivative for the
braking-mode switch.

Assessment: a discrete, correctly delayed brake mode could expand the MPC's
feasible set if positive drive-only commands cannot meet measured deceleration
demands. It would not add plant states, and evaluating a fixed mode adds little
plant cost; searching over per-stage brake/drive choices is the expensive
hybrid optimization problem. A brake mode can also lock the wheels and reduce
steering authority, so it must not be treated as a proportional, gentler
negative-throttle input. The observed v323 failure was an iteration-limited,
steering-reversing candidate with worsening cross-track error; the saved-state
replays do not establish that lack of braking caused that failure. No brake
permission or production bound changed in this review. Any follow-up should
compare a small number of explicit mode schedules on saved states first, with
exact delayed plant rollout and wheel-slip/corridor checks, before a supervised
practice run.

### 2026-10-24 update — finer positive throttle and high-curvature rejection

The production MPC drive lower bound is now `q=0.001` (not `0.01`). The
bridge's three-decimal serialization preserves that value. Unity treats any
positive q as drive torque; exactly q=0 is its discontinuous full-brake mode.
The MPC still excludes q=0.

At v323 practice diagnostic 108, the iteration-limited candidate flipped
steering from about `-0.322` to `+0.112 rad` and used `q=0.416`. On a same-state,
same-duration exact source-plant comparison, it passed the short corridor gate
but scored `24374.69`, versus `10510.14` for the retained certified-plan
prefix. The competition config now enables the existing quality-aware
rejection path and compares both 75 ms prefixes; when both are exact-valid it
publishes the lower-cost one. This event selects the retained action. This is
a targeted rejection-policy correction, not proof that the high-curvature
rollout matches Unity or that the solver now converges there.

Verification: local production MPC and replay targets build; C++ command-envelope
and source-rollout tests pass; the actuator-interface suite passes (6 tests);
Python replay/audit tests pass (25 tests). Re-solving at either branched
endpoint still failed at the 100-iteration limit, so high-curvature convergence
remains open. See
[`MPC_THROTTLE_SLEW_DEADLINE_RECOVERY_20261024.md`](MPC_THROTTLE_SLEW_DEADLINE_RECOVERY_20261024.md)
for the detailed evidence. A short supervised practice run with the same fast
target is the next verification step; no simulator was launched yet.

### 2026-10-24 — minimum positive throttle and high-steering rollout diagnosis

- MPC's minimum positive command is `q=0.001`; Unity's exact-zero full-brake
  branch remains unavailable as an MPC decision. The bridge's three-decimal
  serialization preserves `.001`; any smaller value that serializes to zero
  would select the discontinuous brake mode. This permits near-zero positive
  motor torque, not active braking.
- Fixed the offline continuation tool's float-boundary bug that rejected a
  CSV value of exactly `.001`. Focused verification: 26 Python tests pass.
- Existing practice run `mpc_direct_input_practice_20261024_slew100_recovery2`
  completed six laps, zero collisions, scored-lap mean `6.254 s`; its recorded
  MPC command minimum was `.010`, so it is not live evidence for `.001`.
- Current source-plant replay of recorded practice commands has 1,519 complete
  750 ms windows; 1,516 stay within 5 cm throughout. Max path-error RMSE/p95/
  max are `1.27/2.34/10.72 cm`. The three outliers cluster in a high-steering
  transition near `2.5 m/s`.
- At the largest outlier, `q=0.111` is positive and next steering feedback
  matches the replay target within `0.0001 rad`, but predicted yaw rate is
  `-1.9398` versus Unity `-2.1124 rad/s`; predicted and observed yaw
  accelerations are `+1.7` and `-4.4 rad/s^2`. The next-steering-feedback
  offline oracle leaves the same three windows over 5 cm. This localizes the
  unresolved mismatch to yaw/tire-contact response; it does not support
  restoring full braking or changing global packet-delay assumptions.
- Further offline attribution on the same capture: substituting next throttle
  feedback leaves the same three failures (worst `10.79 cm`); re-anchoring yaw
  rate or `v` from GT every 25 ms removes all three, while re-anchoring `u`
  alone leaves two. This identifies coupled lateral/yaw propagation as the
  source of those path outliers, without uniquely assigning the error to one
  state (the oracle uses future truth and is diagnostic only). Yaw-rate RMSE
  is `0.0047 rad/s` below `0.10 rad` steering and `0.0252 rad/s` at/above it;
  the high-steering bin has p95 `0.0624`, max `0.1730 rad/s`.
- The separate rejected MPC candidate used positive `q=0.416` and reversed
  steering. Its exact-valid 75 ms prefix cost was `24374.69`, versus
  `10510.14` for the retained plan. The quality-aware selection path keeps the
  lower-cost exact-valid action for that saved state. This addresses that bad
  publication case; it does not make the three rollout outliers disappear.
- Detailed evidence and interpretation: [MPC throttle and steering/yaw
  diagnosis](MPC_THROTTLE_SLEW_DEADLINE_RECOVERY_20261024.md#2026-10-24-follow-up--minimum-drive-command-and-steeringyaw-residual).

Next: use the existing practice capture to trace the high-steering yaw-force
residual through per-wheel slip, load, and yaw moment; preserve the exact
source steering branch and keep zero-throttle braking outside the MPC.

### 2026-10-25 — steering-feedback chronology correction and minimum-curvature seed

- The user approved the visual shape of the new practice-track
  minimum-curvature seed. The seed remains a geometric initialization for
  IPOPT, not a validated speed profile or a proven driveable trajectory. The
  optimizer invocation does not fix a world-space start pose; `s=0` is only a
  cyclic seam/index.
- A source-order review of `VehicleController.FixedUpdate()` confirms that
  `currentS` is copied from `SteeringAngle` before `Steer()` changes the rack
  and wheel-collider angles. The runtime had been applying this endpoint
  feedback before simulating the 25 ms interval it closes, effectively
  backdating a future rack state through already elapsed physics.
- Corrected the production chronology: propagate all 25 1 ms vehicle steps
  from the causal rack state, then reconstruct the next boundary rack state
  by applying one exact source `Steer()` step to the endpoint feedback using
  the command associated with the interval. `OdometryObserver` and the
  published vehicle-state message now preserve that reconstructed rack state;
  raw feedback remains separately published. No tire parameters or global
  command-delay assumptions changed.
- Added a direct old-vs-corrected chronology A/B to the existing bag evaluator.
  It uses future feedback only to reproduce the old backdated-reset behavior;
  this is explicitly diagnostic and not a production input. On five clean
  practice captures (7,585 aligned intervals), aggregate one-step RMSE was:

  | Regime | Causal rack yaw RMSE | Backdated-reset yaw RMSE | Causal lateral-velocity RMSE | Backdated-reset lateral-velocity RMSE |
  | --- | ---: | ---: | ---: | ---: |
  | All intervals (n=7,585) | 0.0156 rad/s | 0.0418 rad/s | 0.00226 m/s | 0.00563 m/s |
  | Steering changes ≥0.05 rad (n=660) | 0.0336 rad/s | 0.1340 rad/s | 0.00490 m/s | 0.0179 m/s |
  | Steering ≥0.35 rad (n=143) | 0.0574 rad/s | 0.0765 rad/s | 0.00852 m/s | 0.0112 m/s |
  | Both conditions (n=56) | 0.0479 rad/s | 0.0895 rad/s | 0.00718 m/s | 0.0130 m/s |

  This isolates a timing/state-association defect and shows repeatable gains,
  especially on steering transitions. It does not prove the remaining yaw
  residual is solved; the largest causal one-step yaw outlier in the five
  captures remains 0.282 rad/s.
- On practice capture `mpc_direct_input_practice_20261024_2`, the no-future-
  feedback 750 ms source-plant rollout scored 1,603 windows: every endpoint
  and every intermediate 25 ms position error stayed below 5 cm; maximum path
  error was 4.18 cm. This is a windowed offline rollout score from recorded
  starts/commands, not full-lap recursive odometry accuracy or online
  validation.
- Verification: localization production target builds in the isolated
  `build_task/mpc_screen_*` workspace; 8 source-derived runtime tests and 15
  odometry observer tests pass; evaluator steering tests pass (4), and Python
  syntax validation passes. No simulator was launched for this change.
- IPOPT is still running from the approved minimum-curvature seed with
  `--skip-native`. At the last check it had run 28m35s, used one CPU core and
  about 9.4 GiB RSS; its output directory contained only the mesh and
  provenance files, with no solver iteration/result telemetry yet. It is
  demonstrably active but progress toward a solution is not yet observable.

Next: let the seed optimization finish while monitoring its process/output;
then inspect IPOPT's feasibility and final geometry. After that, do one
supervised practice-simulator run with the rebuilt production odometry to
verify the chronology fix online before treating it as production-validated.

### 2026-10-25 — IPOPT visibility and minimum-curvature length tradeoff

- The approved seed-based IPOPT process (PID 30716) is still inside its first
  Phase-I solve: elapsed 36m12s at the check, one CPU thread near 100%, and
  10.8 GiB RSS. Its output directory still contains only the reference mesh
  and provenance; no iteration report, feasible checkpoint, candidate, or lap
  time has been written. The process is active, but solver progress is not
  observable, so no claim of convergence or improvement is justified yet.
- The seed's geometric objective minimizes squared curvature only; it has no
  path-length penalty. Recomputed from the checked-in practice trajectories,
  its path is 34.367 m versus 28.287 m for the existing v300 path (+6.080 m,
  +21.5%). Against the generator's reconstructed road-midpoint geometry it is
  +3.164 m. The seed lowers RMS curvature from about 0.394 to 0.321 1/m and
  peak absolute curvature from about 1.067 to 0.666 1/m versus v300. These are
  geometry-only comparisons; the seed's carried speed is initialization, not
  a physically feasible time estimate, and the changes do not establish a
  lap-time gain.
- This confirms a material length/curvature tradeoff that the pure
  minimum-curvature seed does not balance. The active time optimizer has not
  yet reported a candidate to tell whether its free-geometry minimum-time
  stage will shorten that route while retaining useful corner speed.

### 2026-10-25 — length-penalized minimum-curvature candidates

- Extended `min_curvature_seed.py` with `--length-penalty-per-m2`, minimizing
  discrete `integral(kappa^2 ds) + lambda * path_length`. Lambda has units
  `1/m^2`; zero preserves the existing pure-curvature mode. The existing seed
  file and the active IPOPT run were not overwritten or restarted.
- A lambda-zero reproduction exactly matched the existing seed's measured
  length (34.3669 m), curvature energy (3.53050 /m), RMS curvature
  (0.320785 /m), peak (0.66308 /m), and 19 solver iterations. Six nonzero
  candidates (`lambda=0.10, 0.15, 0.20, 0.25, 0.30, 0.60 /m^2`)
  also solved in 16–21 geometric IPOPT iterations and passed the dense mapped
  vehicle-footprint check with the requested 5 cm wall margin. Their CSV/JSON
  pairs are in `SDU_Apex_IPOPT_PhysX_MinTime_v2/min_curvature_tradeoff_20261025/`.
- Versus the existing v300 path (28.287 m, RMS curvature about 0.3944 /m,
  peak about 1.067 /m), the lambda `0.15` candidate is 30.808 m with RMS
  `0.3554 /m` and peak `0.8674 /m`; lambda `0.20` is 29.886 m with RMS
  `0.3685 /m` and peak `1.0251 /m`. These are geometry-only Pareto points,
  not predicted lap times; their copied speeds remain IPOPT initial guesses.
- The tight bend around reconstructed road-midpoint station 14–20 m
  (curvature peak near x=-2.08, y=5.49 m) makes the trade visible. The
  midpoint segment is 6.00 m, RMS/peak curvature `0.549/0.981 /m`; pure
  minimum curvature takes about 7.94 m there at `0.425/0.597 /m`. Lambda
  `0.15` shortens that segment to about 6.06 m while retaining lower local
  curvature (`0.507/0.869 /m`). Lambda `0.20` shortens it to about 5.52 m,
  but local curvature (`0.545/1.028 /m`) is essentially back at/slightly
  above the midpoint. Higher penalties increasingly create sharp local peaks.
- `py_compile`, report/schema checks, map-footprint checks, lambda-zero
  parity, and `git diff --check` passed. No candidate was sent to the running
  full vehicle OCP; that process is still active and has not written solver
  progress or a candidate. Next is to use the `0.15` and `0.20` geometries as
  alternative IPOPT seeds only after the current run yields a result or is
  otherwise concluded; compare feasible objective/time and optimized path,
  not the seed's inherited speed profile.

### 2026-10-25 — reference-anchored and straight-section seed

- Visual review identified two desired changes: keep the upper turn closer to
  the current v300 path, and remove small steering-producing curvature from
  the long right-side straight. Extended the seed generator with optional
  periodic v300-path attraction and a localized straight-section target and
  curvature penalty. The default objective remains unchanged when these
  options are omitted.
- Generated `practice_v300_anchored_straight_h32_k500.csv` using the v300 path
  as the reference, length weight `0.40 /m^2`, reference attraction
  `5 /m^4`, 32 Fourier harmonics, and a straight target on map-y `[-2.8, 2.5]`
  at map-x `0.7538 m`. This is a geometry seed; the reference is aligned by
  its nearest point to the source seam and then sampled by normalized lap
  progress.
- Result length is `27.979 m` versus `28.287 m` for v300 (-0.308 m, about
  1.1%). It passes the dense vehicle-footprint map check with the requested
  5 cm extra margin; minimum additional perimeter clearance after that
  margin is `9.77 cm`. In the selected straight interior, curvature RMS/max
  is `0.020/0.043 1/m` versus `0.039/0.096 1/m` for v300. Median/p90 nearest
  distance to v300 is `1.5/4.9 cm` on that straight and `1.6/3.1 cm` on the
  highlighted upper turn.
- The candidate has larger curvature peaks elsewhere (global peak
  `1.133 1/m` versus `1.067 1/m` on v300; lower hairpin peak is also higher).
  Therefore it is not yet an accepted dynamic raceline or evidence of a lap
  time gain. The updated overlay is
  `min_curvature_tradeoff_20261025/raceline_refined_sections_overlay.png`.
- Verification: generator `py_compile` and an end-to-end CasADi/IPOPT solve
  succeeded; dense map/footprint validation passed. No simulator was run and
  the v300 production trajectory was not modified. The full vehicle IPOPT
  process PID 30716 remains active (55m34s at the latest check, ~15.3 GiB RSS)
  and still has only its mesh/provenance files, with no solver-result
  telemetry.

### 2026-10-25 — U-turn entry/exit shaping and straight-to-turn blend

- Added smoothly tapered, map-coordinate local objective regions to
  `min_curvature_seed.py`: the upper U-turn entry can follow a separate
  reference path while paying extra curvature energy; the exit can be pulled
  independently toward v300. Added a least-squares Fourier initialization
  from the requested reference geometry. The straightness preference now has
  a configurable blend length instead of a fixed 0.4 m linear cutoff.
- Generated the separate candidate
  `min_curvature_tradeoff_20261025/practice_v300_orange_uturn_smooth_join_h32_v3.csv`.
  On the entry/apex side (`x < -0.35 m`, `y > 2.65 m`) it is within a nearest
  orange-seed distance of 2.75 cm median / 5.28 cm p90 / 8.64 cm maximum. On
  the exit-to-straight portion (`x >= -0.35 m`, `2.65 < y < 4.2 m`) it is
  within 4.66 cm median / 6.48 cm p90 / 6.69 cm maximum of v300. This
  operationalizes the requested orange entry/apex and original-line exit.
- The upper-right straight/turn transition was checked using the curvature
  gradient along the right branch over map-y 2.2–3.6 m. The candidate's
  absolute `d(kappa)/ds` p90/maximum is `0.733/0.894 1/m^2`, versus
  `1.381/1.542 1/m^2` for the orange seed. In the straight interior
  (`x > 0.4 m`, `-2.5 < y < 1.5 m`), candidate curvature RMS/max remains
  `0.0206/0.0426 1/m` (near-straight, not mathematically zero).
- The candidate path is 28.026 m and passed the dense map-footprint check
  with a requested extra 5 cm wall margin; minimum perimeter clearance
  beyond that margin is 10.07 cm. Geometry-only seed: copied speed is not a
  feasible speed plan, no dynamic simulator or lap-time test was run, and this
  is not yet an IPOPT full-vehicle result. The v300 source trajectory and the
  active full IPOPT process were left untouched. Updated overlay:
  `min_curvature_tradeoff_20261025/raceline_orange_uturn_smooth_join_v3_overlay.png`.
- Verification: end-to-end CasADi/IPOPT seed solve succeeded (18 iterations),
  exact dense footprint/map check passed, Python compile and `git diff --check`
  passed. Earlier local-weight-only candidates were not selected because
  they did not meet the entry-shape objective as well as the separate orange
  reference plus longer straight blend.
- The unrelated full-vehicle IPOPT PID 30716 remains active (1h06m elapsed,
  ~17.6 GB RSS at the latest check) but still has only its initial mesh and
  provenance files; it has not yielded a feasible optimized raceline. This
  geometry work did not restart or interfere with that process.

### 2026-10-25 — correction: target is the middle connector, not a hairpin

- Clarified the requested section from the driver's map description: it is
  the relatively straight connector immediately before a hard turn, between
  the two hard turns. It is **not** the spawn straight and is **not** the
  U-turn geometry. The accepted outer-radius U-turn candidate remains the
  geometry reference for this local edit.
- The current candidate
  `min_curvature_tradeoff_20261025/practice_v300_middle_connector_curvature_smooth_h32.csv`
  applies only a smoothly tapered curvature preference on that connector
  (map-x approximately `[-2.9, -1.0] m`, map-y `[-3.4, 2.7] m`), without
  pulling the path toward a fixed straight-line target. The measured normal
  offset from the accepted U-turn candidate is at most `1.66 cm`; the hard
  turns are therefore effectively preserved. In the broad connector ROI,
  curvature RMS/p90/maximum changes from `0.202/0.308/0.336` to
  `0.194/0.274/0.299 1/m`. The path passes the mapped vehicle-footprint
  clearance check with the requested 5 cm extra margin; the minimum remaining
  clearance beyond that margin is `9.84 cm`.
- Preview: `min_curvature_tradeoff_20261025/middle_connector_curvature_smooth.png`.
  This remains a geometry-only seed; no dynamic vehicle validation or lap-time
  claim is made. The production/reference trajectory and active full-vehicle
  IPOPT process were not changed.

### 2026-10-25 — reject displaced hairpin; refine connector from v300

- Review of the full-map overlay showed the earlier outer-radius seed changed
  both hairpins substantially; its upper-turn geometry differed from v300 by
  as much as about `1.01 m`. It does not represent the requested line and is
  explicitly rejected as a candidate. Do not use it as an IPOPT seed.
- Rebuilt the connector-only seed directly from the original
  `autodrive_practice_20260924_b_v300_optimized_mpc_amcl.csv`, with v300 as the
  reference path, a strong global reference tether, no fixed straight-line
  attraction, and a smooth local curvature preference on the middle connector
  (`x=[-2.9,-1.0] m`, `y=[-3.4,2.7] m`). The generated candidate is
  `min_curvature_tradeoff_20261025/practice_v300_original_uturn_connector_soft_h32.csv`.
- The new candidate stays within `1.98 cm` of v300 overall; nearest-path
  deviation in the two hard-turn regions is at most `0.78 cm`. In the selected
  connector ROI, absolute-curvature median/p90/maximum changes from
  `0.211/0.317/0.362` to `0.215/0.310/0.331 1/m`: peak improves about `9%`,
  p90 about `2%`, while median is essentially unchanged. Total length changes
  from `28.2865 m` to `28.2821 m` (only `4 mm` shorter). It passes the mapped
  vehicle-footprint check with the requested 5 cm margin, leaving `19.49 cm`
  additional minimum clearance after that margin.
- A stronger connector curvature multiplier (`30`, versus `8` above) is
  rejected: although the map check passed, it introduced a local connector
  peak of `0.498 1/m` versus `0.362 1/m` on v300 and its global RMS curvature
  rose to `0.3960 1/m`. The resulting CSV/JSON are retained only as a marked
  diagnostic, not a candidate to use.
- The corrected overlay is
  `min_curvature_tradeoff_20261025/original_uturn_connector_refinement.png`.
  This is still geometry-only; the small connector gain does not establish a
  faster or dynamically feasible lap. The v300 production path remains
  untouched, and full-vehicle IPOPT PID 30716 was not restarted or changed.
- A follow-up location check showed the first local mask stopped short of the
  turn approach. That earlier `...connector_soft_h32.csv` is superseded. The
  current diagnostic seed is
  `min_curvature_tradeoff_20261025/practice_v300_connector_turn_approach_push12_h32.csv`:
  it extends the smoothly tapered curvature preference into the connector's
  approach to the lower hairpin, while retaining v300 as the path reference.
  The comparison region called **middle-connector / lower-hairpin approach**
  is approximately `x=[-3.1,-0.8] m`, `y=[-4.4,-2.6] m`.
- In that approach ROI, absolute-curvature median/p90/maximum is
  `0.217/0.259/0.329 1/m` for v300 and `0.146/0.255/0.280 1/m` for the new
  candidate: median decreases about `33%`, p90 about `1.5%`, and peak about
  `15%`. The full-connector peak decreases about `3%`; total length changes
  by only `-1.1 cm`, so do not infer a meaningful lap-time gain. The maximum
  nearest-path displacement is `2.53 cm`; displacement in the hard-turn ROIs
  is at most `1.5 cm`. The dense map/vehicle-footprint check passes with 5 cm
  added margin and `19.48 cm` additional minimum clearance beyond it.
- The current visual is
  `min_curvature_tradeoff_20261025/v300_connector_tight_section_refinement.png`.
  The stronger-weight `...curvature_push_h32.csv` is rejected because it
  created a `0.498 1/m` local peak; the 64-harmonic trial did not improve on
  the 32-harmonic candidate. The outer-radius U-turn candidate remains
  rejected. No dynamic simulation was run, copied speeds remain initialization
  only, and the production v300 trajectory remains unchanged.

### 2026-10-25 — footprint overlay and outer-wall room in the marked rectangle

- The newly marked work area is the map-coordinate rectangle
  `x=[-3,-1] m`, `y=[-4,4] m`. A footprint visualization is available at
  `min_curvature_tradeoff_20261025/v300_connector_vehicle_footprint_outer_wall_60.png`.
  It draws the actual body outline (`0.510 m × 0.273 m`) and a dashed outline
  expanded by `5 cm`, at representative v300 and candidate poses. The plotting
  helper is `tools/plot_raceline_footprint.py`.
- Using the map raster, v300 headings, the same densely sampled body perimeter
  as the seed generator, and a required additional `5 cm` wall margin, the
  safe positive/outer normal-offset range in this rectangle is `0.25–0.70 m`
  (10th percentile `0.275 m`, median `0.375 m`, 90th percentile `0.500 m`).
  This is geometric map clearance, not a claim that all of that space is
  dynamically desirable.
- Generated `practice_v300_middle_connector_outer_wall_60_h32.csv` directly
  from v300, with a smooth map-derived attraction toward the outside-wall
  safe-offset target in the marked rectangle. Within the rectangle, outward
  left-normal displacement is `0.172 m` median, `0.217 m` p90, and `0.225 m`
  maximum; the path remains tethered to v300 elsewhere. The exact dense
  footprint check passes with the requested 5 cm margin; minimum clearance
  remaining beyond that margin is `19.73 cm` globally and `21.06 cm` within
  the marked rectangle.
- The path has somewhat higher local curvature at the transition: ROI
  `|kappa|` p90/max changes from `0.350/0.414` to `0.507/0.617 1/m`, with
  lower curvature through other parts of the same rectangle. This is a
  geometry candidate, not yet selected or dynamically validated. Its intended
  use is a visual/IPOPT seed experiment only; the production/reference v300
  trajectory was not changed.
- The lower attraction trial (`0.10`) moved the geometry by less than `1 cm`
  and was not retained as the useful preview. No simulator was launched. The
  active full-vehicle IPOPT PID 30716 was left running and untouched.
- Follow-up user direction: keep the outward push on the entry half, but make
  the exit half follow v300. The path traverses this rectangle from lower to
  higher map-y, so the smooth outer-wall objective is now restricted to
  `y=[-4,0] m`; `y=[0,4] m` is held near v300 by the reference tether. The
  resulting preview is
  `min_curvature_tradeoff_20261025/v300_connector_entry_outer_exit_original_footprint.png`,
  with geometry in
  `min_curvature_tradeoff_20261025/practice_v300_connector_entry_outer_wall_exit_v300_h32.csv`.
- Nearest-path comparison against v300 confirms the requested split: in the
  entry half the outward shift is `10.8 cm` median, `25.1 cm` p90, and `26.3 cm`
  maximum. In the exit half the nearest-path distance is only `1.7 mm` median,
  `9.3 mm` p90, and `3.0 cm` maximum. Entry curvature is raised (absolute
  curvature p90/max `0.332/0.360` to `0.505/0.552 1/m`); exit curvature is
  nearly unchanged (`0.378/0.414` to `0.380/0.440 1/m`).
- Exact dense vehicle-perimeter/map checks pass with the 5 cm extra margin:
  minimum remaining clearance beyond that margin is `19.77 cm` globally and
  `20.63 cm` in the marked rectangle. The earlier symmetric whole-rectangle
  attraction preview is superseded for this visual decision. This remains a
  geometry-only candidate; it has not been fed to IPOPT or dynamically tested,
  and production v300 remains unchanged.
- Tested a stronger entry push (`250` wall-attraction weight, `5 cm` setback)
  after correcting the generator's acceptance gate to apply the global-RMS
  improvement requirement only to a *pure* minimum-curvature objective. This
  stronger geometry is map-safe, moves the entry outward by `16.9 cm` median
  and `36.4 cm` maximum, and keeps the exit within `2.5 cm` p90 of v300. However,
  its global curvature RMS is `1.22%` higher than the road-midpoint baseline
  and it perturbs the adjacent lower turn by up to `13 cm`; keep it as a
  diagnostic, not the preferred preview. The `100`-weight candidate above is
  preferred because it preserves the exit more tightly and has less adjacent
  turn spill. No dynamic simulation was started.
- Follow-up correction: the outward move was starting too early. The v300
  connector's strongest curvature transition is near map `y=-1.9 m` (within
  `x=[-3,-1] m`, `y=[-4,0] m`), so generated a new candidate with the wall
  objective limited to `y=[-2.4,-1.4] m`; separate v300 reference tethers cover
  the approach `y=[-4,-2.4] m` and departure `y=[-1.4,0] m`. This centers the
  move on the curvature-change section rather than the early entry.
- In nearest-geometry comparison to v300, the new candidate moves outward in
  the target middle by `13.8 cm` median (`15.6 cm` p90, `15.7 cm` max). Before
  that region, the median shift is effectively zero and p90 is `5.1 cm` (max
  `9.2 cm`); after it, median is effectively zero and p90 is `4.5 cm` (max
  `7.0 cm`). Outside the marked connector rectangle the 90th-percentile / max
  geometric deviation is `1.3 / 1.8 cm`. Dense footprint/map validation passes
  with the extra 5 cm margin, leaving at least `12.9 cm` additional clearance.
- Preview:
  `min_curvature_tradeoff_20261025/v300_connector_mid_transition_tethered_footprint.png`;
  candidate geometry:
  `min_curvature_tradeoff_20261025/practice_v300_connector_mid_transition_tethered_outer_h32.csv`.
  The footprint plot now includes a representative car pose at map `y=-1.8 m`
  so the intended transition is visible. This is still a geometry/IPOPT-seed
  candidate, not a new production raceline; no simulator was launched and
  v300 remains unchanged. The local move raises global curvature RMS by about
  `2.6%`, so its time/vehicle-dynamics value remains unproven.
- The user then requested a blend of the two shown previews: the whole-connector
  wall-biased line and the entry-only wall-biased line. Generated a literal
  equal `50/50` blend of their signed lateral offsets after projection onto
  v300 arc progress (periodic cubic interpolation; not a controller/tracking
  blend). This supersedes the previous middle-only preview for the current
  visual choice.
- The blended CSV is
  `min_curvature_tradeoff_20261025/practice_v300_connector_equal_blend_whole_and_entry.csv`,
  with full-rectangle and connector close-ups at
  `min_curvature_tradeoff_20261025/v300_connector_equal_blend_full_roi.png` and
  `min_curvature_tradeoff_20261025/v300_connector_equal_blend_entry_roi.png`.
  The exact dense footprint/map check passes with the extra 5 cm margin and
  `19.73 cm` additional clearance remaining. Relative to v300, outward offset
  is `8.1 cm` median (`12.8 cm` p90) in the middle transition, `20.0 cm` median
  through the earlier entry, and `8.8 cm` median on the later half; outside the
  marked rectangle p90 deviation is `1.0 cm` (max `7.8 cm`). The path length is
  `28.267 m`; curvature RMS is `0.393 1/m` and peak is `0.914 1/m`.
- This is an equal visual/geometric compromise, not yet the final approved
  path: its outward shift is larger in the earlier entry than at the middle
  transition. The new plot includes a body pose at `y=-1.8 m` to show the
  transition. No simulator was launched; speed values are initialization only
  and the production v300 trajectory remains unchanged.
- 2026-10-25 user feedback: repeated automated edits are not resolving the
  intended geometry. Stop proposing another guessed path and provide direct
  manual control. Added `tools/raceline_sketcher.html` and its usage note
  `tools/RACELINE_SKETCHER.md`: browser-based map/path editor with draggable,
  smooth local shaping points, zoom/pan/undo, candidate-vs-source display,
  conservative 5 cm buffered-body map check, and export to a separate CSV.
  The source line is only a visual reference; it does not tether or constrain
  the candidate. Export recalculates geometry and map ray distances but keeps
  speed/acceleration as copied references, so the output is not represented as
  optimized or dynamically validated. A headless-browser smoke test loaded the
  practice PGM/YAML and v300 CSV, added and undid a shaping point, then exported
  a candidate: 1,024 rows, finite values, strictly increasing arc length, and
  positive map widths. Independent baseline footprint sampling found 0 blocked
  samples out of 54,720. No production/reference trajectory was changed; no
  driving simulation was started for this editor task.
- 2026-10-25 user supplied a hand-edited sketch clarifying the intended
  connector geometry. Preserved the exact file (SHA-256
  `d27919acbba5c1cd2fe0708a1e5b5fb814be882f8e4b81fade61e0c69c9ded4c`) as
  `min_curvature_tradeoff_20261025/user_intended_raceline_sketch_20261025.csv`
  and generated `user_sketch_candidate_review.png` over the practice map. This
  is a broad smooth S-shaped repositioning, not a single local apex move: in
  the connector rectangle `x=[-3,-1]`, `y=[-4,4]`, the sketch shifts mostly
  toward decreasing x through `y=[-2.4,0]` (median x shift about `-0.21 m`),
  then transitions toward increasing x by `y=[2.4,4]` (median about `+0.12 m`).
  The sketch is about `0.268 m` shorter than v300 by its exported arc-length
  column. A dense map-only test at 1 cm along the path and 2 cm over the body
  envelope found 0 blocked/unknown samples among 1,875,965; the map itself is
  5 cm resolution, so this is only a discrete geometric clearance check. No
  optimizer or driving simulation was run, and no production/reference line
  was modified. Treat this user sketch as the intended geometry reference for
  further work; do not replace it with another guessed local-wall objective.
- Compared the user sketch with the immediately preceding
  `practice_v300_connector_equal_blend_whole_and_entry.csv`. The user sketch's
  closed geometric length is `28.019 m` versus `28.266 m` (about `0.247 m`
  shorter). Within the connector rectangle, curvature RMS/p90 drops from
  `0.276/0.480` to `0.228/0.395 1/m`, but p99/max rises from
  `0.547/0.559` to `0.594/0.613 1/m`—a sharper tail at the transition. Across
  the full lap, curvature RMS rises `3.5%` and peak rises from `0.914` to
  `1.039 1/m`. Both candidates pass the same dense map-only body-envelope
  sample with zero blocked/unknown points. Three-line visual comparison is
  `user_sketch_vs_previous_equal_blend_20261025.png`. Neither candidate has a
  re-optimized speed profile or a dynamic lap-time result, so geometric length
  is not being reported as a lap-time gain.

### 2026-10-25 — user sketch sent to IPOPT; v300 rerun after yaw-path rebuild

- Preserved the user's hand-edited raceline as
  `SDU_Apex_IPOPT_PhysX_MinTime_v2/min_curvature_tradeoff_20261025/user_intended_raceline_sketch_20261025.csv`.
  It is passed directly as IPOPT's `--centerline`; no extra smoothing or
  geometry edit was applied. The solve uses a 6 cm spatial mesh (467 nodes),
  the synthetic seed backend, source-feasibility Phase-I, and up to 1,200
  minimum-time iterations. It has entered CasADi/IPOPT but has not yet emitted
  a candidate/report or completed. At the latest check it had run 18 minutes,
  using one full CPU core and about 7.2 GB RSS. Only the input mesh and
  provenance are present in the output directory, so there is no convergence
  or feasibility result to claim yet. The input path is verified in
  `provenance.json`.
- Fixed a launch-order bug in `sdu_apex_autodrive/launch/controller.launch.py`:
  the delayed-MPC `OnProcessStart` event handler was appended after AMCL, so
  it could miss AMCL's start event and leave MPC unlaunched. The handler is now
  registered before AMCL enters the launch action list. A clean dev rebuild
  completed and logs confirmed `Starting MPC after 12.00s AMCL warm-up`.
  `docker/development_entrypoint.sh` also builds `f1tenth_localization` with
  the runtime packages, ensuring the tested image contains the rebuilt yaw
  observer code.
- Ran the prior production v300 trajectory
  `f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_v300_optimized_mpc_amcl.csv`
  on the practice simulator with the rebuilt runtime. The MPC loaded its 1,024
  points (28.29 m). It rejected the full horizon, briefly selected a retained
  plan, then reported a nonlinear stage-2 rejection and latched an MPC fault
  because no sufficiently long validated action sequence remained. The sim
  completed zero laps; the captured collision counter rose from 5 at bag start
  to 65 before shutdown. The recorded actuator feedback was constant at
  throttle `0.182` and steering `-0.4864`; command and MPC-diagnostics topics
  had zero messages in this short bag. The simulator was stopped after the
  collision increase; this is a failed controller/run-start test, not evidence
  that the raceline itself is invalid or that yaw accuracy improved.
- Bag retained for diagnosis:
  `live_runs/mpc_v300_yawfix_v2_clean_5lap_20261025/run` (20.68 s, 10.1 MiB).
  A separate earlier capture from the missed-MPC-launch attempt is also
  retained at
  `live_runs/mpc_v300_post_yaw_fix_practice_5lap_20261025/run`; it is static/
  failed-start data and is not a driving result. Both simulator and dev-stack
  containers are stopped; the IPOPT process remains active.
- The second bag evaluator reports production `u/v` RMSE of `1.88/1.98 m/s`
  and integrated endpoint error of `70.14 m`, with GT speed median/p95/max
  `1.43/2.50/2.61 m/s`. Its reported zero yaw-rate error is only gyro
  pass-through, not evidence of a successful predicted yaw response. These
  metrics are contaminated by the failed run and are not a valid odometry or
  yaw-fix acceptance score.
- The short v300 bag cannot validate the yaw change: it contains no MPC drive
  commands/diagnostics, no completed lap, and rapidly increasing collisions.
  Before another driving attempt, MPC startup must be gated on a usable
  localization state and the stage-2 rejected-plan/fault sequence must be
  addressed or captured with a first-collision watchdog. Do not infer lap time
  or trackability from this run.
- Follow-up retry used the same v300 line and rebuilt yaw-fix runtime, but
  started bagging before MPC and verified practice spawn pose `(0.8005, 3.1583)`
  with both lap and collision counters at zero. It recorded
  `live_runs/mpc_v300_yawfix_retry3_20261025/run`. MPC published 189 throttle
  commands and 188 diagnostics over 4.70 s; status counts were 112
  `accepted_optimal`, 62 `accepted_prior_plan`, 11
  `accepted_candidate_prefix`, 1 `accepted_feasible`, 1
  `rejected_nonlinear_rollout`, and 1 `rejected`. Unity reached 7.524 m/s but
  did not complete a lap. The live first-collision watcher saw the first
  collision and aborted; Unity was stopped immediately, though the count
  reached 12 during shutdown/contact settling. MPC logs show it later found
  neither a candidate nor retained-plan prefix passing exact plant/corridor
  validation and latched its fault.
- Correction to the timing interpretation above: `/mpc/diagnostics` message
  spacing (`24.87/30.27 ms` mean/p95) is the interval between control-cycle
  reports, not MPC execution time. The same bag's measured solver timing was
  RTI `3.155/5.827/7.245 ms` and node solve `4.102/6.515/8.562 ms`
  (p50/p95/max); the publication gate was `4.322/6.822/8.801 ms`, with no
  deadline misses. Thus this retry did **not** show a solver-timing regression;
  it was faster than the 20 ms gate. The preceding successful 2026-10-24 run
  had gate p50/p95/p99/max `4.376/8.333/10.530/19.256 ms`, also with no
  deadline misses. Its different raceline and incomplete run metadata prevent
  an exact apples-to-apples binary/configuration comparison.
- The retry's real failure was trajectory/control feasibility, not compute
  time: after startup the MPC issued commands and reached 7.524 m/s, but later
  neither the current candidate nor retained-plan prefix passed the exact
  plant/corridor validation; the run collided before completing a lap. The
  evaluator's whole-failed-run `u/v` RMSE (`1.39/0.49 m/s`) is collision-
  contaminated, and zero yaw-rate RMSE is gyro pass-through, not yaw-model
  validation. The earlier startup issue was a launch event-handler ordering
  race; registration was moved before adding AMCL to the launch graph and the
  retry confirms MPC did start. Do not conflate that fixed launch issue with
  the later corridor/rollout failure. The simulator and dev runtime are
  stopped; the IPOPT solve on the user sketch remains separate and was active
  at the latest process check, without a candidate or report.

### 2026-10-25 — clearance-aware rejected-prefix choice and IPOPT status

- The retry3 bag contains three consecutive points where both 75 ms prefixes
  passed exact source-plant/corridor validation, but the selector retained the
  lower-objective prior plan while its minimum remaining corridor slack fell
  from `4.996 cm` to `3.249 cm` to `1.810 cm`. The exact-valid candidate
  improved slack by `0.977`, `1.052`, and `1.283 cm`, respectively, at an
  objective cost increase of about `131–137`.
- Added a narrowly gated selection rule to the production MPC: when both
  prefixes pass exact validation, prefer the candidate despite objective cost
  only if retained-plan slack is below `5 cm` and candidate slack improves by
  more than `5 mm`. Objective selection is unchanged outside that reserve.
  The thresholds and whether the override fired are included in diagnostics.
  The exact validity gate remains mandatory; this rule cannot select an
  invalid prefix. Focused selector tests cover objective ordering, reserve
  override, insufficient gain, validity, and non-finite objectives.
- Rebuilt `mpc_controller_node` in the mounted ROS runtime; runtime topic
  policy passed. The focused C++ selector test passed, as did 27 Python tests
  for the trajectory-audit and prefix-branch tools.
- One supervised, full-reference-speed practice attempt used the updated
  runtime and v300 trajectory. It produced 98 `accepted_optimal`, 37
  `accepted_prior_plan`, and 11 `accepted_candidate_prefix` cycles, then
  stopped on an exact nonlinear corridor rejection at stage 2 (`progress_m`
  `32.318`). Both 75 ms recovery prefixes were invalid at that terminal state,
  so the new clearance override was correctly not eligible. The bag spans
  3.93 s, completed zero laps, recorded zero collisions, and reached
  `7.633 m/s` maximum GT speed. This does not show that the selector improves
  lap completion; it isolates a separate case where neither short prefix is
  currently feasible. The watcher stopped the simulator and all runtime
  containers are now stopped.
- IPOPT for the user's sketch is still active. At the latest check it had
  consumed about `43 min` at one full CPU core and `13.6 GB` RSS. Its output
  directory still contains only the initial provenance and track-mesh files,
  timestamped at `02:08` local; there is no iteration checkpoint, optimized
  candidate, or completion report. CPU use confirms active computation, but
  available artifacts do not establish that IPOPT is progressing toward
  convergence or permit an ETA. It has not been stopped.

### 2026-10-25 — candidate-prefix warm-start correction and continued IPOPT

- The prior `retry3` IPOPT attempt completed naturally. Its three Phase-I
  attempts reduced maximum dynamics defect from `5.3341` to `0.9951`, `0.4662`,
  and finally `0.2386719`; the checkpoint is still infeasible and exported no
  candidate. A guarded resume path was added and tested against map, sketch,
  mesh, vehicle-model, optimizer-source, and formulation provenance. The new
  `resume01` process is active from that checkpoint, with 160 Phase-I iterations
  and three restarts. At iteration 60 it has not yet accepted a step or improved
  the checkpoint merit/defect (`3.5535059` / `0.2386719`); it has not finished
  and has not produced a raceline. Do not describe the checkpoint as feasible.
- The high-speed MPC capture's exact-plant branch replay shows 32/52 tested
  next-state solves accepted a full horizon (11 direct nonlinear accepts and
  21 exact-validated line-search accepts); 20/52 rejected after all three
  backtracks. Rejections concentrate in the curvature reversal: from progress
  `33.344 m` onward, heading error and negative yaw rate grow while path
  curvature is positive, and both branches eventually fail corridor at stages
  `0–7`. The accepted 75 ms prefixes themselves remain exact-valid early in the
  chain; the failure is that their following solve starts from a poor nominal.
- A source-plant continuation search at the earlier state `33.577 m` found a
  strictly valid 30-step witness with `0.309 m` minimum corridor slack and
  `5.156 m/s` endpoint speed. A cold MPC solve without line search failed at
  stage 10; the same state with three exact nonlinear backtracks accepted at
  `alpha=0.125`, `0.101 m` minimum slack, and `5.74 ms`. Seeding the source
  witness was accepted unchanged (`alpha=0`, `0.314 m` slack, `5.53 ms`). This
  shows an admissible continuation exists at that earlier state; it does not
  certify closed-loop lap performance.
- Targeted screens on the 52 saved branch states varied steering-slew weight
  `100–2000` and steering-feedforward weight `3–48`. Each kept the same
  `32/52` exact-acceptance count and did not materially advance the positive
  steering reversal. No weight change was promoted.
- Fixed a concrete candidate-prefix seed defect in the MPC. After publishing a
  validated 75 ms candidate prefix, the next nominal previously copied the
  rejected full-horizon QP tail. The seed now contains only the exact-validated
  prefix, followed by the retained certified plan at the correct future action
  index; if there is no retained suffix, the seed length is limited to the
  validated prefix so the nominal builder holds its last action. Both
  quality-aware and candidate-only prefix paths use the same helper. This
  changes only the solver's nominal initialization: exact rollout/corridor
  acceptance and published controls remain unchanged.
- Verification: all 16 focused C++ prefix-selection/seed-composition tests
  passed; the ROS Humble Release build compiled `mpc_controller_node` and test
  targets successfully. The new implementation has not yet been deployed in a
  Unity run, so no lap or online improvement is claimed. The next verification
  is a supervised practice run on the v300 trajectory with full capture and a
  first-collision/controller-fault watcher.
- Build isolation note: the reusable dev image's default entrypoint launches
  the controller stack. Subsequent build-only invocations must use
  `--entrypoint /bin/bash`; the accidental build container was stopped before
  a Unity simulator was connected. The active IPOPT process was not stopped.
