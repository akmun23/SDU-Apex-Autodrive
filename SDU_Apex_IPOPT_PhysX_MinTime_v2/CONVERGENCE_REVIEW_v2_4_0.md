# Comprehensive engineering review — SDU Apex IPOPT/PhysX minimum-time optimizer

**Reviewed build:** v2.3.16 in-place update; archived reports and native-contact telemetry through v2.3.15.  
**Consolidated numerical update:** v2.4.0 (same v2.3.16 vehicle physics).  
**Status:** Research/diagnostic implementation; no fully feasible minimum-time IPOPT circuit and no verified sub-4.9 s native/Unity racing lap.

## 1. Executive assessment

The last eleven revisions largely optimized **model-parity diagnostics**, not **the mathematical feasibility of the full 143-node nonlinear optimal-control problem (OCP)**. These are different objectives. Accurate per-wheel forces given externally measured slips/loads do not imply that the OCP's inferred tire-contact states, suspension forces, collocation dynamics, periodic boundary, wheel integration, geometry and discrete brake mode can all hold simultaneously.

The solution workflow repeatedly paid for independent IPOPT solves, then **discarded the best checkpoint when strict feasibility failed**, and next time started from the original 2.4 m/s native trace. That is a primary methodological regression; debugging each sign/force detail individually created a version treadmill. In v2.3.16, the diagnostic Bash script additionally wrote into the *v2.3.15* output directory—an actual result-contamination risk.

**Recommendation:** Freeze the current physical model for one controlled, multistart/continuation campaign; test two seed strategies under identical physics, reuse hard-constraint-feasible checkpointed states between Phase-I solves, inspect solver progress by dynamic family, and only make physical edits supported by **end-to-end, time-aligned native input/output evidence**. Restore actual CAWB braking topology after a credible symbolic feasibility stage.

## 2. Evidence across versions

All defects below are **maximum absolute residuals of original, differently scaled OCP equations**, not a lap time, a true SI-unit aggregate, or proof of native dynamics parity. Runs differ in physics and starting model; comparing a smaller defect across physics releases is useful for debugging, not a controlled optimizer benchmark.

| Run | Initial max defect | Best max defect | Elastic merit at initial | Result |
| --- | ---: | ---: | ---: | --- |
| v2.3.7 (unprojected native seed) | 1.064776 | 1.062793 | 37.7249 | 50 iterations, no feasibility |
| v2.3.10 (suspension frame fixed) | 1.064776 | 1.038374 | 16.3229 | 40 iterations, no feasibility |
| v2.3.12 (source tire callback) | 1.428774 | 1.333186 | 21.6114 | 40 iterations, no feasibility |
| v2.3.14 (slip sign only) | 8.761113 | 6.511617 | 462.3738 | 40 iterations, force-frame inconsistency |
| v2.3.15 (slip+force signs) | 1.428774 | 1.333186 | 21.6114 | 40 iterations, **identical to v2.3.12** |
| v2.3.16 (full contact velocity) | **not measured yet in this audit** | **not measured** | — | validated in low-speed tire-trace comparison; complete Phase-I solve remains pending |

The last row is deliberately **not a prediction**. Do not claim that improving slip-force parity necessarily improves complete OCP feasibility.

**Concrete IPOPT behavior from v2.3.10:** 6,578 variables; 3,146 equality and 1,430 inequality constraints. IPOPT begins at zero infeasibility **of the modified elastic system**, not of the original physical dynamics. At iteration 40, IPOPT's current trial had constraint violation ~102.55 and objective ~46,863, while a separately safeguarded historical checkpoint had physical max defect ~1.038 and elastic merit ~16.24. The checkpoint thus helped reporting, but **it did not steer IPOPT's internal iterates or seed the next run**. More iterations of the same failed path were not demonstrably useful.

## 3. Physical-model review: what is verified and what is not

### Strong evidence in favor of retaining current fixes

1. **Wheel-substep state indexing** (`x[2,k]=u`, not `x[0,k]=ey`) is a clear index error repaired in v2.3.6. Actual substep count still switches at 5 m/s; high-speed equivalence has not been verified end to end.
2. **Suspension left/right frame** was sign-inverted in the original source-to-model mount conversion. Reflection reduced per-wheel load RMS mismatch from ~2.174 to ~0.992 N and corrected left/right differential sign. It did not fix the total normal-load discrepancy or improve all acceleration predictions.
3. **Native tire callback formula** exactly matches recorded forces when given measured native slip and effective suspension load (~1.34×10⁻⁷ N combined component RMS). This validates the *callback*, not its upstream slip/load reconstruction.
4. **Native lateral slip and tire-force frame** requires both sign conversions. Changing only one produces v2.3.14's 8.76 max initial defect. Changing both recovers v2.3.12 behavior; the algebra explains the exact recurrence.
5. **Roll/pitch contact-point velocities** are omitted by the previous planar model. The v2.3.16 parity estimate decreased lateral-slip RMS ~0.000547 → ~0.000059 rad and lateral tire-force RMS ~0.577 → ~0.068 N. These measurements support the correction, but the rates were interpolated from a 143-node warm start to the 481-record contact trace, not sampled synchronously at each 1 ms native tire update.

### Still unresolved (must not be declared “exact PhysX”)

- The symbolic spring/damper load model is an approximation to the 1 ms vehicle+scene update: raycast recursion, contact solver impulses, suspension constraints, wheel-contact sequencing and body/ground contacts are not fully reproduced.
- Total suspension-load mismatch remained ~3.22 N on average in the earlier controlled frame comparison. The symbolic model therefore is not yet a measured-force oracle.
- Measured post-scene body acceleration, the 25 ms sample interval, and pre-scene per-wheel slip/force telemetry are **staggered in time**. Acceleration differentiated from spatially resampled body states is not the native callback's instantaneous force balance.
- Drive-vs-brake is a discrete physical branch. Every recorded warm start uses drive mode; forcing braking nodes changes the dynamics discontinuously. A 0-brake diagnostic cannot establish a valid 3-brake-zone racing OCP.
- The physical model may be exact to its **own** discrete symbolic equations while still being inaccurate relative to native or Unity. These are separate acceptance gates.

**Do not fit an extra lateral grip coefficient to these residuals.** It could hide a mismatch between contact velocity, tire load, body orientation and event timing.

## 4. Optimization architecture review

### Critical: repeated loss of Phase-I progress — corrected in v2.4.0

The v2.3.16 solver selected a checkpoint with low physical defect and hard safety/actuator constraints, but returned only a failure report when original dynamics did not reach zero. The next run reconstructed the original native seed. v2.4.0 exports the complete selected 16×143 state, wheel loads, accelerations, steering-rate, throttle and phase lengths **only if the checkpoint passed hard constraints**. The runner validates it and can rebuild the OCP from that point for up to two further attempts. It stops if the max physical defect does not meaningfully improve. All restarts remain **infeasible diagnostics**, not certified racing outputs.

**Qualification:** This is primal-only continuation, not a dual-multiplier IPOPT warm start. It can fail or plateau. It is a meaningful change in the experimental method, not a guarantee of convergence.

### Critical: the solver minimizes L2, while checkpoints enforce L2 and L-infinity

Phase I minimizes weighted squared elastic residuals. The checkpoint rule additionally requires the largest absolute raw dynamics residual not to increase. **That checkpoint rule is outside IPOPT.** IPOPT itself can move toward lower L2 while amplifying a localized defect. Future convergence work should study residual scaling and conditioning, not confuse a saved backtrack with improvement of the active solver trajectory.

### High: initial feasibility is elastic-system feasibility only

`inf_pr=0` in the first IPOPT line means the auxiliary elastic variables exactly reproduce initial physical residuals. It says **nothing** about vehicle-dynamics feasibility. The original equality slacks are explicitly forced back to zero before minimum time and native replay; that strict gate must remain.

### High: 143-node collocation, wheel dynamics and suspension are stiff

The OCP has thousands of bounded variables and coupled wheel, heave, roll, pitch, yaw and path constraints. Direct collocation across ~0.199 m intervals approximates 1 ms PhysX contact integration with interval-averaged torques; speed-dependent 5/1 wheel substeps add a discrete branch at 5 m/s. Stiffness, non-smooth force curves and discretization error can defeat standard IPOPT searches even with correct tire-axis signs. Post-campaign work may require an explicit multiple-shooting/implicit integration reformulation, improved algebraic elimination, physical residual scaling and staged/continuation feasibility. Do not remove constraints to make the report appear feasible.

### High: speed and target are distant

The safe warm start is ~2.07–2.36 m/s. A 28.410 m lap under 4.9 s needs an average ground speed above ~5.80 m/s. Even a physically feasible 2.4 m/s OCP is **not** evidence of competitive performance. Optimization must subsequently solve with CAWB braking and pass an actual native full-lap safety check, followed by Unity parity.

### High: no controlled solver-option comparison

The earlier series primarily reran 40–50 L-BFGS iterations under changing physics. Therefore we cannot attribute IPOPT convergence changes to the solver method. v2.4.0 compares two starts with *identical v2.3.16 physics* and checkpoint continuation. An optional exact-Hessian experiment is available separately; its computational cost and value should be decided from the first two runs.

## 5. Implementation and workflow defects fixed in the consolidated update

| Finding | Severity | v2.4.0 response |
| --- | --- | --- |
| Best Phase-I point thrown away after strict failure | Critical | Export all state/algebraic variables and nonuniform mesh; resume only if hard-safe and lower max defect |
| Independent IPOPT runs repeat same initial 40-iteration state | Critical | Up to 3 solve attempts per strategy, with stop-on-stall |
| No side-by-side seed initialization on unchanged physics | High | Run **unprojected** and **symbolic-algebraic-projected** start in one campaign |
| v2.3.16 diagnostic writes into v2.3.15 output folder | High | Corrected v2.3.16 destination; all v2.4.0 runs use distinct result directories |
| Reports conflated when zipped with `zip -j` | High | Preserve directory structure and collect all restart reports separately |
| Hessian mode cache switch may reject the other known generated mode | Medium | Byte-exact known-cache upgrade for both modes, unknown edits still refused |
| Seed provenance mislabeled unprojected as projected | Low | Source label corrected |
| Solver model was being revised after every single parity hypothesis | Process | Freeze v2.3.16 physics and measure full-OCP progress in batch |

## 6. What v2.4.0 tests do and do not establish

Local clean test suite: **143 passed**. Additional checks: generated historical-style solver is syntactically valid in both limited-memory and exact-Hessian modes; cache can switch modes and return without modifying the historical source; unknown cache changes still fail closed; malformed/unsafe checkpoint states cannot restart; no-progress continuation stops; output paths are distinct.

**Not executed in this analysis environment:** restoration of the complete original external Git tree, map-specific full CasADi/IPOPT solve, the user's native executable or Unity. Those resources are not embedded in the in-place update ZIP. End-to-end solver convergence and high-speed native/Unity parity **must be measured on the user's full checkout**. Unit tests are not substituted for that evidence.

## 7. Controlled convergence campaign — the next substantive experiment

`run_convergence_campaign_v240.sh` does the following within the existing repository:

1. Fail-fast check that the generated v2.3.16 CasADi contact model improves native slip and tire-force parity on the recorded flying lap. There is no new simulator run.
2. **Unprojected** native initial state, L-BFGS Phase I, 60 iterations per solve, up to 2 additional restarts using the best hard-safe Phase-I state. Maximum defect must decrease to continue.
3. **Algebraic-projected** native initial state under **the identical physics and solver**, same continuation budgets. This directly tests whether reducing force-balance error at initialization helps the *whole OCP*, rather than presuming that algebraic projection is inherently better.
4. Output a ranked JSON and readable Markdown summary with each attempt's initial/best raw max scaled physical defect, elastic merit, status, iteration count and dominant residual family. Results remain separate.
5. Package all logs and reports into one ZIP for analysis. Any failed/unfinished attempt remains marked as such. `--brake-zones 0` and `--skip-native` remain explicit diagnostic restrictions.

Optional separate exact-Hessian branch: `V240_RUN_EXACT_HESSIAN=1 bash run_convergence_campaign_v240.sh`; expensive, not default. This experiment does not automatically infer an optimized race lap from a symbolic feasibility result.

**Stop/go criteria:**

- If v2.3.16 parity check fails, do not launch IPOPT: fix provenance/state synchronization first.
- If one variant reduces defect substantially across restarts, use that verified best state to design a proper zero-defect continuation, then introduce physical braking. Do not rework the tire signs again.
- If both starts plateau near 1.0+ scaled max defect or IPOPT ends with >10–100 hard constraint infeasibility, stop patching warm-start signs and redesign Phase I: well-scaled constrained least squares, elimination of algebraic slack/loads, block/multiple shooting, and improved treatment of discrete brake/transmission branches. This is a **formulation-level** intervention, not another 40-iteration rerun.
- Exact feasible IPOPT states must pass all original dynamics and track constraints at the solver's audited tolerance; a native 1 ms full lap and ultimately Unity are independent required acceptance tests.

## 8. Reproduction

Install the in-place overlay. It preserves the repository, `.venv`, compiled PhysX executable, recovered historical solver cache, and previous results. Then execute:

```bash
cd ~/Documents/GitHub/SDU-Apex-Autodrive
unzip -o ~/Downloads/SDU_Apex_IPOPT_PhysX_MinTime_v2_4_0_IN_PLACE_REVIEW_UPDATE.zip -d .
cd SDU_Apex_IPOPT_PhysX_MinTime_v2
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/
bash run_convergence_campaign_v240.sh
```

The campaign writes `v240_review_results_upload.zip`. **Do not treat a completed campaign as a completed racing lap.**
