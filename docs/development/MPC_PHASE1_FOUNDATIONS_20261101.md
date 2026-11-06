# MPC Phase 1 foundations audit, 2026-11-01

Status: correctness repairs and specification implemented; Phase 1 is **not
closed**. Physical application-epoch acknowledgment and complete-packet loss detection
remain prerequisites for reliable convergence qualification.
See [the mathematical specification](../MPC_MATHEMATICAL_SPECIFICATION.md).

## Complete control-cycle trace

| Boundary | Data and epoch | Authority / remaining limitation |
| --- | --- | --- |
| Unity response → Bridge | Legal encoders, IMU, LiDAR and actuator feedback; FIFO-associated outgoing host request | No echoed Unity sample time or request sequence in stock player. Incoming steering feedback is radians; outgoing steering command is normalized. |
| Bridge → ROS | One request stamp on packet sensor messages, exact rounded sent command and new stamped actuator-feedback pair | Feedback is published before numeric sensor decoding. Different DDS topics may still dispatch in any order. |
| Sensor packet assembly | Matching left/right encoders and IMU | Earlier partial packets now block newer completed packets; silent interval bypass removed. Overflow fails closed. A completely absent packet remains a transport-observability gap. |
| Actuator join → odometry | Exact command and exact feedback at packet stamp | Headerless receipt-time command fallback and newest-feedback substitution removed. Six arrival permutations tested. |
| Source odometry prediction | One 25 ms interval under preceding request; endpoint rack correction | Hidden source state is estimated. Headerless endpoint throttle never replaces completed-interval torque. |
| Observer correction | Encoders/gyro/IMU and source prediction at same packet epoch | Wheel, lateral velocity, rack and hidden suspension estimates need independent accuracy/uncertainty qualification. |
| `/odom`, `UnityPlantState` | Same packet stamp; body twist at rear axle; explicit correction mask | Plant schema 11 is not Unity truth. |
| EKF / AMCL | Local odometry retains source stamps; scan correction requires an odometry bracket; current map pose propagates map/odom transform at odom stamp | `publish_pose` has an unchecked interpolation return and startup prior initialized at `now()`. Ordinary scan caller already checks a bracket under `pf_mutex_`; startup prior is not an independently measured state. This must be distinguished in future qualification. |
| MPC synchronization | Map anchor transformed through bracketed odometry; latest pose/twist share latest odom epoch | Nearest-epoch substitution removed. A future map anchor waits for odom. Pending generation is retried instead of silently consumed. |
| Hidden-state handoff | Historical plant tracking coordinates reconstructed at plant stamp | Newer `e_y/e_psi/s` no longer attached to older wheel/rack/suspension state. Refuse changed map-anchor generation during that reconstruction. |
| Control-time prediction | Confirmed Bridge request history; sensor fusion at sensor epoch; separate copy propagated to predicted command epoch | Former publication-time events replaced. Sub-ms integration remainder and host phase extrapolation remain approximations. |
| Nominal / references | One frozen initial physical state; source held-input transitions; live path geometry | Same-epoch R2 no longer shifts progress/control seeds. Cross-cycle controls, solver warm state and retained-plan cursors now advance by elapsed command epochs, with cold-start/expiry for unusable epochs. |
| QP / ADMM | Source Jacobians and affine offsets at that nominal; full coupled cost and increment transform; same initial state | Progress split variables rebase; channel penalty/dual shift repaired in preceding work. Restoration Hessian indefiniteness reproduced and fixed. |
| Nonlinear acceptance | Full source rollout at identical initial state, frozen objective, actual future geometry and footprint | A numerically optimal QP does not certify finite-step nonlinear accuracy. |
| Mathematical memory | Selected source-verified nominal and controls | Publication miss now clears unissued iterate/duals, even when a certified prior sequence exists. |
| MPC publication → actuator | Physical steering/throttle `DriveCommand`; strict callback publication gate | ROS publication stamp remains a send time, not guaranteed application time. |
| Actuator → Bridge | New atomic validated `/actuators/drive_command` pair; normalized/rounded under one request lock | Removed two independent Float32 inputs to Unity request sender, which could mix command generations. Float32 outputs remain for hardware/recorders. |

The 5 ms controller timer is a readiness poll: odometry generation gates actual
solves. It does not execute five 25 ms stages per source packet. However,
generation count alone does not solve duplicate/skipped **command** epochs; the new explicit command-epoch policy handles these independently.

## Optimization/output-path disposition

| Path | Disposition and replacement obligation |
| --- | --- |
| Normal source-derived QP / two-pass SQP | Primary optimizer. R2 retains best verified cost at same epoch. |
| Source-primary optimizer | Removed from runtime in preceding work. Offline oracle retained. |
| QP-plus-independent-source polish | Removed from runtime; optional source secant curvature lives inside R2. |
| Alpha search / verified incumbent | Necessary nonlinear globalization until better finite-step directions are validated; one main objective/validator. |
| Feasibility-first / restoration retry | Reuses primary Jacobians and exact physical gate. Still a separate subproblem policy; needs elastic constraints/filter formulation before consolidation is complete. |
| Terminal continuation tie-break | Skipped when integrated terminal mode is enabled; retained while that mode remains unqualified. |
| Source witness reacquisition / viability repair | Additional full-horizon source-verified candidate generators. Not demonstrated redundant; keep pending integrated restoration/terminal evidence. |
| Quality-aware prior/candidate prefix selection | Multiple output-selection policies remain. Exact gates are essential; ranking and candidate generation need consolidation after primary convergence is reliable. |
| Prior-plan continuation / retained prefix | Issued control history provides continuity. Epoch-based cursor alignment implemented; independent current-state verification remains mandatory. |
| Deadline deferral | Scheduling outcome, not infeasibility. Never publish late; discard unissued optimization memory. |
| Sim-only corridor-rejection publication | Removed. The `SDU_APEX_SIM_DIAGNOSTIC_ONLY` environment output bypass no longer exists; IROS runner no longer enables it. |
| Collision, finite-state, actuator-envelope, publication gates | Retained. No convergence experiment bypasses these protections. |

## Hidden-restriction audit

The normal QP has no correction box centered on its nominal. Increment mode
is a full coordinate transform; runtime defaults now span ±pi/3 steering
change and the full throttle range. Narrow command-step settings are rejected
by runtime configuration instead of silently imposing unsupported slew limits.
Generic offline test problems may explicitly request rate constraints.

Full alpha=1 is evaluated before backtracking. Four finite alpha samples and a
deadline gate are incomplete globalization, but they do not reject a direction
because it is far from nominal. Feedback gain damping changes a candidate
rollout policy, not the QP feasible domain. Offline source-optimizer direction
preconditioning has no runtime command authority.

Remaining restrictions must not be mislabeled physical truths: heading ±pi is
a coordinate chart; normal positive throttle excludes Unity's brake branch.
Wheel-rate ±350 entries already exceeded the solver's BIG_BOUND=50 sentinel
and were unconstrained. They now use the standard marker; this is semantic
cleanup, not removal of an active limit. Primary restoration no longer applies undocumented .35 rad heading
and lateral-velocity terminal funnels. Recovery
heading/slip funnels and clipped repair amplitudes constrain recovery search,
not normal SQP, and remain heuristics rather than certified terminal safe sets.

## Evidence and closure criteria

New tests cover causal pose brackets, historical map/odom transforms, bounded
epoch waits, same-epoch nominal invariance, all actuator/sensor arrival orders,
missing-epoch refusal, atomic pair discretization, feedback-radian preservation
and sensor packet ordering. Existing source/adjoint/QP tests remain required.

The prior profile regression expected feedback gain .75 while the checked-in
development profile and preceding trials use .25. Its fingerprint is now
explicitly the **unqualified development profile**; the historical 49.162 s
five-race baseline is still archived. Updating that stale fingerprint does not
promote current behavior to qualified.

Remaining Phase 1 closure work:

1. Prove or enforce requested command epoch through middleware and Unity apply
   tick; distinguish request/response association from a physical timestamp.
2. Qualify the new elapsed-command-stage policy for warm controls, ADMM memory
   and retained-plan/prefix cursors. Repeated/skipped/reversed/expired epochs
   have regression coverage; the physical host/application mapping remains open.
3. Establish complete-packet loss detection. Partial-packet ordering and exact
   actuator joining do not prove that an entire response was observed.
4. Close AMCL startup-prior versus scan-correction epoch semantics explicitly.
5. Unify remaining recovery candidate/selection policies only after primary
   restoration and terminal behavior meet repeatable convergence gates.

## Six-phase work ledger

The user's ordering is binding: later tuning cannot qualify an unresolved
foundation. This ledger separates implementation evidence from completion.

| Requested items | Current state / next evidence |
| --- | --- |
| 1 data flow | Full trace above; concrete epoch bugs repaired; physical apply acknowledgment still open. |
| 2 one pipeline | Source-primary, independent polish and unsafe publication bypass removed; recovery selection remains. |
| 3 hidden trust bounds | Runtime command-step defaults/validation repaired; wheel sentinels normalized and primary-restoration terminal funnels removed; chart/recovery heuristics documented. |
| 4 mathematical reference | Specification added, including component mismatches and uncertainty. |
| 5 Unity fidelity | Existing corrected-command parity evidence; full physical regime qualification incomplete. |
| 6 observer accuracy | Estimated/corrected state provenance documented; independent error/uncertainty corpus needed. |
| 7 actuators | Atomic pair and exact stamped feedback/request association implemented; apply-tick evidence remains. |
| 8 multiscale integration | Source/Unity 25–750 ms tools exist; remeasure after epoch closure. |
| 9 Jacobians | Local source tests exist; branch-boundary/perturbation convergence coverage incomplete. |
| 10 full objective derivatives | Frozen objective/adjoint regression exists; broad physical regimes and finite-step curvature remain open. |
| 11 state coordinates | 23-coordinate chart and omitted timer decisions documented; conditioning/large-slip chart qualification open. |
| 12 objective audit | Actual terms/branches specified; racing/physical justification not yet validated. |
| 13 anticipation | Horizon/delay/positive-throttle restrictions explicit; controlled corner tests needed. |
| 14 tire feasibility | Source dynamics and coupled slip GN retained; no independently qualified force-feasibility constraints. |
| 15 corridor | Four-corner planes and candidate-progress geometry; swept map checking remains optional and endpoint-interpolated. |
| 16 terminal viability | Experimental integrated costs/planes/nonlinear suffix exist; live failure unresolved, default off. |
| 17 QP direction | Source-gradient secant inside R2; one promising run, incomplete repeatability. |
| 18 primary restoration | PSD restoration fixed; elastic/filter primary formulation not completed. |
| 19 QP construction | Full coupling/coordinate tests and OSQP fixtures; expand permanent corpus. |
| 20 ADMM failures | Restoration indefinite-Hessian cause reproduced; capped convex corner QPs still need conditioning analysis. |
| 21 scaling | Exact variable transform tested; principled constraint-row scaling still open. |
| 22 Riccati | Shared factorization and reported regularization; worst-case precision/reuse qualification open. |
| 23 warm starts | Progress/penalty rebasing and missed-publication fix; elapsed-epoch alignment implemented and awaiting live qualification. |
| 24 globalization | Frozen merit and two passes; scalar finite-alpha and invalid-model refinement limitations remain. |
| 25 failure classes | Separate solver/nonlinear/deadline/publication statuses; complete packet/application provenance missing. |
| 26 timing | Existing component diagnostics; new pipeline requires actual-load tail measurements. |
| 27 redundant work | Shared QP adjoint; remaining duplicate geometry/rollout work needs measured profiles. |
| 28 budgets | Hard publication gate and R2 reserve retained; no worst-case completion guarantee. |
| 29 continuity | Confirmed requests and atomic pairing improve input integrity; oscillation/disturbance gates incomplete. |
| 30 fallbacks | Unsafe bypass removed; necessary source/physical gates retained; no blanket removal. |
| 31 failure corpus | Known saved source/QP corner cases exist; exact live solver/middleware warm snapshots not complete. |
| 32 observer perturbations | Not qualified. |
| 33 model mismatch | Not qualified. |
| 34 boundary cases | Partial source/corner tests; systematic measured corpus incomplete. |
| 35 both tracks | Separate configurations retained; competition unqualified. |
| 36 qualification | Five consecutive complete ten-scored-lap races, zero collisions/rejections/hard misses, recovery reduction, 20 ms hard/15–17 ms target and baseline preservation remain required. No gate waived. |

## Initial integration evidence

`phase1_epoch_integrity_20261101_01` completed one warmup plus two scored laps
without reported collision, MPC rejection or recovery. This preceded the final
elapsed-epoch and hidden-bound changes and is not evidence for their live behavior.
All 647 odometry epochs matched exact sent commands, with zero value mismatch.
The recorder discovered the newly created feedback publisher late and missed
its first three samples; 644 recorded matches had at most 5.6e-17 rad roundoff.
The audit correctly refused certification. The publisher now initializes before
the first request response so discovery can complete before feedback arrives.

The six-state unbudgeted corner corpus retained all previous normal/source
costs and completed two passes in every mode. A concurrent 20 ms replay during
container compilation produced eight solver-budget skips, with no ADMM attempt
in those cases. Those scheduling outcomes are preserved, not called infeasibility
or used as isolated worst-case controller timing evidence.
