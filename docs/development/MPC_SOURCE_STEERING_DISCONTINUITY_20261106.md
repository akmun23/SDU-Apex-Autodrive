# Recorded MPC recovery: steering discontinuity and evidence boundaries

6 November 2026. Controller remains unqualified. Numerical replacement and
filter experiments from this investigation were rejected and reverted. No
Unity physics or runtime actuator law was changed.

## Recoverability

A fresh independent original-source oracle evaluation of the archived legal
30-stage witnesses passed **22/22** states. Epoch 775 has minimum constraint
margin `0.00008550286293029785 m`. This is slack beyond the configured original
body/corridor margin, not necessarily literal vehicle-to-wall distance.
Existence under this model does not prove Unity recovery or NLP optimality.

The original witness search minimizes exact nonlinear physical violation over
constant, switched and block-adjusted legal control sequences; it does not
require racing-cost improvement. Twenty-one states had such witnesses. Epoch
775 was recovered by SLSQP from an already improved, near-feasible seed
(previous native result: margin approximately -0.0101412 m), using 300 iterations
and 2007 oracle calls. It did not establish recovery from the recorded warm
start or KKT convergence. Its historical oracle retained an unintended extra
reserve cost; its controls remain physical evidence after original-cost replay.

The latest pinned native qualification uses recorded controls, 2000 maximum
iterations, two repeats: **20/22 feasible, 0/22 KKT**, **5/10 perturbed feasible**.
All 22 recorded repeats agree; every iteration's independent checks passed.
Failures remain 774 and 775. No source unrecoverability was demonstrated.

## A concrete nonsmooth mechanism at the failed epoch-775 iterate

Only zero-based control coordinate 6 (fourth steering command) was changed:

| Quantity | Original | Perturbed |
| --- | ---: | ---: |
| Steering command, rad | -0.39126095175743103 | -0.39126086235046387 |
| Incoming physical rack, rad | +0.39126089215278625 | +0.39126089215278625 |
| Rack after that 25 ms stage, rad | +0.31126111745834351 | -0.39126086235046387 |
| Minimum full-horizon constraint margin, m | -0.037008136510849 | -0.4559504985809326 |

The command changes by **8.940696716308594e-8 rad**. The resulting rack changes
by **0.702522 rad**, about **40.25 degrees**. Earlier source states are identical.
No vehicle-substep schedule changes were reported. Other downstream tire,
reference and peak-slip branch changes accompany the different steering history;
those event counts alone would not have established the initiating cause.

The checked-out Unity `Assets/Scripts/VehicleController.cs:126` contains:

```csharp
if (AutonomousSteering > (lastSteeringAngle / SteeringLimit)) {
    SteeringAngle -= SteeringRate * Time.deltaTime;
    if (SteeringAngle <= -AutonomousSteering * SteeringLimit)
        SteeringAngle = -AutonomousSteering * SteeringLimit;
} else {
    SteeringAngle += SteeringRate * Time.deltaTime;
    if (SteeringAngle >= -AutonomousSteering * SteeringLimit)
        SteeringAngle = -AutonomousSteering * SteeringLimit;
}
```

The internal angle has the opposite sign to physical rack angle. The branch
compares the normalized command with that internal angle, then clamps against
the negative command. Around `command_rad = -physical_rack_rad`, one side slews
and the other can immediately clamp across the steering range. The C++ source
translation preserves this law at `unity_source_plant.cpp:1258`.

A permanent native regression, `OppositeSteeringTargetHasFiniteSourceBranchJump`,
isolates the two commands in one 1 ms source step. One-sided output separation
already exceeds 0.77 rad. It documents source behavior; it does not smooth or
repair the simulator.

This is a finite actuator-mode jump, not missing smooth Hessian curvature.
A smooth SQP/IPOPT model cannot extrapolate a derivative across it reliably.
It is one demonstrated failure mechanism, not proof that every failure has the
same cause. The checked-out C# law still needs comparison with the actual running
Unity build, float arithmetic, input conversion and FixedUpdate ordering.
Correcting C++ to a symmetric actuator without changing Unity would change the
physical problem being predicted.

A diagnostic straight interpolation from the original recorded controls to the
known witness reduced sampled violation from 15.9786 to zero. From the failed
final controls, however, moving only 1e-7 of the way toward the same witness
increased violation from 0.003166 to 2.008169 because of the steering jump.
These sampled paths are not global reachability or monotonicity proofs, and no
witness was supplied to the qualification solver.

## Distinguish local prediction error from Unity fidelity

Saved full SQP trials, independently reintegrated through the source:

| Epoch | Affine minimum margin | Exact source minimum margin | Maximum footprint prediction error |
| --- | ---: | ---: | ---: |
| 765 | +0.02421 m | -0.70827 m | 0.83228 m |
| 769 | +0.01453 m | -0.43120 m | 0.66722 m |
| 775 | +0.01080 m | -0.54327 m | 1.13750 m |

These errors arise between a local model and its own nonlinear forward model.
They cannot be attributed to Unity mismatch. Branch-local quadratic corrections
can be worse across switches. At 0.001 of the epoch-775 saved step, by contrast,
maximum footprint error was about 8 micrometres (affine) and 0.38 micrometres
(quadratic). Local accuracy does not imply a valid finite-step prediction.

Separate historical recorded Unity evaluation propagated 30 source stages
(0.75 s), initialized once. For 144 bridge-command windows: P95 maximum path
error **5.44 cm**, maximum **8.29 cm**, and **9/144** exceeded 5 cm. These are
position errors, not direct rectangular-footprint errors. Overlapping windows
are not independent; host requests do not certify actual FixedUpdate inputs.
This is an earlier source-library build, not new Unity qualification.

## Cost: original formula, local prediction and actual Unity outcome

The authoritative nonlinear evaluator scores the original racing objective.
The unintended additional 3 cm reserve term was removed; genuine original
reserve costs remain. Frozen initial-cost replay and source-objective parity
checks pass. Shared source evaluation is not an independent Unity cost oracle.

The local QP prediction is materially inaccurate on the saved full trials.
At epoch 775, original initial cost is **69952.38886**. The QP predicts change
**-63190.79786**, hence cost **6761.59099**; exact source evaluation gives
**32532.66392**. That trial is infeasible, so lower cost cannot certify success.

The historical SLSQP witness report distinguished augmented cost
**23465.15268** from original racing cost **20802.71761**: the added reserve
inflated that scalar by **2662.43507 (12.80%)**. Its physical witness remains
valid, but that historical optimization cannot certify original-objective
optimality. No result establishes the cost achieved by executing this full
candidate in Unity, nor the original constrained optimum.

## Local evidence and reproduction

Detailed JSON/CSV evidence lives under
`artifacts/source_path_recovery_20261106/`:

- `qualification_fixed_normal/phase1_report.json`: pinned primary qualification.
- `recoverability_recheck.json`: fresh independent 22-witness evaluation.
- `failed775_coordinate_jump.json`: isolated command perturbation.
- `failed775_original_trace.csv`, `failed775_perturbed_trace.csv`: source states
  at each stage; columns are stage, steering command, rack, u, v, r, e_y, e_psi,
  progress.
- `failed775_small_step_branches.json`, `witness_path_775.json`: branch and
  sampled-path diagnostics.
- `curvature_final_{765,769,775}.json`: saved-trial local-model audit.
- `root_cause_review_provenance.json`: hashes and qualification counts.

Historical Unity reports: `artifacts/phase6_competition_20261105/`.
Witness controls remain in the permanent corpus's `investigation.json.gz`.
Artifacts are local evidence; the checked-in corpus and regression are the
portable proof. Run the isolated actuator regression with:

```bash
ctest --test-dir build_task/source_curvature \
  -R '^test_unity_source_plant$' --output-on-failure
```

Next work should validate this actuator switching boundary against Unity first,
then decide the appropriate treatment of the resulting hybrid control problem.
Further solver comparisons alone do not resolve the demonstrated discontinuity.
