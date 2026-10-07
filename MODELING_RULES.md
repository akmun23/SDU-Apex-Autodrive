# Vehicle-model development rules

This is the concise repository-side policy extract for the regime-specific
modeling plan dated 2026-10-07. The external handoff remains the detailed
source of truth.

## Evidence and model structure

- Do not seek one universal vehicle fit. Keep the production transition as the
  shared actuator/kinematic skeleton and treat each distinct response mechanism
  as a separately supported specialist.
- Preserve run, capture, phase, condition, and train/validation/test provenance.
  Support is joint in the measured state and action; marginal speed and
  steering coverage does not imply joint coverage.
- Discover specialist boundaries only where measured response changes. Do not
  substitute hand-picked speed/steering bins, a global lateral-acceleration
  cap, or interpolated limits unsupported by held-out captures.
- Keep simulator truth as an offline label only. Runtime MPC, odometry,
  localization, and filtering may consume only permitted runtime inputs.
- Keep the P0 reference trajectory and simulator physics unchanged unless a
  separate, explicit task authorizes changing them.
- Preserve the trusted Y0/P0 baseline. A candidate stays offline until it
  improves untouched whole-run evidence and passes the handoff's promotion
  gates.
- Do not start general plant fits, broad data sweeps, global lateral-cap
  escalation, or generic specialist reruns without a documented, evidence-led
  Data Gap Ticket.

## Test authorization for this task

Do not launch a simulator, replay a bag as a test, or run a model/controller
evaluation unless both conditions are met:

1. Existing recordings have been directly checked and none can address the
   specific question; and
2. The user has explicitly approved the proposed test.

Existing bags and saved analyses are available for the current atlas. Therefore
no new test is authorized by the current request. Read-only inventory and
analysis of existing evidence are allowed. Keep test/final-test partitions
sealed; record any accidental access and do not present it as blind evidence.

## Required reporting

For each specialist, record its mechanism, source artifacts, exact measured
state/action support, variables, split status, response-boundary evidence,
runtime status, and known limitations. Separate measured open-plane support,
practice evidence, and model-to-model diagnostics. A model score or optimizer
estimate is not a lap-time result.
