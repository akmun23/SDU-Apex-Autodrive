# Unity model-identification diagnostics

`ModelIdentificationDiagnostics.cs` is the versioned source for the
diagnostic-only component used by the disposable Unity identification build.
It is not part of the ROS runtime and does not provide control input or alter
physics. It writes the static Rigidbody/WheelCollider configuration and
per-fixed-step WheelHit/contact trace only when
`AUTODRIVE_MODEL_ID_DIAGNOSTICS_DIR` is set.

For controlled offline experiments, set
`AUTODRIVE_MODEL_ID_EXPERIMENT=ramp_sweep_v1` for the mixed steering/contact
sweep or `AUTODRIVE_MODEL_ID_EXPERIMENT=drive_excitation_v1` for the bounded
straight-line torque plateaus used by continuous wheel-state identification.
Use `AUTODRIVE_MODEL_ID_EXPERIMENT=combined_slip_matrix_v1` for the
diagnostic-only signed throttle/steering plateau matrix.
Use `AUTODRIVE_MODEL_ID_EXPERIMENT=raceline_relevant_speed_sweep_v1` for the
bounded speed-aware steering sweep. Its steering is larger only in the lower
speed plateaus and tapers toward the 16 m/s ceiling; it is diagnostic-only
and does not change the competition scene or physics.
These command sequences exist only in the disposable identification player;
they are not part of the competition scene or runtime controller. The static
dump uses schema `autodrive.simulator_diagnostics.v3` and includes per-wheel
sprung mass, suspension spring/damper/target-position, local rotation, and the
Rigidbody maximum angular velocity. The rigid-body snapshot emits
`bodyInertiaX`, `bodyInertiaY`, and `bodyInertiaZ` from the complete principal
tensor projection, and explicitly declares `yawAxis: body_y`.
`yawInertiaBodyFrame` is a compatibility field containing the body-Y value;
legacy local-Z dumps are rejected by the offline parser.

The trace also contains read-only `Rigidbody.GetAccumulatedForce` and
`GetAccumulatedTorque` fields. In the active F1TENTH vehicle these are zero:
the controller assigns WheelCollider inputs and the internal WheelCollider
solver applies its forces during the physics step. They are retained as an
API-boundary audit only; they are not a substitute for per-wheel solver force
and are never fed into the plant or MPC.

For actuator identification, use
`AUTODRIVE_MODEL_ID_EXPERIMENT=near_zero_actuator_v1` together with
`AUTODRIVE_MODEL_ID_INITIAL_SPEED_MPS` and
`AUTODRIVE_MODEL_ID_THROTTLE_NORM`. The disposable experiment first applies a
small positive preconditioning command so the injected body speed is not paired
with zero WheelCollider RPM. It then marks the requested-command measurement
window in the trace and exits cleanly. Configure the preconditioning and window
with `AUTODRIVE_MODEL_ID_WARMUP_SECONDS`,
`AUTODRIVE_MODEL_ID_WARMUP_THROTTLE_NORM`, and
`AUTODRIVE_MODEL_ID_MEASUREMENT_SECONDS`. The preconditioning rows must be
excluded from actuator fitting.

For E2 continuous powered dynamics, use
`AUTODRIVE_MODEL_ID_EXPERIMENT=powered_drive_repeat_v1` and set
`AUTODRIVE_MODEL_ID_EXPERIMENT_DURATION_SECONDS=51`. The reproducible runner
is `tools/model_id/run_powered_drive_campaign.py`. This profile contains only
positive drive commands, with repeated upsteps and downsteps; zero is excluded
because it invokes the hard-brake branch. Each run starts from rest and must
be fitted with source `fixed_time_s`, while brake-mode data stays in the
separate hybrid actuator report.

The file must be copied into the external Unity checkout before building the
throw-away diagnostic player. The competition scene/player remains the source
of all runtime behavior; no production simulator build is implied by this
directory.
