# Unity model-identification diagnostics

`ModelIdentificationDiagnostics.cs` is the versioned source for the
diagnostic-only component used by the disposable Unity identification build.
It is not part of the ROS runtime and does not provide control input or alter
physics. It writes the static Rigidbody/WheelCollider configuration and
per-fixed-step WheelHit/contact trace only when
`AUTODRIVE_MODEL_ID_DIAGNOSTICS_DIR` is set.

For a controlled offline contact experiment, additionally set
`AUTODRIVE_MODEL_ID_EXPERIMENT=ramp_sweep_v1`. This applies a deterministic
throttle/steering sequence only in the disposable identification player. It
is not part of the competition scene or runtime controller. The static dump
uses schema `autodrive.simulator_diagnostics.v2` and includes per-wheel sprung
mass, suspension spring/damper/target-position, local rotation, and the
Rigidbody maximum angular velocity.

The file must be copied into the external Unity checkout before building the
throw-away diagnostic player. The competition scene/player remains the source
of all runtime behavior; no production simulator build is implied by this
directory.
