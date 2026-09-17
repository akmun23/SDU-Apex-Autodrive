# Simulator trace validation and offline scoring

This directory deliberately contains only the small offline toolchain needed
to validate a new simulator trace and score a causal candidate.  It is not a
real-vehicle identification toolkit and no file here is imported by runtime
odometry, localization, Pure Pursuit, or MPC.

- `assemble_transitions.py` validates source order, 40 Hz cadence, reset
  boundaries, and the packet-frame conversion before writing transitions.
- `check_repeatability.py` compares separate accepted traces without filling
  gaps or fabricating samples.
- `finalize_model_id_run.py` reconstructs trace partitions and timing reports
  after an interrupted recorder shutdown.
- `build_raceline_operating_envelope.py` labels the actual raceline's
  speed/curvature operating regions.
- `score_raceline_model.py` and `score_runtime_control_state.py` score an
  already-causal prediction or runtime estimate offline. Simulator truth is a
  score target only.
- `score_legal_n30.py` is the required acceptance scorer for a future model:
  it accepts only legal sensor-state origins and exactly N4/N10/N20/N30
  predictions at verified 40 Hz cadence.

The discarded utilities included four-wheel/tire/suspension surrogates,
polynomial throttle fits, artifact-pinned candidates, and diagnostic Unity
components. They were disconnected from the deployed controller and made it
too easy to mistake an approximation of a real car for the actual simulated
object.

The active contract and checked worklist are:

- `f1tenth_mpc/config/autodrive_simulator_contract.yaml`
- `f1tenth_mpc/docs/SIMULATOR_NATIVE_MODEL_WORKLIST.md`

New work must use a new accepted 40 Hz trace to identify an observable,
source-command stage map for the 30-stage / 0.75 s horizon. It must not
reintroduce generic friction, cornering-stiffness, Pacejka, load-transfer, or
direct-acceleration terms.
