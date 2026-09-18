# BachelorProject MPC for the AutoDRIVE simulator

This directory contains one MPC core: the BachelorProject Riccati-ADMM solver
and its one AutoDRIVE prediction-model interface. It does not contain an
alternate replay plant, a shadow controller, a historic fit manifest, or a
second controller implementation.

The object to model is the current Unity AutoDRIVE vehicle, not a physical
race car. `config/autodrive_simulator_contract.yaml` is the sole nominal
source contract. It records the current Unity command semantics and serialized
WheelCollider/Rigidbody anchors, and explicitly prohibits treating generic
cornering stiffness, friction coefficient, Pacejka shape, guessed load
transfer, or a direct Unity-acceleration channel as simulator facts.

The MPC is fixed at 30 prediction commands at 40 Hz (0.75 s) and is limited to
a 16 m/s project command ceiling. Simulator truth and hidden WheelCollider
state are offline diagnostics only. A controller model may be enabled only
after an observable stage map is fitted and passes the legal-state recursive
acceptance checks in `docs/SIMULATOR_NATIVE_MODEL_WORKLIST.md`.

Current cleanup status:

- [x] Removed the historic alternate `vehicle_plant` from the MPC package.
- [x] Removed report/artefact-pinned model manifests from the build.
- [x] Documented excluded SynPF and Cartographer work.
- [x] Replaced the legacy bicycle implementation with an explicitly limited
      source-command baseline; it contains no real-car tire or force law.
- [x] Added the minimal `controller:=mpc` ROS adapter and source-timing
      watchdog. Its command authority is inhibited by default.
- [x] Added an identified target-speed/body-speed response state and checked
      its recursive prediction against a separate PP holdout.
- [ ] Validate the updated controller in a closed-loop batchmode run before
      enabling MPC outside an explicitly authorized experiment.

Build from the Humble workspace container:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select f1tenth_mpc
```
