# MPC production/development code inventory

**Inspected:** 2026-10-21  
**Purpose:** identify what is compiled, installed, launched, and retained for
validation before deleting or splitting MPC code.  This is an inventory, not a
claim that dead-code cleanup is complete.

## Production dependency path

The `f1tenth_mpc` package has one production prediction/controller path:

```text
mpc_controller_node (ROS component)
  ├── mpc_control_time (command timing / scheduled queue)
  │    ├── mpc_state_sync (state and timestamp synchronization)
  │    └── mpc_core
  ├── mpc_state_sync
  └── mpc_core
       ├── mpc_reference.c
       ├── vehicle_model.c
       ├── unity_source_plant.cpp + unity_source_plant_c.cpp
       ├── mpc_linearization.c
       ├── mpc_rti.c
       └── riccati_solver.c
```

`f1tenth_mpc/CMakeLists.txt` builds and installs `mpc_core`, `mpc_state_sync`,
`mpc_control_time`, and `mpc_controller_node`. `mpc_core` contains the single
source-derived nonlinear plant used by rollout and validation. There is no
second production vehicle model in this package. The Riccati profile clock
reads are compile-time optional (`SDU_APEX_MPC_RICCATI_PROFILE`, default OFF);
deadline logic, numerical checks and safety validation are not disabled by
that option.

## Runtime launch and image boundary

- `docker/competition_entrypoint.sh` has a fixed startup: source installed ROS
  workspaces, then `ros2 launch sdu_apex_autodrive competition.launch.py`.
- `sdu_apex_autodrive/launch/competition.launch.py` starts the 40 Hz bridge,
  odometry, EKF, map server, AMCL, actuator interface, and then the MPC
  component after localization warm-up. The competition launch does not enable
  MPC diagnostics or the CSV controller.
- `sdu_apex_autodrive/launch/controller.launch.py` is the development launch.
  It supports map/raceline and parameter overlays, diagnostic publication,
  CSV command replay, and optional RViz. It rejects duplicate runtime nodes
  and disallows simultaneous CSV and MPC command publishers.
- The root `Dockerfile` builds the ROS workspace with `BUILD_TESTING=OFF`.
  `.dockerignore` excludes `tools/`, `docs/`, `live_runs/`, and `external/`;
  therefore host-side analysis/replay utilities and their data are not part of
  the competition build context. The same package C++ sources are compiled
  into the runtime image.
- `docker/development_entrypoint.sh` can rebuild mounted runtime source, then
  launch the same controller launch with development-only options.

## Development and evidence dependencies (retain)

These targets/tools are intentionally not installed into the runtime image.
They are currently required to reproduce claims or satisfy the handoff's
future gates, so none is classified as safe-to-delete:

| Item | Build/runtime status | Why it remains |
|---|---|---|
| `mpc_bag_solver_replay` | CMake `EXCLUDE_FROM_ALL`; only defined when the local checkout has `tools/mpc_bag_solver_replay.cpp` | Replays saved states through the production RTI core; used for PR-A/C/D comparisons. |
| `mpc_qp_fixture_replay` | `EXCLUDE_FROM_ALL`; grouped under `mpc_offline_qp_tools` | Replays versioned frozen-QP fixtures against the same solver. |
| `unity_source_plant_eval` | Shared library, `EXCLUDE_FROM_ALL`, not installed or linked to the controller | Lets bag analysis call the exact production plant ABI without creating a second equation implementation. |
| `tools/evaluate_mpc_bag.py` | Host/development utility; absent from Docker context | Decodes bags and scores truth, odometry, source-plant replay, controller and command timing. |
| `tools/export_mpc_run_report.py` | Host/development utility | Produces schema-versioned run reports; needed to classify every supervised attempt. |
| `tools/capture_mpc_qp_regression_corpus.py`, `mpc_qp_fixture.py`, `mpc_qp_fixture_format.hpp`, `validate_mpc_qp_corpus.py` | Host tooling plus explicit offline target | Build and validate exact frozen-QP evidence for PR-A and later solver work. |
| `tools/mpc_qp_oracle.py`, `refine_mpc_qp_oracle_cases.py`, `requirements_mpc_qp_oracle.txt` | Optional host Python analysis | Independent QP accuracy/feasibility comparison; no production imports. |
| `tools/seal_mpc_run_manifest.py` | Host preflight utility | Seals image, source, map, raceline and configuration before controlled runs. |
| `tools/record_debug_bag.sh`, `watch_sim_run.py/.sh` | Development launch supervision and capture | Required for supervised tests, full evidence capture, and immediate collision/fault abort. |
| `tools/sweep_mpc_weights.py`, `benchmark_mpc_replay.py` | Offline experiments | Keep available but do not use for production selection until PR-B/C/G prerequisites pass. |
| `f1tenth_mpc/test/*` and Python tests | Built only with `BUILD_TESTING=ON` | Plant, state sync, command time, envelope, fixture schema, manifest, and regression coverage. |

## Configuration and launch references checked

- Runtime controller config: `f1tenth_mpc/config/mpc_competition.yaml`.
- Development/experiment overlays: `mpc_accuracy_validation.yaml`,
  `mpc_full_speed_overspeed_candidate.yaml`, and
  `mpc_open_loop_horizon_test.yaml`. An overlay is not production config.
- The only launch entrypoints found in this package tree are
  `competition.launch.py`, `controller.launch.py`, and
  `mpc_jacobian_replay_benchmark.launch.py`. The benchmark launch is a
  development utility and is not used by either entrypoint above.
- The compiler profile flag is referenced in CMake and `riccati_solver.c`;
  it is not a ROS runtime parameter.
- Development-only open-loop and diagnostic implementation lives in
  `mpc_controller_node.cpp`; it is not a separate linked controller. No
  extraction was attempted: moving it before focused call-graph and object
  lifetime tests would be a structural rewrite with no demonstrated runtime
  benefit.

## Cleanup decision

No production source or replay utility was deleted in this pass. The current
handoff requires the saved QP corpus, source-plant evaluator, bag decoder,
prelaunch manifest, report exporter, open-loop harness, and supervisor for
PR-A through PR-J. Deleting those would make required evidence irreproducible.
The candidate “split diagnostics out of the 3k-line node” is not yet
demonstrably dead code or a performance win. A future deletion-only change
must name exact symbols, show no launch/build/test references, and preserve the
listed validation modes.

## Commands used to inspect this inventory

```bash
rg --files docker f1tenth_mpc sdu_apex_autodrive/launch
rg -n 'add_executable|add_library|EXCLUDE_FROM_ALL|install\\(TARGETS|ament_add_gtest|add_test\\(' f1tenth_mpc/CMakeLists.txt
rg -n 'mpc_(core|state_sync|control_time|controller_node|bag_solver_replay|source_plant_eval)|MPC_ENABLE_RICCATI_PROFILE' Dockerfile docker f1tenth_mpc sdu_apex_autodrive/launch tools
```

**Conclusion:** production/development boundaries are identifiable and the
inventory deliverable is complete. Actual deletion is intentionally not
claimed; no candidate was proven unreachable without damaging an outstanding
handoff test path.
