# Practice-track path optimization

Current evidence and remaining work:
[structural investigation](../docs/development/IPOPT_STRUCTURAL_RESET_20261028.md).
Production racing remains the separate
[51.030 s baseline](../docs/development/RACING_51P030_FREEZE_20261028.md).

## Two distinct problems

1. `run_ipopt.py`: free spatial geometry/speed search with source-derived force
   equations. It produces a target, not executable held commands. Spatial
   feasibility is neither a KKT certificate nor Unity validation.
2. `command_ocp.py`: periodic command-feasibility polishing at a fixed trial
   duration. It links the actual production C++ plant and forward-mode
   derivatives. Inputs are held steering targets and positive throttle, not
   continuously optimized rack rates. Fewer intervals test a shorter lap;
   they do not prove a global physical minimum.

The command OCP has 21 shooting states: rear-axle pose plus 18 source states,
including all four suspension histories. Sticky histories are a proven zero
invariant in positive-drive mode, not discarded degrees of freedom in an
arbitrary brake/low-speed mode. Controls stay separate from the exported
geometry/speed/curvature target. No runtime reference is replaced automatically.

No fitted dynamics or external PhysX process is used by the command polisher.
The shared production plant remains a source translation, not Unity's private
native solver. Its consistency cannot certify Unity agreement.

## Source contract

- Practice vehicle only; serialized scene guards in `source_vehicle.py`.
- Practice scene motor override: 428 Nm; CAWD sends q*428/4 Nm per wheel.
- q=0 is a distinct hard-brake branch. Optimizers restrict q>=0.001, so they
  cannot rely on that unavailable racing mode.
- Pinned schedule: five internal substeps below 5 m/s, one at/above 5 m/s;
  outer physics tick 1 ms. The former 128 m/s interpretation is invalid.
- WheelCollider radius 0.059 m; Ackermann 0.324/0.236 m; steering limit 30°.
  Constants come from owned code and active Unity assets; unresolved aliases
  fail closed, not through a retired odometry executable.
- Wall margin: 0.05 m beyond the complete vehicle footprint.

## Entry points

Run from the repository root with this package's Python environment. A previous
command witness is only an initializer, never transferred certification:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python -u \
  SDU_Apex_IPOPT_PhysX_MinTime_v2/command_ocp.py \
  --command-seed PATH_TO_PREVIOUS_WITNESS.npz \
  --map f1tenth_planning/maps/autodrive_practice_20260924_b.pgm \
  --intervals 195 --iterations 100 --cpu-seconds 90 --workers 8 \
  --refine-clearance-passes 3 \
  --output optimized_practice_command_campaign/TRIAL
```

195 intervals is 4.875 s. Each solve emits iteration/residual progress, a
checkpoint and report. Bounded refinement adds actual source-prefix wall
constraints at failing 1 ms ticks. It does not move the exported path afterward
or weaken the margin. Numerical domain/ordered-basin guards are reported, not
presented as physical limits.

For a spatial initializer, provide `--checkpoint`, `--mesh` and `--audit`
together. The audit NPZ comes from `run_ipopt.py --audit-time-checkpoint
SAVED_ITERATE.npz` with the original feasible seed and matching map/mesh/model.
That mode evaluates full constraint/dual/stationarity information without a
time solve. `min_curvature_seed.py` makes geometry basins, not time certificates.
Local micro-adjustments remain downstream of a valid initial path.

## Acceptance

Keep spatial feasibility, IPOPT convergence, command realizability, independent
recursive source replay, Unity agreement and MPC tracking separate. Predicted
duration is not an achieved racing time.

Replay command witnesses from one initial state with no future resets. Report
position, yaw, forward/lateral speed, yaw rate and wheel errors separately.
Check the full perimeter at every 1 ms source tick, not only 25 ms nodes.
Continuous inter-tick collision behavior and native Unity parity remain
additional obligations. Only then consider MPC nominal integration and practice
runs. Warmup never counts in ten scored laps; ground truth is offline only.

## Files and generated data

- `command_ocp.py`, `command_ocp_kernel.cpp`: production-map witness NLP.
- `command_realizability.py`: spatial-to-command diagnostic and rack inverse.
- `ipopt_nlp_audit.py`: saved-primal/dual constraint and KKT inspection.
- `source_equation_kernels.py`: compact force/derivative evaluation,
  scaling/trust/barrier policies and batched immutable export.
- `source_vehicle.py`, `path_frame.py`, `map_track.py`, `physics.py`: source
  parameters, coordinates, footprint/map geometry and production-plant adapter.
- `recover.py` and imported overlays: current spatial-model builder. Versioned
  imported layers are dependencies, not alternate racing models. Replacing
  that builder is required before deleting its dependencies.

Generated `_working_solver*`, `_restored_git_history`, `optimized_*`, recordings
and caches are ignored outputs, not source to commit. The spatial builder still
requires base Git commit `ca120e0cde7ae60e70082d38f86c5931f79f148f`; do not prune
that history. Old ZIP uploads are not active entry points.

Focused tests use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, this package's `pytest.ini`
and `tests/test_command_ocp.py`, `test_command_realizability.py`,
`test_ipopt_nlp_audit.py`, `test_source_equation_kernels.py`,
`test_v251_phase1_reduction.py`, `test_v251_source_equation_parity.py`,
`test_path_frame.py`, `test_physics_ctypes_abi.py`, `test_cpp_alias_steering.py`.
Historical optional-mode test failures are documented separately; do not hide
them or call the full historical suite green.
