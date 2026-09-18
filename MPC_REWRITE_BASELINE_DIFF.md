# MPC rewrite baseline delta

The handoff pins `845e621989daaf1e9fbe041723f7e6066a6f01d2`; implementation is
currently at `685ccca0f549468669ac36aa68fcfc24e503d458` (`MPC started`). The
current commit is a substantial MPC rewrite, so Phase 0 must be revalidated
against this SHA before any live MPC authority is enabled.

The delta adds the continuous raceline projector, source-time state
synchronizer, nonlinear 7-state plant and 9-state augmented RTI/QP path,
generic Riccati-ADMM solver, ROS-node wiring/diagnostics, and associated tests
and offline-replay artifacts. It changes MPC equations, timing, and controller
launch integration—the handoff's explicit Phase-0 recheck condition.

No odometry, EKF, AMCL, Unity, or simulator-physics source files differ in the
dev-repository delta. The MPC-side model stage and YAML do change; their use of
the accepted model constants and current weights is part of the Phase-0 audit,
not a simulator-physics change. This is a scope observation, not a live
validation claim. The separate Unity checkout is dirty and is being treated as
user-owned; no Unity files have been changed in this task.

Phase-0 evidence is being regenerated from the current source in temporary
build/artifact locations. The checked-out worktree is dirty only because the
independent constrained-QP oracle regression is currently being added to
`test_riccati_solver.c`.
