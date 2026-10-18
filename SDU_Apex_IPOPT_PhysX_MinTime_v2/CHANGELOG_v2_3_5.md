# v2.3.5 — hard-feasible IPOPT checkpoint backtracking

The v2.3.4 user report had no accepted Phase-I checkpoint after 24 IPOPT steps.
Its elastic initial point was feasible only because the elastic variables
absorbed 1.064776 dynamic defects; IPOPT full iterates violated hard
constraints and could not be accepted.

This update evaluates shorter numerical segments from the last accepted
hard-feasible candidate toward each IPOPT iterate. The numerical elastic
variables are reprojected to the *original* dynamic residual for each trial.
A checkpoint is recorded only if every track/footprint/actuator constraint
still satisfies the original tolerance and the raw dynamic-defect merit
strictly improves. This changes **checkpoint selection, not IPOPT objective,
vehicle dynamics, hard constraints, or final strict zero-slack audit**.
The report exposes `accepted_hard_feasible_backtracks`.

The original recovered source remains unmodified, and recognized v2.3.4
working solvers upgrade in-place with byte-exact validation. The ZIP does
not include `.venv`, native executables, or user output directories.

No full Porto IPOPT or native PhysX rollout was executed in producing this
update. Passing regression tests does not establish feasibility or a race lap.
