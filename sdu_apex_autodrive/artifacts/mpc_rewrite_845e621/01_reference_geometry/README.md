# MPC rewrite Phase 2: continuous path geometry

The controller now delegates path sampling and projection to
`f1tenth_mpc/src/mpc_reference.c`. The module unwraps path heading, validates
strict arc-length ordering and positive corridor bounds, removes a repeated
closing row, computes closed-loop length including the final segment, and
provides periodic interpolation and continuous signed segment projection.

The ROS adapter no longer uses closest-waypoint tracking or integer waypoint
jumps. Its cold-start N+1 reference seed advances by the sampled raceline
speed times 25 ms. The nonlinear rollout will replace this seed with
model-consistent progress in the RTI phase.

The current raceline CSV was loaded in the automated test: 2,587 rows become
2,586 unique points, with a 51.7 m closed-loop length. A deterministic set of
1,000 points offset from actual raceline segments validates cross-track
projection against the segment oracle with a 2 mm bound. No simulator was
started.
