# Phase 0 — frozen reproducible baseline

Captured against the requested base SHA with the existing dirty worktree
preserved. The current package was rebuilt in the ROS 2 Humble dev container,
all package tests and both specified strict RTI replays were rerun, and the
core benchmark was extended to 10,000 cycles. Phase 1's clearance-reporting
correction was already applied before this capture; the pre-fix output is
recorded in `01_metrics_fix/metrics.json` for comparison.

This is offline/static evidence only. No simulator was running during these
checks and no full-lap claim is made.
