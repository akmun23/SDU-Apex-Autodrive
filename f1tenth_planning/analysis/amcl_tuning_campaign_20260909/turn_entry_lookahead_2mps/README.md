# Turn-entry lookahead experiment

This is a real graphical AutoDRIVE ICRA-compete run using `-batchmode` without
`-no-graphics`. It keeps the production localization and odometry code and the
2.0 m/s controller cap from `turn_entry_fix_2mps`.

The only runtime controller change is an explicit `min_lookahead=0.60 m`
parameter override. The purpose is to test whether the 0.166 m startup
raceline offset is being over-corrected by the 0.35 m stationary lookahead.
Ground truth is recorded only for offline scoring; it is not used by control.

This run is a controller diagnostic, not an acceptance pass. A successful
entry must be followed by a collision-free higher-speed run.
