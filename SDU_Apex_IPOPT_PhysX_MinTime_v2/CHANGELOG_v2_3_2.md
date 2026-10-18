# v2.3.2 — safe in-place solver-cache upgrade

Fixes a regression in the v2.3.1 compatibility check: the old-overlay
reconstruction incorrectly included the extra `elastic_label_by_variable`
declaration added in v2.3.1, so an untouched v2.3.0 working solver was
misidentified as having unknown local edits.

The v2.3.0 and v2.2.8 compatibility transforms now reconstruct the earlier
solver byte-for-byte. The updater continues to refuse overwriting an
unrecognized manually edited solver; its historical source and other assets
remain untouched.

No modifications were made to the actual v2.3.1 IPOPT model, its constraints,
its PhysX-derived initialization, or any runtime physics implementation.

If the cache is already in an unrecognized state, the safe manual recovery is
to rename `_working_solver_limited_memory` before running the optimizer. It
will be regenerated from `_restored_git_history`.
