# Phase 6 — dimensioned Riccati-ADMM and parity

The solver's active-dimension matrix loops were already dense and parameterized
in the dirty baseline; this phase verifies that they are correct for the new
nine-state layout instead of claiming an unnecessary full loop rewrite. An
independent condensed-QP oracle checks dense affine A/B/d and state/input cross
cost N for both 9- and 10-state test problems. Nine-state ADMM box projections
are separately exercised.

The control Hessian is symmetrized and checked for positive definiteness. A
bounded diagonal repair preserves off-diagonal coupling, records its count and
maximum lambda, and fails when material indefiniteness remains. The production
controller still uses the old 10-state QP builder; only solver parity for 9
states is established here.
