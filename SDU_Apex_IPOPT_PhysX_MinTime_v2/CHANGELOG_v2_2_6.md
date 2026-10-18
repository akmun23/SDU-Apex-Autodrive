# v2.2.6: feasible-iterate recovery and adaptive dynamics homotopy

The previous v2.2.5 run never retained improved IPOPT primal iterates unless its
**final** iterate was simultaneously feasible. The first stage finished with
7.65 constraint violation, so the feasible initialization was retained and
the next 0.35 stage started with 0.1165 constraint violation.

This numerical update: saves the best feasible IPOPT iteration via Opti's
iteration callback; attempts safeguarded backtracking along a failed end step;
uses intermediate dynamics tolerances chosen from the current measured defect
when the next scheduled tolerance is too strict; stops when the IPOPT iteration
budget runs out or physical defects stagnate. The exact zero-defect equations
and all original hard road/steering/footprint constraints remain acceptance gates.

It does not demonstrate a feasible full lap or Unity/PhysX parity.
The original restored solver is left unchanged, and a verified v2.2.5 cache
is automatically upgraded in the existing install directory.
