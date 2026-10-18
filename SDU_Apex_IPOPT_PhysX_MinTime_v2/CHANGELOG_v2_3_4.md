# v2.3.4 — strict Phase-I diagnostics after an unsuccessful solve

Evidence: the user's v2.3.3 run produced 12 IPOPT iterations in 1.160 s, but
`ipopt_report_pass_00.json` was stripped of the actual physical defects by
`Opti::value: Assertion "solved()" failed`.  The 113 s Python-side elapsed time
includes other NLP setup and model initialization; IPOPT's 1.16 s does not
prove the entire invocation is fast.

## Changes

- A failed elastic IPOPT feasibility solve now returns immediately using the
  already-evaluated **zero-slack strict audit**. It does not query `Opti.value`
  or `Opti.debug.value` on an unsolved NLP.
- Retains labeled physical residuals (`elastic_bound_lateral_force_balance_k*`,
  etc.), `initial_guess`, IPOPT status and iteration count, best raw defect and
  merit in `solve_stages`, and a definitive `feasible_incumbent: false`.
- Preserves the existing successful strict-physics route unchanged. Phase II
  remains accessible **only** for strict physically feasible Phase-I results.
- Accepts byte-exact v2.3.3 generated solver cache for an in-place upgrade;
  unknown local edits still cause fail-closed rejection.
- `diagnose_phase1_v234.sh` runs 24 feasibility iterations with visible IPOPT
  output and a 180 s external timeout, reusing the existing native seed with
  **no native replay**.

## Limitations

This fixes a reporting and control-flow failure, **not** the underlying poor
conditioning of the elastic OCP. v2.3.3's iteration 12 had primal
infeasibility ~45.91 and no accepted strict feasible incumbent. The
historical differentiable Unity-wheel model remains an approximation to native
PhysX, with demonstrated lateral force/load differences. No sub-4.9 s result.

Install over the existing `SDU_Apex_IPOPT_PhysX_MinTime_v2/` folder; do not
delete `.venv`, compiled native PhysX or native_warmstart.npz.
