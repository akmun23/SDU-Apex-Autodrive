# v2.2.7 — elastic dynamic-defect feasibility (in-place)

**Reason:** The v2.2.6 Porto run remained at max raw dynamics defect 0.4665433
through six attempts. The accepted solution never left the synthetic initial
state. A tighter tolerance cannot help until the defect itself decreases.

**New Phase I (CasADi/IPOPT):** Replace each selected dynamics equality
`R(z)=0` by `R(z)-e=0` with an optimizable elastic defect `e`. Initialize
`e=R(z_seed)` for an algebraically consistent starting point. Minimize the
normalized sum of squares of elastic defects with a small seed regularizer.
This avoids the former fixed tolerance schedule and retains the best iterate
satisfying every active constraint via the IPOPT callback.

The lane/corridor and vehicle-footprint bounds, speed, steering, throttle and
normal-load inequalities, and spatial lap geometry are **never softened**.

**Acceptance:** Before minimum-time optimization, set the elastic bounds to
EXACTLY zero. Audit the full original residuals, elastic states and all
constraints against the existing 1e-3 feasibility tolerance. There is no
MPC raceline export without subsequent native PhysX validation.

The historical recovered solver remains byte-for-byte unchanged; the working
copy is automatically upgraded only from recognized, unmodified prior
working copies (including v2.2.6). No .venv/binary/output data in the ZIP.

**Verification scope:** The Python package tests and toy IPOPT regression
models cover the patch and strict rejection; a full Porto minimum-time
solve, native PhysX rollover and Unity parity are **not** verified in this
environment. An exact physics-consistent lap may still require a native
simulator-generated warm-start or correction of the symbolic dynamic model.

**Run in the same folder:**

```bash
python3 run_ipopt.py --repo-root .. --mesh-spacing 0.20 \
  --brake-zones 0 --phase1-hessian limited-memory \
  --feasibility-iterations 1200 --iterations 1200 \
  --physx-binary ./native_physx_vehicle_step \
  --output ./optimized_practice_v2_2_7 --allow-history-fetch
```
