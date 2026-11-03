#!/usr/bin/env python3
"""Regression: optimized source MPC changes subsequent *physical* states.

Two different control policies are started at the SAME frozen predecessor
of a recorded corner failure. After selecting controls, each policy applies
exactly its first source-model action, advances the physical state and
progress, rebuilds the objective/strict QP, and solves from that new epoch.

This is a source-model causal counterfactual, NOT a Unity race or proof of
true simulator feasibility. The test checks the distinction between
(i) an already invalid original future horizon with a valid first command
and (ii) an improved strictly source-feasible future plan.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
CASE=(ROOT/"live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures"
      /"epoch_1793692917233236533_case_2")


def run(strategy:str,folder:Path):
    p=subprocess.run([sys.executable,
        str(ROOT/"tools/mpc_offline_counterfactual_closed_loop.py"),
        str(CASE),"--cycles","2","--iterations","2",
        "--strategy",strategy,"--out",str(folder)],
        capture_output=True,text=True,timeout=45)
    if p.returncode:
        raise AssertionError(f"{strategy} failed:\n{p.stderr[-1000:]}\n{p.stdout[-1000:]}")
    rows=json.loads((folder/"results.json").read_text())
    assert len(rows)==2,rows
    assert all(r["first_step_source_valid"] for r in rows),rows
    for j,r in enumerate(rows):
        assert r['epoch']==j
        assert (folder/f"counterfactual_case_{j+1:02d}"/"source_context.mctx").is_file()
        assert (folder/f"counterfactual_case_{j+1:02d}"/"r1.fqp").is_file()
        assert r["next_progress_m"] is not None
    assert rows[1]['next_progress_m']>rows[0]['next_progress_m']
    return rows


def main():
    if not CASE.is_dir():
        raise RuntimeError("missing captured predecessor case")
    with tempfile.TemporaryDirectory(prefix="mpc_receding_source_regression_") as temp:
        root=Path(temp)
        optimized=run("optimize",root/"optimized")
        shifted=run("shift_seed",root/"shifted")
        assert all(x["selected_source_valid"] for x in optimized),optimized
        assert all(not x["selected_source_valid"] for x in shifted),shifted
        assert all(x["selected_source_squared_corridor_violation_m2"]==0.
                   for x in optimized),optimized
        assert all(x["selected_source_squared_corridor_violation_m2"]>0.
                   for x in shifted),shifted
        # The first control inputs and the *resulting* future progress
        # differ. Later physical states are not replayed from logged frames.
        assert abs(optimized[0]["applied_steering_rad"]-
                   shifted[0]["applied_steering_rad"])>1e-4
        assert abs(optimized[0]["next_progress_m"]-
                   shifted[0]["next_progress_m"])>1e-7
        print("SOURCE_CAUSAL_NEXT_EPOCH_RECOMPUTE_PASS")
        print("UNOPTIMIZED_INFEASIBLE_FUTURE_VS_OPTIMIZED_VALID_TWO_EPOCHS_PASS")
        print(json.dumps({
            'epochs':2,
            'optimized_horizon_valid':2,
            'shifted_horizon_valid':0,
            'optimized_first_step_valid':2,
            'shifted_first_step_valid':2,
            'optimized_first_step_steering':optimized[0]['applied_steering_rad'],
            'shifted_first_step_steering':shifted[0]['applied_steering_rad'],
        }))


if __name__=="__main__":
    main()
