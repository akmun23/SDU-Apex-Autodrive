"""Offline complete 30-stage original-source 0.2ms branch audit.

Every case uses a real first-QP direction from the captured Unity callback.
Every stage is traced through the production Unity-derived source step,
including contact, slip piece, combined saturation, steering, and sticky
state changes. No simulator connection, control publication, or fitted model.
"""
from __future__ import annotations
import collections,json,os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
EXEC=ROOT/"build/f1tenth_mpc/mpc_source_stage_second_order_probe"
CAP=ROOT/"live_runs/mpc_source_second_order_20261104_01"
CASES={"15":"case_15","16":"case_16","live_fast_17":"case_live_fast_17"}
H=(.0001,.001,.01)
def one(case:str):
    folder=CAP/CASES[case]
    direction=(folder/"first_qp_qp_direction.txt" if case!="live_fast_17"
               else folder/"probe_finite_20261104_qp_direction.txt")
    summary={}
    for h in H:
        counts=collections.Counter()
        bad=[]
        stages=[]
        for k in range(30):
            env=dict(os.environ,OMP_NUM_THREADS="6",OMP_PROC_BIND="spread",
                     OMP_PLACES="threads",MPC_SOURCE_INNER_TRACE_STAGE=str(k))
            p=subprocess.run(["taskset","-c","0,2,4,6,8,10",str(EXEC),
                str(folder),str(direction),str(h)],cwd=ROOT,env=env,
                capture_output=True,text=True,timeout=45)
            if p.returncode:
                raise RuntimeError(f"{case} {h} {k}: {p.stderr[-250:]}")
            parsed=json.loads(p.stdout)
            stage=parsed["stages"][k]
            branch=stage["inner_physical_branch"]
            counts.update(branch["switch_counts"])
            if branch["unmatched_internal_evaluations"]>0:
                bad.append((k,branch["unmatched_internal_evaluations"]))
            stages.append({
                "stage":k,
                "source_ad_vs_fd_scaled_first_error":
                    stage.get("max_scaled_local_first_error",0),
                "source_analytic_vs_fd_scaled_second_error":
                    stage.get("branch_local_second_vs_source_fd_scaled_max",0),
                "first_branch_change_ms":
                    branch["earliest_changed_fixed_tick_ms"],
                "events":branch["switch_counts"],
                "first_events":branch["changes"][:4],
                "original_internal_wheel_evaluations":
                    branch["nominal_internal_wheel_evaluations"]
            })
        summary[str(h)]={"switch_totals":dict(counts),
            "stages_with_any_switch":sum(bool(r["events"]) for r in stages),
            "max_scaled_first_error":max(r["source_ad_vs_fd_scaled_first_error"]
                for r in stages),
            "top_first_error_stages":sorted(stages,
                key=lambda r:r["source_ad_vs_fd_scaled_first_error"],
                reverse=True)[:6],
            "top_second_error_stages":sorted(stages,
                key=lambda r:r["source_analytic_vs_fd_scaled_second_error"],
                reverse=True)[:6],
            "stages":stages,
            "substep_configuration_changes":bad}
        print(case,h,"switches",dict(counts),
              "stages",summary[str(h)]["stages_with_any_switch"],
              "max_first",summary[str(h)]["max_scaled_first_error"],
              flush=True)
    output=folder/"source_30stage_real_qp_switch_audit_20261104.json"
    output.write_text(json.dumps({"case":case,"perturbations":summary},
                                 indent=2))
for case in sys.argv[1:] or list(CASES):
    one(case)
