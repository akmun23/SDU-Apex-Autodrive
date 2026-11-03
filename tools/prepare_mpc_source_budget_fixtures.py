#!/usr/bin/env python3
"""Prepare two no-simulator source-SQP cold-start fixtures from a ROS bag report.

Requires --diagnostics exported by tools/export_mpc_run_report.py. The
fixture generator retains all 22 hidden source state channels, command FIFO,
exact practice raceline and ROS tuning. It uses an existing same-build captured
source fixture only as a source-model baseline and solver-config baseline.
"""
import argparse, json, pathlib, subprocess
import yaml

ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics",required=True,type=pathlib.Path,
                   help="JSONL output *_diagnostics.jsonl from export_mpc_run_report.py")
    p.add_argument("--parameters",required=True,type=pathlib.Path,
                   help="exact mpc_parameters.yaml captured with failed Unity run")
    p.add_argument("--trajectory",required=True,type=pathlib.Path,
                   help="the exact practice/competition raceline CSV from run manifest")
    p.add_argument("--base-case",required=True,type=pathlib.Path,
                   help="same-build native source_context.mctx + r1.fqp fixture")
    p.add_argument("--out",required=True,type=pathlib.Path)
    p.add_argument("--indices",nargs=2,type=int,default=[20,1000])
    p.add_argument("--builder",type=pathlib.Path,default=
                   ROOT/"build/f1tenth_mpc/mpc_source_live_fixture")
    args=p.parse_args()
    params=yaml.safe_load(args.parameters.read_text())
    if not isinstance(params,dict):p.error("invalid ROS YAML")
    config=next((v["ros__parameters"] for k,v in params.items()
                  if k in ("/**","/mpc_controller_node","mpc_controller_node")
                  and isinstance(v,dict) and isinstance(v.get("ros__parameters"),dict)),None)
    if config is None:p.error("no matching MPC ROS parameters")
    args.out.mkdir(parents=True,exist_ok=True)
    model=args.out/"captured_parameters.json"
    model.write_text(json.dumps(config,indent=2))
    rows=[]
    with args.diagnostics.open() as stream:
        for line in stream:
            record=json.loads(line)
            obj=record.get("payload",record)
            if (obj.get("status")=="waiting_for_solver_budget"
                and obj.get("plant_state_flags")==[1,1,1]
                and len(obj.get("plant_internal_state",[]))==22
                and len(obj.get("nominal_controls",[]))==30):
                rows.append(obj)
    if len(rows)<=max(args.indices):
        p.error(f"only {len(rows)} source-complete startup records found")
    summary=[]
    for name,index in zip(("cold_early","cold_stable"),args.indices):
        sample=rows[index]
        destination=args.out/name
        destination.mkdir(exist_ok=True)
        sample_path=destination/"live_diagnostic.json"
        sample_path.write_text(json.dumps(sample,indent=2))
        cmd=[str(args.builder),str(sample_path),str(model),
             str(args.trajectory),str(args.base_case),str(destination)]
        subprocess.run(cmd,check=True,cwd=ROOT)
        summary.append({"name":name,"index":index,"source_stamp_ns":
                        sample.get("source_stamp_ns"),"speed_mps":sample["state"][2],
                        "progress_m":sample["progress_m"],"fixture":str(destination)})
    (args.out/"manifest.json").write_text(json.dumps({
        "experiment_regime":"simulator_free_reconstructed_live_cold_start",
        "limits":["Source model/track/tuning from user-provided run; base QP solver config from captured source case",
                  "Source-only replay cannot prove Unity prediction fidelity or ROS command publication"],
        "cases":summary},indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":
    main()
