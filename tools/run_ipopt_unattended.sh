#!/usr/bin/env bash
# Detached optimization only: never launch Unity or promote a runtime path.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_root"
[[ $# -ge 2 ]] || {
  echo 'Usage: run_ipopt_unattended.sh OUTPUT_DIR SEED.npz [--all-day] [--trust-factors LIST] [--iterations N]' >&2; exit 2;
}
output="$(realpath -m -- "$1")"
seed="$(realpath -e -- "$2")"
shift 2
all_day=0
requested_factors=
requested_iterations=
while (( $# )); do
  case "$1" in
    --all-day) all_day=1; shift ;;
    --trust-factors)
      [[ $# -ge 2 ]] || exit 2
      requested_factors="$2"
      [[ -n "$requested_factors" ]] || exit 2
      shift 2 ;;
    --iterations)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || exit 2
      requested_iterations="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$requested_factors" ]]; then
  python3 - "$requested_factors" <<'PY'
import math,sys
try:
    factors=[float(x) for x in sys.argv[1].split(',')]
    assert len(factors) >= 2
    assert all(math.isfinite(x) and x > 0 for x in factors)
    assert all(a < b for a,b in zip(factors,factors[1:]))
except (ValueError,AssertionError):
    sys.exit('Provide at least two finite, positive, strictly increasing trust factors')
PY
fi
[[ "$output" == "$repo_root"/optimized_practice_* && -f "$seed" ]] || exit 2
[[ ! -e "$output/status.json" && ! -e "$output/console.log" ]] || exit 2
mkdir -p "$output"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SDU_IPOPT_STACK_DUMP=1
ipopt_cpu="${SDU_IPOPT_CPU:-15}"
[[ "$ipopt_cpu" =~ ^[0-9]+$ ]] || exit 2
export SDU_IPOPT_STAGE_CAPTURE_DIR="$output/time_stage_iterates"
# Native IPOPT termination returns normally through selection/export code.
# The outer wall guard is an emergency bound, not a convergence claim.
mode=bounded
iterations=1200
factors=64,256,1024
wall_guard=11700
export SDU_IPOPT_PHASE1_MAX_CPU_SECONDS=600
export SDU_IPOPT_TIME_MAX_CPU_SECONDS=2400
export SDU_IPOPT_TIME_MAX_WALL_SECONDS=2700
if (( all_day )); then
  mode=all_day
  iterations=6000
  factors=2,4,16,64,256,1024
  wall_guard=0
  export SDU_IPOPT_PHASE1_MAX_CPU_SECONDS=0
  export SDU_IPOPT_TIME_MAX_CPU_SECONDS=0
  export SDU_IPOPT_TIME_MAX_WALL_SECONDS=0
fi
if [[ -n "$requested_factors" ]]; then
  factors="$requested_factors"
fi
if [[ -n "$requested_iterations" ]]; then
  iterations="$requested_iterations"
fi
write_status() {
  python3 - "$output" "$1" "$2" "$seed" "$mode" "$wall_guard" "$iterations" "$factors" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); state=sys.argv[2]; code=int(sys.argv[3])
data={'state':state,'exit_code':None if state=='running' else code,
      'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'mode':sys.argv[5],
      'hard_wall_guard_seconds':int(sys.argv[6]) or None,
      'phase1_cpu_limit_seconds':float(os.environ['SDU_IPOPT_PHASE1_MAX_CPU_SECONDS']) or None,
      'time_stage_cpu_limit_seconds':float(os.environ['SDU_IPOPT_TIME_MAX_CPU_SECONDS']) or None,
      'time_stage_wall_limit_seconds':float(os.environ['SDU_IPOPT_TIME_MAX_WALL_SECONDS']) or None,
      'iterations_per_time_stage':int(sys.argv[7]),
      'trust_region_factors':[float(x) for x in sys.argv[8].split(',')],
      'requested_cpu':int(os.environ.get('SDU_IPOPT_CPU', '15')),
      'variable_scaling':os.environ.get('SDU_IPOPT_VARIABLE_SCALING', '0') == '1',
      'checkpoint_stride':int(os.environ.get('SDU_IPOPT_TIME_CHECKPOINT_STRIDE', '25')),
      'checkpoint_interval_seconds':float(os.environ.get('SDU_IPOPT_TIME_CHECKPOINT_SECONDS', '120')),
      'seed_path':sys.argv[4],
      'seed_sha256':hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
      'runtime_trajectory_promoted':False,'unity_validated':False,
      'saved_stage_checkpoints':[str(x) for x in sorted(p.glob('time_stage_iterates/*.npz'))]}
report=p/'report.json'
if report.exists():
 try:data['optimizer_report']=json.loads(report.read_text())
 except (OSError,ValueError):data['report_read_error']=True
tmp=p/'status.json.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,p/'status.json')
PY
}
finalized=0
finish_interrupted() {
  local code=$?
  trap - EXIT
  if [[ "$finalized" == 0 ]]; then
    write_status interrupted "$code"
    sync -f "$output/status.json"
  fi
}
trap finish_interrupted EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
write_status running 0
cp -- "$0" "$output/launcher.sh"
command_prefix=()
if (( wall_guard > 0 )); then
  command_prefix=(timeout --signal=TERM --kill-after=30s "${wall_guard}s")
fi
set +e
"${command_prefix[@]}" \
  taskset -c "$ipopt_cpu" nice -n 10 \
  SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python -u \
  SDU_Apex_IPOPT_PhysX_MinTime_v2/run_ipopt.py \
  --repo-root "$repo_root" \
  --map f1tenth_planning/maps/autodrive_practice_20260924_b.pgm \
  --centerline f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv \
  --optimizer-seed-file "$seed" \
  --mesh-spacing 0.06 --max-speed 16 --wall-margin 0.05 \
  --phase1-formulation source-equalities --phase1-hessian exact \
  --time-hessian-approximation exact --seed-backend source --skip-native \
  --feasibility-iterations 200 --iterations "$iterations" \
  --phase1-max-cpu-seconds "$SDU_IPOPT_PHASE1_MAX_CPU_SECONDS" \
  --time-max-cpu-seconds "$SDU_IPOPT_TIME_MAX_CPU_SECONDS" \
  --time-trust-region-factors "$factors" --print-level 5 \
  --output "$output" >"$output/console.log" 2>&1
result=$?
set -e
state=failed
case "$result" in
  0|3) state=finished ;;
  2) state=failed ;;
  124|137) state=wall_guard_stopped ;;
esac
write_status "$state" "$result"
sync -f "$output/status.json"
finalized=1
echo "IPOPT unattended: $state (exit=$result); $output/status.json"
[[ "$result" == 0 || "$result" == 3 || "$result" == 2 ]]
