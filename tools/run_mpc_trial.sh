#!/usr/bin/env bash
# One entry point; each track loads exactly one native MPC profile by map name.
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: tools/run_mpc_trial.sh {show|practice|competition} [RUN_ID]" >&2
  exit 2
fi
# Qualified track profiles are immutable at the public entrypoint. An
# environment speed override would silently change the practice race target.
if [[ "$1" != "show" && -n "$(printenv SDU_APEX_CONTROLLER_MAX_SPEED || true)" ]]; then
  echo 'Canonical track profiles do not permit SDU_APEX_CONTROLLER_MAX_SPEED overrides' >&2
  exit 2
fi
case "$1" in
  show)
    [[ $# -eq 1 ]] || { echo 'show accepts no RUN_ID' >&2; exit 2; }
    echo "Practice:    f1tenth_mpc/config/mpc_practice.yaml"
    echo "             f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800.csv"
    echo "             qualified 5 independent ten-lap races, mean 49.1622 s"
    echo "Competition: f1tenth_mpc/config/mpc_competition.yaml"
    echo "             f1tenth_planning/trajectories/autodrive_iros_2026_compete/ipopt_best_6p925_source_witness.csv"
    echo "             IPOPT source lap 6.925 s; NOT closed-loop qualified"
    echo "Optional overlays are for explicitly requested development experiments only."
    ;;
  practice)
    [[ $# -eq 2 ]] || { echo 'Practice requires unique RUN_ID' >&2; exit 2; }
    if [[ -n "$(printenv SDU_APEX_MPC_PARAMETER_OVERLAY || true)" ||
          -n "$(printenv SDU_APEX_IROS_MPC_PARAMETER_FILE || true)" ]]; then
      echo 'Canonical practice run does not permit MPC overlays. Use the low-level trial runner for deliberate A/B testing.' >&2
      exit 2
    fi
    exec bash tools/run_mpc_practice_trial.sh "$2"
    ;;
  competition)
    [[ $# -eq 2 ]] || { echo 'Competition requires unique RUN_ID' >&2; exit 2; }
    if [[ -n "$(printenv SDU_APEX_MPC_PARAMETER_OVERLAY || true)" ||
          -n "$(printenv SDU_APEX_IROS_MPC_PARAMETER_FILE || true)" ||
          -n "$(printenv SDU_APEX_IROS_TRAJECTORY_FILE || true)" ]]; then
      echo 'Canonical competition run does not permit overrides. Use the low-level trial runner for deliberate A/B testing.' >&2
      exit 2
    fi
    exec bash tools/run_iros_trial.sh "$2"
    ;;
  *)
    echo "Unknown track: $1. Choose practice or competition." >&2
    exit 2 ;;
esac
