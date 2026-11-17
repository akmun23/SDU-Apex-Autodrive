#!/usr/bin/env bash
# Local graphical tuner; reaps only containers launched by this session.
set -Eeuo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export SDU_APEX_TUNER_SESSION="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
controller=sdu_apex_autodrive_competition
sim=sdu_apex_sim_iros_gui
child_pid=0
cleanup() {
  trap - EXIT INT TERM HUP
  if (( child_pid > 0 )); then kill "$child_pid" >/dev/null 2>&1 || true; fi
  for name in "$sim" "$controller"; do
    owner="$(docker inspect -f '{{index .Config.Labels "sdu.apex.tuner.session"}}' "$name" 2>/dev/null || true)"
    if [[ "$owner" == "$SDU_APEX_TUNER_SESSION" ]]; then
      docker stop --time 3 "$name" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
cd "$repo"
python3 "$repo/tools/tuner/competition_tuner.py" "$@" &
child_pid=$!
wait "$child_pid"
