#!/usr/bin/env python3
"""Create exact physical state and command-epoch replay from a controller rejection.

Preserves the complete 22-value source internal plant state and already queued
actuator commands, rather than guessing wheel/slip state from a bag pose.
Replay is an offline approximation of the live controller's QP warm memory,
not a complete recreation of its solver-state z/y/rho history.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def extract(log: Path, directory: Path) -> dict:
    records = []
    for line in log.read_text(errors='replace').splitlines():
        if 'MPC_CYCLE_REJECTED: ' not in line:
            continue
        payload = line.split('MPC_CYCLE_REJECTED: ', 1)[1]
        try:
            records.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    if not records:
        raise ValueError(f'no complete MPC_CYCLE_REJECTED diagnostic in {log}')
    failure = records[0]
    state = failure['state']
    internal = failure['plant_internal_state']
    nominal = failure['nominal_controls']
    if len(state) != 10 or len(internal) != 22 or len(nominal) != 30:
        raise ValueError('incomplete physical-source case or command horizon')
    command_epoch = failure.get('candidate_command_stamp_ns')
    if not isinstance(command_epoch, int) or command_epoch <= 0:
        raise ValueError('missing command epoch for command-time physical state')
    flags = failure.get('plant_state_flags', [])
    if len(flags) != 3 or not flags[0] or not flags[2]:
        raise ValueError('replay requires initialized source and IMU state flags')
    numbers = [
        command_epoch, failure['progress_m'],
        failure['active_speed_ceiling_mps'], *state,
        1.0 if flags[1] else 0.0,
        0.0, 0.0, *internal,
    ]
    assert len(numbers) == 38
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'physical_state.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(numbers)
    with (directory / 'nominal_source_seed.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        # Replay injection itself constructs [seed0,seed0,...seed28]
        # before the standard next-epoch shift. The CSV MUST contain the
        # requested SAME-epoch sequence, not an already-shifted copy.
        writer.writerow([0, *(c for pair in nominal for c in pair)])
    (directory / 'failure.json').write_text(
        json.dumps(failure, indent=2, sort_keys=True) + '\n'
    )
    return {'progress_m':failure['progress_m'],
            'stamp_ns':command_epoch,
            'source_stamp_ns':failure['source_stamp_ns'],
            'failure_reason': failure['nonlinear_failure_reason'],
            'failure_stage': failure['nonlinear_failure_stage'],
            'state_vector_length':len(state),
            'internal_state_length':len(internal)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('watcher_log',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(extract(args.watcher_log,args.output))


if __name__=='__main__':
    main()
