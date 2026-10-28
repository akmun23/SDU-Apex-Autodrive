#!/usr/bin/env python3
"""Read-only terminal dashboard for an unattended IPOPT output directory."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def observed_state(recorded_state, service_state):
    """A stale 'running' JSON must not hide a stopped service after power loss."""
    if recorded_state == 'running' and service_state in {
            'inactive', 'failed', 'unknown', 'not-found'}:
        return f'interrupted / service {service_state} (saved status was running)'
    return f'{recorded_state} / service {service_state}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--unit', help='Optional systemd user service to inspect')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    log = root / 'console.log'
    offset, pending, stage, row, last_result = 0, '', None, None, None
    durable_checkpoint = None
    progress_message = None
    try:
        while True:
            if log.exists():
                if log.stat().st_size < offset:
                    offset, pending, stage, row, last_result = 0, '', None, None, None
                with log.open('rb') as stream:
                    stream.seek(offset)
                    pending += stream.read().decode('utf-8', errors='replace')
                    offset = stream.tell()
                lines = pending.split('\n')
                pending = lines.pop()
                for line in lines:
                    if line.startswith(('Symbolic mesh pass ', 'Phase-I ',
                                        'IPOPT time-continuation solver setup',
                                        'IPOPT exporting audited candidate:')):
                        progress_message = line[:180]
                    match = re.search(r'IPOPT time stage trust_factor=([\d.eE+-]+)', line)
                    if match:
                        stage, row = float(match[1]), None
                        durable_checkpoint = None
                    fields = line.split()
                    if len(fields) >= 4 and re.fullmatch(r'\d+r?', fields[0]):
                        try:
                            row = (fields[0], *(float(x) for x in fields[1:4]))
                            progress_message = 'Numerical time minimization'
                        except ValueError:
                            pass
                    if line.startswith('IPOPT time-stage result: '):
                        try:
                            last_result = json.loads(line.split(': ', 1)[1])
                        except ValueError:
                            pass
                    if line.startswith('IPOPT durable time checkpoint: '):
                        try:
                            durable_checkpoint = json.loads(line.split(': ', 1)[1])
                        except ValueError:
                            pass
            try:
                status = json.loads((root / 'status.json').read_text())
            except (OSError, ValueError):
                status = {}
            state = status.get('state', 'unknown')
            if args.unit:
                try:
                    result = subprocess.run(
                        ['systemctl', '--user', 'is-active', args.unit],
                        capture_output=True, text=True, timeout=2)
                    state = observed_state(
                        state, result.stdout.strip() or 'unknown')
                except (OSError, subprocess.TimeoutExpired):
                    state += ' / service unavailable'
            factors = status.get('trust_region_factors', [])
            number = factors.index(stage) + 1 if stage in factors else '?'
            checkpoints = sorted(root.glob('time_stage_iterates/*.npz'),
                                 key=lambda path: path.stat().st_mtime)
            output = [f'IPOPT   {dt.datetime.now().astimezone():%H:%M:%S}   {state}',
                      f'Run:    {root.name}',
                      f'Stage:  {number}/{len(factors) or "?"}  trust factor {stage:g}'
                      if stage is not None else 'Stage:  initialization / feasibility']
            if row:
                output += [f'Iter:   {row[0]} / {status.get("iterations_per_time_stage", "?")}',
                           f'Lap:    {row[1]:.6f} s  (current iterate)',
                           f'Constraint residual: {row[2]:.3g}  (limit 0.001)',
                           f'Optimality residual: {row[3]:.3g}  (lower is better)']
            elif stage is not None:
                output.append('Phase: solver initialization / derivative compilation; no iterations yet')
            if progress_message:
                output.append(f'Progress: {progress_message}')
            if last_result:
                output.append(f'Last stage: {last_result.get("return_status", "unknown")}')
            if durable_checkpoint:
                best = durable_checkpoint.get('best_sampled_feasible_lap_time_s')
                if best is not None:
                    output.append(f'Best saved feasible sample: {best:.6f} s')
                output.append('Durable save: iteration '
                              + str(durable_checkpoint.get('iteration', '?')))
            if checkpoints:
                latest = checkpoints[-1]
                stamp = dt.datetime.fromtimestamp(latest.stat().st_mtime).astimezone()
                output += [f'Checkpoints: {len(checkpoints)}',
                           f'Latest: {latest.name}  [{stamp:%H:%M:%S}]']
            else:
                output.append('Checkpoints: none yet')
            if log.exists():
                output.append(f'Log updated: {max(0, time.time()-log.stat().st_mtime):.0f} s ago')
            output.append('Current iterate is not a validated lap. Ctrl+C stops viewer only.')
            print(('\033[2J\033[H' if sys.stdout.isatty() and not args.once else '')
                  + '\n'.join(output), flush=True)
            if args.once:
                return
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
