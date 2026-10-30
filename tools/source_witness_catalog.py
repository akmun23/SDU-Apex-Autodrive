"""Read-only catalog of independently certified historical source racelines.

The current optimizer's directory may contain zero elites even when past
same-map workspaces or direct source certification trials contain excellent
alternative lines. This observer never runs physics or promotes files.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _positive_float(number):
    try:
        value = float(number)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _candidate_paths(live_runs):
    # Bounded, explicit source witness layouts; no recursive whole-disk search.
    patterns = (
        'optimizer_*/exports/*/report.json',
        'optimizer_*/starts/*/certified/report.json',
        'optimizer_*/starts/*/campaign/elite_*/*/certified_*/report.json',
        'iros_adaptive_preflight_*/source_*_certification/report.json',
        'iros_new_basin_test_*/automated_*/certified/report.json',
    )
    paths = []
    for pattern in patterns:
        paths.extend(live_runs.glob(pattern))
    return sorted(set(paths))[:600]


def dominates(a, b):
    """Pareto domination: time, clearance, steering and throttle smoothness."""
    # Never allow an invalid metric to masquerade as the best.
    return (a['lap_s'] <= b['lap_s'] + 1.e-9
            and a['clearance_slack_m'] >= b['clearance_slack_m'] - 1.e-7
            and a['steering_step_rms_rad'] <= b['steering_step_rms_rad'] + 1.e-7
            and a['throttle_step_rms'] <= b['throttle_step_rms'] + 1.e-7
            and (a['lap_s'] < b['lap_s'] - 1.e-9
                 or a['clearance_slack_m'] > b['clearance_slack_m'] + 1.e-7
                 or a['steering_step_rms_rad'] < b['steering_step_rms_rad'] - 1.e-7
                 or a['throttle_step_rms'] < b['throttle_step_rms'] - 1.e-7))


def certified_catalog(live_runs, *, source_sha256, margin=.05,
                      map_sha256=None, max_results=25):
    """Read verified source reports only; never mistake optimizer NLP output for certification.

    Other optimizer workspaces must have the exact same bundled image hash.
    Standalone preflight outputs have no bundled map and are displayed with
    source-model validation but with map provenance clearly marked unverified.
    """
    live_runs = Path(live_runs).resolve()
    if not live_runs.is_dir() or not source_sha256:
        return dict(candidates=[],pareto=[],total=0,provenance='unavailable')
    records = {}
    for path in _candidate_paths(live_runs):
        if not path.is_file() or not (path.parent/'witness.npz').is_file():
            continue
        try:
            j=json.loads(path.read_text())
            q=j.get('quality') or {}
            r=j.get('rollout') or {}
            if not (j.get('status')=='source_witness_passed'
                    and j.get('source_plant_sha256')==source_sha256
                    and abs(float(j.get('required_margin_m',-1))-margin)<1e-9
                    and r.get('source_tick_perimeter_validated') is True
                    and not r.get('violating_source_ticks')
                    and (_positive_float(r.get('full_lap_xy_max_m')) is not None)
                    and float(r['full_lap_xy_max_m'])<=.05):
                continue
            fields={
                'lap_s':_positive_float(j.get('trial_lap_s')),
                'clearance_slack_m':_positive_float(r.get('minimum_1ms_footprint_slack_m')),
                'steering_step_rms_rad':_positive_float(q.get('steering_step_rms_rad')),
                'throttle_step_rms':_positive_float(q.get('throttle_step_rms')),
                'steering_step_max_rad':_positive_float(q.get('steering_step_max_rad')),
            }
            if (any(value is None for value in fields.values())
                    or fields['clearance_slack_m']<0):
                continue
            # Never trust a path from inside the JSON; accept only structured paths.
            child=path.relative_to(live_runs).parts[0]
            if child.startswith('optimizer_') and map_sha256:
                candidate_map=live_runs/child/'input/map.pgm'
                if (not candidate_map.is_file()
                        or hashlib.sha256(candidate_map.read_bytes()).hexdigest()!=map_sha256):
                    continue
                same_map=True
            else:
                same_map=False
            witness=path.parent/'witness.npz'
            witness_id=hashlib.sha256(witness.read_bytes()).hexdigest()
            record=dict(**fields,id=witness_id[:16],
                        workspace=child,relative_report=str(path.relative_to(live_runs)),
                        source_only=True,independent_source_passed=True,
                        map_fingerprint_checked=same_map,unity_validated=False,
                        mpc_validated=False)
            previous=records.get(witness_id)
            # Prefer a map-fingerprint-checked report for the same witness.
            if previous is None or (same_map and not previous['map_fingerprint_checked']):
                records[witness_id]=record
        except (OSError,ValueError,TypeError,KeyError,UnicodeError):
            continue
    candidates=sorted(records.values(),key=lambda r:(r['lap_s'],-r['clearance_slack_m'],
                 r['steering_step_rms_rad'],r['id']))
    pareto=[a for a in candidates if not any(
        dominates(b,a) for b in candidates if b is not a)]
    pareto.sort(key=lambda r:(r['lap_s'],-r['clearance_slack_m']))
    # Expose explicit best-quality picks, independent of current optimizer.
    best_clearance=max(candidates,key=lambda x:x['clearance_slack_m']) if candidates else None
    smoothest=min(candidates,key=lambda x:x['steering_step_rms_rad']) if candidates else None
    result=dict(candidates=candidates[:max_results],pareto=pareto[:max_results],
                total=len(candidates),
                pareto_count=len(pareto),
                best_speed=candidates[0] if candidates else None,
                best_clearance=best_clearance,
                smoothest_steering=smoothest,
                provenance='cross-workspace independent source reports',
                unity_validated=False)
    return result
