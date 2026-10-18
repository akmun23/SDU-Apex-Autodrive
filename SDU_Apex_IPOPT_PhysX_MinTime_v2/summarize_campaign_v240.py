#!/usr/bin/env python3
"""Summarize the entire Phase-I campaign without conflating runs or units."""
from pathlib import Path
import json
import math


def summarize(root: Path):
    data = []
    for folder in sorted(root.glob('v240_phase1_*')):
        if not folder.is_dir():
            continue
        reports = sorted(folder.glob('ipopt_report_pass_00_restart_*.json'))
        steps = []
        for p in reports:
            report = json.loads(p.read_text())
            stage = (report.get('solve_stages') or [{}])[0]
            val = stage.get('best_raw_max_defect')
            breakdown = stage.get('selected_dynamic_defect_breakdown') or []
            steps.append({
                'report': str(p.relative_to(root)),
                'initial_max_scaled_dynamic_defect': stage.get('initial_raw_max_defect'),
                'best_max_scaled_dynamic_defect': val,
                'elastic_merit': stage.get('best_elastic_merit'),
                'iterations': stage.get('ipopt_iterations'),
                'ipopt_return_status': stage.get('ipopt_status'),
                'strict_feasible': bool(stage.get('final_exact_physics_feasible')),
                'dominant_residual_family': breakdown[0]['group'] if breakdown else None,
                'top_residual_families': [x['group'] for x in breakdown[:4]],
                'hard_feasible_checkpoint': bool(report.get('phase1_restart_checkpoint')),
            })
        if steps:
            values = [s['best_max_scaled_dynamic_defect'] for s in steps]
            finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
            data.append({
                'variant': folder.name,
                'attempt_count': len(steps),
                'first_initial_defect': steps[0]['initial_max_scaled_dynamic_defect'],
                'best_defect_over_attempts': min(finite) if finite else None,
                'strict_feasible': any(s['strict_feasible'] for s in steps),
                'attempts': steps,
            })
    data.sort(key=lambda row: float('inf') if row['best_defect_over_attempts'] is None else row['best_defect_over_attempts'])
    result = {'diagnostic':'v2.4.0 Phase-I checkpoint-continuation campaign',
              'variants_ranked_by_best_scaled_defect':data,
              'exact_physics_lap_certified':any(x['strict_feasible'] for x in data),
              'native_replay_run':False,
              'important':'Defects combine differently scaled physical equations; only original strictly zero dynamic residual plus hard track constraints passes the IPOPT gate. Drive-only topology is not a valid racing strategy.'}
    (root/'v240_campaign_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# v2.4.0 Phase-I benchmark (drive-only diagnostic)', '',
           'Variant | IPOPT attempts | Initial max scaled defect | Best scaled defect | Strict feasible',
           '--- | ---: | ---: | ---: | ---']
    for x in data:
        lines.append(f"{x['variant']} | {x['attempt_count']} | {x['first_initial_defect']} | {x['best_defect_over_attempts']} | {x['strict_feasible']}")
    lines += ['', '## Interpretation',
              'No result is a minimum-time or native-validated racing lap. Phase I restarts reuse only hard-constraint-feasible diagnostic checkpoints with decreasing raw maximum physical defect. The exact-zero original-dynamics gate remains mandatory.',
              'If neither initialization achieves substantial progress, stop patching curve/coordinate signs and redesign Phase-I scaling, hard-constraint initialization and/or model/discretization systematically.']
    (root/'v240_campaign_summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return result


if __name__ == '__main__':
    summarize(Path(__file__).resolve().parent)
