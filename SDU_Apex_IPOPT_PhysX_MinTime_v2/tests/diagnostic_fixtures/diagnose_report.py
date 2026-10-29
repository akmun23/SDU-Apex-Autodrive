#!/usr/bin/env python3
"""Read IPOPT solver diagnostics from a completed/failed saved JSON run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def summary(report:dict)->str:
    lines=[]
    solver=report.get('solver',{})
    diagnostics=report.get('nlp_diagnostics',{})
    lines.append('IPOPT status: '+str(solver.get('return_status', 'unknown')))
    lines.append('Feasible incumbent: '+str(solver.get('feasible_incumbent',False)))
    lines.append('Max constraint violation: '+str(diagnostics.get('max_constraint_violation')))
    stages=report.get('stages',report.get('stage_reports',[]))
    if isinstance(stages,list):
        for stage in stages:
            lines.append('Stage '+str(stage.get('name','unknown'))+': violation='+str(stage.get('max_constraint_violation')))
    worst=diagnostics.get('worst_constraints',[])
    if not worst:
        lines.append('No labeled final-iterate violations found in report.')
        initial=diagnostics.get('initial_guess',{})
        worst=initial.get('worst_constraints',[])
        if worst:lines.append('Showing INITIAL GUESS violations instead:')
    for entry in worst[:20]:
        lines.append('  '+str(entry.get('label','unknown'))+' : '+str(entry.get('violation','unknown')))
    if worst and worst[0].get('label')=='captured_motion_trust_region':
        lines.append('Largest violation is the numerical captured-motion trust-region cap, not a vehicle physics constraint.')
    return '\n'.join(lines)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('report',type=Path)
    args=ap.parse_args()
    print(summary(json.loads(args.report.read_text())))
