"""Byte-for-byte recovery of the historical IPOPT solver and its dependencies.

The historical copy is untouched; an optional numerical Hessian overlay uses a
separate working copy while keeping all vehicle dynamics constraints intact.
"""
from __future__ import annotations
import io
import subprocess
import sys
import tarfile
from pathlib import Path

REF = 'ca120e0cde7ae60e70082d38f86c5931f79f148f'
LEGACY = 'SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3'
PATHS = [
    LEGACY + '/f1tenth_planning/autodrive_mintime',
    'tools/vehicle_dynamics_learning/unity_vehicle_controller_source.py',
    'tools/vehicle_dynamics_learning/unity_vehicle_mechanics.py',
    'tools/vehicle_dynamics_learning/physx_no_drive_source.py',
    'tools/racing/mpc_vehicle_geometry.py',
]


def restore(repo: Path, destination: Path, allow_fetch: bool = False) -> Path:
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    if not (repo / '.git').exists():
        # Also accept a Git worktree whose .git is a file.
        if not (repo / '.git').is_file():
            raise FileNotFoundError('Repository root must contain Git history: ' + str(repo))
    if (destination / LEGACY / 'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py').exists() and (destination / 'tools/vehicle_dynamics_learning/unity_vehicle_mechanics.py').exists():
        return destination
    git = ['git', '-C', str(repo)]
    if subprocess.run(git + ['cat-file','-e',REF+'^{commit}'], capture_output=True).returncode:
        if not allow_fetch:
            raise RuntimeError('Historical solver commit missing from local clone. Use --allow-history-fetch to fetch it from GitHub.')
        subprocess.run(git + ['fetch','--no-tags','origin',REF], check=True)
    result = subprocess.run(git + ['archive','--format=tar',REF,*PATHS], capture_output=True)
    if result.returncode:
        raise RuntimeError('Git archive failed: ' + result.stderr.decode(errors='replace'))
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as tf:
        dest=destination.resolve()
        for member in tf.getmembers():
            resolved=(dest / member.name).resolve()
            if dest not in resolved.parents and resolved != dest:
                raise RuntimeError('Unexpected archive path traversal')
            if member.issym() or member.islnk():
                raise RuntimeError('Unexpected symlink in historical archive')
        tf.extractall(destination, filter='data')
    return destination


def load_modules(repo: Path, package_root: Path, allow_fetch: bool = False,
                 phase1_hessian: str = 'limited-memory',
                 phase1_formulation: str = 'elastic'):
    recovered = restore(repo, package_root/'_restored_git_history', allow_fetch)
    if phase1_formulation == 'eliminated':
        from solver_overlay_eliminated_v243 import prepare_working_solver_eliminated
        root, provenance = prepare_working_solver_eliminated(
            recovered, package_root / ('_working_solver_eliminated_' +
                phase1_hessian.replace('-', '_')), LEGACY, phase1_hessian)
    elif phase1_formulation == 'condensed':
        from solver_overlay_condensed_v242 import prepare_working_solver_condensed
        root, provenance = prepare_working_solver_condensed(
            recovered, package_root / ('_working_solver_condensed_' +
                phase1_hessian.replace('-', '_')), LEGACY, phase1_hessian)
    elif phase1_formulation == 'elastic':
        from solver_overlay import prepare_working_solver
        root, provenance = prepare_working_solver(
            recovered, package_root / ('_working_solver_' +
                phase1_hessian.replace('-', '_')), LEGACY, phase1_hessian)
    else:
        raise ValueError('Unknown Phase-I formulation')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib
    base=LEGACY+'.f1tenth_planning.autodrive_mintime.'
    o=importlib.import_module(base+'optimizer_unity_wheels')
    m=importlib.import_module(base+'unity_wheel_model')
    t=importlib.import_module(base+'track')
    return o,m,t,root,provenance
