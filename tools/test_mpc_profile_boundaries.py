"""Production map profiles remain completely separate, with protected practice weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "f1tenth_mpc/config"
IPO = ROOT / "f1tenth_planning/trajectories/autodrive_iros_2026_compete/ipopt_best_6p925_source_witness.csv"
# Development profile fingerprint; historical five-race baseline stays archived.
PRACTICE_HASH = "d09b7b1b8a20e8d8cc779496849dbaf05ce5d6423deb24759c5195d63b7feb47"
IPO_HASH = "59e39773f4b44b7768cf7e9130258f812ebe1984ab6b094462932dc8fe64b692"
WEIGHTS = (
    'weight_e_y', 'weight_e_psi', 'weight_u', 'weight_u_overspeed',
    'weight_longitudinal_slip', 'weight_v', 'weight_r',
    'weight_steering_command', 'weight_steering_target',
    'weight_throttle_command', 'weight_steering_slew', 'weight_throttle_slew',
)

def params(profile):
    return yaml.safe_load((CONFIG / profile).read_text())['/**']['ros__parameters']

def test_only_two_active_production_mpc_profiles():
    assert {x.name for x in CONFIG.glob('mpc_*.yaml')} == {
        'mpc_practice.yaml', 'mpc_competition.yaml'}

def test_current_unqualified_practice_fingerprint_preserved():
    p = params('mpc_practice.yaml')
    digest = hashlib.sha256(json.dumps(
        p, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == PRACTICE_HASH
    assert (p['weight_e_y'], p['weight_e_psi']) == (4500.0, 1450.0)
    assert (p['feedback_rollout'], p['feedback_rollout_gain_scale']) == (True, .25)
    assert p['viability_recovery_enabled']
    assert p['viability_proactive_after_fallback_only']
    assert p['end_to_end_cycle_budget_ms'] == 20.

def test_competition_weights_are_independent():
    practice, competition = params('mpc_practice.yaml'), params('mpc_competition.yaml')
    assert all(k in practice and k in competition for k in WEIGHTS)
    assert any(practice[k] != competition[k] for k in WEIGHTS)
    assert competition['max_speed_mps'] == 16.
    assert competition['corridor_margin_m'] == .05
    assert hashlib.sha256(IPO.read_bytes()).hexdigest() == IPO_HASH

def test_track_entrypoint_reports_both_canonical_inputs():
    r = subprocess.run(['bash', str(ROOT/'tools/run_mpc_trial.sh'), 'show'],
                       cwd=ROOT, text=True, capture_output=True, check=True)
    for needle in ('mpc_practice.yaml','mpc_competition.yaml',
                   'ipopt_best_6p925_source_witness.csv','NOT closed-loop qualified'):
        assert needle in r.stdout

def test_entrypoint_blocks_overlays():
    for track, key in [('practice', 'SDU_APEX_MPC_PARAMETER_OVERLAY'),
                       ('competition', 'SDU_APEX_IROS_MPC_PARAMETER_FILE')]:
        env = dict(os.environ)
        env[key] = '/tmp/forbidden-experiment.yaml'
        r = subprocess.run(['bash', str(ROOT/'tools/run_mpc_trial.sh'),
                            track, 'test_refusal'], env=env, cwd=ROOT,
                           text=True, capture_output=True)
        assert r.returncode == 2
        assert 'does not permit' in r.stderr


def test_public_entrypoint_does_not_allow_a_hidden_practice_speed_cap():
    env = dict(os.environ)
    env['SDU_APEX_CONTROLLER_MAX_SPEED'] = '5.0'
    r = subprocess.run(['bash', str(ROOT/'tools/run_mpc_trial.sh'),
                        'practice', 'do_not_run'], env=env, cwd=ROOT,
                       text=True, capture_output=True)
    assert r.returncode == 2
    assert 'do not permit' in r.stderr
