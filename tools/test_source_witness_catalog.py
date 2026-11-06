"""Historical source certificate index: no solver, no implicit promotion."""
import hashlib
import json
from pathlib import Path

from tools.source_witness_catalog import certified_catalog, dominates


def certificate(path, source, lap, clearance, smooth, throttle, *, status='source_witness_passed', map_path=None):
    path.mkdir(parents=True,exist_ok=True)
    (path/'witness.npz').write_bytes(f'{lap}:{clearance}:{smooth}'.encode())
    report=dict(status=status,source_plant_sha256=source,required_margin_m=.05,
        trial_lap_s=lap,quality=dict(clearance_slack_m=clearance,
        steering_step_rms_rad=smooth, steering_step_max_rad=.1,throttle_step_rms=throttle),
        rollout=dict(source_tick_perimeter_validated=True,violating_source_ticks=[],
        full_lap_xy_max_m=.00001,minimum_1ms_footprint_slack_m=clearance))
    (path/'report.json').write_text(json.dumps(report))


def test_catalog_finds_same_time_clearance_and_faster_tight_line(tmp_path):
    map_path=tmp_path/'optimizer_one/input/map.pgm'
    map_path.parent.mkdir(parents=True);map_path.write_bytes(b'map')
    second=tmp_path/'optimizer_two/input/map.pgm'
    second.parent.mkdir(parents=True);second.write_bytes(b'map')
    certificate(tmp_path/'optimizer_one/exports/fast','source',7.075,.000001,.039,.065)
    certificate(tmp_path/'optimizer_two/exports/quality','source',7.1,.017,.043,.068)
    certificate(tmp_path/'optimizer_two/exports/repeated','wrong-source',6.0,.05,.01,.01)
    certificate(tmp_path/'optimizer_one/exports/failed','source',6.9,.04,.035,.055,
                status='no_validated_witness')
    result=certified_catalog(tmp_path,source_sha256='source',
        map_sha256=hashlib.sha256(b'map').hexdigest())
    assert result['total']==2
    assert result['pareto_count']==2
    assert result['best_speed']['lap_s']==7.075
    assert result['best_clearance']['clearance_slack_m']==.017
    assert all(v['map_fingerprint_checked'] for v in result['pareto'])
    assert all(v['source_only'] and not v['unity_validated'] for v in result['pareto'])


def test_catalog_rejects_different_maps_and_incomplete_or_invalid_witnesses(tmp_path):
    p=tmp_path/'optimizer_one/input/map.pgm';p.parent.mkdir(parents=True)
    p.write_bytes(b'other')
    certificate(tmp_path/'optimizer_one/exports/badmap','source',7.0,.005,.03,.02)
    p=tmp_path/'optimizer_two/input/map.pgm';p.parent.mkdir(parents=True)
    p.write_bytes(b'map')
    path=tmp_path/'optimizer_two/exports/invalid'
    certificate(path,'source',7.0,-.001,.04,.05)
    result=certified_catalog(tmp_path,source_sha256='source',
       map_sha256=hashlib.sha256(b'map').hexdigest())
    assert result['total']==0


def test_pareto_does_not_treat_same_time_quality_gain_as_speed_gain():
    faster=dict(lap_s=7.075,clearance_slack_m=.000001,
                steering_step_rms_rad=.039,throttle_step_rms=.063)
    cleaner=dict(lap_s=7.1,clearance_slack_m=.017,
                 steering_step_rms_rad=.044,throttle_step_rms=.069)
    assert not dominates(faster,cleaner)
    assert not dominates(cleaner,faster)
    worse=dict(faster,lap_s=7.2,clearance_slack_m=0.)
    assert dominates(faster,worse)


def test_catalog_deduplicates_identical_witness_export_bytes(tmp_path):
    for name in ['one','two']:
        p=tmp_path/f'optimizer_{name}/input/map.pgm'
        p.parent.mkdir(parents=True);p.write_bytes(b'map')
        certificate(tmp_path/f'optimizer_{name}/exports/identical',
                    'source',7.1,.015,.04,.065)
    result=certified_catalog(tmp_path,source_sha256='source',
        map_sha256=hashlib.sha256(b'map').hexdigest())
    assert result['total']==1


def test_catalog_retains_tolerance_accepted_certificate_without_clamping_raw_value(tmp_path):
    path=tmp_path/'optimizer_one/exports/tolerated'
    certificate(path,'source',7.,-5.e-7,.04,.06)
    report_path=path/'report.json'
    report=json.loads(report_path.read_text())
    # Negative unannotated certificates remain invalid.
    assert certified_catalog(tmp_path,source_sha256='source')['total']==0
    report['clearance_numerical_tolerance_m']=1.e-6
    report_path.write_text(json.dumps(report))
    result=certified_catalog(tmp_path,source_sha256='source')
    assert result['total']==1
    assert result['best_speed']['clearance_slack_m']==-5.e-7
    report['clearance_numerical_tolerance_m']=.01
    report_path.write_text(json.dumps(report))
    assert certified_catalog(tmp_path,source_sha256='source')['total']==0
