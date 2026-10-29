"""Map-only orchestration tests; mocks below do NOT certify vehicle accuracy."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import track_optimizer as optimizer
from command_ocp import save_command_checkpoint


def input_map(tmp_path):
    image, metadata = tmp_path/'new_track.pgm', tmp_path/'new_track.yaml'
    image.write_bytes(b'P5\n2 2\n255\n'+bytes([0, 255, 255, 0]))
    config = dict(image=image.name, resolution=.025, origin=[10., -8., .2],
                  negate=0, occupied_thresh=.65, free_thresh=.25)
    metadata.write_text(yaml.safe_dump(config))
    return image, metadata, config


def witness(path, count=40):
    x = np.zeros((21, count+1))
    angle = np.linspace(0., -2*np.pi, count+1)
    x[0], x[1], x[2] = np.cos(angle), np.sin(angle), angle
    x[3] = 1.
    u = np.vstack([np.full(count, .1), np.full(count, .05)])
    save_command_checkpoint(path, x, u, 'fixture', 0., 0)


def certification(repo, path, map_path, destination):
    destination.mkdir()
    with np.load(path) as data:
        count = data['controls'].shape[1]
    witness(destination/'witness.npz', count)
    (destination/'raceline.csv').write_text('# fixture only\n0,0,0\n')
    report = dict(status='source_witness_passed', source_plant_sha256='fixture',
                  trial_lap_s=count*.025,
                  reference_export=dict(path=str(destination/'raceline.csv')))
    (destination/'report.json').write_text(json.dumps(report))
    return report


def test_map_pair_and_yaml_only_preserve_metric_origin(tmp_path):
    image, metadata, config = input_map(tmp_path)
    assert optimizer.resolve_map(image, metadata) == (image, metadata, config)
    assert optimizer.resolve_map(metadata) == (image, metadata, config)
    other = tmp_path/'wrong.pgm'
    other.write_bytes(b'different track')
    with pytest.raises(ValueError, match='does not match'):
        optimizer.resolve_map(other, metadata)
    config['resolution'] = float('nan')
    metadata.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='resolution'):
        optimizer.resolve_map(image, metadata)


def test_output_identity_depends_on_image_and_metric_metadata(tmp_path):
    image, _, config = input_map(tmp_path)
    original = optimizer.default_output(image, config)
    assert optimizer.default_output(image, config) == original
    assert optimizer.default_output(image, dict(config, resolution=.05)) != original
    image.write_bytes(b'another circuit')
    assert optimizer.default_output(image, config) != original


def test_orphaned_input_bundle_cannot_be_reused_for_another_map(tmp_path):
    image, metadata, config = input_map(tmp_path)
    assets = tmp_path/'input'
    optimizer.bundle_map(image, config, assets)
    optimizer.bundle_map(image, config, assets)  # same input is idempotent
    with pytest.raises(ValueError, match='different map metadata'):
        optimizer.bundle_map(image, dict(config, resolution=.05), assets)
    # This is precisely the crash-before-optimizer.json / reused-output case.
    image.write_bytes(b'P5\n2 2\n255\n'+bytes([0, 0, 255, 0]))
    with pytest.raises(ValueError, match='different map image'):
        optimizer.bundle_map(image, config, assets)


def test_implementation_guard_includes_source_adapter_and_time_helpers(monkeypatch):
    original = Path.read_bytes
    for suffix in ('physics.py', 'path_frame.py', 'command_realizability.py'):
        before = optimizer.implementation_digest()
        with monkeypatch.context() as context:
            context.setattr(Path, 'read_bytes', lambda path: original(path)+(
                b'\n# modified numerical behavior\n' if path.name == suffix else b''))
            assert optimizer.implementation_digest() != before


def test_atomic_export_never_replaces_best_with_slower_or_uncertified(tmp_path):
    source = tmp_path/'certified'
    source.mkdir()
    witness(source/'seed.npz')
    result = tmp_path/'checked'
    certification(tmp_path, source/'seed.npz', tmp_path/'map', result)
    state = dict(best_lap_s=None)
    optimizer.publish_best(tmp_path, result/'witness.npz', 1., state, source_digest='fixture')
    first = (tmp_path/'best.json').read_bytes()
    optimizer.publish_best(tmp_path, result/'witness.npz', 2., state, source_digest='fixture')
    assert (tmp_path/'best.json').read_bytes() == first
    with pytest.raises(ValueError, match='stale-source'):
        optimizer.publish_best(tmp_path, result/'witness.npz', .9, state, source_digest='changed')
    assert (tmp_path/'best.json').read_bytes() == first


def test_verified_unity_reference_orients_optimizer_to_driven_laps():
    ref = optimizer.REPO/'live_runs/iros_mpc_initial_20261029_02/raceline.csv'
    if not ref.is_file():
        pytest.skip('recorded successful IROS reference not mounted')
    direction=optimizer.verified_unity_direction(ref)
    assert direction['winding'] == 1
    assert direction['signed_area_m2'] > 0
    assert direction['geometry_orientation'] != 'clockwise'


def test_opposite_winding_witness_rejected_by_simulator_direction_gate(tmp_path):
    witness(tmp_path/'opposite.npz')
    with pytest.raises(ValueError, match='opposite to verified Unity'):
        optimizer.require_travel_orientation(tmp_path/'opposite.npz', 1, verified=True)


def test_positive_winding_witness_accepted_by_simulator_direction_gate(tmp_path):
    witness(tmp_path/'matching.npz')
    with np.load(tmp_path/'matching.npz', allow_pickle=False) as data:
        state, controls = data['state'].copy(), data['controls'].copy()
    state[1] *= -1
    state[2] *= -1
    save_command_checkpoint(tmp_path/'matching.npz', state, controls, 'test-only', 0., 0)
    optimizer.require_travel_orientation(tmp_path/'matching.npz', 1, verified=True)


def test_counterclockwise_is_rejected_even_with_a_source_certificate(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    witness(source/'seed.npz')
    destination = tmp_path/'certified'
    certification(tmp_path, source/'seed.npz', tmp_path/'map', destination)
    path = destination/'witness.npz'
    with np.load(path) as data:
        x, u = data['state'].copy(), data['controls'].copy()
    x[1] *= -1.
    x[2] *= -1.
    save_command_checkpoint(path, x, u, 'fixture', 0., 0)
    with pytest.raises(ValueError, match='only clockwise'):
        optimizer.publish_best(tmp_path, path, 1., dict(best_lap_s=None), source_digest='fixture')
    assert not (tmp_path/'best.json').exists()


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    image, metadata, config = input_map(tmp_path)
    reference = tmp_path/'generated_centerline.csv'
    reference.write_text('fixture; no physics use')
    monkeypatch.setattr(optimizer, 'REPO', tmp_path)
    monkeypatch.setattr(optimizer, 'production_command_map', lambda *a: (None, 'fixture'))
    def starts(map_yaml, output, directions, cancel):
        assert directions == ('clockwise',)
        return [('map_basin', reference)]
    monkeypatch.setattr(optimizer, 'geometry_starts', starts)
    monkeypatch.setattr(optimizer, 'geometric_seed', lambda ref, path, repo, scale: witness(path))
    monkeypatch.setattr(optimizer, 'certify_witness', certification)
    calls = []
    def solve(**kw):
        calls.append(kw)
        kw['output'].mkdir()
        witness(kw['output']/'witness.npz', kw['count'])
        return dict(status='source_witness_passed', max_constraint_violation=0.,
                    rollout=dict(violating_source_ticks=[]))
    monkeypatch.setattr(optimizer, 'solve_witness', solve)
    def campaign(**kw):
        assert kw['target_seconds'] is None
        assert kw['coarse_fraction'] == .1
        assert kw['seed'].is_file()
        return dict(status='plateau')
    monkeypatch.setattr(optimizer, 'run_campaign', campaign)
    return dict(image=image, metadata=metadata, config=config, output=tmp_path/'out'), calls


def test_map_only_full_workflow_needs_no_bags_reference_or_time_target(pipeline):
    settings, calls = pipeline
    state = optimizer.run_track(**settings, workers=1)
    assert state['status'] == 'plateau' and state['best_lap_s'] == 1.
    assert len(calls) == 1
    assert calls[0]['cpu_seconds'] == 1.e20
    bundled = json.loads((settings['output']/'input/map.yaml').read_text())
    assert bundled['origin'] == [10., -8., .2]
    assert (settings['output']/'best.json').is_file()
    assert not state['global_minimum_proven'] and not state['runtime_promoted']
    # Restart reuses completed bootstrap, without running it again.
    optimizer.run_track(**settings, workers=1)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='settings changed'):
        optimizer.run_track(**settings, workers=2)


def test_larger_budgets_do_not_repeat_a_verified_bootstrap(pipeline):
    settings, calls = pipeline
    optimizer.run_track(**settings, workers=1)
    state = optimizer.run_track(**settings, workers=1, iterations=2500, stagnation_iterations=500)
    assert len(calls) == 1
    assert state['contract']['iterations'] == 2500
    assert state['contract']['stagnation_iterations'] == 500
    assert len(state['budget_updates']) == 1
    with pytest.raises(ValueError, match='settings changed'):
        optimizer.run_track(**settings, workers=1, iterations=1000, stagnation_iterations=100)


def test_explicit_search_code_migration_preserves_map_source_and_seed(pipeline, monkeypatch):
    settings,calls=pipeline
    optimizer.run_track(**settings,workers=1,iterations=900,stagnation_iterations=180)
    previous=json.loads((settings['output']/'optimizer.json').read_text())
    old_digest=previous['contract']['optimizer_sha256']
    monkeypatch.setattr(optimizer,'implementation_digest',lambda: 'search-upgrade-test-only')
    with pytest.raises(ValueError,match='explicitly approve'):
        optimizer.run_track(**settings,workers=1,iterations=1200,stagnation_iterations=300)
    assert json.loads((settings['output']/'optimizer.json').read_text())['contract']==previous['contract']
    result=optimizer.run_track(**settings,workers=1,
        iterations=1200,stagnation_iterations=300,
        allow_search_code_update=True)
    assert result['contract']['iterations']==1200
    assert result['contract']['stagnation_iterations']==300
    assert result['search_code_migrations'][0]['previous_optimizer_sha256']==old_digest
    assert len(calls)==1  # A code upgrade cannot discard previous certified result.
    with pytest.raises(ValueError,match='settings changed'):
        optimizer.run_track(**settings,workers=2,
            iterations=1200,stagnation_iterations=300,
            allow_search_code_update=True)


def test_search_code_migration_does_not_allow_source_or_budget_regression():
    old=dict(optimizer_sha256='a',source_sha256='source',
             image_sha256='map',command_seed_sha256='seed',
             iterations=900,stagnation_iterations=180)
    new=dict(old,optimizer_sha256='b',iterations=1200,stagnation_iterations=300)
    assert optimizer.approved_search_code_update(old,new)
    assert not optimizer.approved_search_code_update(old,dict(new,source_sha256='changed'))
    assert not optimizer.approved_search_code_update(old,dict(new,image_sha256='changed'))
    assert not optimizer.approved_search_code_update(old,dict(new,command_seed_sha256='changed'))
    assert not optimizer.approved_search_code_update(old,dict(new,iterations=700))
    assert not optimizer.approved_search_code_update(old,dict(old,iterations=1200,
        stagnation_iterations=300))


def test_interrupted_bootstrap_resumes_its_durable_best(pipeline, monkeypatch):
    settings, calls = pipeline
    def interrupt(**kw):
        kw['output'].mkdir()
        witness(kw['output']/'best_iterate.npz', kw['count'])
        raise KeyboardInterrupt
    real_solve = optimizer.solve_witness
    monkeypatch.setattr(optimizer, 'solve_witness', interrupt)
    with pytest.raises(KeyboardInterrupt):
        optimizer.run_track(**settings, workers=1)
    state = json.loads((settings['output']/'optimizer.json').read_text())
    assert state['status'] == 'interrupted'
    assert not (settings['output']/'best.json').exists()
    monkeypatch.setattr(optimizer, 'solve_witness', real_solve)
    optimizer.run_track(**settings, workers=1)
    assert calls[0]['command_seed'].name == 'best_iterate.npz'


def test_search_domain_stop_is_not_reported_as_numerical_plateau(pipeline, monkeypatch):
    settings, _ = pipeline
    monkeypatch.setattr(optimizer, 'run_campaign', lambda **kw: dict(status='search_domain_reached'))
    state = optimizer.run_track(**settings, workers=1)
    assert state['status'] == 'search_domain_reached'


def test_no_feasible_bootstrap_is_not_reported_as_raceline_success(pipeline, monkeypatch):
    settings, calls = pipeline
    original = optimizer.solve_witness
    def failed(**kw):
        result = original(**kw)
        result.update(status='no_validated_witness', max_constraint_violation=1.)
        return result
    monkeypatch.setattr(optimizer, 'solve_witness', failed)
    state = optimizer.run_track(**settings, workers=1, refinements=1)
    assert state['status'] == 'no_validated_raceline'
    assert not (settings['output']/'best.json').exists()


def test_saved_command_seed_is_recertified_without_regenerating_geometry(pipeline, monkeypatch):
    settings, calls = pipeline
    seed = settings['output'].parent/'source_seed.npz'
    witness(seed)
    monkeypatch.setattr(optimizer, 'geometry_starts', lambda *a: pytest.fail('must reuse certified seed'))
    state = optimizer.run_track(**settings, workers=1, command_seed=seed)
    assert state['status'] == 'plateau' and state['best_lap_s'] == 1.
    assert len(calls) == 0
    assert (settings['output']/'input/source_seed.npz').read_bytes() == seed.read_bytes()
    assert state['starts']['source_seed']['witness'].endswith('certified/witness.npz')


def test_saved_command_seed_never_bypasses_current_certification(pipeline, monkeypatch):
    settings, _ = pipeline
    seed = settings['output'].parent/'source_seed.npz'
    witness(seed)
    def fail(*args):
        return dict(status='no_validated_witness')
    monkeypatch.setattr(optimizer, 'certify_witness', fail)
    with pytest.raises(ValueError, match='fails current map/source certification'):
        optimizer.run_track(**settings, workers=1, command_seed=seed)
    assert not (settings['output']/'best.json').exists()


def test_geometric_initializer_uses_real_source_state_and_is_not_certified(tmp_path):
    # Artificial circular MAP GEOMETRY for a numerical module check. This is
    # neither simulator telemetry nor evidence of a real car's prediction error.
    theta = np.linspace(0., 2*np.pi, 160, endpoint=False)
    radius = 3.
    table = np.column_stack([theta*radius, radius*np.cos(theta), radius*np.sin(theta),
        theta+np.pi/2, np.full(len(theta), 1/radius), np.full(len(theta), 1.2),
        np.zeros(len(theta)), np.ones((len(theta), 2))])
    reference, seed = tmp_path/'circle.csv', tmp_path/'initializer.npz'
    np.savetxt(reference, table, delimiter=',')
    count = optimizer.geometric_seed(reference, seed, optimizer.REPO)
    with np.load(seed) as data:
        assert data['state'].shape == (21, count+1)
        assert data['controls'].shape == (2, count)
        assert np.isfinite(data['state']).all()
        assert np.min(data['controls'][1]) > 0.
        assert data['state'][2, -1]-data['state'][2, 0] == pytest.approx(2*np.pi)
        assert str(data['source_sha256']) == 'uncertified-map-geometric-initializer'
        assert int(data['iteration']) == -1
        assert np.isinf(data['max_constraint_violation'])
        function, _ = optimizer.production_command_map(optimizer.REPO,
                                                       optimizer.REPO/'build_task/command_ocp')
        predicted = np.asarray(function(np.r_[data['state'][:, 0], data['controls'][:, 0]]))
        assert predicted.shape == (21, 1) and np.isfinite(predicted).all()
