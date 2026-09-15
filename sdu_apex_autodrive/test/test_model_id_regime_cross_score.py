import importlib.util
from pathlib import Path
import sys


def _load():
    path = Path("tools/model_id/cross_score_regime_models.py")
    tools_path = str(path.parent.resolve())
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    spec = importlib.util.spec_from_file_location("regime_cross_score", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regime_discovery_keeps_low_demand_reference_and_dynamic_origins(tmp_path):
    names = (
        "10_static_repeatability_run01_40hz_20260913",
        "30_straight_replay_run01_40hz_20260913",
        "01_longitudinal_throttle_from_rest_40hz_20260913",
        "40_steering_replay_run01_40hz_20260913",
        "70_d1_d2_steady_corner_40hz_20260913",
        "20_combined_throttle_steering_run01_40hz_20260913",
        "60_e1_corner_throttle_40hz_20260913",
    )
    for name in names:
        (tmp_path / name).mkdir()
    groups = _load()._discover_groups(tmp_path)
    assert "10_static_repeatability_run01_40hz_20260913" in groups["low_demand"]
    assert "30_straight_replay_run01_40hz_20260913" in groups["low_demand"]
    assert "30_straight_replay_run01_40hz_20260913" in groups["straight"]
    assert "70_d1_d2_steady_corner_40hz_20260913" in groups["corner_no_throttle"]
    assert "60_e1_corner_throttle_40hz_20260913" in groups["corner_throttle"]


def test_score_summary_reports_active_mpc_horizons():
    module = _load()
    summary = module._score_summary({
        "0.50s": {"position_m": {"count": 4, "p95": 0.1,
                                    "p99": 0.2, "max": 0.3}},
        "0.75s": {"position_m": {"count": 4, "p95": 0.4,
                                    "p99": 0.5, "max": 0.6}},
    })
    assert summary["0.50s"]["p95_m"] == 0.1
    assert summary["0.75s"]["p95_m"] == 0.4
    assert set(summary) == {"0.50s", "0.75s"}
