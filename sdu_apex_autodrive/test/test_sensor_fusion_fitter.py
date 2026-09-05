from pathlib import Path

from sdu_apex_autodrive.scripts.fit_regime_sensor_fusion_model import (
    load_imu_filter_alpha,
    metrics,
    score_deployed_examples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_fitter_reads_the_active_runtime_imu_alpha():
    config = REPO_ROOT / "f1tenth_localization/config/sensor_odometry.yaml"
    assert load_imu_filter_alpha(config) == 0.90


def test_deployed_score_prefers_publish_odom_diagnostic_value():
    examples = [{
        "target": 2.0,
        "truth_regime": "steady",
        "deployed_prediction": 1.8,
        "deployed_source": "odom_diagnostics",
        "deployed_branch": "braking_observer",
        "sensor_regime": "decelerating",
    }]

    rows = score_deployed_examples(examples)

    assert rows[0]["prediction"] == 1.8
    assert rows[0]["prediction_source"] == "odom_diagnostics"
    assert rows[0]["runtime_branch"] == "braking_observer"


def test_runtime_and_offline_metric_families_are_named_separately():
    rows = [{
        "target": 2.0,
        "prediction": 1.9,
        "sensor_regime": "decelerating",
        "runtime_branch": "braking_observer",
    }]

    offline = metrics(rows, "offline_forest_validation")
    deployed = metrics(rows, "deployed_odom_validation")

    assert offline and offline[0]["model"] == "offline_forest_validation"
    assert deployed and deployed[0]["model"] == "deployed_odom_validation"
