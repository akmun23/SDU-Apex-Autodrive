#!/usr/bin/env python3
"""Train and benchmark a causal residual speed estimator for SDU-Apex-Autodrive.

Training file: identification_grid_20260904_113534.csv only.
Validation file: identification_grid_20260904_123232.csv only.

Runtime model structure:
    corrected_speed = current_causal_odom_speed
                      + max(current_causal_odom_speed, 0.5) * learned_fractional_residual

The learned residual uses only signals available causally from the existing
IMU/encoder odometry diagnostics plus their past history. Ground truth is used
only for offline alignment, training targets, weights, and scoring.

The calibration `phase` column is used ONLY to reset offline history at the
known simulator reset boundary. It is never a model feature.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

USECOLS = [
    'source_event_name','source_event_count','source_event_stamp_s','phase',
    'gt_speed_mps','gt_longitudinal_accel_mps2',
    'odom_diagnostics_speed_mps','odom_imu_pair_speed_mps','odom_imu_speed_mps',
    'odom_raw_wheel_speed_mps','odom_corrected_wheel_speed_mps',
    'odom_wheel_observation_confidence','odom_imu_acceleration_bias_mps2',
    'odom_sensor_fusion_window_raw_speed_mps','odom_sensor_fusion_window_mapped_speed_mps',
    'odom_imu_observer_acceleration_mps2','odom_imu_pair_lead_s',
    'odom_sensor_motion_regime',
    'left_encoder_rad','right_encoder_rad','left_encoder_stamp_s','right_encoder_stamp_s',
    'ax_mps2','ay_mps2','imu_yaw_rate_radps','imu_stamp_s',
    'odom_sensor_fusion_model_speed_mps','odom_sensor_fusion_model_active',
    'odom_frozen_encoder_model_active','odom_frozen_encoder_model_decel_mps2',
]


def load_aligned(path: Path) -> pd.DataFrame:
    # New runtime captures expose the pre-v2 causal baseline explicitly. The
    # canonical archive predates that field, so retain the old diagnostic
    # baseline as a compatibility fallback for the frozen holdout replay.
    header = pd.read_csv(path, nrows=0)
    usecols = list(USECOLS)
    if 'odom_v2_base_speed_mps' in header.columns:
        usecols.append('odom_v2_base_speed_mps')
    raw = pd.read_csv(path, usecols=usecols)
    gt = raw[raw.source_event_name.eq('gt_odom')].copy()
    gt = gt.dropna(subset=['source_event_stamp_s','gt_speed_mps'])
    gt = gt.drop_duplicates('source_event_count', keep='last').sort_values('source_event_stamp_s')
    gt = gt.drop_duplicates('source_event_stamp_s', keep='last')

    dg = raw[raw.source_event_name.eq('odom_diagnostics')].copy()
    dg = dg.dropna(subset=['source_event_stamp_s'])
    dg = dg.drop_duplicates('source_event_count', keep='last').sort_values('source_event_stamp_s')

    gt_t = gt.source_event_stamp_s.to_numpy(dtype=float)
    t = dg.source_event_stamp_s.to_numpy(dtype=float)
    inside = (t >= gt_t[0]) & (t <= gt_t[-1])
    dg = dg.loc[inside].copy().reset_index(drop=True)
    t = dg.source_event_stamp_s.to_numpy(dtype=float)
    dg['truth_speed'] = np.interp(t, gt_t, gt.gt_speed_mps.to_numpy(dtype=float))

    ga = gt.gt_longitudinal_accel_mps2.to_numpy(dtype=float)
    finite = np.isfinite(ga)
    if finite.sum() >= 2:
        dg['truth_accel'] = np.interp(t, gt_t[finite], ga[finite])
    else:
        dg['truth_accel'] = np.nan
    return dg


def runtime_baseline_column(df: pd.DataFrame) -> str:
    """Return the exact baseline consumed by the deployed v2 node.

    New captures record the pre-v2 speed explicitly.  Older captures only
    contain the post-v2 diagnostics speed and remain supported for the
    historical replay test.
    """
    if 'odom_v2_base_speed_mps' in df.columns:
        return 'odom_v2_base_speed_mps'
    return 'odom_diagnostics_speed_mps'


def add_episode(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    p = d['phase'].astype(str)
    prev = p.shift(1).fillna('')
    # Offline-only state reset corresponding to the calibration reset command.
    starts = p.str.startswith('grid_settle_') & ~prev.str.startswith('grid_settle_')
    d['episode'] = starts.cumsum().astype(int)
    return d


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = add_episode(df)
    sigs = {
        'base':runtime_baseline_column(d),
        'imu_pair':'odom_imu_pair_speed_mps',
        'imu':'odom_imu_speed_mps',
        'wheel_raw':'odom_raw_wheel_speed_mps',
        'wheel_corr':'odom_corrected_wheel_speed_mps',
        'win_raw':'odom_sensor_fusion_window_raw_speed_mps',
        'win_map':'odom_sensor_fusion_window_mapped_speed_mps',
        'acc_obs':'odom_imu_observer_acceleration_mps2',
        'bias':'odom_imu_acceleration_bias_mps2',
        'conf':'odom_wheel_observation_confidence',
        'ax':'ax_mps2',
        'ay':'ay_mps2',
        'yaw':'imu_yaw_rate_radps',
    }
    X = pd.DataFrame(index=d.index)
    for name, column in sigs.items():
        X[name] = pd.to_numeric(d[column], errors='coerce')

    X['dt'] = d.groupby('episode')['source_event_stamp_s'].diff().clip(lower=0, upper=0.2).fillna(0.025)
    X['gap_win_imu'] = X.win_map - X.imu_pair
    X['gap_raw_imu'] = X.wheel_raw - X.imu_pair
    X['abs_gap_win_imu'] = X.gap_win_imu.abs()
    X['abs_gap_raw_imu'] = X.gap_raw_imu.abs()
    X['wheel_frozen'] = (X.win_raw.abs() < 0.15).astype(float)
    X['wheel_instant_frozen'] = (X.wheel_raw.abs() < 0.15).astype(float)

    group = d['episode']
    lag_sigs = ['base','imu_pair','win_map','win_raw','wheel_raw','acc_obs','ax','bias','conf','gap_win_imu']
    for lag in [1,2,4,8,16,32]:
        for name in lag_sigs:
            shifted = X[name].groupby(group).shift(lag)
            X[f'{name}_lag{lag}'] = shifted
            if name in ['base','imu_pair','win_map','acc_obs','gap_win_imu']:
                X[f'{name}_d{lag}'] = X[name] - shifted

    for window in [4,8,16,32]:
        for name in ['base','imu_pair','win_map','win_raw','acc_obs','ax','gap_win_imu']:
            grouped = X[name].groupby(group)
            X[f'{name}_med{window}'] = grouped.transform(
                lambda s: s.rolling(window, min_periods=1).median())
            X[f'{name}_mean{window}'] = grouped.transform(
                lambda s: s.rolling(window, min_periods=1).mean())
            if name in ['base','imu_pair','win_map','acc_obs','gap_win_imu']:
                X[f'{name}_std{window}'] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=2).std()).fillna(0.0)

    # Explicit causal transient memory. No truth and no commands are used.
    n = len(d)
    decel_time = np.zeros(n)
    accel_time = np.zeros(n)
    frozen_time = np.zeros(n)
    entry_speed = np.zeros(n)
    entry_imu = np.zeros(n)
    integrated_dv = np.zeros(n)
    anchor_age = np.zeros(n)
    anchor_speed = np.zeros(n)

    current_episode = -1
    last_t = None
    was_decel = False
    was_accel = False
    dtime = atime = ftime = 0.0
    entry = entry_i = 0.0
    int_dv = 0.0
    anchor = anchor_t = 0.0

    for pos, idx in enumerate(d.index):
        ep = int(d.at[idx, 'episode'])
        t = float(d.at[idx, 'source_event_stamp_s'])
        dt = 0.0 if last_t is None or ep != current_episode else max(0.0, min(0.2, t - last_t))
        a = float(X.at[idx,'acc_obs']) if np.isfinite(X.at[idx,'acc_obs']) else 0.0
        imu = float(X.at[idx,'imu_pair']) if np.isfinite(X.at[idx,'imu_pair']) else 0.0
        wm = float(X.at[idx,'win_map']) if np.isfinite(X.at[idx,'win_map']) else 0.0
        confidence = float(X.at[idx,'conf']) if np.isfinite(X.at[idx,'conf']) else 0.0
        base = float(X.at[idx,'base']) if np.isfinite(X.at[idx,'base']) else imu

        if ep != current_episode:
            dtime = atime = ftime = int_dv = 0.0
            entry = entry_i = imu
            anchor = wm if wm > 0.0 else imu
            anchor_t = t
            was_decel = was_accel = False

        is_decel = a < -0.5
        is_accel = a > 0.5
        is_frozen = abs(float(X.at[idx,'win_raw']) if np.isfinite(X.at[idx,'win_raw']) else 0.0) < 0.15 and imu > 0.3

        if is_decel and not was_decel:
            dtime = 0.0
            entry = base
            entry_i = imu
            int_dv = 0.0
        if is_accel and not was_accel:
            atime = 0.0

        dtime = dtime + dt if is_decel else 0.0
        atime = atime + dt if is_accel else 0.0
        ftime = ftime + dt if is_frozen else 0.0
        if is_decel:
            int_dv += a * dt

        if wm > 0.2 and confidence > 0.7 and abs(wm - imu) < 0.75 and abs(a) < 1.0:
            anchor = wm
            anchor_t = t

        decel_time[pos] = dtime
        accel_time[pos] = atime
        frozen_time[pos] = ftime
        entry_speed[pos] = entry
        entry_imu[pos] = entry_i
        integrated_dv[pos] = int_dv
        anchor_age[pos] = max(0.0, t - anchor_t)
        anchor_speed[pos] = anchor

        was_decel = is_decel
        was_accel = is_accel
        current_episode = ep
        last_t = t

    X['decel_time'] = decel_time
    X['accel_time'] = accel_time
    X['frozen_time'] = frozen_time
    X['transient_entry_speed'] = entry_speed
    X['transient_entry_imu'] = entry_imu
    X['imu_dv_since_decel'] = integrated_dv
    X['anchor_age'] = anchor_age
    X['anchor_speed'] = anchor_speed
    # A long-memory state referenced to the episode reset. Causal in runtime.
    first_imu = X['imu_pair'].groupby(group).transform('first')
    X['anchor_pred'] = X['anchor_speed'] + (X['imu_pair'] - first_imu)

    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.groupby(group, group_keys=False).ffill().fillna(0.0)
    return d, X


def metric_table(d: pd.DataFrame, prediction: np.ndarray, model_name: str) -> pd.DataFrame:
    temp = d.copy()
    temp['prediction'] = prediction
    temp = temp[np.isfinite(temp.prediction) & np.isfinite(temp.truth_speed) & (temp.truth_speed >= 1.0)].copy()
    temp['abs_error'] = (temp.prediction - temp.truth_speed).abs()
    temp['rel_error_pct'] = 100.0 * temp.abs_error / temp.truth_speed

    regimes = {
        'all': np.ones(len(temp), dtype=bool),
        'accelerating': temp.truth_accel > 0.5,
        'steady': temp.truth_accel.abs() <= 0.5,
        'decelerating': temp.truth_accel < -0.5,
    }
    rows = []
    for regime, mask in regimes.items():
        part = temp.loc[mask]
        if len(part):
            rows.append({
                'model':model_name,'regime':regime,'speed_bin':'all_ge_1_mps','samples':len(part),
                'mae_mps':part.abs_error.mean(),
                'p95_abs_error_mps':np.percentile(part.abs_error,95),
                'median_relative_error_pct':np.median(part.rel_error_pct),
                'p95_relative_error_pct':np.percentile(part.rel_error_pct,95),
            })
        for lo,hi in [(1,3),(3,5),(5,10),(10,15),(15,20),(20,23)]:
            b = part[(part.truth_speed >= lo) & (part.truth_speed < hi if hi < 23 else part.truth_speed <= hi)]
            if not len(b):
                continue
            rows.append({
                'model':model_name,'regime':regime,'speed_bin':f'{lo}-{hi}_mps','samples':len(b),
                'mae_mps':b.abs_error.mean(),
                'p95_abs_error_mps':np.percentile(b.abs_error,95),
                'median_relative_error_pct':np.median(b.rel_error_pct),
                'p95_relative_error_pct':np.percentile(b.rel_error_pct,95),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('training_csv', type=Path)
    ap.add_argument('validation_csv', type=Path)
    ap.add_argument('--out', type=Path, default=Path('odom_model_output'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = load_aligned(args.training_csv)
    valid = load_aligned(args.validation_csv)
    train, X_train = make_features(train)
    valid, X_valid = make_features(valid)

    train_mask = (
        (train.truth_speed >= 0.5) & np.isfinite(train.truth_speed) &
        np.isfinite(train.odom_diagnostics_speed_mps)
    )
    baseline_column = runtime_baseline_column(train)
    baseline_train = train[baseline_column].to_numpy(dtype=float)
    denominator = np.maximum(baseline_train, 0.5)
    target_fraction = np.clip((train.truth_speed.to_numpy(dtype=float) - baseline_train) / denominator, -2.0, 2.0)

    speed = train.truth_speed.to_numpy(dtype=float)
    accel = train.truth_accel.to_numpy(dtype=float)
    weights = np.ones(len(train), dtype=float)
    weights[(speed >= 1.0) & (speed < 3.0)] *= 5.0
    weights[(speed >= 3.0) & (speed < 5.0)] *= 3.0
    weights[np.abs(accel) > 0.5] *= 2.0

    model = lgb.LGBMRegressor(
        objective='huber', alpha=0.9,
        n_estimators=1400, learning_rate=0.02,
        num_leaves=63, min_child_samples=20,
        colsample_bytree=0.75,
        reg_lambda=1.0, reg_alpha=0.05,
        verbosity=-1, n_jobs=-1, random_state=12,
    )
    model.fit(X_train.loc[train_mask], target_fraction[train_mask], sample_weight=weights[train_mask])

    baseline_column = runtime_baseline_column(valid)
    baseline = valid[baseline_column].to_numpy(dtype=float)
    fractional_correction = model.predict(X_valid)
    prediction = np.clip(baseline + np.maximum(baseline, 0.5) * fractional_correction, 0.0, 30.0)

    baseline_metrics = metric_table(valid, baseline, 'current_logged_odom')
    new_metrics = metric_table(valid, prediction, 'causal_lgbm_residual_v1')
    metrics = pd.concat([baseline_metrics,new_metrics], ignore_index=True)
    metrics.to_csv(args.out / 'validation_metrics.csv', index=False)

    prediction_frame = pd.DataFrame({
        'source_event_stamp_s':valid.source_event_stamp_s,
        'truth_speed_mps':valid.truth_speed,
        'truth_accel_mps2':valid.truth_accel,
        'current_odom_speed_mps':baseline,
        'new_speed_mps':prediction,
        'new_fractional_correction':fractional_correction,
    })
    prediction_frame.to_csv(args.out / 'validation_predictions.csv', index=False)

    model.booster_.save_model(str(args.out / 'lgbm_speed_residual.txt'))
    (args.out / 'feature_names.json').write_text(json.dumps(list(X_train.columns), indent=2))

    def row(model_name: str, regime: str='all', speed_bin: str='all_ge_1_mps'):
        r = metrics[(metrics.model == model_name) & (metrics.regime == regime) & (metrics.speed_bin == speed_bin)].iloc[0]
        return {k:(int(r[k]) if k=='samples' else float(r[k])) for k in [
            'samples','mae_mps','p95_abs_error_mps','median_relative_error_pct','p95_relative_error_pct']}

    summary = {
        'training_csv':str(args.training_csv),
        'validation_csv':str(args.validation_csv),
        'validation_is_separate_from_training':True,
        'ground_truth_is_runtime_input':False,
        'current_logged_odom':row('current_logged_odom'),
        'new_model':row('causal_lgbm_residual_v1'),
        'accelerating_current':row('current_logged_odom','accelerating'),
        'accelerating_new':row('causal_lgbm_residual_v1','accelerating'),
        'decelerating_current':row('current_logged_odom','decelerating'),
        'decelerating_new':row('causal_lgbm_residual_v1','decelerating'),
    }
    summary['overall_p95_relative_improvement_pct'] = 100.0 * (
        summary['current_logged_odom']['p95_relative_error_pct'] - summary['new_model']['p95_relative_error_pct']
    ) / summary['current_logged_odom']['p95_relative_error_pct']
    summary['accelerating_p95_relative_improvement_pct'] = 100.0 * (
        summary['accelerating_current']['p95_relative_error_pct'] - summary['accelerating_new']['p95_relative_error_pct']
    ) / summary['accelerating_current']['p95_relative_error_pct']
    summary['decelerating_p95_relative_improvement_pct'] = 100.0 * (
        summary['decelerating_current']['p95_relative_error_pct'] - summary['decelerating_new']['p95_relative_error_pct']
    ) / summary['decelerating_current']['p95_relative_error_pct']
    (args.out / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
