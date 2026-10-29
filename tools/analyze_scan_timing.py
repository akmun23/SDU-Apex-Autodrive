#!/usr/bin/env python3
"""Offline scan/map timing diagnostic; NEVER a live timestamp calibration.

Compare recorded range endpoints with the occupied map at nearby truth poses.
Negative offset means a scan is better explained by an older vehicle pose.
Map/extrinsic errors can also shift the optimum; this is evidence, not proof of
an acquisition timestamp. No ground truth or fitted offset is used at runtime.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates


def endpoint_losses(field, resolution, origin, poses, ranges, angles, laser_xy):
    """One scan, many candidate rear-axle poses; metric squared endpoint error."""
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    bx = laser_xy[0] + ranges*np.cos(angles)
    by = laser_xy[1] + ranges*np.sin(angles)
    x = poses[:, 0, None] + c[:, None]*bx - s[:, None]*by
    y = poses[:, 1, None] + s[:, None]*bx + c[:, None]*by
    # ROS map origin is the lower-left pixel CORNER; raster row 0 is at top.
    col = (x-origin[0])/resolution-.5
    row = field.shape[0]-.5-(y-origin[1])/resolution
    distances = map_coordinates(field, [row.ravel(), col.ravel()], order=1,
        mode='constant', cval=.25).reshape(row.shape)
    return np.mean(np.minimum(distances, .25)**2, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag', type=Path, required=True)
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--parameters', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scan-stride', type=int, default=4)
    args = p.parse_args()
    if args.scan_stride < 1: p.error('positive scan stride required')
    if args.output.exists(): p.error('refusing to overwrite analysis')
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    if __package__:
        from .evaluate_mpc_bag import yaw_from_quaternion, stamp_nanoseconds
    else:
        from evaluate_mpc_bag import yaw_from_quaternion, stamp_nanoseconds

    config = yaml.safe_load(args.map.read_text())
    if config['origin'][2] != 0.: raise ValueError('rotated maps not supported')
    image = np.asarray(Image.open(args.map.parent/config['image']).convert('L'))
    occupancy = image/255. if config.get('negate', 0) else 1.-image/255.
    occupied = occupancy > config['occupied_thresh']
    field = distance_transform_edt(~occupied)*config['resolution']
    params = yaml.safe_load(args.parameters.read_text())['gpu_amcl_cpp']['ros__parameters']
    laser = (params['laser_offset_x'], params['laser_offset_y'])
    reader = SequentialReader()
    storage = 'mcap' if any(args.bag.glob('*.mcap')) else 'sqlite3'
    reader.open(StorageOptions(uri=str(args.bag), storage_id=storage), ConverterOptions('cdr', 'cdr'))
    gt_topic, scan_topic = '/autodrive/roboracer_1/odom', params['scan_topic']
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()
             if t.name in (gt_topic, scan_topic)}
    truth, scans = [], []
    seen = 0
    last_ranges = None
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic not in types: continue
        msg = deserialize_message(raw, types[topic])
        stamp = stamp_nanoseconds(msg)
        if topic == gt_topic:
            pose = msg.pose.pose
            truth.append((stamp, pose.position.x, pose.position.y,
                          yaw_from_quaternion(pose.orientation),
                          msg.twist.twist.linear.x, msg.twist.twist.angular.z))
        else:
            ranges = np.asarray(msg.ranges, dtype=float)
            if last_ranges is not None and np.array_equal(ranges, last_ranges, equal_nan=True): continue
            last_ranges = ranges
            seen += 1
            if seen % args.scan_stride: continue
            indices = np.arange(0, len(ranges), max(1, len(ranges)//270))
            ranges = ranges[indices]
            valid = np.isfinite(ranges) & (ranges >= max(msg.range_min, params['laser_min_range'])) & (ranges < params['laser_max_range'])
            scans.append((stamp, ranges[valid], (msg.angle_min+indices*msg.angle_increment)[valid]))
    if len(truth) < 3 or not scans: raise ValueError('missing truth or scan evidence')
    epoch = truth[0][0]
    times = np.array([(r[0]-epoch)*1.e-9 for r in truth])
    if np.any(np.diff(times) <= 0.): raise ValueError('truth epochs not monotonic')
    values = np.asarray([r[1:] for r in truth])
    values[:, 2] = np.unwrap(values[:, 2])
    offsets = np.arange(-.040, .0201, .0025)
    samples = []
    for stamp, ranges, angles in scans:
        t = (stamp-epoch)*1.e-9
        if t+offsets[0] < times[0] or t+offsets[-1] > times[-1] or len(ranges)<30: continue
        u = float(np.interp(t, times, values[:, 3]))
        if u < 2.: continue
        poses = np.column_stack([np.interp(t+offsets, times, values[:, i]) for i in range(3)])
        losses = endpoint_losses(field, config['resolution'], config['origin'], poses, ranges, angles, laser)
        samples.append(dict(stamp_ns=stamp, u=u,
            r=float(np.interp(t,times,values[:,4])), losses=losses.tolist()))
    if not samples: raise ValueError('no moving scans bracketed by truth')
    zero_index = int(np.argmin(abs(offsets)))
    result = dict(bag=str(args.bag), method=__doc__, map_resolution_m=config['resolution'],
                  laser_offset_m=laser, offset_ms=(1000*offsets).tolist(), groups={})
    for name, selected in [('all', samples), ('slow_2_to_3', [s for s in samples if s['u']<3]),
                           ('fast_above_6', [s for s in samples if s['u']>6]),
                           ('hard_turn', [s for s in samples if abs(s['r'])>2])]:
        if not selected: continue
        losses = np.asarray([s['losses'] for s in selected])
        mean = np.mean(losses, axis=0)
        result['groups'][name] = dict(samples=len(selected), best_offset_ms=float(1000*offsets[np.argmin(mean)]),
            endpoint_rms_curve_m=np.sqrt(mean).tolist(),
            best_offsets_ms=[float(1000*offsets[i]) for i in np.argmin(losses, axis=1)])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({n:{'samples':g['samples'], 'best_offset_ms':g['best_offset_ms'],
        'zero_rms_m':g['endpoint_rms_curve_m'][zero_index], 'best_rms_m':min(g['endpoint_rms_curve_m'])}
        for n,g in result['groups'].items()}, indent=2))


if __name__ == '__main__': main()
