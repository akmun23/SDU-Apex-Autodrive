#!/usr/bin/env python3
"""Read-only host/ROS sensor timing capture for comparing Unity modes.

Execute inside a development ROS container after `source /opt/ros/humble/setup.bash`
and `source /workspace/install/setup.bash`. Not a production controller node.
"""
import argparse
import bisect
from collections import deque
import json
import math
from pathlib import Path
import statistics
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sdu_apex_msgs.msg import DriveCommand
from std_msgs.msg import String


def quantile(values, percentile):
    if not values:return None
    x=sorted(values)
    point=(len(x)-1)*percentile/100
    below=int(point)
    fraction=point-below
    return round(x[below]*(1-fraction)+x[min(below+1,len(x)-1)]*fraction,3)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=14)
    parser.add_argument('--wait-for-imu', action='store_true', help='Begin capture only after Unity Connect')
    parser.add_argument('--wait-for-mpc', action='store_true', help='Begin capture after at least three MPC commands (e.g. Unity Restart)')
    parser.add_argument('--wait-timeout',type=float,default=180)
    parser.add_argument('--json-output',type=Path)
    args=parser.parse_args()
    if not 2 <= args.seconds <= 120:parser.error('seconds must be 2..120')
    rclpy.init(args=[])
    node=rclpy.create_node('development_transport_latency_audit', enable_rosout=False)
    labels=['imu','odom','pose','mpc_command','bridge_request','bridge_feedback','bridge_diagnostic']
    stats={name:{'count':0,'last_received':None,'intervals_ms':[], 'stamped_ages_ms':[]} for name in labels}
    extras={'request_response_ms':[],'request_command_update_age_ms':[]}
    steering_samples={'mpc_command':[],'bridge_request':[],'bridge_feedback':[]}
    subscriptions=[]

    def callback(name, message):
        now=time.monotonic();item=stats[name]
        if item['last_received'] is not None:
            item['intervals_ms'].append((now-item['last_received'])*1000.)
        item['last_received']=now
        item['count']+=1
        if hasattr(message, 'header'):
            stamp=message.header.stamp
            epoch=stamp.sec+stamp.nanosec*1e-9
            if epoch>0:item['stamped_ages_ms'].append((time.time()-epoch)*1000.)
        if name in steering_samples:
            stamp=message.header.stamp
            stamped=stamp.sec+stamp.nanosec*1e-9
            if stamped>0:
                steering_samples[name].append((stamped,
                    float(message.steering_angle_rad),
                    float(message.throttle_normalized)))
        if name == 'bridge_diagnostic':
            try:
                payload=json.loads(message.data)
                req=payload.get('request_monotonic_ns')
                recv=payload.get('bridge_arrival_monotonic_ns')
                if isinstance(req,int) and isinstance(recv,int):
                    extras['request_response_ms'].append((recv-req)*1e-6)
                age=payload.get('steering_command_update_age_ms')
                if isinstance(age,(int,float)) and math.isfinite(age):
                    extras['request_command_update_age_ms'].append(float(age))
            except (TypeError,ValueError,AttributeError):pass

    for label,typ,topic in [
        ('imu',Imu,'/autodrive/roboracer_1/imu'),
        ('odom',Odometry,'/odom'),
        ('pose',PoseWithCovarianceStamped,'/current_map_pose'),
        ('mpc_command',DriveCommand,'/mpc/drive_command'),
        ('bridge_request',DriveCommand,'/odom/bridge_request_command'),
        ('bridge_feedback',DriveCommand,'/odom/bridge_actuator_feedback'),
        ('bridge_diagnostic',String,'/autodrive/roboracer_1/bridge_packet_timing'),
    ]:
        subscriptions.append(node.create_subscription(
            typ,topic,lambda msg, label=label:callback(label,msg),qos_profile_sensor_data))
    if args.wait_for_imu or args.wait_for_mpc:
        source = 'mpc_command' if args.wait_for_mpc else 'imu'
        required_count = 3 if args.wait_for_mpc else 1
        print(('AWAITING_MPC_COMMANDS: press Unity Restart while connected' if args.wait_for_mpc
               else 'AWAITING_UNITY_CONNECT: latency recorder subscribed to permitted ROS topics'),flush=True)
        wait_started=time.monotonic()
        while stats[source]['count']<required_count and time.monotonic()-wait_started<args.wait_timeout:
            rclpy.spin_once(node,timeout_sec=.10)
        if stats[source]['count']<required_count:
            node.destroy_node();rclpy.shutdown()
            raise RuntimeError(f'{source} was not observed before timeout')
        print(f'CAPTURE_TRIGGERED: observed {required_count} {source} messages',flush=True)
    start=time.monotonic()
    # Clear pre-connection samples so all channel rates refer to same interval.
    for item in stats.values():
        item['count']=0
        item['last_received']=None
        item['intervals_ms'].clear()
        item['stamped_ages_ms'].clear()
    for values in extras.values():values.clear()
    for values in steering_samples.values():values.clear()
    while time.monotonic()-start<args.seconds:
        rclpy.spin_once(node,timeout_sec=.05)
    elapsed=time.monotonic()-start
    def effective_lag(command_samples, physical_samples):
        """Alignment of command setpoints and Unity feedback, not pure DDS lag.

        The physical steering dynamics are included in this measured delay.
        Use the SAME method in batchmode and GUI; compare only with sufficient
        command variation and enough samples on both sides.
        """
        if len(command_samples)<35 or len(physical_samples)<35:return None
        cmd_t=[v[0] for v in command_samples]
        results={}
        for kind,index in [('steering',1),('throttle',2)]:
            scores=[]
            for delay_ms in range(0,151,5):
                sq=[]
                for stamp,steer,throttle in physical_samples:
                    if stamp < command_samples[0][0]+0.2:
                        continue
                    j=bisect.bisect_right(cmd_t,stamp-delay_ms/1000.0)-1
                    if j<0:continue
                    actual=steer if index==1 else throttle
                    sq.append((command_samples[j][index]-actual)**2)
                if len(sq)>35:
                    scores.append((math.sqrt(sum(sq)/len(sq)),delay_ms))
            if scores:
                error,delay=min(scores)
                results[kind]={'best_alignment_delay_ms':delay,
                               'rmse':round(error,5),
                               'note':'includes actuator dynamics, 25 ms command sampling, and transport'}
        return results

    result={
        'schema_version':1, 'seconds':round(elapsed,2),
        'channels':{
            k:{'count':v['count'],'rate_hz':round(v['count']/elapsed,2),
                'age_p50_ms':quantile(v['stamped_ages_ms'],50),
                'age_p95_ms':quantile(v['stamped_ages_ms'],95),
                'age_p99_ms':quantile(v['stamped_ages_ms'],99),
                'interarrival_p95_ms':quantile(v['intervals_ms'],95),
                'interarrival_max_ms':round(max(v['intervals_ms']),3) if v['intervals_ms'] else None}
            for k,v in stats.items()},
        'effective_command_to_unity_feedback_alignment':effective_lag(
            steering_samples['mpc_command'],steering_samples['bridge_feedback']),
        'bridge_request_to_feedback_alignment':effective_lag(
            steering_samples['bridge_request'],steering_samples['bridge_feedback']),
        'bridge_diagnostics':{
            k:{'count':len(v),'p50_ms':quantile(v,50),'p95_ms':quantile(v,95),'p99_ms':quantile(v,99)}
            for k,v in extras.items()},
    }
    output=json.dumps(result,indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True,exist_ok=True)
        args.json_output.write_text(output+'\n')
    print(output,flush=True)
    node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
