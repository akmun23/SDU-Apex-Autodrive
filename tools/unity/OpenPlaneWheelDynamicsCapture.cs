#if DEVELOPMENT_BUILD
using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

// Development-player-only, read-only capture of Unity WheelCollider state.
// Copy into the simulator source project's Assets/Scripts and call Capture(this)
// from VehicleController.FixedUpdate after actuator and wheel-pose updates.
internal static class OpenPlaneWheelDynamicsCapture
{
    private const int Capacity = 180000;
    private const long UnixEpochTicks = 621355968000000000L;

    private struct WheelSample
    {
        public float rpm, steerDeg, motorTorque, brakeTorque;
        public float forwardSlip, sidewaysSlip, contactForce;
        public Vector3 contactPointBody, contactNormalBody;
        public Vector3 contactForwardBody, contactSidewaysBody;
        public Vector3 colliderPositionBody, wheelPosePositionBody;
        public Quaternion wheelPoseRotationBody;
        public byte grounded;
    }

    private struct Sample
    {
        public double simulationTime;
        public double realtimeSinceStartup;
        public long captureUnixTimeNs;
        public Vector3 positionWorld, velocityBody, angularVelocityBody;
        public Quaternion rotationWorld;
        public float throttleCommand, steeringCommand;
        public WheelSample frontLeft, frontRight, rearLeft, rearRight;
    }

    private static Sample[] Samples;
    private static bool _enabled;
    private static int _head, _count;
    private static long _overwritten;
    private static long _unixNsAtRealtimeZero;
    private static VehicleController _vehicle;
    private static string _runId;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Initialize()
    {
        _enabled = !string.IsNullOrEmpty(
            Environment.GetEnvironmentVariable("SDU_APEX_DYNAMICS_DIR"));
        if (_enabled)
        {
            Samples = new Sample[Capacity];
            double realtime = Time.realtimeSinceStartupAsDouble;
            long utcNowUnixNs = (DateTime.UtcNow.Ticks - UnixEpochTicks) * 10L;
            _unixNsAtRealtimeZero = utcNowUnixNs - (long)(realtime * 1.0e9);
            _runId = Environment.GetEnvironmentVariable("SDU_APEX_DYNAMICS_RUN_ID");
            if (string.IsNullOrEmpty(_runId))
                _runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            Application.quitting += Flush;
        }
    }

    public static void Capture(VehicleController vehicle)
    {
        if (!_enabled || vehicle == null || vehicle.VehicleRigidBody == null)
            return;

        Rigidbody body = vehicle.VehicleRigidBody;
        Transform bodyTransform = body.transform;
        double realtime = Time.realtimeSinceStartupAsDouble;
        Sample sample = new Sample
        {
            simulationTime = Time.fixedTimeAsDouble,
            realtimeSinceStartup = realtime,
            captureUnixTimeNs = _unixNsAtRealtimeZero + (long)(realtime * 1.0e9),
            positionWorld = body.position,
            rotationWorld = body.rotation,
            velocityBody = bodyTransform.InverseTransformDirection(body.velocity),
            angularVelocityBody = bodyTransform.InverseTransformDirection(body.angularVelocity),
            throttleCommand = vehicle.AutonomousThrottle,
            steeringCommand = vehicle.AutonomousSteering,
            frontLeft = ReadWheel(vehicle.FrontLeftWheelCollider, bodyTransform),
            frontRight = ReadWheel(vehicle.FrontRightWheelCollider, bodyTransform),
            rearLeft = ReadWheel(vehicle.RearLeftWheelCollider, bodyTransform),
            rearRight = ReadWheel(vehicle.RearRightWheelCollider, bodyTransform)
        };

        _vehicle = vehicle;
        if (_count < Capacity)
        {
            Samples[(_head + _count) % Capacity] = sample;
            _count++;
        }
        else
        {
            Samples[_head] = sample;
            _head = (_head + 1) % Capacity;
            _overwritten++;
        }
    }

    private static WheelSample ReadWheel(WheelCollider wheel, Transform body)
    {
        Vector3 wheelPositionWorld;
        Quaternion wheelRotationWorld;
        wheel.GetWorldPose(out wheelPositionWorld, out wheelRotationWorld);

        WheelSample sample = new WheelSample
        {
            rpm = wheel.rpm,
            steerDeg = wheel.steerAngle,
            motorTorque = wheel.motorTorque,
            brakeTorque = wheel.brakeTorque,
            colliderPositionBody = body.InverseTransformPoint(wheel.transform.position),
            wheelPosePositionBody = body.InverseTransformPoint(wheelPositionWorld),
            wheelPoseRotationBody = Quaternion.Inverse(body.rotation) * wheelRotationWorld
        };

        WheelHit hit;
        if (wheel.GetGroundHit(out hit))
        {
            sample.grounded = 1;
            sample.forwardSlip = hit.forwardSlip;
            sample.sidewaysSlip = hit.sidewaysSlip;
            sample.contactForce = hit.force;
            sample.contactPointBody = body.InverseTransformPoint(hit.point);
            sample.contactNormalBody = body.InverseTransformDirection(hit.normal);
            sample.contactForwardBody = body.InverseTransformDirection(hit.forwardDir);
            sample.contactSidewaysBody = body.InverseTransformDirection(hit.sidewaysDir);
        }
        return sample;
    }

    private static void Flush()
    {
        if (!_enabled || _vehicle == null || _count == 0)
            return;

        string directory = Environment.GetEnvironmentVariable("SDU_APEX_DYNAMICS_DIR");
        StringBuilder safeRunId = new StringBuilder(_runId.Length);
        foreach (char c in _runId)
            safeRunId.Append(char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_');

        string path = Path.Combine(directory, "wheel_dynamics_" + safeRunId + ".csv");
        try
        {
            Directory.CreateDirectory(directory);
            using (StreamWriter writer = new StreamWriter(
                new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read)))
            {
                WriteMetadata(writer, _vehicle);
                writer.WriteLine(Header());
                StringBuilder row = new StringBuilder(1024);
                for (int i = 0; i < _count; i++)
                {
                    row.Length = 0;
                    AppendSample(row, Samples[(_head + i) % Capacity]);
                    writer.WriteLine(row.ToString());
                }
            }
            Debug.Log("Open-plane wheel dynamics capture saved: " + path +
                      " (" + _count + " samples, " + _overwritten + " overwritten)");
        }
        catch (Exception exception)
        {
            Debug.LogError("Open-plane wheel dynamics capture could not be saved: " + exception);
        }
    }

    private static void WriteMetadata(StreamWriter writer, VehicleController vehicle)
    {
        Rigidbody body = vehicle.VehicleRigidBody;
        writer.WriteLine("# unity_version=" + Application.unityVersion);
        writer.WriteLine("# fixed_delta_time_s=" + F(Time.fixedDeltaTime));
        writer.WriteLine("# run_id=" + _runId);
        writer.WriteLine("# unix_ns_at_realtime_zero=" +
                         _unixNsAtRealtimeZero.ToString(CultureInfo.InvariantCulture));
        writer.WriteLine("# capture_clock=Time.realtimeSinceStartupAsDouble mapped to Unix UTC at initialization");
        writer.WriteLine("# mass_kg=" + F(body.mass));
        writer.WriteLine("# inertia_tensor_body=" + V(body.inertiaTensor));
        writer.WriteLine("# inertia_tensor_rotation_body=" + Q(body.inertiaTensorRotation));
        writer.WriteLine("# center_of_mass_body=" + V(body.centerOfMass));
        writer.WriteLine("# capacity_samples=" + Capacity.ToString(CultureInfo.InvariantCulture));
        writer.WriteLine("# captured_samples=" + _count.ToString(CultureInfo.InvariantCulture));
        writer.WriteLine("# overwritten_samples=" + _overwritten.ToString(CultureInfo.InvariantCulture));
        WriteWheelMetadata(writer, "front_left", vehicle.FrontLeftWheelCollider);
        WriteWheelMetadata(writer, "front_right", vehicle.FrontRightWheelCollider);
        WriteWheelMetadata(writer, "rear_left", vehicle.RearLeftWheelCollider);
        WriteWheelMetadata(writer, "rear_right", vehicle.RearRightWheelCollider);
    }

    private static void WriteWheelMetadata(StreamWriter writer, string name, WheelCollider wheel)
    {
        WheelFrictionCurve fx = wheel.forwardFriction;
        WheelFrictionCurve fy = wheel.sidewaysFriction;
        JointSpring suspension = wheel.suspensionSpring;
        writer.WriteLine("# " + name + "_radius_m=" + F(wheel.radius));
        writer.WriteLine("# " + name + "_sprung_mass_kg=" + F(wheel.sprungMass));
        writer.WriteLine("# " + name + "_suspension_distance_m=" + F(wheel.suspensionDistance));
        writer.WriteLine("# " + name + "_suspension_spring_n_per_m=" + F(suspension.spring));
        writer.WriteLine("# " + name + "_suspension_damper_ns_per_m=" + F(suspension.damper));
        writer.WriteLine("# " + name + "_suspension_target_position=" + F(suspension.targetPosition));
        writer.WriteLine("# " + name + "_force_app_point_distance_m=" + F(wheel.forceAppPointDistance));
        writer.WriteLine("# " + name + "_wheel_damping_rate=" + F(wheel.wheelDampingRate));
        writer.WriteLine("# " + name + "_forward_curve=" + F(fx.extremumSlip) + "," +
                         F(fx.extremumValue) + "," + F(fx.asymptoteSlip) + "," +
                         F(fx.asymptoteValue) + "," + F(fx.stiffness));
        writer.WriteLine("# " + name + "_sideways_curve=" + F(fy.extremumSlip) + "," +
                         F(fy.extremumValue) + "," + F(fy.asymptoteSlip) + "," +
                         F(fy.asymptoteValue) + "," + F(fy.stiffness));
    }

    private static string Header()
    {
        StringBuilder header = new StringBuilder(
            "sim_time_s,capture_realtime_s,capture_unix_time_ns," +
            "pos_world_x_m,pos_world_y_m,pos_world_z_m,rot_world_x,rot_world_y,rot_world_z,rot_world_w," +
            "vel_body_x_mps,vel_body_y_mps,vel_body_z_mps,omega_body_x_rps,omega_body_y_rps,omega_body_z_rps," +
            "throttle_command,steering_command");
        AppendWheelHeader(header, "fl");
        AppendWheelHeader(header, "fr");
        AppendWheelHeader(header, "rl");
        AppendWheelHeader(header, "rr");
        return header.ToString();
    }

    private static void AppendWheelHeader(StringBuilder header, string prefix)
    {
        header.Append(',').Append(prefix).Append("_rpm,").Append(prefix).Append("_steer_deg,")
            .Append(prefix).Append("_motor_torque_nm,").Append(prefix).Append("_brake_torque_nm,")
            .Append(prefix).Append("_grounded,").Append(prefix).Append("_forward_slip,")
            .Append(prefix).Append("_sideways_slip,").Append(prefix).Append("_contact_force_n,")
            .Append(prefix).Append("_contact_body_x_m,").Append(prefix).Append("_contact_body_y_m,")
            .Append(prefix).Append("_contact_body_z_m,").Append(prefix).Append("_normal_body_x,")
            .Append(prefix).Append("_normal_body_y,").Append(prefix).Append("_normal_body_z,")
            .Append(prefix).Append("_contact_forward_body_x,").Append(prefix).Append("_contact_forward_body_y,")
            .Append(prefix).Append("_contact_forward_body_z,").Append(prefix).Append("_contact_sideways_body_x,")
            .Append(prefix).Append("_contact_sideways_body_y,").Append(prefix).Append("_contact_sideways_body_z,")
            .Append(prefix).Append("_collider_body_x_m,").Append(prefix).Append("_collider_body_y_m,")
            .Append(prefix).Append("_collider_body_z_m,")
            .Append(prefix).Append("_wheel_pose_body_x_m,").Append(prefix).Append("_wheel_pose_body_y_m,")
            .Append(prefix).Append("_wheel_pose_body_z_m,").Append(prefix).Append("_wheel_pose_body_qx,")
            .Append(prefix).Append("_wheel_pose_body_qy,").Append(prefix).Append("_wheel_pose_body_qz,")
            .Append(prefix).Append("_wheel_pose_body_qw");
    }

    private static void AppendSample(StringBuilder row, Sample sample)
    {
        row.Append(sample.simulationTime.ToString("R", CultureInfo.InvariantCulture));
        row.Append(',').Append(sample.realtimeSinceStartup.ToString("R", CultureInfo.InvariantCulture));
        row.Append(',').Append(sample.captureUnixTimeNs.ToString(CultureInfo.InvariantCulture));
        Append(row, sample.positionWorld);
        Append(row, sample.rotationWorld);
        Append(row, sample.velocityBody);
        Append(row, sample.angularVelocityBody);
        Append(row, sample.throttleCommand);
        Append(row, sample.steeringCommand);
        AppendWheel(row, sample.frontLeft);
        AppendWheel(row, sample.frontRight);
        AppendWheel(row, sample.rearLeft);
        AppendWheel(row, sample.rearRight);
    }

    private static void AppendWheel(StringBuilder row, WheelSample wheel)
    {
        Append(row, wheel.rpm);
        Append(row, wheel.steerDeg);
        Append(row, wheel.motorTorque);
        Append(row, wheel.brakeTorque);
        Append(row, wheel.grounded);
        Append(row, wheel.forwardSlip);
        Append(row, wheel.sidewaysSlip);
        Append(row, wheel.contactForce);
        Append(row, wheel.contactPointBody);
        Append(row, wheel.contactNormalBody);
        Append(row, wheel.contactForwardBody);
        Append(row, wheel.contactSidewaysBody);
        Append(row, wheel.colliderPositionBody);
        Append(row, wheel.wheelPosePositionBody);
        Append(row, wheel.wheelPoseRotationBody);
    }

    private static void Append(StringBuilder row, Vector3 value)
    {
        Append(row, value.x);
        Append(row, value.y);
        Append(row, value.z);
    }

    private static void Append(StringBuilder row, Quaternion value)
    {
        Append(row, value.x);
        Append(row, value.y);
        Append(row, value.z);
        Append(row, value.w);
    }

    private static void Append(StringBuilder row, float value)
    {
        row.Append(',').Append(F(value));
    }

    private static string F(float value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    private static string V(Vector3 value)
    {
        return F(value.x) + "," + F(value.y) + "," + F(value.z);
    }

    private static string Q(Quaternion value)
    {
        return F(value.x) + "," + F(value.y) + "," + F(value.z) + "," + F(value.w);
    }
}
#endif
