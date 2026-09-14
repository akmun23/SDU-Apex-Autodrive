using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

// Diagnostic-only component used by the disposable model-identification build.
// It is inert unless AUTODRIVE_MODEL_ID_DIAGNOSTICS_DIR is set and never feeds
// any values back into the bridge, controller, or vehicle physics.
[DefaultExecutionOrder(1000)]
public sealed class ModelIdentificationDiagnostics : MonoBehaviour
{
    private const string OutputEnvironmentVariable =
        "AUTODRIVE_MODEL_ID_DIAGNOSTICS_DIR";
    private const string BuildTagEnvironmentVariable =
        "AUTODRIVE_SIMULATOR_BUILD_TAG";
    private const string ExperimentEnvironmentVariable =
        "AUTODRIVE_MODEL_ID_EXPERIMENT";
    private const string RampSweepExperiment = "ramp_sweep_v1";
    private const string DriveExcitationExperiment = "drive_excitation_v1";
    private const string CombinedSlipMatrixExperiment = "combined_slip_matrix_v1";
    private const string NearZeroActuatorExperiment = "near_zero_actuator_v1";
    private const string PoweredDriveRepeatExperiment =
        "powered_drive_repeat_v1";
    private const string InitialSpeedEnvironmentVariable =
        "AUTODRIVE_MODEL_ID_INITIAL_SPEED_MPS";
    private const string CommandEnvironmentVariable =
        "AUTODRIVE_MODEL_ID_THROTTLE_NORM";
    private const string WarmupSecondsEnvironmentVariable =
        "AUTODRIVE_MODEL_ID_WARMUP_SECONDS";
    private const string WarmupThrottleEnvironmentVariable =
        "AUTODRIVE_MODEL_ID_WARMUP_THROTTLE_NORM";
    private const string MeasurementSecondsEnvironmentVariable =
        "AUTODRIVE_MODEL_ID_MEASUREMENT_SECONDS";
    private const string DurationEnvironmentVariable =
        "AUTODRIVE_MODEL_ID_EXPERIMENT_DURATION_SECONDS";

    public VehicleController Controller;
    public Rigidbody VehicleRigidBody;
    public WheelCollider[] Wheels;
    public int RecordEveryFixedSteps = 1;

    private StreamWriter trace;
    private bool active;
    private bool staticWritten;
    private bool experimentActive;
    private int fixedStep;
    private int recordStride;
    private float experimentInitialSpeedMps;
    private float experimentThrottleNorm;
    private float experimentWarmupSeconds;
    private float experimentWarmupThrottleNorm;
    private float experimentMeasurementSeconds;
    private float experimentDurationSeconds;
    private float experimentElapsedSeconds;
    private float experimentRequestedThrottleNorm;
    private float experimentRequestedSteeringNorm;
    private bool experimentMeasurementActive;

    [Serializable]
    private sealed class FrictionSnapshot
    {
        public float extremumSlip;
        public float extremumValue;
        public float asymptoteSlip;
        public float asymptoteValue;
        public float stiffness;

        public FrictionSnapshot(WheelFrictionCurve curve)
        {
            extremumSlip = curve.extremumSlip;
            extremumValue = curve.extremumValue;
            asymptoteSlip = curve.asymptoteSlip;
            asymptoteValue = curve.asymptoteValue;
            stiffness = curve.stiffness;
        }
    }

    [Serializable]
    private sealed class WheelSnapshot
    {
        public string name;
        public Vector3 positionVehicleFrame;
        public Vector3 localPosition;
        public Quaternion localRotation;
        public Vector3 colliderCenter;
        public float radius;
        public float mass;
        public float sprungMass;
        public float suspensionDistance;
        public float wheelDampingRate;
        public float forceAppPointDistance;
        public SuspensionSnapshot suspensionSpring;
        public FrictionSnapshot forwardFriction;
        public FrictionSnapshot sidewaysFriction;
    }

    [Serializable]
    private sealed class SuspensionSnapshot
    {
        public float spring;
        public float damper;
        public float targetPosition;

        public SuspensionSnapshot(JointSpring value)
        {
            spring = value.spring;
            damper = value.damper;
            targetPosition = value.targetPosition;
        }
    }

    [Serializable]
    private sealed class RigidBodySnapshot
    {
        public float mass;
        public Vector3 centerOfMass;
        public Vector3 worldCenterOfMass;
        public Vector3 inertiaTensor;
        public Quaternion inertiaTensorRotation;
        public float bodyInertiaX;
        public float bodyInertiaY;
        public float bodyInertiaZ;
        public string yawAxis;
        public float yawInertiaBodyFrame;
        public float drag;
        public float angularDrag;
        public float maxAngularVelocity;
    }

    [Serializable]
    private sealed class VehicleSnapshot
    {
        public string objectName;
        public Vector3 rootLocalPosition;
        public Quaternion rootLocalRotation;
        public float wheelbaseM;
        public float trackWidthM;
        public float wheelRadiusControllerM;
        public float steeringLimitRad;
        public float steeringRateRadPerSecond;
        public float motorTorqueNm;
        public string driveType;
        public string steerType;
        public RigidBodySnapshot rigidBody;
        public WheelSnapshot[] wheels;
    }

    [Serializable]
    private sealed class StaticSnapshot
    {
        public string schemaVersion = "autodrive.simulator_diagnostics.v2";
        public bool diagnosticOnly = true;
        public bool runtimeControlInput = false;
        public bool simulatorPhysicsUnmodified = true;
        public string simulatorBuildTag;
        public string productName;
        public string applicationVersion;
        public string unityVersion;
        public string platform;
        public string sceneName;
        public float fixedDeltaTime;
        public float maximumAllowedTimestep;
        public float timeScale;
        public Vector3 gravity;
        public int defaultSolverIterations;
        public int defaultSolverVelocityIterations;
        public float defaultContactOffset;
        public float defaultMaxDepenetrationVelocity;
        public float defaultMaxAngularSpeed;
        public int vSyncCount;
        public int targetFrameRate;
        public string experiment;
        public float experimentInitialSpeedMps;
        public float experimentThrottleNorm;
        public float experimentWarmupSeconds;
        public float experimentWarmupThrottleNorm;
        public float experimentMeasurementSeconds;
        public float experimentDurationSeconds;
        public VehicleSnapshot vehicle;
    }

    private void Awake()
    {
        string outputDirectory = Environment.GetEnvironmentVariable(
            OutputEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            enabled = false;
            return;
        }

        Controller = Controller ?? GetComponent<VehicleController>();
        VehicleRigidBody = VehicleRigidBody ?? GetComponent<Rigidbody>();
        if (Controller == null || VehicleRigidBody == null)
        {
            Debug.LogError("[ModelIdentificationDiagnostics] Missing VehicleController or Rigidbody");
            enabled = false;
            return;
        }

        Wheels = ResolveWheels();
        if (Wheels.Length != 4)
        {
            Debug.LogError("[ModelIdentificationDiagnostics] Expected four wheel colliders");
            enabled = false;
            return;
        }

        try
        {
            Directory.CreateDirectory(outputDirectory);
            string tracePath = Path.Combine(outputDirectory, "wheel_contact_trace.csv");
            trace = new StreamWriter(tracePath, false, new UTF8Encoding(false));
            trace.WriteLine(BuildTraceHeader());
            trace.Flush();
            recordStride = Mathf.Max(1, RecordEveryFixedSteps);
            string experiment = Environment.GetEnvironmentVariable(
                ExperimentEnvironmentVariable);
            experimentActive = string.Equals(
                experiment, RampSweepExperiment, StringComparison.Ordinal) ||
                string.Equals(experiment, DriveExcitationExperiment,
                              StringComparison.Ordinal) ||
                string.Equals(experiment, CombinedSlipMatrixExperiment,
                              StringComparison.Ordinal) ||
                string.Equals(experiment, NearZeroActuatorExperiment,
                              StringComparison.Ordinal) ||
                string.Equals(experiment, PoweredDriveRepeatExperiment,
                              StringComparison.Ordinal);
            experimentInitialSpeedMps = ReadEnvironmentFloat(
                InitialSpeedEnvironmentVariable, 0.0f, 0.0f, 30.0f);
            experimentThrottleNorm = ReadEnvironmentFloat(
                CommandEnvironmentVariable, 0.0f, 0.0f, 1.0f);
            experimentWarmupSeconds = ReadEnvironmentFloat(
                WarmupSecondsEnvironmentVariable, 0.5f, 0.0f, 5.0f);
            experimentWarmupThrottleNorm = ReadEnvironmentFloat(
                WarmupThrottleEnvironmentVariable, 0.65f, 0.0f, 1.0f);
            experimentMeasurementSeconds = ReadEnvironmentFloat(
                MeasurementSecondsEnvironmentVariable, 1.5f, 0.1f, 10.0f);
            experimentDurationSeconds = ReadEnvironmentFloat(
                DurationEnvironmentVariable, 0.0f, 0.0f, 120.0f);
            active = true;
            Debug.Log("[ModelIdentificationDiagnostics] Writing offline diagnostics to " +
                      Path.GetFullPath(outputDirectory));
        }
        catch (Exception exception)
        {
            Debug.LogError("[ModelIdentificationDiagnostics] Could not open output: " +
                           exception.Message);
            enabled = false;
        }
    }

    private void Start()
    {
        // This command profile is opt-in and exists only for the disposable
        // offline identification player. It is never used by the competition
        // scene and does not modify any physics or vehicle parameters.
        if (active && experimentInitialSpeedMps > 0.0f)
        {
            VehicleRigidBody.velocity = VehicleRigidBody.transform.forward *
                                         experimentInitialSpeedMps;
            VehicleRigidBody.angularVelocity = Vector3.zero;
        }
        experimentElapsedSeconds = 0.0f;
        if (active && experimentActive)
            ApplyExperimentCommand(0.0f);
    }

    private void FixedUpdate()
    {
        if (!active)
            return;

        fixedStep++;
        // VehicleController.Start has applied the configured COM by the first
        // fixed update. The dynamic trace is recorded after the controller's
        // FixedUpdate because this component has a later execution order.
        if (!staticWritten)
        {
            WriteStaticSnapshot();
            staticWritten = true;
        }
        if (fixedStep % recordStride == 0)
            WriteTraceRow();
        if (experimentActive)
        {
            ApplyExperimentCommand(experimentElapsedSeconds);
            bool nearZeroComplete = string.Equals(Environment.GetEnvironmentVariable(
                    ExperimentEnvironmentVariable), NearZeroActuatorExperiment,
                    StringComparison.Ordinal) &&
                experimentElapsedSeconds >= experimentWarmupSeconds +
                    experimentMeasurementSeconds;
            bool timedExperimentComplete = experimentDurationSeconds > 0.0f &&
                experimentElapsedSeconds >= experimentDurationSeconds;
            if (nearZeroComplete || timedExperimentComplete)
            {
                Application.Quit();
            }
            experimentElapsedSeconds += Time.fixedDeltaTime;
        }
    }

    private void ApplyExperimentCommand(float timeSeconds)
    {
        if (string.Equals(Environment.GetEnvironmentVariable(
                ExperimentEnvironmentVariable), DriveExcitationExperiment,
                StringComparison.Ordinal))
        {
            ApplyDriveExcitationCommand(timeSeconds);
            return;
        }
        if (string.Equals(Environment.GetEnvironmentVariable(
                ExperimentEnvironmentVariable), CombinedSlipMatrixExperiment,
                StringComparison.Ordinal))
        {
            ApplyCombinedSlipMatrixCommand(timeSeconds);
            return;
        }
        if (string.Equals(Environment.GetEnvironmentVariable(
                ExperimentEnvironmentVariable), NearZeroActuatorExperiment,
                StringComparison.Ordinal))
        {
            experimentMeasurementActive = timeSeconds >= experimentWarmupSeconds;
            experimentRequestedThrottleNorm = experimentMeasurementActive
                ? experimentThrottleNorm : experimentWarmupThrottleNorm;
            experimentRequestedSteeringNorm = 0.0f;
            Controller.AutonomousThrottle = experimentRequestedThrottleNorm;
            Controller.AutonomousSteering = experimentRequestedSteeringNorm;
            return;
        }
        if (string.Equals(Environment.GetEnvironmentVariable(
                ExperimentEnvironmentVariable), PoweredDriveRepeatExperiment,
                StringComparison.Ordinal))
        {
            ApplyPoweredDriveRepeatCommand(timeSeconds);
            return;
        }

        float throttle;
        float steering;
        if (timeSeconds < 4.0f)
        {
            throttle = Mathf.Lerp(0.08f, 0.45f, timeSeconds / 4.0f);
            steering = 0.0f;
        }
        else if (timeSeconds < 10.0f)
        {
            throttle = 0.45f;
            steering = 0.24f;
        }
        else if (timeSeconds < 16.0f)
        {
            throttle = 0.45f;
            steering = -0.24f;
        }
        else if (timeSeconds < 22.0f)
        {
            throttle = 0.65f;
            steering = 0.32f;
        }
        else if (timeSeconds < 28.0f)
        {
            throttle = 0.65f;
            steering = -0.32f;
        }
        else if (timeSeconds < 34.0f)
        {
            throttle = 0.30f;
            steering = 0.0f;
        }
        else
        {
            throttle = 0.0f;
            steering = 0.0f;
        }
        Controller.AutonomousThrottle = throttle;
        Controller.AutonomousSteering = steering;
    }

    private void ApplyDriveExcitationCommand(float timeSeconds)
    {
        // Straight-line, diagnostic-only torque excitation. The plateaus are
        // long enough to reach distinct speed/load regions while the command
        // remains bounded and the competition scene is never modified.
        float throttle;
        if (timeSeconds < 2.0f)
            throttle = 0.08f;
        else if (timeSeconds < 8.0f)
            throttle = 0.20f;
        else if (timeSeconds < 12.0f)
            throttle = 0.0f;
        else if (timeSeconds < 18.0f)
            throttle = 0.40f;
        else if (timeSeconds < 22.0f)
            throttle = 0.0f;
        else if (timeSeconds < 28.0f)
            throttle = 0.65f;
        else if (timeSeconds < 32.0f)
            throttle = 0.30f;
        else if (timeSeconds < 36.0f)
            throttle = 0.0f;
        else if (timeSeconds < 42.0f)
            throttle = 0.45f;
        else if (timeSeconds < 46.0f)
            throttle = 0.10f;
        else
            throttle = 0.0f;

        Controller.AutonomousThrottle = throttle;
        Controller.AutonomousSteering = 0.0f;
    }

    private void ApplyCombinedSlipMatrixCommand(float timeSeconds)
    {
        // Diagnostic-only paired throttle/steering plateaus. This is a
        // straight/open-plane excitation profile for offline combined-slip
        // identification; it never feeds the competition controller.
        float throttle;
        float steering;
        if (timeSeconds < 4.0f)
        {
            throttle = 0.20f;
            steering = 0.0f;
        }
        else if (timeSeconds < 10.0f)
        {
            throttle = 0.35f;
            steering = 0.08f;
        }
        else if (timeSeconds < 16.0f)
        {
            throttle = 0.45f;
            steering = 0.16f;
        }
        else if (timeSeconds < 22.0f)
        {
            throttle = 0.55f;
            steering = 0.24f;
        }
        else if (timeSeconds < 28.0f)
        {
            throttle = 0.65f;
            steering = 0.32f;
        }
        else if (timeSeconds < 34.0f)
        {
            throttle = 0.65f;
            steering = -0.32f;
        }
        else if (timeSeconds < 40.0f)
        {
            throttle = 0.55f;
            steering = -0.24f;
        }
        else if (timeSeconds < 46.0f)
        {
            throttle = 0.45f;
            steering = -0.16f;
        }
        else if (timeSeconds < 52.0f)
        {
            throttle = 0.35f;
            steering = -0.08f;
        }
        else if (timeSeconds < 58.0f)
        {
            throttle = 0.25f;
            steering = 0.0f;
        }
        else if (timeSeconds < 64.0f)
        {
            throttle = 0.65f;
            steering = 0.32f;
        }
        else if (timeSeconds < 70.0f)
        {
            throttle = 0.65f;
            steering = -0.32f;
        }
        else
        {
            throttle = 0.0f;
            steering = 0.0f;
        }
        Controller.AutonomousThrottle = throttle;
        Controller.AutonomousSteering = steering;
    }

    private void ApplyPoweredDriveRepeatCommand(float timeSeconds)
    {
        // Positive-drive-only excitation for the continuous wheel-state fit.
        // Zero is deliberately absent because VehicleController interprets it
        // as a four-wheel brake event. Repeated downsteps provide excitation of
        // wheel relaxation without mixing that hybrid event into the fit.
        float[] throttleSchedule = {
            0.05f, 0.10f, 0.20f, 0.35f, 0.50f, 0.65f,
            0.50f, 0.35f, 0.20f, 0.10f, 0.05f,
            0.20f, 0.50f, 0.65f, 0.35f, 0.10f, 0.05f,
        };
        const float plateauSeconds = 3.0f;
        int plateau = Mathf.Clamp(
            Mathf.FloorToInt(timeSeconds / plateauSeconds),
            0, throttleSchedule.Length - 1);
        float throttle = throttleSchedule[plateau];
        experimentMeasurementActive = true;
        experimentRequestedThrottleNorm = throttle;
        experimentRequestedSteeringNorm = 0.0f;
        Controller.AutonomousThrottle = throttle;
        Controller.AutonomousSteering = 0.0f;
    }

    private WheelCollider[] ResolveWheels()
    {
        if (Controller != null && Controller.FrontLeftWheelCollider != null &&
            Controller.FrontRightWheelCollider != null &&
            Controller.RearLeftWheelCollider != null &&
            Controller.RearRightWheelCollider != null)
        {
            return new[] {
                Controller.FrontLeftWheelCollider,
                Controller.FrontRightWheelCollider,
                Controller.RearLeftWheelCollider,
                Controller.RearRightWheelCollider,
            };
        }

        WheelCollider[] found = GetComponentsInChildren<WheelCollider>(true);
        Array.Sort(found, delegate(WheelCollider left, WheelCollider right)
        {
            return string.CompareOrdinal(left.name, right.name);
        });
        return found;
    }

    private void WriteStaticSnapshot()
    {
        string outputDirectory = Environment.GetEnvironmentVariable(
            OutputEnvironmentVariable);
        var wheels = new WheelSnapshot[Wheels.Length];
        for (int index = 0; index < Wheels.Length; index++)
        {
            WheelCollider wheel = Wheels[index];
            wheels[index] = new WheelSnapshot {
                name = wheel.name,
                positionVehicleFrame = VehicleRigidBody.transform.InverseTransformPoint(
                    wheel.transform.position),
                localPosition = wheel.transform.localPosition,
                localRotation = wheel.transform.localRotation,
                colliderCenter = wheel.center,
                radius = wheel.radius,
                mass = wheel.mass,
                sprungMass = wheel.sprungMass,
                suspensionDistance = wheel.suspensionDistance,
                wheelDampingRate = wheel.wheelDampingRate,
                forceAppPointDistance = wheel.forceAppPointDistance,
                suspensionSpring = new SuspensionSnapshot(wheel.suspensionSpring),
                forwardFriction = new FrictionSnapshot(wheel.forwardFriction),
                sidewaysFriction = new FrictionSnapshot(wheel.sidewaysFriction),
            };
        }

        Vector3 inertiaTensor = VehicleRigidBody.inertiaTensor;
        Quaternion inertiaRotation = VehicleRigidBody.inertiaTensorRotation;
        Vector3 bodyInertias = ComputeBodyFrameInertiaAxes(
            inertiaTensor, inertiaRotation);
        var snapshot = new StaticSnapshot {
            simulatorBuildTag = Environment.GetEnvironmentVariable(
                BuildTagEnvironmentVariable) ?? "unspecified",
            productName = Application.productName,
            applicationVersion = Application.version,
            unityVersion = Application.unityVersion,
            platform = Application.platform.ToString(),
            sceneName = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
            fixedDeltaTime = Time.fixedDeltaTime,
            maximumAllowedTimestep = Time.maximumDeltaTime,
            timeScale = Time.timeScale,
            gravity = Physics.gravity,
            defaultSolverIterations = Physics.defaultSolverIterations,
            defaultSolverVelocityIterations = Physics.defaultSolverVelocityIterations,
            defaultContactOffset = Physics.defaultContactOffset,
            defaultMaxDepenetrationVelocity = Physics.defaultMaxDepenetrationVelocity,
            defaultMaxAngularSpeed = Physics.defaultMaxAngularSpeed,
            vSyncCount = QualitySettings.vSyncCount,
            targetFrameRate = Application.targetFrameRate,
            experiment = Environment.GetEnvironmentVariable(
                ExperimentEnvironmentVariable) ?? "none",
            experimentInitialSpeedMps = experimentInitialSpeedMps,
            experimentThrottleNorm = experimentThrottleNorm,
            experimentWarmupSeconds = experimentWarmupSeconds,
            experimentWarmupThrottleNorm = experimentWarmupThrottleNorm,
            experimentMeasurementSeconds = experimentMeasurementSeconds,
            experimentDurationSeconds = experimentDurationSeconds,
            vehicle = new VehicleSnapshot {
                objectName = VehicleRigidBody.gameObject.name,
                rootLocalPosition = VehicleRigidBody.transform.localPosition,
                rootLocalRotation = VehicleRigidBody.transform.localRotation,
                wheelbaseM = Controller.Wheelbase * 0.001f,
                trackWidthM = Controller.TrackWidth * 0.001f,
                wheelRadiusControllerM = Controller.WheelRadius,
                steeringLimitRad = Controller.SteeringLimit * Mathf.Deg2Rad,
                steeringRateRadPerSecond = Controller.SteeringRate * Mathf.Deg2Rad,
                motorTorqueNm = Controller.MotorTorque,
                driveType = Controller.driveType.ToString(),
                steerType = Controller.steerType.ToString(),
                rigidBody = new RigidBodySnapshot {
                    mass = VehicleRigidBody.mass,
                    centerOfMass = VehicleRigidBody.centerOfMass,
                    worldCenterOfMass = VehicleRigidBody.worldCenterOfMass,
                    inertiaTensor = inertiaTensor,
                    inertiaTensorRotation = inertiaRotation,
                    bodyInertiaX = bodyInertias.x,
                    bodyInertiaY = bodyInertias.y,
                    bodyInertiaZ = bodyInertias.z,
                    yawAxis = "body_y",
                    yawInertiaBodyFrame = bodyInertias.y,
                    drag = VehicleRigidBody.drag,
                    angularDrag = VehicleRigidBody.angularDrag,
                    maxAngularVelocity = VehicleRigidBody.maxAngularVelocity,
                },
                wheels = wheels,
            },
        };

        File.WriteAllText(
            Path.Combine(outputDirectory, "simulator_parameters.json"),
            JsonUtility.ToJson(snapshot, true) + "\n",
            new UTF8Encoding(false));
    }

    private static Vector3 ComputeBodyFrameInertiaAxes(
        Vector3 principalMoments, Quaternion principalRotation)
    {
        principalRotation = principalRotation.normalized;
        Vector3 axis0 = principalRotation * Vector3.right;
        Vector3 axis1 = principalRotation * Vector3.up;
        Vector3 axis2 = principalRotation * Vector3.forward;
        return new Vector3(
            principalMoments.x * axis0.x * axis0.x +
                principalMoments.y * axis1.x * axis1.x +
                principalMoments.z * axis2.x * axis2.x,
            principalMoments.x * axis0.y * axis0.y +
                principalMoments.y * axis1.y * axis1.y +
                principalMoments.z * axis2.y * axis2.y,
            principalMoments.x * axis0.z * axis0.z +
                principalMoments.y * axis1.z * axis1.z +
                principalMoments.z * axis2.z * axis2.z);
    }

    private string BuildTraceHeader()
    {
        var header = new StringBuilder();
        string[] fields = {
            "fixed_time_s", "fixed_step", "render_frame",
            "root_position_x_m", "root_position_y_m", "root_position_z_m",
            "world_com_x_m", "world_com_y_m", "world_com_z_m",
            "root_rotation_x", "root_rotation_y", "root_rotation_z", "root_rotation_w",
            "world_velocity_x_mps", "world_velocity_y_mps", "world_velocity_z_mps",
            "body_velocity_x_mps", "body_velocity_y_mps", "body_velocity_z_mps",
            "world_angular_velocity_x_radps", "world_angular_velocity_y_radps",
            "world_angular_velocity_z_radps", "body_angular_velocity_x_radps",
            "body_angular_velocity_y_radps", "body_angular_velocity_z_radps",
            "controller_physics_step", "applied_command_sequence",
            "applied_throttle_norm", "applied_steering_norm", "applied_steering_rad",
            "experiment_elapsed_s", "experiment_measurement_active",
            "experiment_requested_throttle_norm", "experiment_requested_steering_norm",
        };
        for (int index = 0; index < fields.Length; index++)
        {
            if (index > 0)
                header.Append(',');
            header.Append(fields[index]);
        }
        string[] wheelFields = {
            "steer_angle_deg", "rpm", "motor_torque_nm", "brake_torque_nm",
            "grounded", "forward_slip", "sideways_slip", "contact_force_n",
            "forward_dir_x", "forward_dir_y", "forward_dir_z",
            "sideways_dir_x", "sideways_dir_y", "sideways_dir_z",
            "contact_point_x_m", "contact_point_y_m", "contact_point_z_m",
            "contact_normal_x", "contact_normal_y", "contact_normal_z",
        };
        for (int wheel = 0; wheel < 4; wheel++)
        {
            for (int field = 0; field < wheelFields.Length; field++)
            {
                header.Append(',');
                header.Append("wheel");
                header.Append(wheel);
                header.Append('_');
                header.Append(wheelFields[field]);
            }
        }
        return header.ToString();
    }

    private void WriteTraceRow()
    {
        var row = new StringBuilder();
        Append(row, Time.fixedTime);
        Append(row, fixedStep);
        Append(row, Time.frameCount);
        Append(row, VehicleRigidBody.position);
        Append(row, VehicleRigidBody.worldCenterOfMass);
        Append(row, VehicleRigidBody.rotation);
        Append(row, VehicleRigidBody.velocity);
        Append(row, VehicleRigidBody.transform.InverseTransformDirection(
            VehicleRigidBody.velocity));
        Append(row, VehicleRigidBody.angularVelocity);
        Append(row, VehicleRigidBody.transform.InverseTransformDirection(
            VehicleRigidBody.angularVelocity));
        Append(row, Controller.PhysicsStep);
        Append(row, Controller.AppliedCommandSequence);
        Append(row, Controller.AppliedThrottle);
        float steeringLimit = Mathf.Max(Controller.SteeringLimit * Mathf.Deg2Rad, 1.0e-6f);
        Append(row, Controller.AppliedSteering / steeringLimit);
        Append(row, Controller.AppliedSteering);
        Append(row, experimentElapsedSeconds);
        Append(row, experimentMeasurementActive ? 1 : 0);
        Append(row, experimentRequestedThrottleNorm);
        Append(row, experimentRequestedSteeringNorm);

        for (int index = 0; index < Wheels.Length; index++)
        {
            WheelCollider wheel = Wheels[index];
            Append(row, wheel.steerAngle);
            Append(row, wheel.rpm);
            Append(row, wheel.motorTorque);
            Append(row, wheel.brakeTorque);
            WheelHit hit;
            bool grounded = wheel.GetGroundHit(out hit);
            Append(row, grounded ? 1 : 0);
            if (grounded)
            {
                Append(row, hit.forwardSlip);
                Append(row, hit.sidewaysSlip);
                Append(row, hit.force);
                Append(row, hit.forwardDir);
                Append(row, hit.sidewaysDir);
                Append(row, hit.point);
                Append(row, hit.normal);
            }
            else
            {
                // 15 columns follow the grounded flag: 3 scalars plus four
                // three-component vectors.
                AppendEmpty(row, 15);
            }
        }
        trace.WriteLine(row.ToString());
        if (fixedStep % 100 == 0)
            trace.Flush();
    }

    private static void Append(StringBuilder builder, Vector3 value)
    {
        Append(builder, value.x);
        Append(builder, value.y);
        Append(builder, value.z);
    }

    private static void Append(StringBuilder builder, Quaternion value)
    {
        Append(builder, value.x);
        Append(builder, value.y);
        Append(builder, value.z);
        Append(builder, value.w);
    }

    private static void Append(StringBuilder builder, float value)
    {
        AppendSeparator(builder);
        builder.Append(value.ToString("R", CultureInfo.InvariantCulture));
    }

    private static void Append(StringBuilder builder, int value)
    {
        AppendSeparator(builder);
        builder.Append(value.ToString(CultureInfo.InvariantCulture));
    }

    private static void Append(StringBuilder builder, ulong value)
    {
        AppendSeparator(builder);
        builder.Append(value.ToString(CultureInfo.InvariantCulture));
    }

    private static void AppendEmpty(StringBuilder builder, int count)
    {
        for (int index = 0; index < count; index++)
            AppendSeparator(builder);
    }

    private static void AppendSeparator(StringBuilder builder)
    {
        if (builder.Length > 0)
            builder.Append(',');
    }

    private static float ReadEnvironmentFloat(string name, float fallback,
                                               float minimum, float maximum)
    {
        string raw = Environment.GetEnvironmentVariable(name);
        float parsed;
        if (string.IsNullOrWhiteSpace(raw) ||
            !float.TryParse(raw, NumberStyles.Float,
                            CultureInfo.InvariantCulture, out parsed) ||
            (float.IsNaN(parsed) || float.IsInfinity(parsed)))
            return fallback;
        return Mathf.Clamp(parsed, minimum, maximum);
    }

    private void OnApplicationQuit()
    {
        CloseTrace();
    }

    private void OnDestroy()
    {
        CloseTrace();
    }

    private void CloseTrace()
    {
        if (trace == null)
            return;
        trace.Flush();
        trace.Dispose();
        trace = null;
    }
}
