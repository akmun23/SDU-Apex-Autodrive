// Run the public PhysX 4.1 PxVehicleNoDrive implementation on captured
// Unity WheelCollider parameters. This is a source parity experiment, not a
// claim that NVIDIA's public SDK binary is Unity's patched native plugin.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "PxPhysicsAPI.h"
#include "extensions/PxDefaultAllocator.h"
#include "extensions/PxDefaultCpuDispatcher.h"
#include "extensions/PxDefaultErrorCallback.h"
#include "extensions/PxDefaultSimulationFilterShader.h"
#include "cooking/PxCooking.h"
#include "vehicle/PxVehicleNoDrive.h"
#include "vehicle/PxVehicleSDK.h"
#include "vehicle/PxVehicleTireFriction.h"
#include "vehicle/PxVehicleUpdate.h"
#include "PxVehicleSuspWheelTire4.h"

using json = nlohmann::json;
using namespace physx;

namespace {

PxDefaultAllocator allocator;
PxDefaultErrorCallback errorCallback;

PxVec3 vec3(const json& value) {
    return PxVec3(value.at(0).get<float>(), value.at(1).get<float>(),
                  value.at(2).get<float>());
}

PxQuat quat(const json& value) {
    PxQuat q(value.at(0).get<float>(), value.at(1).get<float>(),
             value.at(2).get<float>(), value.at(3).get<float>());
    q.normalize();
    return q;
}

PxTransform wheelColliderPose(const json& wheel, const json& body,
                              const PxReal jounce, const PxReal steerAngle,
                              const PxReal wheelRotationAngle) {
    const PxVec3 centerOfMass = vec3(body.at("center_of_mass_body_m"));
    const PxVec3 wheelCenterOffset =
        vec3(wheel.at("wheel_center_offset_com_body_m"));
    const PxVec3 suspensionDirection =
        vec3(wheel.at("suspension_direction_body"));
    const PxVec3 position = centerOfMass + wheelCenterOffset
                          - suspensionDirection * jounce;
    const PxQuat steer(steerAngle, PxVec3(0.0f, 1.0f, 0.0f));
    const PxQuat roll(wheelRotationAngle, PxVec3(1.0f, 0.0f, 0.0f));
    return PxTransform(position, steer * roll);
}

std::vector<PxVec3> points3(const json& value) {
    std::vector<PxVec3> points;
    points.reserve(value.size());
    for (const auto& point : value)
        points.push_back(vec3(point));
    if (points.size() < 4)
        throw std::runtime_error("convex hull needs at least four points");
    return points;
}

PxConvexMesh* cookHull(PxCooking& cooking, PxPhysics& physics,
                       const std::vector<PxVec3>& points) {
    PxConvexMeshDesc desc;
    desc.points.count = static_cast<PxU32>(points.size());
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = points.data();
    desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    PxConvexMeshCookingResult::Enum result;
    PxConvexMesh* mesh = cooking.createConvexMesh(
        desc, physics.getPhysicsInsertionCallback(), &result);
    if (!mesh)
        throw std::runtime_error("failed to cook recovered Unity convex hull");
    return mesh;
}

PxShape* attachConvex(PxCooking& cooking, PxPhysics& physics,
                      PxRigidDynamic& actor, PxMaterial& material,
                      const std::vector<PxVec3>& points,
                      const PxTransform& localPose,
                      const PxReal contactOffset) {
    PxConvexMesh* mesh = cookHull(cooking, physics, points);
    PxShape* shape = physics.createShape(PxConvexMeshGeometry(mesh), material, true);
    mesh->release();
    if (!shape)
        throw std::runtime_error("failed to create recovered Unity convex shape");
    shape->setLocalPose(localPose);
    shape->setContactOffset(contactOffset);
    shape->setRestOffset(0.0f);
    shape->setFlag(PxShapeFlag::eSIMULATION_SHAPE, true);
    shape->setFlag(PxShapeFlag::eSCENE_QUERY_SHAPE, true);
    if (!actor.attachShape(*shape)) {
        shape->release();
        throw std::runtime_error("failed to attach recovered Unity convex shape");
    }
    shape->release();
    return shape;
}

float curveResponse(float slip, const std::array<float, 5>& curve) {
    const float x = std::abs(slip);
    const float peakSlip = curve[0];
    const float peakValue = curve[1];
    const float asymptoteSlip = curve[2];
    const float asymptoteValue = curve[3];
    float response;
    if (x < peakSlip) {
        const float t = x / peakSlip;
        response = peakValue * (t + t * t - t * t * t);
    } else if (x < asymptoteSlip) {
        const float t = (x - peakSlip) / (asymptoteSlip - peakSlip);
        response = peakValue + (asymptoteValue - peakValue)
            * (3.0f * t * t - 2.0f * t * t * t);
    } else {
        response = asymptoteValue;
    }
    return std::copysign(response, slip);
}

struct TireCurveData {
    std::array<float, 5> longitudinal{};
    std::array<float, 5> lateral{};
};

void unityCurveTireForce(
    const void* userData, const PxF32 tireFriction,
    const PxF32 longSlip, const PxF32 latSlip, const PxF32,
    const PxF32 wheelOmega, const PxF32 wheelRadius, const PxF32,
    const PxF32, const PxF32, const PxF32 tireLoad,
    const PxF32, const PxF32,
    PxF32& wheelTorque, PxF32& longForce, PxF32& latForce,
    PxF32& aligningMoment) {
    const auto& curve = *static_cast<const TireCurveData*>(userData);
    const float fxCurve = curveResponse(longSlip, curve.longitudinal);
    const float fyCurve = curveResponse(latSlip, curve.lateral);
    const float utilization = (fxCurve / curve.longitudinal[1])
                            * (fxCurve / curve.longitudinal[1])
                            + (fyCurve / curve.lateral[1])
                            * (fyCurve / curve.lateral[1]);
    const float combinedScale = 1.0f / std::sqrt(std::max(1.0f, utilization));
    longForce = tireLoad * tireFriction * fxCurve * combinedScale
              * curve.longitudinal[4];
    latForce = -tireLoad * tireFriction * fyCurve * combinedScale
             * curve.lateral[4];
    wheelTorque = -longForce * wheelRadius;
    aligningMoment = 0.0f;
    (void)wheelOmega;
}

PxQueryHitType::Enum acceptStaticGround(
    PxFilterData queryFilter, PxFilterData hitFilter,
    const void*, PxU32, PxHitFlags&) {
    // Unity's wheel-ray prefilter excludes the vehicle's own colliders. The
    // source replay tags only the ground as wheel-query-drivable (word0=1).
    return queryFilter.word0 == hitFilter.word0
        ? PxQueryHitType::eBLOCK : PxQueryHitType::eNONE;
}

json run(const json& input, const json& shapeAudit) {
    const std::string mode = input.value("mode", std::string("teacher_forced"));
    const bool server = mode == "interactive_recursive_server";
    const bool recursive = mode == "recursive_command_only" || server;
    if (!recursive && mode != "teacher_forced")
        throw std::runtime_error("unknown replay mode: " + mode);
    const json steps = input.contains("steps")
        ? input.at("steps") : json::array({input});
    if (!steps.is_array() || (!server && steps.empty()))
        throw std::runtime_error("input has no steps");
    const std::size_t outputStride = input.value("output_every_n_steps", 1u);
    if (outputStride == 0 || (!server && steps.size() % outputStride != 0))
        throw std::runtime_error(
            "output_every_n_steps must be positive and divide the step count");
    const json& first = recursive ? input.at("initial") : steps.at(0);

    PxFoundation* foundation = PxCreateFoundation(
        PX_PHYSICS_VERSION, allocator, errorCallback);
    if (!foundation) throw std::runtime_error("PxCreateFoundation failed");
    PxTolerancesScale scale;
    PxPhysics* physics = PxCreatePhysics(
        PX_PHYSICS_VERSION, *foundation, scale, true, nullptr);
    if (!physics) throw std::runtime_error("PxCreatePhysics failed");
    if (!PxInitVehicleSDK(*physics))
        throw std::runtime_error("PxInitVehicleSDK failed");
    PxVehicleSetBasisVectors(PxVec3(0, 1, 0), PxVec3(0, 0, 1));

    PxCookingParams cookingParams(scale);
    cookingParams.meshPreprocessParams |= PxMeshPreprocessingFlag::eWELD_VERTICES;
    PxCooking* cooking = PxCreateCooking(PX_PHYSICS_VERSION, *foundation,
                                         cookingParams);
    if (!cooking) throw std::runtime_error("PxCreateCooking failed");

    PxSceneDesc sceneDesc(scale);
    sceneDesc.gravity = PxVec3(0, -9.81f, 0);
    sceneDesc.cpuDispatcher = PxDefaultCpuDispatcherCreate(1);
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    sceneDesc.solverType = PxSolverType::ePGS;
    sceneDesc.frictionType = PxFrictionType::ePATCH;
    const auto& settings = shapeAudit.at("serialized_physics_settings");
    sceneDesc.bounceThresholdVelocity =
        settings.at("bounce_threshold_mps").get<float>();
    sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;
    PxScene* scene = physics->createScene(sceneDesc);
    if (!scene) throw std::runtime_error("scene creation failed");

    PxMaterial* material = physics->createMaterial(0.6f, 0.6f, 0.0f);
    material->setFrictionCombineMode(PxCombineMode::eAVERAGE);
    material->setRestitutionCombineMode(PxCombineMode::eAVERAGE);
    const float contactOffset = settings.at("default_contact_offset_m").get<float>();
    PxRigidStatic* floor = physics->createRigidStatic(
        PxTransform(PxVec3(0.0f, -0.01f, 0.0f)));
    PxShape* floorShape = physics->createShape(
        PxBoxGeometry(500.0f, 0.01f, 500.0f), *material, true);
    floorShape->setContactOffset(contactOffset);
    floorShape->setRestOffset(0.0f);
    floorShape->setQueryFilterData(PxFilterData(1, 0, 0, 0));
    floorShape->setFlag(PxShapeFlag::eSCENE_QUERY_SHAPE, true);
    floor->attachShape(*floorShape);
    floorShape->release();
    scene->addActor(*floor);

    std::array<PxRaycastQueryResult, 4> vehicleRayResults{};
    PxBatchQueryDesc queryDesc(4, 0, 0);
    queryDesc.queryMemory.userRaycastResultBuffer = vehicleRayResults.data();
    queryDesc.preFilterShader = acceptStaticGround;
    PxBatchQuery* batchQuery = scene->createBatchQuery(queryDesc);
    if (!batchQuery) throw std::runtime_error("batch query creation failed");

    const auto& body = first.at("body");
    const float mass = body.at("mass_kg").get<float>();
    PxRigidDynamic* actor = physics->createRigidDynamic(PxTransform(
        vec3(body.at("position_world_m")), quat(body.at("rotation_world_xyzw"))));
    actor->setMass(mass);
    actor->setMassSpaceInertiaTensor(vec3(body.at("inertia_diagonal_kgm2")));
    actor->setCMassLocalPose(PxTransform(
        vec3(body.at("center_of_mass_body_m")),
        quat(body.at("inertia_rotation_xyzw"))));
    actor->setLinearDamping(0.273f);
    actor->setAngularDamping(0.1f);
    actor->setMaxAngularVelocity(
        settings.at("default_max_angular_speed_radps").get<float>());
    actor->setMaxDepenetrationVelocity(
        settings.at("default_max_depenetration_velocity_mps").get<float>());
    actor->setSolverIterationCounts(
        settings.at("default_solver_position_iterations").get<PxU32>(),
        settings.at("default_solver_velocity_iterations").get<PxU32>());
    actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);
    if (recursive) {
        actor->setLinearVelocity(
            vec3(body.at("linear_velocity_world_mps")), false);
        actor->setAngularVelocity(
            vec3(body.at("angular_velocity_world_radps")), false);
    }

    for (const auto& box : shapeAudit.at("active_body_box_colliders"))
        attachConvex(*cooking, *physics, *actor, *material,
                     points3(box.at("corners_body_local_m")),
                     PxTransform(PxIdentity), contactOffset);
    for (const auto& mesh : shapeAudit.at("active_body_convex_meshes_source_hulls"))
        attachConvex(*cooking, *physics, *actor, *material,
                     points3(mesh.at("source_hull_vertices_body_local_m")),
                     PxTransform(PxIdentity), contactOffset);

    static const char* wheelNames[] = {"fl", "fr", "rl", "rr"};
    std::array<PxShape*, 4> wheelShapes{};
    for (PxU32 i = 0; i < 4; ++i) {
        const auto& wheelAudit = shapeAudit.at("active_wheel_mesh_colliders")
                                     .at(wheelNames[i]);
        const auto& matrix = wheelAudit.at("collider_local_to_wheel_pose_matrix");
        std::vector<PxVec3> hullPoints;
        for (const auto& pointJson : wheelAudit.at("source_hull_vertices_mesh_local_m")) {
            const PxVec3 p = vec3(pointJson);
            hullPoints.emplace_back(
                matrix[0][0].get<float>() * p.x + matrix[0][1].get<float>() * p.y
                    + matrix[0][2].get<float>() * p.z + matrix[0][3].get<float>(),
                matrix[1][0].get<float>() * p.x + matrix[1][1].get<float>() * p.y
                    + matrix[1][2].get<float>() * p.z + matrix[1][3].get<float>(),
                matrix[2][0].get<float>() * p.x + matrix[2][1].get<float>() * p.y
                    + matrix[2][2].get<float>() * p.z + matrix[2][3].get<float>());
        }
        const auto& pose = first.at("wheel_shapes").at(i);
        wheelShapes[i] = attachConvex(
            *cooking, *physics, *actor, *material, hullPoints,
            PxTransform(vec3(pose.at("position_body_m")),
                        quat(pose.at("rotation_body_xyzw"))), contactOffset);
    }
    scene->addActor(*actor);

    PxVehicleWheelsSimData* simData = PxVehicleWheelsSimData::allocate(4);
    std::array<TireCurveData, 4> tireData{};
    const auto& initialWheels = first.at("wheel_inputs");
    for (PxU32 i = 0; i < 4; ++i) {
        const auto& in = initialWheels.at(i);
        // Unity's WheelCollider::AddWheelToExistingVehicle explicitly
        // enables each wheel after PxVehicleWheelsSimData allocation.
        simData->enableWheel(i);
        PxVehicleWheelData wheel;
        wheel.mRadius = in.at("radius_m").get<float>();
        wheel.mWidth = in.at("width_m").get<float>();
        wheel.mMass = in.at("mass_kg").get<float>();
        wheel.mMOI = in.at("moment_of_inertia_kgm2").get<float>();
        wheel.mDampingRate = in.at("wheel_damping_rate_nms").get<float>();
        wheel.mMaxBrakeTorque = 1500.0f;
        wheel.mMaxHandBrakeTorque = 0.0f;
        wheel.mMaxSteer = 0.0f;
        simData->setWheelData(i, wheel);

        PxVehicleSuspensionData suspension;
        suspension.mSpringStrength = in.at("spring_rate_npm").get<float>();
        suspension.mSpringDamperRate = in.at("damper_rate_nspm").get<float>();
        suspension.mMaxCompression = in.at("max_compression_m").get<float>();
        suspension.mMaxDroop = in.at("max_droop_m").get<float>();
        suspension.mSprungMass = in.at("sprung_mass_kg").get<float>();
        simData->setSuspensionData(i, suspension);
        simData->setSuspTravelDirection(i, vec3(in.at("suspension_direction_body")));
        simData->setWheelCentreOffset(i, vec3(in.at("wheel_center_offset_com_body_m")));
        const PxVec3 app = vec3(in.at("force_application_offset_com_body_m"));
        simData->setSuspForceAppPointOffset(i, app);
        simData->setTireForceAppPointOffset(i, app);
        simData->setWheelShapeMapping(i, -1);
        simData->setSceneQueryFilterData(i, PxFilterData(1, 0, 0, 0));

        PxVehicleTireData tire;
        tire.mType = 0;
        simData->setTireData(i, tire);
        for (PxU32 j = 0; j < 5; ++j) {
            tireData[i].longitudinal[j] = in.at("forward_curve").at(j).get<float>();
            tireData[i].lateral[j] = in.at("sideways_curve").at(j).get<float>();
        }
    }
    simData->setSubStepCount(5.0f, 5, 1);
    simData->setMinLongSlipDenominator(4.0f);

    PxVehicleNoDrive* vehicle = PxVehicleNoDrive::create(physics, actor, *simData);
    simData->free();
    if (!vehicle) throw std::runtime_error("PxVehicleNoDrive creation failed");
    vehicle->mWheelsDynData.setTireForceShaderFunction(unityCurveTireForce);
    for (PxU32 i = 0; i < 4; ++i)
        vehicle->mWheelsDynData.setTireForceShaderData(i, &tireData[i]);

    const PxMaterial* materials[] = {material};
    PxVehicleDrivableSurfaceType surfaceTypes[] = {{0}};
    PxVehicleDrivableSurfaceToTireFrictionPairs* frictionPairs =
        PxVehicleDrivableSurfaceToTireFrictionPairs::allocate(1, 1);
    frictionPairs->setup(1, 1, materials, surfaceTypes);
    frictionPairs->setTypePairFriction(0, 0, 1.0f);

    PxVehicleWheels* vehicles[] = {vehicle};
    json results = json::array();
    const PxReal dt = first.at("fixed_dt_s").get<float>();
    std::array<PxReal, 4> previousJounce{};
    PxVehicleWheels4DynData* wheelBlocks = nullptr;
    if (recursive) {
        // PxVehicleWheelsDynData has no public jounce setter. The pinned 4.1
        // source stores its four-wheel block pointer as its first field; seed
        // the internal previous-jounce state from the initial WheelCollider
        // pose so the first suspension-damper calculation has the right
        // history instead of the setToRestState sentinel.
        std::memcpy(&wheelBlocks, &vehicle->mWheelsDynData,
                    sizeof(wheelBlocks));
        if (!wheelBlocks)
            throw std::runtime_error("missing PhysX four-wheel dynamics block");
        for (PxU32 i = 0; i < 4; ++i) {
            const auto& wheel = initialWheels.at(i);
            wheelBlocks[0].mJounces[i] =
                wheel.at("initial_pose_jounce_m").get<float>();
            vehicle->mWheelsDynData.setWheelRotationSpeed(
                i, wheel.at("wheel_omega_radps").get<float>());
            vehicle->mWheelsDynData.setWheelRotationAngle(
                i, wheel.at("wheel_rotation_angle_rad").get<float>());
        }
    }
    const auto resetRecursiveState = [&](const json& reset) {
        if (!recursive)
            throw std::runtime_error(
                "only a recursive PhysX server can reset vehicle state");
        const auto& bodyState = reset.at("body");
        const auto& wheels = reset.at("wheel_inputs");
        const auto& shapes = reset.at("wheel_shapes");
        if (!wheels.is_array() || wheels.size() != 4 ||
            !shapes.is_array() || shapes.size() != 4)
            throw std::runtime_error(
                "PhysX reset requires exactly four wheel inputs and shapes");

        // Restore every external plant state used by a rollout. The no-drive
        // reset also clears cached wheel/tire state; WheelCollider jounce and
        // spin are then seeded from the caller's current estimate.
        vehicle->setToRestState();
        actor->setGlobalPose(PxTransform(
            vec3(bodyState.at("position_world_m")),
            quat(bodyState.at("rotation_world_xyzw"))), false);
        actor->setLinearVelocity(
            vec3(bodyState.at("linear_velocity_world_mps")), false);
        actor->setAngularVelocity(
            vec3(bodyState.at("angular_velocity_world_radps")), false);
        for (PxU32 i = 0; i < 4; ++i) {
            const auto& wheel = wheels.at(i);
            const auto& shape = shapes.at(i);
            wheelShapes[i]->setLocalPose(PxTransform(
                vec3(shape.at("position_body_m")),
                quat(shape.at("rotation_body_xyzw"))));
            vehicle->mWheelsDynData.setWheelRotationSpeed(
                i, wheel.at("wheel_omega_radps").get<float>());
            vehicle->mWheelsDynData.setWheelRotationAngle(
                i, wheel.at("wheel_rotation_angle_rad").get<float>());
            previousJounce[i] =
                wheel.at("initial_pose_jounce_m").get<float>();
            wheelBlocks[0].mJounces[i] = previousJounce[i];
        }
        scene->resetFiltering(*actor);
        actor->wakeUp();
    };
    std::size_t stepIndex = 0;
    std::deque<json> pendingServerSteps;
    while (server || stepIndex < steps.size()) {
        json serverStep;
        if (server) {
            if (pendingServerSteps.empty()) {
                std::string line;
                if (!std::getline(std::cin, line))
                    break;
                if (line.empty())
                    continue;
                const json request = json::parse(line);
                const std::string operation =
                    request.value("op", std::string());
                if (operation == "shutdown")
                    break;
                if (operation == "step") {
                    pendingServerSteps.push_back(request);
                } else if (operation == "rollout") {
                    if (request.contains("initial"))
                        resetRecursiveState(request.at("initial"));
                    const auto& stages = request.at("stages");
                    if (!stages.is_array() || stages.empty() ||
                        stages.size() > 128)
                        throw std::runtime_error(
                            "rollout requires 1..128 control stages");
                    std::size_t stageIndex = 0;
                    std::size_t rolloutStepOffset = 0;
                    for (const auto& stage : stages) {
                        constexpr std::size_t kTicksPerControl = 25;
                        const auto profile = stage.find(
                            "wheel_commands_by_tick");
                        const std::size_t ticks = profile != stage.end()
                            ? profile->size()
                            : stage.value("ticks", std::size_t{0});
                        if (ticks != kTicksPerControl ||
                            (profile != stage.end() && !profile->is_array()) ||
                            (profile == stage.end() &&
                                !stage.contains("wheel_commands")))
                            throw std::runtime_error(
                                "each rollout stage must provide exactly 25 1 ms wheel-command ticks");
                        const std::size_t captureRow = stage.value(
                            "capture_row", stepIndex + rolloutStepOffset);
                        const double timeS = stage.value(
                            "time_s",
                            static_cast<double>(stepIndex + rolloutStepOffset) * dt);
                        for (std::size_t tick = 0; tick < ticks; ++tick) {
                            const json& commands = profile != stage.end()
                                ? profile->at(tick)
                                : stage.at("wheel_commands");
                            pendingServerSteps.push_back({
                                {"op", "step"},
                                {"capture_row", captureRow + tick},
                                {"time_s", timeS + tick * dt},
                                {"wheel_commands", commands},
                                {"emit_result", tick + 1 == ticks},
                                {"rollout_stage", stageIndex}
                            });
                        }
                        rolloutStepOffset += ticks;
                        ++stageIndex;
                    }
                } else {
                    throw std::runtime_error(
                        "server input must be step, rollout, or shutdown");
                }
            }
            serverStep = std::move(pendingServerSteps.front());
            pendingServerSteps.pop_front();
        }
        const auto& step = server ? serverStep : steps.at(stepIndex);
        const bool emitResult = server
            ? step.value("emit_result", true)
            : (stepIndex + 1) % outputStride == 0;
        if (!recursive) {
            const auto& b = step.at("body");
            actor->setGlobalPose(PxTransform(
                vec3(b.at("position_world_m")),
                quat(b.at("rotation_world_xyzw"))), false);
            actor->setLinearVelocity(
                vec3(b.at("linear_velocity_world_mps")), false);
            actor->setAngularVelocity(
                vec3(b.at("angular_velocity_world_radps")), false);
        }
        const auto& ws = recursive
            ? step.at("wheel_commands") : step.at("wheel_inputs");
        json shapePosesUsed = json::array();
        for (PxU32 i = 0; i < 4; ++i) {
            const auto& w = ws.at(i);
            vehicle->setSteerAngle(i, w.at("steer_angle_rad").get<float>());
            vehicle->setDriveTorque(i, w.at("drive_torque_nm").get<float>());
            vehicle->setBrakeTorque(i, w.at("brake_torque_nm").get<float>());
            if (!recursive) {
                const auto& pose = step.at("wheel_shapes").at(i);
                wheelShapes[i]->setLocalPose(PxTransform(
                    vec3(pose.at("position_body_m")),
                    quat(pose.at("rotation_body_xyzw"))));
                vehicle->mWheelsDynData.setWheelRotationSpeed(
                    i, w.at("wheel_omega_radps").get<float>());
            } else if (stepIndex > 0) {
                wheelShapes[i]->setLocalPose(wheelColliderPose(
                    initialWheels.at(i), body, previousJounce[i],
                    w.at("steer_angle_rad").get<float>(),
                    vehicle->mWheelsDynData.getWheelRotationAngle(i)));
            }
            if (emitResult) {
                const PxTransform shapePose = wheelShapes[i]->getLocalPose();
                shapePosesUsed.push_back({
                    {"position_body_m", {shapePose.p.x, shapePose.p.y, shapePose.p.z}},
                    {"rotation_body_xyzw", {shapePose.q.x, shapePose.q.y,
                                              shapePose.q.z, shapePose.q.w}}
                });
            }
        }
        PxVehicleSuspensionRaycasts(batchQuery, 1, vehicles, 4,
                                    vehicleRayResults.data());
        json raycasts = json::array();
        if (emitResult) {
            for (const auto& ray : vehicleRayResults) {
                raycasts.push_back({
                    {"has_block", ray.hasBlock},
                    {"query_status", ray.queryStatus},
                    {"hit_position_world_m", ray.hasBlock
                        ? json({ray.block.position.x, ray.block.position.y,
                                ray.block.position.z}) : json(nullptr)},
                    {"hit_normal_world", ray.hasBlock
                        ? json({ray.block.normal.x, ray.block.normal.y,
                                ray.block.normal.z}) : json(nullptr)}
                });
            }
        }
        std::array<PxWheelQueryResult, 4> wheelQueryResults{};
        PxVehicleWheelQueryResult wheelQuery;
        wheelQuery.wheelQueryResults = wheelQueryResults.data();
        wheelQuery.nbWheelQueryResults = static_cast<PxU32>(wheelQueryResults.size());
        PxVehicleUpdates(dt, PxVec3(0, -9.81f, 0), *frictionPairs,
                         1, vehicles, &wheelQuery);
        const PxVec3 vehicleLinear = emitResult
            ? actor->getLinearVelocity() : PxVec3(0.0f);
        const PxVec3 vehicleAngular = emitResult
            ? actor->getAngularVelocity() : PxVec3(0.0f);
        // Unity advances PhysX immediately after VehicleUpdates. In particular,
        // PxVehicleNoDrive's internal suspension/sticky-tire constraints are
        // solved only in this scene step, not in the vehicle update function.
        scene->simulate(dt);
        if (!scene->fetchResults(true))
            throw std::runtime_error("PhysX scene fetchResults failed");
        const PxVec3 linear = emitResult
            ? actor->getLinearVelocity() : PxVec3(0.0f);
        const PxVec3 angular = emitResult
            ? actor->getAngularVelocity() : PxVec3(0.0f);
        const PxTransform pose = emitResult
            ? actor->getGlobalPose() : PxTransform(PxIdentity);
        json wheelResults = json::array();
        for (PxU32 i = 0; i < 4; ++i) {
            if (recursive) {
                const PxReal jounce = wheelQueryResults[i].suspJounce;
                previousJounce[i] = std::isfinite(jounce) ? jounce : 0.0f;
            }
            if (emitResult) {
                wheelResults.push_back({
                    {"omega_radps", vehicle->mWheelsDynData.getWheelRotationSpeed(i)},
                    {"rotation_angle_rad", vehicle->mWheelsDynData.getWheelRotationAngle(i)},
                    {"in_air", wheelQueryResults[i].isInAir},
                    {"jounce_m", wheelQueryResults[i].suspJounce},
                    {"suspension_force_n", wheelQueryResults[i].suspSpringForce},
                    {"longitudinal_slip", wheelQueryResults[i].longitudinalSlip},
                    {"lateral_slip", wheelQueryResults[i].lateralSlip},
                    {"tire_friction", wheelQueryResults[i].tireFriction}
                });
            }
        }
        if (emitResult) {
            json resultRow = {
            {"capture_row", step.at("capture_row")},
            {"time_s", step.at("time_s")},
            {"wheel_shape_local_pose_used", shapePosesUsed},
            {"raycasts", raycasts},
            {"vehicle_stage_linear_velocity_world_mps",
                {vehicleLinear.x, vehicleLinear.y, vehicleLinear.z}},
            {"vehicle_stage_angular_velocity_world_radps",
                {vehicleAngular.x, vehicleAngular.y, vehicleAngular.z}},
            {"scene_solver_linear_velocity_world_mps",
                {linear.x, linear.y, linear.z}},
            {"scene_solver_angular_velocity_world_radps",
                {angular.x, angular.y, angular.z}},
            {"scene_solver_position_world_m",
                {pose.p.x, pose.p.y, pose.p.z}},
            {"scene_solver_rotation_world_xyzw",
                {pose.q.x, pose.q.y, pose.q.z, pose.q.w}},
            {"attached_compound_shape_count", actor->getNbShapes()},
            {"contact_pairs_with_contacts", [&]() {
                PxSimulationStatistics stats;
                scene->getSimulationStatistics(stats);
                return stats.nbDiscreteContactPairsWithContacts;
            }()},
            {"axis_solver_constraints", [&]() {
                PxSimulationStatistics stats;
                scene->getSimulationStatistics(stats);
                return stats.nbAxisSolverConstraints;
            }()},
            {"wheels", wheelResults}
            };
            if (server) {
                json compact = {
                    {"capture_row", resultRow.at("capture_row")},
                    {"time_s", resultRow.at("time_s")},
                    {"position_world_m", resultRow.at("scene_solver_position_world_m")},
                    {"rotation_world_xyzw", resultRow.at("scene_solver_rotation_world_xyzw")},
                    {"linear_velocity_world_mps", resultRow.at("scene_solver_linear_velocity_world_mps")},
                    {"angular_velocity_world_radps", resultRow.at("scene_solver_angular_velocity_world_radps")},
                    {"body_forward_velocity_mps",
                        pose.q.rotateInv(linear).z},
                    {"body_lateral_velocity_mps",
                        -pose.q.rotateInv(linear).x},
                    {"rollout_stage", step.value("rollout_stage", -1)},
                    {"wheels", json::array()},
                    {"wheel_longitudinal_slip", json::array()}
                };
                for (const auto& wheel : resultRow.at("wheels")) {
                    compact["wheels"].push_back({
                        {"omega_radps", wheel.at("omega_radps")},
                        {"rotation_angle_rad", wheel.at("rotation_angle_rad")},
                        {"jounce_m", wheel.at("jounce_m")}
                    });
                    compact["wheel_longitudinal_slip"].push_back(
                        wheel.at("longitudinal_slip"));
                }
                std::cout << compact.dump() << '\n' << std::flush;
            } else {
                results.push_back(std::move(resultRow));
            }
        }
        ++stepIndex;
    }

    json output = {
        {"results", results},
        {"mode", mode},
        {"physics_step_s", dt},
        {"output_stride_steps", outputStride},
        {"simulation_step_s", dt * static_cast<PxReal>(outputStride)},
        {"note", recursive
            ? "Recursive command-only rollout: body and wheel state are initialized once; subsequent steps consume only steering, drive-torque, and brake-torque commands. WheelCollider mesh poses recurse from prior PhysX jounce and wheel rotation-angle outputs. Public PhysX 4.1.2 plus Unity settings/custom tire callback and ordinary scene solve; reconstructed Unity source hulls are recooked from source vertices (not runtime cooked convex data)."
            : "Public PhysX 4.1.2 PxVehicleNoDrive plus Unity WheelCollider settings/custom tire callback and ordinary scene solve; reconstructed Unity source hulls are recooked from source vertices (not runtime cooked convex data). Body truth is reset each row; wheel/vehicle internal dynamics remain sequential."}
    };
    frictionPairs->release();
    vehicle->free();
    batchQuery->release();
    scene->release();
    cooking->release();
    physics->release();
    PxCloseVehicleSDK();
    foundation->release();
    return output;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: physx_native_vehicle_step INPUT.json SHAPE_AUDIT.json OUTPUT.json\n"
                  << "   or: physx_native_vehicle_step --server INITIAL.json SHAPE_AUDIT.json\n";
        return 2;
    }
    try {
        const bool server = std::string(argv[1]) == "--server";
        const char* inputPath = server ? argv[2] : argv[1];
        const char* shapePath = server ? argv[3] : argv[2];
        std::ifstream inputFile(inputPath);
        if (!inputFile) throw std::runtime_error("cannot open input JSON");
        json input;
        inputFile >> input;
        std::ifstream shapeFile(shapePath);
        if (!shapeFile) throw std::runtime_error("cannot open shape audit JSON");
        json shapeAudit;
        shapeFile >> shapeAudit;
        if (server && input.value("mode", std::string()) !=
            "interactive_recursive_server")
            throw std::runtime_error("server initial JSON has the wrong mode");
        const json output = run(input, shapeAudit);
        if (!server) {
            std::ofstream outputFile(argv[3]);
            if (!outputFile) throw std::runtime_error("cannot open output JSON");
            outputFile << output.dump(2) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native vehicle step failed: " << error.what() << '\n';
        return 1;
    }
}
