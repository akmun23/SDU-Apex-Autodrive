// One-tick native PhysX 4.1 scene-solver replay after a source-composed
// PxVehicleNoDrive update. This deliberately isolates Unity's ordinary rigid
// body/contact stage from the already ported WheelCollider vehicle stage.

#include <cmath>
#include <array>
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

using json = nlohmann::json;
using namespace physx;

namespace {

PxDefaultAllocator gAllocator;
PxDefaultErrorCallback gErrorCallback;

PxVec3 vec3(const json& value) {
    if (!value.is_array() || value.size() != 3)
        throw std::runtime_error("expected a 3-vector");
    return PxVec3(value[0].get<float>(), value[1].get<float>(),
                  value[2].get<float>());
}

PxQuat quat_xyzw(const json& value) {
    if (!value.is_array() || value.size() != 4)
        throw std::runtime_error("expected an xyzw quaternion");
    PxQuat q(value[0].get<float>(), value[1].get<float>(),
             value[2].get<float>(), value[3].get<float>());
    q.normalize();
    return q;
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

PxConvexMesh* cook_hull(PxCooking& cooking, PxPhysics& physics,
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
        throw std::runtime_error("PhysX failed to cook a recovered Unity hull");
    return mesh;
}

PxShape* attach_convex(PxCooking& cooking, PxPhysics& physics,
                       PxRigidDynamic& actor, PxMaterial& material,
                       const std::vector<PxVec3>& points,
                       const PxTransform& local_pose, PxReal contact_offset) {
    PxConvexMesh* mesh = cook_hull(cooking, physics, points);
    PxShape* shape = physics.createShape(PxConvexMeshGeometry(mesh), material,
                                         true);
    mesh->release();
    if (!shape)
        throw std::runtime_error("PhysX failed to create a convex shape");
    shape->setLocalPose(local_pose);
    shape->setContactOffset(contact_offset);
    shape->setRestOffset(0.0f);
    shape->setFlag(PxShapeFlag::eSIMULATION_SHAPE, true);
    shape->setFlag(PxShapeFlag::eSCENE_QUERY_SHAPE, true);
    if (!actor.attachShape(*shape)) {
        shape->release();
        throw std::runtime_error("PhysX failed to attach a recovered hull");
    }
    shape->release();
    return shape;
}

json run(const json& shapes, const json& input) {
    const json steps = input.contains("steps")
        ? input.at("steps") : json::array({input});
    if (!steps.is_array() || steps.empty())
        throw std::runtime_error("step input must contain at least one sample");
    const PxReal dt = steps.at(0).at("fixed_dt_s").get<float>();
    const PxReal contact_offset = shapes.at("serialized_physics_settings")
        .at("default_contact_offset_m").get<float>();
    const auto& body_in = steps.at(0).at("body");

    PxFoundation* foundation = PxCreateFoundation(
        PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
    if (!foundation)
        throw std::runtime_error("PxCreateFoundation failed");
    PxTolerancesScale scale;
    PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation,
                                         scale, true, nullptr);
    if (!physics)
        throw std::runtime_error("PxCreatePhysics failed");
    PxCookingParams cooking_params(scale);
    cooking_params.meshPreprocessParams |= PxMeshPreprocessingFlag::eWELD_VERTICES;
    PxCooking* cooking = PxCreateCooking(PX_PHYSICS_VERSION, *foundation,
                                         cooking_params);
    if (!cooking)
        throw std::runtime_error("PxCreateCooking failed");

    PxSceneDesc scene_desc(scale);
    scene_desc.gravity = vec3(shapes.at("serialized_physics_settings")
                                  .at("gravity_mps2"));
    scene_desc.cpuDispatcher = PxDefaultCpuDispatcherCreate(1);
    scene_desc.filterShader = PxDefaultSimulationFilterShader;
    scene_desc.solverType = PxSolverType::ePGS;
    scene_desc.frictionType = PxFrictionType::ePATCH;
    scene_desc.bounceThresholdVelocity = shapes
        .at("serialized_physics_settings").at("bounce_threshold_mps").get<float>();
    scene_desc.flags |= PxSceneFlag::eENABLE_CCD;
    PxScene* scene = physics->createScene(scene_desc);
    if (!scene)
        throw std::runtime_error("PhysX scene creation failed");

    PxMaterial* material = physics->createMaterial(0.6f, 0.6f, 0.0f);
    material->setFrictionCombineMode(PxCombineMode::eAVERAGE);
    material->setRestitutionCombineMode(PxCombineMode::eAVERAGE);
    PxRigidStatic* floor = physics->createRigidStatic(
        PxTransform(PxVec3(0.0f, -0.01f, 0.0f)));
    PxShape* floor_shape = physics->createShape(
        PxBoxGeometry(500.0f, 0.01f, 500.0f), *material, true);
    floor_shape->setContactOffset(contact_offset);
    floor_shape->setRestOffset(0.0f);
    floor->attachShape(*floor_shape);
    floor_shape->release();
    scene->addActor(*floor);

    PxTransform body_pose(vec3(body_in.at("position_world_m")),
                          quat_xyzw(body_in.at("rotation_world_xyzw")));
    PxRigidDynamic* actor = physics->createRigidDynamic(body_pose);
    actor->setMass(body_in.at("mass_kg").get<float>());
    actor->setMassSpaceInertiaTensor(vec3(body_in.at("inertia_diagonal_kgm2")));
    actor->setCMassLocalPose(PxTransform(
        vec3(body_in.at("center_of_mass_body_m")),
        quat_xyzw(body_in.at("inertia_rotation_xyzw"))));
    actor->setLinearDamping(0.273f);
    actor->setAngularDamping(0.1f);
    actor->setMaxAngularVelocity(shapes.at("serialized_physics_settings")
                                     .at("default_max_angular_speed_radps")
                                     .get<float>());
    actor->setMaxDepenetrationVelocity(shapes
        .at("serialized_physics_settings")
        .at("default_max_depenetration_velocity_mps").get<float>());
    actor->setSolverIterationCounts(
        shapes.at("serialized_physics_settings")
            .at("default_solver_position_iterations").get<PxU32>(),
        shapes.at("serialized_physics_settings")
            .at("default_solver_velocity_iterations").get<PxU32>());
    actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);
    actor->setLinearVelocity(vec3(body_in.at(
        "vehicle_stage_linear_velocity_world_mps")));
    actor->setAngularVelocity(vec3(body_in.at(
        "vehicle_stage_angular_velocity_world_radps")));

    for (const auto& box : shapes.at("active_body_box_colliders")) {
        attach_convex(*cooking, *physics, *actor, *material,
                      points3(box.at("corners_body_local_m")),
                      PxTransform(PxIdentity), contact_offset);
    }
    for (const auto& mesh : shapes.at("active_body_convex_meshes_source_hulls")) {
        attach_convex(*cooking, *physics, *actor, *material,
                      points3(mesh.at("source_hull_vertices_body_local_m")),
                      PxTransform(PxIdentity), contact_offset);
    }
    static const char* wheel_names[] = {"fl", "fr", "rl", "rr"};
    std::array<PxShape*, 4> wheel_shapes{};
    for (PxU32 i = 0; i < 4; ++i) {
        const auto& wheel = shapes.at("active_wheel_mesh_colliders")
                                .at(wheel_names[i]);
        const auto& matrix = wheel.at("collider_local_to_wheel_pose_matrix");
        std::vector<PxVec3> hull_points;
        for (const auto& point : wheel.at("source_hull_vertices_mesh_local_m")) {
            const PxVec3 p = vec3(point);
            hull_points.emplace_back(
                matrix[0][0].get<float>() * p.x + matrix[0][1].get<float>() * p.y
                    + matrix[0][2].get<float>() * p.z + matrix[0][3].get<float>(),
                matrix[1][0].get<float>() * p.x + matrix[1][1].get<float>() * p.y
                    + matrix[1][2].get<float>() * p.z + matrix[1][3].get<float>(),
                matrix[2][0].get<float>() * p.x + matrix[2][1].get<float>() * p.y
                    + matrix[2][2].get<float>() * p.z + matrix[2][3].get<float>());
        }
        const auto& pose = steps.at(0).at("wheel_shapes").at(i);
        wheel_shapes[i] = attach_convex(
            *cooking, *physics, *actor, *material, hull_points,
            PxTransform(vec3(pose.at("position_body_m")),
                        quat_xyzw(pose.at("rotation_body_xyzw"))),
            contact_offset);
    }
    scene->addActor(*actor);
    actor->setActorFlag(PxActorFlag::eDISABLE_SIMULATION, false);
    json results = json::array();
    for (const auto& step : steps) {
        const auto& body = step.at("body");
        actor->setGlobalPose(PxTransform(
            vec3(body.at("position_world_m")),
            quat_xyzw(body.at("rotation_world_xyzw"))));
        actor->setLinearVelocity(vec3(body.at(
            "vehicle_stage_linear_velocity_world_mps")));
        actor->setAngularVelocity(vec3(body.at(
            "vehicle_stage_angular_velocity_world_radps")));
        for (PxU32 i = 0; i < 4; ++i) {
            const auto& wheel = step.at("wheel_shapes").at(i);
            wheel_shapes[i]->setLocalPose(PxTransform(
                vec3(wheel.at("position_body_m")),
                quat_xyzw(wheel.at("rotation_body_xyzw"))));
        }
        actor->wakeUp();
        actor->setWakeCounter(1.0f);
        scene->simulate(dt);
        if (!scene->fetchResults(true))
            throw std::runtime_error("PhysX scene fetchResults failed");

        const PxTransform pose = actor->getGlobalPose();
        const PxVec3 linear = actor->getLinearVelocity();
        const PxVec3 angular = actor->getAngularVelocity();
        PxSimulationStatistics stats;
        scene->getSimulationStatistics(stats);
        results.push_back({
            {"capture_row", step.at("capture_row")},
            {"time_s", step.at("time_s")},
            {"position_world_m", {pose.p.x, pose.p.y, pose.p.z}},
            {"rotation_world_xyzw", {pose.q.x, pose.q.y, pose.q.z, pose.q.w}},
            {"linear_velocity_world_mps", {linear.x, linear.y, linear.z}},
            {"angular_velocity_world_radps", {angular.x, angular.y, angular.z}},
            {"contact_pairs_with_contacts", stats.nbDiscreteContactPairsWithContacts},
            {"axis_solver_constraints", stats.nbAxisSolverConstraints}
        });
    }
    return {
        {"results", results},
        {"attached_shapes", actor->getNbShapes()},
        {"simulation_step_s", dt},
        {"note", "public PhysX 4.1.2 scene solver after source-composed vehicle-stage velocities; each row truth-initializes body state while retaining sequential compound-contact cache; convex hulls recooked from Unity source vertices"}
    };
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: physx_compound_contact_step SHAPE_AUDIT.json STEP_INPUT.json\n";
        return 2;
    }
    try {
        std::ifstream shape_stream(argv[1]);
        std::ifstream input_stream(argv[2]);
        if (!shape_stream || !input_stream)
            throw std::runtime_error("could not open input JSON");
        const json shapes = json::parse(shape_stream);
        const json input = json::parse(input_stream);
        const json output = run(shapes, input);
        std::cout << output.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "native PhysX replay failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
