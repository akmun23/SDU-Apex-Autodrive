cmake_minimum_required(VERSION 3.16)
project(f1tenth_mpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
  # Keep IEEE floating-point behaviour during model and solver validation.
  # In particular, the core deliberately checks for non-finite values.
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

option(SDU_APEX_MPC_NATIVE_TUNING
  "Build this MPC for the CPU doing the build (not portable to other CPUs)"
  OFF)
if(SDU_APEX_MPC_NATIVE_TUNING)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
      "SDU_APEX_MPC_NATIVE_TUNING requires GCC or Clang for C++")
  endif()
  # Native ISA vectorization improved the measured Jacobian on the build host.
  # Disable contraction: FMA changed borderline QP statuses/commands on the
  # fixed replay corpus. The default build remains architecture-portable.
  add_compile_options(-march=native -ffp-contract=off)
endif()

option(SDU_APEX_MPC_RICCATI_PROFILE
  "Collect fine-grained Riccati timing counters (development only)"
  OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sdu_apex_msgs REQUIRED)
find_package(OpenMP QUIET COMPONENTS CXX)

# BachelorProject Riccati-ADMM core.  The package deliberately contains one
# prediction path.  Historic replay plants and report-pinned manifest checks
# are not part of the controller build.
add_library(mpc_core STATIC
  src/mpc_reference.cpp
  src/vehicle_model.cpp
  src/unity_source_plant.cpp
  src/unity_source_plant_c.cpp
  src/mpc_linearization.cpp
  src/mpc_rti.cpp
  src/mpc_rti_qp.cpp
  src/mpc_terminal_viability.cpp
  src/mpc_successor_viability.cpp
  src/mpc_source_first_step_viability.cpp
  src/mpc_source_adjoint.cpp
  src/mpc_source_reduced_sqp.cpp
  src/mpc_source_sensitivity.cpp
  src/mpc_source_nonlinear_optimizer.cpp
  src/riccati_solver.cpp
  src/riccati_factorization.cpp
  src/riccati_scaling.cpp
  src/riccati_eigen_kernels.cpp)
target_include_directories(mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_core PUBLIC m PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mpc_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(SDU_APEX_MPC_RICCATI_PROFILE)
  target_compile_definitions(mpc_core PRIVATE MPC_ENABLE_RICCATI_PROFILE)
endif()
set_target_properties(mpc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# On-demand saved-state replay against the exact production RTI core. This
# development-only source is intentionally absent from the competition Docker
# build context; expose the target when a local checkout includes it.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tools/mpc_bag_solver_replay.cpp")
  add_executable(mpc_bag_solver_replay EXCLUDE_FROM_ALL
    ../tools/mpc_bag_solver_replay.cpp)
  target_link_libraries(mpc_bag_solver_replay PRIVATE mpc_core)

  add_executable(mpc_qp_fixture_replay EXCLUDE_FROM_ALL
    ../tools/mpc_qp_fixture_replay.cpp)
  target_link_libraries(mpc_qp_fixture_replay PRIVATE mpc_core)
  add_executable(mpc_live_failure_replay EXCLUDE_FROM_ALL
    ../tools/mpc_live_failure_replay.cpp)
  target_link_libraries(mpc_live_failure_replay PRIVATE mpc_core)
  # Source-exact second-order reduced SQP diagnostics. These tools use
  # production source plant and frozen objective/geometry, but are NOT
  # linked into the live controller and must never publish controls.
  # The Python second-order QP driver invokes these two explicit binaries.
  add_executable(mpc_offline_advance_source_epoch EXCLUDE_FROM_ALL
    ../tools/mpc_offline_advance_source_epoch.cpp)
  target_include_directories(mpc_offline_advance_source_epoch PRIVATE src)
  target_link_libraries(mpc_offline_advance_source_epoch PRIVATE mpc_core)
  add_executable(mpc_native_source_sqp_controller_epoch EXCLUDE_FROM_ALL
    ../tools/mpc_native_source_sqp_controller_epoch.cpp)
  target_link_libraries(mpc_native_source_sqp_controller_epoch PRIVATE mpc_core)
  add_executable(mpc_native_source_sqp_replay EXCLUDE_FROM_ALL
    ../tools/mpc_native_source_sqp_replay.cpp)
  target_include_directories(mpc_native_source_sqp_replay PRIVATE src)
  target_link_libraries(mpc_native_source_sqp_replay PRIVATE mpc_core)
  add_executable(mpc_offline_source_merit EXCLUDE_FROM_ALL
    ../tools/mpc_offline_source_merit.cpp)
  target_include_directories(mpc_offline_source_merit PRIVATE src)
  target_link_libraries(mpc_offline_source_merit PRIVATE mpc_core)
  add_executable(mpc_offline_exact_source_relinearize EXCLUDE_FROM_ALL
    ../tools/mpc_offline_exact_source_relinearize.cpp)
  target_include_directories(mpc_offline_exact_source_relinearize PRIVATE src)
  target_link_libraries(mpc_offline_exact_source_relinearize PRIVATE mpc_core)
  add_custom_target(mpc_offline_qp_tools
    DEPENDS mpc_bag_solver_replay mpc_qp_fixture_replay
            mpc_live_failure_replay mpc_offline_source_merit
            mpc_offline_exact_source_relinearize)
endif()

# Optional C ABI shared object used by tools/evaluate_mpc_bag.py to replay
# logged plant states and command queues against the production rollout. It is
# not linked into or installed with the controller runtime.
add_library(unity_source_plant_eval SHARED EXCLUDE_FROM_ALL
  src/unity_source_plant.cpp
  src/unity_source_plant_c.cpp)
target_include_directories(unity_source_plant_eval PRIVATE include)

add_library(mpc_state_sync STATIC src/mpc_state_synchronizer.cpp)
target_include_directories(mpc_state_sync PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(mpc_state_sync PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(mpc_control_time STATIC src/mpc_control_time_predictor.cpp)
target_include_directories(mpc_control_time PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_control_time PUBLIC mpc_core mpc_state_sync)
set_target_properties(mpc_control_time PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The only ROS layer in this package. It publishes the MPC's steering and
# normalized-throttle decision through the team actuator command message.
add_library(mpc_controller_node SHARED
  src/mpc_controller_node.cpp
  src/mpc_controller_diagnostics.cpp
  src/mpc_controller_recovery.cpp
  src/mpc_diagnostic_worker.cpp)
target_link_libraries(mpc_controller_node mpc_core mpc_state_sync mpc_control_time)
ament_target_dependencies(mpc_controller_node
  ament_index_cpp
  rclcpp
  rclcpp_components
  nav_msgs
  geometry_msgs
  std_msgs
  sdu_apex_msgs)
target_include_directories(mpc_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
rclcpp_components_register_nodes(mpc_controller_node "f1tenth_mpc::MpcControllerNode")

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  ament_add_gtest(test_mpc_trajectory_io test/test_mpc_trajectory_io.cpp)
  target_link_libraries(test_mpc_trajectory_io mpc_core)
  ament_add_gtest(test_unity_source_plant
    test/test_unity_source_plant.cpp
    src/unity_source_plant.cpp
    src/unity_source_plant_c.cpp)
  target_include_directories(test_unity_source_plant PRIVATE include)
  ament_add_gtest(test_vehicle_model_source_rollout
    test/test_vehicle_model_source_rollout.cpp)
  target_link_libraries(test_vehicle_model_source_rollout mpc_core)
  # The test deliberately exercises internal normal-SQP cost equivalence.
  target_include_directories(test_vehicle_model_source_rollout PRIVATE include src)
  ament_add_gtest(test_mpc_control_time_predictor
    test/test_mpc_control_time_predictor.cpp)
  target_link_libraries(test_mpc_control_time_predictor mpc_control_time)
  target_include_directories(test_mpc_control_time_predictor PRIVATE include)
  ament_add_gtest(test_mpc_state_synchronizer
    test/test_mpc_state_synchronizer.cpp)
  target_link_libraries(test_mpc_state_synchronizer mpc_state_sync)
  target_include_directories(test_mpc_state_synchronizer PRIVATE include)
  ament_add_gtest(test_mpc_drive_command_envelope
    test/test_mpc_drive_command_envelope.cpp)
  target_include_directories(test_mpc_drive_command_envelope PRIVATE include)
  ament_add_gtest(test_mpc_publication_deadline
    test/test_mpc_publication_deadline.cpp)
  target_include_directories(test_mpc_publication_deadline PRIVATE include)
  ament_add_gtest(test_mpc_viability_recovery
    test/test_mpc_viability_recovery.cpp)
  target_include_directories(test_mpc_viability_recovery PRIVATE include)
  ament_add_gtest(test_mpc_world_footprint
    test/test_mpc_world_footprint.cpp)
  target_include_directories(test_mpc_world_footprint PRIVATE include)
  ament_add_gtest(test_mpc_fallback_seed
    test/test_mpc_fallback_seed.cpp)
  target_link_libraries(test_mpc_fallback_seed mpc_core)
  target_include_directories(test_mpc_fallback_seed PRIVATE include)
  ament_add_gtest(test_mpc_speed_envelope
    test/test_mpc_speed_envelope.cpp)
  target_include_directories(test_mpc_speed_envelope PRIVATE include)
  ament_add_gtest(test_mpc_tracking_tolerance
    test/test_mpc_tracking_tolerance.cpp)
  target_include_directories(test_mpc_tracking_tolerance PRIVATE include)
  ament_add_gtest(test_mpc_fallback_episode
    test/test_mpc_fallback_episode.cpp)
  target_include_directories(test_mpc_fallback_episode PRIVATE include)
  ament_add_gtest(test_mpc_sqp_convergence test/test_mpc_sqp_convergence.cpp)
  target_include_directories(test_mpc_sqp_convergence PRIVATE include)
  target_link_libraries(test_mpc_sqp_convergence mpc_core Eigen3::Eigen)
  ament_add_gtest(test_mpc_feasibility_restoration
    test/test_mpc_feasibility_restoration.cpp)
  target_include_directories(test_mpc_feasibility_restoration PRIVATE include)
  target_link_libraries(test_mpc_feasibility_restoration mpc_core)
  ament_add_gtest(test_mpc_prefix_selection
    test/test_mpc_prefix_selection.cpp)
  target_include_directories(test_mpc_prefix_selection PRIVATE include)
  ament_add_gtest(test_mpc_qp_fixture_roundtrip
    test/test_mpc_qp_fixture_roundtrip.cpp)
  target_link_libraries(test_mpc_qp_fixture_roundtrip mpc_core)
  target_include_directories(test_mpc_qp_fixture_roundtrip PRIVATE include)
  ament_add_gtest(test_riccati_denormal_scope
    test/test_riccati_denormal_scope.cpp)
  target_link_libraries(test_riccati_denormal_scope mpc_core)
  target_include_directories(test_riccati_denormal_scope PRIVATE include)
  ament_add_gtest(test_mpc_corridor_polygon test/test_mpc_corridor_polygon.cpp)
  target_link_libraries(test_mpc_corridor_polygon mpc_core Eigen3::Eigen)
  target_include_directories(test_mpc_corridor_polygon PRIVATE include)
  add_test(NAME test_mpc_qp_fixture_schema
    COMMAND ${Python3_EXECUTABLE} -m unittest tools.test_mpc_qp_fixture -v)
  set_tests_properties(test_mpc_qp_fixture_schema PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
  add_test(NAME test_mpc_run_manifest
    COMMAND ${Python3_EXECUTABLE} -m unittest tools.test_seal_mpc_run_manifest -v)
  set_tests_properties(test_mpc_run_manifest PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
  add_test(NAME test_mpc_qp_oracle_refinement_selection
    COMMAND ${Python3_EXECUTABLE} -m unittest
      tools.test_refine_mpc_qp_oracle_cases -v)
  set_tests_properties(test_mpc_qp_oracle_refinement_selection PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()

install(TARGETS mpc_core mpc_state_sync mpc_control_time mpc_controller_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(FILES
  config/mpc_competition.yaml
  config/mpc_practice.yaml
  DESTINATION share/${PROJECT_NAME}/config)

ament_export_include_directories(include)
ament_export_libraries(mpc_core)
ament_export_dependencies(ament_index_cpp rclcpp rclcpp_components nav_msgs geometry_msgs
  std_msgs sdu_apex_msgs)

ament_package()
