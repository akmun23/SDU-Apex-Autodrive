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

# Experimental whole-program optimization of original source AD/QP/rollout.
# The original source and objective are unchanged; require byte-for-byte
# cost/command replay parity before enabling in a simulator image.
option(SDU_APEX_MPC_LTO
  "Enable link-time optimization for native original-source MPC builds"
  ${SDU_APEX_MPC_NATIVE_TUNING})
if(SDU_APEX_MPC_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT mpc_ipo_ok OUTPUT mpc_ipo_reason)
  if(NOT mpc_ipo_ok)
    message(FATAL_ERROR "MPC link-time optimization unavailable: ${mpc_ipo_reason}")
  endif()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

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

# The sole MPC numerical runtime: source-exact nonlinear SQP.
# The original Unity source plant, objective, source AD, physical
# inequalities and condensed constrained QP are the only optimization
# dependencies. No Riccati/ADMM, alternative SQP, restoration controller,
# multi-shooting probe or fallback solver links into mpc_core.
add_library(mpc_core STATIC
  src/mpc_reference.cpp
  src/mpc_source_objective_weights.cpp
  src/vehicle_model.cpp
  src/unity_source_plant.cpp
  src/unity_source_plant_c.cpp
  src/mpc_rti.cpp
  src/mpc_source_sensitivity.cpp
  src/mpc_source_exact_nlp.cpp
  src/mpc_source_exact_nlp_sparse.cpp
  src/mpc_source_exact_nlp_reduced.cpp
  src/mpc_source_exact_compact.cpp
  src/mpc_source_exact_nlp_objective.cpp
  src/mpc_source_exact_qp.cpp
  src/mpc_source_convex_qp.cpp
  src/mpc_source_exact_offline_solver.cpp)
target_include_directories(mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_core PUBLIC m PRIVATE Eigen3::Eigen)
# A 60-control KKT solve must not create an Eigen/OpenMP worker team for
# every small Gram product. Source stage AD retains its own explicit workers.
target_compile_definitions(mpc_core PRIVATE EIGEN_DONT_PARALLELIZE)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mpc_core PUBLIC OpenMP::OpenMP_CXX)
endif()
option(SDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES
  "Evaluate independent exact-source derivative batches concurrently" OFF)
if(SDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES)
  if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "Parallel source sensitivities require OpenMP")
  endif()
  target_compile_definitions(mpc_core PRIVATE
    SDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES=1)
endif()

set_target_properties(mpc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Optional source-only development/qualification tools. The competition
# image intentionally omits the repository tools/ directory; no offline tool
# is a runtime dependency. The authoritative MPC core is always compiled.
foreach(source_tool
    mpc_source_exact_offline_optimize
    mpc_source_exact_live_receding_qualifier
    mpc_source_live_fixture
    mpc_source_benchmark_capture_check
    mpc_source_curvature_audit
    mpc_source_nlp_oracle
    mpc_source_feasibility_witness_probe)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tools/${source_tool}.cpp")
    set(source_tool_exclude EXCLUDE_FROM_ALL)
    if(BUILD_TESTING)
      set(source_tool_exclude)
    endif()
    add_executable(${source_tool} ${source_tool_exclude}
      ../tools/${source_tool}.cpp)
    target_link_libraries(${source_tool} PRIVATE mpc_core Eigen3::Eigen)
  endif()
endforeach()

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
  add_test(NAME test_source_only_architecture
    COMMAND ${Python3_EXECUTABLE} -m unittest
      tools.test_source_only_architecture -v)
  add_test(NAME test_source_sqp_regression
    COMMAND ${Python3_EXECUTABLE} -m unittest
      tools.test_source_sqp_regression -v)
  if(TARGET mpc_source_benchmark_capture_check)
    add_test(NAME test_competition_failure_benchmark
      COMMAND ${CMAKE_COMMAND} -E env
        "MPC_SOURCE_BENCHMARK_TOOLS=$<TARGET_FILE_DIR:mpc_source_live_fixture>"
        "MPC_SOURCE_BENCHMARK_REQUIRE_NATIVE=1"
        ${Python3_EXECUTABLE} -m unittest tools.test_competition_failure_benchmark -v)
    add_test(NAME test_mpc_source_phase1_qualification
      COMMAND ${CMAKE_COMMAND} -E env
        "MPC_SOURCE_BENCHMARK_TOOLS=$<TARGET_FILE_DIR:mpc_source_live_fixture>"
        "MPC_SOURCE_BENCHMARK_REQUIRE_NATIVE=1"
        ${Python3_EXECUTABLE} -m unittest tools.test_qualify_mpc_source_phase1
          tools.test_audit_mpc_source_lagrangian
          tools.test_audit_mpc_source_linearization_domain.ConstraintPathIdentityTest -v)
    set_tests_properties(test_mpc_source_phase1_qualification PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
    set_tests_properties(test_competition_failure_benchmark PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
  endif()
  set_tests_properties(test_source_sqp_regression PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
  set_tests_properties(test_source_only_architecture PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
  ament_add_gtest(test_mpc_trajectory_io test/test_mpc_trajectory_io.cpp)
  ament_add_gtest(test_mpc_source_constraint_identity test/test_mpc_source_constraint_identity.cpp)
  target_link_libraries(test_mpc_source_constraint_identity mpc_core)
  ament_add_gtest(test_mpc_source_convex_qp test/test_mpc_source_convex_qp.cpp)
  target_link_libraries(test_mpc_source_convex_qp mpc_core Eigen3::Eigen)
  target_link_libraries(test_mpc_trajectory_io mpc_core)
  ament_add_gtest(test_unity_source_plant
    test/test_unity_source_plant.cpp
    src/unity_source_plant.cpp
    src/unity_source_plant_c.cpp)
  target_include_directories(test_unity_source_plant PRIVATE include)
  ament_add_gtest(test_mpc_control_time_predictor
    test/test_mpc_control_time_predictor.cpp)
  target_link_libraries(test_mpc_control_time_predictor mpc_control_time)
  ament_add_gtest(test_mpc_state_synchronizer
    test/test_mpc_state_synchronizer.cpp)
  target_link_libraries(test_mpc_state_synchronizer mpc_state_sync)
  ament_add_gtest(test_mpc_drive_command_envelope
    test/test_mpc_drive_command_envelope.cpp)
  ament_add_gtest(test_mpc_publication_deadline
    test/test_mpc_publication_deadline.cpp)
  ament_add_gtest(test_mpc_world_footprint
    test/test_mpc_world_footprint.cpp)
  ament_add_gtest(test_mpc_speed_envelope
    test/test_mpc_speed_envelope.cpp)
  ament_add_gtest(test_mpc_tracking_tolerance
    test/test_mpc_tracking_tolerance.cpp)
  foreach(core_test IN ITEMS
      test_mpc_drive_command_envelope test_mpc_publication_deadline
      test_mpc_world_footprint test_mpc_speed_envelope
      test_mpc_tracking_tolerance)
    target_include_directories(${core_test} PRIVATE include)
  endforeach()
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
