cmake_minimum_required(VERSION 3.16)
project(f1tenth_mpc LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
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
  if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" OR
     NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
      "SDU_APEX_MPC_NATIVE_TUNING requires GCC or Clang for C and C++")
  endif()
  # Native ISA vectorization improved the measured Jacobian on the build host.
  # Disable contraction: FMA changed borderline QP statuses/commands on the
  # fixed replay corpus. The default build remains architecture-portable.
  add_compile_options(-march=native -ffp-contract=off)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sdu_apex_msgs REQUIRED)
find_package(OpenMP QUIET COMPONENTS C)

# BachelorProject Riccati-ADMM core.  The package deliberately contains one
# prediction path.  Historic replay plants and report-pinned manifest checks
# are not part of the controller build.
add_library(mpc_core STATIC
  src/mpc_reference.c
  src/vehicle_model.c
  src/unity_source_plant.cpp
  src/unity_source_plant_c.cpp
  src/mpc_linearization.c
  src/mpc_rti.c
  src/riccati_solver.c)
target_include_directories(mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_core PUBLIC m)
if(OpenMP_C_FOUND)
  target_link_libraries(mpc_core PUBLIC OpenMP::OpenMP_C)
endif()
target_compile_definitions(mpc_core PRIVATE MPC_ENABLE_RICCATI_PROFILE)
set_target_properties(mpc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# On-demand saved-state replay against the exact production RTI core. This
# stays outside the default build and is not installed in either runtime
# image; explicitly build the target when diagnosing a captured MPC cycle.
add_executable(mpc_bag_solver_replay EXCLUDE_FROM_ALL
  ../tools/mpc_bag_solver_replay.cpp)
target_link_libraries(mpc_bag_solver_replay PRIVATE mpc_core)

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
add_library(mpc_controller_node SHARED src/mpc_controller_node.cpp)
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
  ament_add_gtest(test_unity_source_plant
    test/test_unity_source_plant.cpp
    src/unity_source_plant.cpp
    src/unity_source_plant_c.cpp)
  target_include_directories(test_unity_source_plant PRIVATE include)
  ament_add_gtest(test_vehicle_model_source_rollout
    test/test_vehicle_model_source_rollout.cpp)
  target_link_libraries(test_vehicle_model_source_rollout mpc_core)
  target_include_directories(test_vehicle_model_source_rollout PRIVATE include)
  ament_add_gtest(test_mpc_control_time_predictor
    test/test_mpc_control_time_predictor.cpp)
  target_link_libraries(test_mpc_control_time_predictor mpc_control_time)
  target_include_directories(test_mpc_control_time_predictor PRIVATE include)
  ament_add_gtest(test_mpc_state_synchronizer
    test/test_mpc_state_synchronizer.cpp)
  target_link_libraries(test_mpc_state_synchronizer mpc_state_sync)
  target_include_directories(test_mpc_state_synchronizer PRIVATE include)
endif()

install(TARGETS mpc_core mpc_state_sync mpc_control_time mpc_controller_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(FILES config/mpc_competition.yaml
  DESTINATION share/${PROJECT_NAME}/config)

ament_export_include_directories(include)
ament_export_libraries(mpc_core)
ament_export_dependencies(ament_index_cpp rclcpp rclcpp_components nav_msgs geometry_msgs
  std_msgs sdu_apex_msgs)

ament_package()
