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

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# BachelorProject Riccati-ADMM core.  The package deliberately contains one
# prediction path.  Historic replay plants and report-pinned manifest checks
# are not part of the controller build.
add_library(mpc_core STATIC
  src/mpc_reference.c
  src/vehicle_model.c
  src/mpc_linearization.c
  src/mpc_rti.c
  src/riccati_solver.c)
target_include_directories(mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_core PUBLIC m)
target_compile_definitions(mpc_core PRIVATE MPC_ENABLE_RICCATI_PROFILE)
set_target_properties(mpc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

# The only ROS layer in this package.  It converts the BachelorProject core's
# target-speed command into the existing Ackermann command contract; it does
# not add a second controller or consume simulator-only state.
add_library(mpc_controller_node SHARED src/mpc_controller_node.cpp)
target_link_libraries(mpc_controller_node mpc_core mpc_state_sync mpc_control_time)
ament_target_dependencies(mpc_controller_node
  rclcpp
  rclcpp_components
  nav_msgs
  geometry_msgs
  ackermann_msgs
  std_msgs)
target_include_directories(mpc_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
rclcpp_components_register_nodes(mpc_controller_node "f1tenth_mpc::MpcControllerNode")

install(TARGETS mpc_core mpc_state_sync mpc_control_time mpc_controller_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(FILES config/mpc_competition.yaml
  DESTINATION share/${PROJECT_NAME}/config)

ament_export_include_directories(include)
ament_export_libraries(mpc_core)
ament_export_dependencies(rclcpp rclcpp_components nav_msgs geometry_msgs
  ackermann_msgs std_msgs)
ament_package()
