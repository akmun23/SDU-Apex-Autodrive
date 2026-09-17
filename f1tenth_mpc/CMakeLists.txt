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

# BachelorProject Riccati-ADMM core.  The package deliberately contains one
# prediction path.  Historic replay plants and report-pinned manifest checks
# are not part of the controller build.
add_library(mpc_core STATIC
  src/util_math.c
  src/vehicle_model.c
  src/riccati_solver.c
  src/mpc.c)
target_include_directories(mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_core PUBLIC m)
set_target_properties(mpc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The only ROS layer in this package.  It converts the BachelorProject core's
# target-speed command into the existing Ackermann command contract; it does
# not add a second controller or consume simulator-only state.
add_library(mpc_controller_node SHARED src/mpc_controller_node.cpp)
target_link_libraries(mpc_controller_node mpc_core)
ament_target_dependencies(mpc_controller_node
  rclcpp
  rclcpp_components
  nav_msgs
  geometry_msgs
  ackermann_msgs)
target_include_directories(mpc_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
rclcpp_components_register_nodes(mpc_controller_node "f1tenth_mpc::MpcControllerNode")

include(CTest)
if(BUILD_TESTING)
  add_executable(vehicle_model_test test/test_vehicle_model.c)
  target_link_libraries(vehicle_model_test PRIVATE mpc_core m)
  add_test(NAME vehicle_model_test COMMAND vehicle_model_test)

  add_executable(mpc_core_test test/test_mpc.c)
  target_link_libraries(mpc_core_test PRIVATE mpc_core m)
  add_test(NAME mpc_core_test COMMAND mpc_core_test)
endif()

install(TARGETS mpc_core mpc_controller_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

ament_export_include_directories(include)
ament_export_libraries(mpc_core)
ament_export_dependencies(rclcpp rclcpp_components nav_msgs geometry_msgs ackermann_msgs)
ament_package()
