cmake_minimum_required(VERSION 3.8)
project(f1tenth_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options("$<$<CONFIG:Release>:-O3>")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sdu_apex_msgs REQUIRED)

set(F1TENTH_CONTROL_DEPENDENCIES
  rclcpp rclcpp_components sensor_msgs nav_msgs std_msgs sdu_apex_msgs)
set(F1TENTH_CONTROL_INCLUDE_DIRS
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

add_library(f1tenth_common SHARED
  src/common/math_utils.cpp
  src/common/lidar_processor.cpp)
target_include_directories(f1tenth_common PUBLIC ${F1TENTH_CONTROL_INCLUDE_DIRS})
ament_target_dependencies(f1tenth_common ${F1TENTH_CONTROL_DEPENDENCIES})

add_library(ftg_algorithm SHARED src/algorithms/follow_the_gap.cpp)
target_include_directories(ftg_algorithm PUBLIC ${F1TENTH_CONTROL_INCLUDE_DIRS})
target_link_libraries(ftg_algorithm f1tenth_common)
ament_target_dependencies(ftg_algorithm ${F1TENTH_CONTROL_DEPENDENCIES})

add_library(ftg_node_component SHARED src/nodes/ftg_node.cpp)
target_include_directories(ftg_node_component PUBLIC ${F1TENTH_CONTROL_INCLUDE_DIRS})
target_link_libraries(ftg_node_component ftg_algorithm f1tenth_common)
ament_target_dependencies(ftg_node_component ${F1TENTH_CONTROL_DEPENDENCIES})
rclcpp_components_register_nodes(ftg_node_component "f1tenth_control::FTGNode")

install(TARGETS f1tenth_common ftg_algorithm ftg_node_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_follow_the_gap test/test_follow_the_gap.cpp)
  target_link_libraries(test_follow_the_gap ftg_algorithm)
  ament_target_dependencies(test_follow_the_gap ${F1TENTH_CONTROL_DEPENDENCIES})
endif()

ament_export_include_directories(include)
ament_export_libraries(f1tenth_common ftg_algorithm ftg_node_component)
ament_export_dependencies(${F1TENTH_CONTROL_DEPENDENCIES})
ament_package()
