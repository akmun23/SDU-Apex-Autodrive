cmake_minimum_required(VERSION 3.8)
project(f1tenth_mpc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sdu_apex_msgs REQUIRED)

set(MPC_DEPS
  ament_index_cpp rclcpp ackermann_msgs nav_msgs sensor_msgs std_msgs sdu_apex_msgs)

add_library(bachelor_mpc_core STATIC
  vendor/bachelor_mpc/src/mpc.c
  vendor/bachelor_mpc/src/vehicle_model.c
  vendor/bachelor_mpc/src/riccati_solver.c
  vendor/bachelor_mpc/src/util_math.c)
set_target_properties(bachelor_mpc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(bachelor_mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor/bachelor_mpc/include>)
target_compile_features(bachelor_mpc_core PRIVATE c_std_99)
target_link_libraries(bachelor_mpc_core PUBLIC m)

add_library(f1tenth_mpc_core SHARED
  src/vehicle_model.cpp
  src/track_model.cpp
  src/bachelor_mpc_controller.cpp)
target_include_directories(f1tenth_mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(f1tenth_mpc_core PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor/bachelor_mpc/include>)
ament_target_dependencies(f1tenth_mpc_core ${MPC_DEPS})
target_link_libraries(f1tenth_mpc_core bachelor_mpc_core)

add_executable(control_state_node src/control_state_node.cpp)
target_link_libraries(control_state_node f1tenth_mpc_core)
ament_target_dependencies(control_state_node ${MPC_DEPS})

add_executable(mpc_shadow_node src/mpc_shadow_node.cpp)
target_link_libraries(mpc_shadow_node f1tenth_mpc_core)
ament_target_dependencies(mpc_shadow_node ${MPC_DEPS})

add_executable(vehicle_model_replay src/vehicle_model_replay.cpp)
target_link_libraries(vehicle_model_replay f1tenth_mpc_core)
ament_target_dependencies(vehicle_model_replay ${MPC_DEPS})

install(TARGETS bachelor_mpc_core f1tenth_mpc_core control_state_node mpc_shadow_node vehicle_model_replay
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  set(ament_cmake_lint_cmake_FOUND TRUE)
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_vehicle_model test/test_vehicle_model.cpp)
  target_link_libraries(test_vehicle_model f1tenth_mpc_core)
  target_include_directories(test_vehicle_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/bachelor_mpc/include)
  ament_target_dependencies(test_vehicle_model ${MPC_DEPS})

  ament_add_gtest(test_track_model test/test_track_model.cpp)
  target_link_libraries(test_track_model f1tenth_mpc_core)
  ament_target_dependencies(test_track_model ${MPC_DEPS})

  ament_add_gtest(test_bachelor_mpc_controller test/test_bachelor_mpc_controller.cpp)
  target_link_libraries(test_bachelor_mpc_controller f1tenth_mpc_core)
  ament_target_dependencies(test_bachelor_mpc_controller ${MPC_DEPS})
endif()

ament_export_include_directories(include)
ament_export_libraries(f1tenth_mpc_core bachelor_mpc_core)
ament_export_dependencies(${MPC_DEPS})
ament_package()
