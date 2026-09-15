cmake_minimum_required(VERSION 3.16)
project(f1tenth_mpc LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT MSVC)
  # Keep IEEE floating-point behaviour during model and solver validation.
  # In particular, the core deliberately checks for non-finite values.
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)

# BachelorProject Riccati-ADMM core. The source/header layout intentionally
# mirrors BachelorProject/MPC: no C++ controller or second vehicle model is
# built in this package.
add_library(mpc_core STATIC
  src/util_math.c
  src/vehicle_plant.c
  src/vehicle_model.c
  src/riccati_solver.c
  src/mpc.c
  src/mpc_observer.c)
target_include_directories(mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_core PUBLIC m)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_test(
    NAME vehicle_model_manifest_check
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/../tools/model_id/check_vehicle_model_manifest.py
      --manifest ${CMAKE_CURRENT_SOURCE_DIR}/config/vehicle_model_manifest_v1.json
      --mpc-header ${CMAKE_CURRENT_SOURCE_DIR}/include/mpc_types.h
      --vehicle-source ${CMAKE_CURRENT_SOURCE_DIR}/src/vehicle_model.c
      --mpc-source ${CMAKE_CURRENT_SOURCE_DIR}/src/mpc.c
      --candidate-lateral-report
        ${CMAKE_CURRENT_SOURCE_DIR}/../sdu_apex_autodrive/artifacts/model_id_work/model_fits_v2/lateral_model_benchmark_fixed_iz_damped_20260915.json
      --candidate-longitudinal-report
        ${CMAKE_CURRENT_SOURCE_DIR}/../sdu_apex_autodrive/artifacts/model_id_work/model_fits_v2/longitudinal_model_benchmark_measured_damping_clean_holdout_brake20_lateral16_20260915.json
      --odom-yaml
        ${CMAKE_CURRENT_SOURCE_DIR}/../f1tenth_localization/config/sensor_odometry.yaml
      --odom-cpp-source
        ${CMAKE_CURRENT_SOURCE_DIR}/../f1tenth_localization/src/odometry_observer.cpp
      --odom-python-source
        ${CMAKE_CURRENT_SOURCE_DIR}/../sdu_apex_autodrive/sdu_apex_autodrive/odometry_analysis/reference_observer.py
      --odom-node-source
        ${CMAKE_CURRENT_SOURCE_DIR}/../f1tenth_localization/src/sensor_odometry_node.cpp
      --plant-source ${CMAKE_CURRENT_SOURCE_DIR}/src/vehicle_plant.c
      --plant-python-source
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/model_id/structured_vehicle_plant.py)

  add_executable(vehicle_model_test test/test_vehicle_model.c)
  target_link_libraries(vehicle_model_test PRIVATE mpc_core m)
  add_test(NAME vehicle_model_test COMMAND vehicle_model_test)

  add_executable(vehicle_plant_test test/test_vehicle_plant.c)
  target_link_libraries(vehicle_plant_test PRIVATE mpc_core m)
  add_test(NAME vehicle_plant_test COMMAND vehicle_plant_test)

  add_executable(mpc_observer_test test/test_mpc_observer.c)
  target_link_libraries(mpc_observer_test PRIVATE mpc_core m)
  add_test(NAME mpc_observer_test COMMAND mpc_observer_test)
endif()

install(TARGETS mpc_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

ament_export_include_directories(include)
ament_export_libraries(mpc_core)
ament_package()
