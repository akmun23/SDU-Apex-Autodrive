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
  src/mpc.c)
target_include_directories(mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_core PUBLIC m)

include(CTest)
if(BUILD_TESTING)
  add_executable(vehicle_model_test test/test_vehicle_model.c)
  target_link_libraries(vehicle_model_test PRIVATE mpc_core m)
  add_test(NAME vehicle_model_test COMMAND vehicle_model_test)

  add_executable(vehicle_plant_test test/test_vehicle_plant.c)
  target_link_libraries(vehicle_plant_test PRIVATE mpc_core m)
  add_test(NAME vehicle_plant_test COMMAND vehicle_plant_test)
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
