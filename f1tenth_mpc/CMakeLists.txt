cmake_minimum_required(VERSION 3.16)
project(f1tenth_mpc LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT MSVC)
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -ffast-math)
endif()

find_package(ament_cmake REQUIRED)

# BachelorProject Riccati-ADMM core. The source/header layout intentionally
# mirrors BachelorProject/MPC: no C++ controller or second vehicle model is
# built in this package.
add_library(mpc_core STATIC
  src/util_math.c
  src/vehicle_model.c
  src/riccati_solver.c
  src/mpc.c)
target_include_directories(mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_core PUBLIC m)

install(TARGETS mpc_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(mpc_core)
ament_package()
