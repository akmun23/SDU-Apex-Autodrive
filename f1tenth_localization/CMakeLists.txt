cmake_minimum_required(VERSION 3.18)
project(f1tenth_localization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_GPU_AMCL "Build the CUDA AMCL node" ON)

if(BUILD_GPU_AMCL)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61 86 89)
  endif()
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sdu_apex_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

set(PHYSX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../external/physx-4.1-source/physx")
set(PHYSX_SHARED_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../external/physx-4.1-source/pxshared")
set(PHYSX_RELEASE_LIBDIR "${PHYSX_ROOT}/bin/linux.clang/release")
set(PHYSX_NATIVE_LIBS
  "${PHYSX_RELEASE_LIBDIR}/libPhysXVehicle_static_64.a"
  "${PHYSX_RELEASE_LIBDIR}/libPhysXExtensions_static_64.a"
  "${PHYSX_RELEASE_LIBDIR}/libPhysXCooking_static_64.a"
  "${PHYSX_RELEASE_LIBDIR}/libPhysX_static_64.a"
  "${PHYSX_RELEASE_LIBDIR}/libPhysXCommon_static_64.a"
  "${PHYSX_RELEASE_LIBDIR}/libPhysXFoundation_static_64.a"
  "${PHYSX_RELEASE_LIBDIR}/libPhysXPvdSDK_static_64.a")
foreach(PHYSX_LIBRARY IN LISTS PHYSX_NATIVE_LIBS)
  if(NOT EXISTS "${PHYSX_LIBRARY}")
    message(FATAL_ERROR "Missing required PhysX source-parity library: ${PHYSX_LIBRARY}")
  endif()
endforeach()

if(BUILD_GPU_AMCL)
  find_package(CUDAToolkit REQUIRED)

  add_library(gpu_amcl_cuda STATIC
    gpu_amcl_cpp/src/cuda/motion_model_kernels.cu
    gpu_amcl_cpp/src/cuda/sensor_model_kernels.cu
    gpu_amcl_cpp/src/cuda/resampling_kernels.cu
    gpu_amcl_cpp/src/cuda/weight_kernels.cu
    gpu_amcl_cpp/src/cuda/estimate_kernels.cu
    gpu_amcl_cpp/src/cuda/cluster_kernels.cu
  )
  target_include_directories(gpu_amcl_cuda PUBLIC gpu_amcl_cpp/include)
  target_link_libraries(gpu_amcl_cuda CUDA::cudart CUDA::curand)
  set_target_properties(gpu_amcl_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  add_library(gpu_amcl_cpp_lib STATIC
    gpu_amcl_cpp/src/helpers/map_utils.cpp
    gpu_amcl_cpp/src/core/motion_model.cpp
    gpu_amcl_cpp/src/core/sensor_model.cpp
    gpu_amcl_cpp/src/core/resampling.cpp
    gpu_amcl_cpp/src/core/particle_filter.cpp
    gpu_amcl_cpp/src/core/amcl_node.cpp
  )
  target_include_directories(gpu_amcl_cpp_lib PUBLIC gpu_amcl_cpp/include)
  target_link_libraries(gpu_amcl_cpp_lib gpu_amcl_cuda Eigen3::Eigen CUDA::cudart CUDA::curand)
  ament_target_dependencies(gpu_amcl_cpp_lib
    ament_index_cpp rclcpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros std_msgs)

  add_executable(gpu_amcl_cpp_node gpu_amcl_cpp/src/main_amcl.cpp)
  target_link_libraries(gpu_amcl_cpp_node gpu_amcl_cpp_lib)
  ament_target_dependencies(gpu_amcl_cpp_node rclcpp)
endif()

add_executable(sensor_odometry_node
  src/sensor_odometry_node.cpp
  src/odometry_observer.cpp
  src/sensor_packet_assembler.cpp
  src/physx_odom_runtime.cpp)
target_include_directories(sensor_odometry_node PRIVATE include)
ament_target_dependencies(sensor_odometry_node
  ament_index_cpp rclcpp sensor_msgs nav_msgs geometry_msgs std_msgs tf2 tf2_ros
  sdu_apex_msgs)

add_executable(physx_native_vehicle_step
  src/physx_native_vehicle_step.cpp
  src/physx_glibc_compat.cpp)
target_include_directories(physx_native_vehicle_step PRIVATE
  "${PHYSX_ROOT}/include"
  "${PHYSX_SHARED_ROOT}/include"
  "${PHYSX_ROOT}/source/physxvehicle/src")
target_compile_definitions(physx_native_vehicle_step PRIVATE
  PX_PHYSX_STATIC_LIB NDEBUG)
target_link_libraries(physx_native_vehicle_step PRIVATE
  -Wl,--start-group ${PHYSX_NATIVE_LIBS} -Wl,--end-group dl pthread)

add_executable(ekf_localization_node
  gpu_amcl_cpp/src/main_ekf.cpp
  gpu_amcl_cpp/src/core/ekf_node.cpp)
target_include_directories(ekf_localization_node PRIVATE gpu_amcl_cpp/include)
target_link_libraries(ekf_localization_node Eigen3::Eigen)
ament_target_dependencies(ekf_localization_node
  rclcpp nav_msgs geometry_msgs tf2_ros)

set(LOCALIZATION_INSTALL_TARGETS
  sensor_odometry_node physx_native_vehicle_step ekf_localization_node)
if(BUILD_GPU_AMCL)
  list(APPEND LOCALIZATION_INSTALL_TARGETS gpu_amcl_cpp_node)
endif()
install(TARGETS ${LOCALIZATION_INSTALL_TARGETS}
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_sensor_packet_assembler
    test/test_sensor_packet_assembler.cpp
    src/sensor_packet_assembler.cpp)
  target_include_directories(test_sensor_packet_assembler PRIVATE include)

  ament_add_gtest(test_odometry_observer_diagnostics
    test/test_odometry_observer_diagnostics.cpp
    src/odometry_observer.cpp)
  target_include_directories(test_odometry_observer_diagnostics PRIVATE include)
endif()

ament_package()
