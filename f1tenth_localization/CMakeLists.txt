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
    set(CMAKE_CUDA_ARCHITECTURES 61 89)
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
find_package(Eigen3 REQUIRED)

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
  src/sensor_packet_assembler.cpp)
target_include_directories(sensor_odometry_node PRIVATE include)
ament_target_dependencies(sensor_odometry_node
  rclcpp sensor_msgs nav_msgs geometry_msgs std_msgs tf2 tf2_ros)

add_executable(odometry_observer_test
  test/odometry_observer_test.cpp
  src/odometry_observer.cpp)
target_include_directories(odometry_observer_test PRIVATE include)
enable_testing()
add_test(NAME odometry_observer_test COMMAND odometry_observer_test)

add_executable(sensor_packet_assembler_test
  test/sensor_packet_assembler_test.cpp
  src/sensor_packet_assembler.cpp)
target_include_directories(sensor_packet_assembler_test PRIVATE include)
add_test(NAME sensor_packet_assembler_test COMMAND sensor_packet_assembler_test)

add_executable(amcl_localization_math_test
  test/amcl_localization_math_test.cpp)
target_include_directories(amcl_localization_math_test PRIVATE gpu_amcl_cpp/include)
target_link_libraries(amcl_localization_math_test Eigen3::Eigen)
add_test(NAME amcl_localization_math_test COMMAND amcl_localization_math_test)

add_executable(odometry_observer_replay
  src/odometry_observer_replay.cpp
  src/odometry_observer.cpp)
target_include_directories(odometry_observer_replay PRIVATE include)

add_executable(ekf_localization_node
  gpu_amcl_cpp/src/main_ekf.cpp
  gpu_amcl_cpp/src/core/ekf_node.cpp)
target_include_directories(ekf_localization_node PRIVATE gpu_amcl_cpp/include)
target_link_libraries(ekf_localization_node Eigen3::Eigen)
ament_target_dependencies(ekf_localization_node
  rclcpp nav_msgs geometry_msgs tf2_ros)

set(LOCALIZATION_INSTALL_TARGETS sensor_odometry_node ekf_localization_node)
if(BUILD_GPU_AMCL)
  list(APPEND LOCALIZATION_INSTALL_TARGETS gpu_amcl_cpp_node)
endif()
install(TARGETS ${LOCALIZATION_INSTALL_TARGETS}
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
