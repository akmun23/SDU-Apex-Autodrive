cmake_minimum_required(VERSION 3.16)
project(mpc_riccati C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT MSVC)
    # Keep IEEE finite-value checks valid: simulator/controller inputs must fail
    # neutral on NaN or infinity.
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# ===========================================================================
# MPC Core Library (float32, platform-independent)
# ===========================================================================
add_library(mpc_core STATIC
    src/util_math.c
    src/vehicle_model.c
    src/riccati_solver.c
    src/mpc.c
)
target_include_directories(mpc_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpc_core PUBLIC m)

# ===========================================================================
# Standalone Core Tests
# ===========================================================================
include(CTest)
if(BUILD_TESTING)
    # The old standalone regression sources were part of the removed hardware
    # tree.  Add a test only when its source is present; this keeps the core
    # build honest while the native AutoDRIVE benchmark is developed here.
    foreach(test_name IN ITEMS riccati_solver frenet_linearization steering_dynamics vehicle_model)
        set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/test/test_${test_name}.c")
        if(EXISTS "${test_source}")
            add_executable("test_${test_name}" "${test_source}")
            target_link_libraries("test_${test_name}" PRIVATE mpc_core m)
            if(NOT MSVC)
                target_compile_options("test_${test_name}" PRIVATE -fno-fast-math)
            endif()
            add_test(NAME "${test_name}" COMMAND "test_${test_name}")
        endif()
    endforeach()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/closed_loop_benchmark.c")
        add_executable(test_closed_loop_benchmark test/closed_loop_benchmark.c)
        target_link_libraries(test_closed_loop_benchmark PRIVATE mpc_core m)
        if(NOT MSVC)
            target_compile_options(test_closed_loop_benchmark PRIVATE -fno-fast-math)
        endif()
        add_test(NAME closed_loop_benchmark COMMAND test_closed_loop_benchmark)
    endif()
endif()

# ===========================================================================
# ROS2 Dependencies
# ===========================================================================
find_package(ament_cmake QUIET)

# Build ROS2 node executables only when inside a ROS2 workspace.
# The core library (mpc_core) builds standalone without ROS.
if(ament_cmake_FOUND)
    find_package(Threads REQUIRED)
    find_package(rclc REQUIRED)
    find_package(rcl REQUIRED)
    find_package(rcutils REQUIRED)
    find_package(nav_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(ackermann_msgs REQUIRED)
    find_package(std_msgs REQUIRED)

    # AutoDRIVE ROS wrapper.  Actuator conversion stays in
    # sdu_apex_autodrive, so this node exposes only the team command boundary.
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/mpc_hardware_node.c)
        add_executable(mpc_autodrive_node src/mpc_hardware_node.c)
        target_link_libraries(mpc_autodrive_node mpc_core m Threads::Threads)
        target_include_directories(mpc_autodrive_node PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )
        target_compile_definitions(mpc_autodrive_node PRIVATE _GNU_SOURCE)
        ament_target_dependencies(mpc_autodrive_node
            rclc rcl rcutils nav_msgs geometry_msgs ackermann_msgs std_msgs
        )
        install(TARGETS mpc_autodrive_node DESTINATION lib/${PROJECT_NAME})
    endif()

    # ===================================================================
    # Install
    # ===================================================================
    install(DIRECTORY include/ DESTINATION include)

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/launch)
        install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trajectories)
        install(DIRECTORY trajectories/ DESTINATION share/${PROJECT_NAME}/trajectories)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
        install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/find_passing_weights.py)
        install(PROGRAMS find_passing_weights.py
                DESTINATION share/${PROJECT_NAME}/test)
    endif()

    ament_package()
endif()
