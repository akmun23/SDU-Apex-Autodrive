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
    add_executable(test_riccati_solver test/test_riccati_solver.c)
    target_link_libraries(test_riccati_solver PRIVATE mpc_core m)
    if(NOT MSVC)
        target_compile_options(test_riccati_solver PRIVATE -fno-fast-math)
    endif()
    add_test(NAME riccati_solver_regressions COMMAND test_riccati_solver)

    add_executable(test_frenet_linearization test/test_frenet_linearization.c)
    target_link_libraries(test_frenet_linearization PRIVATE mpc_core m)
    if(NOT MSVC)
        target_compile_options(test_frenet_linearization PRIVATE -fno-fast-math)
    endif()
    add_test(NAME frenet_linearization COMMAND test_frenet_linearization)

    add_executable(test_steering_dynamics test/test_steering_dynamics.c)
    target_link_libraries(test_steering_dynamics PRIVATE mpc_core m)
    if(NOT MSVC)
        target_compile_options(test_steering_dynamics PRIVATE -fno-fast-math)
    endif()
    add_test(NAME steering_dynamics COMMAND test_steering_dynamics)
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

    # AutoDRIVE ROS wrapper. Legacy gym and VESC executables are intentionally
    # not built; all actuator conversion stays in sdu_apex_autodrive.
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

    ament_package()
endif()
