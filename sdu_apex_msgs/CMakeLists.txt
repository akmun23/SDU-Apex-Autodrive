cmake_minimum_required(VERSION 3.8)
project(sdu_apex_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VehicleControlState.msg"
  "msg/DirectDriveCommand.msg"
  "msg/MpcDiagnostics.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
