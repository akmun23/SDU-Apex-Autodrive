#include "gpu_amcl_cpp/core/ekf_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<gpu_amcl_cpp::EkfNode>());
  rclcpp::shutdown();
  return 0;
}
