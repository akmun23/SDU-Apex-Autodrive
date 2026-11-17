// Replays recorded sensor packets through the production odometry pipeline and
// scores the estimated body velocity against Unity's own odometry.
//
//   odometry_replay replay.csv [--rows]
//
// replay.csv comes from export_odometry_replay.py. One row per sensor packet.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "f1tenth_localization/odometry_pipeline.hpp"

namespace
{

using f1tenth_localization::OdometryObserverConfig;
using f1tenth_localization::OdometryPipeline;
using f1tenth_localization::SensorPacket;

struct Row
{
  SensorPacket packet;
  double steering_command, throttle_command, steering_feedback_rad, throttle_feedback;
  double truth_u, truth_v_com, truth_r, logged_u;
};

std::vector<Row> read_rows(const std::string & path)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<Row> rows;
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    std::istringstream stream(line);
    std::vector<double> v;
    std::string cell;
    while (std::getline(stream, cell, ',')) v.push_back(std::stod(cell));
    if (v.size() != 18) throw std::runtime_error("bad row: " + line);
    Row r{};
    r.packet.stamp_ns = static_cast<int64_t>(v[0]);
    r.packet.left_angle_rad = v[1];
    r.packet.right_angle_rad = v[2];
    r.packet.ax_mps2 = v[3];
    r.packet.ay_mps2 = v[4];
    r.packet.yaw_rate_radps = v[5];
    r.packet.imu_yaw_rad = v[6];
    r.packet.roll_rad = v[7];
    r.packet.pitch_rad = v[8];
    r.packet.roll_rate_radps = v[9];
    r.packet.pitch_rate_radps = v[10];
    r.steering_command = v[11];
    r.throttle_command = v[12];
    r.steering_feedback_rad = v[13];
    r.throttle_feedback = v[14];
    r.truth_u = v[15];
    r.truth_v_com = v[16];
    r.logged_u = v[17];
    rows.push_back(r);
  }
  return rows;
}

double percentile(std::vector<double> values, double p)
{
  std::sort(values.begin(), values.end());
  return values[static_cast<std::size_t>(p * (values.size() - 1))];
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::cerr << "usage: odometry_replay replay.csv [--rows]\n";
    return 2;
  }
  const bool print_rows = argc > 2 && std::string(argv[2]) == "--rows";
  const auto rows = read_rows(argv[1]);
  OdometryPipeline pipeline{OdometryObserverConfig{}};
  std::vector<double> error, v_error, replay_vs_log;
  for (const auto & row : rows) {
    const auto step = pipeline.step(row.packet, row.steering_command, row.throttle_command,
      row.steering_feedback_rad, row.throttle_feedback);
    const double u = step.estimate.body_u_mps;
    // A collision ends the run's valid truth: the car stops while the estimate keeps going.
    error.push_back(u - row.truth_u);
    v_error.push_back(step.estimate.body_v_com_mps - row.truth_v_com);
    replay_vs_log.push_back(u - row.logged_u);
    if (print_rows) {
      const auto & rear_slip = step.raw_plant.longitudinal_slip;
      std::printf("%zu %.3f %.3f %+.3f ax_meas %+.2f ax_model %+.2f ay_meas %+.2f ay_model %+.2f"
        " slip_rl %+.3f slip_rr %+.3f throttle %.2f pred_err %+.4f innov_l %+.5f innov_r %+.5f\n",
        error.size() - 1, row.truth_u, u, u - row.truth_u,
        row.packet.ax_mps2, step.raw_plant.longitudinal_acceleration_mps2,
        row.packet.ay_mps2, step.raw_plant.lateral_acceleration_mps2,
        rear_slip[2], rear_slip[3], row.throttle_command,
        step.raw_plant.state[0] - row.truth_u,
        step.estimate.encoder_model_rotation_residual_rad[0],
        step.estimate.encoder_model_rotation_residual_rad[1]);
    }
  }
  std::vector<double> magnitude;
  double sum = 0.0, sum_sq = 0.0, v_sum_sq = 0.0, log_max = 0.0;
  for (std::size_t i = 0; i < error.size(); ++i) {
    magnitude.push_back(std::abs(error[i]));
    sum += error[i];
    sum_sq += error[i] * error[i];
    v_sum_sq += v_error[i] * v_error[i];
    log_max = std::max(log_max, std::abs(replay_vs_log[i]));
  }
  const double n = static_cast<double>(error.size());
  std::printf("packets %zu  u error: mean %+.4f rms %.4f p95 %.4f max %.4f  v_com rms %.4f"
    "  |replay-logged| max %.2e\n",
    error.size(), sum / n, std::sqrt(sum_sq / n), percentile(magnitude, 0.95),
    *std::max_element(magnitude.begin(), magnitude.end()), std::sqrt(v_sum_sq / n), log_max);
  return 0;
}
