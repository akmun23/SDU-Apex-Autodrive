#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "f1tenth_localization/odometry_observer.hpp"

int main(int argc, char ** argv)
{
  if (argc != 3) {
    std::cerr << "usage: odometry_observer_replay INPUT.csv OUTPUT.csv\n";
    return 2;
  }
  std::ifstream input(argv[1]);
  std::ofstream output(argv[2]);
  if (!input || !output) {
    std::cerr << "unable to open replay files\n";
    return 2;
  }

  f1tenth_localization::OdometryObserver observer;
  output << "stamp_s,speed_pred_mps,speed_mps,body_u_mps,body_v_mps,x_m,y_m"
    ",wheel_update_used,turn_mode,reset_epoch,timing_degraded\n";
  std::string line;
  bool header = true;
  output << std::setprecision(17);
  while (std::getline(input, line)) {
    if (header) {
      header = false;
      continue;
    }
    if (line.empty()) {
      continue;
    }
    std::stringstream stream(line);
    f1tenth_localization::OdometryObservation observation;
    char comma = 0;
    if (!(stream >> observation.stamp_s >> comma >> observation.left_angle_rad >> comma >>
      observation.right_angle_rad >> comma >> observation.ax_mps2 >> comma >>
      observation.ay_mps2 >> comma >> observation.yaw_rate_radps >> comma >>
      observation.yaw_rad))
    {
      std::cerr << "invalid replay row: " << line << "\n";
      return 3;
    }
    const auto estimate = observer.update(observation);
    output << estimate.stamp_s << ',' << estimate.speed_pred_mps << ',' <<
      estimate.speed_mps << ',' << estimate.body_u_mps << ',' <<
      estimate.body_v_mps << ',' << estimate.x_m << ',' << estimate.y_m << ',' <<
      (estimate.wheel_update_used ? 1 : 0) << ',' <<
      (estimate.turn_mode ? 1 : 0) << ',' << (estimate.reset_epoch ? 1 : 0) << ',' <<
      (estimate.timing_degraded ? 1 : 0) << '\n';
  }
  return 0;
}
