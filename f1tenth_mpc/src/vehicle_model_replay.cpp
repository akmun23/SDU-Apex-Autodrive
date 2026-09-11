#include "f1tenth_mpc/vehicle_model.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

using f1tenth_mpc::ModelInput;
using f1tenth_mpc::ModelState;
using f1tenth_mpc::VehicleModel;

namespace {

bool parse_initial_state(const std::string &path, ModelState &state)
{
  std::ifstream input(path);
  if (!input) return false;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream stream(line);
    std::string field;
    std::vector<double> values;
    while (std::getline(stream, field, ',')) {
      try {
        values.push_back(std::stod(field));
      } catch (...) {
        values.clear();
        break;  // permits a human-readable CSV header
      }
    }
    if (values.empty()) continue;
    if (values.size() < 6 || values.size() > 7) return false;
    for (const double value : values) {
      if (!std::isfinite(value)) return false;
    }
    state.x = values[0];
    state.y = values[1];
    state.yaw = values[2];
    state.u = values[3];
    state.v = values[4];
    state.r = values[5];
    if (values.size() == 7) {
      state.steering = values[6];
      state.steering_valid = true;
    }
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char **argv)
{
  if (argc < 2 || argc > 4) {
    std::cerr << "usage: vehicle_model_replay controls.csv [output.csv] [initial_state.csv]\n"
              << "controls.csv columns: dt_s,steering_rad,acceleration_mps2\n"
              << "initial_state.csv columns: x_m,y_m,yaw_rad,u_mps,v_mps,r_radps[,steering_rad]\n";
    return 2;
  }
  std::ifstream input(argv[1]);
  if (!input) {
    std::cerr << "could not open " << argv[1] << "\n";
    return 1;
  }
  std::ostream *output_stream = &std::cout;
  std::ofstream output_file;
  if (argc == 3) {
    output_file.open(argv[2]);
    if (!output_file) {
      std::cerr << "could not open " << argv[2] << "\n";
      return 1;
    }
    output_stream = &output_file;
  }
  ModelState initial_state;
  if (argc == 4 && !parse_initial_state(argv[3], initial_state)) {
    std::cerr << "could not parse initial state " << argv[3] << "\n";
    return 1;
  }
  auto &output = *output_stream;
  output << "time_s,x_m,y_m,yaw_rad,u_mps,v_mps,r_radps,steering_rad\n";
  VehicleModel model;
  ModelState state = initial_state;
  double time_s = 0.0;
  output << std::fixed << std::setprecision(9)
         << time_s << ',' << state.x << ',' << state.y << ',' << state.yaw << ','
         << state.u << ',' << state.v << ',' << state.r << ',' << state.steering << '\n';
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream stream(line);
    std::string field;
    double values[3]{};
    bool valid = true;
    for (double &value : values) {
      if (!std::getline(stream, field, ',')) {
        valid = false;
        break;
      }
      try {
        value = std::stod(field);
      } catch (...) {
        valid = false;
        break;
      }
    }
    if (!valid) continue;  // permits a human-readable CSV header
    const double dt = values[0];
    if (!(dt > 0.0)) {
      std::cerr << "invalid dt in replay input\n";
      return 1;
    }
    state = model.step(state, ModelInput{values[1], values[2]}, dt);
    time_s += dt;
    output << time_s << ',' << state.x << ',' << state.y << ',' << state.yaw << ','
           << state.u << ',' << state.v << ',' << state.r << ',' << state.steering << '\n';
  }
  return 0;
}
