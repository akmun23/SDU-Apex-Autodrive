#ifndef MPC_TRAJECTORY_IO_HPP
#define MPC_TRAJECTORY_IO_HPP

#include "mpc_reference.h"

#include <cmath>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace f1tenth_mpc {

// Startup/offline I/O only. Share the exact schema between production and
// solver replay: silently dropping the source reference changes the problem
// being benchmarked, even when the same CSV filename is supplied.
inline std::vector<MpcTrajectorySample_t> read_mpc_trajectory(
    std::istream &input, double *lap_length)
{
    std::vector<MpcTrajectorySample_t> points;
    std::string line;
    bool witness = false;
    bool input_history = false;
    std::size_t number = 0;
    while (std::getline(input, line)) {
        ++number;
        const auto start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos) continue;
        line.erase(0, start);
        const auto end = line.find_last_not_of(" \t\r");
        line.erase(end+1);
        if (line == "# sdu_apex_source_witness/v1" ||
            line == "# sdu_apex_source_witness/v2") {
            if (!points.empty())
                throw std::runtime_error("source schema after trajectory data");
            witness = true;
            input_history = line == "# sdu_apex_source_witness/v2";
            continue;
        }
        if (line.front() == '#') continue;
        std::vector<double> row;
        std::stringstream fields(line);
        std::string field;
        bool valid = line.back() != ',';
        while (valid && std::getline(fields, field, ',')) {
            try {
                std::size_t consumed = 0;
                const double value = std::stod(field, &consumed);
                valid = std::isfinite(value) &&
                    field.find_first_not_of(" \t\r", consumed) == std::string::npos;
                row.push_back(value);
            } catch (const std::exception &) {
                valid = false;
            }
        }
        if (!valid || row.size() != (input_history ? 16u : witness ? 15u : 9u))
            throw std::runtime_error("invalid trajectory row " +
                std::to_string(number) + ": require corridor or schema-marked source columns");
        MpcTrajectorySample_t point{};
        point.s=row[0]; point.x=row[1]; point.y=row[2];
        point.heading=row[3]; point.curvature=row[4]; point.speed=row[5];
        point.acceleration=row[6]; point.left_bound=row[7]; point.right_bound=row[8];
        if (witness) {
            point.source_witness_valid=1;
            point.source_heading_error=row[9]; point.source_v_rear=row[10];
            point.source_yaw_rate=row[11]; point.source_physical_steering=row[12];
            point.source_steering_target=row[13]; point.source_throttle=row[14];
            point.source_input_history_valid=input_history;
            if (input_history) point.source_previous_steering_target=row[15];
        }
        points.push_back(point);
    }
    if (input.bad()) throw std::runtime_error("trajectory read failure");
    std::size_t count = points.size();
    if (!mpc_trajectory_prepare(points.data(), &count, lap_length))
        throw std::runtime_error("trajectory validation failed");
    points.resize(count);
    return points;
}

}  // namespace f1tenth_mpc
#endif
