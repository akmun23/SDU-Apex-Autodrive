#ifndef F1TENTH_MPC_YAW_RESPONSE_SURFACE_IO_HPP
#define F1TENTH_MPC_YAW_RESPONSE_SURFACE_IO_HPP

#include "vehicle_model.h"

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace f1tenth_mpc
{

inline MpcYawRateResponseSurface_t load_yaw_response_surface_csv(
    const std::string & path, float blend_q_start, float blend_q_end,
    float speed_blend_margin_mps, float low_speed_blend_margin_mps,
    float low_speed_support_fadeout_mps = 0.0f,
    float high_speed_support_fadein_mps = 0.0f,
    float steering_blend_start_rad = 0.0f,
    float steering_blend_full_start_rad = 0.0f,
    float steering_blend_full_end_rad = 0.0f,
    float steering_blend_end_rad = 0.0f,
    float hold_response_time_constant_s = 0.0f,
    float hold_rate_full_radps = 0.0f,
    float hold_rate_zero_radps = 0.0f)
{
    static constexpr std::array<float, MPC_YAW_SURFACE_Q_KNOTS> kSteeringKnots{
        0.15f, 0.20f, 0.21f, 0.22f, 0.23f,
        0.25f, 0.30f, 0.35f, 0.42f, 0.50f};

    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open yaw response surface: " + path);

    MpcYawRateResponseSurface_t result{};
    result.enabled = 1;
    result.blend_q_start = blend_q_start;
    result.blend_q_end = blend_q_end;
    result.speed_blend_margin_mps = speed_blend_margin_mps;
    result.low_speed_blend_margin_mps = low_speed_blend_margin_mps;
    result.low_speed_support_fadeout_mps = low_speed_support_fadeout_mps;
    result.high_speed_support_fadein_mps = high_speed_support_fadein_mps;
    result.steering_blend_start_rad = steering_blend_start_rad;
    result.steering_blend_full_start_rad = steering_blend_full_start_rad;
    result.steering_blend_full_end_rad = steering_blend_full_end_rad;
    result.steering_blend_end_rad = steering_blend_end_rad;
    result.hold_response_time_constant_s = hold_response_time_constant_s;
    result.hold_rate_full_radps = hold_rate_full_radps;
    result.hold_rate_zero_radps = hold_rate_zero_radps;

    std::string line;
    bool header_seen = false;
    size_t row_index = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!header_seen) {
            header_seen = true;
            continue;
        }

        float speed = 0.0f;
        int turn_sign = 0;
        float steering = 0.0f;
        float demand_q = 0.0f;
        float yaw_rate_abs = 0.0f;
        char comma0 = 0, comma1 = 0, comma2 = 0, comma3 = 0;
        std::istringstream row(line);
        if (!(row >> speed >> comma0 >> turn_sign >> comma1 >> steering >> comma2
                  >> demand_q >> comma3 >> yaw_rate_abs) ||
            comma0 != ',' || comma1 != ',' || comma2 != ',' || comma3 != ',') {
            throw std::runtime_error("malformed yaw response row " +
                                     std::to_string(row_index + 1) + " in " + path);
        }
        row >> std::ws;
        if (!row.eof()) throw std::runtime_error("extra data in yaw response row");
        if (row_index >= static_cast<size_t>(MPC_YAW_SURFACE_SPEED_KNOTS *
                                             MPC_YAW_SURFACE_TURN_DIRECTIONS *
                                             MPC_YAW_SURFACE_Q_KNOTS)) {
            throw std::runtime_error("too many yaw response samples in " + path);
        }

        const size_t speed_index = row_index /
            (MPC_YAW_SURFACE_TURN_DIRECTIONS * MPC_YAW_SURFACE_Q_KNOTS);
        const size_t within_speed = row_index %
            (MPC_YAW_SURFACE_TURN_DIRECTIONS * MPC_YAW_SURFACE_Q_KNOTS);
        const size_t direction_index = within_speed / MPC_YAW_SURFACE_Q_KNOTS;
        const size_t knot_index = within_speed % MPC_YAW_SURFACE_Q_KNOTS;
        const int expected_sign = direction_index == 0 ? -1 : 1;
        if (turn_sign != expected_sign ||
            std::abs(steering - kSteeringKnots[knot_index]) > 1.0e-5f ||
            !std::isfinite(speed) || !std::isfinite(demand_q) ||
            !std::isfinite(yaw_rate_abs) || demand_q <= 0.0f ||
            yaw_rate_abs <= 0.0f) {
            throw std::runtime_error("invalid yaw response sample order/value in " + path);
        }
        if (knot_index == 0 && direction_index == 0) {
            result.speed_mps[speed_index] = speed;
        } else if (std::abs(result.speed_mps[speed_index] - speed) > 1.0e-5f) {
            throw std::runtime_error("inconsistent speed knot in yaw response file");
        }
        result.q[speed_index][direction_index][knot_index] = demand_q;
        result.yaw_rate_abs_rps[speed_index][direction_index][knot_index] = yaw_rate_abs;
        ++row_index;
    }

    const size_t rows_per_speed = static_cast<size_t>(
        MPC_YAW_SURFACE_TURN_DIRECTIONS * MPC_YAW_SURFACE_Q_KNOTS);
    if (!header_seen || row_index % rows_per_speed != 0 ||
        row_index < 3 * rows_per_speed) {
        throw std::runtime_error("yaw response surface row count mismatch: " +
                                 std::to_string(row_index));
    }
    result.speed_count = static_cast<int>(row_index / rows_per_speed);
    return result;
}

}  // namespace f1tenth_mpc

#endif  // F1TENTH_MPC_YAW_RESPONSE_SURFACE_IO_HPP
