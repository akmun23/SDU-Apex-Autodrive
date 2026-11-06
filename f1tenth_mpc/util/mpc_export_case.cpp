// Turns one live diagnostics record into a frozen controller problem.
//
//   mpc_export_case DIAGNOSTICS.json RACELINE.csv OUT.case [LATERAL_ACCEL_MPS2 [BRAKING_MPS2]]
//
// DIAGNOSTICS.json is one record of the MPC diagnostics stream (a JSONL line
// from tools/export_mpc_run_report.py, with or without its "payload" wrapper)
// recorded with the `capture_source_benchmark` parameter on. The model
// configuration and objective schedule are taken from that record bit-exactly;
// the optional arguments reproduce the node's curvature speed envelope.
#include "mpc_case.hpp"
#include "mpc_speed_envelope.hpp"
#include "mpc_trajectory_io.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

using nlohmann::json;

namespace {
template <typename T>
void from_hex(const std::string &hex, T *object)
{
    if (hex.size() != 2 * sizeof(T)) throw std::runtime_error("captured struct size mismatch");
    auto *bytes = reinterpret_cast<unsigned char *>(object);
    for (size_t i = 0; i < sizeof(T); ++i)
        bytes[i] = static_cast<unsigned char>(std::stoi(hex.substr(2 * i, 2), nullptr, 16));
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "usage: mpc_export_case DIAGNOSTICS.json RACELINE.csv OUT.case "
                     "[LATERAL_ACCEL [BRAKING]]\n";
        return 2;
    }
    try {
        json record;
        { std::ifstream in(argv[1]); in >> record; }
        const json &live = record.contains("payload") ? record.at("payload") : record;
        sdu_apex::mpc_util::Case c;
        const auto &state = live.at("state"), &internal = live.at("plant_internal_state");
        if (state.size() != 10 || internal.size() != MPC_PLANT_INTERNAL_STATE_COUNT)
            throw std::runtime_error("record lacks the plant state channels");
        auto &s = c.initial_state.plant;
        s.e_y = state[0].get<float>(); s.e_psi = state[1].get<float>();
        s.u = state[2].get<float>(); s.v = state[3].get<float>(); s.r = state[4].get<float>();
        s.delayed_steering_command_1 = state[5].get<float>();
        s.delayed_steering_command_2 = state[6].get<float>();
        s.actual_steering_angle = state[7].get<float>();
        s.delayed_throttle_command_1 = state[8].get<float>();
        s.delayed_throttle_command_2 = state[9].get<float>();
        for (size_t i = 0; i < MPC_PLANT_INTERNAL_STATE_COUNT; ++i)
            s.source_plant_internal_state[i] = internal[i].get<double>();
        const auto &f = live.at("plant_state_flags");
        s.source_plant_initialized = f.at(0).get<int>();
        s.source_suspension_history_initialized = f.at(1).get<int>();
        s.source_imu_state_initialized = f.at(2).get<int>();
        c.progress_m = live.at("progress_m").get<double>();
        c.command_epoch_ns = live.at("candidate_command_stamp_ns").get<std::int64_t>();
        c.prediction_dt = PREDICTION_DT_SECONDS;
        const auto &frozen = live.at("frozen_source_problem");
        from_hex(frozen.at("model_pod_hex").get<std::string>(), &c.model);
        from_hex(frozen.at("objective_pod_hex").get<std::string>(), &c.objective);
        const auto &nominal = live.at("nominal_controls");
        for (int k = 0; k < PREDICTION_HORIZON; ++k) {
            c.controls[k].steering_target_rad = nominal.at(k).at(0).get<float>();
            c.controls[k].throttle_command = nominal.at(k).at(1).get<float>();
        }
        std::ifstream csv(argv[2]);
        if (!csv) throw std::runtime_error("cannot open raceline");
        c.trajectory = f1tenth_mpc::read_mpc_controller_trajectory(csv, &c.lap_length_m);
        if (argc > 4 && std::stod(argv[4]) > 0.)
            f1tenth_mpc::apply_curvature_speed_envelope(
                c.trajectory, c.lap_length_m, std::stod(argv[4]),
                argc > 5 ? std::stod(argv[5]) : 5.);
        if (!sdu_apex::mpc_util::write_case(argv[3], c))
            throw std::runtime_error("cannot write case");
        std::cout << "case " << argv[3] << " progress=" << c.progress_m << " u="
                  << s.u << '\n';
    } catch (const std::exception &error) {
        std::cerr << "mpc_export_case: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
