#pragma once
// A frozen controller problem: everything one mpc_rti_solve_cycle call needs.
// Written by mpc_export_case from a live diagnostics record and read by the
// benchmark. The file is a same-build POD dump (sizes are checked on read);
// regenerate cases after any change to the MPC data structures.
#include "mpc_rti.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace sdu_apex::mpc_util {

struct Case {
    std::int64_t command_epoch_ns{};
    double progress_m{};
    double lap_length_m{};
    float prediction_dt{};
    MpcRtiState_t initial_state{};
    MpcRtiConfiguration_t model{};
    MpcRtiObjectiveSchedule_t objective{};
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    std::vector<MpcTrajectorySample_t> trajectory;
};

static_assert(std::is_trivially_copyable<MpcRtiState_t>::value &&
              std::is_trivially_copyable<MpcRtiConfiguration_t>::value &&
              std::is_trivially_copyable<MpcRtiObjectiveSchedule_t>::value &&
              std::is_trivially_copyable<MpcTrajectorySample_t>::value,
              "cases are POD dumps");

inline constexpr char kCaseMagic[8] = {'M', 'P', 'C', 'C', 'A', 'S', 'E', '3'};

inline bool write_case(const std::string &path, const Case &c)
{
    if (c.trajectory.size() < 3) return false;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::uint32_t sizes[5] = {sizeof(MpcRtiState_t), sizeof(MpcRtiConfiguration_t),
                                    sizeof(MpcRtiObjectiveSchedule_t),
                                    sizeof(MpcTrajectorySample_t),
                                    static_cast<std::uint32_t>(c.trajectory.size())};
    out.write(kCaseMagic, sizeof(kCaseMagic));
    out.write(reinterpret_cast<const char *>(sizes), sizeof(sizes));
    const auto put = [&](const auto &v) {
        out.write(reinterpret_cast<const char *>(&v), sizeof(v));
    };
    put(c.command_epoch_ns); put(c.progress_m); put(c.lap_length_m);
    put(c.prediction_dt); put(c.initial_state); put(c.model); put(c.objective);
    put(c.controls);
    out.write(reinterpret_cast<const char *>(c.trajectory.data()),
              static_cast<std::streamsize>(c.trajectory.size() * sizeof(MpcTrajectorySample_t)));
    return out.good();
}

inline bool read_case(const std::string &path, Case *c)
{
    std::ifstream in(path, std::ios::binary);
    char magic[8];
    std::uint32_t sizes[5];
    if (!in.read(magic, sizeof(magic)) || !in.read(reinterpret_cast<char *>(sizes), sizeof(sizes)) ||
        std::string(magic, 8) != std::string(kCaseMagic, 8) ||
        sizes[0] != sizeof(MpcRtiState_t) || sizes[1] != sizeof(MpcRtiConfiguration_t) ||
        sizes[2] != sizeof(MpcRtiObjectiveSchedule_t) || sizes[3] != sizeof(MpcTrajectorySample_t) ||
        sizes[4] < 3 || sizes[4] > 1000000)
        return false;
    const auto get = [&](auto &v) { in.read(reinterpret_cast<char *>(&v), sizeof(v)); };
    get(c->command_epoch_ns); get(c->progress_m); get(c->lap_length_m);
    get(c->prediction_dt); get(c->initial_state); get(c->model); get(c->objective);
    get(c->controls);
    c->trajectory.resize(sizes[4]);
    in.read(reinterpret_cast<char *>(c->trajectory.data()),
            static_cast<std::streamsize>(sizes[4] * sizeof(MpcTrajectorySample_t)));
    return in.good();
}
} // namespace sdu_apex::mpc_util
