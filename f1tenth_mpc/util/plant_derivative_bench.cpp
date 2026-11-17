// Times the 25 ms plant sensitivity kernel (18 tracked coordinates and both
// controls, 20 lanes) with different caps on the force substeps per tick and
// measures how far the capped derivative blocks are from the plant's own
// schedule. Plant arithmetic only: it says nothing about closed-loop behaviour.
//
//   plant_derivative_bench [REPEATS]
//   plant_derivative_bench rollout COUNT   # loops the exact values-only stage (for a profiler)
#include "unity_source_plant.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace f1tenth_mpc;

namespace {

struct Case {
    const char *name;
    double speed_mps, yaw_rate_radps, steering_rad, throttle;
};

UnitySourcePlantState state_for(const Case &c)
{
    UnitySourcePlantState state;
    state.u_mps = c.speed_mps;
    state.yaw_rate_radps = c.yaw_rate_radps;
    state.physical_steering_rad = c.steering_rad;
    for (double &omega : state.wheel_omega_radps) omega = c.speed_mps / 0.059;
    return state;
}

UnitySourcePlantSensitivitySeeds identity_seeds()
{
    UnitySourcePlantSensitivitySeeds seeds{};
    for (int k = 0; k < 18; ++k) seeds.state[k][k] = 1.;
    seeds.steering_target[18] = 1.;
    seeds.throttle[19] = 1.;
    return seeds;
}

double relative_error(const UnitySourcePlantSensitivity &a, const UnitySourcePlantSensitivity &b,
                      int first_output, int last_output)
{
    double difference = 0., reference = 0.;
    for (int j = first_output; j < last_output; ++j)
        for (int lane = 0; lane < 20; ++lane) {
            const double d = a.state[j][lane] - b.state[j][lane];
            difference += d * d;
            reference += b.state[j][lane] * b.state[j][lane];
        }
    return std::sqrt(difference / std::max(reference, 1.e-300));
}

double displacement_error(const UnitySourcePlantSensitivity &a, const UnitySourcePlantSensitivity &b)
{
    const double *ca[] = {a.body_forward_displacement.data(), a.body_lateral_displacement.data(), a.yaw_delta.data()};
    const double *cb[] = {b.body_forward_displacement.data(), b.body_lateral_displacement.data(), b.yaw_delta.data()};
    double difference = 0., reference = 0.;
    for (int m = 0; m < 3; ++m)
        for (int lane = 0; lane < 20; ++lane) {
            difference += (ca[m][lane] - cb[m][lane]) * (ca[m][lane] - cb[m][lane]);
            reference += cb[m][lane] * cb[m][lane];
        }
    return std::sqrt(difference / std::max(reference, 1.e-300));
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc > 2 && std::string(argv[1]) == "rollout") {
        const auto state = state_for({"hairpin", 3., 1.2, .35, .25});
        const UnitySourcePlantInput input{.35, .25};
        UnitySourcePlantOutput rolled{};
        double sink = 0.;
        for (int i = std::atoi(argv[2]); i > 0; --i) {
            unity_source_plant_step(state, input, .025, &rolled);
            sink += rolled.state.u_mps;
        }
        std::printf("%g\n", sink);
        return 0;
    }
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 200;
    const Case cases[] = {
        {"hairpin 3 m/s, r=1.2", 3., 1.2, .35, .25},
        {"hairpin exit 4 m/s", 4., .7, .2, .6},
        {"fast 8 m/s, r=.5", 8., .5, .12, .3},
    };
    const auto seeds = identity_seeds();
    for (const Case &c : cases) {
        const auto state = state_for(c);
        const UnitySourcePlantInput input{c.steering_rad, c.throttle};
        UnitySourcePlantSensitivity exact{};
        if (!unity_source_plant_step_with_sensitivities(state, input, .025, seeds, &exact, 0)) {
            std::printf("%s: the plant rejected the state\n", c.name);
            continue;
        }
        {
            UnitySourcePlantOutput rolled{};
            std::vector<double> times;
            for (int i = 0; i < repeats; ++i) {
                const auto begin = std::chrono::steady_clock::now();
                const bool ok = unity_source_plant_step(state, input, .025, &rolled);
                const auto end = std::chrono::steady_clock::now();
                if (!ok) { times.clear(); break; }
                times.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
            }
            if (!times.empty()) {
                std::sort(times.begin(), times.end());
                std::printf("%s\n  exact rollout (values only): %.1f us/stage\n", c.name, times[times.size() / 2]);
            }
        }
        std::printf("  cap  us/stage  speedup  err(all)  err(u,v,r rows)  err(displacement)\n");
        double exact_us = 0.;
        for (unsigned cap : {0u, 3u, 2u, 1u}) {
            UnitySourcePlantSensitivity result{};
            std::vector<double> times;
            for (int i = 0; i < repeats; ++i) {
                const auto begin = std::chrono::steady_clock::now();
                const bool ok = unity_source_plant_step_with_sensitivities(state, input, .025, seeds, &result, cap);
                const auto end = std::chrono::steady_clock::now();
                if (!ok) { times.clear(); break; }
                times.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
            }
            if (times.empty()) { std::printf("  %3u  rejected\n", cap); continue; }
            std::sort(times.begin(), times.end());
            const double us = times[times.size() / 2];
            if (cap == 0) exact_us = us;
            std::printf("  %3u  %8.0f  %6.2fx  %8.4f  %14.4f  %14.4f\n", cap, us, exact_us / us,
                        relative_error(result, exact, 0, 18), relative_error(result, exact, 0, 3),
                        displacement_error(result, exact));
        }
    }
    return 0;
}
