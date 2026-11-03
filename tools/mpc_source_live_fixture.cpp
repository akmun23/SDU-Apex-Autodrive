#include "mpc_qp_failure_context.hpp"
#include "mpc_trajectory_io.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using json=nlohmann::json;
namespace fs=std::filesystem;
int main(int argc,char**argv) {
    if(argc!=6) {
        std::cerr<<"usage: mpc_source_live_fixture LIVE_DIAG_JSON PRACTICE_YAML_JSON RACELINE_CSV BASE_CONTEXT_DIR OUT_DIR\n";
        return 2;
    }
    try {
        json live,params;
        { std::ifstream input(argv[1]);input>>live; }
        { std::ifstream input(argv[2]);input>>params; }
        sdu_apex::qp_fixture::FailureReplayContext context{};
        if(!sdu_apex::qp_fixture::read_failure_context(
                (fs::path(argv[4])/"source_context.mctx").string(),&context))
            throw std::runtime_error("cannot load base physical source replay context");
        const auto &state=live.at("state"),&internal=live.at("plant_internal_state");
        if(state.size()!=10 || internal.size()!=MPC_PLANT_INTERNAL_STATE_COUNT)
            throw std::runtime_error("invalid captured live plant state channels");
        auto &s=context.initial_state.plant;
        s.e_y=state[0].get<float>();
        s.e_psi=state[1].get<float>();
        s.u=state[2].get<float>();
        s.v=state[3].get<float>();
        s.r=state[4].get<float>();
        s.delayed_steering_command_1=state[5].get<float>();
        s.delayed_steering_command_2=state[6].get<float>();
        s.actual_steering_angle=state[7].get<float>();
        s.delayed_throttle_command_1=state[8].get<float>();
        s.delayed_throttle_command_2=state[9].get<float>();
        for(size_t i=0;i<MPC_PLANT_INTERNAL_STATE_COUNT;++i)
            s.source_plant_internal_state[i]=internal[i].get<double>();
        auto f=live.at("plant_state_flags");
        if(f.size()!=3)throw std::runtime_error("invalid plant flags");
        s.source_plant_initialized=f[0].get<int>();
        s.source_suspension_history_initialized=f[1].get<int>();
        s.source_imu_state_initialized=f[2].get<int>();
        context.initial_state.incoming_tracking_jacobian_valid=0;
        context.initial_state.incoming_transition_valid=0;
        context.initial_state.incoming_branch_flags=0;
        context.progress_m=live.at("progress_m").get<double>();
        context.command_epoch_ns=live.at("candidate_command_stamp_ns").get<int64_t>();
        context.prediction_dt=PREDICTION_DT_SECONDS;
        context.terminal_viability=0;
        context.world_sweep_was_configured=0;
        std::ifstream csv(argv[3]);
        if(!csv)throw std::runtime_error("cannot open actual raceline");
        context.trajectory=f1tenth_mpc::read_mpc_trajectory(csv,&context.lap_length_m);
        // Exact live mode only consumes raceline geometry/speed/corridor,
        // never the IPOPT witness controls or states.
        for(auto &point:context.trajectory)point.source_witness_valid=0;
        auto &m=context.model;
        m.world_sweep_check=nullptr;m.world_sweep_context=nullptr;
        m.world_map_rejection_only=0;
        // Override all exposed live tuning parameters: the original
        // competition context supplies only the common source physical
        // parameters; it does not supply the practice objective.
#define OVERRIDE(name) if(params.contains(#name)) m.name=params[#name].get<decltype(m.name)>()
        OVERRIDE(weight_e_y);OVERRIDE(weight_e_psi);
        OVERRIDE(tracking_lateral_tolerance_m);
        OVERRIDE(tracking_heading_tolerance_rad);
        OVERRIDE(weight_u);OVERRIDE(weight_u_overspeed);
        OVERRIDE(overspeed_curvature_gain_m);
        OVERRIDE(underspeed_curvature_gain_m);
        OVERRIDE(weight_longitudinal_slip);
        OVERRIDE(longitudinal_slip_curvature_gain_m);
        OVERRIDE(weight_v);OVERRIDE(weight_r);
        OVERRIDE(weight_steering_command);OVERRIDE(weight_steering_target);
        OVERRIDE(weight_throttle_command);OVERRIDE(weight_steering_slew);
        OVERRIDE(weight_steering_correction_variation);
        OVERRIDE(steering_correction_variation_epsilon_rad);
        OVERRIDE(weight_throttle_slew);
        OVERRIDE(command_increment_qp_enabled);
        OVERRIDE(max_steering_command_step_rad);
        OVERRIDE(max_throttle_command_step);
        OVERRIDE(startup_nominal_throttle_seed);
        OVERRIDE(terminal_multiplier);
        OVERRIDE(max_speed_mps);OVERRIDE(max_steering_rad);
        OVERRIDE(corridor_margin_m);
        OVERRIDE(first_prediction_corridor_margin_m);
        OVERRIDE(corridor_preview_halfwidth_m);
        OVERRIDE(nonlinear_corridor_tolerance_m);
        OVERRIDE(reference_clearance_reserve_m);
        OVERRIDE(weight_source_corridor_reserve);
        OVERRIDE(weight_source_lateral_slip);
#undef OVERRIDE
        m.active_speed_ceiling_mps=
            live.at("active_speed_ceiling_mps").get<float>();
        const auto &controls=live.at("nominal_controls");
        if(controls.size()!=PREDICTION_HORIZON)
            throw std::runtime_error("invalid live cold nominal seed");
        for(size_t i=0;i<PREDICTION_HORIZON;++i) {
            context.r1_nominal[i].steering_target_rad=controls[i][0].get<float>();
            context.r1_nominal[i].throttle_command=controls[i][1].get<float>();
        }
        context.r1_captured=1;
        auto target=fs::path(argv[5]);
        fs::create_directories(target);
        if(!sdu_apex::qp_fixture::write_failure_context(
                (target/"source_context.mctx").string(),context))
            throw std::runtime_error("failed to write live cold context");
        fs::copy_file(fs::path(argv[4])/"r1.fqp",target/"r1.fqp",
                      fs::copy_options::overwrite_existing);
        std::cout<<"COLD_SOURCE_FIXTURE progress="<<context.progress_m
                 <<" u="<<s.u
                 <<" states=26 controls=60 trajectory="<<context.trajectory.size()
                 <<" dt="<<context.prediction_dt
                 <<" config_from_live_yaml=true "
                 <<" sensor_flags="<<s.source_plant_initialized
                 <<s.source_suspension_history_initialized
                 <<s.source_imu_state_initialized<<"\n";
        return 0;
    }catch(const std::exception&e) {
        std::cerr<<"SOURCE_FIXTURE_ERROR: "<<e.what()<<"\n";return 1;
    }
}
