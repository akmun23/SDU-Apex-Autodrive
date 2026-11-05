#include "mpc_qp_failure_context.hpp"
#include "mpc_trajectory_io.hpp"
#include "mpc_source_exact_nlp.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <memory>
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
        // Older diagnostic streams omitted the three plant flags.
        // Preserve the source fixture's initialized flags and REQUIRE
        // exact live cost/constraint parity before accepting the replay.
        if(live.contains("plant_state_flags")) {
            const auto &f=live.at("plant_state_flags");
            if(f.size()!=3)throw std::runtime_error("invalid plant flags");
            s.source_plant_initialized=f[0].get<int>();
            s.source_suspension_history_initialized=f[1].get<int>();
            s.source_imu_state_initialized=f[2].get<int>();
        }
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
        // These full input-range limits are required by the source NLP
        // even though the old incremental QP has been deleted. Do NOT
        // rebuild a fixture with the old zero-initialized rates.
        // Match ROS node's canonical missing-parameter defaults, not
        // unrelated historical competition replay values.
        // The live practice profile omits all of these intentionally.
        m.tracking_lateral_tolerance_m=0.f;
        m.tracking_heading_tolerance_rad=0.f;
        m.longitudinal_slip_curvature_gain_m=0.f;
        m.weight_steering_correction_variation=0.f;
        m.steering_correction_variation_epsilon_rad=.02f;
        m.near_wall_tracking_reserve_m=0.f;
        m.weight_source_corridor_reserve=0.f;
        m.weight_source_lateral_slip=0.f;
        m.command_increment_qp_enabled=0;
        m.source_reachable_steering_qp=0;
        m.max_steering_command_step_rad=2.f*m.max_steering_rad;
        m.max_throttle_command_step=
            MPC_THROTTLE_MAX-MPC_THROTTLE_MIN_DRIVE;
        m.active_speed_ceiling_mps=
            live.contains("active_speed_ceiling_mps")?
            live.at("active_speed_ceiling_mps").get<float>():
            params.at("max_speed_mps").get<float>();
        const auto &controls=live.at("nominal_controls");
        if(controls.size()!=PREDICTION_HORIZON)
            throw std::runtime_error("invalid live cold nominal seed");
        for(size_t i=0;i<PREDICTION_HORIZON;++i) {
            context.r1_nominal[i].steering_target_rad=controls[i][0].get<float>();
            context.r1_nominal[i].throttle_command=controls[i][1].get<float>();
        }
        context.r1_captured=1;
        // The source nonlinear MPC freezes a SAME-EPOCH original source
        // objective schedule from its actual nominal, not the schedule
        // captured on an unrelated offline case. Recompute here, and
        // reject actuator-projected controls rather than silently
        // changing what was optimized in the live controller.
        auto nominal=std::make_unique<MpcRtiNominal_t>();
        MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
        if(!mpc_rti_build_nominal_same_epoch(
            &context.initial_state,context.progress_m,nullptr,
            context.r1_nominal,PREDICTION_HORIZON,
            context.trajectory.data(),context.trajectory.size(),
            context.lap_length_m,context.prediction_dt,
            PREDICTION_HORIZON,&context.model,nominal.get(),references))
            throw std::runtime_error("cannot reconstruct live nominal");
        for(int k=0;k<PREDICTION_HORIZON;++k)
            if(nominal->controls[k].steering_target_rad!=
                    context.r1_nominal[k].steering_target_rad ||
               nominal->controls[k].throttle_command!=
                    context.r1_nominal[k].throttle_command)
                throw std::runtime_error(
                    "live nominal was actuator projected; cannot "
                    "claim exact live primal parity");
        if(!mpc_rti_build_objective_schedule(
            nominal.get(),references,&context.model,&context.objective))
            throw std::runtime_error("cannot reconstruct live source score");
        // Re-score the ORIGINAL source nonlinear objective and physical
        // inequality residuals independently. A reconstructed state/ABI
        // is only usable if it reproduces the recorded live nominal
        // exactly; do not silently guess missing source hidden state.
        MpcRtiCycleConfiguration_t validation{};
        validation.model=context.model;
        validation.objective_schedule=&context.objective;
        auto primal=std::make_unique<sdu_apex::source_nlp::Decision>();
        auto score=std::make_unique<sdu_apex::source_nlp::Evaluation>();
        if(!sdu_apex::source_nlp::decision_from_nominal(
               *nominal,primal.get()) ||
           !sdu_apex::source_nlp::evaluate_existing_source_nominal(
               context.initial_state,context.progress_m,
               context.trajectory.data(),context.trajectory.size(),
               context.lap_length_m,context.prediction_dt,validation,
               *nominal,*primal,score.get()))
            throw std::runtime_error(
                "cannot independently re-score reconstructed live source NLP");
        const auto &reported=live.at("solver");
        const double expected_cost=
            reported.at("source_exact_nlp_original_cost_initial").get<double>();
        const double expected_margin=reported.at(
            "source_exact_nlp_original_minimum_corner_margin_m").get<double>();
        const double desired_clearance=
            params.at("source_exact_nlp_extra_clearance_m").get<double>();
        double residual=0.;
        for(const auto &c:score->corridor_corners)
            for(int i=0;i<4;++i) {
                const double l=std::min(0.,c.left[i]-desired_clearance);
                const double r=std::min(0.,c.right[i]-desired_clearance);
                residual+=.5*(l*l+r*r);
            }
        for(const auto &rate:score->command_rate)
            for(double m:rate) {
                const double v=std::min(0.,m);
                residual+=.5*v*v;
            }
        const double expected_violation=reported.at(
            "source_exact_nlp_original_constraint_violation").get<double>();
        if(std::abs(score->objective-expected_cost)>
                1.e-5*std::max(1.,std::abs(expected_cost)) ||
           std::abs(score->minimum_corridor_slack_m-expected_margin)>1.e-4 ||
           std::abs(residual-expected_violation)>
                1.e-5*std::max(1.,expected_violation))
            throw std::runtime_error(
                "live fixture cannot claim source cost, clearance and "
                "physical violation parity: cost_delta="+
                std::to_string(score->objective-expected_cost)+
                " margin_delta="+std::to_string(
                    score->minimum_corridor_slack_m-expected_margin)+
                " violation_delta="+std::to_string(
                    residual-expected_violation));
        auto target=fs::path(argv[5]);
        fs::create_directories(target);
        if(!sdu_apex::qp_fixture::write_failure_context(
                (target/"source_context.mctx").string(),context))
            throw std::runtime_error("failed to write live source context");
        std::ofstream out(target/"initial.controls");
        if(!out)throw std::runtime_error("cannot write initial controls");
        out<<std::setprecision(10);
        for(const auto &u:context.r1_nominal)
            out<<u.steering_target_rad<<' '<<u.throttle_command<<'\n';
        std::cout<<"ORIGINAL_SOURCE_PARITY"
                 <<" cost="<<score->objective
                 <<" margin="<<score->minimum_corridor_slack_m
                 <<" violation="<<residual
                 <<" rate_margin="<<score->minimum_command_rate_margin
                 <<" box_margin="<<score->minimum_command_box_margin
                 <<" required_extra_clearance="<<desired_clearance
                 <<"\n";
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
