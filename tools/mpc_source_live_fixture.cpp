#include "mpc_qp_failure_context.hpp"
#include "mpc_trajectory_io.hpp"
#include "mpc_source_exact_nlp.hpp"
#include "mpc_speed_envelope.hpp"
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
    if(argc!=5 && argc!=6) {
        std::cerr<<"usage: mpc_source_live_fixture LIVE_DIAG_JSON PARAMS_JSON RACELINE_CSV [LEGACY_BASE_CONTEXT_DIR] OUT_DIR\n";
        return 2;
    }
    try {
        json live,params;
        { std::ifstream input(argv[1]);input>>live; }
        { std::ifstream input(argv[2]);input>>params; }
        sdu_apex::qp_fixture::FailureReplayContext context{};
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
        // Never inherit hidden state from an unrelated historical fixture.
        // The extractor may recover omitted flags from the exact-stamp bag.
        if(live.contains("plant_state_flags")) {
            const auto &f=live.at("plant_state_flags");
            if(f.size()!=3)throw std::runtime_error("invalid plant flags");
            s.source_plant_initialized=f[0].get<int>();
            s.source_suspension_history_initialized=f[1].get<int>();
            s.source_imu_state_initialized=f[2].get<int>();
        } else throw std::runtime_error("missing plant_state_flags; recover from exact source epoch, not an unrelated fixture");
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
        context.trajectory=f1tenth_mpc::read_mpc_controller_trajectory(csv,&context.lap_length_m);
        const double ay=params.value("curvature_speed_envelope_lateral_accel_mps2",0.);
        if(ay>0.)f1tenth_mpc::apply_curvature_speed_envelope(
            context.trajectory,context.lap_length_m,ay,
            params.value("curvature_speed_envelope_braking_mps2",5.));
        // Exact live mode only consumes raceline geometry/speed/corridor,
        // never the IPOPT witness controls or states.
        for(auto &point:context.trajectory)point.source_witness_valid=0;
        auto &m=context.model;
        // Canonical ROS defaults for omitted parameters, followed by the
        // recorded profile. Physical source constants are compiled into
        // the source plant, not supplied by another run's model POD.
        m.weight_e_y=4000.f;m.weight_e_psi=1500.f;m.weight_u=35.f;
        m.weight_u_overspeed=20.f;m.overspeed_curvature_gain_m=.5f;
        m.weight_longitudinal_slip=5.f;m.weight_v=5.f;m.weight_r=7.8125f;
        m.weight_steering_command=3.f;m.weight_steering_target=5.f;
        m.weight_throttle_command=.1f;m.weight_steering_slew=2000.f;
        m.weight_throttle_slew=50.f;m.startup_nominal_throttle_seed=.25f;
        m.terminal_multiplier=3.f;m.max_speed_mps=16.f;
        m.max_steering_rad=static_cast<float>(UNITY_SOURCE_MAX_STEERING_RAD);
        m.corridor_margin_m=.05f;m.first_prediction_corridor_margin_m=.05f;
        m.corridor_preview_halfwidth_m=.1f;m.nonlinear_corridor_tolerance_m=.005f;
        m.steering_correction_variation_epsilon_rad=.02f;m.jacobian_threads=8;
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
        OVERRIDE(near_wall_tracking_reserve_m);OVERRIDE(jacobian_threads);
#undef OVERRIDE
        // These full input-range limits are required by the source NLP
        // even though the old incremental QP has been deleted. Do NOT
        // rebuild a fixture with the old zero-initialized rates.
        // Match ROS node's canonical missing-parameter defaults, not
        // unrelated historical competition replay values.
        // The live practice profile omits all of these intentionally.
        m.command_increment_qp_enabled=0;
        m.source_reachable_steering_qp=0;
        m.max_steering_command_step_rad=2.f*m.max_steering_rad;
        m.max_throttle_command_step=
            MPC_THROTTLE_MAX-MPC_THROTTLE_MIN_DRIVE;
        m.active_speed_ceiling_mps=
            live.contains("active_speed_ceiling_mps")?
            live.at("active_speed_ceiling_mps").get<float>():
            params.at("max_speed_mps").get<float>();
        const bool captured_problem=live.contains("frozen_source_problem");
        const auto restore_hex=[](const std::string &hex,void *out,size_t count) {
            if(hex.size()!=2*count)throw std::runtime_error("captured source POD ABI size mismatch");
            const auto nibble=[](char c)->int {
                if(c>='0'&&c<='9')return c-'0';
                if(c>='a'&&c<='f')return c-'a'+10;
                throw std::runtime_error("invalid source POD encoding");
            };
            auto *bytes=static_cast<unsigned char *>(out);
            for(size_t i=0;i<count;++i)bytes[i]=static_cast<unsigned char>(
                16*nibble(hex[2*i])+nibble(hex[2*i+1]));
        };
        if(captured_problem) {
            const auto &f=live.at("frozen_source_problem");
            if(f.at("schema")!="source-pod/v1" ||
               f.at("world_sweep_was_configured").get<bool>() ||
               f.at("model_size").get<size_t>()!=sizeof(m) ||
               f.at("objective_size").get<size_t>()!=sizeof(context.objective))
                throw std::runtime_error("unsupported source POD capture or world-map callback");
            restore_hex(f.at("model_pod_hex").get<std::string>(),&m,sizeof(m));
            m.world_sweep_check=nullptr;m.world_sweep_context=nullptr;
        }
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
        if(captured_problem) {
            restore_hex(live.at("frozen_source_problem").at("objective_pod_hex").get<std::string>(),
                        &context.objective,sizeof(context.objective));
            if(!context.objective.valid || context.objective.horizon!=PREDICTION_HORIZON)
                throw std::runtime_error("invalid captured frozen objective");
        }
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
        // Historic streams only record INITIAL cost, FINAL margin and FINAL
        // violation. Comparing all three to the seed mixed two iterates.
        if(std::abs(score->objective-expected_cost)>
                1.e-7*std::max(1.,std::abs(expected_cost)))
            throw std::runtime_error(
                "live INITIAL source cost parity failed: cost_delta="+
                std::to_string(score->objective-expected_cost));
        auto target=fs::path(argv[argc-1]);
        fs::create_directories(target);
        if(!sdu_apex::qp_fixture::write_failure_context(
                (target/"source_context.mctx").string(),context))
            throw std::runtime_error("failed to write live source context");
        std::ofstream out(target/"initial.controls");
        if(!out)throw std::runtime_error("cannot write initial controls");
        out<<std::setprecision(10);
        for(const auto &u:context.r1_nominal)
            out<<u.steering_target_rad<<' '<<u.throttle_command<<'\n';
        json frozen={{"schema","source-epoch-frozen-problem/v1"},
            {"progress_m",context.progress_m},{"command_epoch_ns",context.command_epoch_ns},
            {"prediction_dt",context.prediction_dt},{"lap_length_m",context.lap_length_m},
            {"initial_cost",score->objective},{"initial_clearance_m",score->minimum_corridor_slack_m},
            {"initial_elastic_violation",residual},
            {"initial_rate_margin",score->minimum_command_rate_margin},
            {"initial_box_margin",score->minimum_command_box_margin},
            {"initial_cost_delta",score->objective-expected_cost},
            {"terminal_speed_viability_active",false},
            {"references",json::array()},{"terminal_targets",json::array()},
            {"model",json::object()}};
        const auto ref=[](const MpcRtiReference_t &r) {return json{
            {"e_y",r.e_y},{"e_psi",r.e_psi},{"u",r.u},{"v",r.v},{"r",r.r},
            {"steering_command",r.steering_command},{"curvature",r.path_curvature},
            {"left_bound",r.left_bound},{"right_bound",r.right_bound},
            {"source_witness_valid",r.source_witness_valid}};};
        for(int k=0;k<=PREDICTION_HORIZON;++k) {
            auto r=ref(context.objective.references[k]);
            r["speed_weight"]=context.objective.speed_weights[k];
            r["nominal_progress_m"]=nominal->progress[k];
            frozen["references"].push_back(r);
        }
        for(int k=0;k<3;++k) {
            auto r=ref(context.objective.terminal_targets[k]);
            r["speed_weight"]=context.objective.terminal_speed_weights[k];
            frozen["terminal_targets"].push_back(r);
        }
        frozen["terminal_targets_valid"]=context.objective.terminal_targets_valid;
        frozen["terminal_prediction_dt"]=context.objective.terminal_prediction_dt;
        const auto hex=[](const void *object,size_t size) {
            constexpr char digits[]="0123456789abcdef";
            const auto *bytes=static_cast<const unsigned char *>(object);
            std::string out;out.reserve(2*size);
            for(size_t i=0;i<size;++i){out+=digits[bytes[i]>>4];out+=digits[bytes[i]&15];}
            return out;
        };
        frozen["frozen_source_problem"]={{"schema","source-pod/v1"},
            {"world_sweep_was_configured",false},
            {"model_size",sizeof(context.model)},{"objective_size",sizeof(context.objective)},
            {"model_pod_hex",hex(&context.model,sizeof(context.model))},
            {"objective_pod_hex",hex(&context.objective,sizeof(context.objective))}};
        frozen["frozen_problem_origin"]=captured_problem?
            "captured_frozen_pod":"reconstructed_from_recorded_seed_cost_verified";
        // Only published cycles contain the final primal in old recordings.
        // Re-score that captured primal independently of today's SQP steps.
        frozen["recorded_final_primal_available"]=false;
        const char *final_key=live.contains("source_optimized_controls") &&
            live.at("source_optimized_controls").size()==PREDICTION_HORIZON?
            "source_optimized_controls":"candidate_controls";
        if(live.contains(final_key) &&
           live.at(final_key).size()==PREDICTION_HORIZON) {
            auto final=std::make_unique<MpcRtiNominal_t>();
            final->horizon=PREDICTION_HORIZON;
            final->states[0]=context.initial_state;
            final->progress[0]=context.progress_m;
            for(int k=0;k<PREDICTION_HORIZON;++k) {
                const auto &u=live.at(final_key)[k];
                final->controls[k]={u[0].get<float>(),u[1].get<float>()};
                if(!mpc_rti_source_exact_stage(&final->states[k],final->progress[k],
                    &final->controls[k],context.trajectory.data(),context.trajectory.size(),
                    context.lap_length_m,context.prediction_dt,&final->states[k+1],
                    &final->progress[k+1]))throw std::runtime_error("final primal rollout failed");
            }
            final->valid=1;
            if(!sdu_apex::source_nlp::decision_from_nominal(*final,primal.get()) ||
               !sdu_apex::source_nlp::evaluate_existing_source_nominal(
                   context.initial_state,context.progress_m,context.trajectory.data(),
                   context.trajectory.size(),context.lap_length_m,context.prediction_dt,
                   validation,*final,*primal,score.get()))
                throw std::runtime_error("final primal scoring failed");
            double violation=0.;
            for(const auto &c:score->corridor_corners)
                for(int i=0;i<4;++i)
                    for(double margin:{c.left[i],c.right[i]}) {
                        const double v=std::min(0.,margin-desired_clearance);
                        violation+=.5*v*v;
                    }
            for(const auto &rate:score->command_rate)
                for(double margin:rate){const double v=std::min(0.,margin);violation+=.5*v*v;}
            frozen["recorded_final_primal_available"]=true;
            frozen["recorded_final_score"]={{"cost",score->objective},
                {"minimum_clearance_m",score->minimum_corridor_slack_m},
                {"elastic_violation",violation},
                {"rate_margin",score->minimum_command_rate_margin},
                {"box_margin",score->minimum_command_box_margin}};
        }
#define SAVE(name) frozen["model"][#name]=m.name
        SAVE(weight_e_y);SAVE(weight_e_psi);SAVE(weight_u);SAVE(weight_u_overspeed);
        SAVE(weight_longitudinal_slip);SAVE(weight_v);SAVE(weight_r);
        SAVE(weight_steering_command);SAVE(weight_steering_target);
        SAVE(weight_throttle_command);SAVE(weight_steering_slew);SAVE(weight_throttle_slew);
        SAVE(terminal_multiplier);SAVE(max_speed_mps);SAVE(active_speed_ceiling_mps);
        SAVE(max_steering_rad);SAVE(max_steering_command_step_rad);SAVE(max_throttle_command_step);
        SAVE(corridor_margin_m);SAVE(first_prediction_corridor_margin_m);
        SAVE(corridor_preview_halfwidth_m);SAVE(nonlinear_corridor_tolerance_m);
        SAVE(jacobian_threads);
        SAVE(tracking_lateral_tolerance_m);SAVE(tracking_heading_tolerance_rad);
        SAVE(overspeed_curvature_gain_m);SAVE(underspeed_curvature_gain_m);
        SAVE(longitudinal_slip_curvature_gain_m);SAVE(weight_steering_correction_variation);
        SAVE(steering_correction_variation_epsilon_rad);SAVE(command_increment_qp_enabled);
        SAVE(startup_nominal_throttle_seed);SAVE(reference_clearance_reserve_m);
        SAVE(near_wall_tracking_reserve_m);SAVE(world_map_rejection_only);
        SAVE(weight_source_corridor_reserve);SAVE(weight_source_lateral_slip);
        SAVE(source_reachable_steering_qp);
#undef SAVE
        const auto physical=vehicle_model_get_parameters();
        frozen["physical_command_configuration"]={
            {"steering_wheelbase_m",physical.steering_wheelbase_m},
            {"max_steering_angle",physical.max_steering_angle},
            {"steering_rate_radps",physical.steering_rate_radps},
            {"maximum_command_speed_mps",physical.maximum_command_speed_mps},
            {"source_throttle_min",MPC_THROTTLE_MIN},
            {"nlp_throttle_min",MPC_THROTTLE_MIN_DRIVE},
            {"throttle_max",MPC_THROTTLE_MAX}};
        std::ofstream(target/"frozen_problem.json")<<frozen.dump(2)<<'\n';
        std::cout<<"INITIAL_SOURCE_COST_PARITY"
                 <<" cost="<<frozen.at("initial_cost")
                 <<" margin="<<frozen.at("initial_clearance_m")
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
