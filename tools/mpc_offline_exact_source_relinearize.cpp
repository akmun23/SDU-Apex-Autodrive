/*
 * Offline exact-source relinearization oracle for a single frozen command epoch.
 *
 * Given a FULL steering/throttle seed, rebuild the 30-stage source trajectory
 * including wheel/suspension and delayed actuator state, then rebuild an R1
 * local QP using the ORIGINAL physical corridor and ONE frozen objective.
 *
 * This tool ONLY exports a QP and diagnostic merit, never publishes controls
 * or changes live racing configuration. A Python outer SQP can use it to
 * assess whether repeated exact-source relinearizations reduce actual V.
 */
#include "mpc_rti.h"
#include "mpc_rti_internal.hpp"
#include "mpc_qp_fixture_format.hpp"
#include "mpc_qp_failure_context.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char **argv) {
    const bool serve=argc>=3 && std::string(argv[2])=="--serve";
    if((!serve && (argc<4 || argc>6)) ||
       (serve && (argc<3 || argc>4))) {
        std::cerr<<"usage: mpc_offline_exact_source_relinearize CASE seed_controls.txt|- output.fqp [mode26|captured]\n"
                 <<"   or: mpc_offline_exact_source_relinearize CASE --serve [mode26|captured]\n";
        return 2;
    }
    const std::string dir=argv[1];
    sdu_apex::qp_fixture::FailureReplayContext context{};
    if(!sdu_apex::qp_fixture::read_failure_context(
          dir+"/source_context.mctx",&context)) {
        std::cerr<<"source context invalid\n";return 3;
    }
    auto original=std::make_unique<MpcRtiQpCapture_t>();
    if(!sdu_apex::qp_fixture::read_capture(dir+"/r1.fqp",original.get())) {
        std::cerr<<"R1 original QP missing\n";return 4;
    }
    // Frozen context and original solver capture are loaded ONCE.
    // Server requests contain: OUTPUT_FIXTURE_PATH then 30 pairs of
    // steering/throttle. There is no nominal-state or optimizer-memory
    // carryover between requests: every QP is rebuilt from the same
    // exact physical command epoch.
    do {
    const std::string output_path=serve ?
        ([&](){std::string name;std::cin>>name;return name;}()) :
        std::string(argv[3]);
    if(output_path.empty() || !std::cin.good() && serve) {
        if(serve && std::cin.eof())return 0;
        return 5;
    }
    MpcModelControl_t command[PREDICTION_HORIZON]{};
    if(!serve && std::string(argv[2])=="-") {
        std::copy_n(context.r1_nominal,PREDICTION_HORIZON,command);
    } else {
        std::ifstream file;
        if(!serve)file.open(argv[2]);
        std::istream &input=serve ? static_cast<std::istream&>(std::cin)
                                 : static_cast<std::istream&>(file);
        if(!input.good())return 5;
        for(auto &control:command)
            if(!(input>>control.steering_target_rad>>control.throttle_command))
                return 5;
    }
    if(!serve) {
        std::ofstream out(output_path+".seed.txt");
        out<<std::setprecision(12);
        for(const auto &c:command)
            out<<c.steering_target_rad<<" "<<c.throttle_command<<"\n";
    }
    MpcRtiConfiguration_t model=context.model;
    const std::string mode=serve
        ?(argc==4?std::string(argv[3]):"mode26")
        :(argc>=5?std::string(argv[4]):"captured");
    if(mode=="mode26") {
        model.command_increment_qp_enabled=1;
        model.source_reachable_steering_qp=1;
    } else if(mode!="captured")return 2;
    model.world_sweep_check=nullptr;
    model.world_sweep_context=nullptr;
    model.world_map_rejection_only=0;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    if(!mpc_rti_build_nominal_same_epoch(
        &context.initial_state,context.progress_m,nullptr,command,
        PREDICTION_HORIZON,context.trajectory.data(),
        context.trajectory.size(),context.lap_length_m,
        context.prediction_dt,PREDICTION_HORIZON,&model,
        nominal.get(),references)) {
        std::cerr<<"exact-source nominal failed\n";return 6;
    }
    auto fresh=std::make_unique<MpcRtiQpCapture_t>();
    if(!mpc_rti_build_ltv_qp_with_progress_corridor(
       nominal->states,nominal->controls,references,PREDICTION_HORIZON,
       context.prediction_dt,&model,&context.objective,&fresh->problem,
       nominal->progress,context.trajectory.data(),
       context.trajectory.size(),context.lap_length_m,nullptr)) {
        std::cerr<<"source QP build failed stage="<<fresh->problem.linearization_failure_stage
             <<" reason="<<fresh->problem.linearization_failure_reason<<"\n";
        return 7;
    }
    fresh->solver_config=original->solver_config;
    for(int k=0;k<=RICCATI_HORIZON;++k)
        fresh->solver_config.state_polygons[k]=fresh->problem.state_polygons[k];
    fresh->solver_config.max_solve_time_us=0.;
    fresh->solver_config.cold_start_each_solve=1;
    fresh->valid=fresh->capture_count=1;
    if(!sdu_apex::qp_fixture::write_capture(output_path,*fresh)) {
        std::cerr<<"QP fixture write failed\n";return 8;
    }
    double source_violation=NAN,min_clearance=NAN;
    if(!mpc_rti_exact_source_corridor_violation(
        &context.initial_state,context.progress_m,command,
        context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,context.prediction_dt,&model,
        &source_violation,&min_clearance))return 9;
    MpcRtiCycleConfiguration_t diagnostic{};
    diagnostic.model=model;
    diagnostic.objective_schedule=&context.objective;
    diagnostic.terminal_viability=0;
    double physical_objective=NAN;
    float physical_min_slack=NAN;
    const int has_cost=mpc_rti_score_candidate_plan_for_diagnostics(
        nominal.get(),context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,&diagnostic,&physical_objective,&physical_min_slack);
    auto strict=std::make_unique<MpcRtiNominal_t>();
    int failure_stage=-1;
    const auto strict_status=mpc_rti_validate_candidate_plan(
        &context.initial_state,context.progress_m,command,
        context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,PREDICTION_HORIZON,context.prediction_dt,
        &diagnostic,strict.get(),&failure_stage);
    std::cout<<std::setprecision(13)
      <<"{\"exact_source_violation_m2\":"<<source_violation
      <<",\"minimum_corridor_slack_m\":"<<min_clearance
      <<",\"physical_objective\":";
    if(has_cost && std::isfinite(physical_objective))std::cout<<physical_objective;
    else std::cout<<"null";
    if(!serve && argc==6) {
        MpcModelControl_t proposal[PREDICTION_HORIZON]{};
        std::ifstream file(argv[5]);
        if(!file.good())return 10;
        for(auto &c:proposal)
            if(!(file>>c.steering_target_rad>>c.throttle_command))return 10;
        auto trial_nominal=std::make_unique<MpcRtiNominal_t>();
        MpcRtiReference_t trial_references[PREDICTION_HORIZON+1]{};
        double trial_v=NAN,trial_min_slack=NAN,trial_cost=NAN;
        float trial_cost_clearance=NAN;
        bool trial_scored=false;
        if(mpc_rti_build_nominal_same_epoch(
            &context.initial_state,context.progress_m,nullptr,proposal,
            PREDICTION_HORIZON,context.trajectory.data(),
            context.trajectory.size(),context.lap_length_m,
            context.prediction_dt,PREDICTION_HORIZON,
            &model,trial_nominal.get(),trial_references) &&
           mpc_rti_exact_source_corridor_violation(
               &context.initial_state,context.progress_m,proposal,
               context.trajectory.data(),context.trajectory.size(),
               context.lap_length_m,context.prediction_dt,&model,
               &trial_v,&trial_min_slack)) {
            trial_scored=mpc_rti_score_candidate_plan_for_diagnostics(
                trial_nominal.get(),context.trajectory.data(),
                context.trajectory.size(),context.lap_length_m,
                &diagnostic,&trial_cost,&trial_cost_clearance);
        }
        std::cout<<",\"proposal_actual_source_violation_m2\":";
        if(std::isfinite(trial_v))std::cout<<trial_v;
        else std::cout<<"null";
        std::cout<<",\"proposal_actual_source_objective\":";
        if(trial_scored && std::isfinite(trial_cost))std::cout<<trial_cost;
        else std::cout<<"null";
        std::cout<<",\"proposal_actual_source_objective_delta\":";
        if(trial_scored && has_cost && std::isfinite(trial_cost) &&
           std::isfinite(physical_objective))
            std::cout<<(trial_cost-physical_objective);
        else std::cout<<"null";
        MpcRtiSqpStepMetrics_t metrics{};
        const bool ok=mpc_rti_measure_qp_source_agreement(
            &context.initial_state,context.progress_m,command,proposal,
            &fresh->problem,context.trajectory.data(),
            context.trajectory.size(),context.lap_length_m,
            context.prediction_dt,&model,&metrics);
        std::cout<<",\"proposal_source_agreement_computed\":"<<(ok?"true":"false")
                 <<",\"maximum_source_state_prediction_error\":";
        if(ok)std::cout<<metrics.maximum_state_error;
        else std::cout<<"null";
        std::cout<<",\"maximum_source_footprint_prediction_error_m\":";
        if(ok)std::cout<<metrics.maximum_footprint_constraint_error_m;
        else std::cout<<"null";
    }
    std::cout<<",\"strict_source_valid\":"
      <<(strict_status==MPC_RTI_ROLLOUT_OK?"true":"false")
      <<",\"strict_source_status\":"<<static_cast<int>(strict_status)
      <<",\"strict_source_failure_stage\":"<<failure_stage
      <<",\"nominal_progress_end\":"<<nominal->progress[PREDICTION_HORIZON]
      <<",\"qp_exported\":true}\n";
    if(!serve)break;
    std::cout<<std::flush;
    } while(true);
    return 0;
}
