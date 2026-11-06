// Check that opt-in benchmark capture preserves the live numerical result,
// including final primals on rejected epochs. Development only.
#include "mpc_qp_failure_context.hpp"
#include "mpc_source_exact_offline_solver.hpp"
#include "mpc_solver_memory_transaction.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>

int main(int argc,char **argv) {
    if(argc!=2)return 2;
    constexpr int64_t epoch=1000000000;
    if(mpc_rti_command_epoch_shift(epoch,epoch-595814,.025f)!=0 ||
       mpc_rti_command_epoch_shift(epoch,epoch-25000000,.025f)!=-1 ||
       mpc_rti_command_epoch_shift(epoch,epoch+25000000,.025f)!=1 ||
       mpc_rti_command_epoch_shift(epoch,epoch,.025f)!=0)return 11;
    auto issued=std::make_unique<MpcRtiMemory_t>();
    issued->nominal.valid=1;issued->warm_start_controls_valid=1;issued->command_epoch_ns=epoch;
    const auto saved=*issued;
    f1tenth_mpc::SolverMemoryTransaction transaction;
    transaction.begin(*issued,epoch+25000000,true);
    issued->command_epoch_ns+=25000000;issued->nominal.valid=0;
    if(!transaction.rollback(issued.get(),epoch+25000000) ||
       std::memcmp(issued.get(),&saved,sizeof(saved))!=0)return 12;
    auto context=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    if(!sdu_apex::qp_fixture::read_failure_context(
        (std::filesystem::path(argv[1])/"source_context.mctx").string(),context.get()))return 3;
    MpcRtiCycleConfiguration_t cfg{};
    cfg.model=context->model;cfg.command_epoch_ns=context->command_epoch_ns;
    cfg.source_exact_nlp_live=cfg.source_exact_nlp_constrained_qp=
        cfg.source_exact_nlp_compact_qp=cfg.source_exact_nlp_full_step_sqp=1;
    cfg.source_exact_nlp_iterations=3;
    cfg.source_exact_nlp_qp_soft_weight=1000000.f;
    cfg.source_exact_nlp_qp_regularization=1000.f;
    cfg.source_exact_nlp_extra_clearance_m=.03f;
    auto plain=std::make_unique<MpcRtiCycleResult_t>();
    auto captured=std::make_unique<MpcRtiCycleResult_t>();
    const auto solve=[&](MpcRtiCycleResult_t *out,bool capture) {
        auto memory=std::make_unique<MpcRtiMemory_t>();
        if(!mpc_rti_memory_set_warm_start_controls_at_epoch(memory.get(),
            context->r1_nominal,PREDICTION_HORIZON,context->command_epoch_ns))return false;
        cfg.capture_source_benchmark=capture;
        mpc_rti_solve_cycle(&context->initial_state,context->progress_m,
            context->trajectory.data(),context->trajectory.size(),context->lap_length_m,
            context->prediction_dt,PREDICTION_HORIZON,&cfg,memory.get(),out);
        return true;
    };
    if(!solve(plain.get(),false) || !solve(captured.get(),true))return 4;
    if(plain->status!=captured->status ||
       plain->source_exact_nlp_final_source_cost!=captured->source_exact_nlp_final_source_cost ||
       plain->source_primal_trace_count!=0 || captured->source_primal_trace_count<1 ||
       captured->source_primal_trace_truncated ||
       captured->source_optimized_control_count!=PREDICTION_HORIZON ||
       std::memcmp(plain->source_optimized_controls,captured->source_optimized_controls,
                   sizeof(plain->source_optimized_controls))!=0) {
        std::cerr<<"status="<<plain->status<<','<<captured->status
            <<" input_failure="<<plain->input_failure_reason<<','<<captured->input_failure_reason
            <<" cost="<<plain->source_exact_nlp_final_source_cost<<','<<captured->source_exact_nlp_final_source_cost
            <<" traces="<<plain->source_primal_trace_count<<','<<captured->source_primal_trace_count
            <<" final_controls="<<captured->source_optimized_control_count<<'\n';
        return 5;
    }
    MpcModelControl_t retained[PREDICTION_HORIZON]{};
    std::copy_n(captured->nominal_controls,PREDICTION_HORIZON,retained);
    for(int i=0;i<captured->source_primal_trace_count;++i) {
        const auto &trace=captured->source_primal_trace[i];
        if(std::memcmp(retained,trace.warm_start,sizeof(retained))!=0)return 6;
        if(trace.accepted) {
            if(!trace.trial_evaluated && !trace.normal_accepted)return 7;
            if(trace.correction_accepted && !trace.correction_evaluated)return 7;
            std::copy_n(trace.accepted_controls,PREDICTION_HORIZON,retained);
        }
    }
    if(std::memcmp(retained,captured->source_optimized_controls,sizeof(retained))!=0)return 8;
    sdu_apex::source_nlp::OfflineOptions late;
    late.absolute_deadline_monotonic_us=1.;
    auto expired=std::make_unique<sdu_apex::source_nlp::OfflineResult>();
    cfg.objective_schedule=&context->objective;
    if(!sdu_apex::source_nlp::optimize_offline_exact_source(context->initial_state,
        context->progress_m,context->trajectory.data(),context->trajectory.size(),
        context->lap_length_m,context->prediction_dt,cfg,context->r1_nominal,
        late,expired.get()))return 9;
    if(!expired->deadline_exhausted || expired->source_nlp_kkt_converged ||
       expired->solver_status!=MPC_SOURCE_DEADLINE_EXCEEDED ||
       !expired->history.empty() || expired->final_source_feasible!=expired->initial_source_feasible ||
       std::memcmp(expired->optimized.controls,context->r1_nominal,
                   sizeof(context->r1_nominal))!=0)return 10;
    std::cout<<"{\"capture_preserves_numerical_result\":true,\"trace_complete\":true,"
        "\"final_primal_retained_independently_of_publication\":true}\n";
    return 0;
}
