#include "mpc_rti.h"
#include "mpc_qp_fixture_format.hpp"
#include "mpc_trajectory_io.hpp"
#include "mpc_world_footprint.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <vector>

namespace {

MpcRtiQpCapture_t g_qp_capture{};
struct FrozenWorldMap {
    std::shared_ptr<const f1tenth_mpc::MpcWorldFootprint> grid;
    long queried_steps=0;
};
int frozen_world_sweep(void *context,
                       double x0,double y0,double yaw0,
                       double x1,double y1,double yaw1,
                       double margin_m,float *slack)
{
    if(!context || !slack)return 0;
    auto *world=static_cast<FrozenWorldMap*>(context);
    if(!world->grid)return 0;
    ++world->queried_steps;
    double clearance=-INFINITY;
    const bool safe=world->grid->interpolated_sweep_passes(
        {x0,y0,yaw0},{x1,y1,yaw1},margin_m+.005,&clearance);
    *slack=static_cast<float>(clearance);
    return safe?1:0;
}


constexpr double kMaxSteeringRad = 0.5235987755982988;
constexpr std::size_t kInputColumnCount = 38;

std::vector<double> parse_csv_numbers(const std::string &line)
{
    std::vector<double> values;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        if (!field.empty()) values.push_back(std::stod(field));
    }
    return values;
}

std::vector<MpcTrajectorySample_t> load_trajectory(const char *path,
                                                   double *lap_length)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open trajectory CSV");
    return f1tenth_mpc::read_mpc_trajectory(input, lap_length);
}

double number_arg(char **argv, int index, double fallback)
{
    return argv[index] ? std::stod(argv[index]) : fallback;
}

std::map<std::size_t, std::array<MpcModelControl_t, PREDICTION_HORIZON>>
load_nominal_seed_controls(const char *path)
{
    std::map<std::size_t,
        std::array<MpcModelControl_t, PREDICTION_HORIZON>> seeds;
    if (!path || path[0] == '\0') return seeds;
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error(
            "cannot open MPC_REPLAY_NOMINAL_SEED_CSV: " + std::string(path));
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line.front() == '#') continue;
        const auto values = parse_csv_numbers(line);
        if (values.size() != 1 + 2 * PREDICTION_HORIZON)
            throw std::runtime_error(
                "nominal seed row must contain index plus 30 steering/throttle "
                "pairs at line " + std::to_string(line_number));
        if (!std::isfinite(values[0]) || values[0] < 0.0 ||
            std::floor(values[0]) != values[0])
            throw std::runtime_error(
                "nominal seed row index must be a non-negative integer");
        const std::size_t row_index =
            static_cast<std::size_t>(values[0]);
        std::array<MpcModelControl_t, PREDICTION_HORIZON> controls{};
        for (int stage = 0; stage < PREDICTION_HORIZON; ++stage) {
            const double steering = values[1 + 2 * stage];
            const double throttle = values[2 + 2 * stage];
            if (!std::isfinite(steering) ||
                std::abs(steering) > kMaxSteeringRad + 1.0e-7 ||
                !std::isfinite(throttle) ||
                throttle < MPC_THROTTLE_MIN_DRIVE - 1.0e-7 ||
                throttle > 1.0 + 1.0e-7)
                throw std::runtime_error(
                    "nominal seed control violates the MPC drive envelope at "
                    "line " + std::to_string(line_number));
            controls[stage].steering_target_rad = static_cast<float>(steering);
            controls[stage].throttle_command = static_cast<float>(throttle);
        }
        if (!seeds.emplace(row_index, controls).second)
            throw std::runtime_error("duplicate nominal seed row index");
    }
    return seeds;
}

void configure(MpcRtiCycleConfiguration_t *config,
               double rho, double rho_steer, double rho_throttle,
               double alpha, double cost_scale,
               double rho_ey, double rho_u_state, double rho_steer_state,
               double rho_max, int rho_period, double tolerance,
               bool cold_start_each_solve, int jacobian_threads,
               int max_iterations,
               double startup_throttle_seed,
               double lateral_error_weight_multiplier,
               double heading_error_weight_multiplier,
               double lateral_state_scale,
               double heading_state_scale,
               double corridor_margin,
               double first_corridor_margin,
               double rho_u,
               double steering_slew_weight,
               double throttle_slew_weight,
               bool adaptive_rho,
               double speed_tracking_weight_multiplier)
{
    *config = MpcRtiCycleConfiguration_t{};
    auto &model = config->model;
    model.weight_e_y = static_cast<float>(
        125.0 * cost_scale * lateral_error_weight_multiplier);
    model.weight_e_psi = static_cast<float>(
        31.25 * cost_scale * heading_error_weight_multiplier);
    model.weight_u = static_cast<float>(500.0 * cost_scale);
    model.weight_u_overspeed = static_cast<float>(300.0 * cost_scale);
    model.weight_u *= static_cast<float>(speed_tracking_weight_multiplier);
    model.weight_u_overspeed *=
        static_cast<float>(speed_tracking_weight_multiplier);
    model.weight_longitudinal_slip = static_cast<float>(5.0 * cost_scale);
    model.weight_v = static_cast<float>(5.0 * cost_scale);
    model.weight_r = static_cast<float>(7.8125 * cost_scale);
    model.weight_steering_command = static_cast<float>(3.0 * cost_scale);
    model.weight_steering_target = static_cast<float>(5.0 * cost_scale);
    model.weight_throttle_command = static_cast<float>(0.1 * cost_scale);
    model.weight_steering_slew = static_cast<float>(steering_slew_weight);
    model.weight_throttle_slew = static_cast<float>(throttle_slew_weight);
    model.startup_nominal_throttle_seed = static_cast<float>(
        startup_throttle_seed);
    model.terminal_multiplier = 3.0f;
    model.max_speed_mps = 16.0f;
    model.active_speed_ceiling_mps = 16.0f;
    model.max_steering_rad = static_cast<float>(kMaxSteeringRad);
    model.corridor_margin_m = static_cast<float>(corridor_margin);
    model.first_prediction_corridor_margin_m =
        static_cast<float>(first_corridor_margin);
    model.corridor_preview_halfwidth_m = 0.10f;
    model.nonlinear_corridor_tolerance_m = 0.005f;
    model.jacobian_threads = jacobian_threads;

    auto &solver = config->solver;
    solver.rho = static_cast<float>(rho);
    solver.rho_u = static_cast<float>(rho_u);
    solver.rho_x_by_state[MPC_OPT_IDX_EY] = static_cast<float>(rho_ey);
    solver.rho_x_by_state[MPC_OPT_IDX_U] = static_cast<float>(rho_u_state);
    solver.rho_x_by_state[MPC_OPT_IDX_ACTUAL_STEERING_ANGLE] =
        static_cast<float>(rho_steer_state);
    solver.rho_u_by_input[0] = static_cast<float>(rho_steer);
    solver.rho_u_by_input[1] = static_cast<float>(rho_throttle);
    solver.rho_max = static_cast<float>(rho_max);
    solver.tolerance = static_cast<float>(tolerance);
    solver.max_iterations = max_iterations;
    solver.adaptive_rho = adaptive_rho ? 1 : 0;
    solver.cold_start_each_solve = cold_start_each_solve ? 1 : 0;
    solver.adaptive_rho_update_period = rho_period;
    solver.shared_rho = 0;
    solver.over_relaxation = static_cast<float>(alpha);
    solver.use_scaling = 1;
    const std::array<float, RICCATI_MAX_NX> state_scale = {{
        0.10f, 0.10f, 0.5f, 1.0f, 4.0f, 0.5f,
        400.0f, 400.0f, 400.0f, 400.0f,
        0.10f, 1.0f, 0.25f, 4.0f, 0.25f, 4.0f,
        0.05f, 0.05f, 0.05f, 0.05f,
        1.0f, 0.5f, 1.0f}};
    static_assert(state_scale.size() == RICCATI_MAX_NX,
        "replay scaling must cover every optimizer state");
    std::copy(state_scale.begin(), state_scale.end(), solver.state_scale);
    solver.state_scale[MPC_OPT_IDX_EY] = static_cast<float>(lateral_state_scale);
    solver.state_scale[MPC_OPT_IDX_EPSI] = static_cast<float>(heading_state_scale);
    solver.input_scale[0] = 0.5f;
    solver.input_scale[1] = 1.0f;

    config->max_cycle_time_us = 0.0;
}

void audit_objective_gradient(const MpcRtiState_t &initial, double progress,
    const std::vector<MpcTrajectorySample_t> &trajectory,double length,
    const MpcRtiCycleConfiguration_t &configuration,const MpcRtiMemory_t &memory)
{
    MpcRtiNominal_t nominal{};
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    const bool has_warm_controls = memory.warm_start_controls_valid &&
        memory.warm_start_control_count > 0;
    const bool built = has_warm_controls
        ? mpc_rti_build_nominal_with_warm_start(&initial,progress,&memory.nominal,
            memory.warm_start_controls,memory.warm_start_control_count,
            trajectory.data(),trajectory.size(),length,PREDICTION_DT_SECONDS,
            PREDICTION_HORIZON,&configuration.model,&nominal,references)
        : mpc_rti_build_nominal(&initial,progress,&memory.nominal,
            trajectory.data(),trajectory.size(),length,PREDICTION_DT_SECONDS,
            PREDICTION_HORIZON,&configuration.model,&nominal,references);
    if (!built)
        throw std::runtime_error("gradient audit could not build nominal; reason="+
            std::to_string(nominal.build_failure_constraint)+
            ", stage="+std::to_string(nominal.build_failure_stage));
    MpcRtiProblem_t problem{};
    if (!mpc_rti_build_ltv_qp(nominal.states,nominal.controls,references,
        PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&configuration.model,&problem))
        throw std::runtime_error("gradient audit could not build QP");
    MpcRtiObjectiveSchedule_t schedule{};
    if (!mpc_rti_build_objective_schedule(&nominal,references,&configuration.model,&schedule))
        throw std::runtime_error("gradient audit could not build objective");
    auto config=configuration;
    config.objective_schedule=&schedule;
    auto wide=trajectory;
    // Offline derivative diagnostic only: remove wall rejection while retaining
    // the original geometry, dynamics and frozen objective. Never publishes.
    for (auto &p:wide) p.left_bound=p.right_bound=100.;
    for (int input:{0,1}) for (int changed:{0,8,20}) {
        double x[RICCATI_MAX_NX],dx[RICCATI_MAX_NX]{};
        std::copy_n(problem.x0,RICCATI_MAX_NX,x);
        double derivative=0.;
        for (int k=0;k<RICCATI_HORIZON;++k) {
            const auto &stage=problem.steps[k];
            const auto prev_steering = k > 0
                ? nominal.controls[k-1].steering_target_rad
                : problem.x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
            const auto prev_throttle = k > 0
                ? nominal.controls[k-1].throttle_command
                : problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
            // In the increment QP, changing ONE absolute command changes
            // the delta at this stage AND at the following stage by -du.
            const double u[2]={
                config.model.command_increment_qp_enabled
                    ? nominal.controls[k].steering_target_rad-prev_steering
                    : nominal.controls[k].steering_target_rad,
                config.model.command_increment_qp_enabled
                    ? nominal.controls[k].throttle_command-prev_throttle
                    : nominal.controls[k].throttle_command};
            double du[2]{};
            if(k==changed) du[input]=1.;
            if(config.model.command_increment_qp_enabled && k==changed+1)
                du[input]=-1.;
            for(int i=0;i<RICCATI_MAX_NX;++i) {
                double gx=stage.Q_diag[i]*x[i]+stage.q[i];
                for(int j=0;j<RICCATI_MAX_NX;++j)
                    gx+=stage.Q_offdiag[i][j]*x[j];
                for(int a=0;a<2;++a)gx+=stage.N[i][a]*u[a];
                derivative+=gx*dx[i];
            }
            for(int a=0;a<2;++a) {
                double gu=stage.R_diag[a]*u[a]+stage.r[a];
                gu+=stage.R_cross*u[1-a];
                for(int i=0;i<RICCATI_MAX_NX;++i)gu+=stage.N[i][a]*x[i];
                derivative+=gu*du[a];
            }
            double next[RICCATI_MAX_NX]{},dnext[RICCATI_MAX_NX]{};
            for(int i=0;i<RICCATI_MAX_NX;++i) {
                next[i]=stage.d[i];
                for(int j=0;j<RICCATI_MAX_NX;++j) {
                    next[i]+=stage.A[i][j]*x[j];dnext[i]+=stage.A[i][j]*dx[j];
                }
                for(int a=0;a<2;++a) {
                    next[i]+=stage.B[i][a]*u[a];dnext[i]+=stage.B[i][a]*du[a];
                }
            }
            std::copy_n(next,RICCATI_MAX_NX,x);std::copy_n(dnext,RICCATI_MAX_NX,dx);
        }
        for(int i=0;i<RICCATI_MAX_NX;++i)
            derivative+=(problem.terminal_Q[i]*x[i]+problem.terminal_q[i])*dx[i];
        for(float epsilon:{.0001f,.00003f}) {
            double cost[2]{};bool valid=true;
            for(int side=0;side<2;++side) {
                MpcModelControl_t controls[PREDICTION_HORIZON];
                std::copy_n(nominal.controls,PREDICTION_HORIZON,controls);
                float &v=input==0?controls[changed].steering_target_rad:
                                  controls[changed].throttle_command;
                v+=(side?epsilon:-epsilon);
                MpcRtiNominal_t exact{};int failure=-1;
                if(mpc_rti_validate_candidate_plan(&initial,progress,controls,
                    wide.data(),wide.size(),length,PREDICTION_HORIZON,
                    PREDICTION_DT_SECONDS,&config,&exact,&failure)!=MPC_RTI_ROLLOUT_OK) {
                    valid=false;break;
                }
                cost[side]=exact.validation_objective;
            }
            if(valid) std::cerr<<"GRADIENT progress="<<progress<<" input="<<input
                <<" stage="<<changed<<" epsilon="<<epsilon<<" qp="<<derivative
                <<" exact="<<(cost[1]-cost[0])/(2.*epsilon)
                <<" walls_relaxed_for_derivative_only=1\n";
        }
    }
}

// Development-only diagnostic: evaluate the *same frozen QP quadratic*
// along the physical-command interpolation that exact PhysX merit scores.
// This distinguishes bad finite-step curvature from a wrong first derivative.
double audit_ltv_objective(const MpcRtiProblem_t &problem,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    bool command_increments)
{
    double x[RICCATI_MAX_NX]{};
    for(int i=0;i<RICCATI_MAX_NX;++i)x[i]=problem.x0[i];
    double cost=0.;
    for(int k=0;k<RICCATI_HORIZON;++k) {
        const auto &stage=problem.steps[k];
        const double u[2]={
            controls[k].steering_target_rad -
                (command_increments ? x[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] : 0.),
            controls[k].throttle_command -
                (command_increments ? x[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] : 0.)};
        for(int i=0;i<RICCATI_MAX_NX;++i) {
            cost+=.5*stage.Q_diag[i]*x[i]*x[i]+stage.q[i]*x[i];
            for(int j=0;j<RICCATI_MAX_NX;++j)
                cost+=.5*stage.Q_offdiag[i][j]*x[i]*x[j];
            for(int a=0;a<2;++a)cost+=x[i]*stage.N[i][a]*u[a];
        }
        for(int a=0;a<2;++a)
            cost+=.5*stage.R_diag[a]*u[a]*u[a]+stage.r[a]*u[a];
        cost+=stage.R_cross*u[0]*u[1];
        double next[RICCATI_MAX_NX]{};
        for(int i=0;i<RICCATI_MAX_NX;++i) {
            next[i]=stage.d[i];
            for(int j=0;j<RICCATI_MAX_NX;++j)next[i]+=stage.A[i][j]*x[j];
            for(int a=0;a<2;++a)next[i]+=stage.B[i][a]*u[a];
        }
        std::copy_n(next,RICCATI_MAX_NX,x);
    }
    for(int i=0;i<RICCATI_MAX_NX;++i)
        cost+=.5*problem.terminal_Q[i]*x[i]*x[i]+problem.terminal_q[i]*x[i];
    return cost;
}


// Accuracy-first OFFLINE solver experiment: true same-epoch nonlinear
// single-shooting SQP. Every SQP nominal and every proposed trajectory is
// integrated with the full source-derived plant. No implicit next-cycle
// warm-start shift; no production publication/fallback policy, no additional
// line-search alpha trials. A full SQP direction that increases the exact
// objective is recorded as a failure, not counted as convergence.
// The experiment may cross inferior source-valid iterates, but only the
// best independently source-valid candidate is reported as an optimizer result.


// Exact-source NOMINAL ADJOINT: derivatives are taken along a physically
// integrated nonlinear trajectory. Each A/B is sourced from Unity-source AD
// at the SAME state/command epoch; Riccati's quadratic stage provides the
// local cost gradient. This does NOT use the QP's inaccurate full-step rollout.
// This should agree with finite differences of the 750ms nonlinear merit
// before anyone uses it as a control-optimization gradient.
bool audit_exact_source_adjoint(
    const MpcRtiState_t &initial, double progress,
    const std::vector<MpcTrajectorySample_t> &trajectory, double length,
    const MpcRtiCycleConfiguration_t &config,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcRtiObjectiveSchedule_t &frozen_objective,
    double gradient_out[PREDICTION_HORIZON][2])
{
    return mpc_rti_source_adjoint_gradient(
        &initial,progress,trajectory.data(),trajectory.size(),length,
        PREDICTION_DT_SECONDS,&config.model,controls,&frozen_objective,
        gradient_out)!=0;
}

// OFFLINE ground-truth reference optimizer, not a production MPC policy.
// Finite-difference the FULL nonlinear 750ms source-plant objective with
// respect to 60 physical commands. Use physically evaluated bounded
// improvements to distinguish a bad QP direction from a truly impossible
// local control improvement. No surrogate prediction is used in its cost.
// The discrete trial choices are for generating a reference data point;
// they MUST NOT be reused as a production SQP globalization fallback.
void audit_exact_source_control_oracle(
    const MpcRtiState_t &initial, double progress,
    const std::vector<MpcTrajectorySample_t> &trajectory,
    const std::vector<MpcTrajectorySample_t> &hard_trajectory,
    double length, MpcRtiCycleConfiguration_t config,
    const MpcModelControl_t seed[PREDICTION_HORIZON], int iterations)
{
    if(iterations<=0)return;
    config.objective_schedule=nullptr;
    config.qp_capture=nullptr;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    auto trial=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    if(!mpc_rti_build_nominal_same_epoch(&initial,progress,nullptr,
        seed,PREDICTION_HORIZON,trajectory.data(),trajectory.size(),length,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config.model,
        nominal.get(),refs))return;
    MpcRtiObjectiveSchedule_t original_objective{};
    if(!mpc_rti_build_objective_schedule(nominal.get(),refs,&config.model,
        &original_objective))return;
    config.objective_schedule=&original_objective;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    std::copy_n(seed,PREDICTION_HORIZON,controls);
    const auto evaluate=[&](const MpcModelControl_t input[PREDICTION_HORIZON]) {
        int failure=-1;
        const auto status=mpc_rti_validate_candidate_plan(
            &initial,progress,input,trajectory.data(),trajectory.size(),
            length,PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&config,
            trial.get(),&failure);
        return status==MPC_RTI_ROLLOUT_OK
            ?trial->validation_objective:INFINITY;
    };
    double current=evaluate(controls);
    if(!std::isfinite(current)) {
        std::cerr<<"SOURCE_ORACLE progress="<<progress
            <<" status=initial_invalid\n";return;
    }
    std::cerr<<std::setprecision(9)<<"SOURCE_ORACLE progress="<<progress
        <<" iter=0 exact_cost="<<current<<"\n";
    constexpr int n=2*PREDICTION_HORIZON;
    for(int iter=1;iter<=std::min(iterations,8);++iter) {
        const auto iteration_start=std::chrono::steady_clock::now();
        double gradient[n]{};
        const bool use_source_adjoint=
            std::getenv("MPC_REPLAY_ORACLE_USE_SOURCE_ADJOINT")!=nullptr;
        const char *fd_scale_env=std::getenv("MPC_REPLAY_ORACLE_FD_SCALE");
        const double fd_scale=fd_scale_env?std::atof(fd_scale_env):1.0;
        if(!std::isfinite(fd_scale)||fd_scale<.02||fd_scale>8.)
            throw std::runtime_error("MPC_REPLAY_ORACLE_FD_SCALE outside [.02,8]");
        const double step[2]={0.0001*fd_scale,0.0003*fd_scale};
        if(!use_source_adjoint)for(int i=0;i<n;++i) {
            const int k=i/2,c=i%2;
            float &input=c==0?controls[k].steering_target_rad:
                controls[k].throttle_command;
            const float lo=c==0?-config.model.max_steering_rad:
                MPC_THROTTLE_MIN_DRIVE;
            const float hi=c==0?config.model.max_steering_rad:
                MPC_THROTTLE_MAX;
            const float saved=input;
            const float above=std::min(hi,float(saved+step[c]));
            const float below=std::max(lo,float(saved-step[c]));
            if(above<=below)continue;
            input=above;const double high=evaluate(controls);
            input=below;const double low=evaluate(controls);
            input=saved;
            if(std::isfinite(high)&&std::isfinite(low))
                gradient[i]=(high-low)/double(above-below);
        }
        double source_ad[PREDICTION_HORIZON][2]{};
        const auto gradient_start=std::chrono::steady_clock::now();
        if(audit_exact_source_adjoint(initial,progress,trajectory,length,
                config,controls,original_objective,source_ad)) {
            if(use_source_adjoint) {
                for(int k=0;k<PREDICTION_HORIZON;++k)
                    for(int a=0;a<2;++a)
                        gradient[2*k+a]=source_ad[k][a];
            } else {
            double sum_sq_error=0.,sum_sq_reference=0.;
            double max_abs_error=0.;int sign_agree=0,active=0;
            for(int k=0;k<PREDICTION_HORIZON;++k)
                for(int a=0;a<2;++a) {
                    const double exact=gradient[2*k+a];
                    const double computed=source_ad[k][a];
                    const double delta=computed-exact;
                    sum_sq_error+=delta*delta;
                    sum_sq_reference+=exact*exact;
                    max_abs_error=std::max(max_abs_error,std::abs(delta));
                    if(std::abs(exact)>1.) {
                        ++active;
                        sign_agree+=((exact>0.)==(computed>0.));
                    }
                }
            std::cerr<<std::setprecision(9)
                <<"SOURCE_ADJOINT_PARITY progress="<<progress
                <<" iter="<<iter
                <<" relative_gradient_error="<<
                    std::sqrt(sum_sq_error/std::max(1e-9,sum_sq_reference))
                <<" max_abs_gradient_error="<<max_abs_error
                <<" sign_agreement="<<sign_agree
                <<" active_components="<<active<<'\n';
            }
        } else {
            std::cerr<<"SOURCE_ADJOINT_PARITY progress="<<progress
                <<" iter="<<iter<<" status=linearization_failed\n";
            if(use_source_adjoint)break;
        }
        const auto gradient_end=std::chrono::steady_clock::now();
        double channel_norm[2]{};
        for(int i=0;i<n;++i)channel_norm[i%2]+=
            gradient[i]*gradient[i];
        for(auto &v:channel_norm)v=std::sqrt(v);
        if(channel_norm[0]<1e-8 && channel_norm[1]<1e-8)break;
        MpcModelControl_t best_controls[PREDICTION_HORIZON]{};
        double best=current;
        double chosen=0.;
        // One independent exact-source reference-optimization step with
        // separate physical-channel scaling, never inferred from QP merit.
        for(double scale:{.02,.05,.1,.2,.4}) {
            MpcModelControl_t candidate[PREDICTION_HORIZON]{};
            for(int k=0;k<PREDICTION_HORIZON;++k) {
                candidate[k].steering_target_rad=std::clamp(
                    float(controls[k].steering_target_rad-
                        scale*gradient[2*k]*.15/
                        std::max(1e-6,channel_norm[0])),
                    -config.model.max_steering_rad,config.model.max_steering_rad);
                candidate[k].throttle_command=std::clamp(
                    float(controls[k].throttle_command-
                        scale*gradient[2*k+1]*.25/
                        std::max(1e-6,channel_norm[1])),
                    MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
            }
            const double cost=evaluate(candidate);
            if(cost<best) {
                best=cost;chosen=scale;
                std::copy_n(candidate,PREDICTION_HORIZON,best_controls);
            }
        }
        if(chosen==0.) {
            std::cerr<<"SOURCE_ORACLE progress="<<progress
                <<" iter="<<iter<<" status=no_exact_descent"
                <<" gradient_steering_norm="<<channel_norm[0]
                <<" gradient_throttle_norm="<<channel_norm[1]<<'\n';
            break;
        }
        std::copy_n(best_controls,PREDICTION_HORIZON,controls);
        int hard_step_failure=-1;
        const auto hard_step_status=mpc_rti_validate_candidate_plan(
            &initial,progress,controls,hard_trajectory.data(),
            hard_trajectory.size(),length,PREDICTION_HORIZON,
            PREDICTION_DT_SECONDS,&config,trial.get(),&hard_step_failure);
        const auto iteration_end=std::chrono::steady_clock::now();
        std::cerr<<std::setprecision(9)
            <<"SOURCE_ORACLE_TIMING progress="<<progress
            <<" iter="<<iter
            <<" jacobian_and_adjoint_ms="
            <<std::chrono::duration<double,std::milli>(
                gradient_end-gradient_start).count()
            <<" total_source_iteration_ms="
            <<std::chrono::duration<double,std::milli>(
                iteration_end-iteration_start).count()
            <<'\n';
        std::cerr<<std::setprecision(9)
            <<"SOURCE_ORACLE_HARD_STEP progress="<<progress
            <<" iter="<<iter
            <<" hard_valid="<<int(hard_step_status==MPC_RTI_ROLLOUT_OK)
            <<" failure_stage="<<hard_step_failure
            <<" minimum_clearance="<<trial->validation_minimum_corridor_slack_m
            <<'\n';
        std::cerr<<std::setprecision(9)
            <<"SOURCE_ORACLE progress="<<progress
            <<" iter="<<iter<<" exact_cost="<<best
            <<" decrease="<<current-best
            <<" gradient_steering_norm="<<channel_norm[0]
            <<" gradient_throttle_norm="<<channel_norm[1]
            <<" source_objective_evaluations="<<(use_source_adjoint?5:n*2+5)
            <<" gradient_source="<<(use_source_adjoint?"source_adjoint":"finite_difference")
            <<" control_step_scale="<<chosen<<'\n';
        current=best;
    }
    int hard_failure=-1;
    const auto hard_status=mpc_rti_validate_candidate_plan(
        &initial,progress,controls,hard_trajectory.data(),
        hard_trajectory.size(),length,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&config,trial.get(),&hard_failure);
    std::cerr<<std::setprecision(9)
        <<"SOURCE_ORACLE_BEST progress="<<progress
        <<" exact_cost="<<current
        <<" hard_valid="<<int(hard_status==MPC_RTI_ROLLOUT_OK)
        <<" hard_failure_stage="<<hard_failure
        <<" hard_min_clearance="<<trial->validation_minimum_corridor_slack_m<<'\n';
    for(int stage=0;stage<PREDICTION_HORIZON;++stage)
        std::cerr<<std::setprecision(9)
            <<"SOURCE_ORACLE_CONTROL progress="<<progress
            <<" stage="<<stage
            <<" steering="<<controls[stage].steering_target_rad
            <<" throttle="<<controls[stage].throttle_command<<'\n';

}

void audit_source_shooting_iterations(
    const MpcRtiState_t &initial, double progress,
    const std::vector<MpcTrajectorySample_t> &trajectory,
    const std::vector<MpcTrajectorySample_t> &hard_trajectory,
    double length, MpcRtiCycleConfiguration_t config,
    const MpcModelControl_t seed[PREDICTION_HORIZON], int requested)
{
    if(requested<=0 || hard_trajectory.empty())return;
    requested=std::min(requested,8);
    config.enable_nonlinear_line_search=0;
    config.enable_feasibility_restoration=0;
    config.prefer_feasibility_first=0;
    config.feedback_rollout=0;
    config.max_sqp_iterations=1;
    config.objective_schedule=nullptr;
    config.qp_capture=nullptr;
    config.solver.max_solve_time_us=0.0;

    auto nominal=std::make_unique<MpcRtiNominal_t>();
    auto exact=std::make_unique<MpcRtiNominal_t>();
    auto problem=std::make_unique<MpcRtiProblem_t>();
    auto solution=std::make_unique<RiccatiSolution_t>();
    auto solver_state=std::make_unique<RiccatiAdmmState_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    MpcRtiObjectiveSchedule_t initial_objective{};
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    MpcModelControl_t best_controls[PREDICTION_HORIZON]{};
    std::copy_n(seed,PREDICTION_HORIZON,controls);
    int failed=-1;

    if(!mpc_rti_build_nominal_same_epoch(&initial,progress,nullptr,controls,
            PREDICTION_HORIZON,trajectory.data(),trajectory.size(),length,
            PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config.model,
            nominal.get(),refs) ||
       !mpc_rti_build_objective_schedule(nominal.get(),refs,&config.model,
            &initial_objective)) {
        std::cerr<<"SOURCE_SHOOTING progress="<<progress
            <<" status=initial_source_nominal_failed\n";return;
    }
    config.objective_schedule=&initial_objective;
    if(mpc_rti_validate_candidate_plan(&initial,progress,controls,
            hard_trajectory.data(),hard_trajectory.size(),length,PREDICTION_HORIZON,
            PREDICTION_DT_SECONDS,&config,exact.get(),&failed)
        !=MPC_RTI_ROLLOUT_OK) {
        std::cerr<<"SOURCE_SHOOTING progress="<<progress
            <<" status=initial_exact_invalid stage="<<failed<<'\n';return;
    }
    double best=exact->validation_objective;
    double last=best;
    std::copy_n(controls,PREDICTION_HORIZON,best_controls);
    std::cerr<<std::setprecision(10)
        <<"SOURCE_SHOOTING progress="<<progress<<" iter=0"
        <<" exact_cost="<<last<<" source_valid=1\n";
    for(int iteration=1;iteration<=requested;++iteration) {
        // Recompute every physical state, complete path progress, nominal
        // curvature schedule and source-derived A/B/Jacobians for THIS iterate.
        if(!mpc_rti_build_nominal_same_epoch(&initial,progress,nullptr,controls,
            PREDICTION_HORIZON,trajectory.data(),trajectory.size(),length,
            PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config.model,
            nominal.get(),refs)) {
            std::cerr<<"SOURCE_SHOOTING progress="<<progress
                <<" iter="<<iteration<<" status=nominal_failed\n";break;
        }
        /* Unlike the obsolete per-pass builder, preserve the original
         * cycle's EXACT objective as the physical nominal moves. Rebuilding
         * speed weights/reference targets here changes problems mid-SQP.
         * Only source A/B and geometric path-frame derivatives move. */
        if(!mpc_rti_build_ltv_qp_with_objective_schedule(
            nominal->states,nominal->controls,refs,
            PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&config.model,
            &initial_objective,problem.get())) {
            std::cerr<<"SOURCE_SHOOTING progress="<<progress
                <<" iter="<<iteration<<" status=jacobian_failed"
                <<" stage="<<problem->linearization_failure_stage<<'\n';break;
        }
        RiccatiAdmmConfig_t solver=config.solver;
        std::copy_n(problem->state_polygons,RICCATI_HORIZON+1,
            solver.state_polygons);
        // Solve with the actual route corridor polygons represented in this
        // SQP. The exact full-source body footprint remains independently
        // authoritative when validating candidate controls. Do not clear
        // geometric constraints to make an optimistic QP look converged.
        riccati_admm_state_init(solver_state.get());
        // An SQP shooting iteration already HAS a fully source-integrated
        // primal; initialize the ADMM split at that physically valid nominal.
        // Avoid the unrelated, unconstrained zero-rho cold Riccati solve,
        // which can be indefinite even if the constrained QP is well-posed.
        solver.cold_start_each_solve=0;
        solver.adaptive_rho=0; // Diagnostic A/B, not a production tuning change.
        solver_state->initialized=1;
        solver_state->rho=solver.rho;
        solver_state->rho_u=solver.rho_u;
        for(int k=0;k<=RICCATI_HORIZON;++k)
            for(int i=0;i<RICCATI_MAX_NX;++i)
                solver_state->z_x[k][i]=problem->x0[i];
        for(int k=0;k<RICCATI_HORIZON;++k) {
            const float prev_s=k?controls[k-1].steering_target_rad:
                problem->x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
            const float prev_t=k?controls[k-1].throttle_command:
                problem->x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
            solver_state->z_u[k][0]=controls[k].steering_target_rad-
                (config.model.command_increment_qp_enabled?prev_s:0.f);
            solver_state->z_u[k][1]=controls[k].throttle_command-
                (config.model.command_increment_qp_enabled?prev_t:0.f);
        }
        const auto solved=riccati_admm_solve(problem->steps,
            problem->terminal_Q,problem->terminal_q,
            problem->terminal_x_lb,problem->terminal_x_ub,
            problem->x0,&solver,solver_state.get(),solution.get());
        if(solved!=RICCATI_STATUS_OPTIMAL &&
           solved!=RICCATI_STATUS_MAX_ITERATIONS) {
            RiccatiDebugInfo_t debug{};
            riccati_debug_get_last(&debug);
            std::cerr<<"SOURCE_SHOOTING_DEBUG reg_count="
                <<debug.control_hessian_regularization_count
                <<" factor_count="<<debug.quadratic_factorization_count
                <<" qpsum="<<problem->steps[10].Q_diag[2]
                <<" qcross="<<problem->steps[10].Q_offdiag[2][4]
                <<" rx="<<problem->steps[10].R_cross
                <<" max_solver_iter="<<solver.max_iterations
                <<" zinit="<<int(solver_state->initialized)<<'\n';
            std::cerr<<"SOURCE_SHOOTING progress="<<progress
                <<" iter="<<iteration<<" status=qp_failed code="<<solved
                <<" iterations="<<solution->iterations
                <<" primal="<<solution->primal_residual
                <<" rho="<<solver.rho<<" rho_u="<<solver.rho_u
                <<" Q00="<<problem->steps[0].Q_diag[0]
                <<" R00="<<problem->steps[0].R_diag[0]
                <<" Rcross="<<problem->steps[0].R_cross
                <<" x0="<<problem->x0[0]<<'\n';
            break;
        }
        MpcModelControl_t candidate[PREDICTION_HORIZON]{};
        float previous_s=problem->x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
        float previous_t=problem->x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
        for(int k=0;k<RICCATI_HORIZON;++k) {
            const auto &step=problem->steps[k];
            const float ds=std::clamp(solution->u[k][0],
                step.u_lb[0],step.u_ub[0]);
            const float dt=std::clamp(solution->u[k][1],
                step.u_lb[1],step.u_ub[1]);
            candidate[k].steering_target_rad=std::clamp(
                ds+(config.model.command_increment_qp_enabled?previous_s:0.f),
                -config.model.max_steering_rad,config.model.max_steering_rad);
            candidate[k].throttle_command=std::clamp(
                dt+(config.model.command_increment_qp_enabled?previous_t:0.f),
                MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
            previous_s=candidate[k].steering_target_rad;
            previous_t=candidate[k].throttle_command;
        }
        const auto valid=mpc_rti_validate_candidate_plan(
            &initial,progress,candidate,
            hard_trajectory.data(),hard_trajectory.size(),
            length,PREDICTION_HORIZON,PREDICTION_DT_SECONDS,
            &config,exact.get(),&failed);
        if(valid!=MPC_RTI_ROLLOUT_OK) {
            std::cerr<<"SOURCE_SHOOTING progress="<<progress
                <<" iter="<<iteration<<" status=nonlinear_invalid"
                <<" failed_stage="<<failed<<" admm="<<int(solved)<<'\n';break;
        }
        // Diagnose the *actual new controls* against the frozen stage QP,
        // keeping command-increment coordinates consistent across the horizon.
        double lin[RICCATI_MAX_NX]{};
        std::copy_n(problem->x0,RICCATI_MAX_NX,lin);
        double err_ey=0.,err_r=0.,err_speed=0.;
        for(int k=0;k<RICCATI_HORIZON;++k) {
            const auto &stage=problem->steps[k];
            const double delta[2]={
                candidate[k].steering_target_rad-
                    (config.model.command_increment_qp_enabled
                        ?lin[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]:0.),
                candidate[k].throttle_command-
                    (config.model.command_increment_qp_enabled
                        ?lin[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]:0.)};
            double next[RICCATI_MAX_NX]{};
            for(int i=0;i<RICCATI_MAX_NX;++i) {
                double val=stage.d[i];
                for(int j=0;j<RICCATI_MAX_NX;++j)val+=stage.A[i][j]*lin[j];
                for(int a=0;a<2;++a)val+=stage.B[i][a]*delta[a];
                next[i]=val;
            }
            err_ey=std::max(err_ey,std::abs(next[MPC_OPT_IDX_EY]-
                exact->states[k+1].plant.e_y));
            err_r=std::max(err_r,std::abs(next[MPC_OPT_IDX_R]-
                exact->states[k+1].plant.r));
            err_speed=std::max(err_speed,std::abs(next[MPC_OPT_IDX_U]-
                exact->states[k+1].plant.u));
            std::copy_n(next,RICCATI_MAX_NX,lin);
        }
        const double actual=exact->validation_objective;
        const bool improvement=actual<best;
        if(improvement) {
            best=actual;
            std::copy_n(candidate,PREDICTION_HORIZON,best_controls);
        }
        std::cerr<<std::setprecision(10)
            <<"SOURCE_SHOOTING progress="<<progress
            <<" iter="<<iteration<<" exact_cost="<<actual
            <<" exact_cost_delta="<<actual-last
            <<" best_exact_cost="<<best
            <<" accepted_best="<<int(improvement)
            <<" source_valid=1 admm="<<int(solved)
            <<" iterations="<<solution->iterations
            <<" max_qp_vs_source_ey="<<err_ey
            <<" max_qp_vs_source_r="<<err_r
            <<" max_qp_vs_source_u="<<err_speed<<'\n';
        // A nonmonotone full step is allowed for EXPERIMENTAL curvature
        // discovery, not reported as an accepted controller action.
        std::copy_n(candidate,PREDICTION_HORIZON,controls);
        last=actual;
    }
    std::cerr<<std::setprecision(10)
        <<"SOURCE_SHOOTING_BEST progress="<<progress
        <<" objective="<<best<<" iterations_requested="<<requested<<'\n';
}

void audit_candidate_merit(const MpcRtiState_t &initial,double progress,
    const std::vector<MpcTrajectorySample_t> &trajectory,double length,
    MpcRtiCycleConfiguration_t config,const MpcRtiCycleResult_t &result,
    const MpcRtiQpCapture_t *frozen_qp)
{
    if(result.candidate_control_count!=PREDICTION_HORIZON ||
       !result.comparison_objective.valid)return;
    config.objective_schedule=&result.comparison_objective;
    auto wide=trajectory;
    for(auto &p:wide)p.left_bound=p.right_bound=100.;
    if(frozen_qp && frozen_qp->valid) {
        // The Riccati QP solves a *raw* input sequence. Feedback rollout
        // subsequently modifies absolute commands. Separate those effects:
        // a small ADMM residual certifies the RAW QP, not its nonlinear
        // feedback-modified command trajectory.
        MpcModelControl_t raw[PREDICTION_HORIZON]{};
        float previous_s=frozen_qp->problem.x0[
            MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
        float previous_t=frozen_qp->problem.x0[
            MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
        float max_steering_change=0.,max_throttle_change=0.;
        for(int k=0;k<PREDICTION_HORIZON;++k) {
            const auto &stage=frozen_qp->problem.steps[k];
            const float decision_s=std::clamp(
                frozen_qp->raw_solution.u[k][0],stage.u_lb[0],stage.u_ub[0]);
            const float decision_t=std::clamp(
                frozen_qp->raw_solution.u[k][1],stage.u_lb[1],stage.u_ub[1]);
            raw[k].steering_target_rad=std::clamp(
                decision_s+(config.model.command_increment_qp_enabled?previous_s:0.f),
                -config.model.max_steering_rad,config.model.max_steering_rad);
            raw[k].throttle_command=std::clamp(
                decision_t+(config.model.command_increment_qp_enabled?previous_t:0.f),
                MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
            previous_s=raw[k].steering_target_rad;
            previous_t=raw[k].throttle_command;
            max_steering_change=std::max(max_steering_change,
                std::abs(raw[k].steering_target_rad -
                    result.candidate_controls[k].steering_target_rad));
            max_throttle_change=std::max(max_throttle_change,
                std::abs(raw[k].throttle_command -
                    result.candidate_controls[k].throttle_command));
        }
        // OFFLINE CURVATURE PROBE: along the exact same multi-stage raw-QP
        // command direction, measure the finite-step nonlinearity of the
        // source model itself. Does one source-derived second derivative
        // improve the complete QP rollout, or is higher-order relinearization
        // required? Source is ground truth for this experiment, not a new
        // acceptance shortcut in production.
        {
            constexpr double h=.02;
            auto make_trial=[&](double a,MpcRtiNominal_t *out) {
                MpcModelControl_t input[PREDICTION_HORIZON]{};
                for(int k=0;k<PREDICTION_HORIZON;++k) {
                    input[k].steering_target_rad=
                        result.nominal_controls[k].steering_target_rad+
                        float(a)*(raw[k].steering_target_rad-
                             result.nominal_controls[k].steering_target_rad);
                    input[k].throttle_command=
                        result.nominal_controls[k].throttle_command+
                        float(a)*(raw[k].throttle_command-
                             result.nominal_controls[k].throttle_command);
                }
                int fail=-1;
                return mpc_rti_validate_candidate_plan(
                    &initial,progress,input,wide.data(),wide.size(),length,
                    PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&config,
                    out,&fail)==MPC_RTI_ROLLOUT_OK;
            };
            MpcRtiNominal_t y0{},y1{},y2{};
            if(make_trial(0.,&y0) && make_trial(h,&y1) &&
               make_trial(2*h,&y2)) {
                for(double alpha : {.005,.01,.02,.04,.08,.125,.25,.5,1.}) {
                    MpcRtiNominal_t exact{};
                    if(!make_trial(alpha,&exact))continue;
                    // State-reanchored ONE-STAGE Taylor defect, evaluated
                    // at true source states along this proposed trajectory.
                    // This distinguishes local unmodeled curvature from its
                    // propagated, whole-horizon amplification.
                    {
                        double local_max_ey=0.,local_max_r=0.,
                            local_max_u=0.,local_max_steer=0.;
                        int local_ey_stage=-1,local_r_stage=-1;
                        for(int k=0;k<RICCATI_HORIZON;++k) {
                            const auto& st=frozen_qp->problem.steps[k];
                            const auto& ps=exact.states[k].plant;
                            const auto& pn=exact.states[k+1].plant;
                            double state[RICCATI_MAX_NX]{};
                            state[MPC_OPT_IDX_EY]=ps.e_y;
                            state[MPC_OPT_IDX_EPSI]=ps.e_psi;
                            state[MPC_OPT_IDX_U]=ps.u;
                            state[MPC_OPT_IDX_V]=ps.v;
                            state[MPC_OPT_IDX_R]=ps.r;
                            state[MPC_OPT_IDX_ACTUAL_STEERING_ANGLE]=
                                ps.actual_steering_angle;
                            for(int j=0;j<14;++j)
                                state[6+j]=ps.source_plant_internal_state[j];
                            state[MPC_OPT_IDX_PROGRESS_DEVIATION]=
                                exact.progress[k]-y0.progress[k];
                            state[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]=k==0
                                ?frozen_qp->problem.x0[
                                    MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]
                                :result.nominal_controls[k-1].steering_target_rad+
                                    alpha*(raw[k-1].steering_target_rad-
                                        result.nominal_controls[k-1].steering_target_rad);
                            state[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]=k==0
                                ?frozen_qp->problem.x0[
                                    MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]
                                :result.nominal_controls[k-1].throttle_command+
                                    alpha*(raw[k-1].throttle_command-
                                        result.nominal_controls[k-1].throttle_command);
                            const double current_steering=
                                result.nominal_controls[k].steering_target_rad+
                                alpha*(raw[k].steering_target_rad-
                                    result.nominal_controls[k].steering_target_rad);
                            const double current_throttle=
                                result.nominal_controls[k].throttle_command+
                                alpha*(raw[k].throttle_command-
                                    result.nominal_controls[k].throttle_command);
                            const double input[2]={
                                current_steering-(config.model.command_increment_qp_enabled
                                    ?state[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]:0.),
                                current_throttle-(config.model.command_increment_qp_enabled
                                    ?state[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]:0.)};
                            double predicted[6]{};
                            for(int m=0;m<6;++m) {
                                double z=st.d[m];
                                for(int j=0;j<RICCATI_MAX_NX;++j)
                                    z+=st.A[m][j]*state[j];
                                for(int a=0;a<2;++a)z+=st.B[m][a]*input[a];
                                predicted[m]=z;
                            }
                            const double errs[4]={
                                std::abs(predicted[0]-pn.e_y),
                                std::abs(predicted[4]-pn.r),
                                std::abs(predicted[2]-pn.u),
                                std::abs(predicted[5]-pn.actual_steering_angle)};
                            if(errs[0]>local_max_ey) {
                                local_max_ey=errs[0];local_ey_stage=k+1;
                            }
                            if(errs[1]>local_max_r) {
                                local_max_r=errs[1];local_r_stage=k+1;
                            }
                            local_max_u=std::max(local_max_u,errs[2]);
                            local_max_steer=std::max(local_max_steer,errs[3]);
                        }
                        std::cerr<<std::setprecision(9)
                            <<"SOURCE_ONE_STAGE_DEFECT progress="<<progress
                            <<" alpha="<<alpha
                            <<" ey="<<local_max_ey<<" ey_stage="<<local_ey_stage
                            <<" yaw="<<local_max_r<<" yaw_stage="<<local_r_stage
                            <<" u="<<local_max_u
                            <<" steering="<<local_max_steer<<'\n';
                    }
                    double linear[RICCATI_MAX_NX]{};
                    std::copy_n(frozen_qp->problem.x0,
                        RICCATI_MAX_NX,linear);
                    double max_linear[5]{},max_second[5]{};
                    constexpr int channels[5]={MPC_OPT_IDX_EY,
                        MPC_OPT_IDX_EPSI,MPC_OPT_IDX_U,
                        MPC_OPT_IDX_V,MPC_OPT_IDX_R};
                    for(int k=0;k<RICCATI_HORIZON;++k) {
                        const auto& st=frozen_qp->problem.steps[k];
                        const float steering= result.nominal_controls[k].steering_target_rad+
                            float(alpha)*(raw[k].steering_target_rad-
                                result.nominal_controls[k].steering_target_rad);
                        const float throttle= result.nominal_controls[k].throttle_command+
                            float(alpha)*(raw[k].throttle_command-
                                result.nominal_controls[k].throttle_command);
                        const double u[2]={steering-
                            (config.model.command_increment_qp_enabled?
                                linear[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]:0.),
                            throttle-
                            (config.model.command_increment_qp_enabled?
                                linear[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]:0.)};
                        double next[RICCATI_MAX_NX]{};
                        for(int i=0;i<RICCATI_MAX_NX;++i) {
                            next[i]=st.d[i];
                            for(int j=0;j<RICCATI_MAX_NX;++j)
                                next[i]+=st.A[i][j]*linear[j];
                            for(int a=0;a<2;++a)next[i]+=st.B[i][a]*u[a];
                        }
                        const auto sample=[&](const MpcRtiNominal_t &n,int c)->double {
                            const auto &plant=n.states[k+1].plant;
                            switch(c) {
                                case 0:return plant.e_y;
                                case 1:return plant.e_psi;
                                case 2:return plant.u;
                                case 3:return plant.v;
                                default:return plant.r;
                            }
                        };
                        for(int c=0;c<5;++c) {
                            const double z0=sample(y0,c),zh=sample(y1,c),
                                z2=sample(y2,c),z=sample(exact,c);
                            const double slope=(4.*zh-z2-3.*z0)/(2.*h);
                            const double curvature=(z2-2.*zh+z0)/(h*h);
                            const double second=z0+alpha*slope+
                                0.5*alpha*alpha*curvature;
                            max_linear[c]=std::max(max_linear[c],
                                std::abs(next[channels[c]]-z));
                            max_second[c]=std::max(max_second[c],
                                std::abs(second-z));
                        }
                        std::copy_n(next,RICCATI_MAX_NX,linear);
                    }
                    std::cerr<<std::setprecision(9)
                        <<"SOURCE_SECOND_ORDER_AUDIT progress="<<progress
                        <<" alpha="<<alpha
                        <<" first_ey="<<max_linear[0]
                        <<" second_ey="<<max_second[0]
                        <<" first_yaw="<<max_linear[4]
                        <<" second_yaw="<<max_second[4]
                        <<" first_speed="<<max_linear[2]
                        <<" second_speed="<<max_second[2]<<'\n';
                }
            } else {
                std::cerr<<"SOURCE_SECOND_ORDER_AUDIT progress="<<progress
                    <<" status=nominal_perturbation_invalid\n";
            }
        }
        // OFFLINE exact-state midpoint relinearization. No search damping,
        // no controller acceptance change: quantify whether recentring
        // first-order source sensitivities on a *real nonlinear rollout*
        // improves the physical prediction for the same large command.
        {
            MpcModelControl_t midpoint[PREDICTION_HORIZON]{};
            for(int k=0;k<PREDICTION_HORIZON;++k) {
                midpoint[k].steering_target_rad=0.5f*(
                    result.nominal_controls[k].steering_target_rad+
                    raw[k].steering_target_rad);
                midpoint[k].throttle_command=0.5f*(
                    result.nominal_controls[k].throttle_command+
                    raw[k].throttle_command);
            }
            MpcRtiNominal_t centered{};
            MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
            const bool valid=mpc_rti_build_nominal_same_epoch(                &initial,progress,nullptr,midpoint,PREDICTION_HORIZON,
                wide.data(),wide.size(),length,PREDICTION_DT_SECONDS,
                PREDICTION_HORIZON,&config.model,&centered,refs);
            MpcRtiProblem_t midpoint_qp{};
            if(valid) {
                double max_s=0.,max_t=0.;
                for(int k=0;k<PREDICTION_HORIZON;++k) {
                    max_s=std::max(max_s,std::abs(double(
                        centered.controls[k].steering_target_rad-
                        midpoint[k].steering_target_rad)));
                    max_t=std::max(max_t,std::abs(double(
                        centered.controls[k].throttle_command-
                        midpoint[k].throttle_command)));
                }
                MpcRtiNominal_t independent{};int error=-1;
                auto exact=mpc_rti_validate_candidate_plan(&initial,progress,
                    midpoint,wide.data(),wide.size(),length,
                    PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&config,
                    &independent,&error);
                double err_ey=0.,err_r=0.;
                if(exact==MPC_RTI_ROLLOUT_OK) {
                    for(int k=0;k<=PREDICTION_HORIZON;++k) {
                        err_ey=std::max(err_ey,std::abs(double(
                            independent.states[k].plant.e_y-
                            centered.states[k].plant.e_y)));
                        err_r=std::max(err_r,std::abs(double(
                            independent.states[k].plant.r-
                            centered.states[k].plant.r)));
                    }
                }
                std::cerr<<std::setprecision(10)
                    <<"MIDPOINT_SEED_CONSISTENCY progress="<<progress
                    <<" ctrl_s="<<max_s<<" ctrl_t="<<max_t
                    <<" nominal_vs_exact_ey="<<err_ey
                    <<" nominal_vs_exact_r="<<err_r
                    <<" exact_status="<<int(exact)<<'\n';
            }
            if(valid && mpc_rti_build_ltv_qp(centered.states,
                    centered.controls,refs,PREDICTION_HORIZON,
                    PREDICTION_DT_SECONDS,&config.model,&midpoint_qp)) {
                for(double alpha : {0.5,0.6,0.75,1.0}) {
                    MpcModelControl_t u_abs[PREDICTION_HORIZON]{};
                    for(int k=0;k<PREDICTION_HORIZON;++k) {
                        u_abs[k].steering_target_rad=
                            result.nominal_controls[k].steering_target_rad+
                            float(alpha)*(raw[k].steering_target_rad-
                                result.nominal_controls[k].steering_target_rad);
                        u_abs[k].throttle_command=
                            result.nominal_controls[k].throttle_command+
                            float(alpha)*(raw[k].throttle_command-
                                result.nominal_controls[k].throttle_command);
                    }
                    MpcRtiNominal_t truth{};int failure=-1;
                    const auto valid_target=mpc_rti_validate_candidate_plan(
                        &initial,progress,u_abs,wide.data(),wide.size(),length,
                        PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&config,
                        &truth,&failure);
                    if(valid_target!=MPC_RTI_ROLLOUT_OK)continue;
                    // Reanchor EACH 25ms LTV transition on the true
                    // nonlinear state. This isolates Jacobian curvature
                    // without confounding cumulative previous-stage errors.
                    double local_yaw=0.,local_speed=0.,local_ey=0.;
                    int yaw_stage=-1;
                    for(int k=0;k<RICCATI_HORIZON;++k) {
                        const auto &real=truth.states[k].plant;
                        const auto &next_real=truth.states[k+1].plant;
                        const auto &st=midpoint_qp.steps[k];
                        double current[RICCATI_MAX_NX]{};
                        current[MPC_OPT_IDX_EY]=real.e_y;
                        current[MPC_OPT_IDX_EPSI]=real.e_psi;
                        current[MPC_OPT_IDX_U]=real.u;
                        current[MPC_OPT_IDX_V]=real.v;
                        current[MPC_OPT_IDX_R]=real.r;
                        current[MPC_OPT_IDX_ACTUAL_STEERING_ANGLE]=
                            real.actual_steering_angle;
                        for(int j=0;j<14;++j)
                            current[6+j]=real.source_plant_internal_state[j];
                        current[MPC_OPT_IDX_PROGRESS_DEVIATION]=
                            truth.progress[k]-centered.progress[k];
                        current[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]=k==0
                            ?midpoint_qp.x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]
                            :u_abs[k-1].steering_target_rad;
                        current[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]=k==0
                            ?midpoint_qp.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]
                            :u_abs[k-1].throttle_command;
                        const double inputs[2]={
                            double(u_abs[k].steering_target_rad)-
                                (config.model.command_increment_qp_enabled
                                ?current[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]:0.),
                            double(u_abs[k].throttle_command)-
                                (config.model.command_increment_qp_enabled
                                ?current[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]:0.)};
                        double prediction[5]{};
                        for(int m=0;m<5;++m) {
                            double v=st.d[m];
                            for(int j=0;j<RICCATI_MAX_NX;++j)
                                v+=st.A[m][j]*current[j];
                            for(int a=0;a<2;++a)v+=st.B[m][a]*inputs[a];
                            prediction[m]=v;
                        }
                        const double err_r=std::abs(prediction[4]-next_real.r);
                        if(err_r>local_yaw){local_yaw=err_r;yaw_stage=k+1;}
                        local_speed=std::max(local_speed,
                            std::abs(prediction[2]-next_real.u));
                        local_ey=std::max(local_ey,
                            std::abs(prediction[0]-next_real.e_y));
                    }
                    std::cerr<<std::setprecision(9)
                        <<"MIDPOINT_LOCAL_DEFECT progress="<<progress
                        <<" alpha="<<alpha
                        <<" max_ey="<<local_ey
                        <<" max_yaw="<<local_yaw
                        <<" yaw_stage="<<yaw_stage
                        <<" max_speed="<<local_speed<<'\n';
                    float x[RICCATI_MAX_NX]{};
                    std::copy_n(midpoint_qp.x0,RICCATI_MAX_NX,x);
                    double ey=0.,yaw=0.,v=0.;
                    for(int k=0;k<RICCATI_HORIZON;++k) {
                        const auto &stage=midpoint_qp.steps[k];
                        const float input[2]={
                            u_abs[k].steering_target_rad-
                                (config.model.command_increment_qp_enabled
                                    ?x[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]:0.f),
                            u_abs[k].throttle_command-
                                (config.model.command_increment_qp_enabled
                                    ?x[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]:0.f)};
                        float next[RICCATI_MAX_NX]{};
                        for(int i=0;i<RICCATI_MAX_NX;++i) {
                            double value=stage.d[i];
                            for(int j=0;j<RICCATI_MAX_NX;++j)
                                value+=stage.A[i][j]*x[j];
                            for(int a=0;a<2;++a)
                                value+=stage.B[i][a]*input[a];
                            next[i]=float(value);
                        }
                        const auto &t=truth.states[k+1].plant;
                        ey=std::max(ey,std::abs(double(next[MPC_OPT_IDX_EY]-t.e_y)));
                        yaw=std::max(yaw,std::abs(double(next[MPC_OPT_IDX_R]-t.r)));
                        v=std::max(v,std::abs(double(next[MPC_OPT_IDX_V]-t.v)));
                        std::copy_n(next,RICCATI_MAX_NX,x);
                    }
                    std::cerr<<std::setprecision(9)
                        <<"MIDPOINT_RELINEARIZATION progress="<<progress
                        <<" alpha="<<alpha
                        <<" max_ey="<<ey
                        <<" max_yaw="<<yaw
                        <<" max_v="<<v<<'\n';
                }
            } else {
                std::cerr<<"MIDPOINT_RELINEARIZATION progress="<<progress
                    <<" status=nominal_or_qp_failed\n";
            }
        }
        // Distinguish a defective nominal-point affine identity from normal
        // first-order Taylor-model divergence away from the nominal.
        MpcRtiNominal_t nominal_exact{};int nominal_failure=-1;
        const auto nominal_status=mpc_rti_validate_candidate_plan(&initial,
            progress,result.nominal_controls,trajectory.data(),
            trajectory.size(),length,PREDICTION_HORIZON,PREDICTION_DT_SECONDS,
            &config,&nominal_exact,&nominal_failure);
        if(nominal_status==MPC_RTI_ROLLOUT_OK) {
            double x[RICCATI_MAX_NX]{};
            std::copy_n(frozen_qp->problem.x0,RICCATI_MAX_NX,x);
            double max_nominal_ey=0.,max_nominal_epsi=0.,max_nominal_u=0.;
            for(int k=0;k<RICCATI_HORIZON;++k) {
                const auto &stage=frozen_qp->problem.steps[k];
                const double u[2]={
                    result.nominal_controls[k].steering_target_rad-
                        (config.model.command_increment_qp_enabled
                            ?x[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]:0.),
                    result.nominal_controls[k].throttle_command-
                        (config.model.command_increment_qp_enabled
                            ?x[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]:0.)};
                double next[RICCATI_MAX_NX]{};
                for(int i=0;i<RICCATI_MAX_NX;++i) {
                    next[i]=stage.d[i];
                    for(int j=0;j<RICCATI_MAX_NX;++j)
                        next[i]+=stage.A[i][j]*x[j];
                    for(int a=0;a<2;++a)next[i]+=stage.B[i][a]*u[a];
                }
                const auto &real=nominal_exact.states[k+1].plant;
                max_nominal_ey=std::max(max_nominal_ey,
                    std::abs(next[MPC_OPT_IDX_EY]-real.e_y));
                max_nominal_epsi=std::max(max_nominal_epsi,
                    std::abs(next[MPC_OPT_IDX_EPSI]-real.e_psi));
                max_nominal_u=std::max(max_nominal_u,
                    std::abs(next[MPC_OPT_IDX_U]-real.u));
                std::copy_n(next,RICCATI_MAX_NX,x);
            }
            std::cerr<<"LTV_NOMINAL_PARITY progress="<<progress
                <<" max_ey="<<max_nominal_ey
                <<" max_epsi="<<max_nominal_epsi
                <<" max_u="<<max_nominal_u<<'\n';
        } else {
            std::cerr<<"LTV_NOMINAL_PARITY progress="<<progress
                <<" invalid_exact_stage="<<nominal_failure<<'\n';
        }
        MpcRtiNominal_t raw_exact{};int failure=-1;
        const auto raw_status=mpc_rti_validate_candidate_plan(&initial,progress,
            raw,trajectory.data(),trajectory.size(),length,
            PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&config,
            &raw_exact,&failure);
        // Measure whether the actual QP step is a descent direction for
        // the ORIGINAL frozen exact-source objective (not its surrogate).
        // A first-order QP can generate a full-step physical prediction
        // error, but at infinitesimal alpha its true gradient MUST have
        // the correct sign for a locally useful descent proposal.
        double source_gradient[PREDICTION_HORIZON][2]{};
        if (mpc_rti_source_adjoint_gradient(
                &initial,progress,trajectory.data(),trajectory.size(),
                length,PREDICTION_DT_SECONDS,&config.model,
                result.nominal_controls,&result.comparison_objective,
                source_gradient)) {
            double raw_dot=0.,candidate_dot=0.;
            double gradient_norm=0.,raw_norm=0.,candidate_norm=0.;
            for(int k=0;k<PREDICTION_HORIZON;++k) {
                const double direction[2]={
                    raw[k].steering_target_rad-
                        result.nominal_controls[k].steering_target_rad,
                    raw[k].throttle_command-
                        result.nominal_controls[k].throttle_command};
                const double accepted[2]={
                    result.candidate_controls[k].steering_target_rad-
                        result.nominal_controls[k].steering_target_rad,
                    result.candidate_controls[k].throttle_command-
                        result.nominal_controls[k].throttle_command};
                for(int c=0;c<2;++c) {
                    const double g=source_gradient[k][c];
                    raw_dot+=g*direction[c];
                    candidate_dot+=g*accepted[c];
                    gradient_norm+=g*g;
                    raw_norm+=direction[c]*direction[c];
                    candidate_norm+=accepted[c]*accepted[c];
                }
            }
            const double raw_cos=raw_dot/std::max(1e-20,
                std::sqrt(gradient_norm*raw_norm));
            const double candidate_cos=candidate_dot/std::max(1e-20,
                std::sqrt(gradient_norm*candidate_norm));
            std::cerr<<std::setprecision(9)
                <<"SOURCE_QP_DIRECTION progress="<<progress
                <<" nominal_exact_adjoint_dot_raw="<<raw_dot
                <<" raw_cos="<<raw_cos
                <<" nominal_exact_adjoint_dot_candidate="<<candidate_dot
                <<" candidate_cos="<<candidate_cos
                <<" raw_direction_norm="<<std::sqrt(raw_norm)
                <<" gradient_norm="<<std::sqrt(gradient_norm)
                <<" local_raw_descent="<<int(raw_dot<0.)
                <<" local_feedback_candidate_descent="<<int(candidate_dot<0.)
                <<'\n';
        } else {
            std::cerr<<"SOURCE_QP_DIRECTION progress="<<progress
                <<" status=source_adjoint_failed\n";
        }
        const double raw_qp=audit_ltv_objective(frozen_qp->problem,raw,
            config.model.command_increment_qp_enabled);
        const double adjusted_qp=audit_ltv_objective(frozen_qp->problem,
            result.candidate_controls,config.model.command_increment_qp_enabled);
        if(raw_status==MPC_RTI_ROLLOUT_OK) {
            double x[RICCATI_MAX_NX]{};
            std::copy_n(frozen_qp->problem.x0,RICCATI_MAX_NX,x);
            double max_ey=0.,max_epsi=0.,max_u=0.,max_v=0.,max_r=0.;
            int ey_stage=-1,epsi_stage=-1,u_stage=-1,r_stage=-1;
            for(int k=0;k<RICCATI_HORIZON;++k) {
                const auto &stage=frozen_qp->problem.steps[k];
                const double u[2]={
                    raw[k].steering_target_rad-
                        (config.model.command_increment_qp_enabled
                            ?x[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]:0.),
                    raw[k].throttle_command-
                        (config.model.command_increment_qp_enabled
                            ?x[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]:0.)};
                double next[RICCATI_MAX_NX]{};
                for(int i=0;i<RICCATI_MAX_NX;++i) {
                    next[i]=stage.d[i];
                    for(int j=0;j<RICCATI_MAX_NX;++j)
                        next[i]+=stage.A[i][j]*x[j];
                    for(int a=0;a<2;++a)next[i]+=stage.B[i][a]*u[a];
                }
                const auto &real=raw_exact.states[k+1].plant;
                auto record=[&](double value,double& best,int* at){
                    if(value>best){best=value;if(at)*at=k+1;}
                };
                record(std::abs(next[MPC_OPT_IDX_EY]-real.e_y),max_ey,&ey_stage);
                record(std::abs(next[MPC_OPT_IDX_EPSI]-real.e_psi),max_epsi,&epsi_stage);
                record(std::abs(next[MPC_OPT_IDX_U]-real.u),max_u,&u_stage);
                record(std::abs(next[MPC_OPT_IDX_V]-real.v),max_v,nullptr);
                record(std::abs(next[MPC_OPT_IDX_R]-real.r),max_r,&r_stage);
                std::copy_n(next,RICCATI_MAX_NX,x);
            }
            std::cerr<<"LTV_EXACT progress="<<progress
                <<" max_abs_ey="<<max_ey<<" at="<<ey_stage
                <<" max_abs_epsi="<<max_epsi<<" at="<<epsi_stage
                <<" max_abs_u="<<max_u<<" at="<<u_stage
                <<" max_abs_v="<<max_v
                <<" max_abs_r="<<max_r<<" at="<<r_stage
                <<" final_linear_ey="<<x[MPC_OPT_IDX_EY]
                <<" final_exact_ey="<<raw_exact.states[RICCATI_HORIZON].plant.e_y
                <<'\n';
        }
        // Curvature diagnosis on the *raw QP direction*, without feedback
        // modifying the control sequence and without trusting a bounded
        // backtracking acceptance metric. QP and nonlinear objective values
        // have different additive constants, so compare differences from
        // alpha=0. Keep corridor validity as an independent outcome.
        for (const double alpha : {0., .005, .01, .02, .04, .08,
                                   .125, .25, .5, .75, 1.}) {
            MpcModelControl_t interpolated[PREDICTION_HORIZON]{};
            for (int k=0;k<PREDICTION_HORIZON;++k) {
                interpolated[k].steering_target_rad=static_cast<float>(
                    result.nominal_controls[k].steering_target_rad+
                    alpha*(raw[k].steering_target_rad-
                        result.nominal_controls[k].steering_target_rad));
                interpolated[k].throttle_command=static_cast<float>(
                    result.nominal_controls[k].throttle_command+
                    alpha*(raw[k].throttle_command-
                        result.nominal_controls[k].throttle_command));
            }
            const double quadratic=audit_ltv_objective(
                frozen_qp->problem,interpolated,
                config.model.command_increment_qp_enabled);
            MpcRtiNominal_t exact_no_wall{}; int failure=-1;
            const auto unconstrained=mpc_rti_validate_candidate_plan(
                &initial,progress,interpolated,wide.data(),wide.size(),
                length,PREDICTION_HORIZON,PREDICTION_DT_SECONDS,
                &config,&exact_no_wall,&failure);
            double exact_cost=unconstrained==MPC_RTI_ROLLOUT_OK
                ? exact_no_wall.validation_objective:NAN;
            MpcRtiNominal_t exact_with_wall{};int hard_stage=-1;
            const auto hard=mpc_rti_validate_candidate_plan(
                &initial,progress,interpolated,trajectory.data(),
                trajectory.size(),length,PREDICTION_HORIZON,
                PREDICTION_DT_SECONDS,&config,&exact_with_wall,&hard_stage);
            if(unconstrained==MPC_RTI_ROLLOUT_OK &&
               (alpha==0. || alpha==.04 || alpha==.08 || alpha==.125 ||
                alpha==.25 || alpha==.5 || alpha==1.)) {
                double linear[RICCATI_MAX_NX]{};
                std::copy_n(frozen_qp->problem.x0,RICCATI_MAX_NX,linear);
                double ey=0.,epsi=0.,speed=0.,yaw=0.;
                int ey_stage=-1;
                for(int k=0;k<RICCATI_HORIZON;++k) {
                    const auto &step=frozen_qp->problem.steps[k];
                    double command[2]={
                        interpolated[k].steering_target_rad-
                            (config.model.command_increment_qp_enabled
                                ?linear[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]:0.),
                        interpolated[k].throttle_command-
                            (config.model.command_increment_qp_enabled
                                ?linear[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]:0.)};
                    double next[RICCATI_MAX_NX]{};
                    for(int i=0;i<RICCATI_MAX_NX;++i) {
                        next[i]=step.d[i];
                        for(int j=0;j<RICCATI_MAX_NX;++j)
                            next[i]+=step.A[i][j]*linear[j];
                        for(int a=0;a<2;++a)
                            next[i]+=step.B[i][a]*command[a];
                    }
                    const auto &real=exact_no_wall.states[k+1].plant;
                    const auto eye=std::abs(next[MPC_OPT_IDX_EY]-real.e_y);
                    if(eye>ey){ey=eye;ey_stage=k+1;}
                    epsi=std::max(epsi,std::abs(
                        next[MPC_OPT_IDX_EPSI]-real.e_psi));
                    speed=std::max(speed,std::abs(
                        next[MPC_OPT_IDX_U]-real.u));
                    yaw=std::max(yaw,std::abs(
                        next[MPC_OPT_IDX_R]-real.r));
                    std::copy_n(next,RICCATI_MAX_NX,linear);
                }
                std::cerr<<std::setprecision(13)
                    <<"CURVATURE_STATE_ERROR progress="<<progress
                    <<" alpha="<<alpha
                    <<" max_ey_m="<<ey<<" ey_stage="<<ey_stage
                    <<" max_heading_rad="<<epsi
                    <<" max_speed_mps="<<speed
                    <<" max_yaw_radps="<<yaw<<'\n';
            }
            // Distinguish continuous corner-state coupling from discrete
            // brake/clamp branch transitions along the same physical plan.
            double max_slip=0.,max_lateral_accel=0.,max_abs_yaw_rate=0.;
            int brake_stages=0,clamped_stages=0;
            if(unconstrained==MPC_RTI_ROLLOUT_OK) {
                for(int k=1;k<=RICCATI_HORIZON;++k) {
                    const auto &state=exact_no_wall.states[k];
                    max_abs_yaw_rate=std::max(max_abs_yaw_rate,
                        std::abs((double)state.plant.r));
                    max_lateral_accel=std::max(max_lateral_accel,
                        std::abs((double)state.plant.u*state.plant.r));
                    if(state.incoming_branch_flags&MPC_STAGE_PLANT_BRAKE)
                        ++brake_stages;
                    if(state.incoming_branch_flags&(
                            MPC_STAGE_CLIPPED_STEERING_TARGET|
                            MPC_STAGE_CLIPPED_THROTTLE))
                        ++clamped_stages;
                    for(double slip:state.incoming_wheel_slip_excess_mps)
                        max_slip=std::max(max_slip,slip);
                }
            }
            std::cerr<<std::setprecision(13);
            std::cerr<<"RAW_DIRECTION_CURVATURE progress="<<progress
                <<" alpha="<<alpha<<" quadratic="<<quadratic
                <<" max_lateral_accel="<<max_lateral_accel
                <<" max_slip_excess="<<max_slip
                <<" max_abs_yaw_rate="<<max_abs_yaw_rate
                <<" brake_stages="<<brake_stages
                <<" clamp_stages="<<clamped_stages
                <<" exact="<<exact_cost
                <<" no_wall_status="<<static_cast<int>(unconstrained)
                <<" hard_status="<<static_cast<int>(hard)
                <<" hard_failure_stage="<<hard_stage
                <<'\n';
        }
        // Isolate nonlinear coupling by varying steering only, throttle
        // only, and their joint QP direction. Diagnostics never publish.
        for (int mode=0;mode<3;++mode) {
            for (double alpha : {0.,0.005,0.01,0.02,0.04,0.1,0.25,0.5,1.}) {
                MpcModelControl_t u[PREDICTION_HORIZON]{};
                for (int k=0;k<PREDICTION_HORIZON;++k) {
                    const auto &n=result.nominal_controls[k];
                    u[k].steering_target_rad=static_cast<float>(
                        n.steering_target_rad+
                        ((mode==1)?0.:alpha*(raw[k].steering_target_rad-
                                                  n.steering_target_rad)));
                    u[k].throttle_command=static_cast<float>(
                        n.throttle_command+
                        ((mode==0)?0.:alpha*(raw[k].throttle_command-
                                                  n.throttle_command)));
                }
                MpcRtiNominal_t nonlinear{}; int failure=-1;
                const auto relaxed=mpc_rti_validate_candidate_plan(
                    &initial,progress,u,wide.data(),wide.size(),length,
                    PREDICTION_HORIZON,PREDICTION_DT_SECONDS,
                    &config,&nonlinear,&failure);
                const double cost=(relaxed==MPC_RTI_ROLLOUT_OK)
                    ? nonlinear.validation_objective:NAN;
                const double qpcost=audit_ltv_objective(frozen_qp->problem,u,
                    config.model.command_increment_qp_enabled);
                int wall_failure=-1;
                const auto physical=mpc_rti_validate_candidate_plan(
                    &initial,progress,u,trajectory.data(),trajectory.size(),
                    length,PREDICTION_HORIZON,PREDICTION_DT_SECONDS,
                    &config,&nonlinear,&wall_failure);
                std::cerr<<std::setprecision(12)
                    <<"CONTROL_CHANNEL_CURVATURE progress="<<progress
                    <<" mode="<<(mode==0?"steering":mode==1?"throttle":"joint")
                    <<" alpha="<<alpha<<" qp="<<qpcost<<" exact="<<cost
                    <<" hard_valid="<<(physical==MPC_RTI_ROLLOUT_OK)
                    <<" failure_stage="<<wall_failure<<'\n';
            }
        }
        std::cerr<<"RAW_VS_FEEDBACK progress="<<progress
            <<" delta_steering_max="<<max_steering_change
            <<" delta_throttle_max="<<max_throttle_change
            <<" raw_qp="<<raw_qp<<" adjusted_qp="<<adjusted_qp
            <<" raw_exact_valid="<<(raw_status==MPC_RTI_ROLLOUT_OK)
            <<" raw_exact_objective="<<raw_exact.validation_objective
            <<" raw_failure_stage="<<failure<<'\n';
    }
    if(const char *iterations=std::getenv("MPC_REPLAY_EXACT_SOURCE_ORACLE_ITERATIONS")) {
        const int count=std::atoi(iterations);
        if(count>0 && count<=8) {
            auto source_track=trajectory;
            for(auto &sample:source_track)
                sample.left_bound=sample.right_bound=100.;
            const bool physical_geometry=
                std::getenv("MPC_REPLAY_ORACLE_HARD_GEOMETRY")!=nullptr;
            audit_exact_source_control_oracle(initial,progress,
                physical_geometry?trajectory:source_track,
                trajectory,length,config,result.nominal_controls,count);
        }
    }
    if(const char *iterations=std::getenv(
            "MPC_REPLAY_SOURCE_NONLINEAR_CORE_ITERATIONS")) {
        const int count=std::atoi(iterations);
        if(count>0 && count<=8) {
            MpcRtiSourceNonlinearOptions_t options{};
            options.max_iterations=count;
            options.max_exact_trials_per_iteration=9;
            if (const char *v=std::getenv(
                    "MPC_REPLAY_SOURCE_NONLINEAR_MAX_TRIALS"))
                options.max_exact_trials_per_iteration=std::atoi(v);
            options.initial_step_scale=.1f;
            options.minimum_step_scale=.005f;
            options.armijo_fraction=1.e-4f;
            options.steering_direction_norm_rad=.15f;
            options.throttle_direction_norm=.25f;
            options.directional_check_step_scale=.0003f;
            if (const char *v=std::getenv(
                    "MPC_REPLAY_SOURCE_NONLINEAR_TIME_BUDGET_US"))
                options.maximum_wall_time_us=std::atof(v);
            if (const char *v=std::getenv(
                    "MPC_REPLAY_SOURCE_NONLINEAR_INITIAL_SCALE"))
                options.initial_step_scale=std::atof(v);
            if (const char *v=std::getenv(
                    "MPC_REPLAY_SOURCE_NONLINEAR_GRADIENT_CHECK_SCALE"))
                options.directional_check_step_scale=std::atof(v);
            if (const char *v=std::getenv(
                    "MPC_REPLAY_SOURCE_NONLINEAR_STEERING_NORM"))
                options.steering_direction_norm_rad=std::atof(v);
            if (const char *v=std::getenv(
                    "MPC_REPLAY_SOURCE_NONLINEAR_THROTTLE_NORM"))
                options.throttle_direction_norm=std::atof(v);
            auto improvement=std::make_unique<MpcRtiSourceNonlinearResult_t>();
            const int success=mpc_rti_source_nonlinear_optimize(
                &initial,progress,trajectory.data(),trajectory.size(),
                length,PREDICTION_DT_SECONDS,&config,
                &result.comparison_objective,result.nominal_controls,
                &options,improvement.get());
            std::cerr<<std::setprecision(10)
                <<"SOURCE_NONLINEAR_CORE progress="<<progress
                <<" success="<<success
                <<" initial_valid="<<improvement->initial_valid
                <<" final_valid="<<improvement->final_valid
                <<" initial_failure_reason="<<improvement->initial_failure_reason
                <<" initial_failure_stage="<<improvement->initial_failure_stage
                <<" phase1_attempts="<<improvement->feasibility_iterations_attempted
                <<" phase1_accepted="<<improvement->feasibility_iterations_accepted
                <<" phase1_feasible="<<improvement->feasible_seed_recovered
                <<" phase1_initial_violation="<<improvement->initial_corridor_violation_m2
                <<" phase1_final_violation="<<improvement->final_corridor_violation_m2
                <<" initial_exact_cost="<<improvement->initial_objective
                <<" final_exact_cost="<<improvement->final_objective
                <<" accepted_iterations="<<improvement->iterations_accepted
                <<" attempted_iterations="<<improvement->iterations_attempted
                <<" full_source_rollouts="<<improvement->exact_rollout_evaluations
                <<" source_adjoint_evaluations="<<improvement->source_adjoint_evaluations
                <<" gradient_checks="<<improvement->source_derivative_consistency_checks
                <<" gradient_discrepancies="<<improvement->source_derivative_discrepancies
                <<" max_directional_relative_error="<<improvement->max_directional_derivative_relative_error
                <<" corridor_rejections="<<improvement->trial_corridor_rejections
                <<" model_rejections="<<improvement->trial_model_rejections
                <<" time_budget_exhausted="<<improvement->budget_exhausted
                <<" clearance_m="<<improvement->minimum_corridor_clearance_m
                <<" max_steering_change="<<improvement->largest_control_step_rad
                <<" max_throttle_change="<<improvement->largest_throttle_step
                <<" wall_time_ms="<<improvement->wall_time_ms<<'\n';
            for(int stage=0;stage<improvement->iterations_attempted;++stage) {
                std::cerr<<std::setprecision(10)
                    <<"SOURCE_NONLINEAR_CORE_ITER progress="<<progress
                    <<" index="<<stage+1
                    <<" accepted="<<int(stage<improvement->iterations_accepted)
                    <<" exact_cost="<<improvement->accepted_exact_objectives[
                        std::min(stage+1,improvement->iterations_accepted)]
                    <<" selected_scale="<<improvement->accepted_step_scales[stage]
                    <<" directional_error="<<improvement->directional_relative_error_by_iteration[stage]
                    <<'\n';
            }
            for(int stage=0;stage<PREDICTION_HORIZON && success;++stage) {
                std::cerr<<std::setprecision(10)
                    <<"SOURCE_NONLINEAR_CORE_CONTROL progress="<<progress
                    <<" stage="<<stage
                    <<" steering="<<improvement->controls[stage].steering_target_rad
                    <<" throttle="<<improvement->controls[stage].throttle_command
                    <<'\n';
            }
        }
    }
    if(const char *iterations=std::getenv("MPC_REPLAY_NONLINEAR_SHOOTING_ITERATIONS")) {
        const int count=std::atoi(iterations);
        if(count>0 && count<=8) {
            audit_source_shooting_iterations(
                initial,progress,trajectory,trajectory,
                length,config,result.nominal_controls,count);
        }
    }
    for(float alpha:{0.f,1.f,.5f,.25f,.125f,.0625f,.03125f,.015625f,.0078125f}) {
        MpcModelControl_t controls[PREDICTION_HORIZON];
        for(int k=0;k<PREDICTION_HORIZON;++k) {
            const auto &a=result.nominal_controls[k],&b=result.candidate_controls[k];
            controls[k].steering_target_rad=a.steering_target_rad+
                alpha*(b.steering_target_rad-a.steering_target_rad);
            controls[k].throttle_command=a.throttle_command+
                alpha*(b.throttle_command-a.throttle_command);
        }
        MpcRtiNominal_t exact{};int failure=-1;
        auto status=mpc_rti_validate_candidate_plan(&initial,progress,controls,
            wide.data(),wide.size(),length,PREDICTION_HORIZON,
            PREDICTION_DT_SECONDS,&config,&exact,&failure);
        const double objective=status==MPC_RTI_ROLLOUT_OK?exact.validation_objective:NAN;
        const double quadratic=(frozen_qp && frozen_qp->valid)
            ? audit_ltv_objective(frozen_qp->problem,controls,
                config.model.command_increment_qp_enabled) : NAN;
        status=mpc_rti_validate_candidate_plan(&initial,progress,controls,
            trajectory.data(),trajectory.size(),length,PREDICTION_HORIZON,
            PREDICTION_DT_SECONDS,&config,&exact,&failure);
        std::cerr<<"MERIT progress="<<progress<<" alpha="<<alpha<<" objective="
            <<objective<<" quadratic="<<quadratic
            <<" exact_status="<<status<<" failure_stage="<<failure<<'\n';
    }
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 8 || (argc > 31 && argc != 43 && argc != 44 &&
                     argc != 45 && argc != 46 && argc != 47 && argc != 48 && argc != 49 && argc != 50)) {
        std::cerr << "usage: mpc_bag_solver_replay TRAJECTORY INPUT_CSV "
                     "RHO RHO_STEER RHO_THROTTLE ALPHA COST_SCALE "
                     "[RHO_EY RHO_U_STATE RHO_STEER_STATE "
                     "[RHO_MAX PERIOD TOLERANCE "
                     "[sequence|independent|nominal-cold|reference-seed] "
                     "[warm|cold|dual-cold] "
                     "[JACOBIAN_THREADS [MAX_ITERATIONS [STARTUP_SEED "
                     "[EY_WEIGHT_MULTIPLIER [EPSI_WEIGHT_MULTIPLIER "
                     "[EY_STATE_SCALE [EPSI_STATE_SCALE "
                     "[CORRIDOR_MARGIN [FIRST_CORRIDOR_MARGIN "
                     "[RHO_U [STEERING_SLEW_WEIGHT "
                     "[THROTTLE_SLEW_WEIGHT [ADAPTIVE_RHO "
                     "[LINE_SEARCH_BACKTRACKS "
                     "[SPEED_TRACKING_WEIGHT_MULTIPLIER "
                     "[DIRECT_WEIGHTS: EY EPSI U U_OVERSPEED SLIP V R "
                     "STEER_COMMAND STEER_TARGET THROTTLE_COMMAND "
                     "STEER_SLEW THROTTLE_SLEW [MAX_SQP_ITERATIONS: 1..2 "
                     "[OVERSPEED_CURVATURE_GAIN_M: 0..10 "
                     "[UNDERSPEED_CURVATURE_GAIN_M: 0..10 "
                     "[SLIP_CURVATURE_GAIN_M: 0..10]]]]]]]]]]]]]]]]]]]] "
                     "(MAX_ITERATIONS: 1..300; optional final arguments after "
                     "direct weights: FEEDBACK_ROLLOUT [REFERENCE_CLEARANCE_RESERVE_M])\n";
        return 2;
    }

    try {
        const double rho = number_arg(argv, 3, 2.0);
        const double rho_steer = number_arg(argv, 4, 1.0);
        const double rho_throttle = number_arg(argv, 5, 7.0);
        const double alpha = number_arg(argv, 6, 1.6);
        const double cost_scale = number_arg(argv, 7, 0.001);
        const double rho_ey = argc >= 9 ? number_arg(argv, 8, 0.0) : 0.0;
        const double rho_u_state = argc >= 10 ? number_arg(argv, 9, 0.0) : 0.0;
        const double rho_steer_state = argc >= 11
            ? number_arg(argv, 10, 0.0) : 0.0;
        const double rho_max = argc >= 12 ? number_arg(argv, 11, 32.0) : 32.0;
        const int rho_period = argc >= 13
            ? static_cast<int>(number_arg(argv, 12, 10.0)) : 10;
        const double tolerance = argc >= 14
            ? number_arg(argv, 13, 0.02) : 0.02;
        const std::string memory_mode = argc >= 15 ? argv[14] : "sequence";
        const std::string solver_start = argc >= 16 ? argv[15] : "warm";
        const int jacobian_threads = argc >= 17
            ? static_cast<int>(number_arg(argv, 16, 4.0)) : 4;
        const int max_iterations = argc >= 18
            ? static_cast<int>(number_arg(argv, 17, 100.0)) : 100;
        const double startup_seed = argc >= 19
            ? number_arg(argv, 18, 0.25) : 0.25;
        const double ey_weight_multiplier = argc >= 20
            ? number_arg(argv, 19, 1.0) : 1.0;
        const double epsi_weight_multiplier = argc >= 21
            ? number_arg(argv, 20, 1.0) : 1.0;
        const double ey_state_scale = argc >= 22
            ? number_arg(argv, 21, 0.10) : 0.10;
        const double epsi_state_scale = argc >= 23
            ? number_arg(argv, 22, 0.10) : 0.10;
        const double corridor_margin = argc >= 24
            ? number_arg(argv, 23, 0.05) : 0.05;
        const double first_corridor_margin = argc >= 25
            ? number_arg(argv, 24, corridor_margin) : corridor_margin;
        const double rho_u = argc >= 26 ? number_arg(argv, 25, 7.0) : 7.0;
        const double steering_slew_weight = argc >= 27
            ? number_arg(argv, 26, 25.0) : 25.0;
        const double throttle_slew_weight = argc >= 28
            ? number_arg(argv, 27, 2.0) : 2.0;
        const double adaptive_rho_value = argc >= 29
            ? number_arg(argv, 28, 1.0) : 1.0;
        const bool adaptive_rho = adaptive_rho_value != 0.0;
        const int line_search_backtracks = argc >= 30
            ? static_cast<int>(number_arg(argv, 29, 0.0)) : 0;
        const double speed_tracking_weight_multiplier = argc >= 31
            ? number_arg(argv, 30, 1.0) : 1.0;
        if (!(cost_scale > 0.0) ||
            !(tolerance > 0.0) ||
            (adaptive_rho_value != 0.0 && adaptive_rho_value != 1.0) ||
            !std::isfinite(rho_u) || !(rho_u > 0.0) || rho_u > 127.0 ||
            (memory_mode != "sequence" && memory_mode != "independent" &&
             memory_mode != "nominal-cold" &&
             memory_mode != "reference-seed") ||
            (solver_start != "warm" && solver_start != "cold" &&
             solver_start != "dual-cold") ||
            jacobian_threads < 0 ||
                jacobian_threads > MPC_RTI_MAX_JACOBIAN_THREADS ||
            max_iterations < 1 || max_iterations > MAX_ITERATIONS ||
            startup_seed < 0.0 || startup_seed > 1.0 ||
            !(ey_weight_multiplier > 0.0) || ey_weight_multiplier > 100.0 ||
            !(epsi_weight_multiplier > 0.0) ||
            epsi_weight_multiplier > 100.0 ||
            !(ey_state_scale > 0.0) || ey_state_scale > 10.0 ||
            !(epsi_state_scale > 0.0) || epsi_state_scale > 10.0 ||
            corridor_margin < 0.0 ||
            first_corridor_margin < 0.0 ||
            first_corridor_margin > corridor_margin ||
            !std::isfinite(steering_slew_weight) ||
            steering_slew_weight < 0.0 ||
            !std::isfinite(throttle_slew_weight) ||
            throttle_slew_weight < 0.0 ||
            line_search_backtracks < 0 || line_search_backtracks > 4 ||
            !std::isfinite(speed_tracking_weight_multiplier) ||
            !(speed_tracking_weight_multiplier > 0.0) ||
            speed_tracking_weight_multiplier > 100.0)
            throw std::runtime_error("invalid replay configuration");

        double lap_length = 0.0;
        const auto trajectory = load_trajectory(argv[1], &lap_length);
        std::ifstream input(argv[2]);
        if (!input) throw std::runtime_error("cannot open input CSV");
        const auto nominal_seed_controls = load_nominal_seed_controls(
            std::getenv("MPC_REPLAY_NOMINAL_SEED_CSV"));
        std::set<std::size_t> consumed_nominal_seed_rows;

        MpcRtiCycleConfiguration_t config{};
        configure(&config, rho, rho_steer, rho_throttle, alpha, cost_scale,
                  rho_ey, rho_u_state, rho_steer_state,
                  rho_max, rho_period, tolerance, solver_start == "cold",
                  jacobian_threads, max_iterations, startup_seed,
                  ey_weight_multiplier,
                  epsi_weight_multiplier, ey_state_scale, epsi_state_scale,
                  corridor_margin, first_corridor_margin, rho_u,
                  steering_slew_weight, throttle_slew_weight, adaptive_rho,
                  speed_tracking_weight_multiplier);
        config.max_sqp_iterations = argc >= 44
            ? static_cast<int>(number_arg(argv, 43, 1.0)) : 1;
        if (config.max_sqp_iterations < 1 ||
            config.max_sqp_iterations > 2)
            throw std::runtime_error("MAX_SQP_ITERATIONS must be in [1, 2]");
        if(std::getenv("MPC_REPLAY_FEASIBILITY_FIRST")) {
            config.enable_feasibility_restoration=1;config.prefer_feasibility_first=1;
            config.feasibility_restoration_qp_relaxation_m=.18f;
            config.feasibility_restoration_wall_weight=8000.f;
        }
        if(const char *budget=std::getenv("MPC_REPLAY_CYCLE_BUDGET_US")) {
            const double value=std::stod(budget);
            if(!std::isfinite(value) || value<=0.)
                throw std::runtime_error("invalid replay cycle budget");
            config.max_cycle_time_us=value;
        }
        config.terminal_viability=std::getenv("MPC_REPLAY_TERMINAL_VIABILITY")?1:0;
        config.source_informed_sqp=std::getenv("MPC_REPLAY_SOURCE_INFORMED_SQP")?1:0;
        config.prefer_terminal_continuable_refinement =
            std::getenv("MPC_REPLAY_TERMINAL_REFINEMENT") ? 1 : 0;
        // Offline parity with live increment-QP controller. These switches
        // are intentionally independent of old positional replay arguments.
        if(const char *enabled=std::getenv("MPC_REPLAY_COMMAND_INCREMENT_QP")) {
            if(std::string(enabled)!="1" && std::string(enabled)!="0")
                throw std::runtime_error("MPC_REPLAY_COMMAND_INCREMENT_QP must be 0 or 1");
            config.model.command_increment_qp_enabled=*enabled=='1';
            if(config.model.command_increment_qp_enabled) {
                config.model.max_steering_command_step_rad =
                    2.f*config.model.max_steering_rad;
                config.model.max_throttle_command_step=1.f;
            }
        }
        if(const char *gain=std::getenv("MPC_REPLAY_FEEDBACK_GAIN")) {
            const double value=std::stod(gain);
            if(!std::isfinite(value) || value<0. || value>1.)
                throw std::runtime_error("MPC_REPLAY_FEEDBACK_GAIN must be in [0,1]");
            config.feedback_rollout_gain_scale=static_cast<float>(value);
        }
        FrozenWorldMap frozen_world;
        if(const char *file=std::getenv("MPC_REPLAY_WORLD_MAP_FILE")) {
            // Offline test-only fixture: recorded 2026 IROS map metadata
            // accompanies exactly 238x903 signed occupancy pixels.
            constexpr int width=238,height=903;
            std::ifstream stream(file,std::ios::binary);
            if(!stream)throw std::runtime_error("cannot open frozen world map");
            std::vector<int8_t> cells(width*height);
            if(!stream.read(reinterpret_cast<char*>(cells.data()),
                            static_cast<std::streamsize>(cells.size())))
                throw std::runtime_error("frozen map is truncated");
            frozen_world.grid=f1tenth_mpc::MpcWorldFootprint::from_occupancy(
                width,height,0.02500000037252903,-.742,-20.,0.,cells);
            if(!frozen_world.grid)
                throw std::runtime_error("frozen world map rejected");
            config.model.world_sweep_check=frozen_world_sweep;
            config.model.world_sweep_context=&frozen_world;
            config.model.world_map_rejection_only=0;
        }
        /* Production tuning sweeps use the exact twelve production-stage
         * weights, rather than trying to synthesize them from this replay
         * executable's historical global cost multipliers. Keep the older
         * multiplier interface intact for existing benchmark artifacts. */
        if (argc >= 43) {
            float * const direct_weights[] = {
                &config.model.weight_e_y,
                &config.model.weight_e_psi,
                &config.model.weight_u,
                &config.model.weight_u_overspeed,
                &config.model.weight_longitudinal_slip,
                &config.model.weight_v,
                &config.model.weight_r,
                &config.model.weight_steering_command,
                &config.model.weight_steering_target,
                &config.model.weight_throttle_command,
                &config.model.weight_steering_slew,
                &config.model.weight_throttle_slew,
            };
            for (int index = 0; index < 12; ++index) {
                const double value = number_arg(argv, 31 + index, 0.0);
                if (!std::isfinite(value) || value < 0.0 || value > 1.0e8)
                    throw std::runtime_error(
                        "direct replay weights must be finite and in [0, 1e8]");
                *direct_weights[index] = static_cast<float>(value);
            }
        }
        if (argc >= 45) {
            const double curvature_gain = number_arg(argv, 44, 0.0);
            if (!std::isfinite(curvature_gain) || curvature_gain < 0.0 ||
                curvature_gain > 10.0)
                throw std::runtime_error(
                    "overspeed curvature gain must be finite and in [0, 10]");
            config.model.overspeed_curvature_gain_m =
                static_cast<float>(curvature_gain);
        }
        if (argc >= 46) {
            const double curvature_gain = number_arg(argv, 45, 0.0);
            if (!std::isfinite(curvature_gain) || curvature_gain < 0.0 ||
                curvature_gain > 10.0)
                throw std::runtime_error(
                    "underspeed curvature gain must be finite and in [0, 10]");
            config.model.underspeed_curvature_gain_m =
                static_cast<float>(curvature_gain);
        }
        if (argc >= 47) {
            const double curvature_gain = number_arg(argv, 46, 0.0);
            if (!std::isfinite(curvature_gain) || curvature_gain < 0.0 ||
                curvature_gain > 10.0)
                throw std::runtime_error(
                    "slip curvature gain must be finite and in [0, 10]");
            config.model.longitudinal_slip_curvature_gain_m =
                static_cast<float>(curvature_gain);
        }
        if (argc >= 48) {
            const double feedback_mode = number_arg(argv, 47, 0);
            if (feedback_mode != 0. && feedback_mode != 1.)
                throw std::runtime_error("feedback rollout must be 0 or 1");
            config.feedback_rollout = feedback_mode == 1.;
        }
        if (argc >= 49) {
            const double reserve = number_arg(argv, 48, 0.);
            if (!std::isfinite(reserve) || reserve < 0.)
                throw std::runtime_error("reference clearance reserve must be finite and nonnegative");
            config.model.reference_clearance_reserve_m = static_cast<float>(reserve);
        }
        if (argc >= 50) {
            const double tolerance = number_arg(argv, 49, .005);
            if (!std::isfinite(tolerance) || tolerance < 0. || tolerance > .02)
                throw std::runtime_error("invalid nonlinear corridor tolerance");
            config.model.nonlinear_corridor_tolerance_m = static_cast<float>(tolerance);
        }
        config.enable_nonlinear_line_search = line_search_backtracks > 0;
        config.line_search_max_backtracks = line_search_backtracks;
        /* This executable is an offline fixed-state evaluator; it has no
         * bridge command epoch. Live code leaves the corresponding gate off. */
        config.line_search_command_epoch_valid = 1;
        const char *trace_row_text = std::getenv("MPC_REPLAY_TRACE_ROW");
        const long trace_row = trace_row_text
            ? std::strtol(trace_row_text, nullptr, 10) : -1;
        const char *diagnostic_row_text = std::getenv("MPC_REPLAY_DIAG_ROW");
        const long diagnostic_row = diagnostic_row_text
            ? std::strtol(diagnostic_row_text, nullptr, 10) : -1;
        const char *gradient_row_text=std::getenv("MPC_REPLAY_GRADIENT_AUDIT_ROW");
        const long gradient_row=gradient_row_text?std::strtol(gradient_row_text,nullptr,10):-1;
        const char *qp_capture_dir_text =
            std::getenv("MPC_REPLAY_QP_CAPTURE_DIR");
        const char *qp_capture_row_text =
            std::getenv("MPC_REPLAY_QP_CAPTURE_ROW");
        const char *qp_capture_rows_text =
            std::getenv("MPC_REPLAY_QP_CAPTURE_ROWS");
        const long qp_capture_row = qp_capture_row_text
            ? std::strtol(qp_capture_row_text, nullptr, 10) : -1;
        std::set<long> qp_capture_rows;
        if (qp_capture_rows_text && qp_capture_rows_text[0] != '\0') {
            std::stringstream rows_stream(qp_capture_rows_text);
            std::string row_token;
            while (std::getline(rows_stream, row_token, ',')) {
                if (row_token.empty())
                    throw std::runtime_error(
                        "empty row in MPC_REPLAY_QP_CAPTURE_ROWS");
                std::size_t parsed = 0;
                const long row = std::stol(row_token, &parsed);
                if (parsed != row_token.size() || row < 0 ||
                    !qp_capture_rows.insert(row).second)
                    throw std::runtime_error(
                        "invalid or duplicate MPC_REPLAY_QP_CAPTURE_ROWS row");
            }
        }
        const bool qp_capture_enabled = qp_capture_dir_text &&
            qp_capture_dir_text[0] != '\0';
        const std::filesystem::path qp_capture_dir = qp_capture_enabled
            ? std::filesystem::path(qp_capture_dir_text)
            : std::filesystem::path{};
        if (qp_capture_enabled)
            std::filesystem::create_directories(qp_capture_dir);
        riccati_debug_set_trace_enabled(trace_row >= 0);
        std::cout << "stamp_ns,status,admm_status,time_budget_exhausted,iterations,"
                     "sqp_iterations_attempted,sqp_iterations_completed,"
                     "sqp_iterations_accepted,refinement_prefix_selected_pass,"
                     "refinement_full_horizon_selected_pass,"
                     "refinement_terminal_probe_count,"
                     "refinement_terminal_selected_pass,"
                     "refinement_terminal_probe_us,"
                     "refinement_nominal_build_us,"
                     "refinement_first_prefix_objective,"
                     "refinement_second_prefix_objective,"
                     "refinement_prefix_validation_us,primal,dual,"
                     "rho_changes,factorizations,"
                     "input_u_mps,input_v_mps,input_r_radps,"
                     "input_abs_steering_rad,input_delayed_throttle,"
                     "reference_speed_mps,reference_curvature_per_m,"
                     "solve_us,cycle_us,nominal_build_us,reference_seed_us,"
                     "cold_start_queue_seed_us,rollout_a_us,"
                     "progress_reference_resampling_us,"
                     "qp_build_us,jacobian_wall_us,jacobian_thread_cpu_us,"
                     "jacobian_slowest_stage_us,jacobian_stages_completed,"
                     "jacobian_threads_used,jacobian_source_sensitivity_calls,"
                     "jacobian_finite_difference_columns,admm_us,"
                     "solver_cold_start_initialization_ns,"
                     "riccati_factorization_ns,riccati_rhs_ns,"
                     "admm_projection_ns,rho_update_ns,scaling_transform_ns,"
                     "output_unscale_ns,rollout_us,"
                     "nonlinear_validation_us,min_corridor_slack_m,"
                     "first_pass_objective,second_pass_objective,exact_objective_improvement,source_directional_derivative,source_curvature_steering,source_curvature_throttle,nonlinear_objective,"
                     "nonlinear_failure_reason,nonlinear_failure_stage,"
                     "line_search_attempted,line_search_trials,"
                     "line_search_accepted,line_search_skipped_budget,"
                     "line_search_merit_triggered,"
                     "line_search_nominal_exact_valid,"
                     "line_search_nominal_objective,"
                     "line_search_skipped_command_epoch,"
                     "line_search_full_step_failure_reason,"
                     "line_search_full_step_failure_stage,"
                     "line_search_last_failure_reason,"
                     "line_search_last_failure_stage,"
                     "line_search_selected_alpha,line_search_us,"
                     "first_steering,first_throttle,"
                     "max_plan_steering_delta,max_plan_throttle_delta,"
                     "frenet_geometry_closure_max_m,frenet_geometry_closure_max_rad,"
                     "predicted_cross_track_rms_m,predicted_heading_rms_rad,"
                     "predicted_speed_rms_mps,predicted_terminal_cross_track_m,"
                     "nominal_seed_injected\n";

        std::string line;
        std::size_t row_index = 0;
        MpcRtiMemory_t memory{};
        mpc_rti_memory_reset(&memory);
        while (std::getline(input, line)) {
            if (line.empty() || line.front() == '#') continue;
            const auto row = parse_csv_numbers(line);
            if (row.size() != kInputColumnCount)
                throw std::runtime_error("input row has unexpected column count");

            MpcRtiState_t state{};
            state.plant.e_y = static_cast<float>(row[3]);
            state.plant.e_psi = static_cast<float>(row[4]);
            state.plant.u = static_cast<float>(row[5]);
            state.plant.v = static_cast<float>(row[6]);
            state.plant.r = static_cast<float>(row[7]);
            state.plant.actual_steering_angle = static_cast<float>(row[8]);
            state.plant.delayed_steering_command_1 = static_cast<float>(row[9]);
            state.plant.delayed_steering_command_2 = static_cast<float>(row[10]);
            state.plant.delayed_throttle_command_1 = static_cast<float>(row[11]);
            state.plant.delayed_throttle_command_2 = static_cast<float>(row[12]);
            state.plant.source_suspension_history_initialized =
                row[13] != 0.0 ? 1 : 0;
            state.plant.source_plant_initialized = 1;
            state.plant.source_imu_state_initialized = 1;
            /* Columns 14-15 retain the exported bag schema's optional yaw
             * innovation fields. The production state no longer contains or
             * applies that correction, so replay intentionally ignores them. */
            for (int i = 0; i < MPC_PLANT_INTERNAL_STATE_COUNT; ++i)
                state.plant.source_plant_internal_state[i] = row[16 + i];

            MpcTrajectorySample_t operating_reference{};
            const bool operating_reference_valid = mpc_trajectory_sample(
                trajectory.data(), trajectory.size(), lap_length, row[1],
                &operating_reference) != 0;

            config.model.active_speed_ceiling_mps = static_cast<float>(row[2]);
            const bool capture_this_qp = qp_capture_enabled &&
                ((qp_capture_row < 0 && qp_capture_rows.empty()) ||
                 qp_capture_row == static_cast<long>(row_index) ||
                 qp_capture_rows.count(static_cast<long>(row_index)) != 0);
            if (capture_this_qp) {
                g_qp_capture = MpcRtiQpCapture_t{};
                config.qp_capture = &g_qp_capture;
            } else {
                config.qp_capture = nullptr;
            }
            MpcRtiCycleResult_t result{};
            if (memory_mode == "independent") {
                mpc_rti_memory_reset(&memory);
            } else if (memory_mode == "nominal-cold") {
                /* Keep the solver's ADMM iterate, but discard only the
                 * previous nonlinear nominal. This isolates whether a
                 * shifted trajectory—not the QP warm state—is the source of
                 * sequence-replay convergence failures. */
                memory.nominal.valid = 0;
            } else if (memory_mode == "reference-seed") {
                /* Offline counterfactual for PR-C: preserve ADMM z/y/rho
                 * history but rebuild the next physical nominal from the
                 * trajectory/source-plant initializer, not the previous
                 * rejected QP suffix. The current sensor state still carries
                 * the immutable in-flight actuator queue. This is a replay
                 * offline analysis only; it is not a publishable recovery action
                 * and does not reconstruct an unavailable certified suffix. */
                memory.nominal.valid = 0;
                memory.warm_start_controls_valid = 0;
                memory.warm_start_control_count = 0;
            }
            if (solver_start == "dual-cold") {
                /* Hybrid ADMM seed: retain shifted primal
                 * projections (z), adapted rho values, and nominal/command
                 * history, but discard dual multipliers (y). This isolates
                 * whether stale duals hurt difficult corridor states without
                 * discarding the current command-consistent warm start.
                 * Offline replay only; production remains warm. */
                for (int stage = 0; stage <= RICCATI_HORIZON; ++stage) {
                    for (int state_index = 0;
                         state_index < RICCATI_MAX_NX; ++state_index) {
                        memory.solver_state.y_x[stage][state_index] = 0.0f;
                    }
                }
                for (int stage = 0; stage < RICCATI_HORIZON; ++stage) {
                    for (int input_index = 0;
                         input_index < RICCATI_MAX_NU; ++input_index) {
                        memory.solver_state.y_u[stage][input_index] = 0.0f;
                    }
                }
            }
            const auto seed_it = nominal_seed_controls.find(row_index);
            if (!nominal_seed_controls.empty() &&
                seed_it == nominal_seed_controls.end())
                throw std::runtime_error(
                    "nominal seed file has no entry for replay row " +
                    std::to_string(row_index));
            const bool nominal_seed_injected =
                seed_it != nominal_seed_controls.end();
            if (nominal_seed_injected) {
                /* Offline source-plant continuation analysis. The
                 * seed file stores 30 actions already checked from this
                 * saved state by the source-plant audit. The warm-start
                 * contract shifts out element zero (the action in flight),
                 * so duplicate seed[0] in slot zero and put seed[0..28] in
                 * slots 1..29. The final nominal action is held at seed[28],
                 * matching normal horizon-shift semantics. Production never
                 * reads this environment variable. */
                memory.warm_start_controls[0] = seed_it->second[0];
                for (int stage = 1; stage < PREDICTION_HORIZON; ++stage)
                    memory.warm_start_controls[stage] =
                        seed_it->second[stage - 1];
                memory.warm_start_control_count = PREDICTION_HORIZON;
                memory.warm_start_controls_valid = 1;
                consumed_nominal_seed_rows.insert(row_index);
            }
            // Offline-only independent same-state source plant comparison.
            // This exact production C rollout is necessary to verify the
            // Python source-audit's claimed geometric feasibility before
            // reasoning about rescue search order or objective weights.
            if(nominal_seed_injected &&
               std::getenv("MPC_REPLAY_VALIDATE_NOMINAL_SEED")) {
                MpcRtiNominal_t validated{};
                int failing_stage=-1;
                const auto exact=mpc_rti_validate_candidate_plan(
                    &state,row[1],seed_it->second.data(),
                    trajectory.data(),trajectory.size(),lap_length,
                    PREDICTION_HORIZON,PREDICTION_DT_SECONDS,
                    &config,&validated,&failing_stage);
                std::cerr<<"SOURCE_VALIDATION row="<<row_index
                    <<" status="<<static_cast<int>(exact)
                    <<" failure_stage="<<failing_stage
                    <<" min_slack="<<validated.validation_minimum_corridor_slack_m
                    <<" terminal_u="<<validated.states[PREDICTION_HORIZON].plant.u
                    <<" terminal_epsi="<<validated.states[PREDICTION_HORIZON].plant.e_psi
                    <<" terminal_v="<<validated.states[PREDICTION_HORIZON].plant.v
                    <<'\n';
            }
            if(static_cast<long>(row_index)==gradient_row)
                audit_objective_gradient(state,row[1],trajectory,lap_length,config,memory);
            const MpcRtiCycleStatus_t status = mpc_rti_solve_cycle(
                &state, row[1], trajectory.data(), trajectory.size(),
                lap_length, PREDICTION_DT_SECONDS, PREDICTION_HORIZON,
                &config, &memory, &result);
            if(static_cast<long>(row_index)==gradient_row)
                audit_candidate_merit(state,row[1],trajectory,lap_length,config,result,
                    g_qp_capture.valid ? &g_qp_capture : nullptr);
            if (capture_this_qp && g_qp_capture.valid) {
                char filename[32];
                std::snprintf(filename, sizeof(filename), "qp_%06zu.fqp",
                              row_index);
                const auto path = qp_capture_dir / filename;
                if (!sdu_apex::qp_fixture::write_capture(
                        path.string(), g_qp_capture))
                    throw std::runtime_error(
                        "failed to write frozen QP fixture: " + path.string());
                std::cerr << "QP_CAPTURE row=" << row_index
                          << " passes=" << g_qp_capture.capture_count
                          << " path=" << path.string() << '\n';
            }
            float max_plan_steering_delta = 0.0f;
            float max_plan_throttle_delta = 0.0f;
            double geometry_closure_m = NAN;
            double geometry_closure_rad = NAN;
            double predicted_ey_rms=NAN,predicted_heading_rms=NAN,
                predicted_speed_rms=NAN,predicted_terminal_ey=NAN;
            if ((status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
                 status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
                 status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH) &&
                memory.nominal.valid && memory.nominal.horizon > 0) {
                const int horizon=memory.nominal.horizon;
                double ey_sum=0.,heading_sum=0.,speed_sum=0.;
                for (int k=1;k<=horizon;++k) {
                    const auto &predicted=memory.nominal.states[k].plant;
                    const auto &reference=result.comparison_objective.references[k];
                    const double ey=predicted.e_y-reference.e_y;
                    const double heading=predicted.e_psi-reference.e_psi;
                    const double speed=predicted.u-reference.u;
                    ey_sum+=ey*ey;heading_sum+=heading*heading;speed_sum+=speed*speed;
                }
                predicted_ey_rms=std::sqrt(ey_sum/horizon);
                predicted_heading_rms=std::sqrt(heading_sum/horizon);
                predicted_speed_rms=std::sqrt(speed_sum/horizon);
                predicted_terminal_ey=memory.nominal.states[horizon].plant.e_y-
                    result.comparison_objective.references[horizon].e_y;
                float previous_steering = state.plant.delayed_steering_command_2;
                float previous_throttle = state.plant.delayed_throttle_command_2;
                for (int k = 0; k < memory.nominal.horizon; ++k) {
                    const MpcModelControl_t &control = memory.nominal.controls[k];
                    max_plan_steering_delta = std::max(
                        max_plan_steering_delta,
                        std::abs(control.steering_target_rad - previous_steering));
                    max_plan_throttle_delta = std::max(
                        max_plan_throttle_delta,
                        std::abs(control.throttle_command - previous_throttle));
                    previous_steering = control.steering_target_rad;
                    previous_throttle = control.throttle_command;
                }
                MpcTrajectorySample_t anchor{};
                if (!mpc_trajectory_sample(trajectory.data(), trajectory.size(),
                        lap_length, memory.nominal.progress[0], &anchor))
                    throw std::runtime_error("cannot anchor geometry closure");
                const auto &initial = memory.nominal.states[0].plant;
                double world_x = anchor.x - std::sin(anchor.heading) * initial.e_y;
                double world_y = anchor.y + std::cos(anchor.heading) * initial.e_y;
                double world_yaw = anchor.heading + initial.e_psi;
                geometry_closure_m = geometry_closure_rad = 0.0;
                for (int k = 1; k <= memory.nominal.horizon; ++k) {
                    const auto &next = memory.nominal.states[k];
                    const double c = std::cos(world_yaw), s = std::sin(world_yaw);
                    world_x += c * next.incoming_body_forward_delta_m -
                        s * next.incoming_body_lateral_delta_m;
                    world_y += s * next.incoming_body_forward_delta_m +
                        c * next.incoming_body_lateral_delta_m;
                    world_yaw += next.incoming_body_yaw_delta_rad;
                    MpcTrajectorySample_t reference{};
                    if (!mpc_trajectory_sample(trajectory.data(), trajectory.size(),
                            lap_length, memory.nominal.progress[k], &reference))
                        throw std::runtime_error("cannot sample geometry closure");
                    const double path_x = reference.x -
                        std::sin(reference.heading) * next.plant.e_y;
                    const double path_y = reference.y +
                        std::cos(reference.heading) * next.plant.e_y;
                    const double yaw_error = world_yaw -
                        reference.heading - next.plant.e_psi;
                    geometry_closure_m = std::max(geometry_closure_m,
                        std::hypot(world_x - path_x, world_y - path_y));
                    geometry_closure_rad = std::max(geometry_closure_rad,
                        std::abs(std::atan2(std::sin(yaw_error),
                            std::cos(yaw_error))));
                }
            }
            if (static_cast<long>(row_index) == trace_row) {
                const int trace_count = riccati_debug_get_trace_count();
                for (int i = 0; i < trace_count; ++i) {
                    RiccatiDebugIterSample_t sample{};
                    if (riccati_debug_get_trace_sample(i, &sample) == 0) {
                        std::cerr << "TRACE row=" << row_index
                                  << " iter=" << sample.iter
                                  << " p=" << sample.primal_residual
                                  << " d=" << sample.dual_residual
                                  << " px=" << sample.state_primal_residual
                                  << " dx=" << sample.state_dual_residual
                                  << " pu=" << sample.ctrl_primal_residual
                                  << " du=" << sample.ctrl_dual_residual
                                  << " rho=" << sample.rho
                                  << " rho_u=" << sample.rho_u
                                  << " u0=" << sample.u0_steer << ','
                                  << sample.u0_throttle
                                  << " z0=" << sample.z0_steer << ','
                                  << sample.z0_throttle
                                  << " rho_scale=" << sample.scale_rho << ','
                                  << sample.scale_rho_u << '\n';
                    }
                }
            }
            if (static_cast<long>(row_index) == diagnostic_row) {
                std::cerr << "CYCLE row=" << row_index
                          << " status=" << status
                          << " admm=" << result.admm_status
                          << " iterations=" << result.solver_iterations
                          << " nonlinear_failure="
                          << result.nonlinear_failure_reason
                          << " stage=" << result.nonlinear_failure_stage
                          << " state={ey:" << state.plant.e_y
                          << ",e_psi:" << state.plant.e_psi
                          << ",u:" << state.plant.u
                          << ",r:" << state.plant.r << "}\n";
                std::cerr << "NOMINAL_FIRST row=" << row_index
                          << " steering="
                          << result.nominal_first_control.steering_target_rad
                          << " throttle="
                          << result.nominal_first_control.throttle_command
                          << " progress=" << row[1]
                          << " curvature=" << row[2] << '\n';
                const auto &failure = result.nonlinear_failure;
                std::cerr << "NONLINEAR_FAILURE valid=" << failure.valid
                          << " stage=" << failure.stage
                          << " ey=" << failure.state.plant.e_y
                          << " epsi=" << failure.state.plant.e_psi
                          << " lower=" << failure.lower_bound_m
                          << " upper=" << failure.upper_bound_m << '\n';
                if (result.candidate_control_count > 0) {
                    std::cerr << "REJECTED_CANDIDATE_CONTROLS row="
                              << row_index;
                    for (int k = 0;
                         k < result.candidate_control_count; ++k) {
                        const MpcModelControl_t &control =
                            result.candidate_controls[k];
                        std::cerr << " [k=" << k << ",delta="
                                  << control.steering_target_rad
                                  << ",q=" << control.throttle_command << ']';
                    }
                    std::cerr << '\n';
                }
                if ((status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
                     status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
                     status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH) &&
                    memory.nominal.horizon > 0) {
                    const int horizon = memory.nominal.horizon;
                    const int sample_stages[] = {
                        0, 1, 2, 3, 4, horizon / 3,
                        (2 * horizon) / 3, horizon - 1};
                    std::cerr << "PLAN_SAMPLES row=" << row_index;
                    int previous_stage = -1;
                    for (const int stage : sample_stages) {
                        if (stage < 0 || stage >= horizon ||
                            stage == previous_stage) continue;
                        previous_stage = stage;
                        const MpcModelControl_t &control =
                            memory.nominal.controls[stage];
                        const float predicted_u =
                            memory.nominal.states[stage + 1].plant.u;
                        std::cerr << " [k=" << stage << ",delta="
                                  << control.steering_target_rad
                                  << ',' << control.throttle_command << ']';
                        std::cerr << " u=" << predicted_u;
                    }
                    std::cerr << " terminal_u="
                              << memory.nominal.states[horizon].plant.u;
                    std::cerr << '\n';
                    std::cerr << "PLAN_CONTROLS row=" << row_index;
                    for (int stage = 0; stage < horizon; ++stage) {
                        const MpcModelControl_t &control =
                            memory.nominal.controls[stage];
                        std::cerr << " [k=" << stage << ':'
                                  << control.steering_target_rad << ','
                                  << control.throttle_command << ']';
                    }
                    std::cerr << '\n';
                }
            }
            if (status == MPC_RTI_CYCLE_REJECTED_INPUT &&
                result.input_failure_reason != MPC_RTI_INPUT_FAILURE_NONE)
                std::cerr << "input row " << row_index << " rejected before solve: "
                          << result.input_failure_reason << '\n';
            std::cout.precision(9);
            std::cout << static_cast<std::uint64_t>(row[0]) << ',' << status << ','
                      << result.admm_status << ','
                      << result.solver_time_budget_exhausted << ','
                      << result.solver_iterations << ','
                      << result.sqp_iterations_attempted << ','
                      << result.sqp_iterations_completed << ','
                      << result.sqp_iterations_accepted << ','
                      << result.refinement_prefix_selected_pass << ','
                      << result.refinement_full_horizon_selected_pass << ','
                      << result.refinement_terminal_probe_count << ','
                      << result.refinement_terminal_selected_pass << ','
                      << result.refinement_terminal_probe_us << ','
                      << result.refinement_nominal_build_us << ','
                      << result.refinement_first_prefix_objective << ','
                      << result.refinement_second_prefix_objective << ','
                      << result.refinement_prefix_validation_us << ','
                      << result.primal_residual << ',' << result.dual_residual << ','
                      << result.rho_change_count << ','
                      << result.factorization_count << ','
                      << state.plant.u << ',' << state.plant.v << ','
                      << state.plant.r << ','
                      << std::abs(state.plant.actual_steering_angle) << ','
                      << state.plant.delayed_throttle_command_2 << ','
                      << (operating_reference_valid
                          ? operating_reference.speed
                          : std::numeric_limits<double>::quiet_NaN()) << ','
                      << (operating_reference_valid
                          ? operating_reference.curvature
                          : std::numeric_limits<double>::quiet_NaN()) << ','
                      << result.total_solve_us << ',' << result.total_cycle_us << ','
                      << result.nominal_build_us << ',' << result.reference_seed_us << ','
                      << result.cold_start_queue_seed_us << ','
                      << result.nominal_rollout_a_us << ','
                      << result.progress_reference_resampling_us << ','
                      << result.qp_build_us << ',' << result.jacobian_wall_us << ','
                      << result.jacobian_thread_cpu_us << ','
                      << result.jacobian_slowest_stage_us << ','
                      << result.jacobian_stages_completed << ','
                      << result.jacobian_threads_used << ','
                      << result.jacobian_source_sensitivity_calls << ','
                      << result.jacobian_finite_difference_columns << ','
                      << result.admm_us << ','
                      << result.solver_cold_start_initialization_time_ns << ','
                      << result.factorization_time_ns << ','
                      << result.riccati_rhs_time_ns << ','
                      << result.admm_projection_time_ns << ','
                      << result.rho_update_time_ns << ','
                      << result.solver_scaling_time_ns << ','
                      << result.solver_output_unscale_time_ns << ','
                      << result.exact_rollout_us << ','
                      << result.nonlinear_validation_us << ','
                      << result.minimum_predicted_corridor_slack_m << ','
                      << result.first_pass_objective << ','
                      << result.second_pass_objective << ','
                      << result.exact_objective_improvement << ','
                      << result.source_directional_derivative << ','
                      << result.source_curvature_regularization[0] << ','
                      << result.source_curvature_regularization[1] << ','
                      << result.nonlinear_objective << ','
                      << result.nonlinear_failure_reason << ','
                      << result.nonlinear_failure_stage << ','
                      << result.line_search_attempted << ','
                      << result.line_search_trials << ','
                      << result.line_search_accepted << ','
                      << result.line_search_skipped_budget << ','
                      << result.line_search_merit_triggered << ','
                      << result.line_search_nominal_exact_valid << ','
                      << result.line_search_nominal_objective << ','
                      << result.line_search_skipped_command_epoch << ','
                      << result.line_search_full_step_failure_reason << ','
                      << result.line_search_full_step_failure_stage << ','
                      << result.line_search_last_failure_reason << ','
                      << result.line_search_last_failure_stage << ','
                      << result.line_search_selected_alpha << ','
                      << result.line_search_us << ','
                      << result.first_control.steering_target_rad << ','
                      << result.first_control.throttle_command << ','
                      << max_plan_steering_delta << ','
                      << max_plan_throttle_delta << ','
                      << geometry_closure_m << ','
                      << geometry_closure_rad << ','
                      << predicted_ey_rms << ',' << predicted_heading_rms << ','
                      << predicted_speed_rms << ',' << predicted_terminal_ey << ','
                      << (nominal_seed_injected ? 1 : 0) << '\n';
            ++row_index;
        }
        if (consumed_nominal_seed_rows.size() !=
            nominal_seed_controls.size())
            throw std::runtime_error(
                "nominal seed file contains rows outside the replay input");
        return row_index == 0 ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << "mpc_bag_solver_replay: " << error.what() << '\n';
        return 2;
    }
}
