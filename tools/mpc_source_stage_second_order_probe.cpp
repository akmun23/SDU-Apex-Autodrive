// Development-only full-source directional second-order rollout profiler.
// Input is a captured exact-source MPC fixture and a REAL first-QP
// normalized 60-control direction. No ROS publication or fallback path.
#include "mpc_qp_failure_context.hpp"
#include "mpc_rti.h"
#include "vehicle_model.h"
#include "unity_source_plant_c.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <map>
#include <tuple>
#include "unity_source_plant.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif

using json=nlohmann::json;
namespace fs=std::filesystem;
namespace {
constexpr int N=PREDICTION_HORIZON;
constexpr int X=UNITY_SOURCE_PLANT_STATE_SIZE;
constexpr int L=UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;
double us(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-t).count();
}

// Offline 1 ms branch trace of the SAME 26-state source plant. Do not
// instrument or alter the live plant, SQP, or contact solver.
constexpr double kCom=.15532;
struct MicroTrace {
    std::array<UnitySourcePlantOutputC,25> values{};
    std::array<double,25> incoming_u{};
    std::array<double,25> incoming_steering{};
    double target_steering{};
    UnitySourcePlantOutputC all25{};
    bool split_equal{};
    double max_split_state_error{};
};
bool trace_source_microsteps(const MpcModelState_t &plant,
        const MpcModelControl_t &control,
        const double dx[X],const double du[2],double epsilon,
        MicroTrace *trace)
{
    if(!trace || !plant.source_plant_initialized)return false;
    UnitySourcePlantInputC input{};
    input.state[UNITY_SOURCE_U]=plant.u;
    input.state[UNITY_SOURCE_V]=plant.v+plant.r*kCom;
    input.state[UNITY_SOURCE_YAW_RATE]=plant.r;
    input.state[UNITY_SOURCE_PHYSICAL_STEERING]=
        plant.actual_steering_angle;
    for(int i=0;i<MPC_PLANT_INTERNAL_STATE_COUNT;++i) {
        const int coordinate=i<10?3+i:i+4;
        input.state[coordinate]=plant.source_plant_internal_state[i];
    }
    for(int i=0;i<X;++i)input.state[i]+=epsilon*dx[i];
    input.steering_target_rad=double(control.steering_target_rad)+
        epsilon*du[0];
    input.throttle_normalized=double(control.throttle_command)+
        epsilon*du[1];
    input.suspension_history_initialized=
        plant.source_suspension_history_initialized;
    if(std::abs(input.steering_target_rad)>
           UNITY_SOURCE_MAX_STEERING_RAD+1.e-7||
       input.throttle_normalized<.001-1.e-7||
       input.throttle_normalized>1.+1.e-7)return false;
    // Match the physical float-to-double steering boundary exactly.
    input.steering_target_rad=std::clamp(input.steering_target_rad,
        -UNITY_SOURCE_MAX_STEERING_RAD,
        UNITY_SOURCE_MAX_STEERING_RAD);
    trace->target_steering=input.steering_target_rad;
    UnitySourcePlantInputC tick=input;
    for(int k=0;k<25;++k) {
        trace->incoming_u[k]=tick.state[UNITY_SOURCE_U];
        trace->incoming_steering[k]=
            tick.state[UNITY_SOURCE_PHYSICAL_STEERING];
        if(!unity_source_plant_step_c(&tick,.001,
                                     &trace->values[k]))return false;
        const auto &out=trace->values[k];
        std::copy_n(out.state,X,tick.state);
        std::copy_n(out.wheel_rotation_angle_rad,4,
                    tick.wheel_rotation_angle_rad);
        tick.suspension_history_initialized=
            out.suspension_history_initialized;
    }
    if(!unity_source_plant_step_c(&input,.025,&trace->all25))
        return false;
    double worst=0.;
    for(int i=0;i<X;++i)
        worst=std::max(worst,std::abs(
           trace->values[24].state[i]-trace->all25.state[i]));
    trace->max_split_state_error=worst;
    trace->split_equal=worst<1.e-8;
    return true;
}

f1tenth_mpc::UnitySourcePlantState as_full_source_state(
    const MpcModelState_t &state,const double dx[X],double epsilon)
{
    double x[X]{};
    x[UNITY_SOURCE_U]=state.u;
    x[UNITY_SOURCE_V]=state.v+kCom*state.r;
    x[UNITY_SOURCE_YAW_RATE]=state.r;
    x[UNITY_SOURCE_PHYSICAL_STEERING]=state.actual_steering_angle;
    for(int i=0;i<MPC_PLANT_INTERNAL_STATE_COUNT;++i) {
        const int coord=i<10?i+3:i+4;
        x[coord]=state.source_plant_internal_state[i];
    }
    for(int i=0;i<X;++i)x[i]+=epsilon*dx[i];
    f1tenth_mpc::UnitySourcePlantState result{};
    result.u_mps=x[UNITY_SOURCE_U];
    result.v_mps=x[UNITY_SOURCE_V];
    result.yaw_rate_radps=x[UNITY_SOURCE_YAW_RATE];
    result.physical_steering_rad=x[UNITY_SOURCE_PHYSICAL_STEERING];
    result.body_height_m=x[UNITY_SOURCE_BODY_HEIGHT];
    result.body_height_rate_mps=x[UNITY_SOURCE_BODY_HEIGHT_RATE];
    result.roll_rad=x[UNITY_SOURCE_ROLL];
    result.roll_rate_radps=x[UNITY_SOURCE_ROLL_RATE];
    result.pitch_rad=x[UNITY_SOURCE_PITCH];
    result.pitch_rate_radps=x[UNITY_SOURCE_PITCH_RATE];
    result.suspension_history_initialized=
        state.source_suspension_history_initialized!=0;
    for(int w=0;w<4;++w) {
        result.wheel_omega_radps[w]=x[UNITY_SOURCE_WHEEL_OMEGA_FL+w];
        result.suspension_jounce_m[w]=x[
            UNITY_SOURCE_SUSPENSION_JOUNCE_FL+w];
        result.low_forward_speed_time_s[w]=x[
            UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL+w];
        result.low_side_speed_time_s[w]=x[
            UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL+w];
    }
    return result;
}
json exact_inner_substep_branch_comparison(
    const MpcModelState_t &state,
    const MpcModelControl_t &control,
    const double dx[X],const double du[2],double epsilon)
{
    using Event=f1tenth_mpc::UnitySourceBranchObservation;
    using Key=std::tuple<int,int,int>;
    using Index=std::map<Key,Event>;
    std::array<Index,3> index{};
    std::array<std::size_t,3> counts{};
    for(int pass=0;pass<3;++pass) {
        const double sign=pass==0?0.:(pass==1?1.:-1.);
        const auto physical=as_full_source_state(
            state,dx,sign*epsilon);
        f1tenth_mpc::UnitySourcePlantInput cmd{
            std::clamp(double(control.steering_target_rad)+
                         sign*epsilon*du[0],
                 -UNITY_SOURCE_MAX_STEERING_RAD,
                  UNITY_SOURCE_MAX_STEERING_RAD),
            double(control.throttle_command)+sign*epsilon*du[1]};
        std::vector<Event> observations;
        if(!f1tenth_mpc::unity_source_plant_trace_branches(
             physical,cmd,.025,&observations))
             throw std::runtime_error("source internal substep trace failed");
        counts[pass]=observations.size();
        for(const auto &o:observations)
            index[pass].emplace(Key(o.fixed_tick,o.vehicle_substep,o.wheel),o);
    }
    json result;
    result["nominal_internal_wheel_evaluations"]=counts[0];
    result["positive_internal_wheel_evaluations"]=counts[1];
    result["negative_internal_wheel_evaluations"]=counts[2];
    result["changes"]=json::array();
    result["switch_counts"]=json::object();
    int earliest=-1,unmatched=0;
    for(const auto &[key,base]:index[0]) {
        const auto found_plus=index[1].find(key);
        const auto found_minus=index[2].find(key);
        if(found_plus==index[1].end()||
           found_minus==index[2].end()) {++unmatched;continue;}
        const auto &plus=found_plus->second;
        const auto &minus=found_minus->second;
        const auto record=[&](const char *name,
                int n,int p,int m) {
            if(n==p&&n==m)return;
            const int tick=std::get<0>(key);
            if(earliest<0 || tick<earliest)earliest=tick;
            auto &counter=result["switch_counts"][name];
            if(counter.is_null())counter=0;
            counter=counter.get<int>()+1;
            if(result["changes"].size()<60) {
                result["changes"].push_back({
                    {"name",name},{"tick_ms",tick},
                    {"substep",std::get<1>(key)},
                    {"substeps_per_ms",base.vehicle_substep_count},
                    {"wheel",std::get<2>(key)},
                    {"nominal",n},{"positive",p},{"negative",m},
                    {"nominal_slip_x",base.longitudinal_slip},
                    {"nominal_slip_y",base.lateral_slip},
                    {"nominal_combined_utilization",
                     base.combined_slip_utilization}
                });
            }
        };
        record("grounded",base.grounded,plus.grounded,minus.grounded);
        record("sticky_forward",base.sticky_forward,
               plus.sticky_forward,minus.sticky_forward);
        record("sticky_side",base.sticky_side,
               plus.sticky_side,minus.sticky_side);
        record("longitudinal_curve_region",
               base.longitudinal_curve_region,
               plus.longitudinal_curve_region,
               minus.longitudinal_curve_region);
        record("lateral_curve_region",
               base.lateral_curve_region,
               plus.lateral_curve_region,
               minus.lateral_curve_region);
        record("tire_load_filter_region",
               base.tire_load_filter_region,
               plus.tire_load_filter_region,
               minus.tire_load_filter_region);
        record("combined_slip_utilization",
               base.combined_slip_saturated,
               plus.combined_slip_saturated,
               minus.combined_slip_saturated);
    }
    result["earliest_changed_fixed_tick_ms"]=earliest;
    result["unmatched_internal_evaluations"]=unmatched;
    return result;
}

int tire_regime(double slip,int is_lateral) {
    const double value=std::abs(slip);
    const double peak=is_lateral?.01:.15;
    const double asymptote=is_lateral?.1:.25;
    return value<peak?0:(value<asymptote?1:2);
}
double curve_response_ratio(double slip,bool lateral)
{
    const double peak=lateral?.01:.15;
    const double asym=lateral?.1:.25;
    const double asym_response=lateral?.5:.58;
    const double peak_response=lateral?1.:.9;
    const double magnitude=std::abs(slip);
    double response=0.;
    if(magnitude<peak) {
        const double t=magnitude/peak;
        response=peak_response*(t+t*t-t*t*t);
    } else if(magnitude<asym) {
        const double t=(magnitude-peak)/(asym-peak);
        response=peak_response+(asym_response-peak_response)*
            (3.*t*t-2.*t*t*t);
    } else response=asym_response;
    return std::copysign(response/peak_response,slip);
}
int load_filter_regime(double unfiltered_force,int wheel)
{
    constexpr double sprung[4]={.817194700124942,.8160186332083913,
                                 .9189924292167562,.9177942374499102};
    const double rest=(sprung[wheel]+.109)*9.81;
    const double normalized=unfiltered_force/rest;
    return normalized<=0.?0:(normalized>=3.?2:1);
}

void append_first_change(json &report,const char *category,
        int tick,int wheel,double nominal,double positive,double negative)
{
    json entry={{"event",category},{"tick_ms",tick},
                {"wheel",wheel},{"nominal",nominal},
                {"positive",positive},{"negative",negative}};
    report["first_switches"].push_back(entry);
}
json compare_microbranches(const MicroTrace &nominal,
        const MicroTrace &positive,const MicroTrace &negative)
{
    json report;
    report["one_ms_matches_25ms_source"]=
        nominal.split_equal&&positive.split_equal&&negative.split_equal;
    report["max_split_state_error"]=std::max({
        nominal.max_split_state_error,
        positive.max_split_state_error,negative.max_split_state_error});
    report["first_switches"]=json::array();
    report["switch_counts"]=json::object();
    report["largest_normalized_yaw_rate_departure"]=0.;
    report["largest_lat_force_departure_N"]=0.;
    report["largest_long_force_departure_N"]=0.;
    report["largest_suspension_force_departure_N"]=0.;
    report["largest_slip_y_departure_rad"]=0.;
    int first=-1;
    const auto record=[&](const char *name,int tick,int wheel,
                          double n,double p,double m) {
        if(p==n&&m==n)return;
        if(first<0)first=tick;
        auto &count=report["switch_counts"][name];
        if(count.is_null())count=0;
        count=count.get<int>()+1;
        if(report["first_switches"].size()<35)
            append_first_change(report,name,tick,wheel,n,p,m);
    };
    for(int tick=0;tick<25;++tick) {
        const double speeds[3]={
            nominal.incoming_u[tick],
            positive.incoming_u[tick],
            negative.incoming_u[tick]};
        record("wheel_substeps_5mps",tick,-1,
            std::abs(speeds[0])<5?5:1,
            std::abs(speeds[1])<5?5:1,
            std::abs(speeds[2])<5?5:1);
        const auto &nc=nominal.values[tick];
        const auto &pc=positive.values[tick];
        const auto &mc=negative.values[tick];
        // Source steering is rate limited with an exact snap at the
        // target. Detect a change in that physical actuator branch too.
        const auto steer_snap=[](const MicroTrace &t,int k) {
            return std::abs(t.values[k].
                state[UNITY_SOURCE_PHYSICAL_STEERING]-
                t.target_steering)<1.e-7;
        };
        record("steering_slew_snap",tick,-1,
            steer_snap(nominal,tick),steer_snap(positive,tick),
            steer_snap(negative,tick));
        const double yaw=nc.state[UNITY_SOURCE_YAW_RATE];
        const double departure=std::max(
            std::abs(pc.state[UNITY_SOURCE_YAW_RATE]-yaw),
            std::abs(mc.state[UNITY_SOURCE_YAW_RATE]-yaw));
        report["largest_normalized_yaw_rate_departure"]=
            std::max(report["largest_normalized_yaw_rate_departure"].get<double>(),
                departure);
        for(int w=0;w<4;++w) {
            const double load[3]={
                nc.normal_load_n[w],pc.normal_load_n[w],
                mc.normal_load_n[w]};
            record("wheel_contact",tick,w,load[0]>0.,
                load[1]>0.,load[2]>0.);
            const double longslip[3]={
                nc.longitudinal_slip[w],pc.longitudinal_slip[w],
                mc.longitudinal_slip[w]};
            record("longitudinal_curve_region",tick,w,
                tire_regime(longslip[0],0),
                tire_regime(longslip[1],0),
                tire_regime(longslip[2],0));
            const double latslip[3]={
                nc.lateral_slip[w],pc.lateral_slip[w],
                mc.lateral_slip[w]};
            record("lateral_curve_region",tick,w,
                tire_regime(latslip[0],1),
                tire_regime(latslip[1],1),
                tire_regime(latslip[2],1));
            const double suspension[3]={
                nc.suspension_force_n[w],pc.suspension_force_n[w],
                mc.suspension_force_n[w]};
            record("tire_load_filter_region",tick,w,
                load_filter_regime(suspension[0],w),
                load_filter_regime(suspension[1],w),
                load_filter_regime(suspension[2],w));
            const auto combined_slip=[&](int j) {
                const double lon=curve_response_ratio(longslip[j],false);
                const double lat=curve_response_ratio(latslip[j],true);
                return lon*lon+lat*lat>1.0;
            };
            record("combined_slip_utilization",tick,w,
                combined_slip(0),combined_slip(1),combined_slip(2));
            const double latforce=std::max(
                std::abs(pc.tire_lateral_force_n[w]-
                         nc.tire_lateral_force_n[w]),
                std::abs(mc.tire_lateral_force_n[w]-
                         nc.tire_lateral_force_n[w]));
            const double longforce=std::max(
                std::abs(pc.tire_longitudinal_force_n[w]-
                         nc.tire_longitudinal_force_n[w]),
                std::abs(mc.tire_longitudinal_force_n[w]-
                         nc.tire_longitudinal_force_n[w]));
            const double susforce=std::max(
                std::abs(pc.suspension_force_n[w]-
                         nc.suspension_force_n[w]),
                std::abs(mc.suspension_force_n[w]-
                         nc.suspension_force_n[w]));
            report["largest_lat_force_departure_N"]=
                std::max(report["largest_lat_force_departure_N"].get<double>(),
                         latforce);
            report["largest_long_force_departure_N"]=
                std::max(report["largest_long_force_departure_N"].get<double>(),
                         longforce);
            report["largest_suspension_force_departure_N"]=
                std::max(report["largest_suspension_force_departure_N"].get<double>(),
                         susforce);
            const double delta_slip=std::max(
                std::abs(pc.lateral_slip[w]-nc.lateral_slip[w]),
                std::abs(mc.lateral_slip[w]-nc.lateral_slip[w]));
            report["largest_slip_y_departure_rad"]=
                std::max(report["largest_slip_y_departure_rad"].get<double>(),
                         delta_slip);
        }
    }
    report["earliest_switch_ms"]=first;
    return report;
}

bool load_direction(const fs::path &path,double direction[2*N]) {
    std::ifstream input(path);
    if(!input)return false;
    std::string line;
    int count=0;
    while(std::getline(input,line)) {
        if(line.empty()||line[0]=='#')continue;
        std::istringstream row(line);
        int index=-1;double value=NAN;
        if(!(row>>index>>value)||index!=count||
           !std::isfinite(value))return false;
        direction[count++]=value;
        if(count>2*N)return false;
    }
    return count==2*N;
}
} // namespace

int main(int argc,char **argv) {
    if(argc<3||argc>4) {
        std::cerr<<"usage: mpc_source_stage_second_order_probe CASE_DIR "
            "QP_DIRECTION.txt [EPSILON=0.01]\n";
        return 2;
    }
    auto context=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    if(!sdu_apex::qp_fixture::read_failure_context(
        (fs::path(argv[1])/"source_context.mctx").string(),context.get()))
        return 3;
    double direction[2*N]{};
    if(!load_direction(argv[2],direction))return 4;
    const double epsilon=argc==4?std::stod(argv[3]):.01;
    if(!(epsilon>0.&&epsilon<=.2))return 5;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[N+1]{};
    if(!mpc_rti_build_nominal_same_epoch(
        &context->initial_state,context->progress_m,nullptr,
        context->r1_nominal,N,context->trajectory.data(),
        context->trajectory.size(),context->lap_length_m,
        context->prediction_dt,N,&context->model,nominal.get(),refs))
        return 6;
    const double scale[2]={.15,.25};
    double source_tangent[X]{};
    std::array<std::array<double,X>,N> source_tangent_by_stage{};
    std::array<MpcSourceDirectionalSecondOrder_t,N> serial_curvatures{};
    std::array<unsigned char,N> serial_valid{};
    json output;
    output["schema"]="source-stage-directional-curvature/v1";
    output["epsilon"]=epsilon;
    output["recorded_progress_m"]=context->progress_m;
    output["nominal_valid"]=nominal->valid!=0;
    output["stages"]=json::array();
    double total_second_us=0.,total_ad_us=0.;
    double total_branch_local_analytic_us=0.;
    int second_ok=0,second_not_smooth=0;
    const bool enable_branch_trace=
        std::getenv("MPC_SOURCE_TRACE_MICROBRANCHES")!=nullptr;
    const char *inner_stage_text=
        std::getenv("MPC_SOURCE_INNER_TRACE_STAGE");
    const int inner_stage=inner_stage_text?std::atoi(inner_stage_text):-1;
    double maximum_split_error=0.;
    double maximum_relative_difference=0.;
    for(int k=0;k<N;++k) {
        const auto &state=nominal->states[k].plant;
        const auto &control=nominal->controls[k];
        const double physical_direction[2]={
            scale[0]*direction[2*k],scale[1]*direction[2*k+1]};
        for(int i=0;i<X;++i)
            source_tangent_by_stage[k][i]=source_tangent[i];
        double seeds[X][L]{},controls[2][L]{};
        for(int i=0;i<X;++i)seeds[i][0]=source_tangent[i];
        controls[0][0]=physical_direction[0];
        controls[1][0]=physical_direction[1];
        double next[X][L]{},body[3][L]{},slip[4][L]{};
        const auto start_ad=std::chrono::steady_clock::now();
        if(!vehicle_model_full_source_interval_sensitivities(
            &state,&control,context->prediction_dt,
            seeds,controls,next,body,slip))return 7;
        const double ad_us=us(start_ad);
        total_ad_us+=ad_us;
        MpcSourceDirectionalSecondOrder_t second{};
        const auto start_second=std::chrono::steady_clock::now();
        const bool ok=vehicle_model_full_source_interval_directional_second_order(
            &state,&control,context->prediction_dt,
            source_tangent,physical_direction,epsilon,&second)!=0;
        const double second_us=us(start_second);
        total_second_us+=second_us;
        MpcSourceDirectionalSecondOrder_t analytic{};
        const auto start_analytic=std::chrono::steady_clock::now();
        const bool analytic_ok=
            vehicle_model_full_source_interval_analytic_directional_second_order(
                &state,&control,context->prediction_dt,
                source_tangent,physical_direction,&analytic)!=0;
        const double analytic_us=us(start_analytic);
        if(analytic_ok) {
            serial_curvatures[k]=analytic;
            serial_valid[k]=1;
        }
        total_branch_local_analytic_us+=analytic_us;
        json stage;
        stage["branch_local_analytic_valid"]=analytic_ok;
        stage["branch_local_analytic_us"]=analytic_us;
        if(analytic_ok) {
            double largest_scaled_first_ad_error=0.;
            double largest_scaled_second_fd_error=0.;
            json coordinates=json::object();
            for(int i:{UNITY_SOURCE_U,UNITY_SOURCE_V,
                 UNITY_SOURCE_YAW_RATE,UNITY_SOURCE_PHYSICAL_STEERING}) {
                const double predicted=analytic.first_source[i];
                const double expected_ad=next[i][0];
                const double true_fd=ok?second.second_source[i]:NAN;
                const double scaled_first=std::abs(predicted-expected_ad)/
                    std::max(1.,std::abs(expected_ad));
                largest_scaled_first_ad_error=std::max(
                    largest_scaled_first_ad_error,scaled_first);
                if(ok && std::isfinite(true_fd)) {
                    const double scaled_second=std::abs(
                        analytic.second_source[i]-true_fd)/
                        std::max(1.,std::abs(analytic.second_source[i]));
                    largest_scaled_second_fd_error=std::max(
                        largest_scaled_second_fd_error,scaled_second);
                }
                coordinates[std::to_string(i)]={{
                   "analytical_first",predicted},
                   {"existing_source_ad_first",expected_ad},
                   {"analytical_second",analytic.second_source[i]},
                   {"finite_difference_second",true_fd}};
            }
            stage["branch_local_analytic_state"]=coordinates;
            stage["branch_local_first_vs_source_ad_scaled_max"]=
                largest_scaled_first_ad_error;
            stage["branch_local_second_vs_source_fd_scaled_max"]=
                largest_scaled_second_fd_error;
        }
        stage["index"]=k;
        stage["steering_target"]=control.steering_target_rad;
        stage["throttle_target"]=control.throttle_command;
        stage["physical_direction"]=physical_direction;
        stage["ad_us"]=ad_us;
        stage["second_order_us"]=second_us;
        stage["physical_curvature_valid"]=ok;
        if(k==inner_stage) {
            stage["inner_physical_branch"]=
                exact_inner_substep_branch_comparison(
                    state,control,source_tangent,physical_direction,epsilon);
        }
        if(enable_branch_trace) {
            MicroTrace base{},pos{},neg{};
            if(!trace_source_microsteps(state,control,source_tangent,
                    physical_direction,0.,&base) ||
               !trace_source_microsteps(state,control,source_tangent,
                    physical_direction,epsilon,&pos) ||
               !trace_source_microsteps(state,control,source_tangent,
                    physical_direction,-epsilon,&neg))return 9;
            stage["microbranch_comparison"]=
                compare_microbranches(base,pos,neg);
            maximum_split_error=std::max(maximum_split_error,
                stage["microbranch_comparison"]["max_split_state_error"].get<double>());
        }
        if(ok) {
            ++second_ok;
            const int critical[4]={
                UNITY_SOURCE_U,UNITY_SOURCE_V,
                UNITY_SOURCE_YAW_RATE,UNITY_SOURCE_PHYSICAL_STEERING};
            double max_scaled_relative_first=0.;
            json outputs=json::object();
            for(int coordinate:critical) {
                const double predicted=next[coordinate][0];
                const double measured=second.first_source[coordinate];
                max_scaled_relative_first=std::max(max_scaled_relative_first,
                    std::abs(predicted-measured)/
                       std::max(1.,std::abs(measured)));
                outputs[std::to_string(coordinate)]={{
                    "first_ad",predicted},
                    {"first_exact_centered",measured},
                    {"second_exact_centered",second.second_source[coordinate]}};
            }
            stage["source_outputs"]=outputs;
            stage["body_second_order"]={
                second.second_body[0],second.second_body[1],
                second.second_body[2]};
            stage["slip_second_order"]={
                second.second_slip_excess_mps[0],
                second.second_slip_excess_mps[1],
                second.second_slip_excess_mps[2],
                second.second_slip_excess_mps[3]};
            stage["max_scaled_local_first_error"]=max_scaled_relative_first;
            if(max_scaled_relative_first>.05)++second_not_smooth;
            maximum_relative_difference=std::max(
                maximum_relative_difference,max_scaled_relative_first);
        }
        output["stages"].push_back(stage);
        for(int i=0;i<X;++i)source_tangent[i]=next[i][0];
    }
    // The source transition is independent at each stage once the exact
    // nominal and tangent direction are frozen. Benchmark the same 26-state
    // second-order algebra as SIX independently pinned workers. The existing
    // native SQP already parallelizes independent stage Jacobians; this
    // benchmark measures whether the richer source Hessian can share that
    // computational structure. It does NOT activate live second-order MPC.
    std::array<MpcSourceDirectionalSecondOrder_t,N> parallel_curvatures{};
    std::array<unsigned char,N> parallel_valid{};
    int workers=1;
#if defined(_OPENMP)
    workers=std::min(6,omp_get_max_threads());
#endif
    // One warm worker team prevents the first OpenMP startup overhead from
    // masquerading as source-plant computation in the timing comparison.
#if defined(_OPENMP)
#pragma omp parallel num_threads(workers) if(workers>1)
    {
        (void)omp_get_thread_num();
    }
#endif
    const auto parallel_start=std::chrono::steady_clock::now();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(workers) if(workers>1)
#endif
    for(int k=0;k<N;++k) {
        const double physical_direction[2]={
            scale[0]*direction[2*k],scale[1]*direction[2*k+1]};
        parallel_valid[k]=
            vehicle_model_full_source_interval_analytic_directional_second_order(
                &nominal->states[k].plant,&nominal->controls[k],
                context->prediction_dt,source_tangent_by_stage[k].data(),
                physical_direction,&parallel_curvatures[k])?1:0;
    }
    output["parallel_analytic_wall_us"]=us(parallel_start);
    output["parallel_analytic_threads"]=workers;
    double max_parallel_discrepancy=0.;
    bool all_valid=true;
    for(int k=0;k<N;++k) {
        if(!serial_valid[k] || !parallel_valid[k]) {
            all_valid=false;continue;
        }
        for(int row=0;row<X;++row) {
            max_parallel_discrepancy=std::max(max_parallel_discrepancy,
                std::abs(serial_curvatures[k].first_source[row]-
                         parallel_curvatures[k].first_source[row]));
            max_parallel_discrepancy=std::max(max_parallel_discrepancy,
                std::abs(serial_curvatures[k].second_source[row]-
                         parallel_curvatures[k].second_source[row]));
        }
    }
    output["parallel_analytic_all_valid"]=all_valid;
    output["parallel_analytic_max_absolute_parity_difference"]=
        max_parallel_discrepancy;
    output["second_order_valid_stages"]=second_ok;
    output["microstep_max_split_state_error"]=maximum_split_error;
    output["microsteps_traced"]=enable_branch_trace;
    output["local_first_error_above_5pct_stages"]=second_not_smooth;
    output["max_scaled_local_first_error"]=maximum_relative_difference;
    output["total_branch_local_analytic_us"]=
        total_branch_local_analytic_us;
    output["total_second_order_us"]=total_second_us;
    output["total_ad_us"]=total_ad_us;
    output["total_us"]=total_second_us+total_ad_us;
    std::cout<<std::setprecision(17)<<output.dump(2)<<"\n";
    return 0;
}
