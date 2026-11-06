#include "unity_source_plant_c.h"

#include "unity_source_plant.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <tuple>

namespace {

f1tenth_mpc::UnitySourcePlantState from_c(
    const double values[UNITY_SOURCE_PLANT_STATE_SIZE], int history_initialized)
{
  f1tenth_mpc::UnitySourcePlantState state;
  state.u_mps = values[UNITY_SOURCE_U];
  state.v_mps = values[UNITY_SOURCE_V];
  state.yaw_rate_radps = values[UNITY_SOURCE_YAW_RATE];
  std::copy_n(values + UNITY_SOURCE_WHEEL_OMEGA_FL, 4,
              state.wheel_omega_radps.begin());
  state.wheel_rotation_angle_rad = {0.0, 0.0, 0.0, 0.0};
  state.body_height_m = values[UNITY_SOURCE_BODY_HEIGHT];
  state.body_height_rate_mps = values[UNITY_SOURCE_BODY_HEIGHT_RATE];
  state.roll_rad = values[UNITY_SOURCE_ROLL];
  state.roll_rate_radps = values[UNITY_SOURCE_ROLL_RATE];
  state.pitch_rad = values[UNITY_SOURCE_PITCH];
  state.pitch_rate_radps = values[UNITY_SOURCE_PITCH_RATE];
  state.physical_steering_rad = values[UNITY_SOURCE_PHYSICAL_STEERING];
  std::copy_n(values + UNITY_SOURCE_SUSPENSION_JOUNCE_FL, 4,
              state.suspension_jounce_m.begin());
  std::copy_n(values + UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL, 4,
              state.low_forward_speed_time_s.begin());
  std::copy_n(values + UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL, 4,
              state.low_side_speed_time_s.begin());
  state.suspension_history_initialized = history_initialized != 0;
  return state;
}

void to_c(const f1tenth_mpc::UnitySourcePlantOutput &source,
          UnitySourcePlantOutputC *output)
{
  const auto &state = source.state;
  output->state[UNITY_SOURCE_U] = state.u_mps;
  output->state[UNITY_SOURCE_V] = state.v_mps;
  output->state[UNITY_SOURCE_YAW_RATE] = state.yaw_rate_radps;
  output->integrated_body_forward_displacement_m =
      source.integrated_body_forward_displacement_m;
  output->integrated_body_lateral_displacement_m =
      source.integrated_body_lateral_displacement_m;
  output->integrated_yaw_delta_rad = source.integrated_yaw_delta_rad;
  std::copy(state.wheel_omega_radps.begin(), state.wheel_omega_radps.end(),
            output->state + UNITY_SOURCE_WHEEL_OMEGA_FL);
  std::copy(state.wheel_rotation_angle_rad.begin(),
            state.wheel_rotation_angle_rad.end(),
            output->wheel_rotation_angle_rad);
  output->state[UNITY_SOURCE_BODY_HEIGHT] = state.body_height_m;
  output->state[UNITY_SOURCE_BODY_HEIGHT_RATE] = state.body_height_rate_mps;
  output->state[UNITY_SOURCE_ROLL] = state.roll_rad;
  output->state[UNITY_SOURCE_ROLL_RATE] = state.roll_rate_radps;
  output->state[UNITY_SOURCE_PITCH] = state.pitch_rad;
  output->state[UNITY_SOURCE_PITCH_RATE] = state.pitch_rate_radps;
  output->state[UNITY_SOURCE_PHYSICAL_STEERING] =
      state.physical_steering_rad;
  std::copy(state.suspension_jounce_m.begin(), state.suspension_jounce_m.end(),
            output->state + UNITY_SOURCE_SUSPENSION_JOUNCE_FL);
  std::copy(state.low_forward_speed_time_s.begin(),
            state.low_forward_speed_time_s.end(),
            output->state + UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL);
  std::copy(state.low_side_speed_time_s.begin(),
            state.low_side_speed_time_s.end(),
            output->state + UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL);
  output->suspension_history_initialized =
      state.suspension_history_initialized ? 1 : 0;
  std::copy(source.longitudinal_slip.begin(), source.longitudinal_slip.end(),
            output->longitudinal_slip);
  std::copy(source.maximum_positive_longitudinal_slip.begin(),
            source.maximum_positive_longitudinal_slip.end(),
            output->maximum_positive_longitudinal_slip);
  std::copy(source.maximum_longitudinal_slip_excess_mps.begin(),
            source.maximum_longitudinal_slip_excess_mps.end(),
            output->maximum_longitudinal_slip_excess_mps);
  std::copy(source.lateral_slip.begin(), source.lateral_slip.end(),
            output->lateral_slip);
  std::copy(source.normal_load_n.begin(), source.normal_load_n.end(),
            output->normal_load_n);
  std::copy(source.suspension_force_n.begin(), source.suspension_force_n.end(),
            output->suspension_force_n);
  std::copy(source.suspension_jounce_rate_mps.begin(),
            source.suspension_jounce_rate_mps.end(),
            output->suspension_jounce_rate_mps);
  std::copy(source.tire_longitudinal_force_n.begin(),
            source.tire_longitudinal_force_n.end(),
            output->tire_longitudinal_force_n);
  std::copy(source.tire_lateral_force_n.begin(),
            source.tire_lateral_force_n.end(), output->tire_lateral_force_n);
  output->longitudinal_acceleration_mps2 =
      source.longitudinal_acceleration_mps2;
  output->lateral_acceleration_mps2 = source.lateral_acceleration_mps2;
  output->yaw_acceleration_radps2 = source.yaw_acceleration_radps2;
}

}  // namespace

extern "C" int unity_source_plant_step_c(
    const UnitySourcePlantInputC *input, double duration_s,
    UnitySourcePlantOutputC *output)
{
  if (!input || !output) return 0;
  const f1tenth_mpc::UnitySourcePlantInput plant_input{
      input->steering_target_rad, input->throttle_normalized};
  auto plant_state = from_c(input->state, input->suspension_history_initialized);
  std::copy_n(input->wheel_rotation_angle_rad, 4,
              plant_state.wheel_rotation_angle_rad.begin());
  f1tenth_mpc::UnitySourcePlantOutput plant_output;
  if (!f1tenth_mpc::unity_source_plant_step(
          plant_state,
          plant_input, duration_s, &plant_output)) {
    return 0;
  }
  to_c(plant_output, output);
  return 1;
}

extern "C" int unity_source_plant_step_directional_curvature_c(
    const UnitySourcePlantInputC *input,double duration_s,
    const double direction_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double direction_control[2],
    UnitySourcePlantDirectionalCurvatureC *out)
{
    if(!input||!direction_state||!direction_control||!out)return 0;
    auto physical=from_c(input->state,
        input->suspension_history_initialized);
    std::copy_n(input->wheel_rotation_angle_rad,4,
                physical.wheel_rotation_angle_rad.begin());
    const f1tenth_mpc::UnitySourcePlantInput command{
        input->steering_target_rad,input->throttle_normalized};
    std::array<double,UNITY_SOURCE_PLANT_STATE_SIZE> ds{};
    std::array<double,2> du{};
    std::copy_n(direction_state,ds.size(),ds.begin());
    std::copy_n(direction_control,du.size(),du.begin());
    f1tenth_mpc::UnitySourcePlantDirectionalCurvature curvature{};
    if(!f1tenth_mpc::unity_source_plant_step_with_directional_curvature(
        physical,command,duration_s,ds,du,&curvature))return 0;
    std::copy(curvature.first_state.begin(),
        curvature.first_state.end(),out->first_state);
    std::copy(curvature.second_state.begin(),
        curvature.second_state.end(),out->second_state);
    std::copy(curvature.first_body.begin(),
        curvature.first_body.end(),out->first_body);
    std::copy(curvature.second_body.begin(),
        curvature.second_body.end(),out->second_body);
    std::copy(curvature.first_slip_excess.begin(),
        curvature.first_slip_excess.end(),out->first_slip_excess);
    std::copy(curvature.second_slip_excess.begin(),
        curvature.second_slip_excess.end(),out->second_slip_excess);
    return 1;
}

extern "C" int unity_source_plant_step_one_sided_directional_c(
    const UnitySourcePlantInputC *input,double duration_s,
    const double direction_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double direction_control[2],
    UnitySourcePlantDirectionalCurvatureC *out,
    UnitySourceDirectionalBranchReportC *report)
{
    if(!report)return 0;
    const double second_state[UNITY_SOURCE_PLANT_STATE_SIZE]{};
    return unity_source_plant_step_path_curvature_c(input,duration_s,
        direction_state,second_state,direction_control,out,report);
}

extern "C" int unity_source_plant_step_path_curvature_c(
    const UnitySourcePlantInputC *input,double duration_s,
    const double direction_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double second_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double direction_control[2],
    UnitySourcePlantDirectionalCurvatureC *out,
    UnitySourceDirectionalBranchReportC *report)
{
    if(!input||!direction_state||!second_state||!direction_control||!out)
        return 0;
    auto physical=from_c(input->state,
        input->suspension_history_initialized);
    std::copy_n(input->wheel_rotation_angle_rad,4,
                physical.wheel_rotation_angle_rad.begin());
    const f1tenth_mpc::UnitySourcePlantInput command{
        input->steering_target_rad,input->throttle_normalized};
    std::array<double,UNITY_SOURCE_PLANT_STATE_SIZE> ds{};
    std::array<double,UNITY_SOURCE_PLANT_STATE_SIZE> dds{};
    std::array<double,2> du{};
    std::copy_n(direction_state,ds.size(),ds.begin());
    std::copy_n(second_state,dds.size(),dds.begin());
    std::copy_n(direction_control,du.size(),du.begin());
    f1tenth_mpc::UnitySourcePlantDirectionalCurvature curvature{};
    f1tenth_mpc::UnitySourceDirectionalBranchReport source_report{};
    if(!f1tenth_mpc::unity_source_plant_step_path_curvature(
        physical,command,duration_s,ds,dds,du,&curvature,report?&source_report:nullptr))
        return 0;
    std::copy(curvature.first_state.begin(),
        curvature.first_state.end(),out->first_state);
    std::copy(curvature.second_state.begin(),
        curvature.second_state.end(),out->second_state);
    std::copy(curvature.first_body.begin(),
        curvature.first_body.end(),out->first_body);
    std::copy(curvature.second_body.begin(),
        curvature.second_body.end(),out->second_body);
    std::copy(curvature.first_slip_excess.begin(),
        curvature.first_slip_excess.end(),out->first_slip_excess);
    std::copy(curvature.second_slip_excess.begin(),
        curvature.second_slip_excess.end(),out->second_slip_excess);
    if(report) {
    report->exact_physical_switch_ties=source_report.exact_physical_switch_ties;
    report->derivative_selected_switch_ties=
        source_report.derivative_selected_switch_ties;
    report->contact_switch_ties=source_report.contact_switch_ties;
    report->zero_throttle_brake_switches=
        source_report.zero_throttle_brake_switches;
    report->discrete_substep_schedule_switches=
        source_report.discrete_substep_schedule_switches;
    report->valid_one_sided_directional=
        source_report.valid_one_sided_directional;
    }
    return 1;
}

extern "C" int unity_source_plant_compare_physical_branches_c(
    const UnitySourcePlantInputC *original,
    const UnitySourcePlantInputC *candidate,
    double duration_s,UnitySourcePhysicalBranchComparisonC *out)
{
    if(!original||!candidate||!out)return 0;
    *out={};
    out->first_changed_fixed_tick=-1;
    out->first_changed_vehicle_substep=-1;
    out->first_changed_wheel=-1;
    const auto build=[&](
        const UnitySourcePlantInputC *input,
        std::vector<f1tenth_mpc::UnitySourceBranchObservation> *trace) {
        auto state=from_c(input->state,
            input->suspension_history_initialized);
        std::copy_n(input->wheel_rotation_angle_rad,4,
                    state.wheel_rotation_angle_rad.begin());
        const f1tenth_mpc::UnitySourcePlantInput command{
            input->steering_target_rad,input->throttle_normalized};
        return f1tenth_mpc::unity_source_plant_trace_branches(
            state,command,duration_s,trace);
    };
    using Observation=f1tenth_mpc::UnitySourceBranchObservation;
    std::vector<Observation> a,b;
    if(!build(original,&a)||!build(candidate,&b))return 0;
    out->original_internal_events=static_cast<int>(a.size());
    out->perturbed_internal_events=static_cast<int>(b.size());
    using Key=std::tuple<int,int,int>;
    std::map<Key,const Observation*> measured;
    for(const auto &record:b)
        measured[{record.fixed_tick,record.vehicle_substep,
                  record.wheel}]=&record;
    for(const auto &src:a) {
        const Key key{src.fixed_tick,src.vehicle_substep,src.wheel};
        auto it=measured.find(key);
        if(it==measured.end()) {
            ++out->mismatched_substep_schedule_events;
            if(out->first_changed_fixed_tick<0) {
                out->first_changed_fixed_tick=src.fixed_tick;
                out->first_changed_vehicle_substep=src.vehicle_substep;
                out->first_changed_wheel=src.wheel;
            }
            continue;
        }
        const auto &dst=*it->second;
        bool changed=false;
        const auto check=[&](bool different,int &counter) {
            if(different) {++counter;changed=true;}
        };
        check(src.vehicle_substep_count!=dst.vehicle_substep_count,
              out->mismatched_substep_schedule_events);
        check(src.grounded!=dst.grounded,
              out->contact_grounded_changes);
        check(src.sticky_forward!=dst.sticky_forward,
              out->sticky_forward_changes);
        check(src.sticky_side!=dst.sticky_side,
              out->sticky_side_changes);
        check(src.longitudinal_curve_region!=
              dst.longitudinal_curve_region,
              out->longitudinal_spline_region_changes);
        check(src.lateral_curve_region!=dst.lateral_curve_region,
              out->lateral_spline_region_changes);
        check(src.tire_load_filter_region!=dst.tire_load_filter_region,
              out->tire_load_filter_changes);
        check(src.combined_slip_saturated!=dst.combined_slip_saturated,
              out->combined_slip_saturation_changes);
        check(src.suspension_jounce_region!=
              dst.suspension_jounce_region,
              out->suspension_jounce_region_changes);
        check(src.longitudinal_slip_denominator_region!=
              dst.longitudinal_slip_denominator_region,
              out->slip_denominator_region_changes);
        check(src.steering_target_reached!=dst.steering_target_reached,
              out->steering_target_reached_changes);
        check(src.zero_throttle_brake_mode!=dst.zero_throttle_brake_mode,
              out->throttle_brake_mode_changes);
        if(changed && out->first_changed_fixed_tick<0) {
            out->first_changed_fixed_tick=src.fixed_tick;
            out->first_changed_vehicle_substep=src.vehicle_substep;
            out->first_changed_wheel=src.wheel;
        }
        measured.erase(it);
    }
    out->mismatched_substep_schedule_events+=
        static_cast<int>(measured.size());
    if(out->first_changed_fixed_tick<0&&!measured.empty()) {
        const auto &src=*measured.begin()->second;
        out->first_changed_fixed_tick=src.fixed_tick;
        out->first_changed_vehicle_substep=src.vehicle_substep;
        out->first_changed_wheel=src.wheel;
    }
    out->valid=1;
    return 1;
}

extern "C" int unity_source_plant_instantaneous_lateral_slip_c(
    const double state[UNITY_SOURCE_PLANT_STATE_SIZE], double slip[4])
{
    if(!state || !slip)return 0;
    const auto physical=from_c(state,1);
    std::array<double,4> result{};
    if(!f1tenth_mpc::unity_source_plant_lateral_slip_at_state(
           physical,&result))return 0;
    std::copy(result.begin(),result.end(),slip);
    return 1;
}

extern "C" int unity_source_plant_instantaneous_lateral_slip_ad_c(
    const double state[UNITY_SOURCE_PLANT_STATE_SIZE],double slip[4],
    double jacobian[4][UNITY_SOURCE_PLANT_STATE_SIZE])
{
    if(!state || !slip || !jacobian)return 0;
    const auto physical=from_c(state,1);
    std::array<double,4> values{};
    std::array<std::array<double,UNITY_SOURCE_PLANT_STATE_SIZE>,4> deriv{};
    if(!f1tenth_mpc::unity_source_plant_lateral_slip_with_sensitivities(
            physical,&values,&deriv))return 0;
    for(int w=0;w<4;++w) {
        slip[w]=values[w];
        for(int c=0;c<UNITY_SOURCE_PLANT_STATE_SIZE;++c)
            jacobian[w][c]=deriv[w][c];
    }
    return 1;
}

extern "C" int unity_source_plant_instantaneous_lateral_slip_second_c(
    const double state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double first_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    const double second_state[UNITY_SOURCE_PLANT_STATE_SIZE],
    double slip[4],double first_slip[4],double second_slip[4])
{
    if(!state || !first_state || !second_state ||
       !slip || !first_slip || !second_slip)return 0;
    auto physical=from_c(state,1);
    std::array<double,UNITY_SOURCE_PLANT_STATE_SIZE> first{},second{};
    std::copy_n(first_state,first.size(),first.begin());
    std::copy_n(second_state,second.size(),second.begin());
    std::array<double,4> values{},d1{},d2{};
    if(!f1tenth_mpc::unity_source_plant_lateral_slip_directional_second(
            physical,first,second,&values,&d1,&d2))return 0;
    std::copy(values.begin(),values.end(),slip);
    std::copy(d1.begin(),d1.end(),first_slip);
    std::copy(d2.begin(),d2.end(),second_slip);
    return 1;
}

extern "C" int unity_source_plant_full_and_substep_sensitivity_c(
    const UnitySourcePlantInputC *input,double duration_s,
    const UnitySourcePlantSensitivitySeedsC *seeds,
    UnitySourcePlantSensitivityC *out,
    UnitySourceSlipSubstepSensitivityC *samples,
    int capacity,int *count)
{
    if(!input||!seeds||!out||!count||capacity<0||
       (capacity>0&&!samples))return 0;
    auto physical=from_c(input->state,
        input->suspension_history_initialized);
    std::copy_n(input->wheel_rotation_angle_rad,4,
        physical.wheel_rotation_angle_rad.begin());
    const f1tenth_mpc::UnitySourcePlantInput control{
        input->steering_target_rad,input->throttle_normalized};
    f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeded{};
    for(int src=0;src<UNITY_SOURCE_PLANT_STATE_SIZE;++src)
        for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane)
            seeded.state[src][lane]=seeds->state[src][lane];
    for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane) {
        seeded.steering_target[lane]=seeds->steering_target[lane];
        seeded.throttle[lane]=seeds->throttle[lane];
    }
    f1tenth_mpc::UnitySourcePlantSensitivity complete{};
    std::vector<f1tenth_mpc::UnitySourceSlipSubstepSensitivity> rows;
    if(!f1tenth_mpc::unity_source_plant_step_with_sensitivities(
        physical,control,duration_s,seeded,&complete,&rows))
        return 0;
    *count=static_cast<int>(rows.size());
    if(*count>capacity)return 0;
    for(int wheel=0;wheel<4;++wheel) {
        out->maximum_longitudinal_slip_residual_mps[wheel]=
            complete.maximum_longitudinal_slip_residual_mps[wheel];
        for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane)
            out->maximum_longitudinal_slip_residual_derivative[wheel][lane]=
                complete.maximum_longitudinal_slip_residual_derivative[wheel][lane];
    }
    for(int src=0;src<UNITY_SOURCE_PLANT_STATE_SIZE;++src)
        for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane)
            out->state[src][lane]=complete.state[src][lane];
    for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane) {
        out->body_forward_displacement[lane]=
            complete.body_forward_displacement[lane];
        out->body_lateral_displacement[lane]=
            complete.body_lateral_displacement[lane];
        out->yaw_delta[lane]=complete.yaw_delta[lane];
    }
    for(int i=0;i<*count;++i) {
        const auto &r=rows[i];
        samples[i].original={
            r.fixed_tick,r.vehicle_substep,r.wheel,r.positive_excess_mps};
        std::copy(r.derivative.begin(),r.derivative.end(),
                  samples[i].derivative);
    }
    return 1;
}

extern "C" int unity_source_plant_all_slip_substep_sensitivity_c(
    const UnitySourcePlantInputC *input,double duration_s,
    const UnitySourcePlantSensitivitySeedsC *seeds,
    UnitySourceSlipSubstepSensitivityC *samples,int capacity,int *count)
{
    if(!input||!seeds||!count||capacity<0||
       (capacity>0&&!samples))return 0;
    auto physical=from_c(input->state,
        input->suspension_history_initialized);
    std::copy_n(input->wheel_rotation_angle_rad,4,
        physical.wheel_rotation_angle_rad.begin());
    const f1tenth_mpc::UnitySourcePlantInput control{
        input->steering_target_rad,input->throttle_normalized};
    f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeded{};
    for(int src=0;src<UNITY_SOURCE_PLANT_STATE_SIZE;++src)
        for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane)
            seeded.state[src][lane]=seeds->state[src][lane];
    for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane) {
        seeded.steering_target[lane]=seeds->steering_target[lane];
        seeded.throttle[lane]=seeds->throttle[lane];
    }
    f1tenth_mpc::UnitySourcePlantSensitivity source_jacobian{};
    std::vector<f1tenth_mpc::UnitySourceSlipSubstepSensitivity> all;
    if(!f1tenth_mpc::unity_source_plant_step_with_sensitivities(
        physical,control,duration_s,seeded,&source_jacobian,&all))
        return 0;
    *count=static_cast<int>(all.size());
    if(*count>capacity)return 0;
    for(int i=0;i<*count;++i) {
        const auto &original=all[i];
        samples[i].original={original.fixed_tick,
            original.vehicle_substep,original.wheel,
            original.positive_excess_mps};
        std::copy(original.derivative.begin(),
            original.derivative.end(),samples[i].derivative);
    }
    return 1;
}

extern "C" int unity_source_plant_all_slip_substeps_c(
    const UnitySourcePlantInputC *input,double duration_s,
    UnitySourceSlipSubstepC *samples,int capacity,int *count,
    double exact_source_maximum[4])
{
    if(!input||!count||!exact_source_maximum||capacity<0||
       (capacity>0&&!samples))return 0;
    auto physical=from_c(input->state,
        input->suspension_history_initialized);
    std::copy_n(input->wheel_rotation_angle_rad,4,
        physical.wheel_rotation_angle_rad.begin());
    const f1tenth_mpc::UnitySourcePlantInput control{
        input->steering_target_rad,input->throttle_normalized};
    std::vector<f1tenth_mpc::UnitySourceBranchObservation> events;
    f1tenth_mpc::UnitySourcePlantOutput source{};
    if(!f1tenth_mpc::unity_source_plant_trace_branches(
        physical,control,duration_s,&events,&source))return 0;
    *count=static_cast<int>(events.size());
    if(*count>capacity)return 0;
    for(int wheel=0;wheel<4;++wheel)
        exact_source_maximum[wheel]=
            source.maximum_longitudinal_slip_excess_mps[wheel];
    double observed_max[4]{};
    for(int i=0;i<*count;++i) {
        const auto &e=events[i];
        if(e.wheel<0||e.wheel>=4)return 0;
        const double excess=e.grounded?
            e.positive_longitudinal_slip_excess_mps:0.;
        if(!std::isfinite(excess)||excess<0.)return 0;
        samples[i]={e.fixed_tick,e.vehicle_substep,e.wheel,excess};
        observed_max[e.wheel]=std::max(observed_max[e.wheel],excess);
    }
    for(int wheel=0;wheel<4;++wheel)
        if(std::abs(observed_max[wheel]-
                    exact_source_maximum[wheel])>1.e-10)return 0;
    return 1;
}

extern "C" int unity_source_plant_peak_slip_trace_c(
    const UnitySourcePlantInputC *input,double duration_s,
    UnitySourcePeakSlipTraceC out[4])
{
    if(!input||!out)return 0;
    auto physical=from_c(input->state,
        input->suspension_history_initialized);
    std::copy_n(input->wheel_rotation_angle_rad,4,
        physical.wheel_rotation_angle_rad.begin());
    const f1tenth_mpc::UnitySourcePlantInput control{
        input->steering_target_rad,input->throttle_normalized};
    std::vector<f1tenth_mpc::UnitySourceBranchObservation> events;
    f1tenth_mpc::UnitySourcePlantOutput source{};
    if(!f1tenth_mpc::unity_source_plant_trace_branches(
        physical,control,duration_s,&events,&source))return 0;
    std::array<std::vector<double>,4> values{};
    for(int wheel=0;wheel<4;++wheel) {
        out[wheel]={};
        out[wheel].actual_source_max_excess_mps=
            source.maximum_longitudinal_slip_excess_mps[wheel];
        out[wheel].active_fixed_tick=-1;
        out[wheel].active_vehicle_substep=-1;
    }
    for(const auto &event:events) {
        if(!event.grounded)continue;
        const int wheel=event.wheel;
        if(wheel<0||wheel>=4)return 0;
        const double v=event.positive_longitudinal_slip_excess_mps;
        if(!std::isfinite(v))return 0;
        values[wheel].push_back(v);
        ++out[wheel].traced_substeps;
        if(v>out[wheel].traced_max_excess_mps) {
            out[wheel].traced_max_excess_mps=v;
            out[wheel].active_fixed_tick=event.fixed_tick;
            out[wheel].active_vehicle_substep=event.vehicle_substep;
        }
    }
    for(int wheel=0;wheel<4;++wheel) {
        auto &v=values[wheel];
        std::sort(v.begin(),v.end(),std::greater<double>());
        if(!v.empty()) {
            out[wheel].runner_up_excess_mps=
                v.size()>1?v[1]:0.;
            out[wheel].equal_max_substeps=static_cast<int>(
                std::count(v.begin(),v.end(),v[0]));
        }
        if(!std::isfinite(out[wheel].actual_source_max_excess_mps)||
           std::abs(out[wheel].actual_source_max_excess_mps-
                    out[wheel].traced_max_excess_mps)>1.e-10)
            return 0;
    }
    return 1;
}

extern "C" int unity_source_plant_reconcile_suspension_jounce_c(
    double state[UNITY_SOURCE_PLANT_STATE_SIZE],
    int *suspension_history_initialized)
{
  if (!state || !suspension_history_initialized) return 0;
  auto plant_state = from_c(state, *suspension_history_initialized);
  if (!f1tenth_mpc::unity_source_plant_reconcile_suspension_jounce(
          &plant_state))
    return 0;
  std::copy(plant_state.suspension_jounce_m.begin(),
            plant_state.suspension_jounce_m.end(),
            state + UNITY_SOURCE_SUSPENSION_JOUNCE_FL);
  *suspension_history_initialized =
      plant_state.suspension_history_initialized ? 1 : 0;
  return 1;
}

extern "C" int unity_source_plant_step_sensitivity_c(
    const UnitySourcePlantInputC *input, double duration_s,
    const UnitySourcePlantSensitivitySeedsC *seeds,
    UnitySourcePlantSensitivityC *sensitivity)
{
  if (!input || !seeds || !sensitivity) return 0;
  auto plant_state = from_c(input->state,
                            input->suspension_history_initialized);
  std::copy_n(input->wheel_rotation_angle_rad, 4,
              plant_state.wheel_rotation_angle_rad.begin());
  const f1tenth_mpc::UnitySourcePlantInput plant_input{
      input->steering_target_rad, input->throttle_normalized};
  f1tenth_mpc::UnitySourcePlantSensitivitySeeds typed_seeds;
  for (std::size_t state = 0; state < UNITY_SOURCE_PLANT_STATE_SIZE; ++state)
    for (std::size_t lane = 0;
         lane < UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH; ++lane)
      typed_seeds.state[state][lane] = seeds->state[state][lane];
  for (std::size_t lane = 0;
       lane < UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH; ++lane) {
    typed_seeds.steering_target[lane] = seeds->steering_target[lane];
    typed_seeds.throttle[lane] = seeds->throttle[lane];
  }
  f1tenth_mpc::UnitySourcePlantSensitivity typed_sensitivity;
  if (!f1tenth_mpc::unity_source_plant_step_with_sensitivities(
          plant_state, plant_input, duration_s, typed_seeds,
          &typed_sensitivity)) return 0;
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    sensitivity->maximum_longitudinal_slip_residual_mps[wheel] =
        typed_sensitivity.maximum_longitudinal_slip_residual_mps[wheel];
    for (std::size_t lane = 0;
         lane < UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH; ++lane)
      sensitivity->maximum_longitudinal_slip_residual_derivative[wheel][lane] =
          typed_sensitivity.maximum_longitudinal_slip_residual_derivative[wheel][lane];
  }
  for (std::size_t state = 0; state < UNITY_SOURCE_PLANT_STATE_SIZE; ++state)
    for (std::size_t lane = 0;
         lane < UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH; ++lane)
      sensitivity->state[state][lane] =
          typed_sensitivity.state[state][lane];
  for (std::size_t lane = 0;
       lane < UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH; ++lane) {
    sensitivity->body_forward_displacement[lane] =
        typed_sensitivity.body_forward_displacement[lane];
    sensitivity->body_lateral_displacement[lane] =
        typed_sensitivity.body_lateral_displacement[lane];
    sensitivity->yaw_delta[lane] = typed_sensitivity.yaw_delta[lane];
  }
  return 1;
}

extern "C" int unity_source_plant_slip_excess_c(
    const double state[UNITY_SOURCE_PLANT_STATE_SIZE],
    double longitudinal_slip[4], double excess_speed_mps[4],
    double excess_derivative_wheel_omega[4])
{
  if (!state || !longitudinal_slip || !excess_speed_mps ||
      !excess_derivative_wheel_omega) return 0;
  const auto source_state = from_c(state, 0);
  std::array<double, 4> slip{}, excess{}, derivative{};
  if (!f1tenth_mpc::unity_source_plant_slip_excess(
          source_state, &slip, &excess, &derivative)) return 0;
  std::copy(slip.begin(), slip.end(), longitudinal_slip);
  std::copy(excess.begin(), excess.end(), excess_speed_mps);
  std::copy(derivative.begin(), derivative.end(),
            excess_derivative_wheel_omega);
  return 1;
}
