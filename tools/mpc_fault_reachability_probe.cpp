// Offline diagnostic only: exact production-source MPC validator, no runtime modifications.
// Usage: mpc_fault_reachability_probe SOURCE_CSV WATCHER_LOG
#include "mpc_rti.h"
#include "mpc_trajectory_io.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using Controls = std::array<MpcModelControl_t, PREDICTION_HORIZON>;
static constexpr float kSteer=0.5235987756f;

static json fault_snapshot(std::istream &input) {
  std::string line;
  const std::string header="MPC_CYCLE_REJECTED: ";
  while(std::getline(input,line)) if(line.rfind(header,0)==0)
    return json::parse(line.substr(header.size()));
  throw std::runtime_error("no rejected MPC cycle in watcher");
}
static MpcRtiState_t parse_state(const json &j) {
  MpcRtiState_t s{};
  auto &p=s.plant;
  auto a=j.at("state");
  if(a.size()!=10) throw std::runtime_error("unexpected state dimensions");
  p.e_y=a[0].get<float>();p.e_psi=a[1].get<float>();
  p.u=a[2].get<float>();p.v=a[3].get<float>();p.r=a[4].get<float>();
  p.delayed_steering_command_1=a[5].get<float>();
  p.delayed_steering_command_2=a[6].get<float>();
  p.actual_steering_angle=a[7].get<float>();
  p.delayed_throttle_command_1=a[8].get<float>();
  p.delayed_throttle_command_2=a[9].get<float>();
  const auto internal=j.at("plant_internal_state");
  if(internal.size()!=MPC_PLANT_INTERNAL_STATE_COUNT)
    throw std::runtime_error("unexpected internal source-state dimensions");
  for(size_t i=0;i<internal.size();i++) p.source_plant_internal_state[i]=internal[i].get<double>();
  auto flags=j.at("plant_state_flags");
  p.source_plant_initialized=flags[0].get<int>();
  p.source_suspension_history_initialized=flags[1].get<int>();
  p.source_imu_state_initialized=flags[2].get<int>();
  return s;
}
static MpcRtiCycleConfiguration_t configuration(const json &j) {
  MpcRtiCycleConfiguration_t c{};
  auto &m=c.model;
  m.max_speed_mps=16.f;
  m.active_speed_ceiling_mps=j.at("active_speed_ceiling_mps").get<float>();
  m.max_steering_rad=kSteer;
  m.corridor_margin_m=.05f;
  m.first_prediction_corridor_margin_m=.05f;
  m.corridor_preview_halfwidth_m=.10f;
  m.nonlinear_corridor_tolerance_m=.005f;
  m.reference_clearance_reserve_m=.07f;
  m.startup_nominal_throttle_seed=.25f;
  m.terminal_multiplier=3.f;
  m.weight_e_y=300.f;
  m.weight_e_psi=75.f;
  m.weight_u=.35f;
  m.weight_u_overspeed=4.f;
  m.overspeed_curvature_gain_m=.5f;
  m.weight_longitudinal_slip=.05f;
  m.weight_v=.05f;
  m.weight_r=2.f;
  m.weight_steering_command=.03f;
  m.weight_steering_target=.05f;
  m.weight_throttle_command=.001f;
  m.weight_steering_slew=1.f;
  m.weight_throttle_slew=.1f;
  m.sqp_steering_trust_region_rad=.08f;
  m.sqp_throttle_trust_region=.10f;
  return c;
}
static Controls from_json(const json &a) {
  Controls controls{};
  for(size_t i=0;i<a.size()&&i<controls.size();i++){
    controls[i].steering_target_rad=a[i][0].get<float>();
    controls[i].throttle_command=a[i][1].get<float>();
  }
  return controls;
}
static void add(std::vector<float>& values, float x) {
  x=std::clamp(x,-kSteer,kSteer);
  for(auto previous:values) if(std::abs(x-previous)<.001f) return;
  values.push_back(x);
}
int main(int argc,char**argv) {
  try {
    if(argc!=3) throw std::runtime_error("usage: probe TRAJECTORY_CSV WATCHER_LOG");
    std::ifstream csv(argv[1]),log(argv[2]);
    if(!csv||!log) throw std::runtime_error("cannot open input");
    double length=0;
    auto trajectory=f1tenth_mpc::read_mpc_trajectory(csv,&length);
    json j=fault_snapshot(log);
    auto state=parse_state(j);
    auto config=configuration(j);
    const double progress=j.at("progress_m").get<double>();
    std::cout<<std::fixed<<std::setprecision(4);
    std::cout<<"TRACK_S "<<std::fmod(progress,length)<<" START "<<state.plant.e_y
             <<" "<<state.plant.e_psi<<" "<<state.plant.u
             <<" original_failure_stage="<<j.at("nonlinear_failure_stage")<<"\n";
    const int H=3;
    auto validate=[&](const Controls &controls, float *slack, int *stage) {
      MpcRtiNominal_t n{};
      auto code=mpc_rti_validate_candidate_plan(&state,progress,controls.data(),
        trajectory.data(),trajectory.size(),length,H,PREDICTION_DT_SECONDS,
        &config,&n,stage);
      if(code==MPC_RTI_ROLLOUT_OK&&slack) *slack=n.validation_minimum_corridor_slack_m;
      return code==MPC_RTI_ROLLOUT_OK;
    };
    if(j.contains("rejected_candidate_controls")){
      auto controls=from_json(j.at("rejected_candidate_controls"));
      float slack=-1;int stage=-1;
      auto ok=validate(controls,&slack,&stage);
      std::cout<<"CANDIDATE_75MS "<<ok<<" stage "<<stage<<" slack "<<slack<<"\n";
      if (!ok && stage < 0) {
        MpcRtiNominal_t verified{};int reason_stage=-1;
        const auto reason=mpc_rti_validate_candidate_plan(
          &state,progress,controls.data(),trajectory.data(),trajectory.size(),
          length,H,PREDICTION_DT_SECONDS,&config,&verified,&reason_stage);
        std::cout<<"UNREPLAYABLE_PRODUCTION_CANDIDATE status="<<(int)reason
                 <<" stage="<<reason_stage
                 <<" (do not infer controllability from grid)\n";
        return 3;
      }
    }
    if(j.contains("nominal_controls")){
      auto controls=from_json(j.at("nominal_controls"));
      float slack=-1;int stage=-1;
      auto ok=validate(controls,&slack,&stage);
      std::cout<<"NOMINAL_75MS "<<ok<<" stage "<<stage<<" slack "<<slack<<"\n";
    }
    std::vector<float> steering;
    for(float v: {-.5236f,-.4f,-.25f,-.12f,0.f,.12f,.25f,.4f,.5236f,
       state.plant.actual_steering_angle,
       state.plant.delayed_steering_command_2}) add(steering,v);
    auto candidate=j.at("rejected_candidate_controls");
    for(int i=0;i<3;i++) add(steering,candidate[i][0].get<float>());
    auto nominal=j.at("nominal_controls");
    for(int i=0;i<3;i++) add(steering,nominal[i][0].get<float>());
    std::vector<float> throttles={MPC_THROTTLE_MIN_DRIVE,.025f,.06f,.10f,.20f,.35f};
    throttles.push_back(state.plant.delayed_throttle_command_2);
    int attempted=0,valid=0,failed_first=0,failed_second=0,failed_third=0;
    float best_slack=-std::numeric_limits<float>::infinity();
    float best_steer=0,best_throttle=0;
    for(float steer:steering) for(float q:throttles) {
      Controls seq{};
      for(int i=0;i<H;i++){seq[i].steering_target_rad=steer;seq[i].throttle_command=q;}
      float slack=-1;int stage=-1;attempted++;
      bool ok=validate(seq,&slack,&stage);
      if(!ok) {if(stage==0) failed_first++; else if(stage==1) failed_second++; else if(stage==2) failed_third++;}
      else {
        valid++;
        if(slack>best_slack) {best_slack=slack;best_steer=steer;best_throttle=q;}
      }
    }
    std::cout<<"EXACT_GRID_75MS tested "<<attempted<<" valid "<<valid
             <<" failed_stage_0/1/2 "<<failed_first<<"/"<<failed_second<<"/"<<failed_third
             <<" best_slack "<<best_slack
             <<" best_steer "<<best_steer<<" best_q "<<best_throttle<<"\n";
    // Diagnostic extension: constant-action reachability over longer horizons.
    // This is a deliberately limited family, not a proof of viability.
    for(int horizon: {6,12,20}) {
      int feasible=0;float maximum_slack=-std::numeric_limits<float>::infinity();
      for(float steer:steering) for(float q:throttles) {
        Controls seq{};
        for(int i=0;i<horizon;i++) {seq[i].steering_target_rad=steer;seq[i].throttle_command=q;}
        MpcRtiNominal_t n{};int fail=-1;
        auto status=mpc_rti_validate_candidate_plan(&state,progress,seq.data(),
           trajectory.data(),trajectory.size(),length,horizon,PREDICTION_DT_SECONDS,
           &config,&n,&fail);
        if(status==MPC_RTI_ROLLOUT_OK) {
          feasible++;
          maximum_slack=std::max(maximum_slack,n.validation_minimum_corridor_slack_m);
        }
      }
      std::cout<<"EXACT_GRID_"<<(horizon*25)<<"MS viable "<<feasible
               <<" max_slack "<<maximum_slack<<"\n";
    }
  } catch (const std::exception &e) {std::cerr<<"ERROR "<<e.what()<<"\n"; return 2;}
}
