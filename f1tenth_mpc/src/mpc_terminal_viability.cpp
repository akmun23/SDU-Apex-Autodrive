#include "mpc_rti_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
constexpr int tail_length=3;
constexpr int residual_count=tail_length*7;
struct Tail {
    MpcRtiNominal_t nominal{};
    double residual[residual_count]{};
    double weight[residual_count]{};
    double constraint[RICCATI_TERMINAL_MAX_PLANES]{};
};

bool continuation(const MpcRtiState_t &terminal, double progress,
    MpcModelControl_t held, const MpcTrajectorySample_t *trajectory,
    size_t count, double lap, float dt, const MpcRtiConfiguration_t &model,
    const MpcRtiObjectiveSchedule_t *schedule, Tail &out)
{
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(int k=0;k<tail_length;++k)controls[k]=held;
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    if(!mpc_rti_build_nominal_same_epoch(&terminal,progress,nullptr,controls,
        tail_length,trajectory,count,lap,dt,tail_length,&model,
        &out.nominal,references))return false;
    if(!schedule)return true;
    for(int k=1;k<=tail_length;++k) {
        const auto &state=out.nominal.states[k].plant;
        const auto &target=schedule->terminal_targets[k-1];
        const double r[5]={state.e_y-target.e_y,state.e_psi-target.e_psi,
            state.u-target.u,state.v-target.v,state.r-target.r};
        const double w[5]={model.weight_e_y,model.weight_e_psi,
            schedule->terminal_speed_weights[k-1],model.weight_v,model.weight_r};
        const int offset=(k-1)*7;
        for(int i=0;i<5;++i) {
            out.residual[offset+i]=r[i];out.weight[offset+i]=w[i];
        }
        float lower=0.f,upper=0.f;
        if(!mpc_rti_compute_corridor_slack(&state,references[k].left_bound,
            references[k].right_bound,&model,PREDICTION_HORIZON,&lower,&upper))
            return false;
        const double slack[2]={lower,upper};
        for(int side=0;side<2;++side) {
            out.constraint[(k-1)*2+side]=-slack[side];
            out.residual[offset+5+side]=std::max(0.,
                model.reference_clearance_reserve_m-slack[side]);
            out.weight[offset+5+side]=model.weight_e_y;
        }
        out.constraint[6+k-1]=state.u-model.max_speed_mps;
    }
    // A legal continuation must permit further progress. At stationary
    // startup zero progress remains legal; no artificial heading/slip box.
    out.constraint[9]=progress-out.nominal.progress[tail_length]+
        (terminal.plant.u>.5f?1.e-5:0.);
    return true;
}

/* Propagate the SAME source sensitivities and exact loaded-path frame
 * differential as the main QP. No finite differences of source trajectories. */
bool continuation_sensitivities(const Tail &tail,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    const MpcRtiConfiguration_t &model,float dt,
    double jacobian[residual_count][MPC_RTI_NX],
    double constraints[RICCATI_TERMINAL_MAX_PLANES][MPC_RTI_NX])
{
    double state[MPC_OPT_PHYSICAL_NX][MPC_RTI_NX]{};
    double station[MPC_RTI_NX]{};
    for(int i=0;i<MPC_OPT_PHYSICAL_NX;++i)state[i][i]=1.;
    station[MPC_OPT_IDX_PROGRESS_DEVIATION]=1.;
    const auto full_index=[](int compact){return compact<5?compact:compact==5?7:compact+4;};
    for(int k=0;k<tail_length;++k) {
        const auto &incoming=tail.nominal.states[k+1];
        if(!incoming.incoming_tracking_jacobian_valid)return false;
        MpcStageResult_t known{};
        known.next=incoming.plant;known.valid=incoming.incoming_transition_valid;
        known.branch_flags=incoming.incoming_branch_flags;
        known.body_forward_delta_m=incoming.incoming_body_forward_delta_m;
        known.body_lateral_delta_m=incoming.incoming_body_lateral_delta_m;
        known.body_yaw_delta_rad=incoming.incoming_body_yaw_delta_rad;
        for(int w=0;w<4;++w)known.output.maximum_longitudinal_slip_excess_mps[w]=
            incoming.incoming_wheel_slip_excess_mps[w];
        MpcStageLinearization_t lin{};
        if(!mpc_model_linearize_sensitivities_from_known_step(
            &tail.nominal.states[k].plant,&tail.nominal.controls[k],dt,0.f,&known,&lin))return false;
        const auto &geometry=incoming.incoming_tracking_jacobian;
        double next[MPC_OPT_PHYSICAL_NX][MPC_RTI_NX]{};
        double next_station[MPC_RTI_NX]{};
        for(int i=0;i<MPC_OPT_PHYSICAL_NX;++i) {
            for(int j=0;j<MPC_OPT_PHYSICAL_NX;++j) {
                double a=lin.A[full_index(i)][full_index(j)];
                if(i<2) {
                    a=j<2?geometry[i][j]:0.;
                    for(int m=0;m<3;++m)a+=geometry[i][m+2]*lin.body_motion_A[m][full_index(j)];
                }
                for(int c=0;c<MPC_RTI_NX;++c)next[i][c]+=a*state[j][c];
            }
            for(int a=0;a<2;++a) {
                double b=lin.B[full_index(i)][a];
                if(i<2) {
                    b=0.;for(int m=0;m<3;++m)b+=geometry[i][m+2]*lin.body_motion_B[m][a];
                }
                next[i][MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND+a]+=b;
            }
            if(i<2)for(int c=0;c<MPC_RTI_NX;++c)next[i][c]+=geometry[i][5]*station[c];
        }
        for(int c=0;c<MPC_RTI_NX;++c)next_station[c]=geometry[2][5]*station[c];
        for(int j=0;j<MPC_OPT_PHYSICAL_NX;++j) {
            double a=j<2?geometry[2][j]:0.;
            for(int m=0;m<3;++m)a+=geometry[2][m+2]*lin.body_motion_A[m][full_index(j)];
            for(int c=0;c<MPC_RTI_NX;++c)next_station[c]+=a*state[j][c];
        }
        for(int a=0;a<2;++a)
            for(int m=0;m<3;++m)next_station[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND+a]+=
                geometry[2][m+2]*lin.body_motion_B[m][a];
        std::memcpy(state,next,sizeof(state));std::memcpy(station,next_station,sizeof(station));
        for(int c=0;c<MPC_RTI_NX;++c)constraints[6+k][c]=state[2][c];
        const int offset=k*7;
        for(int r=0;r<5;++r)
            for(int c=0;c<MPC_RTI_NX;++c)jacobian[offset+r][c]=state[r][c];
        // Only the geometry sampler/footprint is differenced: inexpensive,
        // independent of plant propagation, and branch-consistent at knots.
        double wall[2][3]{};
        for(int axis=0;axis<3;++axis) {
            constexpr double h=1.e-4;
            auto plus=incoming.plant,minus=incoming.plant;
            double sp=tail.nominal.progress[k+1],sm=sp;
            if(axis==0){plus.e_y+=h;minus.e_y-=h;}
            if(axis==1){plus.e_psi+=h;minus.e_psi-=h;}
            if(axis==2){sp+=h;sm-=h;}
            MpcRtiReference_t rp{},rm{};
            if(!mpc_rti_reference_at_progress(trajectory,count,lap,sp,&model,&rp) ||
               !mpc_rti_reference_at_progress(trajectory,count,lap,sm,&model,&rm))return false;
            float lp=0.f,up=0.f,lm=0.f,um=0.f;
            if(!mpc_rti_compute_corridor_slack(&plus,rp.left_bound,rp.right_bound,
                &model,PREDICTION_HORIZON,&lp,&up) ||
               !mpc_rti_compute_corridor_slack(&minus,rm.left_bound,rm.right_bound,
                &model,PREDICTION_HORIZON,&lm,&um))return false;
            wall[0][axis]=-(lp-lm)/(2.*h);wall[1][axis]=-(up-um)/(2.*h);
        }
        for(int side=0;side<2;++side)
            for(int c=0;c<MPC_RTI_NX;++c) {
                const double d=wall[side][0]*state[0][c]+wall[side][1]*state[1][c]+wall[side][2]*station[c];
                constraints[k*2+side][c]=d;
                jacobian[offset+5+side][c]=tail.residual[offset+5+side]>0.?d:0.;
            }
    }
    for(int c=0;c<MPC_RTI_NX;++c)
        constraints[9][c]=(c==MPC_OPT_IDX_PROGRESS_DEVIATION?1.:0.)-station[c];
    return true;
}
}

int mpc_rti_initialize_terminal_targets(const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,float dt,
    const MpcRtiConfiguration_t *model,MpcRtiObjectiveSchedule_t *schedule)
{
    if(!nominal || !nominal->valid || !model || !schedule ||
        nominal->horizon!=PREDICTION_HORIZON)return 0;
    Tail tail;
    if(!continuation(nominal->states[PREDICTION_HORIZON],
        nominal->progress[PREDICTION_HORIZON],nominal->controls[PREDICTION_HORIZON-1],
        trajectory,count,lap,dt,*model,nullptr,tail))return 0;
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    if(!mpc_rti_references_from_exact_nominal(&tail.nominal,trajectory,count,
        lap,model,references))return 0;
    for(int k=0;k<tail_length;++k) {
        schedule->terminal_targets[k]=references[k+1];
        schedule->terminal_speed_weights[k]=tail.nominal.states[k+1].plant.u>
            references[k+1].u?overspeed_weight_for_reference(model,&references[k+1]):
            underspeed_weight_for_reference(model,&references[k+1]);
    }
    schedule->terminal_prediction_dt=dt;
    schedule->terminal_targets_valid=1;
    return 1;
}

int mpc_rti_terminal_value(const MpcRtiNominal_t *plan,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    const MpcRtiConfiguration_t *model,const MpcRtiObjectiveSchedule_t *schedule,
    double *cost,int verify)
{
    if(!plan || !plan->valid || plan->horizon!=PREDICTION_HORIZON || !model ||
        !schedule || !schedule->terminal_targets_valid || !cost)return 0;
    Tail tail;
    if(!continuation(plan->states[PREDICTION_HORIZON],
        plan->progress[PREDICTION_HORIZON],plan->controls[PREDICTION_HORIZON-1],
        trajectory,count,lap,schedule->terminal_prediction_dt,*model,schedule,tail))return 0;
    *cost=0.;
    for(int r=0;r<residual_count;++r)
        *cost+=tail.weight[r]*tail.residual[r]*tail.residual[r];
    if(!std::isfinite(*cost))return 0;
    if(verify) {
        for(double c:tail.constraint)if(!std::isfinite(c) || c>0.)return 0;
        // Reuse the production physical/world-sweep gate for every tail step.
        MpcRtiCycleConfiguration_t validation{};
        validation.model=*model;
        MpcRtiNominal_t checked{};
        int failure=-1;
        if(mpc_rti_validate_candidate_plan(&plan->states[PREDICTION_HORIZON],
            plan->progress[PREDICTION_HORIZON],tail.nominal.controls,
            trajectory,count,lap,tail_length,schedule->terminal_prediction_dt,
            &validation,&checked,&failure)!=MPC_RTI_ROLLOUT_OK)return 0;
        if(plan->states[PREDICTION_HORIZON].plant.u>.5f &&
            checked.progress[tail_length]<=plan->progress[PREDICTION_HORIZON])return 0;
    }
    return 1;
}

int mpc_rti_add_terminal_viability(MpcRtiProblem_t *problem,
    const MpcRtiNominal_t *nominal,const MpcTrajectorySample_t *trajectory,
    size_t count,double lap,const MpcRtiConfiguration_t *model,
    const MpcRtiObjectiveSchedule_t *schedule)
{
    if(!problem || !nominal || !model || !schedule ||
        !schedule->terminal_targets_valid)return 0;
    const auto &terminal=nominal->states[PREDICTION_HORIZON];
    const double progress=nominal->progress[PREDICTION_HORIZON];
    const auto held=nominal->controls[PREDICTION_HORIZON-1];
    Tail base;
    if(!continuation(terminal,progress,held,trajectory,count,lap,
        schedule->terminal_prediction_dt,*model,schedule,base))return 0;
    double jacobian[residual_count][MPC_RTI_NX]{};
    double constraints[RICCATI_TERMINAL_MAX_PLANES][MPC_RTI_NX]{};
    // Physical scale for a PSD diagonal majorizer of the dense source GN
    // curvature. Preserve its gradient; no additional nominal tracking bias.
    double scale[MPC_RTI_NX];
    for(int i=0;i<MPC_RTI_NX;++i)
        scale[i]=i<2?.1:i<5?1.:i==5?.1:i<10?100.:i<20?.01:i==20?1.:.1;
    if(!continuation_sensitivities(base,trajectory,count,lap,*model,
        schedule->terminal_prediction_dt,jacobian,constraints))return 0;
    float x[MPC_RTI_NX]{};serialize_state(&terminal,x);
    x[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]=held.steering_target_rad;
    x[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]=held.throttle_command;
    for(int i=0;i<MPC_RTI_NX;++i) {
        double gradient=0.,majorizer=0.;
        for(int r=0;r<residual_count;++r) {
            gradient+=2.*base.weight[r]*base.residual[r]*jacobian[r][i];
            double norm=0.;
            for(int j=0;j<MPC_RTI_NX;++j)norm+=std::abs(jacobian[r][j])*scale[j];
            majorizer+=2.*base.weight[r]*std::abs(jacobian[r][i])*norm/scale[i];
        }
        if(!std::isfinite(gradient) || !std::isfinite(majorizer))return 0;
        problem->terminal_Q[i]+=(float)majorizer;
        problem->terminal_q[i]+=(float)(gradient-majorizer*x[i]);
    }
    auto &planes=problem->terminal_constraints;
    planes.count=0;planes.rho=7.f;
    for(int f=0;f<RICCATI_TERMINAL_MAX_PLANES;++f) {
        double bound=-base.constraint[f],norm=0.;
        for(int i=0;i<MPC_RTI_NX;++i) {
            bound+=constraints[f][i]*x[i];
            norm+=constraints[f][i]*constraints[f][i]*scale[i]*scale[i];
        }
        // A stopped source branch can make the progress constraint a
        // satisfied constant. Do not send a zero-normal plane to ADMM.
        if(norm==0.) {
            if(base.constraint[f]>0.)return 0;
            continue;
        }
        // Normalize by physical scales, not arbitrary state magnitudes.
        norm=std::max(1.e-6,std::sqrt(norm));
        const int row=planes.count++;
        for(int i=0;i<MPC_RTI_NX;++i)
            planes.normal[row][i]=(float)(constraints[f][i]/norm);
        planes.upper[row]=(float)(bound/norm);
    }
    return 1;
}
