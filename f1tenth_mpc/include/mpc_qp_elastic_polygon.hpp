#pragma once

#include "mpc_types.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <limits>

/* Exact convex objective (elastic-slack eliminated), solved with a
 * semismooth box-active Newton step and monotone globalization.
 *
 *    min_{lower <= z <= upper}
 *        0.5 (z-v)'diag(rho)(z-v)
 *        + 0.5 w sum_face max(0,a_face*z-b_face)^2 .
 *
 * Unlike shifting the walls by 0.18m, THIS QP retains the original physical
 * halfplane boundaries: each violation is represented by a nonnegative
 * penalized optimization slack. The returned z is a soft QP auxiliary state,
 * NOT a certified vehicle clearance or publishable command.
 * Handles both the old 2D geometry and progress-dependent 3D geometry with
 * no new Riccati horizon dimensions. Hard control/state boxes remain exact.
 */
inline int riccati_elastic_polygon_prox(
    const RiccatiStatePolygon_t *polygon,
    const float input[3],const float lower[3],const float upper[3],
    const float rho[3],float result[3])
{
    if(!polygon || !input || !lower || !upper || !rho || !result ||
       polygon->count<1 || polygon->count>RICCATI_STATE_POLYGON_MAX_PLANES ||
       !std::isfinite(polygon->elastic_weight) ||
       polygon->elastic_weight<=0.f)return 0;
    constexpr int n=3;
    const double w=polygon->elastic_weight;
    double a[RICCATI_STATE_POLYGON_MAX_PLANES][n]{};
    double b[RICCATI_STATE_POLYGON_MAX_PLANES]{};
    double x[n],v[n],lb[n],ub[n],r[n];
    for(int j=0;j<n;++j) {
        if(!std::isfinite(input[j]) || !std::isfinite(lower[j]) ||
           !std::isfinite(upper[j]) || lower[j]>upper[j] ||
           !std::isfinite(rho[j]) || rho[j]<=0.f)return 0;
        v[j]=input[j];lb[j]=lower[j];ub[j]=upper[j];r[j]=rho[j];
        x[j]=std::clamp(v[j],lb[j],ub[j]);
    }
    for(int f=0;f<polygon->count;++f) {
        a[f][0]=polygon->normal[f][0];
        a[f][1]=polygon->normal[f][1];
        a[f][2]=polygon->progress_normal[f];
        b[f]=polygon->upper[f];
        double length2=0.;
        for(int j=0;j<n;++j) {
            if(!std::isfinite(a[f][j]))return 0;
            length2+=a[f][j]*a[f][j];
        }
        if(!std::isfinite(b[f]) || length2<=0.)return 0;
    }
    const auto energy=[&](const double z[3]) {
        double cost=0.;
        for(int j=0;j<n;++j)
            cost+=.5*r[j]*(z[j]-v[j])*(z[j]-v[j]);
        for(int f=0;f<polygon->count;++f) {
            double h=-b[f];
            for(int j=0;j<n;++j)h+=a[f][j]*z[j];
            cost+=.5*w*std::max(0.,h)*std::max(0.,h);
        }
        return cost;
    };
    for(int iter=0;iter<75;++iter) {
        double g[n]{};
        double H[n][n]{};
        for(int j=0;j<n;++j) {
            g[j]=r[j]*(x[j]-v[j]);
            H[j][j]=r[j];
        }
        for(int f=0;f<polygon->count;++f) {
            double h=-b[f];
            for(int j=0;j<n;++j)h+=a[f][j]*x[j];
            if(h<=0.)continue;
            for(int j=0;j<n;++j) {
                g[j]+=w*h*a[f][j];
                for(int k=0;k<n;++k)
                    H[j][k]+=w*a[f][j]*a[f][k];
            }
        }
        double projected_stationarity=0.;
        int active[n]{};
        for(int j=0;j<n;++j) {
            const double scaled_step=g[j]/H[j][j];
            const double prox=std::clamp(x[j]-scaled_step,lb[j],ub[j]);
            projected_stationarity=std::max(projected_stationarity,
                                           std::abs(x[j]-prox));
            active[j]=(x[j]<=lb[j]+1.e-11 && g[j]>0.) ||
                      (x[j]>=ub[j]-1.e-11 && g[j]<0.);
        }
        if(projected_stationarity <= 5.e-10) {
            for(int j=0;j<n;++j)result[j]=(float)x[j];
            return 1;
        }
        double M[n][n+1]{};
        for(int j=0;j<n;++j) {
            for(int k=0;k<n;++k)
                M[j][k]=active[j]?double(j==k):
                    (active[k]?0.:H[j][k]);
            M[j][n]=active[j]?0.:-g[j];
        }
        for(int pivot=0;pivot<n;++pivot) {
            int row=pivot;
            for(int j=pivot+1;j<n;++j)
                if(std::abs(M[j][pivot])>std::abs(M[row][pivot]))row=j;
            if(std::abs(M[row][pivot])<1.e-16)return 0;
            if(row!=pivot)
                for(int j=pivot;j<=n;++j)
                    std::swap(M[row][j],M[pivot][j]);
            const double diag=M[pivot][pivot];
            for(int j=pivot;j<=n;++j)M[pivot][j]/=diag;
            for(int j=0;j<n;++j)if(j!=pivot) {
                const double scale=M[j][pivot];
                for(int k=pivot;k<=n;++k)
                    M[j][k]-=scale*M[pivot][k];
            }
        }
        const double current=energy(x);
        double next[n]{};
        bool accepted=false;
        double max_move=0.;
        for(double alpha=1.;alpha>=1./(1<<24);alpha*=.5) {
            max_move=0.;
            for(int j=0;j<n;++j) {
                next[j]=std::clamp(x[j]+alpha*M[j][n],lb[j],ub[j]);
                max_move=std::max(max_move,std::abs(next[j]-x[j]));
            }
            if(max_move<=1.e-13)break;
            double slope=0.;
            for(int j=0;j<n;++j)
                slope+=g[j]*(next[j]-x[j]);
            const double trial_cost=energy(next);
            if(std::isfinite(trial_cost) &&
               trial_cost<=current+1.e-4*slope+1.e-12*
                   std::max(1.,std::abs(current))) {
                accepted=true;break;
            }
        }
        if(!accepted)return 0;
        for(int j=0;j<n;++j)x[j]=next[j];
    }
    return 0;
}
