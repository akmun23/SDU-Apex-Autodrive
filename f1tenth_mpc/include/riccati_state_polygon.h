#ifndef RICCATI_STATE_POLYGON_H
#define RICCATI_STATE_POLYGON_H

#include "mpc_types.h"
#include <math.h>
#include <float.h>

/* Exact diagonal-metric projection onto the intersection of a 2D box and
 * affine half-planes. Most iterates are interior or project onto one face;
 * only a multi-face corner needs the bounded vertex enumeration. No heap,
 * iteration cap, approximate alternating projections or hidden fallback. */
static inline int riccati_project_state_polygon(
    const RiccatiStatePolygon_t *polygon, const float input[2],
    const float lower[2], const float upper[2], const float metric[2],
    float output[2])
{
    if (!polygon || polygon->count < 0 ||
        polygon->count > RICCATI_STATE_POLYGON_MAX_PLANES) return 0;
    float a[RICCATI_STATE_POLYGON_MAX_PLANES + 4][2];
    float b[RICCATI_STATE_POLYGON_MAX_PLANES + 4];
    int count = polygon->count;
    for (int axis = 0; axis < 2; ++axis) {
        if (!isfinite(input[axis]) || !isfinite(lower[axis]) ||
            !isfinite(upper[axis]) || lower[axis] > upper[axis] ||
            !isfinite(metric[axis]) || metric[axis] <= 0.f) return 0;
    }
    for (int i = 0; i < count; ++i) {
        a[i][0] = polygon->normal[i][0];
        a[i][1] = polygon->normal[i][1];
        b[i] = polygon->upper[i];
        if (!isfinite(a[i][0]) || !isfinite(a[i][1]) || !isfinite(b[i]) ||
            !(a[i][0]*a[i][0] + a[i][1]*a[i][1] > 0.f)) return 0;
    }
    for (int axis = 0; axis < 2; ++axis) {
        if (lower[axis] > -BIG_BOUND) {
            a[count][0] = a[count][1] = 0.f;
            a[count][axis] = -1.f;
            b[count++] = -lower[axis];
        }
        if (upper[axis] < BIG_BOUND) {
            a[count][0] = a[count][1] = 0.f;
            a[count][axis] = 1.f;
            b[count++] = upper[axis];
        }
    }
    int inside = 1;
    for (int i = 0; i < count; ++i)
        if (a[i][0]*input[0] + a[i][1]*input[1] > b[i]) inside = 0;
    if (inside) {
        output[0] = input[0]; output[1] = input[1];
        return 1;
    }
    const double inverse0 = 1. / metric[0], inverse1 = 1. / metric[1];
    for (int i = 0; i < count; ++i) {
        const double violation = a[i][0]*(double)input[0] +
            a[i][1]*(double)input[1] - b[i];
        if (violation <= 0.) continue;
        const double denominator = a[i][0]*(double)a[i][0]*inverse0 +
            a[i][1]*(double)a[i][1]*inverse1;
        const double lambda = violation / denominator;
        const double x = input[0] - lambda*inverse0*a[i][0];
        const double y = input[1] - lambda*inverse1*a[i][1];
        int feasible = 1;
        for (int j = 0; j < count; ++j)
            if (a[j][0]*x + a[j][1]*y > b[j] + 1.e-7) { feasible = 0; break; }
        /* A projection onto one violated half-space is a lower bound on
         * distance to the intersection. If feasible, it is the exact answer. */
        if (feasible) { output[0] = (float)x; output[1] = (float)y; return 1; }
    }
    double best = DBL_MAX;
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            const double determinant = a[i][0]*(double)a[j][1] -
                a[i][1]*(double)a[j][0];
            if (fabs(determinant) <= 1.e-12) continue;
            const double x = (b[i]*(double)a[j][1] - a[i][1]*(double)b[j]) / determinant;
            const double y = (a[i][0]*(double)b[j] - b[i]*(double)a[j][0]) / determinant;
            int feasible = 1;
            for (int k = 0; k < count; ++k)
                if (a[k][0]*x + a[k][1]*y > b[k] + 1.e-7) { feasible = 0; break; }
            if (!feasible) continue;
            const double dx = x-input[0], dy = y-input[1];
            const double cost = metric[0]*dx*dx + metric[1]*dy*dy;
            if (cost < best) {
                best = cost;
                output[0] = (float)x; output[1] = (float)y;
            }
        }
    }
    return best < DBL_MAX;
}

/* Exact weighted Euclidean projection in (e_y, e_psi, delta_s).
 * Convex QP KKT conditions admit an active set of at most three independent
 * faces in 3D. Enumerate 1/2/3 active faces and retain the feasible candidate
 * with non-negative multipliers and minimum squared metric distance.
 * The legacy 2D solver remains the fast path when progress slopes are zero.
 * No iterative alternating-projection tolerance can silently soften walls. */
static inline int riccati_project_state_progress_polygon(
    const RiccatiStatePolygon_t *polygon, const float input[3],
    const float lower[3], const float upper[3], const float metric[3],
    float output[3])
{
    if (!polygon || polygon->count < 0 ||
        polygon->count > RICCATI_STATE_POLYGON_MAX_PLANES) return 0;
    double a[RICCATI_STATE_POLYGON_MAX_PLANES + 6][3]{};
    double b[RICCATI_STATE_POLYGON_MAX_PLANES + 6]{};
    const double inv[3]={1.0/metric[0],1.0/metric[1],1.0/metric[2]};
    int count=polygon->count;
    for (int j=0;j<3;++j)
        if(!isfinite(input[j]) || !isfinite(lower[j]) ||
           !isfinite(upper[j]) || lower[j]>upper[j] ||
           !isfinite(metric[j]) || metric[j]<=0.f) return 0;
    for(int i=0;i<count;++i) {
        a[i][0]=polygon->normal[i][0];
        a[i][1]=polygon->normal[i][1];
        a[i][2]=polygon->progress_normal[i];
        b[i]=polygon->upper[i];
        double n=0;
        for(double value:a[i]) {if(!isfinite(value))return 0;n+=value*value;}
        if(!isfinite(b[i]) || !(n>0.))return 0;
    }
    for(int j=0;j<3;++j) {
        if(lower[j]>-BIG_BOUND) {a[count][j]=-1.;b[count++]=-lower[j];}
        if(upper[j]<BIG_BOUND) {a[count][j]= 1.;b[count++]=upper[j];}
    }
    auto feasible=[&](const double v[3]) {
        for(int i=0;i<count;++i) {
            double lhs=0;
            for(int j=0;j<3;++j)lhs+=a[i][j]*v[j];
            if(lhs>b[i]+1.e-7)return false;
        }
        return true;
    };
    const double x[3]={input[0],input[1],input[2]};
    if(feasible(x)) {
        for(int j=0;j<3;++j)output[j]=input[j];
        return 1;
    }
    double best=DBL_MAX;
    // For an active set, solve (A W^-1 A^T) lambda = A x - b.
    // The KKT multiplier sign is required for true metric projection.
    auto evaluate=[&](int size,const int ids[3]) {
        double m[3][4]{};
        for(int i=0;i<size;++i) {
            double rhs=-b[ids[i]];
            for(int j=0;j<3;++j)rhs+=a[ids[i]][j]*x[j];
            m[i][size]=rhs;
            for(int h=0;h<size;++h)
                for(int j=0;j<3;++j)
                    m[i][h]+=a[ids[i]][j]*inv[j]*a[ids[h]][j];
        }
        for(int col=0;col<size;++col) {
            int pivot=col;
            for(int i=col+1;i<size;++i)
                if(fabs(m[i][col])>fabs(m[pivot][col]))pivot=i;
            if(fabs(m[pivot][col])<1.e-12)return;
            if(pivot!=col)
                for(int j=col;j<=size;++j) {
                    const double v=m[col][j];m[col][j]=m[pivot][j];
                    m[pivot][j]=v;
                }
            const double divisor=m[col][col];
            for(int j=col;j<=size;++j)m[col][j]/=divisor;
            for(int i=0;i<size;++i) if(i!=col) {
                const double factor=m[i][col];
                for(int j=col;j<=size;++j)m[i][j]-=factor*m[col][j];
            }
        }
        double projected[3]={x[0],x[1],x[2]};
        for(int i=0;i<size;++i) {
            const double lambda=m[i][size];
            if(lambda < -1.e-7 || !isfinite(lambda))return;
            for(int j=0;j<3;++j)
                projected[j]-=inv[j]*a[ids[i]][j]*lambda;
        }
        if(!feasible(projected))return;
        double cost=0;
        for(int j=0;j<3;++j)
            cost+=metric[j]*(projected[j]-x[j])*(projected[j]-x[j]);
        if(cost<best) {
            best=cost;
            for(int j=0;j<3;++j)output[j]=(float)projected[j];
        }
    };
    for(int i=0;i<count;++i) {
        const int ids[3]={i,0,0};
        evaluate(1,ids);
    }
    for(int i=0;i<count;++i)
        for(int j=i+1;j<count;++j) {
            const int ids[3]={i,j,0};
            evaluate(2,ids);
        }
    for(int i=0;i<count;++i)
        for(int j=i+1;j<count;++j)
            for(int k=j+1;k<count;++k) {
                const int ids[3]={i,j,k};
                evaluate(3,ids);
            }
    return best<DBL_MAX;
}

#endif
