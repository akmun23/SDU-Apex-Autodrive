#pragma once
#include "mpc_rti.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>

/* The native source-suspension Gauss-Newton Hessian is PSD before
 * binary32 rounding. Large near-rank-deficient wheel/body terms can lose PSD
 * after independent Q/N/R casts AND the incremental-command transform.
 *
 * Project only a roundoff-scale negative spectral component back to the
 * positive-semidefinite cone. Apply the gradient correction around the exact
 * physical nominal (both states and *solver-coordinate* input) so that the
 * source model first derivative is unchanged. There is no trust radius,
 * candidate fallback or change to the physical constraints.
 *
 * A genuinely indefinite Hessian is rejected: second-order source dynamics
 * should be incorporated using a regularized SQP Lagrangian Hessian with
 * explicit checks, not by disguising negative physical curvature as rounding.
 */
inline int mpc_rti_convexify_float_source_hessian(
    RiccatiStepData_t *stage,const float nominal_state[MPC_RTI_NX],
    const float nominal_input[MPC_RTI_NU],
    float *applied_max)
{
    if(!stage || !nominal_state || !nominal_input)return 0;
    constexpr int nx=MPC_RTI_NX,nu=MPC_RTI_NU,n=nx+nu;
    using Hessian=Eigen::Matrix<double,n,n>;
    const auto collect=[&](const RiccatiStepData_t &s) {
        Hessian h=Hessian::Zero();
        for(int i=0;i<nx;++i) {
            h(i,i)=double(s.Q_diag[i])+s.Q_offdiag[i][i];
            for(int j=i+1;j<nx;++j)
                h(i,j)=h(j,i)=.5*(double(s.Q_offdiag[i][j])+
                                        s.Q_offdiag[j][i]);
            for(int a=0;a<nu;++a)
                h(i,nx+a)=h(nx+a,i)=s.N[i][a];
        }
        for(int a=0;a<nu;++a)
            h(nx+a,nx+a)=s.R_diag[a];
        h(nx,nx+1)=h(nx+1,nx)=s.R_cross;
        return h;
    };
    const RiccatiStepData_t original=*stage;
    const Hessian h0=collect(original);
    Eigen::SelfAdjointEigenSolver<Hessian> eig(h0);
    if(eig.info()!=Eigen::Success)return 0;
    const auto lambda=eig.eigenvalues();
    const double initial=lambda[0];
    if(!std::isfinite(initial))return 0;
    if(applied_max)*applied_max=0.f;
    if(initial>=-1.e-7)return 1;
    const double amplitude=h0.cwiseAbs().maxCoeff();
    const double rounding_limit=
        4.*double(std::numeric_limits<float>::epsilon())*
        std::max(1.,amplitude);
    if(initial < -rounding_limit)return 0;

    Hessian repair=Hessian::Zero();
    // Minimal Frobenius correction before floating-point quantization.
    for(int i=0;i<n;++i) {
        if(lambda[i]>=0.)continue;
        const auto v=eig.eigenvectors().col(i);
        repair.noalias()+=(-lambda[i])*(v*v.transpose());
    }
    Eigen::Matrix<double,n,1> nominal{};
    for(int i=0;i<nx;++i) {
        if(!std::isfinite(nominal_state[i]))return 0;
        nominal[i]=nominal_state[i];
    }
    for(int a=0;a<nu;++a) {
        if(!std::isfinite(nominal_input[a]))return 0;
        nominal[nx+a]=nominal_input[a];
    }
    double padding=std::max(1.e-6,-initial*.01);
    for(int attempt=0;attempt<22 && padding<=rounding_limit*4.;++attempt) {
        *stage=original;
        // Positive padding is a controlled, *roundoff-bounded* numerical
        // guard against another float32 quantization of the spectral repair.
        const Hessian correction=repair+padding*Hessian::Identity();
        for(int i=0;i<nx;++i) {
            stage->Q_diag[i]=(float)(double(original.Q_diag[i])+correction(i,i));
            for(int j=i+1;j<nx;++j) {
                const float v=(float)(h0(i,j)+correction(i,j));
                stage->Q_offdiag[i][j]=v;
                stage->Q_offdiag[j][i]=v;
            }
            for(int a=0;a<nu;++a)
                stage->N[i][a]=(float)(h0(i,nx+a)+correction(i,nx+a));
        }
        for(int a=0;a<nu;++a)
            stage->R_diag[a]=(float)(double(original.R_diag[a])+
                                    correction(nx+a,nx+a));
        stage->R_cross=(float)(double(original.R_cross)+
                                correction(nx,nx+1));
        const Hessian realized=collect(*stage)-h0;
        const auto gradient_shift=realized*nominal;
        for(int i=0;i<nx;++i)
            stage->q[i]=(float)(double(original.q[i])-gradient_shift[i]);
        for(int a=0;a<nu;++a)
            stage->r[a]=(float)(double(original.r[a])-gradient_shift[nx+a]);

        Eigen::SelfAdjointEigenSolver<Hessian> check(
            collect(*stage),Eigen::EigenvaluesOnly);
        if(check.info()!=Eigen::Success)break;
        if(check.eigenvalues()[0]>=-1.e-7) {
            if(applied_max)
                *applied_max=(float)realized.cwiseAbs().maxCoeff();
            return 1;
        }
        padding*=2.;
    }
    *stage=original;
    return 0;
}
