// Fixed-size Eigen alternative for Riccati P quadratic update.
// No heap allocation, no Eigen objects cross the C ABI.
#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Core>
#include <cmath>
#include "riccati_solver.h"
extern "C" void mpc_riccati_eigen_quadratic_update(
    const float *P,
    const float *A,
    const float *G,
    const float *K,
    const float Q_diag[RICCATI_MAX_NX],
    const float *Q_offdiag,
    float *P_out) {
    static_assert(RICCATI_MAX_NX == 23 && RICCATI_MAX_NU == 2);
    using Mat=Eigen::Matrix<float,23,23,Eigen::RowMajor>;
    using Controls=Eigen::Matrix<float,2,23,Eigen::RowMajor>;
    const Eigen::Map<const Mat,Eigen::Unaligned> p(P),a(A);
    const Eigen::Map<const Controls,Eigen::Unaligned> g(G),k(K);
    Eigen::Map<Mat,Eigen::Unaligned> out(P_out);
    Mat intermediate;
    intermediate.noalias()=p*a;
    out.noalias()=a.transpose()*intermediate;
    out.noalias()+=g.transpose()*k;
    out+=Eigen::Map<const Mat,Eigen::Unaligned>(Q_offdiag);
    for(int i=0;i<23;++i) {
        out(i,i)+=Q_diag[i];
        for(int j=i+1;j<23;++j)out(j,i)=out(i,j);
    }
}

// One call covers all 30 backward/forward stages; no per-stage crossing of
// the C++/C boundary. The public factorized interface may change d without
// requiring a quadratic refactorization, so that pathway never caches P*d.
extern "C" int mpc_riccati_eigen_factored_rhs(
    const RiccatiStepData_t *steps, const float *terminal_q,
    const float *x0, const RiccatiFactorization_t *factor,
    const float *z_x, const float *y_x,
    const float *z_u, const float *y_u,
    float *x_out, float *u_out)
{
    using Vec = Eigen::Matrix<float,23,1>;
    using Control = Eigen::Matrix<float,2,1>;
    using Matrix = Eigen::Matrix<float,23,23,Eigen::RowMajor>;
    using BMatrix = Eigen::Matrix<float,23,2,Eigen::RowMajor>;
    using GMatrix = Eigen::Matrix<float,2,23,Eigen::RowMajor>;
    using SMatrix = Eigen::Matrix<float,2,2,Eigen::RowMajor>;
    Vec p[RICCATI_HORIZON+1];
    Control kk[RICCATI_HORIZON];
    constexpr int n=RICCATI_HORIZON;
    for(int i=0;i<23;++i) {
        p[n](i)=terminal_q[i] -
           (factor->x_is_constrained[n][i]
            ? factor->rho_state[n][i]*(z_x[n*23+i]-y_x[n*23+i])
            : 0.f);
        if(!std::isfinite(p[n](i)))return 0;
    }
    for(int stage=n-1;stage>=0;--stage) {
        const auto &s=steps[stage];
        const Eigen::Map<const Matrix,Eigen::Unaligned> a(&s.A[0][0]);
        const Eigen::Map<const BMatrix,Eigen::Unaligned> b(&s.B[0][0]);
        const Eigen::Map<const GMatrix,Eigen::Unaligned> g(&factor->G[stage][0][0]);
        const Eigen::Map<const SMatrix,Eigen::Unaligned> inv(&factor->S_inv[stage][0][0]);
        const Eigen::Map<const Matrix,Eigen::Unaligned> pn(&factor->P[stage+1][0][0]);
        Vec shifted=p[stage+1];
        if(factor->private_fixed_qp_affine_cache_enabled) {
            shifted+=Eigen::Map<const Vec,Eigen::Unaligned>(
                factor->affine_state_shift[stage]);
        }else {
            shifted.noalias()+=pn*Eigen::Map<const Vec,Eigen::Unaligned>(s.d);
        }
        Vec q;
        Control r;
        for(int i=0;i<23;++i) {
            q(i)=s.q[i]-
                 (factor->x_is_constrained[stage][i]
                  ? factor->rho_state[stage][i]*
                     (z_x[stage*23+i]-y_x[stage*23+i])
                  : 0.f);
        }
        for(int i=0;i<2;++i)
            r(i)=s.r[i]-factor->rho_input[stage][i]*
               (z_u[stage*2+i]-y_u[stage*2+i]);
        if(!shifted.allFinite() || !q.allFinite() || !r.allFinite())
            return 0;
        const Control btp=b.transpose()*shifted;
        kk[stage].noalias()=-inv*(r+btp);
        if(!kk[stage].allFinite())return 0;
        p[stage].noalias()=a.transpose()*shifted;
        p[stage].noalias()+=g.transpose()*kk[stage];
        p[stage]+=q;
        if(!p[stage].allFinite())return 0;
    }
    Eigen::Map<Vec,Eigen::Unaligned> first_state(x_out);
    first_state=Eigen::Map<const Vec,Eigen::Unaligned>(x0);
    for(int stage=0;stage<n;++stage) {
        const auto &s=steps[stage];
        const Eigen::Map<const Matrix,Eigen::Unaligned> a(&s.A[0][0]);
        const Eigen::Map<const BMatrix,Eigen::Unaligned> b(&s.B[0][0]);
        const Eigen::Map<const GMatrix,Eigen::Unaligned> k(&factor->K[stage][0][0]);
        const Eigen::Map<const Vec,Eigen::Unaligned> current(&x_out[stage*23]);
        Eigen::Map<Control,Eigen::Unaligned> cmd(&u_out[stage*2]);
        cmd.noalias()=k*current;
        cmd+=kk[stage];
        if(!cmd.allFinite())return 0;
        Eigen::Map<Vec,Eigen::Unaligned> next(&x_out[(stage+1)*23]);
        next.noalias()=a*current;
        next.noalias()+=b*cmd;
        next+=Eigen::Map<const Vec,Eigen::Unaligned>(s.d);
        if(!next.allFinite())return 0;
    }
    return 1;
}
