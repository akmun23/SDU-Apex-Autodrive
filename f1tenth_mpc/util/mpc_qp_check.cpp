// JSONL interface for offline QP property tests. No ROS or production hooks.
#include "mpc_source_qp.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

using nlohmann::json;
using namespace sdu_apex::source_sqp;

namespace {
Eigen::VectorXd vector(const json &j)
{
    Eigen::VectorXd v(j.size());
    for(int i=0;i<v.size();++i)v(i)=j.at(i).get<double>();
    return v;
}
Eigen::MatrixXd matrix(const json &j,int n)
{
    Eigen::MatrixXd m(j.size(),n);
    for(int i=0;i<m.rows();++i) {
        if(j.at(i).size()!=size_t(n))throw std::runtime_error("matrix dimensions");
        for(int k=0;k<n;++k)m(i,k)=j.at(i).at(k).get<double>();
    }
    return m;
}
}

int main()
{
    try {
        std::string line;
        while(std::getline(std::cin,line)) {
            const auto input=json::parse(line);
            ConvexQp qp;
            qp.gradient=vector(input.at("gradient"));const int n=qp.gradient.size();
            qp.hessian=matrix(input.at("hessian"),n);
            qp.lower=vector(input.at("lower"));qp.upper=vector(input.at("upper"));
            qp.hard_jacobian=matrix(input.at("hard_jacobian"),n);qp.hard_margin=vector(input.at("hard_margin"));
            qp.elastic_jacobian=matrix(input.at("elastic_jacobian"),n);qp.elastic_margin=vector(input.at("elastic_margin"));
            qp.elastic_weight=vector(input.at("elastic_weight"));qp.elastic_linear_cost=vector(input.at("elastic_linear_cost"));
            ConvexQpResult result;
            const bool valid=solve_convex_qp(qp,input.value("maximum_iterations",120),input.value("tolerance",1.e-6),&result);
            json out={{"valid",valid},{"converged",result.converged},{"iterations",result.iterations},
                {"primal_residual",result.primal_residual},{"dual_residual",result.dual_residual},
                {"complementarity_residual",result.complementarity_residual},{"duality_gap",result.duality_gap}};
            out["objective_error_bound"]=result.objective_error_bound;
            if(valid)out["step"]=std::vector<double>(result.step.data(),result.step.data()+result.step.size());
            std::cout<<out.dump()<<'\n';
        }
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
