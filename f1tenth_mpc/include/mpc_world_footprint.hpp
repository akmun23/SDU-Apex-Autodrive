#pragma once
// Conservative discrete occupancy/body certificate, independent of Frenet
// raceline bounds. Immutable precomputed grid. No allocations on control path.
// Unknown cells and map edges are obstacles. Map data are ROS OccupancyGrid
// row-major bottom-left, not vertically inverted PGM image rows.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace f1tenth_mpc {
struct MpcBodyPose {
    double x, y, yaw;
};
class MpcWorldFootprint {
public:
    static std::shared_ptr<const MpcWorldFootprint> from_occupancy(
        int width, int height, double resolution,
        double origin_x, double origin_y, double origin_yaw,
        const std::vector<int8_t> &occupancy)
    {
        if (width < 3 || height < 3 || width > 4096 || height > 4096 ||
            !std::isfinite(resolution) || resolution <= 0.0 ||
            !std::isfinite(origin_x) || !std::isfinite(origin_y) ||
            !std::isfinite(origin_yaw) ||
            occupancy.size() != static_cast<std::size_t>(width) * height)
            return {};
        auto p = std::shared_ptr<MpcWorldFootprint>(new MpcWorldFootprint);
        p->width_=width; p->height_=height; p->resolution_=resolution;
        p->origin_x_=origin_x; p->origin_y_=origin_y;
        p->c_=std::cos(origin_yaw);p->s_=std::sin(origin_yaw);
        const int n=std::max(width,height);
        std::vector<double> f(n),tmp(static_cast<std::size_t>(width)*height);
        std::vector<double> d(n);
        constexpr double kLarge=1e12;
        // Exact squared Euclidean distance transform to obstacle cell CENTERS.
        for(int y=0;y<height;++y) {
            for(int x=0;x<width;++x) {
                const bool occupied=x==0||y==0||x==width-1||y==height-1||
                    occupancy[static_cast<std::size_t>(y)*width+x]!=0;
                f[x]=occupied?0.0:kLarge;
            }
            edt1d(f.data(),width,d.data());
            for(int x=0;x<width;++x)
                tmp[static_cast<std::size_t>(y)*width+x]=d[x];
        }
        p->distances_.resize(static_cast<std::size_t>(width)*height);
        for(int x=0;x<width;++x) {
            for(int y=0;y<height;++y)
                f[y]=tmp[static_cast<std::size_t>(y)*width+x];
            edt1d(f.data(),height,d.data());
            for(int y=0;y<height;++y)
                p->distances_[static_cast<std::size_t>(y)*width+x]=
                    std::nextafter(static_cast<float>(
                        std::sqrt(std::max(0.0,d[y]))*resolution),-INFINITY);
        }
        // One-time, 64x64 fractional grid-cell offset table. Every entry
        // stores an UPPER bound on the distance from a query inside that
        // fraction bin to each of its four neighboring cell centers.
        // The runtime never computes square roots for field interpolation.
        for (int by=0;by<kFractionBins;++by) {
            for (int bx=0;bx<kFractionBins;++bx) {
                for (int j=0;j<=1;++j)for(int i=0;i<=1;++i) {
                    const double fx0=double(bx)/kFractionBins;
                    const double fx1=double(bx+1)/kFractionBins;
                    const double fy0=double(by)/kFractionBins;
                    const double fy1=double(by+1)/kFractionBins;
                    const double rx=std::max(std::abs(fx0-i),std::abs(fx1-i));
                    const double ry=std::max(std::abs(fy0-j),std::abs(fy1-j));
                    const double upper=resolution*std::sqrt(rx*rx+ry*ry);
                    p->upper_center_offset_[
                        (by*kFractionBins+bx)*4+j*2+i]=
                        std::nextafter(static_cast<float>(upper),INFINITY);
                }
            }
        }
        return p;
    }

    // Lower bound on clearance to the entire occupied pixel square.
    // Four independent 1-Lipschitz lower bounds, no potentially optimistic
    // bilinear interpolation. Off-grid and nonfinite coordinates fail closed.
    double point_clearance_lower_bound(double x,double y) const {
        if (!std::isfinite(x)||!std::isfinite(y))return -INFINITY;
        const double dx=x-origin_x_,dy=y-origin_y_;
        const double col=(c_*dx+s_*dy)/resolution_-0.5;
        const double row=(-s_*dx+c_*dy)/resolution_-0.5;
        if (!(col>=0&&row>=0&&col<width_-1&&row<height_-1))return -INFINITY;
        const int ix=static_cast<int>(col),iy=static_cast<int>(row);
        const int bx=std::min(kFractionBins-1,
            static_cast<int>((col-ix)*kFractionBins));
        const int by=std::min(kFractionBins-1,
            static_cast<int>((row-iy)*kFractionBins));
        const float *const upper=&upper_center_offset_[
            (by*kFractionBins+bx)*4];
        const std::size_t base=static_cast<std::size_t>(iy)*width_+ix;
        // Distance-to-obstacle is 1-Lipschitz. Subtracting an UPPER bound
        // on each center-to-query offset yields a certified lower bound.
        // Unlike bilinear interpolation this can never overestimate safety.
        const double lower=std::max(
            std::max(double(distances_[base])-upper[0],
                     double(distances_[base+1])-upper[1]),
            std::max(double(distances_[base+width_])-upper[2],
                     double(distances_[base+width_+1])-upper[3]));
        // Occupied CELL extends one half diagonal beyond its center.
        return lower-0.7071067811865476*resolution_;
    }

    // Vehicle is 0.510 x 0.273m, rear-axle origin at x=0. Samples edges
    // densely and subtracts the maximum unmeasured edge-to-probe distance.
    // Direct geometric clearance to map obstacles; does not include
    // additional localization/model reserve supplied by the caller.
    double footprint_clearance_lower_bound(const MpcBodyPose &pose) const {
        if (!std::isfinite(pose.yaw))return -INFINITY;
        const double c=std::cos(pose.yaw),s=std::sin(pose.yaw);
        double minimum=INFINITY;
        // 26 and 14 intervals ensure <=~2cm point spacing.
        for(int i=0;i<=26;++i){
            const double along=-.080+.510*double(i)/26.;
            for(const double side:{-.1365,.1365}){
                const double d=point_clearance_lower_bound(
                    pose.x+c*along-s*side,pose.y+s*along+c*side);
                minimum=std::min(minimum,d);
            }
        }
        for(int i=0;i<=14;++i){
            const double side=-.1365+.273*double(i)/14.;
            for(const double along:{-.080,.430}){
                const double d=point_clearance_lower_bound(
                    pose.x+c*along-s*side,pose.y+s*along+c*side);
                minimum=std::min(minimum,d);
            }
        }
        return minimum-0.010; // edge probe interval/2 upper bound
    }

    // Conservative interpolation certificate of the *sampled source-step
    // endpoints*. This is not proof of intermediate PhysX substep dynamics.
    // Reserve additional translation/heading uncertainty separately.
    bool interpolated_sweep_passes(const MpcBodyPose &a,const MpcBodyPose &b,
        double required_clearance_m,double *min_slack=nullptr) const
    {
        const double dx=b.x-a.x,dy=b.y-a.y;
        const double dyaw=std::atan2(std::sin(b.yaw-a.yaw),
                                    std::cos(b.yaw-a.yaw));
        if (!std::isfinite(dx)||!std::isfinite(dy)||!std::isfinite(dyaw)||
            !std::isfinite(required_clearance_m)||required_clearance_m<0)
            return false;
        const double translation=std::hypot(dx,dy);
        // Bound point movement between successive interpolated rigid poses.
        const int steps=std::max(1,static_cast<int>(std::ceil(std::max(
            translation/.01,std::abs(dyaw)/.015))));
        if(steps>200)return false;
        double slack=INFINITY;
        constexpr double radius=.455;
        const double segment_bound=.5*(translation+
            radius*std::abs(dyaw))/steps;
        for(int k=0;k<=steps;++k){
            const double t=double(k)/steps;
            const MpcBodyPose pose{a.x+t*dx,a.y+t*dy,a.yaw+t*dyaw};
            // Rectangle circumscribed by radius <0.290m around its
            // geometric center (0.175m in front of the rear axle).
            // For open regions one EDT query certifies the *entire* rigid
            // footprint; otherwise retain all dense perimeter probes.
            const double c=std::cos(pose.yaw),s=std::sin(pose.yaw);
            const double center_lower=point_clearance_lower_bound(
                pose.x+0.175*c,pose.y+0.175*s)-0.290;
            const double safe_threshold=required_clearance_m+segment_bound;
            const double footprint_lower=center_lower>=safe_threshold
                ? center_lower : footprint_clearance_lower_bound(pose);
            const double d=footprint_lower-safe_threshold;
            slack=std::min(slack,d);
            if(d<0){
                if(min_slack)*min_slack=slack;
                return false;
            }
        }
        if(min_slack)*min_slack=slack;
        return true;
    }
    double resolution() const{return resolution_;}
private:
    MpcWorldFootprint()=default;
    static void edt1d(const double *f,int n,double *d) {
        std::vector<int> v(n);
        std::vector<double> z(n+1);
        int k=0;v[0]=0;z[0]=-INFINITY;z[1]=INFINITY;
        for(int q=1;q<n;++q){
            double s;
            do {
                const int vk=v[k];
                s=((f[q]+double(q)*q)-(f[vk]+double(vk)*vk))/
                    (2.0*(q-vk));
                if(s<=z[k]&&k>0)--k;
                else break;
            }while(true);
            if(s<=z[k]){k=0;v[0]=q;z[0]=-INFINITY;z[1]=INFINITY;continue;}
            ++k;v[k]=q;z[k]=s;z[k+1]=INFINITY;
        }
        k=0;
        for(int q=0;q<n;++q){
            while(z[k+1]<q)++k;
            const double dx=q-v[k];
            d[q]=dx*dx+f[v[k]];
        }
    }
    static constexpr int kFractionBins=64;
    int width_{},height_{};
    double resolution_{},origin_x_{},origin_y_{},c_{},s_{};
    std::array<float,kFractionBins*kFractionBins*4> upper_center_offset_{};
    std::vector<float> distances_;
};
} // namespace f1tenth_mpc
