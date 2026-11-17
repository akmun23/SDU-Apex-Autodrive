#include "gpu_amcl_cpp/helpers/map_utils.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {

/** Intersection where EDT parabola `u` becomes cheaper than parabola `i`. */
inline float edt_intersection(const std::vector<float>& f, int i, int u) {
    const double fi = static_cast<double>(f[i]);
    const double fu = static_cast<double>(f[u]);
    const double ii = static_cast<double>(i) * static_cast<double>(i);
    const double uu = static_cast<double>(u) * static_cast<double>(u);
    return static_cast<float>(((fu + uu) - (fi + ii)) /
                              (2.0 * static_cast<double>(u - i)));
}

/** Felzenszwalb-Huttenlocher 1D squared-distance transform.
 * `f` contains 0 at sites and `inf_value` elsewhere; `d` receives squared
 * distances. `v` stores envelope sites, `z` boundaries, and `n` the slice length. */
void edt_1d_felzenszwalb(const std::vector<float>& f,
                         std::vector<float>& d,
                         std::vector<int>& v,
                         std::vector<float>& z,
                         int n,
                         float inf_value) {
    int first_site = -1;
    for (int i = 0; i < n; ++i) {
        if (f[i] < inf_value) {
            first_site = i;
            break;
        }
    }

    if (first_site < 0) {
        std::fill(d.begin(), d.begin() + n, inf_value);
        return;
    }

    int k = 0;
    v[0] = first_site;
    z[0] = -std::numeric_limits<float>::infinity();
    z[1] = std::numeric_limits<float>::infinity();

    for (int q = first_site + 1; q < n; ++q) {
        if (f[q] >= inf_value) {
            continue;
        }

        float s = edt_intersection(f, v[k], q);
        while (s <= z[k]) {
            --k;
            s = edt_intersection(f, v[k], q);
        }

        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = std::numeric_limits<float>::infinity();
    }

    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < static_cast<float>(q)) {
            ++k;
        }

        const float diff = static_cast<float>(q - v[k]);
        d[q] = diff * diff + f[v[k]];
    }
}

}  // namespace

namespace gpu_amcl_cpp {


void MapProcessor::load_from_msg(const nav_msgs::msg::OccupancyGrid& msg) {
    width_      = static_cast<int>(msg.info.width);
    height_     = static_cast<int>(msg.info.height);
    map_size_   = width_ * height_;
    resolution_ = static_cast<float>(msg.info.resolution);
    origin_x_   = static_cast<float>(msg.info.origin.position.x);
    origin_y_   = static_cast<float>(msg.info.origin.position.y);

    if (msg.data.size() != static_cast<size_t>(map_size_)) {
        throw std::runtime_error("OccupancyGrid data size does not match width*height");
    }

    occupancy_.resize(map_size_);
    for (int i = 0; i < map_size_; ++i) {
        occupancy_[i] = msg.data[i];
    }

    compute_distance_field();
    find_free_cells();
    loaded_ = true;
}

/** Euclidean obstacle distances in metres, using separable
 * Felzenszwalb-Huttenlocher transforms over rows then columns. */
void MapProcessor::compute_distance_field() {
    const int size = map_size_;

    if (size <= 0 || width_ <= 0 || height_ <= 0) {
        distance_field_.clear();
        return;
    }

    static constexpr float kInf = 1e20f;
    distance_field_.assign(size, kInf);

    bool has_occupied = false;
    for (int i = 0; i < size; ++i) {
        if (occupancy_[i] >= 50) {
            distance_field_[i] = 0.0f;
            has_occupied = true;
        }
    }

    if (!has_occupied) {
        return; // No occupied cells, so all distances remain at infinity.
    }

    const int max_dim = std::max(width_, height_);
    std::vector<float> f(max_dim);
    std::vector<float> d(max_dim);
    std::vector<int> v(max_dim);
    std::vector<float> z(max_dim + 1);

    // Pass 1: EDT across rows.
    for (int y = 0; y < height_; ++y) {
        const int row_offset = y * width_;             
        for (int x = 0; x < width_; ++x) {
            f[x] = distance_field_[row_offset + x];
        }

        edt_1d_felzenszwalb(f, d, v, z, width_, kInf);

        for (int x = 0; x < width_; ++x) {
            distance_field_[row_offset + x] = d[x];
        }
    }

    // Pass 2: EDT across columns.
    for (int x = 0; x < width_; ++x) {
        for (int y = 0; y < height_; ++y) {
            f[y] = distance_field_[y * width_ + x];
        }

        edt_1d_felzenszwalb(f, d, v, z, height_, kInf);

        for (int y = 0; y < height_; ++y) {
            distance_field_[y * width_ + x] = d[y];
        }
    }

    // Convert squared cell distances to metres.
    for (float& dist_sq : distance_field_) {
        dist_sq = std::sqrt(dist_sq) * resolution_;
    }
}


void MapProcessor::find_free_cells() {
    free_cells_.clear();
    const int size = map_size_;
    for (int i = 0; i < size; ++i) {
        if (occupancy_[i] == 0) {
            free_cells_.push_back(i);
        }
    }
}

}  // namespace gpu_amcl_cpp
