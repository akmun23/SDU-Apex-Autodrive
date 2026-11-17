#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <Eigen/Core>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace gpu_amcl_cpp {

/** Occupancy grid, Euclidean obstacle distances and world/map conversions. */
class MapProcessor {
public:
    MapProcessor() = default;

    /** Load the occupancy grid and rebuild obstacle distances and free-cell indices. */
    void load_from_msg(const nav_msgs::msg::OccupancyGrid& msg);

    /** Obstacle distances in metres. */
    const std::vector<float>& distance_field() const {
        return distance_field_; 
    }

    /** Occupancy values: 0 free, 100 occupied, -1 unknown. */
    const std::vector<int8_t>& occupancy() const {
        return occupancy_; 
    }

    /** Flattened indices of cells with occupancy 0. */
    const std::vector<int>& free_cells() const {
        return free_cells_; 
    }

    // ---- Geometry ----
    /** Map width in cells. */
    int width()  const { 
        return width_; 
    }
    /** Map height in cells. */
    int height() const { 
        return height_; 
    }
    /** Metres per cell. */
    float resolution() const { 
        return resolution_; 
    }
    /** World-frame x coordinate of the map origin. */
    float origin_x() const { 
        return origin_x_; 
    }
    /** World-frame y coordinate of the map origin. */
    float origin_y() const { 
        return origin_y_; 
    }
    bool  is_loaded() const { 
        return loaded_; 
    }

    /** Convert world coordinates to cell indices (`mx` column, `my` row). */
    inline void world_to_map(double wx, double wy,
                             int& mx, int& my) const {
        assert(loaded_ && "world_to_map requires a loaded map");
        assert(resolution_ > 0.0f && "world_to_map requires resolution_ > 0");
        mx = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
        my = static_cast<int>(std::floor((wy - origin_y_) / resolution_));
    }

    /** Return world coordinates at the cell centre (`mx` column, `my` row). */
    inline void map_to_world(int mx, int my,
                             double& wx, double& wy) const {
        wx = origin_x_ + (mx + 0.5) * resolution_;
        wy = origin_y_ + (my + 0.5) * resolution_;
    }

    inline bool is_in_bounds(int mx, int my) const {
        return mx >= 0 && mx < width_ && my >= 0 && my < height_;
    }

    /** True for in-bounds cells with occupancy 0. */
    inline bool is_free(int mx, int my) const {
        return is_in_bounds(mx, my) &&
               occupancy_[my * width_ + mx] == 0;
    }

    /** True for in-bounds cells with occupancy >= 50. */
    inline bool is_occupied(int mx, int my) const {
        return is_in_bounds(mx, my) &&
               occupancy_[my * width_ + mx] >= 50;
    }

private:
    void compute_distance_field();
    void find_free_cells();

    std::vector<int8_t> occupancy_;
    std::vector<float>  distance_field_;
    std::vector<int>    free_cells_;  ///< flat indices of free cells.

    int   width_      = 0;
    int   height_     = 0;
    int   map_size_   = 0;
    float resolution_ = 0.0f;
    float origin_x_   = 0.0f;
    float origin_y_   = 0.0f;
    bool  loaded_     = false;
};

}  // namespace gpu_amcl_cpp
