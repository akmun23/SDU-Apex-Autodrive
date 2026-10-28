#ifndef SDU_APEX_MPC_QP_FIXTURE_FORMAT_HPP
#define SDU_APEX_MPC_QP_FIXTURE_FORMAT_HPP

#include "mpc_rti.h"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

namespace sdu_apex::qp_fixture {

constexpr std::uint32_t kSchemaVersion = 4;
constexpr char kMagic[8] = {'S', 'D', 'U', 'F', 'Q', 'P', '4', '\n'};

class Writer {
 public:
  explicit Writer(const std::string &path)
      : stream_(path, std::ios::binary | std::ios::trunc) {}

  bool good() const { return stream_.good(); }
  bool bytes(const void *data, std::size_t size) {
    stream_.write(static_cast<const char *>(data),
                  static_cast<std::streamsize>(size));
    return stream_.good();
  }
  bool u32(std::uint32_t value) {
    const unsigned char data[4] = {
        static_cast<unsigned char>(value),
        static_cast<unsigned char>(value >> 8),
        static_cast<unsigned char>(value >> 16),
        static_cast<unsigned char>(value >> 24)};
    return bytes(data, sizeof(data));
  }
  bool i32(std::int32_t value) {
    return u32(static_cast<std::uint32_t>(value));
  }
  bool u64(std::uint64_t value) {
    const unsigned char data[8] = {
        static_cast<unsigned char>(value),
        static_cast<unsigned char>(value >> 8),
        static_cast<unsigned char>(value >> 16),
        static_cast<unsigned char>(value >> 24),
        static_cast<unsigned char>(value >> 32),
        static_cast<unsigned char>(value >> 40),
        static_cast<unsigned char>(value >> 48),
        static_cast<unsigned char>(value >> 56)};
    return bytes(data, sizeof(data));
  }
  bool f32(float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t),
                  "fixture requires IEEE-754 binary32");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return u32(bits);
  }
  bool f64(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "fixture requires IEEE-754 binary64");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return u64(bits);
  }

 private:
  std::ofstream stream_;
};

class Reader {
 public:
  explicit Reader(const std::string &path)
      : stream_(path, std::ios::binary) {}

  bool good() const { return stream_.good(); }
  bool bytes(void *data, std::size_t size) {
    stream_.read(static_cast<char *>(data),
                 static_cast<std::streamsize>(size));
    return stream_.good();
  }
  bool u32(std::uint32_t *value) {
    unsigned char data[4]{};
    if (!bytes(data, sizeof(data))) return false;
    *value = static_cast<std::uint32_t>(data[0]) |
        (static_cast<std::uint32_t>(data[1]) << 8) |
        (static_cast<std::uint32_t>(data[2]) << 16) |
        (static_cast<std::uint32_t>(data[3]) << 24);
    return true;
  }
  bool i32(std::int32_t *value) {
    std::uint32_t bits = 0;
    if (!u32(&bits)) return false;
    *value = static_cast<std::int32_t>(bits);
    return true;
  }
  bool u64(std::uint64_t *value) {
    unsigned char data[8]{};
    if (!bytes(data, sizeof(data))) return false;
    *value = 0;
    for (int i = 0; i < 8; ++i)
      *value |= static_cast<std::uint64_t>(data[i]) << (8 * i);
    return true;
  }
  bool f32(float *value) {
    std::uint32_t bits = 0;
    if (!u32(&bits)) return false;
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }
  bool f64(double *value) {
    std::uint64_t bits = 0;
    if (!u64(&bits)) return false;
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }
  bool at_eof() {
    char extra = 0;
    stream_.read(&extra, 1);
    return stream_.eof();
  }

 private:
  std::ifstream stream_;
};

inline bool write_f32_array(Writer &writer, const float *values,
                            std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    if (!writer.f32(values[i])) return false;
  return true;
}

inline bool write_capture(const std::string &path,
                          const MpcRtiQpCapture_t &capture) {
  static_assert(RICCATI_MAX_NX == 23, "unexpected MPC QP state ABI");
  static_assert(RICCATI_MAX_NU == 2, "unexpected MPC QP control ABI");
  if (!capture.valid || capture.problem.horizon != RICCATI_HORIZON)
    return false;

  Writer writer(path);
  if (!writer.good() || !writer.bytes(kMagic, sizeof(kMagic)) ||
      !writer.u32(kSchemaVersion) || !writer.u32(0x01020304u) ||
      !writer.u32(RICCATI_MAX_NX) || !writer.u32(RICCATI_MAX_NU) ||
      !writer.u32(static_cast<std::uint32_t>(capture.problem.horizon)) ||
      !writer.u32(static_cast<std::uint32_t>(capture.capture_count)) ||
      !writer.i32(static_cast<std::int32_t>(capture.raw_status)))
    return false;

  const auto &problem = capture.problem;
  for (int k = 0; k < problem.horizon; ++k) {
    const auto &step = problem.steps[k];
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      if (!write_f32_array(writer, step.A[i], RICCATI_MAX_NX)) return false;
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      if (!write_f32_array(writer, step.B[i], RICCATI_MAX_NU)) return false;
    if (!write_f32_array(writer, step.d, RICCATI_MAX_NX) ||
        !write_f32_array(writer, step.Q_diag, RICCATI_MAX_NX) ||
        !write_f32_array(writer, step.q, RICCATI_MAX_NX)) return false;
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      if (!write_f32_array(writer, step.N[i], RICCATI_MAX_NU)) return false;
    if (!write_f32_array(writer, step.R_diag, RICCATI_MAX_NU) ||
        !write_f32_array(writer, step.r, RICCATI_MAX_NU) ||
        !write_f32_array(writer, step.x_lb, RICCATI_MAX_NX) ||
        !write_f32_array(writer, step.x_ub, RICCATI_MAX_NX) ||
        !write_f32_array(writer, step.u_lb, RICCATI_MAX_NU) ||
        !write_f32_array(writer, step.u_ub, RICCATI_MAX_NU)) return false;
  }
  if (!write_f32_array(writer, problem.terminal_Q, RICCATI_MAX_NX) ||
      !write_f32_array(writer, problem.terminal_q, RICCATI_MAX_NX) ||
      !write_f32_array(writer, problem.terminal_x_lb, RICCATI_MAX_NX) ||
      !write_f32_array(writer, problem.terminal_x_ub, RICCATI_MAX_NX) ||
      !write_f32_array(writer, problem.x0, RICCATI_MAX_NX)) return false;

  const auto &config = capture.solver_config;
  for (const auto &polygon : config.state_polygons) {
    if (!writer.i32(polygon.count)) return false;
    for (int face = 0; face < RICCATI_STATE_POLYGON_MAX_PLANES; ++face)
      if (!write_f32_array(writer, polygon.normal[face], 2) ||
          !writer.f32(polygon.upper[face])) return false;
  }
  if (!writer.f32(config.rho) || !writer.f32(config.rho_u) ||
      !write_f32_array(writer, config.rho_x_by_state, RICCATI_MAX_NX) ||
      !write_f32_array(writer, config.rho_u_by_input, RICCATI_MAX_NU) ||
      !writer.f32(config.rho_max) || !writer.f32(config.tolerance) ||
      !writer.i32(config.max_iterations) ||
      !writer.i32(config.adaptive_rho) ||
      !writer.i32(config.cold_start_each_solve) ||
      !writer.i32(config.reuse_same_epoch) ||
      !writer.i32(config.adaptive_rho_update_period) ||
      !writer.i32(config.shared_rho) || !writer.f32(config.over_relaxation) ||
      !writer.f64(config.max_solve_time_us) ||
      !writer.i32(config.use_scaling) ||
      !write_f32_array(writer, config.state_scale, RICCATI_MAX_NX) ||
      !write_f32_array(writer, config.input_scale, RICCATI_MAX_NU)) return false;

  const auto &state = capture.initial_solver_state;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, state.z_x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, state.z_u[k], RICCATI_MAX_NU)) return false;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, state.y_x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, state.y_u[k], RICCATI_MAX_NU)) return false;
  if (!writer.f32(state.rho) || !writer.f32(state.rho_u)) return false;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, state.rho_state[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, state.rho_input[k], RICCATI_MAX_NU)) return false;
  if (!writer.i32(state.initialized) || !writer.i32(state.scaling_enabled) ||
      !write_f32_array(writer, state.state_scale, RICCATI_MAX_NX) ||
      !write_f32_array(writer, state.input_scale, RICCATI_MAX_NU)) return false;

  const auto &final_state = capture.final_solver_state;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, final_state.z_x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, final_state.z_u[k], RICCATI_MAX_NU)) return false;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, final_state.y_x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, final_state.y_u[k], RICCATI_MAX_NU)) return false;
  if (!writer.f32(final_state.rho) || !writer.f32(final_state.rho_u)) return false;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, final_state.rho_state[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, final_state.rho_input[k], RICCATI_MAX_NU)) return false;
  if (!writer.i32(final_state.initialized) ||
      !writer.i32(final_state.scaling_enabled) ||
      !write_f32_array(writer, final_state.state_scale, RICCATI_MAX_NX) ||
      !write_f32_array(writer, final_state.input_scale, RICCATI_MAX_NU)) return false;

  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, capture.raw_primal_u[k],
                         RICCATI_MAX_NU)) return false;

  const auto &solution = capture.raw_solution;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, solution.x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!write_f32_array(writer, solution.u[k], RICCATI_MAX_NU)) return false;
  if (!writer.i32(solution.iterations) ||
      !writer.i32(solution.time_budget_exhausted) ||
      !writer.f32(solution.primal_residual) ||
      !writer.f32(solution.dual_residual) ||
      !writer.f32(solution.state_primal_residual) ||
      !writer.f32(solution.state_dual_residual) ||
      !writer.f32(solution.input_primal_residual) ||
      !writer.f32(solution.input_dual_residual) ||
      !writer.i32(solution.state_primal_stage) ||
      !writer.i32(solution.state_primal_index) ||
      !writer.i32(solution.state_dual_stage) ||
      !writer.i32(solution.state_dual_index) ||
      !writer.i32(solution.input_primal_stage) ||
      !writer.i32(solution.input_primal_index) ||
      !writer.i32(solution.input_dual_stage) ||
      !writer.i32(solution.input_dual_index) ||
      !writer.i32(static_cast<std::int32_t>(solution.status))) return false;
  const auto &debug = capture.debug_info;
  if (!writer.f32(debug.rho_start) || !writer.f32(debug.rho_u_start) ||
      !writer.f32(debug.rho) || !writer.f32(debug.rho_u) ||
      !writer.i32(debug.rho_change_count) ||
      !writer.f32(debug.primal_residual) ||
      !writer.f32(debug.dual_residual) ||
      !writer.f32(debug.last_invert_det) ||
      !writer.f32(debug.last_fallback_s00) ||
      !writer.f32(debug.last_fallback_s11) ||
      !writer.i32(debug.invert_fallback_count) ||
      !writer.f32(debug.max_control_hessian_regularization) ||
      !writer.i32(debug.control_hessian_regularization_count) ||
      !writer.i32(debug.quadratic_factorization_count) ||
      !writer.u64(debug.quadratic_factorization_time_ns) ||
      !writer.u64(debug.quadratic_rhs_time_ns) ||
      !writer.u64(debug.projection_residual_time_ns) ||
      !writer.u64(debug.cold_start_initialization_time_ns) ||
      !writer.u64(debug.rho_update_time_ns) ||
      !writer.u64(debug.scaling_transform_time_ns) ||
      !writer.u64(debug.output_unscale_time_ns)) return false;
  return writer.good();
}

inline bool read_f32_array(Reader &reader, float *values,
                           std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    if (!reader.f32(&values[i])) return false;
  return true;
}

inline bool read_capture(const std::string &path,
                         MpcRtiQpCapture_t *capture) {
  if (!capture) return false;
  *capture = MpcRtiQpCapture_t{};
  Reader reader(path);
  char magic[sizeof(kMagic)]{};
  std::uint32_t version = 0, endian = 0, nx = 0, nu = 0, horizon = 0;
  std::uint32_t passes = 0;
  std::int32_t raw_status = 0;
  if (!reader.good() || !reader.bytes(magic, sizeof(magic)) ||
      std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
      !reader.u32(&version) || !reader.u32(&endian) || !reader.u32(&nx) ||
      !reader.u32(&nu) || !reader.u32(&horizon) || !reader.u32(&passes) ||
      !reader.i32(&raw_status) || version != kSchemaVersion ||
      endian != 0x01020304u || nx != RICCATI_MAX_NX ||
      nu != RICCATI_MAX_NU || horizon != RICCATI_HORIZON || passes == 0)
    return false;

  auto &problem = capture->problem;
  problem.horizon = static_cast<int>(horizon);
  for (int k = 0; k < problem.horizon; ++k) {
    auto &step = problem.steps[k];
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      if (!read_f32_array(reader, step.A[i], RICCATI_MAX_NX)) return false;
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      if (!read_f32_array(reader, step.B[i], RICCATI_MAX_NU)) return false;
    if (!read_f32_array(reader, step.d, RICCATI_MAX_NX) ||
        !read_f32_array(reader, step.Q_diag, RICCATI_MAX_NX) ||
        !read_f32_array(reader, step.q, RICCATI_MAX_NX)) return false;
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      if (!read_f32_array(reader, step.N[i], RICCATI_MAX_NU)) return false;
    if (!read_f32_array(reader, step.R_diag, RICCATI_MAX_NU) ||
        !read_f32_array(reader, step.r, RICCATI_MAX_NU) ||
        !read_f32_array(reader, step.x_lb, RICCATI_MAX_NX) ||
        !read_f32_array(reader, step.x_ub, RICCATI_MAX_NX) ||
        !read_f32_array(reader, step.u_lb, RICCATI_MAX_NU) ||
        !read_f32_array(reader, step.u_ub, RICCATI_MAX_NU)) return false;
  }
  if (!read_f32_array(reader, problem.terminal_Q, RICCATI_MAX_NX) ||
      !read_f32_array(reader, problem.terminal_q, RICCATI_MAX_NX) ||
      !read_f32_array(reader, problem.terminal_x_lb, RICCATI_MAX_NX) ||
      !read_f32_array(reader, problem.terminal_x_ub, RICCATI_MAX_NX) ||
      !read_f32_array(reader, problem.x0, RICCATI_MAX_NX)) return false;

  auto &config = capture->solver_config;
  for (int k = 0; k <= RICCATI_HORIZON; ++k) {
    auto &polygon = config.state_polygons[k];
    if (!reader.i32(&polygon.count) || polygon.count < 0 ||
        polygon.count > RICCATI_STATE_POLYGON_MAX_PLANES) return false;
    for (int face = 0; face < RICCATI_STATE_POLYGON_MAX_PLANES; ++face) {
      if (!read_f32_array(reader, polygon.normal[face], 2) ||
          !reader.f32(&polygon.upper[face])) return false;
      if (!std::isfinite(polygon.normal[face][0]) ||
          !std::isfinite(polygon.normal[face][1]) ||
          !std::isfinite(polygon.upper[face])) return false;
      if (face < polygon.count &&
          !(std::hypot(polygon.normal[face][0], polygon.normal[face][1]) > 0.f)) return false;
    }
    problem.state_polygons[k] = polygon;
  }
  if (!reader.f32(&config.rho) || !reader.f32(&config.rho_u) ||
      !read_f32_array(reader, config.rho_x_by_state, RICCATI_MAX_NX) ||
      !read_f32_array(reader, config.rho_u_by_input, RICCATI_MAX_NU) ||
      !reader.f32(&config.rho_max) || !reader.f32(&config.tolerance) ||
      !reader.i32(&config.max_iterations) ||
      !reader.i32(&config.adaptive_rho) ||
      !reader.i32(&config.cold_start_each_solve) ||
      !reader.i32(&config.reuse_same_epoch) ||
      !reader.i32(&config.adaptive_rho_update_period) ||
      !reader.i32(&config.shared_rho) ||
      !reader.f32(&config.over_relaxation) ||
      !reader.f64(&config.max_solve_time_us) ||
      !reader.i32(&config.use_scaling) ||
      !read_f32_array(reader, config.state_scale, RICCATI_MAX_NX) ||
      !read_f32_array(reader, config.input_scale, RICCATI_MAX_NU)) return false;

  auto &state = capture->initial_solver_state;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, state.z_x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, state.z_u[k], RICCATI_MAX_NU)) return false;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, state.y_x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, state.y_u[k], RICCATI_MAX_NU)) return false;
  if (!reader.f32(&state.rho) || !reader.f32(&state.rho_u)) return false;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, state.rho_state[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, state.rho_input[k], RICCATI_MAX_NU)) return false;
  if (!reader.i32(&state.initialized) || !reader.i32(&state.scaling_enabled) ||
      !read_f32_array(reader, state.state_scale, RICCATI_MAX_NX) ||
      !read_f32_array(reader, state.input_scale, RICCATI_MAX_NU)) return false;

  auto &final_state = capture->final_solver_state;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, final_state.z_x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, final_state.z_u[k], RICCATI_MAX_NU)) return false;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, final_state.y_x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, final_state.y_u[k], RICCATI_MAX_NU)) return false;
  if (!reader.f32(&final_state.rho) || !reader.f32(&final_state.rho_u)) return false;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, final_state.rho_state[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, final_state.rho_input[k], RICCATI_MAX_NU)) return false;
  if (!reader.i32(&final_state.initialized) ||
      !reader.i32(&final_state.scaling_enabled) ||
      !read_f32_array(reader, final_state.state_scale, RICCATI_MAX_NX) ||
      !read_f32_array(reader, final_state.input_scale, RICCATI_MAX_NU)) return false;

  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, capture->raw_primal_u[k],
                        RICCATI_MAX_NU)) return false;

  auto &solution = capture->raw_solution;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, solution.x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!read_f32_array(reader, solution.u[k], RICCATI_MAX_NU)) return false;
  std::int32_t solution_status = 0;
  if (!reader.i32(&solution.iterations) ||
      !reader.i32(&solution.time_budget_exhausted) ||
      !reader.f32(&solution.primal_residual) ||
      !reader.f32(&solution.dual_residual) ||
      !reader.f32(&solution.state_primal_residual) ||
      !reader.f32(&solution.state_dual_residual) ||
      !reader.f32(&solution.input_primal_residual) ||
      !reader.f32(&solution.input_dual_residual) ||
      !reader.i32(&solution.state_primal_stage) ||
      !reader.i32(&solution.state_primal_index) ||
      !reader.i32(&solution.state_dual_stage) ||
      !reader.i32(&solution.state_dual_index) ||
      !reader.i32(&solution.input_primal_stage) ||
      !reader.i32(&solution.input_primal_index) ||
      !reader.i32(&solution.input_dual_stage) ||
      !reader.i32(&solution.input_dual_index) ||
      !reader.i32(&solution_status)) return false;

  auto &debug = capture->debug_info;
  if (!reader.f32(&debug.rho_start) || !reader.f32(&debug.rho_u_start) ||
      !reader.f32(&debug.rho) || !reader.f32(&debug.rho_u) ||
      !reader.i32(&debug.rho_change_count) ||
      !reader.f32(&debug.primal_residual) ||
      !reader.f32(&debug.dual_residual) ||
      !reader.f32(&debug.last_invert_det) ||
      !reader.f32(&debug.last_fallback_s00) ||
      !reader.f32(&debug.last_fallback_s11) ||
      !reader.i32(&debug.invert_fallback_count) ||
      !reader.f32(&debug.max_control_hessian_regularization) ||
      !reader.i32(&debug.control_hessian_regularization_count) ||
      !reader.i32(&debug.quadratic_factorization_count) ||
      !reader.u64(&debug.quadratic_factorization_time_ns) ||
      !reader.u64(&debug.quadratic_rhs_time_ns) ||
      !reader.u64(&debug.projection_residual_time_ns) ||
      !reader.u64(&debug.cold_start_initialization_time_ns) ||
      !reader.u64(&debug.rho_update_time_ns) ||
      !reader.u64(&debug.scaling_transform_time_ns) ||
      !reader.u64(&debug.output_unscale_time_ns) || !reader.at_eof())
    return false;

  solution.status = static_cast<RiccatiStatus_t>(solution_status);
  capture->raw_status = static_cast<RiccatiStatus_t>(raw_status);
  capture->capture_count = static_cast<int>(passes);
  capture->valid = 1;

  const auto finite_values = [](const float *values, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
      if (!std::isfinite(values[i])) return false;
    return true;
  };
  for (int k = 0; k < problem.horizon; ++k) {
    const auto &step = problem.steps[k];
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      if (!finite_values(step.A[i], RICCATI_MAX_NX) ||
          !finite_values(step.B[i], RICCATI_MAX_NU) ||
          !finite_values(step.N[i], RICCATI_MAX_NU)) return false;
    if (!finite_values(step.d, RICCATI_MAX_NX) ||
        !finite_values(step.Q_diag, RICCATI_MAX_NX) ||
        !finite_values(step.q, RICCATI_MAX_NX) ||
        !finite_values(step.R_diag, RICCATI_MAX_NU) ||
        !finite_values(step.r, RICCATI_MAX_NU)) return false;
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      if (!std::isfinite(step.x_lb[i]) || !std::isfinite(step.x_ub[i]) ||
          step.x_lb[i] > step.x_ub[i]) return false;
    for (int i = 0; i < RICCATI_MAX_NU; ++i)
      if (!std::isfinite(step.u_lb[i]) || !std::isfinite(step.u_ub[i]) ||
          step.u_lb[i] > step.u_ub[i]) return false;
  }
  for (int i = 0; i < RICCATI_MAX_NX; ++i)
    if (!std::isfinite(problem.terminal_x_lb[i]) ||
        !std::isfinite(problem.terminal_x_ub[i]) ||
        problem.terminal_x_lb[i] > problem.terminal_x_ub[i]) return false;

  const auto finite_solver_config =
      finite_values(&config.rho, 1) && finite_values(&config.rho_u, 1) &&
      finite_values(config.rho_x_by_state, RICCATI_MAX_NX) &&
      finite_values(config.rho_u_by_input, RICCATI_MAX_NU) &&
      finite_values(&config.rho_max, 1) &&
      finite_values(&config.tolerance, 1) &&
      finite_values(&config.over_relaxation, 1) &&
      std::isfinite(config.max_solve_time_us) &&
      finite_values(config.state_scale, RICCATI_MAX_NX) &&
      finite_values(config.input_scale, RICCATI_MAX_NU);
  if (!finite_solver_config || !(config.rho > 0.0f) ||
      !(config.rho_u > 0.0f) || config.rho_max < 1.0f ||
      config.rho_max > 127.0f || !(config.tolerance > 0.0f) ||
      config.max_iterations < 1 || config.max_iterations > 200 ||
      config.adaptive_rho < 0 || config.adaptive_rho > 1 ||
      config.cold_start_each_solve < 0 || config.cold_start_each_solve > 1 ||
      config.reuse_same_epoch < 0 || config.reuse_same_epoch > 1 ||
      config.shared_rho < 0 || config.shared_rho > 1 ||
      config.adaptive_rho_update_period < 1 ||
      config.over_relaxation < 1.0f || config.over_relaxation > 2.0f ||
      config.use_scaling < 0 || config.use_scaling > 1 ||
      config.max_solve_time_us < 0.0) return false;
  for (int i = 0; i < RICCATI_MAX_NX; ++i)
    if (config.rho_x_by_state[i] < 0.0f ||
        config.rho_x_by_state[i] > 127.0f ||
        (config.rho_x_by_state[i] > 0.0f &&
         config.rho_x_by_state[i] < 1.0f) ||
        !(config.state_scale[i] > 0.0f)) return false;
  for (int i = 0; i < RICCATI_MAX_NU; ++i)
    if (config.rho_u_by_input[i] < 0.0f ||
        config.rho_u_by_input[i] > 127.0f ||
        !(config.input_scale[i] > 0.0f)) return false;
  if (!finite_values(state.state_scale, RICCATI_MAX_NX) ||
      !finite_values(state.input_scale, RICCATI_MAX_NU) ||
      !finite_values(final_state.state_scale, RICCATI_MAX_NX) ||
      !finite_values(final_state.input_scale, RICCATI_MAX_NU) ||
      !std::isfinite(state.rho) || !std::isfinite(state.rho_u) ||
      !std::isfinite(final_state.rho) || !std::isfinite(final_state.rho_u))
    return false;
  if ((state.initialized != 0 && state.initialized != 1) ||
      (state.scaling_enabled != 0 && state.scaling_enabled != 1) ||
      (final_state.initialized != 0 && final_state.initialized != 1) ||
      (final_state.scaling_enabled != 0 && final_state.scaling_enabled != 1))
    return false;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!finite_values(state.z_x[k], RICCATI_MAX_NX) ||
        !finite_values(state.y_x[k], RICCATI_MAX_NX) ||
        !finite_values(state.rho_state[k], RICCATI_MAX_NX) ||
        !finite_values(final_state.z_x[k], RICCATI_MAX_NX) ||
        !finite_values(final_state.y_x[k], RICCATI_MAX_NX) ||
        !finite_values(final_state.rho_state[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!finite_values(state.z_u[k], RICCATI_MAX_NU) ||
        !finite_values(state.y_u[k], RICCATI_MAX_NU) ||
        !finite_values(state.rho_input[k], RICCATI_MAX_NU) ||
        !finite_values(final_state.z_u[k], RICCATI_MAX_NU) ||
        !finite_values(final_state.y_u[k], RICCATI_MAX_NU) ||
        !finite_values(final_state.rho_input[k], RICCATI_MAX_NU) ||
        !finite_values(capture->raw_primal_u[k], RICCATI_MAX_NU)) return false;
  if (!finite_values(problem.x0, RICCATI_MAX_NX) ||
      !finite_values(problem.terminal_Q, RICCATI_MAX_NX) ||
      !finite_values(problem.terminal_q, RICCATI_MAX_NX) ||
      !finite_values(solution.x[0], RICCATI_MAX_NX) ||
      !finite_values(solution.u[0], RICCATI_MAX_NU)) return false;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    if (!finite_values(solution.x[k], RICCATI_MAX_NX)) return false;
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    if (!finite_values(solution.u[k], RICCATI_MAX_NU)) return false;
  const auto &debug_values = capture->debug_info;
  if (!std::isfinite(debug_values.rho_start) ||
      !std::isfinite(debug_values.rho_u_start) ||
      !std::isfinite(debug_values.rho) ||
      !std::isfinite(debug_values.rho_u) ||
      !std::isfinite(debug_values.primal_residual) ||
      !std::isfinite(debug_values.dual_residual) ||
      !std::isfinite(debug_values.last_invert_det) ||
      !std::isfinite(debug_values.last_fallback_s00) ||
      !std::isfinite(debug_values.last_fallback_s11) ||
      !std::isfinite(debug_values.max_control_hessian_regularization) ||
      static_cast<int>(capture->raw_status) < RICCATI_STATUS_OPTIMAL ||
      static_cast<int>(capture->raw_status) > RICCATI_STATUS_ERROR ||
      static_cast<int>(solution.status) < RICCATI_STATUS_OPTIMAL ||
      static_cast<int>(solution.status) > RICCATI_STATUS_ERROR) return false;

  return true;
}

}  // namespace sdu_apex::qp_fixture

#endif
