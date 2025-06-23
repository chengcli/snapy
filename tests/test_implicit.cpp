// C/C++
#include <fstream>
#include <iostream>

// external
#include <gtest/gtest.h>

// base
#include <globals.h>

// fvm
#include <fvm/index.h>

#include <fvm/implicit/vertical_implicit.hpp>

// tests
#include "device_testing.hpp"

using namespace canoe;

template <typename T, int N>
torch::Tensor mass_matrix(float dt, torch::IntArrayRef shape,
                          torch::TensorOptions options) {
  torch::Tensor out = torch::zeros(shape, options);
  torch::TensorIteratorConfig iter_config;
  iter_config.add_output(out);
  auto iter = iter_config.build();

  auto set_identity = [dt](char **data, const int64_t *strides, int64_t n0,
                           int64_t n1) {
    auto *out = data[0];

    for (int64_t j = 0; j < n0; j++)
      for (int64_t i = 0; i < n1; i++) {
        for (int k = 0; k < N; k++) {
          reinterpret_cast<T *>(out)[k * N + k] = 1. / dt;
        }
        out += strides[0];
      }
  };

  iter.for_each(set_identity);

  return out;
}

template <typename T, int N>
torch::Tensor forcing_matrix(float grav, torch::IntArrayRef shape,
                             torch::TensorOptions options) {
  torch::Tensor out = torch::zeros(shape, options);
  torch::TensorIteratorConfig iter_config;
  iter_config.add_output(out);
  auto iter = iter_config.build();

  auto set_forcing = [grav](char **data, const int64_t *strides, int64_t n0,
                            int64_t n1) {
    auto *out = data[0];

    for (int64_t j = 0; j < n0; j++)
      for (int64_t i = 0; i < n1; i++) {
        reinterpret_cast<T *>(out)[index::IVX * N + index::IDN] = grav;
        reinterpret_cast<T *>(out)[index::IPR * N + index::IVX] = grav;
        out += strides[0];
      }
  };

  iter.for_each(set_forcing);

  return out;
}

TEST_P(DeviceTest, test_roe_average) {
  auto w = torch::randn({5, 1, 1, 10}, torch::device(device).dtype(dtype));
  w[index::IDN] = torch::abs(w[index::IDN]);
  w[index::IPR] = torch::abs(w[index::IPR]);

  ReconstructOptions op;
  auto precon = Reconstruct(op);
  auto wlr = precon->forward(w, 3);

  auto gm1 = torch::ones_like(w[index::IDN]) * 0.4;

  // std::cout << "w = " << w << std::endl;
  auto y = roe_average(wlr, gm1);
  // std::cout << "y = " << y << std::endl;
}

TEST_P(DeviceTest, test_flux_jacobian) {
  auto w = torch::randn({5, 1, 1, 10}, torch::device(device).dtype(dtype));
  w[index::IDN] = torch::abs(w[index::IDN]);
  w[index::IPR] = torch::abs(w[index::IPR]);

  auto gm1 = torch::ones_like(w[index::IDN]) * 0.4;
  ReconstructOptions op;
  auto precon = Reconstruct(op);
  auto wlr = precon->forward(w, 3);
  auto wroe = roe_average(wlr, gm1);

  auto y = flux_jacobian(wroe, gm1);
  std::cout << "y = " << y.sizes() << std::endl;
}

TEST_P(DeviceTest, test_eigen_value) {
  auto w = torch::randn({5, 1, 1, 10}, torch::device(device).dtype(dtype));
  w[index::IDN] = torch::abs(w[index::IDN]);
  w[index::IPR] = torch::abs(w[index::IPR]);

  EquationOfStateOptions op;
  auto peos = IdealGas(op);

  auto cs = peos->sound_speed(w);
  std::cout << "cs = " << cs << std::endl;

  // auto y = eigen_value(w[index::IVX], cs);
  // std::cout << "y = " << y.sizes() << std::endl;
}

TEST_P(DeviceTest, test_eigen_vectors) {
  auto w = torch::randn({5, 1, 1, 10}, torch::device(device).dtype(dtype));
  w[index::IDN] = torch::abs(w[index::IDN]);
  w[index::IPR] = torch::abs(w[index::IPR]);

  auto gm1 = torch::ones_like(w[index::IDN]) * 0.4;

  EquationOfStateOptions op;
  auto peos = IdealGas(op);

  auto cs = peos->sound_speed(w);

  auto y = eigen_vectors(w, gm1, cs);
  std::cout << "y = " << y.first.sizes() << std::endl;
  std::cout << "y = " << y.second.sizes() << std::endl;
}

TEST_P(DeviceTest, test_diffusion_matrix) {
  auto w = torch::randn({5, 160, 160, 40}, torch::device(device).dtype(dtype));
  w[index::IDN] = torch::abs(w[index::IDN]);
  w[index::IPR] = torch::abs(w[index::IPR]);

  auto du = torch::randn({5, 160, 160, 40}, torch::device(device).dtype(dtype));

  auto gm1 = torch::ones_like(w[index::IDN]) * 0.4;

  auto op = VerticalImplicitOptions();
  auto pvic = VerticalImplicit(op);

  auto start = std::chrono::high_resolution_clock::now();

  auto result = pvic->diffusion_matrix(w, gm1);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by diffusion matrix: " << elapsed.count()
            << " seconds" << std::endl;

  std::cout << "result = " << result.sizes() << std::endl;
}

TEST_P(DeviceTest, test_mass_matrix) {
  auto w = torch::randn({5, 160, 160, 40}, torch::device(device).dtype(dtype));

  torch::Tensor Phi = torch::zeros({5, 5}, w.options());
  Phi[index::IVX][index::IDN] = -9.8;
  Phi[index::IPR][index::IVX] = -9.8;
  Phi = Phi.view({1, 1, 5, 5});

  std::cout << "Phi = " << Phi << std::endl;
}

TEST_P(DeviceTest, test_foward) {
  auto w = torch::randn({5, 160, 160, 40}, torch::device(device).dtype(dtype));
  w[index::IDN] = torch::abs(w[index::IDN]);
  w[index::IPR] = torch::abs(w[index::IPR]);

  auto du = torch::randn({5, 160, 160, 40}, torch::device(device).dtype(dtype));

  auto gm1 = torch::ones_like(w[index::IDN]) * 0.4;

  auto op = VerticalImplicitOptions().nghost(2);
  op.coord().nx1(40).nx2(160).nx3(160);
  auto pvic = VerticalImplicit(op);
  pvic->to(device, dtype);

  auto start = std::chrono::high_resolution_clock::now();

  auto result = pvic->forward(w, du, gm1, 0.1);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by forward: " << elapsed.count() << " seconds"
            << std::endl;

  std::cout << "result = " << result.sizes() << std::endl;
}

int main(int argc, char **argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  // Get the number of threads to verify
  int intra_op_threads = torch::get_num_threads();
  int inter_op_threads = torch::get_num_interop_threads();

  std::cout << "Intra-op parallelism: " << intra_op_threads << std::endl;
  std::cout << "Inter-op parallelism: " << inter_op_threads << std::endl;

  testing::InitGoogleTest(&argc, argv);
  start_logging(argc, argv);

  return RUN_ALL_TESTS();
}
