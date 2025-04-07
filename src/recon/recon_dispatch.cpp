// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// fvm
#include "interp_simple.hpp"
#include "interpolation.hpp"

namespace canoe {
InterpOptions::InterpOptions(ParameterInput pin, std::string section,
                             std::string xorder) {
  if (pin->DoesParameterExist(section, xorder)) {
    type(pin->GetString(section, xorder));
  } else {
    type(pin->GetOrAddString(section, xorder, "dc"));
  }
}

InterpOptions::InterpOptions(std::string type_) { type(type_); }

void call_cp3_cpu(at::TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "cp3_cpu", [&] {
    at::native::cpu_kernel(
        iter, [](scalar_t in1, scalar_t in2, scalar_t in3) -> scalar_t {
          return interp_cp3(in1, in2, in3);
        });
  });
}

void call_cp5_cpu(at::TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "cp5_cpu", [&] {
    at::native::cpu_kernel(iter,
                           [](scalar_t in1, scalar_t in2, scalar_t in3,
                              scalar_t in4, scalar_t in5) -> scalar_t {
                             return interp_cp5(in1, in2, in3, in4, in5);
                           });
  });
}

void call_weno3_cpu(at::TensorIterator& iter, bool scale) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "weno3_cpu", [&] {
    at::native::cpu_kernel(
        iter, [scale](scalar_t in1, scalar_t in2, scalar_t in3) -> scalar_t {
          auto s =
              scale ? (fabs(in1) + fabs(in2) + fabs(in3)) / 3. + 1.e-10 : 1;
          return s * interp_weno3(in1 / s, in2 / s, in3 / s);
        });
  });
}

void call_weno5_cpu(at::TensorIterator& iter, bool scale) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "weno5_cpu", [&] {
    at::native::cpu_kernel(
        iter,
        [scale](scalar_t in1, scalar_t in2, scalar_t in3, scalar_t in4,
                scalar_t in5) -> scalar_t {
          auto s = scale ? (fabs(in1) + fabs(in2) + fabs(in3) + fabs(in4) +
                            fabs(in5)) /
                                   5. +
                               1.e-10
                         : 1;
          return s * interp_weno5(in1 / s, in2 / s, in3 / s, in4 / s, in5 / s);
        });
  });
}

}  // namespace canoe
