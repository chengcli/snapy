#pragma once

// spdlog
#include <spdlog/spdlog.h>

// fvm
#include <fvm/eos/eos_formatter.hpp>
#include <fvm/recon/recon_formatter.hpp>
#include <fvm/riemann/riemann_formatter.hpp>
#include <fvm/thermo/thermo_formatter.hpp>

#include "hydro.hpp"

template <>
struct fmt::formatter<snap::HydroOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::HydroOptions& p, FormatContext& ctx) {
    return fmt::format_to(
        ctx.out(),
        "(nghost = {}; eos = {}; riemann = {}; recon1 = {}; recon23 = {})",
        p.nghost(), p.eos(), p.riemann(), p.recon1(), p.recon23());
  }
};

template <>
struct fmt::formatter<snap::PrimitiveProjectorOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::PrimitiveProjectorOptions& p, FormatContext& ctx) {
    return fmt::format_to(
        ctx.out(),
        "(type = {}; grav = {}; nghost = {}; margin = {}; thermo = {})",
        p.type(), p.grav(), p.nghost(), p.margin(), p.thermo());
  }
};
