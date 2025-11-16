#pragma once

// C/C++
#include <sstream>

// snap
#include <snap/eos/eos_formatter.hpp>
#include <snap/forcing/forcing_formatter.hpp>
#include <snap/recon/recon_formatter.hpp>
#include <snap/riemann/riemann_formatter.hpp>

#include "hydro.hpp"

template <>
struct fmt::formatter<snap::HydroOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::HydroOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    p.report(ss);
    ss << "Thermodynamics options:\n";
    p.thermo().report(ss);
    ss << "Const gravity forcing options:\n";
    p.grav().report(ss);
    ss << "Coriolis forcing options:\n";
    p.coriolis().report(ss);
    ss << "Viscosity forcing options:\n";
    p.visc().report(ss);
    ss << "Frictional heating options:\n";
    p.fricHeat().report(ss);
    ss << "Body heating options:\n";
    p.bodyHeat().report(ss);
    ss << "Bottom heating options:\n";
    p.botHeat().report(ss);
    ss << "Top cooling options:\n";
    p.topCool().report(ss);
    ss << "Relaxation bottom component options:\n";
    p.relaxBotComp().report(ss);
    ss << "Relaxation bottom temperature options:\n";
    p.relaxBotTemp().report(ss);
    ss << "Relaxation bottom velocity options:\n";
    p.relaxBotVelo().report(ss);
    ss << "Top sponge layer options:\n";
    p.topSpongeLyr().report(ss);
    ss << "Bottom sponge layer options:\n";
    p.botSpongeLyr().report(ss);
    ss << "Coordinate options:\n";
    p.coord().report(ss);
    ss << "Equation of state options:\n";
    p.eos().report(ss);
    ss << "Primitive projector options:\n";
    p.proj().report(ss);
    ss << "Reconstruction options (vertical):\n";
    p.recon1().report(ss);
    ss << "Reconstruction options (horizontal):\n";
    p.recon23().report(ss);
    ss << "Riemann solver options:\n";
    p.riemann().report(ss);
    ss << "Internal boundary options:\n";
    p.ib().report(ss);
    ss << "Implicit options:\n";
    p.imp().report(ss);
    ss << "Sedimentation options:\n";
    p.sed().report(ss);

    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};

template <>
struct fmt::formatter<snap::PrimitiveProjectorOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::PrimitiveProjectorOptions& p,
              FormatContext& ctx) const {
    std::stringstream ss;
    p.report(ss);
    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};
