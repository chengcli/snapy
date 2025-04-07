#pragma once

// spdlog
#include <spdlog/spdlog.h>

// fvm
#include "boundary_condition.hpp"
#include "internal_boundary.hpp"

template <>
struct fmt::formatter<canoe::InternalBoundaryOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::InternalBoundaryOptions& p, FormatContext& ctx) {
    return fmt::format_to(
        ctx.out(),
        "(nghost = {}; max_iter = {}, solid_density = {}; solid_pressure = {})",
        p.nghost(), p.max_iter(), p.solid_density(), p.solid_pressure());
  }
};

template <>
struct fmt::formatter<canoe::BoundaryFlag> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::BoundaryFlag& p, FormatContext& ctx) {
    switch (p) {
      case canoe::BoundaryFlag::kExchange:
        return fmt::format_to(ctx.out(), "Exchange");
      case canoe::BoundaryFlag::kUser:
        return fmt::format_to(ctx.out(), "User");
      case canoe::BoundaryFlag::kReflect:
        return fmt::format_to(ctx.out(), "Reflect");
      case canoe::BoundaryFlag::kOutflow:
        return fmt::format_to(ctx.out(), "Outflow");
      case canoe::BoundaryFlag::kPeriodic:
        return fmt::format_to(ctx.out(), "Periodic");
      case canoe::BoundaryFlag::kShearPeriodic:
        return fmt::format_to(ctx.out(), "ShearPeriodic");
      case canoe::BoundaryFlag::kPolar:
        return fmt::format_to(ctx.out(), "Polar");
      case canoe::BoundaryFlag::kPolarWedge:
        return fmt::format_to(ctx.out(), "PolarWedge");
      default:
        return fmt::format_to(ctx.out(), "Unknown");
    }
  }
};

template <>
struct fmt::formatter<std::vector<canoe::BoundaryFlag>> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const std::vector<canoe::BoundaryFlag>& p, FormatContext& ctx) {
    std::string result = "(";
    for (size_t i = 0; i < p.size(); ++i) {
      result += fmt::format("{}", p[i]);
      if (i < p.size() - 1) {
        result += ", ";
      }
    }
    result += ")";
    return fmt::format_to(ctx.out(), "{}", result);
  }
};

template <>
struct fmt::formatter<canoe::BoundaryFuncOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::BoundaryFuncOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(type = {}; nghost = {})", p.type(),
                          p.nghost());
  }
};
