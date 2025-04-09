#pragma once

// fmt
#include <fmt/format.h>

// snap
#include "boundary_condition.hpp"
#include "internal_boundary.hpp"

template <>
struct fmt::formatter<snap::InternalBoundaryOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::InternalBoundaryOptions& p,
              FormatContext& ctx) const {
    return fmt::format_to(
        ctx.out(),
        "(nghost = {}; max_iter = {}, solid_density = {}; solid_pressure = {})",
        p.nghost(), p.max_iter(), p.solid_density(), p.solid_pressure());
  }
};

template <>
struct fmt::formatter<snap::BoundaryFlag> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::BoundaryFlag& p, FormatContext& ctx) const {
    switch (p) {
      case snap::BoundaryFlag::kExchange:
        return fmt::format_to(ctx.out(), "Exchange");
      case snap::BoundaryFlag::kUser:
        return fmt::format_to(ctx.out(), "User");
      case snap::BoundaryFlag::kReflect:
        return fmt::format_to(ctx.out(), "Reflect");
      case snap::BoundaryFlag::kOutflow:
        return fmt::format_to(ctx.out(), "Outflow");
      case snap::BoundaryFlag::kPeriodic:
        return fmt::format_to(ctx.out(), "Periodic");
      case snap::BoundaryFlag::kShearPeriodic:
        return fmt::format_to(ctx.out(), "ShearPeriodic");
      case snap::BoundaryFlag::kPolar:
        return fmt::format_to(ctx.out(), "Polar");
      case snap::BoundaryFlag::kPolarWedge:
        return fmt::format_to(ctx.out(), "PolarWedge");
      default:
        return fmt::format_to(ctx.out(), "Unknown");
    }
  }
};

template <>
struct fmt::formatter<std::vector<snap::BoundaryFlag>> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const std::vector<snap::BoundaryFlag>& p,
              FormatContext& ctx) const {
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
struct fmt::formatter<snap::BoundaryFuncOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::BoundaryFuncOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "(type = {}; nghost = {})", p.type(),
                          p.nghost());
  }
};
