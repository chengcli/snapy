#pragma once

// fmt
#include <fmt/format.h>

//! \brief fmt::formatter specialization for std::vector
//!
//! Provides custom formatting for std::vector types using the fmt library.
//! Formats vectors as comma-separated values enclosed in square brackets.
//!
//! \tparam A Element type of the vector
template <typename A>
struct fmt::formatter<std::vector<A>> {
  //! \brief Parse format specification
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  //! \brief Format vector to output
  //! \tparam FormatContext Context type for formatting
  //! \param[in] vec Vector to format
  //! \param[in,out] ctx Format context
  //! \return Iterator to output
  template <typename FormatContext>
  auto format(const std::vector<A>& vec, FormatContext& ctx) const {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      result += fmt::format("{}", vec[i]);
      if (i < vec.size() - 1) {
        result += ", ";
      }
    }
    result += "]";
    return fmt::format_to(ctx.out(), "{}", result);
  }
};
