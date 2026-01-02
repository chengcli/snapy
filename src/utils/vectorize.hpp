#pragma once

// C/C++
#include <cstring>
#include <string>
#include <vector>

namespace snap {

//! \brief Split a string into a vector of values
//!
//! Parses a C-string and splits it into a vector based on the given delimiter.
//! Values are parsed as floats using std::stof and then cast to type A.
//!
//! \tparam A Target type for vector elements (e.g., float, double, int)
//! \param[in] cstr Input C-string to parse
//! \param[in] delimiter Delimiter string for splitting (default: " ")
//! \return Vector of parsed values of type A
//!
//! \note Values are parsed as float then cast to type A, which may lose
//! precision
template <typename A>
std::vector<A> Vectorize(const char* cstr, const char* delimiter = " ") {
  std::vector<A> arr;
  char str[1028], *p;
  snprintf(str, sizeof(str), "%s", cstr);
  p = std::strtok(str, delimiter);
  while (p != NULL) {
    arr.push_back(static_cast<A>(std::stof(p)));
    p = std::strtok(NULL, delimiter);
  }
  return arr;
}

//! \brief Template specialization for parsing strings
//!
//! Specialized version of Vectorize for std::string type that preserves
//! string values instead of converting to numeric types.
//!
//! \param[in] cstr Input C-string to parse
//! \param[in] delimiter Delimiter string for splitting
//! \return Vector of strings
template <>
std::vector<std::string> Vectorize(const char* cstr, const char* delimiter);

}  // namespace snap
