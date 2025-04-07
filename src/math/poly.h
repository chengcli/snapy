#pragma once

// C/C++
#include <cstdio>
#include <cstdlib>

// base
#include <configure.h>

// -----------------------------------------------------------------------------
// Computes all coefficients of P(x) = (x1 - x)*(x2 - x)*...*(xn - x),
// storing them in 'coeff' (size n+1) IN PLACE.
//
// After calling, coeff[j] will be the coefficient of x^j.
//
// Requirements:
//   - 'coeff' must be an array of length n+1 (allocated by the caller).
//   - 'xs' is an array of length n holding x1,...,xn.
//   - All memory must be pre-allocated; function returns void.
// -----------------------------------------------------------------------------
template <typename T>
inline DISPATCH_MACRO void poly_coeffs_from_elementary_symmetric(T* coeff,
                                                                 T const* xs,
                                                                 int n) {
  // (1) Initialize coefficients to represent the constant polynomial "1".
  //     i.e. coeff[0] = 1, and coeff[1..n] = 0.
  for (int i = 0; i <= n; i++) {
    coeff[i] = 0.0;
  }
  coeff[0] = 1.0;

  // (2) Iteratively multiply by each factor (x_i - x).
  int deg = 0;  // current polynomial degree starts at 0 ("1" is degree 0)
  for (int i = 0; i < n; i++) {
    auto x_i = xs[i];

    // Multiply the polynomial of degree 'deg' by (x_i - x),
    // which yields a new polynomial of degree 'deg+1'.
    // We'll do it backwards to avoid overwriting data we still need.
    for (int j = deg; j >= 0; j--) {
      auto tmp = coeff[j];

      // The x^(j+1) term gets += -tmp
      coeff[j + 1] += -tmp;

      // The x^j term becomes x_i * old coeff[j]
      coeff[j] = x_i * tmp;
    }

    // Now the polynomial degree has increased by 1
    deg++;
  }
}

// Evaluate the polynomial f(x) = a_n x^n + ... + a_1 x + a_0
//   coeff[0] = a_0
//   coeff[1] = a_1
//   ...
//   coeff[n] = a_n
//   So the polynomial is sum_{k=0..n} [ coeff[k] * x^k ].
// But if you prefer storing in reverse, just adapt accordingly.
template <typename T>
inline DISPATCH_MACRO T poly_eval(T const* coeff, int n, T x) {
  T val = 0.0;
  // We'll assume here that coeff[k] is the coefficient of x^k, i.e. a_k.
  for (int k = n; k >= 0; k--) {
    val = val * x + coeff[k];
  }
  return val;
}

// Evaluate the derivative f'(x) = n*a_n x^(n-1) + (n-1)*a_{n-1} x^(n-2) + ...
// In the same ordering:
//   coeff[0] = a_0
//   coeff[1] = a_1
//   ...
//   coeff[n] = a_n
template <typename T>
inline DISPATCH_MACRO T poly_ddx(T const* coeff, int n, T x) {
  // If n=0, the polynomial is a constant => derivative is 0
  if (n == 0) return 0.0;

  T val = 0.0;
  // We'll build the derivative polynomial from highest term down
  // derivative of a_k * x^k = k*a_k * x^(k-1).
  // So for k in [n..1], derivative has (k*a_k) as the coefficient for x^(k-1).
  for (int k = n; k >= 1; k--) {
    val = val * x + k * coeff[k];
  }
  return val;
}

template <typename T>
inline DISPATCH_MACRO T poly_solve(T const* coeff, int n, T x0,
                                   int maxIter = 100, T tol = 1e-12) {
  auto xk = x0;  // current approximation
  for (int i = 0; i < maxIter; i++) {
    auto fval = poly_eval(coeff, n, xk);
    auto dfval = poly_ddx(coeff, n, xk);

    // Check if derivative is too small => risk of division by zero
    if (fabs(dfval) < 1e-16) {
      printf("Derivative near 0! Newton's method may fail.\n");
      break;
    }

    auto x_next = xk - fval / dfval;

    // Check for convergence
    if (fabs(x_next - xk) < tol) {
      return x_next;  // Converged
    }
    xk = x_next;

    // If we didn't break, then we continue to next iteration
    if (i == maxIter - 1) {
      printf("Reached maximum iterations without full convergence.\n");
    }
  }
}
