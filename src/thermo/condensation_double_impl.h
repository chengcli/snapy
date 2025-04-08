
// x1 + x2 -> y at constant volume (mole concentration)
inline pair<double, double> satfunc2v(double s, double x1, double x2, double y,
                                      double logs_ddT = 0.) {
  double delta = (x1 - x2) * (x1 - x2) + 4 * s;
  double rate = (x1 + x2 - sqrt(delta)) / 2.;

  if (rate > 0. || (rate < 0. && y > -rate)) {
    return {rate, -s * logs_ddT / sqrt(delta)};
  }
  return {-y, -s * logs_ddT / sqrt(delta)};
}

inline void set_jac2v(Eigen::MatrixXd& jac, double s, double const* conc, int j,
                      int ix1, int ix2, int iy) {
  double const& x1 = conc[ix1];
  double const& x2 = conc[ix2];
  double const& y = conc[iy];
  double delta = (x1 - x2) * (x1 - x2) + 4 * s;
  double rate = (x1 + x2 - sqrt(delta)) / 2.;

  if (rate > 0. || (rate < 0. && y > -rate)) {
    jac(j, ix1) = (1. - (x1 - x2) / sqrt(delta)) / 2.;
    jac(j, ix2) = (1. - (x2 - x1) / sqrt(delta)) / 2.;
  } else {
    jac(j, iy) = -1.;
  }
}

// x1 + x2 -> y at constant pressure (mole fraction)
inline double satfunc2p(double s, double x1, double x2, double y, double g) {
  double delta = (x1 - x2) * (x1 - x2) + 4 * s * (g - 2. * x1) * (g - 2. * x2);
  double rate = (x1 + x2 - 4 * g * s - sqrt(delta)) / (2. * (1. - 4. * s));

  if (rate > 0. || (rate < 0. && y > -rate)) {
    return rate;
  }
  return -y;
}

inline void set_jac2p(Eigen::MatrixXd& jac, double s, double const* frac,
                      double g, int j, int ix1, int ix2, int iy) {
  double const& x1 = frac[ix1];
  double const& x2 = frac[ix2];
  double const& y = frac[iy];

  double delta = (x1 - x2) * (x1 - x2) + 4 * s * (g - 2. * x1) * (g - 2. * x2);
  double rate = (x1 + x2 - 4 * g * s - sqrt(delta)) / (2. * (1. - 4. * s));

  if (rate > 0. || (rate < 0. && y > -rate)) {
    jac(j, ix1) = (1. - (x1 - x2) / sqrt(delta)) / 2.;
    jac(j, ix2) = (1. - (x2 - x1) / sqrt(delta)) / 2.;
  } else {
    jac(j, iy) = -1.;
  }
}
