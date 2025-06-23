
//! \brief Calculate temperature from primitive variable
/*!
 * $T = p/(\rho R) = p/(\rho \frac{R}{R_d} Rd)$
 * \param w primitive variable
 * \return $T$
 */
torch::Tensor get_temp(torch::Tensor w) const;

//! \brief Effective adiabatic index
/*!
 * Eq.71 in Li2019
 * \param w primitive variable
 * \return $\chi$
 */
torch::Tensor get_chi(torch::Tensor w) const;

//! \brief Calculate potential temperature from primitive variable
/*!
 * $\theta = T (p0/p)^{R_d/c_p}$
 * \param w primitive variable
 * \return $\theta$
 */
torch::Tensor get_theta(torch::Tensor w, double p0) const;
