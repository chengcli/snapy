"""Independent analytic-limit checks for the compressible spherical reference."""

import math
import unittest

from spherical_compressible_reference import critical


class BoussinesqLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {
            (eta, degree): critical(degree, eta, a=1e-4, dissipation=5e-5)
            for eta in (0.99, 0.9999)
            for degree in (24, 32)
        }

    def test_weak_stratification_thin_shell_recovers_analytic_plane(self):
        exact_ra = 27 * math.pi**4 / 4
        exact_k = math.pi / math.sqrt(2)
        coarse = self.results[0.99, 32]
        thin = self.results[0.9999, 32]
        self.assertLess(abs(thin["Ra"] - exact_ra), 1e-5)
        self.assertLess(abs(thin["k"] - exact_k), 3e-5)
        self.assertLess(abs(thin["Ra"] - exact_ra), abs(coarse["Ra"] - exact_ra) / 1000)

    def test_radial_refinement_preserves_limit_and_selected_degree(self):
        for eta in (0.99, 0.9999):
            low, high = (self.results[eta, degree] for degree in (24, 32))
            self.assertEqual(low["ell"], high["ell"])
            self.assertLess(abs(low["Ra"] - high["Ra"]), 1e-8)


if __name__ == "__main__":
    unittest.main()
