"""PDF degrees, probability conservation, and Fisher serialization of the order."""
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.interpolate import PchipInterpolator, PPoly

from fisherA2Z import nz_quantiles as nq
from fisherA2Z import nz_quantiles_v2 as nq2
from fisherA2Z import _quantile_interpolation as interpolation
from fisherA2Z.fisher_flex import FisherFlex, _FlexModel


class QuantileInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.levels = np.linspace(0., 1., 10)
        self.z = np.r_[0., np.linspace(.003, .9, 227), 1.]
        # Analytic quantiles of the positive linear PDF (1 + .4*z)/1.2.
        self.q = (np.sqrt(1. + .96*self.levels)-1.)/.4
        self.q[[0, -1]] = [0., 1.]

    def test_all_degrees_preserve_probabilities_and_normalization(self):
        for order in range(9):
            with self.subTest(order=order):
                pdf = nq.evaluate_quantile_nz(self.z, self.levels, self.q,
                                               np.zeros(8), order)
                self.assertGreaterEqual(pdf.min(), 0.)
                self.assertAlmostEqual(np.trapz(pdf, self.z), 1., places=13)
                if order:
                    cdf = interpolation._cdf_spline(self.q, self.levels, order)
                    np.testing.assert_allclose(cdf(self.q), self.levels, atol=2e-13)
                    polynomial = cdf if isinstance(cdf, PPoly) else PPoly.from_spline(cdf)
                    self.assertEqual(polynomial.c.shape[0]-2, order)

    def test_higher_orders_reproduce_an_analytic_quadratic_cdf(self):
        edges = np.r_[self.z[0], .5*(self.z[1:]+self.z[:-1]), self.z[-1]]
        expected = (1. + .2*(edges[1:]+edges[:-1]))/1.2
        for order in range(3, 9):
            pdf = nq.evaluate_quantile_nz(self.z, self.levels, self.q, np.zeros(8), order)
            np.testing.assert_allclose(pdf, expected, rtol=1e-10, atol=1e-10)

    def test_uniform_pdf_all_orders_and_numpy_integer(self):
        for order in range(9):
            pdf = nq.evaluate_quantile_nz(self.z, self.levels, self.levels,
                                           np.zeros(8), np.int64(order))
            np.testing.assert_allclose(pdf, 1., atol=2e-11)

    def test_orders_zero_and_two_match_previous_reconstructions(self):
        edges = np.r_[self.z[0], .5*(self.z[1:]+self.z[:-1]), self.z[-1]]
        expected = {0: np.interp(edges, self.q, self.levels),
                    2: PchipInterpolator(self.q, self.levels)(edges)}
        for order in (0, 2):
            pdf = nq.evaluate_quantile_nz(self.z, self.levels, self.q, np.zeros(8), order)
            np.testing.assert_allclose(pdf, np.diff(expected[order])/np.diff(edges), atol=1e-12)

    def test_linear_pdf_is_nonnegative_for_highly_unequal_quantile_gaps(self):
        rng = np.random.default_rng(192)
        for _ in range(50):
            gaps = np.exp(rng.uniform(-6., 3., 9))
            q = np.r_[0., np.cumsum(gaps)]
            q /= q[-1]
            cdf = interpolation._cdf_spline(q, self.levels, 1)
            nodes = np.sort(np.r_[q, .5*(q[1:]+q[:-1])])
            self.assertGreaterEqual(cdf.derivative()(nodes).min(), -1e-10)
            np.testing.assert_allclose(cdf(q), self.levels, atol=1e-12)
            interpolation._check_nonnegative_pdf(cdf, q, self.levels, 1)

    def test_negative_pdf_rejected_even_with_a_coarse_evaluation_grid(self):
        q = np.array([0., .01, .02, .03, .1, .2, .4, .7, .9, 1.])
        with self.assertRaisesRegex(ValueError, 'negative n\\(z\\)'):
            nq.evaluate_quantile_nz(np.array([0., .5, 1.]), self.levels, q, np.zeros(8), 4)

    def test_invalid_order_rejected(self):
        for order in (-1, 1.5, 1., True, 'linear', 9):
            with self.subTest(order=order), self.assertRaises(ValueError):
                nq.evaluate_quantile_nz(self.z, self.levels, self.q, np.zeros(8), order)

    def test_v2_keeps_configurable_quantiles_and_its_default_order(self):
        for n in (3, 6, 10, 14):
            for order in (0, 1, 2):
                m = nq2.build_quantile_model(self.z, np.ones_like(self.z),
                                             n_quantiles=n, interpolation_order=order)
                self.assertEqual(m.locs.shape, (1, n))
                self.assertEqual(m.interpolation_order, order)
                np.testing.assert_allclose(m.nz_fid, 1., atol=2e-12)
        self.assertEqual(nq2.build_quantile_model(self.z, np.ones_like(self.z)).interpolation_order, 0)


class FisherInterpolationTests(unittest.TestCase):
    def make_flex(self, order=None, realizations=False):
        z = np.linspace(.01, 2., 101)
        nz = np.array([1.+.2*z, 1.+.4*z])
        rng = np.random.default_rng(195)
        samples = nz[:, None, :]*np.exp(rng.normal(0, .03, (2, 24, len(z))))
        return FisherFlex(nz_source=nz, nz_realizations=samples if realizations else None,
                          realizations_axis=1, z_grid=z, neff_source=[2., 2.],
                          fsky=.5, sigma_e=.26, mode='cosmic_shear', nz_model='quantiles',
                          quantile_interpolation_order=order, n_lens_bins=2,
                          ell_edges=np.geomspace(100, 2500, 25), verbose=False)

    @staticmethod
    def synthetic_payload(f):
        rng = np.random.default_rng(123)
        b, nell = f.layout.n_blocks, len(f.ell)
        f.deriv = rng.normal(size=(f.n_params, b, nell))
        f.cov_blocks = np.repeat(np.eye(b)[None], nell, axis=0)
        f.data_vector = np.ones((b, nell))
        f.chi_eff_lens_Mpc = np.full(f.n_lens, 1000.)
        f._computed = True
        return np.ones((b, nell), dtype=bool)

    def test_default_and_explicit_orders_survive_worker_serialization(self):
        for order in (0, 1, 2, 4):
            with patch.object(nq, 'INTERPOLATION_ORDER', order):
                f = self.make_flex()
            self.assertEqual(f.quantile_interpolation_order, order)
            worker = _FlexModel(pickle.loads(pickle.dumps(f._to_state())))
            with patch.object(nq, 'INTERPOLATION_ORDER', 8):
                np.testing.assert_allclose(worker.source_nz(f.fid), f.nz_source_fid)
                shifted = dict(f.fid)
                shifted['qshift1_1'] += 1e-5
                expected = nq.evaluate_quantile_nz(f.z, f.quantile_levels, f.quantile_locs[0],
                                                   np.r_[1e-5, np.zeros(7)], order)
                np.testing.assert_allclose(worker.source_nz(shifted)[0], expected)
            self.assertEqual(worker.to_state()['quantile_interpolation_order'], order)

    def test_changing_order_does_not_change_quantile_locations_or_prior(self):
        base = self.make_flex(0, realizations=True)
        for order in (1, 2, 4):
            f = self.make_flex(order, realizations=True)
            np.testing.assert_array_equal(f.quantile_locs, base.quantile_locs)
            np.testing.assert_array_equal(f.realization_params, base.realization_params)
            np.testing.assert_array_equal(f.pz_prior_cov, base.pz_prior_cov)

    def test_saved_forecast_keeps_order_and_rejects_other_or_unknown_order(self):
        for order in (0, 1, 2, 4):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as temp:
                f = self.make_flex(order)
                mask = self.synthetic_payload(f)
                path = Path(temp)/'order.npz'
                f.save(path)
                loaded = FisherFlex.from_npz(path, verbose=False)
                self.assertEqual(loaded.quantile_interpolation_order, order)
                np.testing.assert_array_equal(loaded.forecast(mask=mask).fisher,
                                               f.forecast(mask=mask).fisher)
                self.make_flex(order)._load_npz(path)
                with self.assertRaisesRegex(ValueError, 'Different quantile interpolation'):
                    self.make_flex(1 if order == 0 else 0)._load_npz(path)
                with np.load(path, allow_pickle=False) as data:
                    content = {key: data[key] for key in data.files}
                for key in ('quantile_interpolation_order', 'quantile_interpolation_version'):
                    incomplete = dict(content)
                    incomplete.pop(key)
                    np.savez(path, **incomplete)
                    with self.assertRaisesRegex(ValueError, 'compatible interpolation order'):
                        FisherFlex.from_npz(path, verbose=False)


if __name__ == '__main__':
    unittest.main()
