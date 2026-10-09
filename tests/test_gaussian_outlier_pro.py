"""Mathematical and Fisher integration regressions for the new model."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import ndtr

from fisherA2Z import nz_gaussian_outlier_pro as go
from fisherA2Z import fisher_flex as ff
from fisherA2Z.fisher_flex import FisherFlex


class GaussianOutlierTests(unittest.TestCase):
    def setUp(self):
        self.z = np.linspace(0.01, 3.01, 301)
        self.nz = go.normalize_nz(self.z, np.array([
            np.exp(-0.5 * ((self.z - 0.7) / 0.15)**2),
            np.exp(-0.5 * ((self.z - 1.6) / 0.2)**2)]))
        self.c = np.array([[0, 0.05], [0.02, 0]])

    def flex(self, **kwargs):
        config = kwargs.pop('gaussian_outlier_pro_kwargs',
                            dict(dz=0.001, std_b=0.12, c=self.c))
        return FisherFlex(nz_source=self.nz, z_grid=self.z, neff_source=[2, 3],
                          fsky=0.5, sigma_e=0.26, mode='cosmic_shear',
                          nz_model='gaussian_outlier_pro', verbose=False,
                          gaussian_outlier_pro_kwargs=config, **kwargs)

    def test_convolution_uniform_pdf_analytic_nonuniform_grid(self):
        t = np.linspace(0, 1, 1801)**1.2 * 3
        x = np.linspace(-0.5, 3.5, 21)
        shift, width = 0.15, 0.3
        expected = (ndtr((x-shift)/width) - ndtr((x-shift-3)/width)) / 3
        actual = go.gaussian_int(x, np.ones_like(t)/3, t, shift, width)
        assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)

    def test_shift_and_variance_match_gaussian_convolution(self):
        z = np.linspace(-4, 6, 3001)
        n = np.exp(-0.5*((z-0.7)/0.3)**2)/(np.sqrt(2*np.pi)*0.3)
        x = np.linspace(-2, 4, 31)
        sigma = np.hypot(0.3, 0.4)
        expected = np.exp(-0.5*((x-0.7-0.12)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
        assert_allclose(go.gaussian_int(x, n, z, 0.12, 0.4), expected, rtol=1e-11, atol=1e-14)

    def test_row_mixing_and_normalization(self):
        components, _ = go.get_dnGdz(self.nz, self.z, 2, 0.001, 0.12)
        result = go.evaluate_nz(self.nz, self.z, 0.001, 0.12, self.c)
        assert_allclose(result[0], 0.95*components[0]+0.05*components[1])
        assert_allclose(result[1], 0.98*components[1]+0.02*components[0])
        assert_allclose(np.trapz(result, self.z, axis=1), 1, atol=1e-14)
        assert_array_equal(go.outlier_fraction(self.c), [0.05, 0.02])
        only_first = self.c.copy(); only_first[0, 1] = 0.3
        assert_array_equal(go.evaluate_nz(self.nz, self.z, 0.001, 0.12, only_first)[1], result[1])

    def test_reference_interface_zero_outside_support(self):
        z, fun = go.run(self.nz, self.z, 2, 0, 0.12, self.c)
        expected = go.evaluate_nz(self.nz, self.z, 0, 0.12, self.c)
        assert_array_equal(z, self.z)
        for i in range(2):
            assert_array_equal(fun[i](z), expected[i])
            assert_array_equal(fun[i]([-1, 5]), [0, 0])

    def test_invalid_matrix_kernel_and_prior(self):
        for c in (np.ones((2, 2)), [[0, -0.1], [0, 0]], [[0, 1.1], [0, 0]]):
            with self.assertRaises(ValueError):
                go.validate_mixing(c, 2)
        with self.assertRaises(ValueError):
            go.get_dnGdz(self.nz, self.z, 2, 0, 0.001)
        with self.assertRaises(ValueError):
            go.validate_prior_cov(np.zeros((2, 3, 3)), 2)

    def test_parameter_order_fiducials_and_model(self):
        flex = self.flex()
        self.assertEqual(flex._pz_param_names(0), ['gdz1', 'gsigma1', 'gc1_2'])
        self.assertEqual(flex.n_pz_per_bin, 3)
        self.assertIsNone(flex.pz_prior_cov)
        assert_allclose(flex.nz_source_fid, go.evaluate_nz(self.nz, self.z, 0.001, 0.12, self.c))
        model = flex._model()
        assert_array_equal(model.source_nz(flex.fid), flex.nz_source_fid)
        values = dict(flex.fid, gc1_2=0.1)
        varied = model.source_nz(values)
        assert_allclose(varied[1], flex.nz_source_fid[1])
        self.assertGreater(np.max(np.abs(varied[0]-flex.nz_source_fid[0])), 0.01)
        # Width of template 1 also affects bin 2 through c[2,1].
        varied = model.source_nz(dict(flex.fid, gsigma1=0.13))
        self.assertGreater(np.max(np.abs(varied[1]-flex.nz_source_fid[1])), 1e-5)

    def test_fixed_mixing_and_prior(self):
        prior = np.tile(np.diag([0.002**2, 0.01**2]), (2, 1, 1))
        flex = self.flex(gaussian_outlier_pro_kwargs=dict(dz=0, std_b=0.12,
                         c=self.c, vary_mixing=False), photoz_prior_cov=prior)
        self.assertEqual(flex._pz_param_names(0), ['gdz1', 'gsigma1'])
        assert_array_equal(flex.pz_prior_cov, prior)
        self.assertFalse(any(p.startswith('gc') for p in flex.param_order))

    def test_automatic_calibration_requires_fixed_mixing(self):
        with self.assertRaisesRegex(ValueError, 'vary_mixing=False'):
            self.flex(nz_realizations=self.nz[None])

    def test_boundary_derivatives_remain_physical(self):
        flex = self.flex(gaussian_outlier_pro_kwargs=dict(dz=0, std_b=0.12,
                         c=np.zeros((2, 2))))
        self.assertEqual(flex.param_deriv_method['gc1_2'], 'forward')
        for p in flex._pz_param_names(0):
            values = dict(flex.fid)
            values[p] += flex.param_step[p]
            self.assertTrue(np.all(np.isfinite(flex._model().source_nz(values))))

    def test_worker_honors_forward_derivative_at_zero_fraction(self):
        flex = self.flex(gaussian_outlier_pro_kwargs=dict(dz=0, std_b=0.12,
                         c=np.zeros((2, 2))))
        ff._init_worker(flex._to_state(), flex._layout_state(), flex.fid,
                        flex.param_step, 'central', flex.param_deriv_method)
        def evaluator(param, value):
            self.assertGreaterEqual(value, 0)
            return np.array([1 + value*value])
        with patch.object(ff, '_eval_at', side_effect=evaluator):
            actual = ff._deriv_one_param_dispatch('gc1_2')
        assert_allclose(actual, [flex.param_step['gc1_2']], rtol=1e-7)

    def payload(self, flex):
        rng = np.random.default_rng(45)
        blocks, nell = flex.layout.n_blocks, flex.ell.size
        flex.deriv = rng.normal(size=(flex.n_params, blocks, nell))
        flex.data_vector = np.ones((blocks, nell))
        flex.cov_blocks = np.repeat(np.eye(blocks)[None], nell, axis=0)
        flex.chi_eff_lens_Mpc = np.ones(flex.n_lens) * 1000
        flex._computed = True

    def test_save_load_and_fiducial_model_without_external_prior(self):
        flex = self.flex()
        self.payload(flex)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'test.npz')
            flex.save(path)
            loaded = FisherFlex.from_npz(path, verbose=False)
            assert_array_equal(loaded.forecast().cov, flex.forecast().cov)
            assert_array_equal(loaded._model().source_nz(loaded.fid), flex.nz_source_fid)
            assert_array_equal(loaded.gpro_templates, flex.gpro_templates)
            self.assertIsNone(loaded.pz_prior_cov)
            changed = self.flex(gaussian_outlier_pro_kwargs=dict(dz=0.1, std_b=0.12, c=self.c))
            with self.assertRaisesRegex(ValueError, 'different gpro_dz'):
                changed.compute(precompute=path)
            other = FisherFlex(nz_source=self.nz, z_grid=self.z, neff_source=[2,3],
                               fsky=0.5, sigma_e=0.26, mode='cosmic_shear',
                               nz_model='no_uncertainty', verbose=False)
            with self.assertRaisesRegex(ValueError, 'cannot reuse'):
                other.compute(precompute=path)

    def test_full_correlated_prior_and_fixed_mixing_roundtrip(self):
        prior = np.tile(np.array([[1e-4, 2e-5], [2e-5, 4e-4]]), (2,1,1))
        flex = self.flex(gaussian_outlier_pro_kwargs=dict(dz=0, std_b=0.12,
                         c=self.c, vary_mixing=False), photoz_prior_cov=prior)
        self.payload(flex)
        with_prior, no_prior = flex.forecast(), flex.forecast(prior=False)
        ids = [flex.param_order.index(p) for p in flex._pz_param_names(0)]
        actual = (with_prior.fisher-no_prior.fisher)[np.ix_(ids, ids)]
        assert_allclose(actual, np.linalg.inv(prior[0]), rtol=1e-13, atol=1e-10)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'test.npz')
            flex.save(path)
            loaded = FisherFlex.from_npz(path, verbose=False)
            assert_array_equal(loaded.forecast().cov, with_prior.cov)
            assert_array_equal(loaded.pz_prior_cov, prior)
            self.assertFalse(loaded.gpro_vary_mixing)

    def test_degenerate_forecast_is_rejected_without_fake_precision(self):
        flex = self.flex()
        self.payload(flex)
        first = flex.param_order.index('gc1_2')
        second = flex.param_order.index('gc2_1')
        flex.deriv[second] = flex.deriv[first]
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            flex.forecast()



if __name__ == '__main__':
    unittest.main(verbosity=2)
