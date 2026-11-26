"""Configurable quantiles and no external photo-z prior (unittest compatible)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from fisherA2Z import nz_quantiles as nq
from fisherA2Z.fisher_flex import FisherFlex


class QuantileConfigurationTests(unittest.TestCase):
    def make_flex(self, n=6, cov=None, real=None):
        z = np.linspace(0.01, 3, 301)
        nz = np.array([np.exp(-0.5*((z-0.7)/0.2)**2),
                       np.exp(-0.5*((z-1.2)/0.3)**2)])
        return FisherFlex(nz_source=nz, z_grid=z, neff_source=[2., 2.],
                          fsky=0.5, sigma_e=0.26, mode='cosmic_shear',
                          nz_model='quantiles', n_quantiles=n,
                          n_lens_bins=2, ell_edges=np.geomspace(100, 2500, 25),
                          photoz_prior_cov=cov, nz_realizations=real,
                          verbose=False)

    def synthetic_payload(self, flex):
        # Full-rank synthetic observables isolate Fisher algebra from CCL.
        b, ell = flex.layout.n_blocks, flex.ell.size
        rng = np.random.default_rng(42)
        flex.deriv = rng.normal(size=(flex.n_params, b, ell))
        flex.cov_blocks = np.repeat(np.eye(b)[None], ell, axis=0)
        flex.data_vector = np.ones((b, ell))
        flex.chi_eff_lens_Mpc = np.full(flex.n_lens, 1000.)
        flex._computed = True
        return np.ones((b, ell), dtype=bool)

    def test_uniform_pdf_analytic_quantiles_and_normalization(self):
        z = np.linspace(0.01, 3, 301)
        for n in (3, 6, 10, 14):
            with self.subTest(n=n):
                m = nq.build_quantile_model(z, np.ones_like(z), n_quantiles=n)
                np.testing.assert_allclose(m.locs[0], np.linspace(z[0], z[-1], n))
                np.testing.assert_allclose(m.nz_fid, 1/(z[-1]-z[0]), rtol=1e-11)
                np.testing.assert_allclose(np.trapz(m.nz_fid, z, axis=1), 1)

    def test_independent_instances_and_free_parameters(self):
        a, b = self.make_flex(4), self.make_flex(12)
        for f, n in ((a, 4), (b, 12), (a, 4)):
            names = [p for p in f.param_order if p.startswith('qshift')]
            self.assertEqual(len(names), 2*(n-2))
            self.assertIsNone(f.pz_prior_cov)
            self.assertTrue(all(np.isinf(f.prior_sigma[p]) for p in names))
            model = f._model()
            moved = dict(f.fid)
            moved[names[0]] += f.param_step[names[0]]
            self.assertFalse(np.array_equal(model.source_nz(moved), model.source_nz(f.fid)))

    def test_no_photoz_precision_but_other_priors_remain(self):
        f = self.make_flex()
        mask = self.synthetic_payload(f)
        a, b = f.forecast(mask=mask, prior=True), f.forecast(mask=mask, prior=False)
        extra = a.fisher - b.fisher
        for i, p in enumerate(f.param_order):
            if p.startswith('qshift'):
                np.testing.assert_array_equal(extra[i], 0.)
        self.assertGreater(np.max(np.diag(extra)), 0.)
        np.testing.assert_allclose(a.fisher @ a.cov, np.eye(f.n_params), atol=1e-10)

    def test_supplied_correlated_prior_is_used(self):
        n = 6
        block = np.eye(n-2)*0.002 + np.ones((n-2,n-2))*0.0001
        f = self.make_flex(n, cov=np.repeat(block[None], 2, axis=0))
        mask = self.synthetic_payload(f)
        extra = f.forecast(mask=mask).fisher - f.forecast(mask=mask, prior=False).fisher
        for b in range(2):
            idx = [f.param_order.index(p) for p in f._pz_param_names(b)]
            np.testing.assert_allclose(extra[np.ix_(idx,idx)], np.linalg.inv(block))

    def test_roundtrip_with_and_without_prior(self):
        for n in (4, 10):
            for prior in (False, True):
                with self.subTest(n=n, prior=prior), tempfile.TemporaryDirectory() as tmp:
                    cov = np.repeat((np.eye(n-2)*0.001)[None], 2, axis=0) if prior else None
                    f = self.make_flex(n, cov=cov)
                    mask = self.synthetic_payload(f)
                    path = Path(tmp)/'forecast.npz'
                    f.save(path)
                    loaded = FisherFlex.from_npz(path, verbose=False)
                    self.assertEqual(loaded.n_pz_per_bin, n-2)
                    self.assertEqual(loaded.param_order, f.param_order)
                    np.testing.assert_array_equal(loaded.forecast(mask=mask).fisher,
                                                  f.forecast(mask=mask).fisher)
                    if not prior:
                        self.assertIsNone(loaded.pz_prior_cov)
                    else:
                        np.testing.assert_array_equal(loaded.pz_prior_cov, cov)
                    with np.load(path, allow_pickle=False) as data:
                        for key in data.files:
                            self.assertNotEqual(data[key].dtype, object)
                    other = self.make_flex(n+1)
                    with self.assertRaises(ValueError):
                        other._load_npz(path)

    def test_realizations_still_generate_a_prior(self):
        for n in (3, 5):
            with self.subTest(n=n):
                f = self.make_flex(n)
                rng = np.random.default_rng(12)
                real = f.nz_source[None] * np.exp(rng.normal(0, .1, (40, 2, f.z.size)))
                fitted = self.make_flex(n, real=real)
                self.assertEqual(fitted.pz_prior_cov.shape, (2,n-2,n-2))
                nq.validate_prior_cov(fitted.pz_prior_cov, 2, n-2)

    def test_invalid_dimensions_rejected(self):
        for n in (0, 2, 3.5, True):
            with self.assertRaises(ValueError):
                self.make_flex(n)
        with self.assertRaises(ValueError):
            self.make_flex(6, cov=np.zeros((2,8,8)))


if __name__ == '__main__':
    unittest.main()
