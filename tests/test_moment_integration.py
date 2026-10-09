"""Moments/Fisher integration contracts; run with unittest, no CCL spectra.

The full prior blocks are deliberately ragged. Synthetic observables separate
parameter/prior/persistence correctness from slow forward-model integration.
"""
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import fisherA2Z.fisher_flex as ff

from fisherA2Z import nz_moments as nzm
from fisherA2Z.fisher_flex import FisherFlex, NZ_MODELS, _FlexModel


Z = np.linspace(0.05, 2.5, 161)
COUNTS = (3, 4)  # Includes the fixed m0=1.
NZ = np.array([np.exp(-0.5*((Z-0.7)/0.3)**2),
               np.exp(-0.5*((Z-1.3)/0.42)**2)])
NZ /= np.trapz(NZ, Z, axis=-1)[:, None]
BLOCKS = (
    np.array([[4e-4, 2e-4], [2e-4, 9e-4]]),
    np.array([[4e-4, 1e-4, -1e-4],
              [1e-4, 9e-4, 2e-4],
              [-1e-4, 2e-4, 16e-4]]),
)


def make_flex(counts=COUNTS, cov=None, real=None, nz=NZ, **kwargs):
    options = dict(
        nz_source=nz, z_grid=Z, neff_source=[2., 2.], fsky=0.1,
        sigma_e=0.26, mode='cosmic_shear', nz_model='moments',
        n_moments=list(counts), n_lens_bins=2,
        ell_edges=np.geomspace(100., 2500., 25),
        photoz_prior_cov=cov, nz_realizations=real, deriv_method='central',
        verbose=False,
    )
    options.update(kwargs)
    return FisherFlex(**options)


def synthetic_payload(flex):
    """A full-rank experiment, with known independent measurement errors."""
    b, ell = flex.layout.n_blocks, flex.ell.size
    rng = np.random.default_rng(42)
    flex.deriv = rng.normal(size=(flex.n_params, b, ell))
    flex.cov_blocks = np.repeat(np.eye(b)[None], ell, axis=0)
    flex.data_vector = np.ones((b, ell))
    flex.chi_eff_lens_Mpc = np.full(flex.n_lens, 1000.)
    flex._computed = True
    return np.ones((b, ell), dtype=bool)


def pz_names(flex):
    return [p for b in range(flex.n_source) for p in flex._pz_param_names(b)]


def prior_increment(flex, mask, drop=None):
    with_prior = flex.forecast(mask=mask, prior=True, drop_params=drop)
    without_prior = flex.forecast(mask=mask, prior=False, drop_params=drop)
    return with_prior.param_order, with_prior.fisher - without_prior.fisher


class MomentIntegrationTests(unittest.TestCase):
    def test_registration_counts_names_and_fixed_normalization(self):
        self.assertIn('moments', NZ_MODELS)
        f = make_flex()
        self.assertEqual(f._pz_param_names(0), ['mshift1_1', 'mshift2_1'])
        self.assertEqual(f._pz_param_names(1), ['mshift1_2', 'mshift2_2', 'mshift3_2'])
        self.assertEqual(f.n_params, 7 + 4 + 5 + 2)
        self.assertFalse(any(p.startswith('mshift0') for p in f.param_order))
        for p in pz_names(f):
            self.assertEqual(f.fid[p], 0.)
            self.assertTrue(np.isinf(f.prior_sigma[p]))
            self.assertGreater(f.param_step[p], 0.)
            self.assertEqual(f.param_deriv_method[p], 'central')
        self.assertIsNone(f.pz_prior_cov)
        np.testing.assert_allclose(np.trapz(f.nz_source_fid, f.z, axis=-1), 1., atol=2e-4)
        self.assertTrue(np.all(f.nz_source_fid >= 0))

    def test_instance_counts_are_independent(self):
        a = make_flex(counts=(3, 4))
        b = make_flex(counts=(4, 3))
        self.assertEqual([len(a._pz_param_names(i)) for i in range(2)], [2, 3])
        self.assertEqual([len(b._pz_param_names(i)) for i in range(2)], [3, 2])
        self.assertEqual([len(a._pz_param_names(i)) for i in range(2)], [2, 3])

    def test_each_parameter_changes_only_its_source_bin(self):
        f = make_flex()
        model = f._model()
        base = model.source_nz(f.fid)
        np.testing.assert_allclose(base, f.nz_source_fid, rtol=1e-9, atol=1e-12)
        for b in range(2):
            for p in f._pz_param_names(b):
                with self.subTest(parameter=p):
                    vals = dict(f.fid)
                    vals[p] += f.param_step[p]
                    shifted = model.source_nz(vals)
                    self.assertGreater(np.max(np.abs(shifted[b]-base[b])), 1e-12)
                    np.testing.assert_array_equal(shifted[1-b], base[1-b])
                    np.testing.assert_allclose(np.trapz(shifted, f.z, axis=-1), 1., atol=2e-4)
                    self.assertTrue(np.all(shifted >= 0))
        for p in ('omega_m', 'A0', 'gbias1'):
            vals = dict(f.fid)
            vals[p] += 0.001
            np.testing.assert_array_equal(model.source_nz(vals), base)

    def test_worker_state_pickle_roundtrip_preserves_model(self):
        f = make_flex()
        restored = _FlexModel(pickle.loads(pickle.dumps(f._to_state())))
        vals = dict(f.fid)
        vals['mshift1_2'] = 0.5*f.param_step['mshift1_2']
        np.testing.assert_allclose(restored.source_nz(vals), f._model().source_nz(vals), rtol=1e-12)

    def test_free_moments_add_no_photoz_precision(self):
        f = make_flex()
        mask = synthetic_payload(f)
        names, extra = prior_increment(f, mask)
        for p in pz_names(f):
            np.testing.assert_array_equal(extra[names.index(p)], 0.)
        self.assertGreater(extra[names.index('omega_m'), names.index('omega_m')], 0.)

    def test_full_ragged_prior_precision_is_used(self):
        f = make_flex(cov=BLOCKS)
        mask = synthetic_payload(f)
        names, extra = prior_increment(f, mask)
        for b, block in enumerate(BLOCKS):
            idx = [names.index(p) for p in f._pz_param_names(b)]
            np.testing.assert_allclose(extra[np.ix_(idx, idx)], np.linalg.inv(block), rtol=1e-11, atol=1e-10)
        idx0 = [names.index(p) for p in f._pz_param_names(0)]
        idx1 = [names.index(p) for p in f._pz_param_names(1)]
        np.testing.assert_array_equal(extra[np.ix_(idx0, idx1)], 0.)

    def test_drop_params_conditions_full_correlated_prior(self):
        f = make_flex(cov=BLOCKS)
        mask = synthetic_payload(f)
        # Dropped parameters are FIXED, not marginalized. Select from full precision.
        dropped = ['mshift1_1', 'mshift2_2']
        names, extra = prior_increment(f, mask, drop=dropped)
        for b, block in enumerate(BLOCKS):
            group = f._pz_param_names(b)
            local = [j for j, p in enumerate(group) if p not in dropped]
            idx = [names.index(group[j]) for j in local]
            expected = np.linalg.inv(block)[np.ix_(local, local)]
            np.testing.assert_allclose(extra[np.ix_(idx, idx)], expected, rtol=1e-11, atol=1e-10)

    def test_diagonal_prior_uses_marginal_variances(self):
        f = make_flex(cov=BLOCKS, prior_mode='diag')
        mask = synthetic_payload(f)
        names, extra = prior_increment(f, mask)
        for b, block in enumerate(BLOCKS):
            idx = [names.index(p) for p in f._pz_param_names(b)]
            np.testing.assert_allclose(extra[np.ix_(idx, idx)], np.diag(1./np.diag(block)), rtol=1e-11)

    def test_realizations_measure_raw_moment_scatter(self):
        rng = np.random.default_rng(91)
        real = NZ[None]*np.exp(rng.normal(0., 0.12, (80, 2, len(Z))))
        f = make_flex(real=real, realizations_axis=0)
        for b, count in enumerate(COUNTS):
            measured = nzm.raw_moments(Z, real[:, b], count)[..., 1:]
            expected = np.cov(measured, rowvar=False, ddof=1)
            self.assertEqual(f.pz_prior_cov[b].shape, (count-1, count-1))
            np.testing.assert_allclose(f.pz_prior_cov[b], expected, rtol=1e-10, atol=1e-14)
            self.assertGreater(np.min(np.linalg.eigvalsh(f.pz_prior_cov[b])), 0.)

    def test_invalid_counts_and_prior_blocks_are_rejected(self):
        for counts in ((3,), (3, 1), (3, 3.5), (3, True)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                make_flex(counts=counts)
        bad_covs = [
            (np.eye(2),),
            (np.eye(3), np.eye(3)),
            (np.zeros((2, 2)), np.eye(3)),
            (np.array([[1., 2.], [0., 1.]]), np.eye(3)),
            (np.array([[1., 2.], [2., 1.]]), np.eye(3)),
        ]
        for cov in bad_covs:
            with self.subTest(cov=cov), self.assertRaises(ValueError):
                make_flex(cov=cov)

    def test_save_load_preserves_forecast_and_forward_model(self):
        for cov in (None, BLOCKS):
            with self.subTest(with_prior=cov is not None), tempfile.TemporaryDirectory() as tmp:
                f = make_flex(cov=cov)
                mask = synthetic_payload(f)
                path = Path(tmp)/'moments.npz'
                f.save(path)
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(int(archive['schema_version']), ff.SCHEMA_VERSION)
                    for key in archive.files:
                        self.assertNotEqual(archive[key].dtype, object, key)
                loaded = FisherFlex.from_npz(path, verbose=False)
                self.assertEqual(loaded.param_order, f.param_order)
                self.assertEqual(loaded.param_step, f.param_step)
                np.testing.assert_array_equal(loaded.forecast(mask=mask).fisher,
                                              f.forecast(mask=mask).fisher)
                vals = dict(f.fid)
                vals['mshift1_1'] = 0.5*f.param_step['mshift1_1']
                np.testing.assert_allclose(loaded._model().source_nz(vals),
                                           f._model().source_nz(vals), rtol=1e-12)
                if cov is None:
                    self.assertIsNone(loaded.pz_prior_cov)
                else:
                    for got, expected in zip(loaded.pz_prior_cov, cov):
                        np.testing.assert_array_equal(got, expected)

    def test_precompute_rejects_different_counts_fiducial_or_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = make_flex()
            synthetic_payload(f)
            path = Path(tmp)/'moments.npz'
            f.save(path)
            alternatives = [
                make_flex(counts=(4, 3)),
                make_flex(nz=NZ*np.exp(0.01*Z)[None]),
                make_flex(z_grid=Z+0.005),
                make_flex(nz_model='no_uncertainty'),
            ]
            for other in alternatives:
                with self.subTest(model=other.nz_model), self.assertRaises(ValueError):
                    other._load_npz(path)

    def test_legacy_nonmoment_schema_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = make_flex(nz_model='no_uncertainty')
            mask = synthetic_payload(f)
            path = Path(tmp)/'base.npz'
            f.save(path)
            with np.load(path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            for version in (2, 3):
                payload['schema_version'] = np.array(version)
                old = Path(tmp)/('schema%d.npz' % version)
                np.savez_compressed(old, **payload)
                loaded = FisherFlex.from_npz(old, verbose=False)
                np.testing.assert_array_equal(loaded.forecast(mask=mask).fisher,
                                              f.forecast(mask=mask).fisher)

    def test_old_moment_solver_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = make_flex()
            synthetic_payload(f)
            path = Path(tmp)/"current.npz"
            f.save(path)
            with np.load(path, allow_pickle=False) as data:
                payload = {k: data[k] for k in data.files}
            self.assertEqual(str(payload["moment_model_version"]), nzm.MODEL_VERSION)
            for version in (None, "other_solver"):
                if version is None:
                    payload.pop("moment_model_version", None)
                else:
                    payload["moment_model_version"] = np.array(version)
                legacy = Path(tmp)/"legacy.npz"
                np.savez_compressed(legacy, **payload)
                with self.assertRaisesRegex(ValueError, "Recompute derivatives"):
                    FisherFlex.from_npz(legacy, verbose=False)
                with self.assertRaisesRegex(ValueError, "Recompute derivatives"):
                    f.compute(precompute=legacy)

    def test_bias_algebra_keeps_moment_names_and_zero_truth_residual(self):
        f = make_flex(cov=BLOCKS)
        mask = synthetic_payload(f)
        model = f._model()
        # This test isolates bias propagation; root smoke-tests the actual spectra.
        with patch.object(model, 'data_vector', return_value=f.data_vector.copy()):
            bias = f.forecast_bias(f.nz_source_fid, forecast_params={'mask': mask}, model=model)
        self.assertEqual(bias.param_order, f.param_order)
        np.testing.assert_array_equal(bias.delta_cl, 0.)
        np.testing.assert_array_equal(list(bias.delta_p.values()), 0.)


if __name__ == '__main__':
    unittest.main()
