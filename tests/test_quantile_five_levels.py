"""Regression for five levels in both plotting and Fisher worker state."""
import pickle
import unittest
import numpy as np
from fisherA2Z import nz_quantiles as nq
from fisherA2Z.fisher_flex import FisherFlex, _FlexModel


class FiveQuantileTests(unittest.TestCase):
    def test_three_displacements_for_five_levels_independent_of_module_defaults(self):
        z = np.linspace(.01, 3., 301)
        levels = np.linspace(0., 1., 5)
        locs = nq.quantile_locations(z, np.exp(-.5*((z-.8)/.2)**2), levels)
        self.assertEqual(nq.N_QUANTILES, 10)
        self.assertEqual(nq.N_PARAMS, 8)
        for order in (0, 1, 2):
            nz = nq.evaluate_quantile_nz(z, levels, locs, np.zeros(3), order)
            self.assertGreaterEqual(nz.min(), 0.)
            self.assertAlmostEqual(np.trapz(nz, z), 1., places=12)
        with self.assertRaisesRegex(ValueError, '5 quantile locations and 3 interior'):
            nq.evaluate_quantile_nz(z, levels, locs, np.zeros(nq.N_PARAMS), 0)

    def test_five_bins_prior_and_worker_state_use_three_parameters_per_bin(self):
        z = np.linspace(.01, 3., 40)
        central = np.array([np.exp(-.5*((z-c)/.25)**2)+.01 for c in (.4,.7,1.,1.3,1.6)])
        rng = np.random.default_rng(314)
        samples = central[:, None, :]*np.exp(rng.normal(0., .1, (5, 30, len(z))))
        f = FisherFlex(nz_source=central, nz_realizations=samples, realizations_axis=1,
                       z_grid=z, neff_source=np.full(5, 2.), fsky=.5, sigma_e=.26,
                       mode='cosmic_shear', nz_model='quantiles', n_quantiles=5,
                       quantile_interpolation_order=0, verbose=False)
        self.assertEqual(f.quantile_locs.shape, (5, 5))
        self.assertEqual(f.n_pz_per_bin, 3)
        self.assertEqual(f.pz_prior_cov.shape, (5, 3, 3))
        self.assertEqual(f.realization_params.shape, (30, 5, 3))
        names = [p for p in f.param_order if p.startswith('qshift')]
        self.assertEqual(len(names), 15)
        worker = _FlexModel(pickle.loads(pickle.dumps(f._to_state())))
        np.testing.assert_allclose(worker.source_nz(f.fid), f.nz_source_fid)
        moved = dict(f.fid)
        moved['qshift3_5'] += f.param_step['qshift3_5']
        changed = worker.source_nz(moved)
        np.testing.assert_array_equal(changed[:4], f.nz_source_fid[:4])
        self.assertFalse(np.array_equal(changed[4], f.nz_source_fid[4]))


if __name__ == '__main__':
    unittest.main()
