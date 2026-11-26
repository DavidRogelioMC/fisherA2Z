"""Regression for the two-argument validator of the active quantiles_v1 module."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from fisherA2Z.fisher_flex import FisherFlex

class QuantileV1PriorTests(unittest.TestCase):
    def test_validator_accepts_two_and_three_arguments(self):
        from fisherA2Z import nz_quantiles as nq
        cov=np.repeat(np.eye(nq.N_PARAMS)[None],5,axis=0)
        np.testing.assert_array_equal(nq.validate_prior_cov(cov,5),
                                      nq.validate_prior_cov(cov,5,nq.N_PARAMS))

    def test_validator_rejects_mismatched_explicit_parameter_count(self):
        from fisherA2Z import nz_quantiles as nq
        cov=np.repeat(np.eye(nq.N_PARAMS)[None],5,axis=0)
        with self.assertRaises(ValueError):nq.validate_prior_cov(cov,5,nq.N_PARAMS-1)

    def make_flex(self, prior_mode='full', cov=None):
        z=np.linspace(.01,3,40)
        central=np.array([np.exp(-.5*((z-c)/.35)**2)+.05 for c in (.4,.7,1.,1.3,1.6)])
        random=np.random.default_rng(25)
        # Same orientation as the user's samples: bins, realizations, redshift.
        samples=central[:,None,:]*np.exp(random.normal(0,.2,(5,35,40)))
        return FisherFlex(nz_source=central,nz_realizations=samples if cov is None else None,
            z_grid=z,photoz_prior_cov=cov,realizations_axis=1,
            neff_source=np.full(5,2.),fsky=.5,sigma_e=.26,mode='cosmic_shear',
            nz_model='quantiles',prior_mode=prior_mode,n_lens_bins=2,
            ell_edges=np.geomspace(100,2500,21),verbose=False)

    def test_prior_from_realizations_has_expected_shape_and_is_positive(self):
        f=self.make_flex()
        self.assertEqual(f.realization_params.shape,(35,5,8))
        self.assertEqual(f.pz_prior_cov.shape,(5,8,8))
        for c in f.pz_prior_cov:self.assertGreater(np.min(np.linalg.eigvalsh(c)),0)
        direct=self.make_flex(cov=f.pz_prior_cov)
        np.testing.assert_array_equal(direct.pz_prior_cov,f.pz_prior_cov)

    def test_diagonal_prior_from_realizations(self):
        f=self.make_flex(prior_mode='diag')
        for c in f.pz_prior_cov:np.testing.assert_array_equal(c,np.diag(np.diag(c)))

    def test_save_load_keeps_auto_prior_and_forecast(self):
        f=self.make_flex(); rng=np.random.default_rng(9)
        b,nell=f.layout.n_blocks,len(f.ell)
        f.deriv=rng.normal(size=(f.n_params,b,nell))
        f.data_vector=np.ones((b,nell))
        f.cov_blocks=np.repeat(np.eye(b)[None],nell,axis=0)
        f.chi_eff_lens_Mpc=np.full(f.n_lens,1000.)
        f._computed=True
        mask=np.ones((b,nell),dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'quantiles.npz';f.save(path)
            loaded=FisherFlex.from_npz(path,verbose=False)
            np.testing.assert_array_equal(loaded.pz_prior_cov,f.pz_prior_cov)
            np.testing.assert_allclose(loaded.forecast(mask=mask).fisher,f.forecast(mask=mask).fisher)

if __name__=='__main__':unittest.main()
