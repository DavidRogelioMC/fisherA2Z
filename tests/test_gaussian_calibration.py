import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import fisherA2Z
from fisherA2Z import fisher_flex as ff
from fisherA2Z import nz_gaussian_calibration as cal
from fisherA2Z import nz_gaussian_outlier_pro as go

class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.z=np.linspace(.02,3.02,161)
        self.templates=go.normalize_nz(self.z,np.array([
            np.exp(-.5*((self.z-.6)/.12)**2),np.exp(-.5*((self.z-1.4)/.16)**2)]))
        self.c=np.array([[0,.03],[.05,0]])
        phase=np.arange(12)*2*np.pi/12
        self.params=np.empty((12,2,2))
        self.params[:,0,0]=.01+.003*np.sin(phase)
        self.params[:,1,0]=-.01+.004*np.sin(phase+.5)
        self.params[:,0,1]=.12+.008*np.cos(phase)
        self.params[:,1,1]=.2+.01*np.cos(phase+.5)
        self.real=np.array([go.evaluate_nz(self.templates,self.z,p[:,0],p[:,1],self.c) for p in self.params])

    def make(self,real=None,**kwargs):
        return ff.FisherFlex(nz_source=self.templates,z_grid=self.z,neff_source=[2,2],
                            fsky=.5,sigma_e=.26,mode='cosmic_shear',verbose=False,
                            nz_model='gaussian_outlier_pro',nz_realizations=self.real if real is None else real,
                            gaussian_outlier_pro_kwargs=dict(dz=0,std_b=1,c=self.c,vary_mixing=False,
                                                             calibration_n_jobs=1),**kwargs)

    def test_recover_known_parameters_and_covariance(self):
        flex=self.make()
        assert_allclose(flex.realization_params,self.params,atol=2e-6,rtol=2e-5)
        assert_allclose(flex.gpro_dz,self.params[:,:,0].mean(axis=0),atol=2e-7)
        assert_allclose(flex.gpro_std_b,self.params[:,:,1].mean(axis=0),atol=2e-7)
        for b in range(2):
            assert_allclose(flex.pz_prior_cov[b],np.cov(self.params[:,b],rowvar=False),rtol=1e-4,atol=1e-10)
        self.assertEqual(flex.gpro_calibration['n_realizations'],12)
        self.assertFalse(flex.gpro_calibration['subsampled'])

    def test_axis1_and_explicit_realization_grid(self):
        native=self.z[::2]
        samples=self.real[:,:,::2].transpose(1,0,2)
        flex=self.make(samples,z_realizations=native,realizations_axis=1)
        self.assertEqual(flex.realization_params.shape,(12,2,2))
        self.assertEqual(flex.gpro_calibration['n_realization_nodes'],81)
        self.assertEqual(flex.pz_prior_cov.shape,(2,2,2))
        with self.assertRaisesRegex(ValueError,'z_realizations'):
            self.make(samples,realizations_axis=1)

    def test_fft_and_nonuniform_kernel_jacobian_match_forward_model(self):
        for z in (self.z,np.linspace(0,1,161)**1.1*3+.02):
            template=go.normalize_nz(z,np.exp(-.5*((z-.6)/.12)**2))
            model=cal._ConvolutionFit(z,template)
            x=np.array([.02,.16])
            pdf,jac=model.evaluate(x)
            exact=go.get_dnGdz(template[None],z,1,*x)[0][0]
            assert_allclose(pdf,exact,rtol=1e-10,atol=1e-12)
            for j in range(2):
                shift=np.eye(2)[j]*1e-6
                numeric=(model.evaluate(x+shift)[0]-model.evaluate(x-shift)[0])/2e-6
                assert_allclose(jac[:,j],numeric,rtol=1e-5,atol=1e-8)

    def test_roundtrip_keeps_empirical_prior_and_diagnostics(self):
        flex=self.make()
        rng=np.random.default_rng(1)
        blocks,nell=flex.layout.n_blocks,flex.ell.size
        flex.data_vector=np.ones((blocks,nell))
        flex.deriv=rng.normal(size=(flex.n_params,blocks,nell))
        flex.cov_blocks=np.repeat(np.eye(blocks)[None],nell,axis=0)
        flex.chi_eff_lens_Mpc=np.ones(flex.n_lens)*1000
        flex._computed=True
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.npz'
            flex.save(path)
            loaded=ff.FisherFlex.from_npz(path,verbose=False)
            assert_array_equal(loaded.forecast().cov,flex.forecast().cov)
            self.assertEqual(loaded.gpro_calibration,flex.gpro_calibration)

if __name__=='__main__':
    unittest.main(verbosity=2)
