"""Continuous MaxEnt: reference equation, quadrature, multipliers and derivatives."""
import unittest
import numpy as np
from scipy.optimize import fsolve
from fisherA2Z import nz_moments as nm


class MomentNumericsTests(unittest.TestCase):
    def test_exact_input_moments_on_nonuniform_grid(self):
        z=np.array([0., .03, .19, .71, 1.])
        np.testing.assert_allclose(nm.raw_moments(z,2*z,8),
                                   2/(np.arange(8)+2),rtol=2e-14,atol=2e-14)

    def test_input_moments_and_prior_coordinates_survive_linear_regridding(self):
        z=np.array([0., .08, .3, .7, 1.2, 2.7, 3.])
        n=np.array([0., .4, 1., .2, .7, .02, 0.])
        fine=np.unique(np.concatenate([np.linspace(a,b,33) for a,b in zip(z[:-1],z[1:])]))
        np.testing.assert_allclose(nm.raw_moments(z,n,8),
            nm.raw_moments(fine,123*np.interp(fine,z,n),8),rtol=3e-14,atol=2e-14)

    def test_agrees_with_literal_numpy_translation_of_reference_continuous_branch(self):
        # Independent transcription of pymaxent_jax (float64), not its unavailable JAX runtime.
        xs=np.linspace(0.,1.,1000)
        p=np.exp(.6*xs-2*xs**2);p/=np.trapz(p,xs)
        mu=np.array([np.trapz(xs**k*p,xs) for k in range(3)])
        def original_integrand(x,lamb,k=0):
            xi=np.array([x**i for i in range(len(lamb))])
            return x**k*np.exp(np.dot(lamb,xi))
        def original_residual(lamb):
            return np.array([np.trapz(original_integrand(xs,lamb,k),xs)-mu[k]
                             for k in range(len(mu))])
        reference=fsolve(original_residual,np.zeros(3))
        sol,lambdas=nm.reconstruct(mu,bnds=[0.,1.])
        np.testing.assert_allclose(lambdas,reference,rtol=1e-8,atol=1e-8)
        points=np.array([0.,.023,.227,.581,.991,1.])
        np.testing.assert_allclose(sol(points),original_integrand(points,reference),rtol=1e-9)

    def test_returns_physical_lagrange_multipliers_and_sampling_does_not_change_solution(self):
        z=np.linspace(.05,2.5,181)
        fid=nm.raw_moments(z,np.exp(-.5*((z-.9)/.35)**2),4)
        small=np.linspace(z[0],z[-1],19);fine=np.linspace(z[0],z[-1],1000)
        density,lambdas=nm.evaluate_moment_nz(small,fid,np.zeros(3),return_lambdas=True)
        dense,other=nm.evaluate_moment_nz(fine,fid,np.zeros(3),return_lambdas=True)
        np.testing.assert_array_equal(lambdas,other)
        np.testing.assert_allclose(density,np.exp(sum(lambdas[k]*small**k for k in range(4))),rtol=1e-12)
        np.testing.assert_allclose([np.trapz(fine**k*dense,fine) for k in range(4)],fid,atol=1e-10)

    def test_uniform_and_exponential_known_maximum_entropy_solutions(self):
        x=np.linspace(0.,1.,1000)
        for exponent in (0.,1.):
            truth=np.exp(exponent*x);truth/=np.trapz(truth,x)
            mu=np.array([np.trapz(x**k*truth,x) for k in range(4)])
            sol,lambdas=nm.reconstruct(mu,bnds=[0.,1.])
            np.testing.assert_allclose(sol(x),truth,rtol=2e-9,atol=2e-9)
            np.testing.assert_allclose(nm.residual_c(lambdas,mu,[0.,1.]),0.,atol=2e-11)

    def test_each_shift_satisfies_continuous_moment_constraints(self):
        z=np.linspace(.01,3,301)
        m=nm.build_moment_model(z,np.exp(-.5*((z-1.1)/.6)**2)+.2,8)
        fid=m.moments[0];steps=nm.recommended_steps(z,fid)
        x=np.linspace(z[0],z[-1],1000)
        for k,h in enumerate(steps):
            for sign in (-1.,1.):
                shift=np.zeros(7);shift[k]=sign*h
                density,lambdas=nm.evaluate_moment_nz(x,fid,shift,return_lambdas=True)
                target=fid+np.r_[0.,shift]
                self.assertTrue(np.all(density>=0))
                np.testing.assert_allclose([np.trapz(x**j*density,x) for j in range(8)],
                                           target,atol=5e-10,rtol=5e-10)
        self.assertEqual(len(m.lambdas[0]),8)

    def test_smooth_observable_derivative_convergence(self):
        z=np.linspace(0.,3.,301)
        fid=nm.raw_moments(z,np.exp(-z)+.1,7)
        steps=nm.recommended_steps(z,fid,.003)
        kernel=np.exp(-.8*z)
        for k in (0,5):
            derivatives=[]
            for scale in (1.,.5):
                shift=np.zeros(6);shift[k]=steps[k]*scale
                plus=nm.evaluate_moment_nz(z,fid,shift)
                minus=nm.evaluate_moment_nz(z,fid,-shift)
                derivatives.append(np.trapz(kernel*(plus-minus),z)/(2*shift[k]))
            np.testing.assert_allclose(derivatives[0],derivatives[1],rtol=.002,atol=1e-7)

    def test_pdf_maxent_style_wrappers(self):
        z=np.linspace(.01,2.,100)
        p=np.exp(-z)
        x,y,lambdas=nm.reconstruct_from_moments(p,z,4,np.zeros(4),return_lambdas=True)
        np.testing.assert_array_equal(x,z)
        np.testing.assert_allclose(y,nm.integrand(z,lambdas))
        xs,solutions=nm.run([p,p],z,2,[3,4],[np.zeros(2),np.zeros(3)])
        np.testing.assert_array_equal(xs[0],z)
        self.assertEqual(float(solutions[0](-1.)),0.)
        self.assertTrue(np.all(solutions[1](z)>=0))

    def test_bad_moments_pdf_and_solver_inputs_raise(self):
        z=np.linspace(0,2,101)
        for moments in ([.9,1.,1.2],[1.,3.,10.],[1.,1.,.5],[1.,1.,8.]):
            with self.assertRaises(ValueError):
                nm.evaluate_moment_nz(z,np.array(moments),np.zeros(2))
        for p in (np.zeros(101),-np.ones(101),np.full(101,np.nan)):
            with self.assertRaises(ValueError):nm.raw_moments(z,p,4)
        with self.assertRaises(ValueError):nm.reconstruct([1.,.5],bnds=[0.,1.],maxiter=0)
        with self.assertRaises(ValueError):nm.reconstruct([1.,.5],bnds=[0.,1.],lambguess=[0.])
        with self.assertRaises(ValueError):nm.evaluate_moment_nz(z,[1.,.5],[0.],tolerance=-1)
        with self.assertRaises(ValueError):nm.reconstruct([1.,.5],rndvar=z)

    def test_singular_and_indefinite_priors_are_not_regularized(self):
        for cov in (np.ones((2,2)),np.array([[1.,2.],[2.,1.]]),np.zeros((2,2))):
            with self.assertRaises(ValueError):nm.validate_prior_cov([cov],[3])


if __name__=='__main__':unittest.main()
