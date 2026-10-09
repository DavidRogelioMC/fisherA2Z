"""Fit Gaussian convolution parameters to an ensemble with FIXED mixing.

Each row of realizations is a joint tomographic draw. First undo the fixed
mixing matrix, then fit each component by integrated squared PDF error.
This defines the fitting metric in component space, not in observable space.
Negative values after undoing mixing are retained as targets, not clipped.

Fiducials are the MEAN FITTED PARAMETERS and the prior is their sample
covariance (not covariance divided by the number of draws). This differs
from fitting the ensemble-mean PDF, which need not be represented exactly.
Boundary fractions and fit residuals must be inspected: the Gaussian Fisher
prior is only an approximation to this potentially non-Gaussian ensemble.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve


def _weights(z):
    h = np.diff(z)
    return np.r_[h[0]/2, (h[:-1]+h[1:])/2, h[-1]/2]


class _ConvolutionFit:
    """The same trapezoidal convolution as gaussian_outlier_pro, with Jacobian."""
    def __init__(self, z, template):
        self.z = z
        self.weights = _weights(z)
        self.sqrtw = np.sqrt(self.weights)
        self.weighted = template * self.weights
        self.uniform = np.allclose(np.diff(z), np.diff(z)[0], rtol=1e-10, atol=1e-13)
        if self.uniform:
            self.distance = np.arange(1-z.size, z.size)*np.diff(z)[0]
        else:
            self.distance = z[:, None]-z[None, :]
        self.lower_width = 2*np.max(np.diff(z))*(1+1e-4)
        self.span = z[-1]-z[0]
        self.mean = np.sum(self.weighted*z)
        self.variance = np.sum(self.weighted*(z-self.mean)**2)
        self.last_x = None

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return self.pdf, self.jac
        shift, width = x
        u = (self.distance-shift)/width
        kernel = np.exp(-u*u/2)/(np.sqrt(2*np.pi)*width)
        arrays = []
        for k in (kernel, kernel*u/width, kernel*(u*u-1)/width):
            if self.uniform:
                value = fftconvolve(self.weighted, k)[self.z.size-1:2*self.z.size-1]
            else:
                value = k @ self.weighted
            arrays.append(value)
        mass = arrays[0] @ self.weights
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError('Convolved template has no mass on the calibration support.')
        self.pdf = arrays[0]/mass
        self.jac = np.column_stack([(d-self.pdf*(d@self.weights))/mass for d in arrays[1:]])
        self.last_x = np.array(x, copy=True)
        return self.pdf, self.jac

    def fit(self, target):
        mu = np.sum(target*self.weights*self.z)
        var = np.sum(target*self.weights*(self.z-mu)**2)
        start = np.array([mu-self.mean, np.sqrt(max(var-self.variance, (2*self.lower_width)**2))])
        lower = np.array([-self.span, self.lower_width])
        upper = np.array([self.span, self.span])
        start = np.clip(start, lower+1e-8, upper-1e-8)
        def residual(x):
            return (self.evaluate(x)[0]-target)*self.sqrtw
        def jacobian(x):
            return self.evaluate(x)[1]*self.sqrtw[:, None]
        result = least_squares(residual, start, jac=jacobian, bounds=(lower, upper),
                               x_scale='jac', max_nfev=100, ftol=1e-7, xtol=1e-7, gtol=1e-9)
        if not result.success:
            result = least_squares(residual, result.x, jac=jacobian, bounds=(lower, upper),
                                   x_scale='jac', max_nfev=300, ftol=1e-7, xtol=1e-7, gtol=1e-9)
        if not result.success:
            raise RuntimeError('Gaussian convolution fit did not converge; no realization was discarded.')
        boundary = result.x[1] <= self.lower_width*1.001
        relative_error = np.sqrt(2*result.cost / max(np.sum(target**2*self.weights), 1e-300))
        return result.x, boundary, relative_error


def fit_chunk(task):
    """Independent batch; also callable by a multiprocessing pool."""
    bin_index, start, z, template, z_real, targets = task
    fitter = _ConvolutionFit(z, template)
    parameters = np.empty((targets.shape[0], 2))
    boundary = np.zeros(targets.shape[0], dtype=bool)
    errors = np.empty(targets.shape[0])
    for i, target in enumerate(targets):
        dense_target = np.interp(z, z_real, target)
        parameters[i], boundary[i], errors[i] = fitter.fit(dense_target)
    return bin_index, start, parameters, boundary, errors


def fit_ensemble(z, templates, z_real, real, c, pool=None, progress=None):
    """Use ALL realizations; return fitted parameters and calibration diagnostics.

    z_real is explicit when samples use a different grid. Fine-grid arrays
    are constructed only for one realization at a time. A supplied process
    pool must support imap; the caller manages its notebook-safe lifecycle.
    """
    from . import nz_gaussian_outlier_pro as go
    z = go._grid(z)
    z_real = go._grid(z_real)
    templates = go.normalize_nz(z, templates)
    real = np.asarray(real, dtype=float)
    ns = templates.shape[0]
    if real.ndim != 3 or real.shape[1:] != (ns, z_real.size):
        raise ValueError('real must have shape (n_realizations, n_bins, len(z_realizations)).')
    if not np.allclose(z[[0,-1]], z_real[[0,-1]], rtol=1e-12, atol=1e-12):
        raise ValueError('The realization and Fisher grids must cover the same redshift interval.')
    if real.shape[0] <= 2:
        raise ValueError('At least three realizations are needed for two-parameter prior blocks.')
    c = go.validate_mixing(c, ns)
    mixing = c + np.diag(1-c.sum(axis=1))
    if np.linalg.cond(mixing) > 1e8:
        raise ValueError('The fixed mixing matrix is too ill-conditioned to calibrate its components.')
    inverse = np.linalg.inv(mixing)
    parameters = np.empty((real.shape[0], ns, 2))
    boundary = np.zeros((real.shape[0], ns), dtype=bool)
    errors = np.empty((real.shape[0], ns))

    def tasks():
        for start in range(0, real.shape[0], 128):
            stop = min(start+128, real.shape[0])
            batch = go.normalize_nz(z_real, real[start:stop])
            components = np.einsum('ij,rjz->riz', inverse, batch)
            for b in range(ns):
                yield b, start, z, templates[b], z_real, components[:,b]

    # Bounded submission: imap can prefetch the complete ensemble and needlessly
    # retain all demixed PDFs. Submit at most two tasks per worker instead.
    def bounded_results():
        from collections import deque
        pending = deque()
        task_iterator = iter(tasks())
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < 2*pool._processes:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(pool.apply_async(fit_chunk, (task,)))
            if pending:
                yield pending.popleft().get()
    results = map(fit_chunk, tasks()) if pool is None else bounded_results()
    completed = 0
    for b, start, values, flags, residuals in results:
        stop = start+len(values)
        parameters[start:stop,b] = values
        boundary[start:stop,b] = flags
        errors[start:stop,b] = residuals
        completed += len(values)
        if progress:
            progress(completed, real.shape[0]*ns)
    report = dict(n_realizations=int(real.shape[0]), n_bins=ns,
                  n_realization_nodes=int(z_real.size), n_fisher_nodes=int(z.size),
                  method='nonlinear_least_squares_in_unmixed_pdf_space',
                  fiducial_definition='mean_fitted_parameters',
                  lower_width=float(2*np.max(np.diff(z))*(1+1e-4)),
                  boundary_fraction=boundary.mean(axis=0).tolist(),
                  median_relative_fit_error=np.median(errors,axis=0).tolist(),
                  p95_relative_fit_error=np.quantile(errors,.95,axis=0).tolist(),
                  cross_bin_prior_correlations='not included; per-bin blocks only')
    return parameters, report
