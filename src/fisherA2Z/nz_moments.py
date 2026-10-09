"""Continuous maximum-entropy n(z), following pdf_maxent / pymaxent_jax.

The reconstructed PDF is n(z) = exp(sum(lambdas[k] * z**k)). We solve
its moment equations with scipy.optimize.fsolve and 1000 trapezoid nodes,
as in pymaxent_jax.py. There is no discrete-entropy or Legendre solver.

The continuous reconstruction functions below are adapted from the MIT
PyMaxEnt code by Tony Saad and Giovanna Ruai (copyright 2019 Tony Saad),
as supplied in the user's pymaxent_jax.py. NumPy replaces JAX; SciPy fsolve
is the solver in both implementations. See MAXENT_NOTICE.txt.

Fisher fixes m0=1 and varies additive RAW m1,...,mK. Its input moments keep
the existing exact integration of the piecewise-linear input PDF, so the
prior is unchanged by linear regridding. This intentionally differs from
the rectangular sum in pdf_maxent.py. Output values sample the continuous
exponential: their coarse-grid trapezoidal moments need not match exactly.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

DEFAULT_N_MOMENTS = (7, 8, 4, 4, 4)
N_INTEGRATION_POINTS = 1000
MODEL_VERSION = "continuous_maxent_v1"

def _grid(z):
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 3 or not np.all(np.isfinite(z)):
        raise ValueError("z must be a finite one-dimensional grid with >=3 nodes.")
    if z[0] < 0 or np.any(np.diff(z) <= 0):
        raise ValueError("z must be nonnegative and strictly increasing.")
    return z

def _weights(z):
    dz = np.diff(z)
    return np.r_[dz[0] / 2, (dz[:-1] + dz[1:]) / 2, dz[-1] / 2]

def normalize_nz(z, nz):
    z = _grid(z)
    nz = np.asarray(nz, dtype=float)
    if nz.ndim < 1 or nz.shape[-1] != z.size:
        raise ValueError("The last PDF axis must match the redshift grid.")
    if not np.all(np.isfinite(nz)) or np.any(nz < 0):
        raise ValueError("PDF values must be finite and nonnegative.")
    area = np.sum(nz * _weights(z), axis=-1)
    if np.any(area <= 0) or not np.all(np.isfinite(area)):
        raise ValueError("Every PDF must have a finite positive integral.")
    return nz / area[..., None]

def _n_moments(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError("n_moments must be an integer >=2, including m0.")
    integer = int(value)
    if integer != value or integer < 2:
        raise ValueError("n_moments must be an integer >=2, including m0.")
    return integer

def moment_counts(n_moments, n_source):
    """Expand a scalar or validate one moment count (including m0) per bin."""
    if np.isscalar(n_moments):
        return tuple(_n_moments(n_moments) for _ in range(n_source))
    counts = tuple(_n_moments(n) for n in n_moments)
    if len(counts) != n_source:
        raise ValueError("Supply one n_moments entry per source bin.")
    return counts

@lru_cache(maxsize=16)
def _moment_weights(z_bytes, size, n_moments):
    """Exact integral weights for z**k times a piecewise-linear nodal PDF."""
    z = np.frombuffer(z_bytes, dtype=np.float64, count=size) 
    nodes, weights = leggauss((n_moments + 2) // 2)
    t = (nodes + 1) / 2
    dz = np.diff(z)
    points = z[:-1, None] + dz[:, None] * t
    powers = points[..., None] ** np.arange(n_moments)[None, None, :]
    common = dz[:, None] * weights[None, :] / 2
    result = np.zeros((size, n_moments))
    result[:-1] += np.einsum("ig,igk,g->ik", common, powers, 1 - t)
    result[1:] += np.einsum("ig,igk,g->ik", common, powers, t)
    result[:, 0] = _weights(z)
    return result

def raw_moments(z, nz, n_moments):
    """Integrate m0,...,mK of normalized PIECEWISE-LINEAR input PDFs.

    Leading dimensions are preserved. Polynomial-weighted linear segments
    are integrated exactly by Gaussian quadrature; refining z with linear
    interpolation leaves these moments unchanged, up to roundoff.
    """
    z = _grid(z)
    n_moments = _n_moments(n_moments)
    pdf = normalize_nz(z, nz)
    weights = _moment_weights(z.tobytes(), z.size, n_moments)
    result = pdf @ weights
    result[..., 0] = 1.0
    return result

def integrand(x, lamb, k=0):
    """Integrand of the kth moment; lamb contains the Lagrange multipliers."""
    x = np.asarray(x, dtype=float)
    xi = np.array([x**i for i in range(len(lamb))])
    with np.errstate(over="ignore", invalid="ignore"):
        return x**k * np.exp(np.tensordot(lamb, xi, axes=1))


def residual_c(lamb, mu, bnds):
    """Moment integrals minus targets, using the reference's 1000-node rule."""
    xs = np.linspace(bnds[0], bnds[1], N_INTEGRATION_POINTS)
    density = integrand(xs, lamb)
    return np.array([np.trapz(xs**k * density, xs) - mu[k]
                     for k in range(len(mu))])


def _moment_jacobian(lamb, bnds):
    """d residual[k] / d lambda[j] = integral z**(k+j) n(z) dz."""
    xs = np.linspace(bnds[0], bnds[1], N_INTEGRATION_POINTS)
    powers = xs[:, None] ** np.arange(len(lamb))
    return (powers.T * (_weights(xs) * integrand(xs, lamb))) @ powers


def maxent_reconstruct_c(mu, bnds, *, lambguess=None, tolerance=2e-12, maxiter=150):
    """Reconstruct a continuous distribution from its known moments.

    Returns (recon, lambsol), exactly the two outputs of pymaxent_jax.
    The analytic Jacobian and residual check make fsolve usable for Fisher
    derivatives; a failed solve raises instead of returning a spurious PDF.
    """
    mu = np.asarray(mu, dtype=float)
    bounds = np.asarray(bnds, dtype=float)
    if (mu.ndim != 1 or mu.size < 2 or not np.all(np.isfinite(mu))
            or mu[0] <= 0 or bounds.shape != (2,)
            or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]):
        raise ValueError("Supply finite moments (m0>0) and finite increasing bounds.")
    if not bounds[0] < mu[1] / mu[0] < bounds[1]:
        raise ValueError("The requested mean must be strictly inside the support.")
    if mu.size > 2 and mu[2] / mu[0] <= (mu[1] / mu[0])**2:
        raise ValueError("Requested moments have nonpositive variance.")
    if tolerance <= 0 or not np.isfinite(tolerance) or maxiter < 1:
        raise ValueError("tolerance and maxiter must be positive.")
    if lambguess is None:
        lambguess = np.zeros(len(mu))
        lambguess[0] = np.log(mu[0] / (bounds[1] - bounds[0]))
    lambguess = np.asarray(lambguess, dtype=float)
    if lambguess.shape != mu.shape or not np.all(np.isfinite(lambguess)):
        raise ValueError("lambguess must contain one finite multiplier per moment.")
    lambsol, info, status, message = fsolve(
        residual_c, lambguess, args=(mu, bounds),
        fprime=lambda lamb, *_: _moment_jacobian(lamb, bounds),
        full_output=True, xtol=tolerance, maxfev=maxiter * (len(mu) + 1), factor=0.1)
    residual = residual_c(lambsol, mu, bounds)
    error = np.max(np.abs(residual) / np.maximum(1.0, np.abs(mu)))
    if not np.all(np.isfinite(lambsol)) or not np.isfinite(error) or error > 5e-10:
        raise ValueError("Maximum-entropy fsolve failed: residual={:.3g}; {}"
                         .format(error, message.strip()))
    recon = lambda x: integrand(x, lambsol, k=0)
    return recon, lambsol


def reconstruct(mu, rndvar=None, bnds=None, **kwargs):
    """Main call: return the continuous maximum-entropy PDF and multipliers.

    Only the continuous branch used by pdf_maxent/Fisher is implemented.
    """
    if rndvar is not None:
        raise ValueError("Use bnds=[z_min, z_max]; this n(z) model is continuous.")
    return maxent_reconstruct_c(mu, bnds, **kwargs)


@lru_cache(maxsize=32)
def _fiducial_lambdas(moments, bounds):
    """Reuse the fiducial solve as an initial guess for nearby Fisher shifts."""
    return reconstruct(np.asarray(moments), bnds=bounds)[1]


def evaluate_moment_nz(z, moments_fid, params, *, tolerance=2e-12,
                       maxiter=150, return_lambdas=False):
    """Sample the continuous maximum-entropy PDF after shifting raw moments.

    m0=1 is fixed. With return_lambdas=True, also return lambda_0,...,lambda_K
    in physical powers of redshift (not scaled coordinates or moments).
    No post-normalization is applied to the sampled output: that would change
    the solved continuous distribution and its moment constraints.
    """
    z = _grid(z)
    moments = np.asarray(moments_fid, dtype=float).copy()
    shifts = np.asarray(params, dtype=float)
    if (moments.ndim != 1 or moments.size < 2
            or shifts.shape != (moments.size - 1,)
            or not np.all(np.isfinite(moments)) or not np.all(np.isfinite(shifts))):
        raise ValueError("Supply finite fiducials and n_moments-1 moment shifts.")
    if not np.isclose(moments[0], 1., atol=1e-13, rtol=0):
        raise ValueError("The zeroth moment is fixed to 1; do not shift it.")
    if tolerance <= 0 or not np.isfinite(tolerance) or maxiter < 1:
        raise ValueError("tolerance and maxiter must be positive.")
    bounds = (float(z[0]), float(z[-1]))
    guess = _fiducial_lambdas(tuple(moments), bounds)
    moments[1:] += shifts
    if np.any(shifts) or tolerance != 2e-12 or maxiter != 150:
        sol, lambdas = reconstruct(moments, bnds=bounds, lambguess=guess,
                                   tolerance=tolerance, maxiter=maxiter)
        density = sol(z)
    else:
        lambdas = guess.copy()
        density = integrand(z, lambdas)
    return (density, lambdas) if return_lambdas else density


def reconstruct_from_moments(p, x, N_moments, shifts=None, *, return_lambdas=False):
    """pdf_maxent-style wrapper; preserve m0=1 and use the Fisher input moments.

    shifts has N_moments-1 entries, or N_moments entries with shifts[0]=0.
    Returns (x, density), or (x, density, lambdas) when requested.
    """
    x = _grid(x)
    moments = raw_moments(x, p, N_moments)
    shifts = np.zeros(N_moments - 1) if shifts is None else np.asarray(shifts, dtype=float)
    if shifts.shape == (N_moments,) and shifts[0] == 0:
        shifts = shifts[1:]
    density, lambdas = evaluate_moment_nz(x, moments, shifts, return_lambdas=True)
    return (x, density, lambdas) if return_lambdas else (x, density)


def run(nzs, zs, N, N_moments, shifts=None):
    """pdf_maxent-style multi-bin wrapper, returning grids and interpolators.

    shifts is None or one vector of free moment shifts per bin. Interpolation
    uses zero outside the support; it does not extrapolate a PDF to negatives.
    """
    if len(nzs) != N:
        raise ValueError("N must match the number of input distributions.")
    counts = moment_counts(N_moments, N)
    shifts = [None] * N if shifts is None else shifts
    if len(shifts) != N:
        raise ValueError("Supply one shift vector per bin.")
    xs, sol = [], []
    for i in range(N):
        x_0, sol_0 = reconstruct_from_moments(nzs[i], zs, counts[i], shifts[i])
        xs.append(x_0)
        sol.append(interp1d(x_0, sol_0, bounds_error=False, fill_value=0.))
    return xs, sol


@dataclass
class MomentModel:
    nz_fid: np.ndarray
    moments: tuple
    n_moments: tuple
    pivot: np.ndarray
    sigma: np.ndarray
    lambdas: tuple


def build_moment_model(z, nz_source, n_moments=DEFAULT_N_MOMENTS):
    """Build each fiducial continuous PDF and retain its Lagrange multipliers."""
    z = _grid(z)
    pdfs = np.atleast_2d(normalize_nz(z, nz_source))
    if pdfs.ndim != 2:
        raise ValueError("nz_source must have shape (n_source,n_z).")
    counts = moment_counts(n_moments, len(pdfs))
    moments = tuple(raw_moments(z, p, n) for p, n in zip(pdfs, counts))
    solutions = [evaluate_moment_nz(z, m, np.zeros(n-1), return_lambdas=True)
                 for m, n in zip(moments, counts)]
    densities, lambdas = zip(*solutions)
    xs = np.linspace(z[0], z[-1], N_INTEGRATION_POINTS)
    first_three = np.array([[np.trapz(xs**k * integrand(xs, l), xs)
                             for k in range(3)] for l in lambdas])
    pivot = first_three[:, 1]
    sigma = np.sqrt(np.maximum(first_three[:, 2] - pivot**2, 0.))
    return MomentModel(np.array(densities), moments, counts, pivot, sigma, lambdas)


def fit_moments(z, real, n_moments, moments_fid):
    """Return a tuple of (n_realizations,n_free_bin) raw-moment shifts.

    real is (n_realizations,n_source,n_z). No max-entropy reconstruction
    is necessary when measuring moments in the input ensemble.
    """
    z = _grid(z)
    real = np.asarray(real, dtype=float)
    if real.ndim != 3 or real.shape[-1] != z.size:
        raise ValueError("real must have shape (n_realizations,n_source,n_z).")
    counts = moment_counts(n_moments, real.shape[1])
    if len(moments_fid) != real.shape[1]:
        raise ValueError("Supply fiducial moments for every source bin.")
    result = []
    for b, n in enumerate(counts):
        fid = np.asarray(moments_fid[b], dtype=float)
        if fid.shape != (n,):
            raise ValueError("Incorrect fiducial moment count in bin {}.".format(b))
        result.append(raw_moments(z, real[:, b], n)[:, 1:] - fid[None, 1:])
    return tuple(result)

def validate_prior_cov(cov, n_moments=DEFAULT_N_MOMENTS, n_source=None):
    """Validate heterogeneous per-bin RAW-moment covariances, without regularization."""
    if n_source is None:
        n_source = len(cov)
    counts = moment_counts(n_moments, n_source)
    if len(cov) != n_source:
        raise ValueError("Supply one covariance block per source bin.")
    result = []
    for b, n in enumerate(counts):
        block = np.asarray(cov[b], dtype=float)
        if block.shape != (n - 1, n - 1) or not np.all(np.isfinite(block)):
            raise ValueError("Prior block {} must have shape ({},{}), finite values.".format(b, n - 1, n - 1))
        if np.any(np.diag(block) <= 0):
            raise ValueError("Prior variances must be strictly positive in bin {}.".format(b))
        sd = np.sqrt(np.diag(block))
        correlation = block / np.outer(sd, sd)
        if not np.allclose(correlation, correlation.T, atol=1e-10, rtol=1e-10):
            raise ValueError("Prior covariance must be symmetric in bin {}.".format(b))
        eig = np.linalg.eigvalsh(correlation)
        threshold = (n - 1) * np.finfo(float).eps * eig[-1]
        if eig[0] <= threshold:
            raise ValueError("Prior covariance is singular or numerically rank deficient in bin {}.".format(b))
        result.append(block)
    return tuple(result)

def recommended_steps(z, moments_fid, default_step=0.001):
    """Choose raw-moment steps from the probability-weighted log-PDF response.

    H d(lambda) = d(moment), where H[k,j] = integral z**(k+j) n(z) dz.
    The RMS response weights the populated part of n(z), so tiny tails do
    not force steps below the numerical noise of the CCL spectra.
    These numerical derivative steps are not prior widths.
    """
    z = _grid(z)
    fid = np.asarray(moments_fid, dtype=float)
    if not np.isscalar(default_step) or not np.isfinite(default_step) or default_step <= 0:
        raise ValueError("default_step must be positive and finite.")
    bounds = (float(z[0]), float(z[-1]))
    lambdas = _fiducial_lambdas(tuple(fid), bounds)
    hessian = _moment_jacobian(lambdas, bounds)
    xs = np.linspace(z[0], z[-1], N_INTEGRATION_POINTS)
    powers = xs[:, None] ** np.arange(len(fid))
    mass = _weights(xs) * integrand(xs, lambdas)
    result = []
    for k in range(1, len(fid)):
        rhs = np.zeros(len(fid)); rhs[k] = 1.
        response = np.linalg.solve(hessian, rhs)
        sensitivity = np.sqrt(np.sum(mass * (powers @ response)**2))
        if not np.isfinite(sensitivity) or sensitivity <= 0:
            raise ValueError("Invalid moment sensitivity when choosing a derivative step.")
        step = default_step / sensitivity
        for attempt in range(15):
            shift = np.zeros(len(fid)-1); shift[k-1] = 2*step
            try:
                evaluate_moment_nz(z, fid, shift)
                evaluate_moment_nz(z, fid, -shift)
                break
            except ValueError:
                step *= .5
        else:
            raise ValueError("No reliable finite-difference step for moment {}.".format(k))
        if step <= 128*abs(np.spacing(fid[k])):
            raise ValueError("Moment {} needs a step below numerical precision.".format(k))
        result.append(step)
    return np.array(result)
