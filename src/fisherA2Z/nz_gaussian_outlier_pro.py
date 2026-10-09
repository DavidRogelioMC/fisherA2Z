"""Gaussian convolution followed by mixing between tomographic bins.

Adapted from the user's ``Gaussian_with_outliers.py``.  Its useful function
names are retained, with NumPy instead of JAX.  Unlike FisherA2Z's original
``gaussian_outlier``, the input PDFs need not be Gaussian and the outliers
come from other bins, not from a fitted residual within the same bin.

First convolve each template n_j with N(dz_j, std_b_j**2), then normalize
on the supplied finite redshift support to obtain g_j.  The output is

    n_i = (1 - sum_{j != i} c[i,j]) * g_i + sum_{j != i} c[i,j] * g_j.

Thus c[i,j] is the fraction of OUTPUT bin i supplied by template j.  This
does not move galaxy counts or change neff.  The diagonal of c is zero.
The Gaussian width std_b is in redshift units, NOT a variance or a factor
multiplying the original PDF width.  A positive dz shifts toward larger z.

Corrections to the reference: integrate with redshift weights, compute all
templates before mixing, use one row of c per output bin, normalize each
component on the finite support, and return zero outside that support.
The reference cites https://arxiv.org/abs/2007.12795 .
"""

import numpy as np
from scipy.interpolate import interp1d


MODEL_VERSION = "gaussian_convolution_row_mixing_v1"


def _grid(z):
    z = np.asarray(z, dtype=float)
    if (z.ndim != 1 or z.size < 2 or not np.all(np.isfinite(z))
            or np.any(np.diff(z) <= 0)):
        raise ValueError("Redshift nodes must be finite and strictly increasing.")
    return z


def normalize_nz(z, nz):
    """Normalize each nonnegative PDF using trapezoidal redshift weights."""
    z = _grid(z)
    nz = np.asarray(nz, dtype=float)
    if (nz.ndim < 1 or nz.shape[-1] != z.size
            or not np.all(np.isfinite(nz)) or np.any(nz < 0)):
        raise ValueError("PDFs must be finite, nonnegative and match the grid.")
    mass = np.trapz(nz, z, axis=-1)
    if np.any(mass <= 0):
        raise ValueError("Each PDF must have positive mass on the grid.")
    return nz / np.expand_dims(mass, -1)


def _per_bin(value, n_bins, name):
    value = np.asarray(value, dtype=float)
    if value.ndim > 1 or (value.ndim == 1 and value.size not in (1, n_bins)):
        raise ValueError(f"{name} must be a scalar or one value per bin.")
    result = np.broadcast_to(value, (n_bins,)).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


def validate_mixing(c, n_bins):
    """Check c[i,j] >= 0, c[i,i] = 0, and sum_j c[i,j] <= 1."""
    c = np.asarray(c, dtype=float)
    if (c.shape != (n_bins, n_bins) or not np.all(np.isfinite(c))
            or np.any(c < 0) or np.any(np.diag(c) != 0)
            or np.any(c.sum(axis=1) > 1)):
        raise ValueError("c must be a finite nonnegative square matrix with "
                         "zero diagonal and each row sum <= 1.")
    return c.copy()


def gaussian_int(z, nzs_b, zs, dz, std_b):
    """Integral n(t) * N(z-t; dz, std_b**2) dt at one or several z.

    Uses the trapezoid rule, including on nonuniform grids.  The grid must
    resolve the Gaussian width; refining only the plotting grid is not enough.
    Finite-support normalization is performed later by ``get_dnGdz``.
    """
    zs = _grid(zs)
    nzs_b = np.asarray(nzs_b, dtype=float)
    if (nzs_b.shape != zs.shape or not np.all(np.isfinite(nzs_b))
            or np.any(nzs_b < 0)):
        raise ValueError("nzs_b must be one finite nonnegative PDF on zs.")
    dz, std_b = float(dz), float(std_b)
    if not np.isfinite(dz) or not np.isfinite(std_b) or std_b <= 0:
        raise ValueError("dz must be finite and std_b must be positive and finite.")
    z = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("Evaluation redshifts must be finite.")
    delta = (z[..., None] - zs - dz) / std_b
    kernel = np.exp(-0.5 * delta**2) / (np.sqrt(2 * np.pi) * std_b)
    return np.trapz(kernel * nzs_b, zs, axis=-1)


def get_dnGdz(nzs, zs, Nbins, dz, std_b):
    """Convolve ALL input bins, returning (normalized PDFs, redshift grid).

    dz and std_b may be scalars or arrays of length Nbins.  Small batches
    bound the temporary kernel memory when evaluating a fine redshift grid.
    """
    zs = _grid(zs)
    nzs = normalize_nz(zs, nzs)
    if nzs.shape != (Nbins, zs.size):
        raise ValueError("nzs must have shape (Nbins, len(zs)).")
    dz = _per_bin(dz, Nbins, "dz")
    std_b = _per_bin(std_b, Nbins, "std_b")
    if np.any(std_b <= 0):
        raise ValueError("Every std_b must be positive.")
    # An unresolved kernel gives incorrect derivatives even after normalization.
    if np.any(std_b < 2 * np.max(np.diff(zs))):
        raise ValueError("The redshift grid does not resolve std_b: refine zs "
                         "and interpolate the templates so max(diff(zs)) <= std_b/2.")
    dnGdz = np.empty_like(nzs)
    for b in range(Nbins):
        for start in range(0, zs.size, 256):
            stop = min(start + 256, zs.size)
            dnGdz[b, start:stop] = gaussian_int(
                zs[start:stop], nzs[b], zs, dz[b], std_b[b])
    return normalize_nz(zs, dnGdz), zs.copy()


def reconstructed_pdf(dnGdz, c, binn, Nbins):
    """Mix row binn of c; each row of dnGdz is already normalized."""
    dnGdz = np.asarray(dnGdz, dtype=float)
    c = validate_mixing(c, Nbins)
    if (dnGdz.ndim != 2 or dnGdz.shape[0] != Nbins
            or not np.all(np.isfinite(dnGdz)) or np.any(dnGdz < 0)):
        raise ValueError("dnGdz must contain one finite nonnegative PDF per bin.")
    if not isinstance(binn, (int, np.integer)) or not 0 <= binn < Nbins:
        raise ValueError("binn must be a zero-based bin index.")
    return (1 - c[binn].sum()) * dnGdz[binn] + c[binn] @ dnGdz


def outlier_fraction(c):
    """Outlier fraction PER OUTPUT BIN, without summing over other rows."""
    c = np.asarray(c, dtype=float)
    if c.ndim != 2:
        raise ValueError("c must be a square matrix.")
    return validate_mixing(c, c.shape[0]).sum(axis=1)


def evaluate_nz(nzs, zs, dz, std_b, c):
    """Convolve the templates and mix them; return shape (n_bins, n_z)."""
    nzs = np.asarray(nzs, dtype=float)
    if nzs.ndim != 2:
        raise ValueError("nzs must have shape (n_bins, n_z).")
    n_bins = nzs.shape[0]
    c = validate_mixing(c, n_bins)
    dnGdz, _ = get_dnGdz(nzs, zs, n_bins, dz, std_b)
    return np.array([reconstructed_pdf(dnGdz, c, b, n_bins)
                     for b in range(n_bins)])


def run(nzs, zs, Nbins, dz, std_b, c):
    """Reference-style interface: return (zgrid, list of PDF callables).

    This code follows the convolution/mixing structure of the supplied
    Gaussian_with_outliers.py, with the corrections described above.
    Linear interpolation is zero outside the specified redshift support.
    """
    if np.shape(nzs) != (Nbins, np.size(zs)):
        raise ValueError("nzs must have shape (Nbins, len(zs)).")
    values = evaluate_nz(nzs, zs, dz, std_b, c)
    sol = [interp1d(zs, row, bounds_error=False, fill_value=0.0)
           for row in values]
    return np.asarray(zs, dtype=float).copy(), sol


def parameter_names(binn, n_bins, vary_mixing=True):
    """Per-bin Fisher/prior order; indices in parameter names start at one."""
    names = [f"gdz{binn + 1}", f"gsigma{binn + 1}"]
    if vary_mixing:
        names += [f"gc{binn + 1}_{j + 1}" for j in range(n_bins) if j != binn]
    return names


def configuration(z, nzs, *, dz, std_b, c, vary_mixing=True):
    """Validated numeric state for Fisher; fiducial values are explicit.

    Keep the original templates separate from the convolved fiducial PDFs:
    applying a convolution to an already convolved PDF would broaden it twice.
    """
    templates = normalize_nz(z, nzs)
    if templates.ndim != 2:
        raise ValueError("nzs must have shape (n_bins, n_z).")
    n_bins = templates.shape[0]
    if not isinstance(vary_mixing, (bool, np.bool_)):
        raise ValueError("vary_mixing must be True or False.")
    dz = _per_bin(dz, n_bins, "dz")
    std_b = _per_bin(std_b, n_bins, "std_b")
    if np.any(std_b <= 2 * np.max(np.diff(z))):
        raise ValueError("Refine the redshift grid: each std_b must be greater "
                         "than 2*max(diff(z)), including room for derivative steps.")
    c = validate_mixing(c, n_bins)
    if vary_mixing and np.any(c.sum(axis=1) >= 1):
        raise ValueError("Free mixing fractions require each row sum < 1.")
    return dict(gpro_templates=templates, gpro_dz=dz, gpro_std_b=std_b,
                gpro_c=c, gpro_vary_mixing=bool(vary_mixing))


def validate_prior_cov(cov, n_bins, vary_mixing=True):
    """Validate per-bin covariance blocks in ``parameter_names`` order.

    These are priors on dz, std_b and (optionally) the off-diagonal c[i,j],
    NOT the mu, sigma and residual fraction of the original Gaussian model.
    No correlations between bins are represented by this block interface.
    """
    if cov is None:
        return None
    n_params = 2 + (n_bins - 1 if vary_mixing else 0)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (n_bins, n_params, n_params) or not np.all(np.isfinite(cov)):
        raise ValueError(f"photoz_prior_cov must have shape {(n_bins, n_params, n_params)}.")
    for block in cov:
        if not np.allclose(block, block.T, rtol=1e-10, atol=1e-14):
            raise ValueError("Each prior covariance must be symmetric.")
        if np.any(np.diag(block) <= 0):
            raise ValueError("Prior variances must be positive.")
        sd = np.sqrt(np.diag(block))
        try:
            np.linalg.cholesky(block / np.outer(sd, sd))
        except np.linalg.LinAlgError as error:
            raise ValueError("Prior covariances must be positive definite.") from error
    return cov.copy()
