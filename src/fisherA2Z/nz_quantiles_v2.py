"""Configurable probability levels, with free interior redshift locations.

The support endpoints are fixed to z[0] and z[-1].  Interior locations are
the inverse CDF of the *original*, normalized, piecewise-linear input PDF.
INTERPOLATION_ORDER selects the PDF degree: 0 constant, 1 linear, 2 quadratic, etc.
Order 1 preserves interval masses using an extra interpolation midpoint.
Order 2 uses a monotone PCHIP CDF; invalid higher-order PDFs are rejected.

For CCL we return cell-averaged densities on the requested z nodes.  This
conservative projection keeps moving-boundary probability in numerical
derivatives.  It differs from sampling qp's discontinuous PDF at the nodes;
joint redshift-grid and finite-difference convergence must still be checked.
No qp objects, interpolators or CCL objects are stored in the model state.
"""

from dataclasses import dataclass

import numpy as np
from ._quantile_interpolation import (
    INTERPOLATION_VERSION, quantile_cdf_values, validate_interpolation_order,
)

# Order of n(z): 0 constant, 1 linear, 2 quadratic (PCHIP), 3 cubic, etc.
INTERPOLATION_ORDER = 0

N_QUANTILES = 10
N_PARAMS = N_QUANTILES - 2


# Checar bias


def _grid(z):
    z = np.asarray(z, dtype=float)
    if (z.ndim != 1 or z.size < 3 or not np.all(np.isfinite(z))
            or np.any(np.diff(z) <= 0) or z[0] < 0):
        raise ValueError("z must be a finite, nonnegative, increasing 1D grid.")
    return z


def _integral(y, z):
    return np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(z), axis=-1)


def normalize_nz(z, nz):
    """Normalize nonnegative PDFs; reject negative data instead of clipping."""
    z = _grid(z)
    nz = np.asarray(nz, dtype=float)
    if nz.ndim < 1 or nz.shape[-1] != z.size:
        raise ValueError("The final n(z) dimension must match z.")
    if not np.all(np.isfinite(nz)) or np.any(nz < 0):
        raise ValueError(
            "n(z) must be finite and nonnegative. Negative noisy estimates "
            "require an explicit statistical treatment before fitting quantiles."
        )
    area = _integral(nz, z)
    if np.any(area <= 0) or not np.all(np.isfinite(area)):
        raise ValueError("Every n(z) must have a finite positive integral.")
    return nz / area[..., None]


def _levels(levels):
    levels = np.asarray(levels, dtype=float)
    if (levels.ndim != 1 or levels.size < 3 or not np.all(np.isfinite(levels))
            or levels[0] != 0 or levels[-1] != 1
            or np.any(np.diff(levels) <= 0)):
        raise ValueError("levels must contain at least three increasing values from 0 to 1.")
    return levels


def quantile_locations(z, nz, levels):
    """Invert the integrated piecewise-linear input PDF without CDF regridding.

    Leading dimensions are preserved. At a CDF plateau the leftmost location
    is selected (generalized inverse); endpoints are fixed support boundaries.
    """
    z = _grid(z)
    levels = _levels(levels)
    rows = normalize_nz(z, nz)
    out = np.empty(rows.shape[:-1] + (levels.size,))
    for row, loc in zip(rows.reshape(-1, z.size), out.reshape(-1, levels.size)):
        masses = 0.5 * (row[1:] + row[:-1]) * np.diff(z)
        cdf = np.concatenate(([0.0], np.cumsum(masses)))
        cdf[-1] = 1.0
        loc[0], loc[-1] = z[0], z[-1]
        p = levels[1:-1]
        idx = np.searchsorted(cdf, p, side="left") - 1
        idx = np.clip(idx, 0, z.size - 2)
        area = np.maximum(p - cdf[idx], 0.0)
        slope = (row[idx + 1] - row[idx]) / (z[idx + 1] - z[idx])
        root = np.sqrt(np.maximum(row[idx] ** 2 + 2.0 * slope * area, 0.0))
        denominator = row[idx] + root
        distance = np.divide(2.0 * area, denominator,
                             out=np.zeros_like(area), where=denominator > 0)
        loc[1:-1] = z[idx] + np.minimum(distance, z[idx + 1] - z[idx])
    if np.any(np.diff(out, axis=-1) <= 0):
        raise ValueError("Input PDFs produce coincident quantile locations.")
    return out


def evaluate_quantile_nz(z, levels, locs_fid, params, interpolation_order=None):
    """Evaluate one bin at the interior redshift displacements delta q.

    Cell edges are z[0], adjacent-node midpoints, z[-1]. Their widths equal
    the trapezoidal node weights, hence trapz(result, z) equals one. CCL
    subsequently interpolates these samples; refine z to assess that error.
    """
    z = _grid(z)
    levels = _levels(levels)
    locs_fid = np.asarray(locs_fid, dtype=float)
    params = np.asarray(params, dtype=float)
    if locs_fid.shape != levels.shape or params.shape != (levels.size - 2,):
        raise ValueError("Expected one location per level and one displacement per interior level.")
    if not np.all(np.isfinite(locs_fid)) or not np.all(np.isfinite(params)):
        raise ValueError("Quantile locations and displacements must be finite.")
    if locs_fid[0] != z[0] or locs_fid[-1] != z[-1]:
        raise ValueError("The outer quantile locations must equal the grid support.")
    locs = locs_fid.copy()
    locs[1:-1] += params
    if np.any(np.diff(locs) <= 0):
        raise ValueError("Quantile locations crossed or coincide; reduce the step.")
    order = INTERPOLATION_ORDER if interpolation_order is None else interpolation_order
    edges = np.r_[z[0], 0.5*(z[:-1]+z[1:]), z[-1]]
    cdf = quantile_cdf_values(edges, locs, levels, order)
    return np.diff(cdf) / np.diff(edges)


@dataclass
class QuantileModel:
    nz_fid: np.ndarray
    levels: np.ndarray
    locs: np.ndarray
    pivot: np.ndarray
    sigma: np.ndarray
    interpolation_order: int


def build_quantile_model(z, nz_source, n_quantiles=N_QUANTILES, interpolation_order=None):
    """Build the reconstructed fiducial, keeping original quantile locations."""
    z = _grid(z)
    nz_source = np.atleast_2d(np.asarray(nz_source, dtype=float))
    if nz_source.ndim != 2 or nz_source.shape[0] == 0:
        raise ValueError("nz_source must have shape (n_source, n_z).")
    if (isinstance(n_quantiles, (bool, np.bool_))
            or not isinstance(n_quantiles, (int, np.integer)) or n_quantiles < 3):
        raise ValueError("n_quantiles must be an integer >= 3 (including endpoints).")
    levels = np.linspace(0.0, 1.0, n_quantiles)
    order = validate_interpolation_order(
        INTERPOLATION_ORDER if interpolation_order is None else interpolation_order, len(levels))
    locs = quantile_locations(z, nz_source, levels)
    nz_fid = np.array([evaluate_quantile_nz(z, levels, q, np.zeros(levels.size - 2), order)
                       for q in locs])
    pivot = _integral(nz_fid * z, z)
    variance = _integral(nz_fid * (z - pivot[:, None]) ** 2, z)
    return QuantileModel(nz_fid, levels, locs, pivot, np.sqrt(variance), order)


def fit_quantiles(z, nz_reference, real, levels, locs_fid):
    """Measure delta q on original realizations, returning (R, B, n_quantiles - 2).

    nz_reference is used only for checking the bin count. Fiducial locations
    must come from build_quantile_model, never by re-quantiling nz_fid.
    """
    z = _grid(z)
    levels = _levels(levels)
    reference = np.atleast_2d(np.asarray(nz_reference, dtype=float))
    real = np.asarray(real, dtype=float)
    locs_fid = np.asarray(locs_fid, dtype=float)
    if (reference.ndim != 2 or reference.shape[-1] != z.size
            or real.ndim != 3 or real.shape[1:] != reference.shape
            or locs_fid.shape != (reference.shape[0], levels.size)):
        raise ValueError("Expected realizations (R, B, n_z) and locations (B, n_quantiles).")
    return quantile_locations(z, real, levels)[..., 1:-1] - locs_fid[None, :, 1:-1]


def validate_prior_cov(cov, n_source, n_params=N_PARAMS):
    """Reject invalid or rank-deficient priors; never add artificial variance."""
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (n_source, n_params, n_params):
        raise ValueError(f"Quantile prior must have shape ({n_source}, {n_params}, {n_params}).")
    if not np.all(np.isfinite(cov)):
        raise ValueError("Quantile prior contains non-finite values.")
    for i, block in enumerate(cov):
        if not np.allclose(block, block.T, rtol=1e-10, atol=0):
            raise ValueError(f"Quantile prior in bin {i + 1} is not symmetric.")
        sd = np.sqrt(np.maximum(np.diag(block), 0.0))
        if np.any(sd <= 0):
            raise ValueError(f"Quantile prior in bin {i + 1} has zero/negative variance.")
        correlation = block / np.outer(sd, sd)
        eigenvalues = np.linalg.eigvalsh(correlation)
        tolerance = n_params * np.finfo(float).eps * eigenvalues[-1]
        if eigenvalues[0] <= tolerance:
            raise ValueError(
                f"Quantile prior in bin {i + 1} is singular or numerically rank "
                "deficient. Use sufficient independent realizations or reduce "
                "the model dimension; no arbitrary prior floor is applied."
            )
    return cov


def recommended_steps(locs_fid, default_step):
    """Local central-difference steps; convergence checks are still required."""
    q = np.asarray(locs_fid, dtype=float)
    if (q.ndim < 1 or q.shape[-1] < 3
            or not np.all(np.isfinite(q)) or np.any(np.diff(q, axis=-1) <= 0)):
        raise ValueError("Expected ordered fiducial locations (..., n_quantiles).")
    if not np.isfinite(default_step) or default_step <= 0:
        raise ValueError("default_step must be finite and positive.")
    gaps = np.diff(q, axis=-1)
    return np.minimum(float(default_step), 0.1 * np.minimum(gaps[..., :-1], gaps[..., 1:]))
