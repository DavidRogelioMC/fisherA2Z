"""Integer PDF degrees for quantile reconstructions that retain their masses.

PDF order 0 uses a linear CDF; order 1 uses a nonnegative linear PDF with
an extra midpoint per quantile interval; order 2 retains the monotone cubic
PCHIP CDF. Higher orders use an interpolating CDF B-spline of degree order+1. A spline
that decreases anywhere is rejected: clipping would change the quantiles.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator, PPoly, make_interp_spline

INTERPOLATION_VERSION = "quantile_pdf_order_v1"


def validate_interpolation_order(order, n_levels=None):
    """The number is the polynomial degree of n(z), not of the CDF."""
    if (isinstance(order, (bool, np.bool_))
            or not isinstance(order, (int, np.integer)) or order < 0):
        raise ValueError("interpolation_order must be an integer >= 0 (the degree of n(z)).")
    order = int(order)
    # PCHIP uses local Hermite interpolation and needs only two locations.
    if n_levels is not None and order != 2 and order + 1 >= n_levels:
        raise ValueError(
            f"Order {order} for n(z) needs at least {order + 2} quantile levels; "
            f"received {n_levels}."
        )
    return order


def _cdf_spline(locs, levels, order):
    """Interpolate the quantile constraints F(locs[j]) = levels[j]."""
    if order == 1:
        return _linear_pdf_cdf(locs, levels)
    if order == 2:
        return PchipInterpolator(locs, levels, extrapolate=False)
    degree = order + 1
    # Averaged interpolation knots work for both even and odd degrees.
    interior = [np.mean(locs[j:j+degree]) for j in range(1, len(locs)-degree)]
    knots = np.r_[np.repeat(locs[0], degree+1), interior,
                  np.repeat(locs[-1], degree+1)]
    return make_interp_spline(locs, levels, k=degree, t=knots)


def _linear_pdf_cdf(locs, levels):
    """A continuous linear PDF preserving the mass in each quantile interval.

    A straight segment between each pair of quantiles need not permit both
    nonnegative endpoint heights and the required mass. One extra midpoint
    per interval resolves that constraint. Harmonic-mean endpoint heights
    guarantee nonnegative midpoint heights; these are reconstruction nodes,
    not additional uncertain parameters.
    """
    mean_pdf = np.diff(levels) / np.diff(locs)
    heights = np.r_[mean_pdf[0],
                    2.0 / (1.0/mean_pdf[:-1] + 1.0/mean_pdf[1:]), mean_pdf[-1]]
    # Integrating two straight segments gives width*(left+2*middle+right)/4.
    middle = 2.0*mean_pdf - 0.5*(heights[:-1]+heights[1:])
    nodes = np.empty(2*len(locs)-1)
    values = np.empty_like(nodes)
    nodes[::2], nodes[1::2] = locs, 0.5*(locs[:-1]+locs[1:])
    values[::2], values[1::2] = heights, middle
    return make_interp_spline(nodes, values, k=1).antiderivative()


def _check_nonnegative_pdf(cdf, locs, levels, order):
    """Check minima on every polynomial piece, not just on plotting nodes."""
    polynomial = cdf if isinstance(cdf, PPoly) else PPoly.from_spline(cdf)
    pdf = polynomial.derivative()
    extrema = pdf.derivative().roots(extrapolate=False)
    candidates = np.r_[polynomial.x, extrema]
    candidates = candidates[np.isfinite(candidates)
                            & (candidates >= locs[0]) & (candidates <= locs[-1])]
    values = pdf(candidates)
    scale = max(float(np.max(np.diff(levels)/np.diff(locs))),
                float(np.max(np.abs(values))))
    if not np.all(np.isfinite(values)) or np.min(values) < -1e-12*scale:
        raise ValueError(
            f"interpolation_order={order} produces a negative n(z) for these quantiles. "
            "Use order 0, 1, or 2 (monotone PCHIP), or another valid order. "
            "The PDF was not clipped or replaced by a different interpolation."
        )


def quantile_cdf_values(edges, locs, levels, order):
    """CDF at cell edges; differencing it conserves each quantile's probability."""
    order = validate_interpolation_order(order, len(levels))
    if order == 0:
        return np.interp(edges, locs, levels, left=0., right=1.)
    cdf = _cdf_spline(locs, levels, order)
    _check_nonnegative_pdf(cdf, locs, levels, order)
    values = np.asarray(cdf(edges), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("The interpolated quantile CDF is not finite.")
    if np.any(np.diff(values) < -1e-12):
        raise ValueError("The interpolated quantile CDF decreases; reduce the order.")
    # Only roundoff remains after the continuous positivity check above.
    values = np.maximum.accumulate(np.clip(values, 0., 1.))
    values[0], values[-1] = 0., 1.
    return values
