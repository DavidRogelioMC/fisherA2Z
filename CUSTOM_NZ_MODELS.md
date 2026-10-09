# Adding a custom redshift-distribution model to FisherFlex

The recommended extension point for a new source-redshift-distribution
parameterization is `src/fisherA2Z/fisher_flex.py`. The older
`src/fisherA2Z/fisher.py` implementation is comparatively hard-coded.

## Fiducial distributions versus uncertainty models

`FisherFlex` already accepts an arbitrary tabulated fiducial distribution:

```python
flex = FisherFlex(
    nz_source=my_nz,                 # (n_tomo, n_z)
    nz_realizations=my_nz_samples,   # (n_real, n_tomo, n_z)
    z_grid=z,
    nz_model="my_model",
    # survey and analysis options...
)
```

Adding an `nz_model` means defining:

1. the nuisance parameters that deform each fiducial distribution;
2. the forward model `n(z | parameters)`;
3. how realizations are converted into parameter values; and
4. the prior covariance of those parameters.

## Recommended module structure

Create a module such as `src/fisherA2Z/nz_my_model.py` containing three
conceptual operations:

```python
def build_my_model(z, nz_source):
    """Validate, normalize, and prepare the fiducial model."""


def evaluate_my_nz(z, nz_fid, params, model_metadata):
    """Return n(z | params) for one tomographic bin."""


def fit_my_model(z, nz_reference, nz_realizations):
    """Return parameters with shape (n_real, n_tomo, n_parameters)."""
```

At the fiducial point, the model should satisfy

```python
evaluate_my_nz(z, nz_fid, fiducial_params, metadata) == nz_fid
```

up to numerical precision. Every evaluated distribution should be finite,
non-negative, defined on `z_grid`, and have a positive integral.

## Required changes in `fisher_flex.py`

### 1. Register the model

Add its name and parameter count to `NZ_MODELS` and `N_PZ_PER_BIN`:

```python
NZ_MODELS = (
    "shift_stretch",
    "gaussian_outlier",
    "no_uncertainty",
    "my_model",
)

N_PZ_PER_BIN = {
    "shift_stretch": 2,
    "gaussian_outlier": 3,
    "no_uncertainty": 0,
    "my_model": K,
}
```

`K` is the number of nuisance parameters per source bin.

### 2. Initialize the model

Extend `_setup_photoz()` with an explicit branch:

```python
if self.nz_model == "shift_stretch":
    self._setup_shift_stretch(decompose_kwargs)
elif self.nz_model == "gaussian_outlier":
    self._setup_gaussian_outlier(decompose_kwargs)
elif self.nz_model == "my_model":
    self._setup_my_model(decompose_kwargs)
else:
    raise ValueError(...)
```

Implement `_setup_my_model()`. It should populate the fiducial distribution
and all metadata needed by the forward model. Existing save/reporting code also
expects these attributes:

```python
self.nz_source_fid
self.pz_pivot
self.pz_mu
self.pz_sigma
self.pz_fout
self.nz_out
self.pz_degenerate
```

If some fields do not apply, use documented neutral arrays with the expected
shapes.

### 3. Implement the forward model

Add a branch to `_FlexModel.source_nz()`:

```python
if self.nz_model == "my_model":
    for i in range(self.n_source):
        params = [
            vals[f"nzparam1_{i + 1}"],
            vals[f"nzparam2_{i + 1}"],
        ]
        out[i] = evaluate_my_nz(
            self.z,
            self.nz_source_fid[i],
            params,
            # model metadata...
        )
    return out
```

Fisher derivatives are calculated numerically, so analytic derivatives are
normally unnecessary.

### 4. Fit the realizations or provide a prior directly

Add a branch to `_fit_realizations()`:

```python
if self.nz_model == "my_model":
    return fit_my_model(self.z, self.nz_source_fid, real)
```

The output must have shape `(n_realizations, n_source_bins, K)`. The existing
`photoz_prior_from_realizations()` function can then construct a `K x K`
covariance for each bin.

If the prior was obtained externally, realization fitting can be skipped by
passing:

```python
photoz_prior_cov=my_covariance
```

with shape `(n_source_bins, K, K)`.

### 5. Define parameter names

Extend `_pz_param_names()`:

```python
if self.nz_model == "my_model":
    return [
        f"nzparam1_{i + 1}",
        f"nzparam2_{i + 1}",
    ]
```

This order must exactly match both the final dimension of the fitted
realization parameters and the rows/columns of `photoz_prior_cov`.

### 6. Add parameters to the Fisher calculation

Extend `_setup_params()` with:

- one unique name per parameter and tomographic bin;
- plotting labels;
- fiducial values in `self.fid`;
- appropriate finite-difference steps in `self.param_step`; and
- derivative-method overrides in `self.param_deriv_method`, when necessary.

Use a forward difference for a parameter at a physical boundary if a central
difference would evaluate an invalid model.

### 7. Make the state multiprocessing-safe

Any additional templates, basis functions, knots, or metadata must be included
in both:

- `_FlexModel.STATE_KEYS`; and
- `FisherFlex._to_state()`.

Use serializable values such as NumPy arrays, strings, scalars, and dictionaries.

### 8. Update saved-file support

Keep the custom model in `N_PZ_PER_BIN`, because `.npz` loading uses that
mapping. Save and reload any additional arrays required by the model. Increment
`SCHEMA_VERSION` if the saved-file structure changes.

## Suggested general non-Gaussian model

A basis-deformation model preserves an arbitrary, possibly skewed or multimodal,
fiducial distribution:

```text
n_i(z; a_i) = n_i,fid(z) exp[sum_alpha a_i,alpha e_i,alpha(z)] / normalization
```

Here `e_i,alpha(z)` can be spline, histogram, or PCA modes derived from the
realizations. This construction has useful properties:

- the supplied, non-Gaussian `n(z)` remains the exact fiducial model;
- the exponential guarantees non-negativity;
- normalization is automatic;
- zero coefficients reproduce the fiducial distribution; and
- PCA coefficients measured from the realizations directly define the prior.

This is generally safer than independently varying histogram heights, which
can become negative during finite differences.

## Tests to add

Use `tests/test_nz_shift_stretch.py` as a template. At minimum, test:

1. exact fiducial reconstruction;
2. positivity, finiteness, and normalization;
3. recovery of known parameters from synthetic realizations;
4. prior covariance shape and parameter ordering;
5. numerical derivative convergence;
6. agreement between serial and parallel computation; and
7. `.npz` save/load round-tripping.

