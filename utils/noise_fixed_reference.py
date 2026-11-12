"""Fixed-rank covariance fits, with optional frozen uncorrected incoherent noise."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import plot_cache as cache
from .analytic_artifacts import sha
from .plot_fit_provenance import noise_fit_definition
from .plot_fits import json_safe, noise_model

VARIANTS = ('independent', 'fixedD_uncorrected')


def label(config, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown noise-fit variant: {variant}')
    return f'{variant}_nc{config.diagnostics["n_coherent_noise_model"]}'


def fit_directory(config, root, variant):
    stem = (f'nc{config.diagnostics["n_coherent_noise_model"]}' if variant == 'independent'
            else label(config, variant))
    return Path(root)/f'{stem}_iter{config.plotting["plot_noise_maxiter"]}'


def fit_key(variant):
    return 'noise_model' if variant == 'independent' else 'noise_model_fixedD_uncorrected'


def attach_fits(config, module, info, products, *, create=True):
    from .paths import resolve_paths
    from .inference_config import method_directory
    root = Path(resolve_paths(config)['target']['noise_model_fits'])/module
    variants = config.plotting['plot_noise_fit_variants']
    if 'fixedD_uncorrected' in variants and 'uncorrected' not in products:
        raise ValueError('fixedD_uncorrected requires uncorrected in methods')
    reference = None
    if 'uncorrected' in products:
        reference = product(config, module, 'uncorrected', products['uncorrected']['covariance'],
            root/method_directory(config, 'uncorrected'), info['nch'], 'independent', create=create)
    for method, row in products.items():
        for variant in variants:
            row[fit_key(variant)] = (reference if method == 'uncorrected' else
                product(config, module, method, row['covariance'], root/method_directory(config, method),
                        info['nch'], variant, reference=reference,
                        raw_covariance=products.get('uncorrected', {}).get('covariance'), create=create))
    return products


def definition(config, covariance, variant, reference=None):
    out = noise_fit_definition(config, covariance.token)
    if variant == 'fixedD_uncorrected':
        if reference is None:
            raise ValueError('A verified uncorrected reference fit is required')
        out.update(schema='hgcal_cm_fixed_incoherent_reference_v1', reference_fit=reference.token,
                   reference_covariance=reference.record['definition']['covariance'],
                   reference_observed_channels=reference.record['summary'].get('observed_channels'),
                   algorithm='positive spectral truncation of C-D for complete masks; L-BFGS-B for missing pairs',
                   fixed_reference_code_sha256=sha(Path(__file__)))
    return out


def require_fit(product, context):
    if not product.record['summary'].get('success', False):
        raise RuntimeError(f'Noise fit did not converge for {context}: '
                           f'{product.record["summary"].get("message")}. '
                           f'Inspect {product.path / "fit.json"}; saved failure is not an adopted fit.')
    return product


def paired_reference(corrected, raw, reference, nch):
    """Require identical per-pair populations and observed channels before sharing D."""
    a, b = corrected.arrays(), raw.arrays()
    if (a['covariance'].shape != b['covariance'].shape
            or not np.array_equal(a['counts'], b['counts'])):
        raise ValueError('Fixed-D reference and corrected covariance pair counts differ')
    observed = []
    for data in (a, b):
        c, k = data['covariance'][:nch, :nch], data['counts'][:nch, :nch]
        channels = np.isfinite(np.diag(c)) & (np.diag(k) >= 2)
        observed.append(channels)
    if not np.array_equal(*observed):
        raise ValueError('Fixed-D reference and corrected observed channels differ')
    ix = np.ix_(observed[0], observed[0])
    masks = [np.isfinite(data['covariance'][:nch, :nch][ix])
             & (data['counts'][:nch, :nch][ix] >= 2) for data in (a, b)]
    if not np.array_equal(*masks):
        raise ValueError('Fixed-D reference and corrected observed-pair masks differ')
    values = reference.arrays()['incoherent']
    if values.shape != (nch, nch):
        raise ValueError('Uncorrected reference channel dimensions differ')
    return a, np.diag(values)


def objective(parameters, covariance, mask, diagonal, rank):
    loading = parameters.reshape(len(diagonal), rank)
    residual = np.where(mask, np.diag(diagonal)+loading@loading.T-covariance, 0.)
    count = int(mask.sum())
    return float(np.sum(residual**2)/count), (4*residual@loading/count).ravel()


def fixed_diagonal_model(covariance, counts, diagonal, rank, maxiter, *, gtol, ftol):
    """Same full covariance MSE as noise_model; D is held exactly fixed.

    For a complete observed matrix the closest positive-semidefinite matrix of
    rank <= nc is its positive spectral truncation: no iterative optimizer is
    needed. Missing pairs retain the masked MSE and use its analytic gradient.
    """
    covariance, counts, diagonal = map(np.asarray, (covariance, counts, diagonal))
    n = len(covariance)
    if covariance.shape != (n, n) or counts.shape != (n, n) or diagonal.shape != (n,):
        raise ValueError('Fixed-D covariance/count/diagonal dimensions differ')
    if rank < 1 or any(not np.isfinite(v) or v <= 0 for v in (gtol, ftol)):
        raise ValueError('Fixed-D rank and optimizer tolerances must be positive')
    selected = np.flatnonzero(np.isfinite(np.diag(covariance)) & (np.diag(counts) >= 2))
    if len(selected) <= rank:
        raise ValueError('Too few observed channels for fixed-D coherent rank')
    ix = np.ix_(selected, selected)
    cov = .5*(covariance[ix]+covariance[ix].T)
    mask = np.isfinite(cov) & (counts[ix] >= 2)
    if not np.array_equal(mask, mask.T):
        raise ValueError('Fixed-D observed-pair mask must be symmetric')
    d = diagonal[selected].astype(float, copy=True)
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError('Invalid incoherent variances in uncorrected reference')
    target = np.where(mask, cov-np.diag(d), 0.)
    values, vectors = np.linalg.eigh(target)
    order = np.argsort(values)[::-1][:rank]
    loading = vectors[:, order]*np.sqrt(np.maximum(values[order], 0.))[None, :]
    options = {'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol}
    if mask.all():
        status = dict(success=True, status=0, message='Exact positive spectral truncation for complete covariance',
                      iterations=0, evaluations=1, solver='spectral')
    else:
        from scipy.optimize import minimize
        result = minimize(objective, loading.ravel(), args=(cov, mask, d, rank), jac=True,
                          method='L-BFGS-B', options=options)
        loading = result.x.reshape(len(selected), rank)
        status = dict(success=bool(result.success), status=int(result.status), message=str(result.message),
                      iterations=int(result.nit), evaluations=int(result.nfev), solver='L-BFGS-B')
    loading = loading[:, np.argsort(np.sum(loading*loading, axis=0))[::-1]]
    for j in range(rank):
        if loading[np.argmax(np.abs(loading[:, j])), j] < 0:
            loading[:, j] *= -1
    coherent, incoherent = loading@loading.T, np.diag(d)
    model = coherent+incoherent
    arrays = {key: np.full((n, n), np.nan) for key in ('model', 'coherent', 'incoherent', 'residual')}
    for key, value in [('model', model), ('coherent', coherent), ('incoherent', incoherent), ('residual', cov-model)]:
        arrays[key][ix] = value
    arrays['loadings'] = np.full((n, rank), np.nan)
    arrays['loadings'][selected] = loading
    mse, gradient = objective(loading.ravel(), cov, mask, d, rank)
    finite = np.isfinite(loading).all() and np.isfinite(mse) and np.isfinite(gradient).all()
    total = float(np.trace(cov))
    summary = dict(status, success=bool(status['success'] and finite), rank=rank, mse=mse,
        optimizer_options=options, observed_channels=selected.tolist(), trace_total=total,
        trace_coherent=float(np.trace(coherent)), coherent_fraction=float(np.trace(coherent)/total) if total > 0 else None,
        gradient_inf=float(np.max(np.abs(gradient))), incoherent_definition='fixed uncorrected fitted variances',
        objective='mean squared covariance error over valid entries, including diagonal')
    return arrays, json_safe(summary)


def product(config, module, method, covariance, root, nch, variant, *, reference=None,
            raw_covariance=None, create=True):
    """Reuse the original independent fit exactly; publish fixed-D separately."""
    if variant == 'fixedD_uncorrected' and method == 'uncorrected':
        return require_fit(reference, f'{module}/uncorrected reference')
    directory = fit_directory(config, root, variant)
    expected = definition(config, covariance, variant, reference)
    saved = cache.matching(directory, expected)
    if saved is not None:
        return require_fit(saved, f'{module}/{method}/{label(config, variant)}')
    if not create:
        raise FileNotFoundError(f'Missing current noise fit: {directory}. Run --steps plots first.')
    data = covariance.arrays()
    rank = config.diagnostics['n_coherent_noise_model']
    settings = config.plotting
    print(f'Noise fit {module}/{method}: {label(config, variant)}', flush=True)
    if variant == 'independent':
        arrays, summary = noise_model(data['covariance'][:nch, :nch], data['counts'][:nch, :nch], rank,
            settings['plot_noise_maxiter'], gtol=settings['plot_noise_gtol'], ftol=settings['plot_noise_ftol'])
    else:
        require_fit(reference, f'{module}/uncorrected reference')
        data, diagonal = paired_reference(covariance, raw_covariance, reference, nch)
        arrays, summary = fixed_diagonal_model(data['covariance'][:nch, :nch], data['counts'][:nch, :nch],
            diagonal, rank, settings['plot_noise_maxiter'], gtol=settings['plot_noise_gtol'], ftol=settings['plot_noise_ftol'])
        summary['reference_fit'] = reference.token
    def build(work):
        np.savez_compressed(work/'data.npz', **arrays)
        (work/'fit.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        return summary
    return require_fit(cache.publish(directory, expected, build), f'{module}/{method}/{label(config, variant)}')
