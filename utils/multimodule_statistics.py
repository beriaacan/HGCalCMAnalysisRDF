"""Checked covariance/fit prerequisites and run-level noise summaries."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .inference_config import method_directory
from .paths import resolve_paths
from . import plot_cache as cache
from .noise_fixed_reference import attach_fits, fit_key


class MissingCovarianceError(FileNotFoundError):
    """An absent/current-definition cache, distinct from damaged saved products."""


def noise_products(config, modules):
    """Reuse numerical products; calculate only missing covariances and fits."""
    import plots
    from dataclasses import replace
    covariances = {}
    initialized = False
    # Select the existing collector's covariance path, without spectra or fits.
    covariance_config = replace(config, plotting=dict(config.plotting, plot_groups=['covariance']))
    for module, info in modules.items():
        print(f'Checking multimodule covariance products: {module}', flush=True)
        try:
            current = cached_covariances(config, {module: info})[module]
        except MissingCovarianceError:
            if not initialized:
                from .plot_statistics import initialize
                ROOT = initialize()
                if ROOT.IsImplicitMTEnabled():
                    ROOT.DisableImplicitMT()
                ROOT.EnableImplicitMT(config.resources['cpus'])
                initialized = True
            print(f'Completing covariance products for multimodule summaries: {module}', flush=True)
            parts = plots.contributions(covariance_config, module, info)
            combined = {key: plots.combine(covariance_config, module, key, rows)
                        for key, rows in parts.items()}
            plots.diagnostic_products(covariance_config, module, info, combined)
            # Recheck the same complete numerical provenance as cached products.
            current = cached_covariances(config, {module: info})[module]
        covariances[module] = current
    return cached_noise_products(config, modules, covariances, create=True)


def cached_covariances(config, modules):
    """Check every original source and combined covariance; never read events."""
    import plots
    root = Path(resolve_paths(config)['target']['covariances']).parent
    products = {}
    def required(directory, definition, *, immutable=False):
        result = cache.matching(directory, definition, immutable=immutable)
        if result is None:
            raise MissingCovarianceError(f'Missing/current saved covariance required: {directory}. '
                                         'Run --steps multimodule_plot or --steps plots to calculate it.')
        return result
    for module, info in modules.items():
        combined = {}
        for kind, method in [('cm', None)]+[('covariance', m) for m in config.methods]:
            folder = root/'statistics'/module/(method_directory(config, method) if method else Path('common'))/kind
            parts = []
            for source in info['sources']:
                directory = folder/f'Run{source["run"]}'/Path(source['input_file']).stem
                parts.append(required(directory, plots.source_definition(config,module,source,kind,method),immutable=True))
            combined[(kind, method)] = required(folder/'combined',
                {'kind':kind,'method':method,'contributions':[p.token for p in parts]})
        products[module] = {}
        for method in config.methods:
            adc, cm = combined[('covariance',method)], combined[('cm',None)]
            cov = required(root/'covariances'/module/method_directory(config,method),
                {'adc':adc.token,'cm':cm.token,'formula':'centered pairwise population covariance, N denominator'})
            data = cov.arrays()
            shape=(info['nch']+info['nerx'],)*2
            if data['covariance'].shape != shape or data['counts'].shape != shape:
                raise ValueError(f'Covariance dimensions differ for {module}/{method}')
            products[module][method] = cov
    return products


def cached_noise_products(config, modules, covariances=None, *, create=False):
    """Load checked covariance/fit variants, optionally creating missing fits."""
    covariances = cached_covariances(config,modules) if covariances is None else covariances
    variants = {variant:{} for variant in config.plotting['plot_noise_fit_variants']}
    for module, info in modules.items():
        rows={m:{'covariance':cov} for m,cov in covariances[module].items()}
        print(f'Checking/reusing noise fits for multimodule summaries: {module}', flush=True)
        attach_fits(config,module,info,rows,create=create)
        for variant, products in variants.items():
            products[module] = {}
            for method, row in rows.items():
                fit=row[fit_key(variant)];cov=row['covariance'];arrays=fit.arrays()
                nch=info['nch']
                if any(arrays[k].shape != (nch,nch) for k in ('coherent','incoherent','model')):
                    raise ValueError(f'Noise-fit dimensions differ for {module}/{method}/{variant}')
                products[module][method]={'covariance':cov,'fit':fit,
                    'data':components(np.diag(cov.arrays()['covariance'])[:nch],
                                      np.diag(arrays['coherent']),np.diag(arrays['incoherent']))}
    return variants


def ratio(corrected, raw):
    corrected, raw = np.asarray(corrected), np.asarray(raw)
    if corrected.shape != raw.shape:
        raise ValueError('Noise ratio channel dimensions differ')
    return np.divide(corrected, raw, out=np.full(raw.shape, np.nan),
                     where=np.isfinite(corrected) & np.isfinite(raw) & (raw > 0))


def components(measured_var, coherent_var, incoherent_var):
    """Fit RMS components and measured RMS / fitted RMS - 1."""
    values = {'total': coherent_var+incoherent_var, 'coherent': coherent_var, 'incoherent': incoherent_var}
    out = {key: np.sqrt(np.maximum(value, 0.)) for key, value in values.items()}
    closure = np.full(len(measured_var), np.nan)
    valid = np.isfinite(measured_var) & np.isfinite(values['total']) & (measured_var >= 0) & (values['total'] > 0)
    closure[valid] = np.sqrt(measured_var[valid]/values['total'][valid])-1.
    return dict(out, closure=closure)


def performance(modules, products, method, minimum):
    """Reference medians of per-channel RMS ratios, separately for each axis."""
    all_x, all_y, points = [], [], []
    connected_count = unconnected_count = threshold_count = 0
    for name, info in modules.items():
        raw, corrected = (products[name][m]['data'] for m in ('uncorrected', method))
        connected = np.asarray(info['area_fraction']) > 0
        x = 1.-ratio(corrected['coherent'], raw['coherent'])
        x[~(connected & (raw['coherent'] >= minimum))] = np.nan
        y = ratio(corrected['incoherent'], raw['incoherent'])
        y[~connected] = np.nan

        def median(values):
            finite = values[np.isfinite(values)]
            return float(np.median(finite)) if len(finite) else np.nan

        all_x.append(x[np.isfinite(x)]); all_y.append(y[np.isfinite(y)])
        erx = np.asarray([(median(x[i:i+37]), median(y[i:i+37])) for i in range(0, info['nch'], 37)])
        points.append({'module': name, 'family': re.sub(r'\d+$', '', name),
                       'point': [median(x), median(y)], 'erx_points': erx,
                       'coherent_channels': int(np.isfinite(x).sum()), 'incoherent_channels': int(np.isfinite(y).sum())})
        connected_count += int(connected.sum()); unconnected_count += int((~connected).sum())
        threshold_count += int((connected & np.isfinite(raw['coherent']) & (raw['coherent'] < minimum)).sum())
    x, y = np.concatenate(all_x), np.concatenate(all_y)
    return {'modules': points, 'channel_coherent_rejection': x, 'channel_incoherent_retention': y,
            'overall_point': [median(x), median(y)], 'n_connected': connected_count,
            'n_unconnected': unconnected_count, 'n_coherent_threshold_excluded': threshold_count,
            'min_coherent_sigma': minimum}


def scaled_median(names, modules, products, method, component):
    """Median across modules of (component - module median) / median total RMS."""
    nch = modules[names[0]]['nch']
    rows = []
    for name in names:
        if modules[name]['nch'] != nch:
            raise ValueError('Canonical aggregation mixes channel dimensions')
        data = products[name][method]['data']
        connected = np.asarray(modules[name]['area_fraction']) > 0
        valid, total = connected & np.isfinite(data[component]), connected & np.isfinite(data['total'])
        if not valid.any() or not total.any():
            continue
        scale = np.median(data['total'][total])
        if scale <= 0:
            continue
        row = np.full(nch, np.nan)
        row[valid] = (data[component][valid]-np.median(data[component][valid]))/scale
        rows.append(row)
    values, counts = np.full(nch, np.nan), np.zeros(nch, dtype=np.int64)
    if rows:
        stacked = np.stack(rows)
        for channel in range(nch):
            finite = stacked[:, channel][np.isfinite(stacked[:, channel])]
            counts[channel] = len(finite)
            if len(finite): values[channel] = np.median(finite)
    return values, counts, len(rows)
