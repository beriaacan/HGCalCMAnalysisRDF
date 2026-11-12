"""Conditional per-channel least squares and compact, physical-unit predictors."""
from __future__ import annotations

import json
from pathlib import Path

from .training import initialize_training_root

SCHEMA = 'hgcal_cm_analytic_allinputs_v3'
_READY = False


def initialize_analytic_root():
    global _READY
    ROOT = initialize_training_root()
    if not _READY:
        header = Path(__file__).with_name('analytic_helpers.hxx')
        if not ROOT.gInterpreter.Declare(
                '#ifndef EIGEN_DONT_PARALLELIZE\n#define EIGEN_DONT_PARALLELIZE\n#endif\n'
                f'#include {json.dumps(str(header))}'):
            raise RuntimeError('Could not compile analytic RDF helpers; use the current hgc environment')
        _READY = True
    return ROOT


def candidate_columns(spec: dict, module: str) -> list[int]:
    """Exclude module/target constants and known padding before allocating moments."""
    meta = spec['modules'][module]
    result = []
    for index, name in enumerate(spec['feature_names'][:spec['global_size']]):
        if name in {'module_id', 'module_nch'}:
            continue
        if name.startswith('cm_erx') and int(name.removeprefix('cm_erx')) >= meta['nerx']:
            continue
        if name.startswith('adc_allch_') and int(name.removeprefix('adc_allch_')) >= meta['nch']:
            continue
        result.append(index)
    return result


def solve(fitter, spec: dict, module: str, columns: list[int], rcond: float) -> dict:
    """Solve conditional correlation matrices; return coefficients in raw ADC units.

    Applying the saved affine training preprocessing and then the per-fit
    centering/unit-variance scaling gives this same system. Working directly in
    physical units avoids an unnecessary float32 normalization round trip.
    """
    import numpy as np
    nch, width, p = spec['modules'][module]['nch'], len(spec['feature_names']), len(columns)
    weights = np.zeros((nch, width), dtype=np.float64)
    intercept = np.full(nch, np.nan)
    counts = np.array([fitter.count(c) for c in range(nch)], dtype=np.int64)
    fitted = np.zeros(nch, dtype=bool)
    active = np.zeros((nch, width), dtype=bool)
    ranks = np.zeros(nch, dtype=np.int64)
    conditions = np.full(nch, np.nan)
    details = []
    columns_array = np.asarray(columns, dtype=np.int64)
    for channel, count in enumerate(counts):
        row = {'channel': channel, 'training_targets': int(count)}
        if count < 2:
            row.update(status='unavailable', reason='fewer than two retained training targets',
                       active_predictors=0, rank=0, retained_condition=None,
                       underdetermined=False)
            details.append(row)
            continue
        mean = np.asarray(fitter.mean(channel), dtype=np.float64)
        my = float(fitter.target_mean(channel))
        covariance = np.asarray(fitter.scatter(channel), dtype=np.float64).reshape(p, p)/count
        cross = np.asarray(fitter.cross(channel), dtype=np.float64)/count
        selected = np.asarray(fitter.varying(channel), dtype=int)
        own = spec['adc_columns'][channel]
        selected = selected[columns_array[selected] != own]
        if not (np.isfinite(mean).all() and np.isfinite(covariance).all()
                and np.isfinite(cross).all() and np.isfinite(my)):
            raise ValueError(f'Nonfinite conditional statistics: {module}/channel {channel}')
        if len(selected):
            matrix = covariance[np.ix_(selected, selected)]
            matrix = (matrix + matrix.T)*0.5
            variances = np.diag(matrix)
            if np.any(variances <= 0):
                raise ValueError(f'Nonpositive variance for varying predictors: {module}/channel {channel}')
            scales = np.sqrt(variances)
            correlation = matrix/scales[:, None]/scales[None, :]
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            cutoff = rcond*eigenvalues[-1]
            tolerance = 128*np.finfo(float).eps*max(1, len(selected))*max(1., eigenvalues[-1])
            if eigenvalues[0] < -tolerance:
                raise ValueError(f'Conditional covariance is not positive semidefinite: {module}/channel {channel}')
            keep = eigenvalues > max(cutoff, 0.)
            ranks[channel] = int(keep.sum())
            if keep.any():
                basis = eigenvectors[:, keep]
                beta = (basis @ ((basis.T @ (cross[selected]/scales))/eigenvalues[keep]))/scales
                weights[channel, columns_array[selected]] = beta
                conditions[channel] = eigenvalues[-1]/eigenvalues[keep][0]
            active[channel, columns_array[selected]] = True
        intercept[channel] = my - weights[channel, columns_array] @ mean
        fitted[channel] = True
        if own >= 0 and weights[channel, own] != 0:
            raise ValueError('Own-channel ADC leaked into regression')
        row.update(status='fitted' if len(selected) else 'intercept_only',
                   active_predictors=len(selected), rank=int(ranks[channel]),
                   retained_condition=float(conditions[channel]) if np.isfinite(conditions[channel]) else None,
                   underdetermined=bool(count < len(selected)+1))
        details.append(row)
    if not np.isfinite(weights).all() or not np.isfinite(intercept[fitted]).all():
        raise ValueError(f'Nonfinite regression coefficients: {module}')
    return {'coefficients': weights, 'intercepts': intercept, 'fitted': fitted,
            'training_counts': counts, 'active_predictors': active, 'rank': ranks,
            'retained_condition': conditions, 'diagnostics': details}


def make_evaluator(model: dict, columns: list[int], chunk: int):
    import numpy as np
    ROOT = initialize_analytic_root()
    return ROOT.hgcal_cm_analytic.Evaluator(
        ROOT.VecOps.RVec['int'](columns),
        ROOT.VecOps.RVec['double'](np.ascontiguousarray(model['coefficients'][:, columns]).ravel()),
        ROOT.VecOps.RVec['double'](model['intercepts']),
        ROOT.VecOps.RVec['int'](model['fitted'].astype(np.int32)), chunk)


def validation_payload(evaluator) -> list[dict]:
    import math
    result = []
    for channel, moment in enumerate(evaluator.moments):
        n = int(moment.n)
        row = {'channel': channel, 'eligible_targets': int(moment.eligible), 'evaluated_targets': n}
        for label, mean, m2 in (('uncorrected', moment.target_mean, moment.target_m2),
                                ('residual', moment.residual_mean, moment.residual_m2)):
            variance = max(0., float(m2)/n) if n else None
            row[label] = {'mean': float(mean) if n else None,
                          'std': math.sqrt(variance) if n else None,
                          'rmse': math.sqrt(variance+float(mean)**2) if n else None}
        result.append(row)
    return result


def load_module(directory: Path, module: str) -> tuple[dict, dict]:
    """Load a completed module, checking file hashes; callers must respect fitted."""
    from .analytic_artifacts import load_module as read_module
    return read_module(directory, module)
