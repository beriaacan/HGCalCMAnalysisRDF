"""Sequential coherent-rank selection using the unchanged production fitter."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import plot_cache as cache
from .analytic_artifacts import sha
from .plot_fit_provenance import noise_fit_definition
from .plot_fits import json_safe, noise_model

SCHEMA = 'hgcal_cm_sequential_noise_rank_v1'


def settings(config):
    return {key: config.plotting['plot_noise_'+key] for key in (
        'max_rank', 'min_improvement', 'localization_min', 'variance_fraction_min',
        'objective_atol', 'objective_rtol')}


def selected_name(config):
    return f'selected_max{config.plotting["plot_noise_max_rank"]}_iter{config.plotting["plot_noise_maxiter"]}'


def selection_definition(config, covariance_token, connected):
    return {'schema': SCHEMA, 'covariance': covariance_token, 'selection': settings(config),
            'connected_channels': np.flatnonzero(connected).tolist(), 'channel_count': len(connected),
            'maxiter': config.plotting['plot_noise_maxiter'],
            'gtol': config.plotting['plot_noise_gtol'], 'ftol': config.plotting['plot_noise_ftol'],
            'fit_code_sha256': sha(Path(__file__).with_name('plot_fits.py')),
            'selection_code_sha256': sha(Path(__file__)),
            'initialization': 'original spectral initialization independently at every rank',
            'metric': 'fractional reduction of connected-channel off-diagonal covariance RMSE',
            'stopping': 'first localized or insufficient-improvement candidate; numerical failures unresolved'}


def offdiag_mask(covariance, counts, connected):
    covariance, counts = np.asarray(covariance), np.asarray(counts)
    connected = np.asarray(connected, dtype=bool)
    if covariance.shape != counts.shape or covariance.shape != (len(connected), len(connected)):
        raise ValueError('Noise-selection covariance/count/geometry dimensions differ')
    observed = connected & np.isfinite(np.diag(covariance)) & (np.diag(covariance) > 0) & (np.diag(counts) >= 2)
    mask = np.outer(observed, observed) & np.isfinite(covariance) & (counts >= 2)
    np.fill_diagonal(mask, False)
    if not mask.any():
        raise ValueError('No valid connected-channel off-diagonal covariance pairs for rank selection')
    return mask


def unfitted_offdiag_rmse(covariance, counts, connected):
    mask = offdiag_mask(covariance, counts, connected)
    return float(np.sqrt(np.mean(np.asarray(covariance)[mask]**2)))


def candidate_metrics(arrays, mask, connected, options):
    residual = arrays['residual'][mask]
    if not np.isfinite(residual).all():
        return {'offdiag_rmse': None, 'localized_channels': [], 'localized_channel_count': 0}
    # Identical rotation-invariant formula and inclusive thresholds to the
    # validated scan's channel_values/spike_channels; keep its archived sources intact.
    k = arrays['coherent']
    power = np.nansum(k*k, axis=1)
    localization = np.divide(np.diag(k)**2, power, out=np.full(len(k), np.nan), where=power > 0)
    total = np.diag(arrays['model'])
    fraction = np.divide(np.diag(k), total, out=np.zeros(len(k)), where=total > 0)
    flagged = np.flatnonzero(np.asarray(connected, dtype=bool)
        & (localization >= options['localization_min']) & (fraction >= options['variance_fraction_min']))
    return {'offdiag_rmse': float(np.sqrt(np.mean(residual**2))),
            'localized_channels': flagged.tolist(), 'localized_channel_count': len(flagged)}


def select_rank(covariance, counts, connected, options, candidate):
    """Evaluate nc=0 upwards; candidate(rank) returns arrays, raw summary, audit token."""
    mask = offdiag_mask(covariance, counts, connected)
    attempts, selected, previous = [], None, None
    status, reason, stopped = 'selected', 'maximum_rank', None
    for rank in range(options['max_rank']+1):
        arrays, summary, token = candidate(rank)
        metrics = candidate_metrics(arrays, mask, connected, options)
        improvement = ((previous['offdiag_rmse']-metrics['offdiag_rmse'])/previous['offdiag_rmse']
                       if previous is not None and metrics['offdiag_rmse'] is not None else None)
        row = {'rank': rank, **metrics, 'fractional_improvement': improvement,
               'converged': bool(summary.get('success')), 'mse': summary.get('mse'),
               'message': summary.get('message'), 'accepted': False, 'rejection_reasons': [], 'fit': token}
        attempts.append(row)
        finite_objective = row['mse'] is not None and np.isfinite(row['mse'])
        if not row['converged'] or metrics['offdiag_rmse'] is None or not finite_objective:
            status, reason, stopped = 'unresolved', 'optimizer_failed_or_invalid', rank
            row['rejection_reasons'] = [reason]
            break
        if previous is not None and row['mse'] > previous['mse']*(1+options['objective_rtol'])+options['objective_atol']:
            status, reason, stopped = 'unresolved', 'full_objective_increased', rank
            row['rejection_reasons'] = [reason]
            break
        if rank:
            if row['localized_channel_count']:
                row['rejection_reasons'].append('localized_channels')
            if improvement < options['min_improvement']:
                row['rejection_reasons'].append('insufficient_improvement')
            if row['rejection_reasons']:
                reason, stopped = '+'.join(row['rejection_reasons']), rank
                break
        row['accepted'] = True
        selected, previous = (arrays, summary, row), row
        if row['offdiag_rmse'] == 0:
            reason, stopped = 'zero_offdiag_residual', rank
            break
    if selected is None:
        raise ValueError('Noise rank-zero fit failed; no valid diagonal baseline: '+str(attempts[-1]))
    arrays, summary, chosen = selected
    selection = {'status': status, 'selected_rank': chosen['rank'], 'stop_reason': reason,
                 'stopped_at_rank': stopped, 'criteria': options,
                 'valid_connected_offdiag_pairs': int(mask.sum()), 'attempts': attempts}
    summary = dict(summary, success=status == 'selected', fit_success=bool(summary['success']),
                   rank=chosen['rank'], offdiag_rmse=chosen['offdiag_rmse'],
                   localized_channels=chosen['localized_channels'], selection=selection)
    return arrays, json_safe(summary)


def require_selected(product, label):
    summary = product.record['summary']
    if not summary['success'] or summary['selection']['status'] != 'selected':
        decision = summary['selection']
        raise RuntimeError(f'Noise rank selection unresolved for {label}: nc={decision["stopped_at_rank"]}, '
            f'{decision["stop_reason"]}. Last accepted nc={summary["rank"]} is diagnostic only. '
            f'Inspect {product.path / "fit.json"}; review the failed candidate or increase plot_noise_maxiter '
            'before interpreting the rank. Completed candidate fits are saved.')
    return product


def selected_fit(config, module, method, covariance, connected, root):
    """Reuse candidates independently of the selection thresholds; publish a checked selection."""
    options = settings(config)
    connected = np.asarray(connected, dtype=bool)
    root = Path(root)
    directory = root/selected_name(config)
    definition = selection_definition(config, covariance.token, connected)
    previous = cache.matching(directory, definition)
    label = f'{module}/{method}'
    if previous is not None:
        require_selected(previous, label)
        print(f'Reusing noise rank selection {label}: nc={previous.record["summary"]["rank"]}', flush=True)
        return previous
    data = covariance.arrays(); n = len(connected)
    cov, counts = data['covariance'][:n, :n], data['counts'][:n, :n]

    def publish_arrays(work, arrays, summary):
        np.savez_compressed(work/'data.npz', **arrays)
        (work/'fit.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        return summary

    def candidate(rank):
        print(f'Noise rank selection {label}: trying nc={rank}', flush=True)
        candidate_directory = root/f'nc{rank}_iter{config.plotting["plot_noise_maxiter"]}'
        candidate_definition = noise_fit_definition(config, covariance.token, rank=rank)
        product = cache.matching(candidate_directory, candidate_definition)
        reused = product is not None
        if product is None:
            # The original fitter also handles rank zero: no loading columns,
            # exact diagonal initialization and an already-stationary objective.
            arrays, summary = noise_model(cov, counts, rank, config.plotting['plot_noise_maxiter'],
                gtol=config.plotting['plot_noise_gtol'], ftol=config.plotting['plot_noise_ftol'])
            product = cache.publish(candidate_directory, candidate_definition,
                lambda work: publish_arrays(work, arrays, summary))
        arrays, summary = product.arrays(), product.record['summary']
        for key in ('model', 'coherent', 'incoherent', 'residual'):
            if arrays[key].shape != (n, n):
                raise ValueError(f'Invalid candidate dimensions: {candidate_directory}/{key}')
        if arrays['loadings'].shape != (n, rank):
            raise ValueError(f'Invalid candidate loading dimensions: {candidate_directory}')
        metrics = candidate_metrics(arrays, offdiag_mask(cov, counts, connected), connected, options)
        print(f'  {"reused" if reused else "computed"}; converged={summary.get("success")}; '
              f'offdiag RMSE={metrics["offdiag_rmse"]}; localized channels={metrics["localized_channel_count"]}', flush=True)
        return arrays, summary, product.token

    arrays, summary = select_rank(cov, counts, connected, options, candidate)
    product = cache.publish(directory, definition, lambda work: publish_arrays(work, arrays, summary))
    require_selected(product, label)
    print(f'Noise rank selected {label}: nc={summary["rank"]}; '
          f'stop={summary["selection"]["stop_reason"]}; offdiag RMSE={summary["offdiag_rmse"]:.6g}', flush=True)
    return product
