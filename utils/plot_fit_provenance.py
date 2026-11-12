"""Shared noise-fit identity and precise reuse of unchanged spectrum code."""
from __future__ import annotations

from pathlib import Path
from .analytic_artifacts import sha


def spectrum_code_hash(path):
    return sha(Path(path))


def noise_fit_definition(config, covariance_token, *, rank=None):
    return {'covariance': covariance_token, 'rank': config.diagnostics['n_coherent_noise_model'] if rank is None else rank,
            'maxiter': config.plotting['plot_noise_maxiter'],
            'gtol': config.plotting['plot_noise_gtol'], 'ftol': config.plotting['plot_noise_ftol'],
            'fit_code_sha256': sha(Path(__file__).with_name('plot_fits.py'))}
