"""Derive the CM-only linear correction using the shared conditional OLS workflow."""
from __future__ import annotations

from pathlib import Path

from analytic_allinputs import _definition as full_definition, _produce, cached_predictors
from utils.paths import resolve_paths
from utils.analytic_artifacts import sha

ROOT_DIR = Path(__file__).resolve().parent


def cm_manifest(manifest):
    """Project the feature contract, without changing source identities or targets."""
    original = manifest['feature_spec']
    width = original['max_channels']//37
    names = [f'cm_erx{i:02d}' for i in range(width)]
    spec = dict(original, feature_names=names, feature_transforms=['zscore']*width,
                global_groups=['cm_erx'], global_size=width, local_features=[],
                adc_columns=[-1]*original['max_channels'])
    return dict(manifest, feature_spec=spec)


def describe(config):
    return {'destination': str(Path(resolve_paths(config)['analytic_predictors'])/'cm'),
            'training_inputs': resolve_paths(config)['training_inputs'],
            'method': 'analytic', 'predictors': 'calibration-centered CM averages from every real eRx; no channel ADC inputs',
            'fit': 'per module and target: centered/unit-variance conditional OLS with intercept',
            'population': 'same retained training events and per-target ADC selection as analytic_allinputs',
            'analytic_rcond': config.diagnostics['analytic_rcond'],
            'sample_weighting': 'unweighted; sample_weighting applies only to the DNN loss',
            'validation': 'not used or evaluated',
            'publication': 'same per-module NPZ/JSON artifacts, completion checks and resume as analytic_allinputs'}


def definition(config, directory, manifest):
    projected = cm_manifest(manifest)
    result = full_definition(config, directory, projected)
    result.update(method='analytic', predictors='per-eRx CM minus the calibration reference; real eRx columns; target-conditional intercept',
                  reference_difference='centered conditional OLS with intercept, rather than the old mixed-population zero-intercept estimator')
    result['code_sha256']['analytic.py'] = sha(ROOT_DIR/'analytic.py')
    return result


def run(config):
    from training_inputs import load_inputs
    print('Checking training-input provenance for analytic (CM only)', flush=True)
    directory, manifest = load_inputs(config)
    expected = definition(config, directory, manifest)
    destination = Path(describe(config)['destination'])
    existing = cached_predictors(destination, expected)
    _produce(config, directory, cm_manifest(manifest), expected, destination, existing=existing)
