"""RDF application of physical-unit analytic coefficients to prepared measurements."""
from __future__ import annotations

import json
from pathlib import Path

from .analytic import candidate_columns
from .prepare_rdf import analysis_view
from .training import global_expression, initialize_training_root

SCHEMA = 'hgcal_cm_application_v2'
TREE = 'Predictions'
COLUMNS = ['source_run', 'source_file_index', 'source_entry', 'event_id', 'event',
           'prediction', 'adc_corrected']
_READY = False


def initialize_application_root():
    global _READY
    ROOT = initialize_training_root()
    if not _READY:
        header = Path(__file__).with_name('apply_helpers.hxx')
        if not ROOT.gInterpreter.Declare(
                '#ifndef EIGEN_DONT_PARALLELIZE\n#define EIGEN_DONT_PARALLELIZE\n#endif\n'
                f'#include {json.dumps(str(header))}'):
            raise RuntimeError('Could not compile application helpers; use the current hgc environment')
        _READY = True
    return ROOT


def validate_model(model: dict, spec: dict, module: str) -> list[int]:
    import numpy as np
    nch, width = spec['modules'][module]['nch'], len(spec['feature_names'])
    from .preprocessing import CENTERED_CM
    if spec.get('cm_convention') != CENTERED_CM:
        raise ValueError('Linear predictors require calibration-centered CM inputs; rederive old raw-CM predictors')
    columns = candidate_columns(spec, module)
    required = {'coefficients', 'intercepts', 'fitted', 'feature_names', 'target_channels', 'candidate_columns'}
    if required - model.keys():
        raise ValueError(f'Incomplete analytic model for {module}: {sorted(required-model.keys())}')
    weights, offsets, fitted = model['coefficients'], model['intercepts'], model['fitted']
    if (weights.shape != (nch, width) or offsets.shape != (nch,) or fitted.shape != (nch,)
            or fitted.dtype.kind != 'b'
            or not np.array_equal(model['feature_names'], spec['feature_names'])
            or not np.array_equal(model['target_channels'], np.arange(nch))
            or not np.array_equal(model['candidate_columns'], columns)):
        raise ValueError(f'Analytic feature/channel mapping mismatch: {module}')
    if not np.isfinite(weights).all() or not np.isfinite(offsets[fitted]).all():
        raise ValueError(f'Nonfinite fitted analytic coefficients: {module}')
    excluded = np.ones(width, dtype=bool)
    excluded[columns] = False
    if np.any(weights[:, excluded] != 0):
        raise ValueError(f'Analytic model fits omitted constant or padded features: {module}')
    for channel in range(nch):
        own = spec['adc_columns'][channel]
        if own >= 0 and weights[channel, own] != 0:
            raise ValueError(f'Own-channel ADC coefficient is nonzero: {module}/channel {channel}')
    return columns


def make_predictor(model: dict, spec: dict, module: str):
    import numpy as np
    columns = validate_model(model, spec, module)
    ROOT = initialize_application_root()
    return ROOT.hgcal_cm_apply.Predictor(
        ROOT.VecOps.RVec['int'](columns),
        ROOT.VecOps.RVec['double'](np.ascontiguousarray(model['coefficients'][:, columns], dtype=np.float64).ravel()),
        ROOT.VecOps.RVec['double'](np.asarray(model['intercepts'], dtype=np.float64)),
        ROOT.VecOps.RVec['int'](model['fitted'].astype(np.int32)), spec['global_size'])


def application_view(rdf, config, spec: dict, module: str, predictor):
    """Apply target event/channel selections; build all predictors from uncut ADC."""
    ROOT = initialize_application_root()
    if config.target.selection not in {'full', 'trigtime'}:
        raise NotImplementedError('apply_selection test/test_trigtime requires a later explicit evaluation '
                                  'implementation; use full or trigtime for application')
    node = analysis_view(rdf, config.target).Filter('module_present', 'module present for application')
    node = node.Define('application_features', global_expression(config, spec, module, recipe=config.target))
    return ROOT.hgcal_cm_apply.define_predictions(ROOT.RDF.AsRNode(node), predictor)
