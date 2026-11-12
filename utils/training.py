"""Feature contract, RDF views and bounded NumPy batches for normal MSE inputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .prepare_rdf import analysis_view, initialize_root, threshold_label

SCHEMA = 'hgcal_cm_training_inputs_v2'
_READY = False
LOCAL_KINDS = {'channel_in_erx': 0, 'erx_index': 1, 'cell_area_fraction': 2}


def initialize_training_root():
    global _READY
    ROOT = initialize_root()
    if not _READY:
        header = Path(__file__).with_name('training_helpers.hxx')
        if not ROOT.gInterpreter.Declare(f'#include {json.dumps(str(header))}'):
            raise RuntimeError('Could not compile training input helpers')
        _READY = True
    return ROOT


def source_salt(campaign: str, run: int, uuid: str) -> int:
    identity = json.dumps([campaign, run, uuid.lower()], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big')


def make_feature_spec(config, modules: dict) -> dict:
    settings = config.dnn
    width = settings['max_channels']
    areas = json.loads(settings['cell_area_file'].read_text())
    geometries = {}
    from .preprocessing import calibration_only, CENTERED_CM, reference_from_file
    persistent_ids = None
    if calibration_only(settings):
        from .module_registry import module_ids
        persistent_ids = module_ids(sorted(modules), path=str(settings['module_registry_file']))
        if any(index > 2**24 for index in persistent_ids.values()):
            raise ValueError('Module ID is not exactly representable as a float32 input')
    for module_id, name in enumerate(sorted(modules), 1):
        if persistent_ids is not None:
            module_id = persistent_ids[name]
        nerx = modules[name]['identity'][2]
        nch = 37*nerx
        if nch > width:
            raise ValueError(f'{name} has {nch} channels, above max_channels={width}')
        kind = name[:4]
        values = areas.get(kind, {}).get('SF')
        if (not isinstance(values, list) or len(values) != nch
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError(f'Invalid central cell-area fractions for {name} ({kind}, {nch} channels)')
        geometries[name] = {'module_id': module_id, 'nch': nch, 'nerx': nerx,
                            'cell_area_fraction': values}
    names, transforms, adc_columns, local = [], [], [-1]*width, []
    global_groups = []
    for group in settings['input_features']:
        if group in LOCAL_KINDS:
            local.append(group)
            continue
        global_groups.append(group)
        if group == 'cm_erx':
            fields = [f'cm_erx{i:02d}' for i in range(width//37)]
            modes = ['zscore']*len(fields)
        elif group == 'adc_allchannels':
            fields = [f'adc_allch_{i:03d}' for i in range(width)]
            adc_columns = list(range(len(names), len(names)+width))
            modes = ['zscore']*width
        elif group == 'activity_fractions':
            fields = ['nchtoa_fraction', 'nchtot_fraction'] + [
                f'nchadcgt{threshold_label(t)}_fraction' for t in config.preparation['activity_thresholds']]
            modes = ['identity']*len(fields)
        else:
            fields = [group]
            modes = ['zscore' if group == 'module_nch' else 'identity']
        names.extend(fields)
        transforms.extend(modes)
    global_size = len(names)
    names.extend(local)
    transforms.extend(['identity']*len(local))
    result = {'schema': SCHEMA, 'observation_model': 'mse', 'feature_names': names,
            'feature_transforms': transforms, 'global_size': global_size,
            'global_groups': global_groups, 'local_features': local,
            'adc_columns': adc_columns, 'max_channels': width,
            'missing_input_value': settings['missing_input_value'], 'modules': geometries,
            'unknown_module_id': 0, 'target_adc_mask': 'configured derivation bounds; finite real channels only',
            'target_input_mask': 'replace own-channel ADC with missing_input_value BEFORE preprocessing',
            'cm_convention': CENTERED_CM,
            'missing_inputs': 'fill nonfinite and padded values; no additional observation flags',
            'target_preprocessing': 'per-channel statistics pooled across training modules',
            'split_identity': 'uint64 Nano event within campaign/run/source-file UUID; module independent',
            'split_labels': {'0': 'train', '1': 'validation'}, 'independent_test_set': False}
    if 'module_fraction' in global_groups:
        result['module_fraction_denominator'] = 444
    if calibration_only(settings):
        result.update(preprocessing_mode='calibration_only', cm_convention=CENTERED_CM,
                      module_id_convention='stable_registry_v1',
                      feature_transforms=['identity']*len(names),
                      target_preprocessing='none; pedestal-subtracted ADC units')
    # Both regressions and the DNN use the same calibration-defined CM reference.
    if 'cm_erx' in global_groups:
        for name, meta in geometries.items():
            reference_from_file(config.correction.calibration_file, name, meta['nch'],meta['cell_area_fraction'])
    return result


def global_expression(config, spec: dict, module: str, *, recipe=None) -> str:
    meta = spec['modules'][module]
    width, fill = spec['max_channels'], spec['missing_input_value']
    statements = ['ROOT::RVec<float> out;']
    from .preprocessing import CENTERED_CM, reference_from_file
    cm = 'cm_erx'
    if 'cm_erx' in spec['global_groups']:
        if spec.get('cm_convention') != CENTERED_CM:
            raise ValueError('CM features must use the calibration-centered convention; regenerate linear inputs/predictors')
        from .prepare_rdf import _vector
        reference = reference_from_file((recipe or config.correction).calibration_file, module, meta['nch'],meta['cell_area_fraction'])
        statements.append(f'static const auto cm_reference = {_vector(reference,allow_missing=True)};')
        cm = '(cm_erx - cm_reference)'

    for group in spec['global_groups']:
        if group == 'cm_erx':
            statements.append(f'hgcal_cm_training::append(out,hgcal_cm_training::padded({cm},{width//37},{fill:.17g}));')
        elif group == 'adc_allchannels':
            statements.append(f'hgcal_cm_training::append(out,hgcal_cm_training::padded(adc_pedsub,{width},{fill:.17g}));')
        elif group == 'activity_fractions':
            columns = ['nchtoa', 'nchtot'] + [f'nchadcgt{threshold_label(t)}' for t in config.preparation['activity_thresholds']]
            statements.extend(f'out.push_back(float({column})/{meta["nch"]}.f);' for column in columns)
        elif group == 'module_id':
            statements.append(f'out.push_back({meta["module_id"]}.f);')
        elif group == 'module_nch':
            statements.append(f'out.push_back({meta["nch"]}.f);')
        elif group == 'module_fraction':
            if spec.get('module_fraction_denominator') != 444:
                raise ValueError('module_fraction requires the saved fixed denominator 444')
            statements.append(f'out.push_back({meta["nch"]}.f / 444.f);')
    return ' '.join(statements + ['return out;'])


def training_view(rdf, config, spec: dict, module: str, source: dict, *, statistics_only=False):
    """Reconstruct selected events and features from a shared prepared module file."""
    initialize_training_root()
    if 'test' in config.correction.selection:
        raise ValueError('training_inputs requires derive_selection full or trigtime; '
                         'held-out selections cannot define training events')
    seed = config.dnn['split_seed']
    salt = source_salt(config.correction.campaign, source['run'], source['uuid'])
    rdf = rdf.Define('training_split', f'hgcal_cm_training::split(ULong64_t(event),{salt}ULL,{seed}ULL,'
                     f'{config.dnn["validation_split_fraction"]:.17g})')
    rdf = rdf.Define('training_retained', f'hgcal_cm_training::retained(ULong64_t(event),{salt}ULL,{seed}ULL,'
                     f'training_split,{config.dnn["train_event_fraction"]:.17g},'
                     f'{config.dnn["validation_event_fraction"]:.17g})')
    rdf = analysis_view(rdf, config.correction)
    rdf = rdf.Filter('module_present', 'module present for training inputs')
    rdf = rdf.Define('training_n_targets', 'hgcal_cm_training::count_targets(adc_selected)')
    rdf = rdf.Filter('training_n_targets>0', 'at least one selected training target')
    expression = global_expression(config, spec, module)
    if statistics_only:
        enabled = str(config.dnn['preprocess_inputs']).lower()
        expression = (f'if (!(training_split==0 && training_retained && {enabled})) '
                      'return ROOT::RVec<float>{}; ' + expression)
    return rdf.Define('training_global_features', expression)


def book_statistics(rdf, config, spec: dict, module: str):
    ROOT = initialize_training_root()
    node = rdf.Define('training_payload', 'hgcal_cm_training::Row{training_global_features, '
                      'adc_selected, training_split, training_retained}')
    return ROOT.hgcal_cm_training.book_statistics(
        ROOT.RDF.AsRNode(node), len(spec['feature_names']), spec['max_channels'],
        ROOT.VecOps.RVec['int'](spec['adc_columns']),
        ROOT.VecOps.RVec['int']([LOCAL_KINDS[name] for name in spec['local_features']]),
        ROOT.VecOps.RVec['float'](spec['modules'][module]['cell_area_fraction']),
        spec['missing_input_value'], config.dnn['preprocess_inputs'])


def preprocessing_payload(stats, spec: dict, config) -> dict:
    from .preprocessing import calibration_only, descriptor
    if calibration_only(spec):
        return descriptor(spec)
    def moments(values, enabled, modes=None):
        means, scales, counts = [], [], []
        for index, item in enumerate(values):
            active = enabled and (modes is None or modes[index] == 'zscore')
            variance = max(0., float(item.m2)/item.count) if item.count else 0.
            scale = math.sqrt(variance) if variance > 0 else 1.
            means.append(float(item.mean) if active else 0.)
            scales.append(scale if active else 1.)
            counts.append(int(item.count))
        return {'mean': means, 'std': scales, 'count': counts}
    return {'fit_partition': 'retained train events, finite selected targets only',
            'feature_names': spec['feature_names'], 'feature_transforms': spec['feature_transforms'],
            'inputs_enabled': config.dnn['preprocess_inputs'],
            'targets_enabled': config.dnn['preprocess_targets'],
            'inputs': moments(stats.inputs, config.dnn['preprocess_inputs'], spec['feature_transforms']),
            'targets': moments(stats.targets, config.dnn['preprocess_targets']),
            'zero_std_policy': 'mean zero and scale one for unobserved; scale one for constants',
            'variance_convention': 'population; stable weighted Welford/Chan accumulation'}


def iter_sample_batches(global_features, targets, spec: dict, module: str, batch_size: int,
                        preprocessing: dict | None = None):
    """Expand an event chunk into bounded (event,target) NumPy batches; no pandas."""
    import numpy as np
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    meta = spec['modules'][module]
    if preprocessing is not None:
        from .preprocessing import validate_descriptor
        validate_descriptor(preprocessing, spec)
    global_features = np.asarray(global_features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if global_features.ndim != 2 or global_features.shape[1] != spec['global_size']:
        raise ValueError('Global feature matrix has an incompatible shape')
    if targets.shape != (len(global_features), meta['nch']):
        raise ValueError('Target matrix must have one column per real module channel')
    rows, channels = np.nonzero(np.isfinite(targets))
    areas = np.asarray(meta['cell_area_fraction'], dtype=np.float32)
    adc_columns = np.asarray(spec['adc_columns'], dtype=int)
    for start in range(0, len(rows), batch_size):
        rr, cc = rows[start:start+batch_size], channels[start:start+batch_size]
        x = np.empty((len(rr), len(spec['feature_names'])), dtype=np.float32)
        x[:, :spec['global_size']] = global_features[rr]
        own = adc_columns[cc]
        valid = own >= 0
        x[np.flatnonzero(valid), own[valid]] = spec['missing_input_value']
        for offset, feature in enumerate(spec['local_features'], spec['global_size']):
            x[:, offset] = cc%37 if feature == 'channel_in_erx' else cc//37 if feature == 'erx_index' else areas[cc]
        y = targets[rr, cc].copy()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite expanded training sample')
        yield {'inputs': x, 'targets': y, 'event_rows': rr, 'channels': cc}


def sample_weights(manifest: dict, module: str, run: int, partition: str, channels, *, weighting: str):
    """Mean-one inverse (source run, target channel) frequency within each module."""
    import numpy as np
    if partition not in {'train', 'validation'}:
        raise ValueError('partition must be train or validation')
    channels = np.asarray(channels, dtype=int)
    if weighting not in {'none', 'source_run_channel'}:
        raise ValueError('weighting must be none or source_run_channel')
    if weighting == 'none':
        return np.ones(channels.shape, dtype=np.float32)
    counts = np.asarray(manifest['counts_by_module_run'][module][str(run)][partition], dtype=np.int64)
    selected = counts[channels]
    if np.any(selected <= 0):
        raise ValueError('Requested weights for a channel with no retained targets in this partition')
    scale = manifest['inverse_frequency_scales'][module][partition]
    return (scale/selected).astype(np.float32)
