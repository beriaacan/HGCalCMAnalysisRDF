"""Resolve and verify DNNs identified by the derivation/training recipe."""
from __future__ import annotations

import json
from pathlib import Path
import re

from .analytic_artifacts import sha, signature


def model_directory(config,module):
    """Only the complete derivation/training recipe selects the source model."""
    from .identity import model_directory as automatic_directory
    group='combined'
    if config.training['training_grouping']=='per_module':
        from .prepared import derivation_modules
        if module not in derivation_modules(config):
            raise ValueError(f'No per-module DNN is derived for {module}; use combined training for cross-module transfer')
        group=module
    return automatic_directory(config,group)


def read_model(folder):
    """Verify the saved bundle itself, without opening its historical inputs."""
    from .dnn_training import SCHEMA, cached
    from .dnn_application import _preprocessing
    folder = Path(folder)
    manifest = folder / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'No DNN for this derivation/training recipe: {manifest}; run train or submit_train with the same source recipe')
    record = json.loads(manifest.read_text())
    definition = record.get('definition')
    if (record.get('schema') != SCHEMA or not isinstance(definition, dict)
            or definition.get('schema') != SCHEMA):
        raise ValueError(f'Unsupported DNN model bundle: {folder}')
    checked = cached(folder, definition)
    if checked is None or checked.get('complete') is not True:
        raise ValueError(f'Selected DNN model is not complete: {folder}')
    from .preprocessing import dnn_definition
    if definition.get('preprocessing') != dnn_definition():
        raise ValueError(f'Unsupported DNN preprocessing contract: {folder}')
    spec = definition['feature_spec']
    if spec.get('schema') != 'hgcal_cm_training_inputs_v2' or spec.get('observation_model') != 'mse':
        raise ValueError(f'Only normal-MSE model application is supported: {folder}')
    if (not isinstance(definition.get('modules'), list) or not definition['modules']
            or set(definition['modules']) - spec['modules'].keys()):
        raise ValueError(f'Invalid saved training-module membership: {folder}')
    if json.loads((folder / 'model_spec.json').read_text()).get('definition') != definition:
        raise ValueError(f'DNN model_spec/manifest mismatch: {folder}')
    pre = json.loads((folder / 'preprocessing.json').read_text())
    _preprocessing(pre, spec)
    validate_features(spec)
    from .training_metadata import read, check_model_identity
    check_model_identity(folder, read(folder, checked))
    return {'folder': folder, 'record': checked, 'definition': definition}


def activity_thresholds(spec):
    if 'activity_fractions' not in spec['global_groups']:
        return []
    values = []
    for name in spec['feature_names'][:spec['global_size']]:
        match = re.fullmatch(r'nchadcgt([0-9]+(?:p[0-9]+)?)_fraction', name)
        if match:
            values.append(float(match[1].replace('p', '.')))
    return values


def validate_features(spec):
    """Require a feature layout understood by the existing shared expanders."""
    import math
    from .prepare_rdf import threshold_label
    width = spec['max_channels']
    if type(width) is not int or width != 444:
        raise ValueError('Selected DNN must use the common 444-channel representation')
    names, adc = [], [-1]*width
    for group in spec['global_groups']:
        if group == 'cm_erx':
            names += [f'cm_erx{i:02d}' for i in range(width//37)]
        elif group == 'activity_fractions':
            names += ['nchtoa_fraction', 'nchtot_fraction']
            names += [f'nchadcgt{threshold_label(t)}_fraction' for t in activity_thresholds(spec)]
        elif group in {'module_id', 'module_fraction'}:
            names.append(group)
        elif group == 'adc_allchannels':
            adc = list(range(len(names), len(names)+width))
            names += [f'adc_allch_{i:03d}' for i in range(width)]
        else:
            raise ValueError(f'Unsupported saved DNN feature group: {group}')
    if (spec['global_size'] != len(names) or spec['adc_columns'] != adc
            or set(spec['local_features']) - {'channel_in_erx', 'erx_index', 'cell_area_fraction'}
            or names + spec['local_features'] != spec['feature_names']
            or spec['unknown_module_id'] != 0 or not math.isfinite(spec['missing_input_value'])):
        raise ValueError('Selected DNN feature ordering/masking is incompatible with the inference reader')
    from .preprocessing import calibration_only, CENTERED_CM
    if 'module_fraction' in spec['global_groups']:
        if ('module_nch' in spec['global_groups'] or spec.get('module_fraction_denominator') != 444
                or not calibration_only(spec)):
            raise ValueError('Invalid saved module_fraction feature convention')
    if not calibration_only(spec) or 'module_fraction' not in spec['global_groups']:
        raise ValueError('DNN requires the fixed physical preprocessing and module_fraction feature')
    if calibration_only(spec) and spec.get('module_id_convention') != 'stable_registry_v1':
        raise ValueError('Calibration-only models require the persistent module-ID convention')
    if spec.get('cm_convention') != CENTERED_CM:
        raise ValueError('Unsupported saved DNN CM convention')


def resolve_application(config):
    """Resolve source-recipe models and application-only prepared inputs."""
    import apply as analytic
    from .prepared import prepared_manifest, module_signature, select_modules
    from . import dnn_application as application
    from .application import SCHEMA, TREE, COLUMNS
    from .inference_config import method_directory
    from .paths import resolve_paths
    from .prepared import plan, resolve, cached_files
    metadata, sources, available_thresholds = {}, {}, {}
    for job in plan(config):
        if 'apply' not in job.roles:
            continue
        prepared = resolve(config, job)
        prepared = select_modules(prepared, analytic._selected(config.target.modules, prepared['modules']))
        if cached_files(prepared) is None:
            raise FileNotFoundError(f'Run --steps prepare before DNN application: {prepared["destination"]}')
        manifest = prepared_manifest(prepared)
        for name in analytic._selected(config.target.modules, prepared['modules']):
            info = prepared['modules'][name]
            if name in metadata and metadata[name]['identity'][2] != info['identity'][2]:
                raise ValueError(f'Application module channel count varies across sources: {name}')
            metadata[name] = info
            available = set(manifest['definition']['activity_thresholds'])
            available_thresholds[name] = available_thresholds.get(name, available) & available
            for source in prepared['files']:
                original = next(row for row in manifest['outputs'] if row['module'] == name
                                and row['source_file_index'] == source['index'])
                sources.setdefault(name, []).append({
                    'run': job.source.run, 'source_file_index': source['index'], 'source_uuid': source['uuid'],
                    'input_file': str(prepared['destination']/original['file']),
                    'prepared_signature': module_signature(prepared, name), 'input_entries': source['entries'],
                    'input_size': original['size'], 'input_mtime_ns': original['mtime_ns'],
                    'output_file': str(Path(f'Run{job.source.run}')/Path(original['file']).name)})
    if not metadata:
        raise ValueError('No application modules selected')
    models, modules, specs = {}, {}, {}
    for name, inputs in sources.items():
        folder = model_directory(config, name)
        group = str(folder)
        if group not in models:
            models[group] = read_model(folder)
        model = models[group]
        if name not in model['definition']['modules'] and not config.dnn['allow_unknown_modules']:
            raise ValueError(f'{name} was not trained in selected model {folder}; '
                             'set allow_unknown_modules: true explicitly to apply it')
        required = activity_thresholds(model['definition']['feature_spec'])
        missing = set(required) - available_thresholds[name]
        if missing:
            raise ValueError(f'{name} prepared cache lacks activity thresholds required by its saved model: {sorted(missing)}')
        per_target = application.application_spec(config, model['definition']['feature_spec'], {name: metadata[name]})
        # Each target may use a different model's vocabulary and normalization.
        specs[name] = per_target
        if len({item['output_file'] for item in inputs}) != len(inputs):
            raise ValueError(f'Application output filenames overlap for {name}')
        model_hash = next(row['sha256'] for row in model['record']['outputs'] if row['file'] == 'model.pt')
        modules[name] = {'inputs': inputs, 'group': group, 'definition': {
            'nch': per_target['modules'][name]['nch'], 'geometry': per_target['modules'][name],
            'predictor': str(folder/'model.pt'), 'predictor_sha256': model_hash,
            'model_signature': model['record']['signature'],
            'feature_spec_sha256': signature(per_target), 'inputs': inputs}}
    recipe = config.target
    definition = {'schema': SCHEMA, 'method': 'dnn',
        'target': {'campaign': recipe.campaign, 'runs': [s.run for s in recipe.dataset.sources],
            'pedestal_run': recipe.pedestal.run, 'selection': recipe.selection,
            'trigger_time_window': recipe.trigger_time_window, 'adc_min': recipe.dataset.adcmin, 'adc_max': recipe.dataset.adcmax},
        'allow_unknown_modules': config.dnn['allow_unknown_modules'],
        'modules': {name: row['definition'] for name, row in modules.items()}, 'tree': TREE, 'columns': COLUMNS,
        'invalid_channels': 'both vectors NaN for rejected/nonfinite application ADC',
        'vector_dtype': 'float32; corrected ADC = selected pedestal-subtracted ADC - stored prediction',
        'code_sha256': application.application_code_hashes(config)}
    return {'destination': Path(resolve_paths(config)['target']['predictions'])/method_directory(config, 'dnn'),
            'definition': json.loads(json.dumps(definition)), 'modules': modules, 'models': models,
            'feature_spec': next(iter(specs.values())), 'feature_specs': specs}
