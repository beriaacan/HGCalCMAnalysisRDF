"""Automatic recipe identities; labels never define which physical data are used."""
from __future__ import annotations

from functools import lru_cache
import glob
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'rdf2'


def canonical(value):
    return json.loads(json.dumps(value, default=str, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def variant(value):
    return FORMAT + '_' + digest(value)[:20]


@lru_cache(maxsize=256)
def _sha(path, size, mtime):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_identity(path):
    path = Path(path)
    if not path.is_file():
        return {'path': str(path), 'missing': True}
    stat = path.stat()
    return {'path': str(path), 'sha256': _sha(str(path), stat.st_size, stat.st_mtime_ns)}


def code_identity(names):
    return {name: file_identity(ROOT/name)['sha256'] for name in names}


def inventory(source):
    """Lightweight discovery for path resolution; ROOT UUIDs are checked at preflight."""
    paths = sorted(path for path in glob.glob(source.input_pattern) if Path(path).is_file())
    if len({Path(path).parent for path in paths}) > 1:
        raise ValueError(f'Run{source.run} selects multiple Nano productions; pin one production')
    return {'run': source.run, 'pattern': source.input_pattern,
            'files': [{'path': path, 'size': Path(path).stat().st_size,
                       'mtime_ns': Path(path).stat().st_mtime_ns} for path in paths]}


def preparation_recipe(source, campaign, pedestal_run, calibration, locator, thresholds, conditions=None, cell_areas=None):
    central = ROOT.parents[1]/'LocalCalibration/python/DigiAnalysisUtils.py'
    result = {'format': FORMAT, 'campaign': campaign, 'source': inventory(source),
            'pedestal_run': pedestal_run, 'calibration': file_identity(calibration),
            'module_map': file_identity(locator), 'activity_thresholds': list(thresholds),
            'code': code_identity(('prepare.py', 'utils/prepared.py', 'utils/prepare_rdf.py',
                                   'utils/prepare_helpers.hxx', 'utils/nano.py', 'utils/identity.py',
                                   'utils/preprocessing.py','utils/datasets.py','utils/conditions.py')),
            'central_selection': file_identity(central)}
    if conditions:
        result['conditions']=conditions
    if cell_areas is not None:
        result['cell_areas']=file_identity(cell_areas)
    return result


def population(config, *, target=False):
    if not target and config.derivations:
        from .datasets import recipes, single
        return {'datasets':[population(single(config,row)) for row in recipes(config)]}
    recipe = config.target if target else config.correction
    out = {'format': FORMAT, 'campaign': recipe.campaign,
           'sources': [preparation_recipe(source, recipe.campaign, recipe.pedestal.run,
                       recipe.calibration_file, recipe.module_map,
                       config.preparation['activity_thresholds'],recipe.conditions,
                       config.dnn['cell_area_file']) for source in recipe.dataset.sources],
           'calibration': file_identity(recipe.calibration_file),
           'module_map': file_identity(recipe.module_map), 'pedestal_run': recipe.pedestal.run,
           'selection': recipe.selection, 'trigger_time_window': recipe.trigger_time_window,
           'adc_min': recipe.dataset.adcmin, 'adc_max': recipe.dataset.adcmax,
           'activity_thresholds': config.preparation['activity_thresholds']}
    if recipe.conditions:
        out['conditions']=recipe.conditions
    if not target:
        out['modules'] = recipe.modules
        out['split'] = {key: config.dnn[key] for key in ('split_seed', 'validation_split_fraction',
                       'train_event_fraction', 'validation_event_fraction')}
    return canonical(out)


def selected_module_ids(config):
    """Pin relevant permanent IDs without invalidation from unrelated registry appends."""
    if config.dnn['preprocessing_mode'] != 'calibration_only':
        return None
    path = config.dnn['module_registry_file']
    registry = json.loads(path.read_text()).get('modules', {}) if path.is_file() else {}
    selected = config.correction.modules
    if config.derivations:
        from .prepared import derivation_modules
        selected=derivation_modules(config)
    if not isinstance(selected, str):
        return {name: registry.get(name) for name in selected}
    # A selector's exact observed inventory is signed during preparation. The
    # locator limits this lightweight lookup without opening Nano event data.
    from .prepared import _locator
    available = _locator(config.correction.module_map) if config.correction.module_map.is_file() else {}
    return {name: registry.get(name) for name in sorted(available)
            if selected=='all' or selected=='ld' and name.startswith('ML_')
            or selected=='hd' and name.startswith('MH_')}


def input_recipe(config):
    return {'data': population(config), 'module_ids': selected_module_ids(config),
            'features': {key: config.dnn[key] for key in ('preprocessing_mode', 'preprocess_inputs',
                         'preprocess_targets', 'input_features', 'max_channels', 'missing_input_value')},
            'cell_areas': file_identity(config.dnn['cell_area_file']),
            'code': code_identity(('training_inputs.py', 'utils/training.py', 'utils/training_helpers.hxx',
                    'utils/preprocessing.py', 'utils/module_registry.py', 'utils/datasets.py', 'utils/conditions.py', 'utils/prepare_rdf.py',
                    'utils/prepare_helpers.hxx', 'utils/identity.py'))}


def input_id(config):
    return variant(input_recipe(config))


def linear_id(config):
    from .preprocessing import statistics_config
    return variant({'inputs': input_recipe(statistics_config(config)),
                    'rcond': config.diagnostics['analytic_rcond'],
                    'code': code_identity(('analytic.py', 'analytic_allinputs.py', 'utils/analytic.py',
                                          'utils/analytic_fit.py', 'utils/analytic_helpers.hxx'))})


def model_recipe(config, group):
    dnn_keys = ('nodes', 'dropout', 'epochs', 'batch_events', 'learning_rate', 'weight_decay',
                'shuffle_buffer_chunks', 'sample_weighting')
    training_keys = ('training_grouping', 'training_seed', 'training_deterministic',
        'training_events_per_chunk', 'optimizer_betas', 'optimizer_epsilon', 'early_stopping_patience',
        'early_stopping_min_delta', 'lr_scheduler_factor', 'lr_scheduler_patience',
        'lr_scheduler_threshold', 'lr_scheduler_min_lr')
    from .preprocessing import dnn_definition
    return {'inputs': input_recipe(config), 'group': group, 'preprocessing': dnn_definition(),
            'dnn': {key: config.dnn[key] for key in dnn_keys},
            'training': {key: config.training[key] for key in training_keys},
            'runtime': {'manifest':file_identity(config.training['training_environment']/'runtime.json'),
                'python_environment':file_identity(config.training['training_environment']/'pyvenv.cfg'),
                'torch_build':file_identity(config.training['training_environment']/
                    f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/torch/version.py')},
            'code': code_identity(('train.py', 'utils/dnn_training.py', 'utils/dnn_batches.py',
                'utils/dnn_prefetch.py', 'utils/dnn_tensors.py', 'utils/training_cache.py',
                'utils/dnn_internal_normalization.py', 'utils/training_metadata.py', 'utils/identity.py',
                'utils/training.py', 'utils/training_helpers.hxx', 'utils/train_config.py',
                'utils/dnn_provenance.py', 'utils/training_progress.py', 'utils/training_runtime.py',
                'utils/model_compatibility.py','utils/datasets.py','utils/conditions.py'))}


def model_id(config, group='combined'):
    return variant(model_recipe(config, group))


def model_directory(config, group='combined'):
    return config.model_dir/'dnn'/model_id(config, group)/group


def data_id(training_recipe, modules):
    """Full, automatic physical population identity, independent of DNN architecture."""
    if training_recipe.get('datasets'):
        population={'datasets':training_recipe['datasets'],'modules':list(modules),
                    'split':training_recipe['options'],'split_identity':training_recipe['split_identity'],
                    'prepared':[{k:row['preparation_definition'][k] for k in
                        ('campaign','run','files','pedestal_run','calibration_sha256','module_map_sha256','activity_thresholds')}
                        | {'dataset_index':row['dataset_index'],
                           'conditions':row['preparation_definition'].get('conditions')}
                        for row in training_recipe['prepared']]}
        return digest(population)
    population = {key: training_recipe[key] for key in ('campaign', 'selection',
                  'trigger_time_window', 'adc_min', 'adc_max', 'split_identity')}
    population['split'] = {key: training_recipe['options'][key] for key in
        ('split_seed', 'validation_split_fraction', 'train_event_fraction', 'validation_event_fraction')}
    population['modules'] = list(modules)
    population['prepared'] = []
    for row in training_recipe['prepared']:
        definition = row['preparation_definition']
        population['prepared'].append({
            key: definition[key] for key in ('campaign', 'run', 'files', 'pedestal_run',
                         'calibration_sha256', 'module_map_sha256', 'activity_thresholds')})
    return digest(population)
