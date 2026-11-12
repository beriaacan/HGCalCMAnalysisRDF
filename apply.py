"""Apply saved analytic predictors and persist compact prediction/corrected-ADC trees."""
from __future__ import annotations
from utils.prepared import prepared_manifest, module_signature, select_modules
from utils.inference_config import method_directory

import json
import os
from pathlib import Path
import tempfile

from analytic_allinputs import _definition as model_definition
from analytic_allinputs import _inventory, _write_manifest, cached_predictors
from utils.analytic import load_module
from utils.analytic_artifacts import sha, signature
from utils.application import COLUMNS, SCHEMA, TREE, validate_model
from utils.preprocessing import reference_from_file
from utils.config import AnalysisConfig
from utils.paths import resolve_paths
from utils.prepared import cached_files, plan, resolve

ROOT_DIR = Path(__file__).resolve().parent
SUPPORTED_METHODS = {'uncorrected', 'analytic_allinputs'}


def describe(config: AnalysisConfig) -> dict:
    return {'runs': [source.run for source in config.target.dataset.sources],
            'methods': config.methods,
            'unsupported_methods': [name for name in config.methods if name not in SUPPORTED_METHODS],
            'destination': str(Path(resolve_paths(config)['target']['predictions']) / method_directory(config, 'analytic_allinputs')),
            'predictors': str(Path(resolve_paths(config)['analytic_predictors']) / 'allinputs'),
            'selection': config.target.selection, 'trigger_time_window': config.target.trigger_time_window,
            'adc_min': config.target.dataset.adcmin, 'adc_max': config.target.dataset.adcmax,
            'tree': TREE, 'columns': COLUMNS, 'vector_dtype': 'float32',
            'uncorrected': 'read pedestal-subtracted ADC from the shared prepared cache; no new files',
            'publication': 'one completed module at a time; matching modules reused on restart',
            'evaluation': 'no noise statistics, validation evaluation or plots'}


def _selected(selection, available):
    if not isinstance(selection, str):
        missing = set(selection) - available.keys()
        if missing:
            raise ValueError(f'Application modules missing from prepared inputs: {sorted(missing)}')
        return list(selection)
    return [name for name in available if selection == 'all'
            or selection == 'ld' and name.startswith('ML_')
            or selection == 'hd' and name.startswith('MH_')]


def resolve_application(config: AnalysisConfig) -> dict:
    """Validate models and target caches once, without processing target events."""
    from training_inputs import load_inputs
    unsupported = set(config.methods) - SUPPORTED_METHODS
    if unsupported:
        raise NotImplementedError(f'Application methods not implemented: {sorted(unsupported)}. '
                                  'For this increment use methods: [uncorrected, analytic_allinputs].')
    if config.target.selection not in {'full', 'trigtime'}:
        raise NotImplementedError('Application currently supports full or trigtime; held-out '
                                  'test selectors belong to a later explicit evaluation increment')
    directory, training = load_inputs(config)
    expected_model = model_definition(config, directory, training)
    model_dir = Path(resolve_paths(config)['analytic_predictors']) / 'allinputs'
    stored = cached_predictors(model_dir, expected_model)
    if stored is None:
        raise FileNotFoundError(f'Run --steps analytic_allinputs before application: {model_dir}')
    spec = training['feature_spec']
    modules = {}
    for job in plan(config):
        if 'apply' not in job.roles:
            continue
        prepared = resolve(config, job)
        prepared = select_modules(prepared, _selected(config.target.modules, prepared['modules']))
        if cached_files(prepared) is None:
            raise FileNotFoundError(f'Run --steps prepare for application inputs: {prepared["destination"]}')
        manifest = prepared_manifest(prepared)
        names = _selected(config.target.modules, prepared['modules'])
        for name in names:
            if name not in stored['modules'] or name not in spec['modules']:
                raise ValueError(f'No completed analytic predictor for application module {name}')
            metadata = prepared['modules'][name]
            if 37*metadata['identity'][2] != spec['modules'][name]['nch']:
                raise ValueError(f'Application/model channel count mismatch: {name}')
            if 'cm_erx' in spec['global_groups']:
                reference_from_file(config.target.calibration_file, name, spec['modules'][name]['nch'])
            if name not in modules:
                model, _ = load_module(model_dir, name)
                validate_model(model, spec, name)
                record = next(row for row in stored['outputs'] if row['file'] == f'{name}/predictor.npz')
                modules[name] = {'model': model, 'inputs': [], 'definition': {
                    'nch': spec['modules'][name]['nch'],
                    'predictor': str(model_dir / name / 'predictor.npz'),
                    'predictor_sha256': record['sha256'],
                    'fitted_channels': int(model['fitted'].sum())}}
            for source in prepared['files']:
                original = next(row for row in manifest['outputs'] if row['module'] == name
                                and row['source_file_index'] == source['index'])
                item = {'run': job.source.run, 'source_file_index': source['index'],
                        'source_uuid': source['uuid'],
                        'input_file': str(prepared['destination'] / original['file']),
                        'prepared_signature': module_signature(prepared, name),
                        'input_entries': source['entries'],
                        'input_size': original['size'], 'input_mtime_ns': original['mtime_ns'],
                        'output_file': str(Path(f'Run{job.source.run}') / Path(original['file']).name)}
                modules[name]['inputs'].append(item)
    if not modules:
        raise ValueError('No application modules selected')
    for name, row in modules.items():
        files = [item['output_file'] for item in row['inputs']]
        if len(files) != len(set(files)):
            raise ValueError(f'Application output filenames overlap for {name}')
        row['definition']['inputs'] = row['inputs']
    recipe = config.target
    definition = {'schema': SCHEMA, 'method': 'analytic_allinputs',
                  'target': {'campaign': recipe.campaign,
                             'runs': [source.run for source in recipe.dataset.sources],
                             'pedestal_run': recipe.pedestal.run, 'selection': recipe.selection,
                             'trigger_time_window': recipe.trigger_time_window,
                             'adc_min': recipe.dataset.adcmin, 'adc_max': recipe.dataset.adcmax},
                  'model_definition_signature': signature(expected_model),
                  'feature_spec_sha256': signature(spec),
                  'cm_convention': spec['cm_convention'],
                  'modules': {name: row['definition'] for name, row in modules.items()},
                  'tree': TREE, 'columns': COLUMNS,
                  'invalid_channels': 'both vectors NaN for unavailable models or rejected/nonfinite target ADC',
                  'vector_dtype': 'float32; corrected ADC = selected pedestal-subtracted ADC - stored prediction',
                  'code_sha256': {name: sha(ROOT_DIR / name) for name in
                                  ('apply.py', 'utils/application.py', 'utils/apply_helpers.hxx')}}
    definition = json.loads(json.dumps(definition))
    return {'destination': Path(describe(config)['destination']), 'definition': definition,
            'modules': modules, 'feature_spec': spec}


def _check_record(record, path, definition, module=None):
    stored = record.get('definition', {})
    common = lambda value: {key: item for key, item in value.items() if key != 'modules'}
    if (record.get('schema') != SCHEMA or record.get('signature') != signature(stored)
            or common(stored) != common(definition)
            or module is not None and stored.get('modules', {}).get(module) != definition['modules'][module]):
        raise ValueError(f'Application provenance mismatch: {path}. Existing outputs were not replaced.')


def _check_outputs(directory, outputs, expected):
    if [row['file'] for row in outputs] != expected:
        raise ValueError(f'Incomplete application output inventory: {directory}')
    for row in outputs:
        path = directory / row['file']
        if not path.is_file():
            raise ValueError(f'Missing application output: {path}')
        stat = path.stat()
        if stat.st_size != row['size'] or stat.st_mtime_ns != row['mtime_ns']:
            raise ValueError(f'Modified application output: {path}')


def cached_application(resolved: dict) -> dict | None:
    destination, definition = resolved['destination'], resolved['definition']
    if not destination.exists():
        return None
    path = destination / 'manifest.json'
    if not path.is_file():
        raise ValueError(f'Incomplete application outputs: {destination}')
    overall = json.loads(path.read_text())
    _check_record(overall, path, definition)
    if type(overall.get('complete')) is not bool:
        raise ValueError(f'Invalid application completion status: {path}')
    records = {}
    for module, info in definition['modules'].items():
        target = destination / module
        if not target.exists():
            continue
        path = target / 'manifest.json'
        if not path.is_file():
            raise ValueError(f'Incomplete application module: {target}')
        record = json.loads(path.read_text())
        _check_record(record, path, definition, module)
        if record.get('module') != module or record.get('complete') is not True:
            raise ValueError(f'Invalid application module completion record: {path}')
        _check_outputs(target, record.get('outputs', []), [row['output_file'] for row in info['inputs']])
        records[module] = record
    names = list(definition['modules'])
    summaries, outputs = _inventory(records, names)
    return dict(overall, definition=definition, signature=signature(definition),
                complete=all(name in records for name in names),
                modules=summaries, outputs=outputs, module_records=records)


def _overview(resolved, records, *, complete):
    definition = resolved['definition']
    summaries, outputs = _inventory(records, list(definition['modules']))
    return {'schema': SCHEMA, 'signature': signature(definition), 'definition': definition,
            'complete': complete, 'modules': summaries, 'outputs': outputs}


def _compute_module(config, resolved, module, work):
    from utils.application import application_view, initialize_application_root, make_predictor
    from utils.nano import root_path
    ROOT = initialize_application_root()
    if ROOT.IsImplicitMTEnabled() and ROOT.GetThreadPoolSize() != config.resources['cpus']:
        ROOT.DisableImplicitMT()
    if not ROOT.IsImplicitMTEnabled():
        ROOT.EnableImplicitMT(config.resources['cpus'])
    row = resolved['modules'][module]
    predictor = make_predictor(row['model'], resolved['feature_spec'], module)
    outputs = []
    for source in row['inputs']:
        rdf = ROOT.RDataFrame('Events', root_path(source['input_file']))
        input_count = rdf.Count()
        node = application_view(rdf, config, resolved['feature_spec'], module, predictor)
        selected_count = node.Count()
        target = work / source['output_file']
        target.parent.mkdir(parents=True, exist_ok=True)
        options = ROOT.RDF.RSnapshotOptions()
        options.fLazy = True
        options.fAutoFlush = config.resources['events_per_chunk']
        snapshot = node.Snapshot(TREE, str(target), COLUMNS, options)
        snapshot.GetValue()
        if int(input_count.GetValue()) != source['input_entries']:
            raise ValueError(f'Application prepared-input event count changed: {source["input_file"]}')
        entries = int(selected_count.GetValue())
        stat = target.stat()
        outputs.append({'file': source['output_file'], 'run': source['run'],
                        'source_file_index': source['source_file_index'],
                        'input_events': source['input_entries'], 'selected_events': entries,
                        'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        print(f'{module}: Run{source["run"]}/file{source["source_file_index"]}: '
              f'{entries}/{source["input_entries"]} selected events', flush=True)
    summary = {'input_events': sum(row['input_events'] for row in outputs),
               'selected_events': sum(row['selected_events'] for row in outputs),
               'channels': resolved['feature_spec']['modules'][module]['nch'],
               'fitted_channels': int(row['model']['fitted'].sum())}
    return outputs, summary


def _produce(config, resolved, existing=None):
    destination, definition = resolved['destination'], resolved['definition']
    if existing is None:
        existing = cached_application(resolved)
    if existing is not None and existing['complete']:
        print(f'Reusing applied corrections: {destination}', flush=True)
        return
    records = dict(existing['module_records']) if existing is not None else {}
    if existing is None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
    _write_manifest(destination, _overview(resolved, records, complete=False))
    for module in definition['modules']:
        if module in records:
            print(f'Reusing applied module: {destination / module}', flush=True)
            continue
        work = Path(tempfile.mkdtemp(prefix=f'.{module}.work-', dir=destination))
        try:
            print(f'Applying analytic_allinputs to {module}', flush=True)
            outputs, summary = _compute_module(config, resolved, module, work)
            record = {'schema': SCHEMA, 'signature': signature(definition), 'definition': definition,
                      'module': module, 'complete': True, 'summary': summary, 'outputs': outputs}
            _write_manifest(work, record)
            _check_outputs(work, outputs, [row['output_file'] for row in definition['modules'][module]['inputs']])
            target = destination / module
            if target.exists():
                raise FileExistsError(f'Application module appeared during production: {target}')
            os.rename(work, target)
            records[module] = record
            print(f'Applied module ready: {target}', flush=True)
            _write_manifest(destination, _overview(resolved, records, complete=False))
        except Exception as exc:
            raise RuntimeError(f'{exc}. Completed modules remain in {destination}; '
                               f'unfinished work, if present: {work}') from exc
    _write_manifest(destination, _overview(resolved, records, complete=True))
    print(f'Applied corrections ready: {destination}', flush=True)


def run(config: AnalysisConfig) -> None:
    unsupported = set(config.methods) - SUPPORTED_METHODS
    if unsupported:
        raise NotImplementedError(f'Application methods not implemented: {sorted(unsupported)}. '
                                  'Use methods: [uncorrected, analytic_allinputs] for this increment.')
    if 'analytic_allinputs' not in config.methods:
        print('Uncorrected ADC uses the shared prepared cache; no prediction files are needed.', flush=True)
        return
    print('Checking predictor and prepared-input provenance for application', flush=True)
    resolved = resolve_application(config)
    existing = cached_application(resolved)
    _produce(config, resolved, existing)
