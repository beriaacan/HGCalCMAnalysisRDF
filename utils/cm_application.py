"""CM-only adapter to the existing linear ROOT evaluator and publication format."""
from __future__ import annotations
from utils.prepared import prepared_manifest, module_signature, select_modules
from utils.inference_config import method_directory

import json
import os
from pathlib import Path
import tempfile

import analytic
import apply as linear
from .analytic_artifacts import sha, signature
from .analytic import load_module
from .application import COLUMNS, SCHEMA, TREE, validate_model
from .preprocessing import reference_from_file
from .paths import resolve_paths
from .prepared import plan, resolve, cached_files

ROOT_DIR = Path(__file__).resolve().parents[1]


def describe(config):
    return {'method': 'analytic',
            'destination': str(Path(resolve_paths(config)['target']['predictions'])/method_directory(config, 'analytic')),
            'predictors': analytic.describe(config)['destination'],
            'predictor_inputs': 'per-eRx CM minus the application calibration reference; saved physical-unit coefficients and intercepts',
            'selection': config.target.selection, 'adc_min': config.target.dataset.adcmin,
            'adc_max': config.target.dataset.adcmax, 'tree': TREE, 'columns': COLUMNS}


def resolve_application(config):
    from training_inputs import load_inputs
    if config.target.selection not in {'full', 'trigtime'}:
        raise NotImplementedError('CM application supports full or trigtime application selections')
    directory, training = load_inputs(config)
    expected_model = analytic.definition(config, directory, training)
    model_dir = Path(analytic.describe(config)['destination'])
    stored = analytic.cached_predictors(model_dir, expected_model)
    if stored is None:
        raise FileNotFoundError(f'Run --steps analytic before CM-only application: {model_dir}')
    spec = analytic.cm_manifest(training)['feature_spec']
    modules = {}
    for job in plan(config):
        if 'apply' not in job.roles:
            continue
        prepared = resolve(config, job)
        prepared = select_modules(prepared, linear._selected(config.target.modules, prepared['modules']))
        if cached_files(prepared) is None:
            raise FileNotFoundError(f'Run --steps prepare for application inputs: {prepared["destination"]}')
        manifest = prepared_manifest(prepared)
        for name in linear._selected(config.target.modules, prepared['modules']):
            if name not in stored['modules'] or name not in spec['modules']:
                raise ValueError(f'No completed CM-only predictor for application module {name}')
            if 37*prepared['modules'][name]['identity'][2] != spec['modules'][name]['nch']:
                raise ValueError(f'CM application/model channel count mismatch: {name}')
            reference_from_file(config.target.calibration_file, name, spec['modules'][name]['nch'])
            if name not in modules:
                model, _ = load_module(model_dir, name)
                validate_model(model, spec, name)
                record = next(r for r in stored['outputs'] if r['file']==f'{name}/predictor.npz')
                modules[name] = {'model': model, 'inputs': [], 'definition': {
                    'nch': spec['modules'][name]['nch'], 'predictor': str(model_dir/name/'predictor.npz'),
                    'predictor_sha256': record['sha256'], 'fitted_channels': int(model['fitted'].sum())}}
            for source in prepared['files']:
                original = next(r for r in manifest['outputs'] if r['module']==name
                                and r['source_file_index']==source['index'])
                modules[name]['inputs'].append({
                    'run': job.source.run, 'source_file_index': source['index'], 'source_uuid': source['uuid'],
                    'input_file': str(prepared['destination']/original['file']),
                    'prepared_signature': module_signature(prepared, name), 'input_entries': source['entries'],
                    'input_size': original['size'], 'input_mtime_ns': original['mtime_ns'],
                    'output_file': str(Path(f'Run{job.source.run}')/Path(original['file']).name)})
    if not modules:
        raise ValueError('No CM-only application modules selected')
    for name, row in modules.items():
        files = [r['output_file'] for r in row['inputs']]
        if len(files)!=len(set(files)):
            raise ValueError(f'Application output filenames overlap for {name}')
        row['definition']['inputs'] = row['inputs']
    from .cm_application_provenance import code_hashes
    recipe = config.target
    definition = {'schema': SCHEMA, 'method': 'analytic',
        'target': {'campaign': recipe.campaign, 'runs': [s.run for s in recipe.dataset.sources],
                   'pedestal_run': recipe.pedestal.run, 'selection': recipe.selection,
                   'trigger_time_window': recipe.trigger_time_window,
                   'adc_min': recipe.dataset.adcmin, 'adc_max': recipe.dataset.adcmax},
        'model_definition_signature': signature(expected_model), 'feature_spec_sha256': signature(spec),
        'cm_convention': spec['cm_convention'],
        'modules': {name: row['definition'] for name, row in modules.items()},
        'tree': TREE, 'columns': COLUMNS,
        'invalid_channels': 'both vectors NaN for unavailable models or rejected/nonfinite target ADC',
        'vector_dtype': 'float32; corrected ADC = selected pedestal-subtracted ADC - stored prediction',
        'code_sha256': code_hashes(ROOT_DIR)}
    return {'destination': Path(describe(config)['destination']),
            'definition': json.loads(json.dumps(definition)), 'modules': modules, 'feature_spec': spec}


def run(config, resolved=None):
    """Publish modules using the existing linear evaluator and completion checks."""
    resolved = resolve_application(config) if resolved is None else resolved
    destination, definition = resolved['destination'], resolved['definition']
    existing = linear.cached_application(resolved)
    if existing is not None and existing['complete']:
        print(f'Reusing applied CM-only corrections: {destination}', flush=True)
        return
    records = dict(existing['module_records']) if existing is not None else {}
    if existing is None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
    linear._write_manifest(destination, linear._overview(resolved, records, complete=False))
    for module in definition['modules']:
        if module in records:
            print(f'Reusing applied CM-only module: {destination/module}', flush=True)
            continue
        work = Path(tempfile.mkdtemp(prefix=f'.{module}.work-', dir=destination))
        try:
            print(f'Applying analytic (CM only) to {module}', flush=True)
            outputs, summary = linear._compute_module(config, resolved, module, work)
            record = {'schema': SCHEMA, 'signature': signature(definition), 'definition': definition,
                      'module': module, 'complete': True, 'summary': summary, 'outputs': outputs}
            linear._write_manifest(work, record)
            linear._check_outputs(work, outputs, [r['output_file'] for r in definition['modules'][module]['inputs']])
            target = destination/module
            if target.exists():
                raise FileExistsError(f'Application module appeared during production: {target}')
            os.rename(work, target)
            records[module] = record
            print(f'Applied CM-only module ready: {target}', flush=True)
            linear._write_manifest(destination, linear._overview(resolved, records, complete=False))
        except Exception as exc:
            raise RuntimeError(f'{exc}. Completed modules remain in {destination}; unfinished work: {work}') from exc
    linear._write_manifest(destination, linear._overview(resolved, records, complete=True))
    print(f'Applied CM-only corrections ready: {destination}', flush=True)
