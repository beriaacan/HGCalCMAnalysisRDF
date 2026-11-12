"""Compute, publish and reuse independent per-module linear regressions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from utils.analytic_artifacts import (MODULE_FILES, PUBLICATION, check_definition,
                                      check_outputs, checked_module, sha, signature)
from utils.config import AnalysisConfig
from utils.paths import resolve_paths

ROOT_DIR = Path(__file__).resolve().parent


def describe(config: AnalysisConfig) -> dict:
    return {'destination': str(Path(resolve_paths(config)['analytic_predictors']) / 'allinputs'),
            'training_inputs': resolve_paths(config)['training_inputs'],
            'fit': 'separate module and target; all moments use the same accepted retained training events',
            'predictors': 'configured varying inputs; omit module ID, constants, padding and own-channel ADC',
            'input_adc_bounds': 'none; configured ADC bounds select targets only',
            'analytic_rcond': config.diagnostics['analytic_rcond'],
            'sample_weighting': 'unweighted ordinary least squares; sample_weighting is a DNN loss option',
            'validation': 'not evaluated during derivation; later explicit evaluation only',
            'artifacts': 'per-module predictor.npz, diagnostics.json and manifest.json; overall completion manifest',
            'publication': 'publish and reuse each module independently after coefficients and fit diagnostics are complete'}


def _definition(config, directory, manifest):
    from utils.analytic import SCHEMA
    calculation = (ROOT_DIR / 'utils/analytic.py').read_text().split('def load_module(', 1)[0]
    from utils.model_compatibility import population_request
    return {'schema': SCHEMA, 'training_inputs': str(directory),
            'application_population_request': population_request(config),
            'training_signature': manifest['signature'],
            'training_manifest_sha256': sha(directory / 'manifest.json'),
            'preprocessing_sha256': manifest['preprocessing_sha256'],
            'feature_spec': manifest['feature_spec'],
            'analytic_rcond': config.diagnostics['analytic_rcond'],
            'estimator': 'unweighted target-conditional centered/unit-variance OLS; eigenvalue pseudoinverse',
            'coefficient_units': 'physical feature values, including calibration-centered CM, to pedestal-subtracted target ADC',
            'unavailable_policy': 'fewer than 2 targets: fitted=false, intercept=NaN; never substitute another fit',
            'code_sha256': {
                'utils/analytic_fit.py': sha(ROOT_DIR / 'utils/analytic_fit.py'),
                'utils/analytic.py:calculation': hashlib.sha256(calculation.encode()).hexdigest(),
                'utils/analytic_helpers.hxx': sha(ROOT_DIR / 'utils/analytic_helpers.hxx')}}


def _inventory(records: dict, names: list[str]) -> tuple[dict, list]:
    summaries = {name: records[name]['summary'] for name in names if name in records}
    outputs = [dict(row, file=f'{name}/{row["file"]}') for name in names if name in records
               for row in records[name]['outputs']]
    return summaries, outputs


def cached_predictors(destination: Path, definition: dict) -> dict | None:
    """Check all published modules, returning partial progress or full completion."""
    if not destination.exists():
        return None
    path = destination / 'manifest.json'
    if not path.is_file():
        raise ValueError(f'Incomplete analytic predictors: {destination}')
    manifest = json.loads(path.read_text())
    check_definition(manifest, path, definition)
    names = list(definition['feature_spec']['modules'])
    if manifest.get('publication') == PUBLICATION:
        if type(manifest.get('complete')) is not bool:
            raise ValueError(f'Invalid analytic completion status: {path}')
        records = {}
        for name in names:
            record = checked_module(destination, name, definition)
            if record is not None:
                records[name] = record
        if set(manifest.get('modules', {})) - records.keys():
            raise ValueError(f'Missing completed analytic module: {destination}')
        summaries, outputs = _inventory(records, names)
        if manifest['complete'] and (list(summaries) != names
                or manifest.get('modules') != summaries or manifest.get('outputs') != outputs):
            raise ValueError(f'Incomplete analytic predictor inventory: {destination}')
        # The module rename may have succeeded immediately before interruption,
        # leaving the overall progress record one module behind. Modules decide.
        return dict(manifest, modules=summaries, outputs=outputs, module_records=records)
    raise ValueError(f'Unsupported analytic publication format: {path}; regenerate predictors')


def _write_manifest(directory: Path, record: dict) -> None:
    """Replace only the progress record, never a module's coefficient files."""
    descriptor, name = tempfile.mkstemp(prefix='.manifest-', suffix='.tmp', dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as output:
            output.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, directory / 'manifest.json')
    finally:
        temporary.unlink(missing_ok=True)


def _overview(definition, records, *, complete):
    from utils.analytic import SCHEMA
    names = list(definition['feature_spec']['modules'])
    summaries, outputs = _inventory(records, names)
    return {'schema': SCHEMA, 'signature': signature(definition), 'definition': definition,
            'publication': PUBLICATION, 'complete': complete, 'modules': summaries,
            'outputs': outputs, 'runtime': {name: records[name]['runtime'] for name in names if name in records},
            'publication_code_sha256': {name: sha(ROOT_DIR / name) for name in
                                        ('analytic_allinputs.py', 'utils/analytic_artifacts.py')}}


def _produce(config, directory, manifest, definition, destination, *, existing=None):
    from utils.analytic_fit import compute_module
    if existing is None:
        existing = cached_predictors(destination, definition)
    if existing is not None and existing.get('complete', True):
        print(f'Reusing analytic predictors: {destination}', flush=True)
        return
    records = dict(existing['module_records']) if existing is not None else {}
    if existing is None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Refuse a destination created by another invocation after preflight.
        destination.mkdir()
    _write_manifest(destination, _overview(definition, records, complete=False))
    for module in manifest['feature_spec']['modules']:
        if module in records:
            print(f'Reusing analytic module: {destination / module}', flush=True)
            continue
        work = Path(tempfile.mkdtemp(prefix=f'.{module}.work-', dir=destination))
        try:
            summary, runtime = compute_module(config, manifest, module, work)
            outputs = [{'file': name, 'sha256': sha(work / name)} for name in MODULE_FILES]
            record = {'schema': definition['schema'], 'signature': signature(definition),
                      'definition': definition, 'publication': PUBLICATION, 'complete': True,
                      'module': module, 'summary': summary, 'outputs': outputs, 'runtime': runtime}
            _write_manifest(work, record)
            check_outputs(work, outputs, list(MODULE_FILES))
            target = destination / module
            if target.exists():
                raise FileExistsError(f'Analytic module appeared during production: {target}')
            os.rename(work, target)
            records[module] = record
            print(f'Analytic module ready: {target}', flush=True)
            _write_manifest(destination, _overview(definition, records, complete=False))
        except Exception as exc:
            raise RuntimeError(f'{exc}. Completed modules remain in {destination}; '
                               f'unfinished work, if present: {work}') from exc
    _write_manifest(destination, _overview(definition, records, complete=True))
    print(f'Analytic predictors ready: {destination}', flush=True)


def run(config: AnalysisConfig) -> None:
    from training_inputs import load_inputs
    print('Checking training-input provenance for analytic_allinputs', flush=True)
    directory, manifest = load_inputs(config)
    definition = _definition(config, directory, manifest)
    destination = Path(describe(config)['destination'])
    existing = cached_predictors(destination, definition)
    if existing is not None and existing.get('complete', True):
        print(f'Reusing analytic predictors: {destination}', flush=True)
        return
    _produce(config, directory, manifest, definition, destination, existing=existing)
