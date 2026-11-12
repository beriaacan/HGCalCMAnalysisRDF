"""Plan, validate and reopen provenance-validated prepared ROOT datasets."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .calibrations import validate_calibration_run
from .config import AnalysisConfig, DatasetConfig, SourceRun
from .nano import REQUIRED_NANO_COLUMNS, read_module_metadata, root_path
from .paths import discover_files

SCHEMA = 'hgcal_cm_prepared_v2'
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PreparationJob:
    campaign: str
    source: SourceRun
    pedestal_run: int
    calibration_file: Path
    module_map: Path
    selections: tuple[str | tuple[str, ...], ...]
    roles: tuple[str, ...]
    root: Path
    variant: str
    conditions: dict | None = None


@lru_cache(maxsize=128)
def _metadata_modules(filename, size, mtime_ns):
    """Only Runs metadata; cache invalidates when source file identity changes."""
    return tuple(read_module_metadata(filename)) if _file_info(filename)['entries'] else ()


def derivation_modules(config):
    """Exact derivation inventory for apply_modules: all, never a target selector."""
    if config.derivations:
        from .datasets import recipes, single
        return tuple(sorted({name for row in recipes(config) for name in derivation_modules(single(config,row))}))
    selection = config.correction.modules
    if not isinstance(selection, str):
        return selection
    names=set()
    for filename in discover_files(config.correction.dataset):
        stat=Path(filename).stat()
        names.update(name for name in _metadata_modules(filename,stat.st_size,stat.st_mtime_ns)
                     if selection=='all' or selection=='ld' and name.startswith('ML_')
                     or selection=='hd' and name.startswith('MH_'))
    if not names:
        raise ValueError('derive_modules selects no modules in the derivation Nano metadata')
    return tuple(sorted(names))


def plan(config: AnalysisConfig) -> tuple[PreparationJob, ...]:
    grouped: dict[tuple, list[tuple[str, DatasetConfig]]] = {}
    target=config.target
    if target.modules=='all':
        target=replace(target,modules=derivation_modules(config))
    from .datasets import recipes
    for role, recipe in [*(('derive',r) for r in recipes(config)), ('apply', target)]:
        for source in recipe.dataset.sources:
            key = (recipe.campaign, source, recipe.pedestal.run,
                   recipe.calibration_file, recipe.module_map, json.dumps(recipe.conditions,sort_keys=True))
            grouped.setdefault(key, []).append((role, recipe))
    jobs = tuple(PreparationJob(campaign, source, pedestal, calibration, locator,
                                tuple(dict.fromkeys(recipe.modules for _, recipe in entries)),
                                tuple(dict.fromkeys(role for role, _ in entries)),
                                config.output_dir / campaign / 'prepared' / f'Run{source.run}',
                                _variant(config, campaign, source, pedestal, calibration, locator,json.loads(conditions)),
                                json.loads(conditions))
                 for (campaign, source, pedestal, calibration, locator,conditions), entries in grouped.items())
    destinations = [cache_directory(job) for job in jobs]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Different preparation recipes target the same readable cache directory. '
                         'Use separate output_dir values for these variants.')
    return jobs


def _variant(config, campaign, source, pedestal, calibration, locator, conditions=None):
    from .identity import preparation_recipe, variant
    return variant(preparation_recipe(source, campaign, pedestal, calibration, locator,
                                      config.preparation['activity_thresholds'],conditions,config.dnn['cell_area_file']))


def cache_directory(job: PreparationJob) -> Path:
    label=(f'conditions_{job.conditions["era"].replace("/", "")}_Run{job.conditions["run"]}'
           if job.conditions else f'pedestals_{job.pedestal_run}')
    return job.root / label / job.variant



def describe(config: AnalysisConfig) -> list[dict]:
    return [{'run': job.source.run, 'roles': job.roles,
             'input_pattern': job.source.input_pattern, 'cache_root': str(job.root),
             'cache_directory': str(cache_directory(job)),
             'calibration_file': str(job.calibration_file), 'modules': job.selections,
             'analysis_cuts_applied': False} for job in plan(config)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locator(path: Path) -> dict[str, dict[str, str]]:
    lines = [line.split() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    if not lines or not {'typecode', 'fedid', 'econdidx'}.issubset(lines[0]):
        raise ValueError(f'Invalid module locator: {path}')
    output = {}
    for values in lines[1:]:
        if len(values) != len(lines[0]):
            raise ValueError(f'Malformed module locator row: {path}')
        row = dict(zip(lines[0], values))
        name = row['typecode'].replace('-', '_')
        if name in output:
            raise ValueError(f'Duplicate module {name} in {path}')
        output[name] = row
    return output


def _file_info(filename: str) -> dict:
    from .prepare_rdf import initialize_root
    ROOT = initialize_root()
    handle = ROOT.TFile.Open(root_path(filename), 'READ')
    if not handle or handle.IsZombie():
        raise OSError(f'Cannot open Nano file: {filename}')
    try:
        tree = handle.Get('Events')
        if not tree:
            raise ValueError(f'Missing Events tree: {filename}')
        branches = {str(branch.GetName()) for branch in tree.GetListOfBranches()}
        entries = int(tree.GetEntries())
        required = set(REQUIRED_NANO_COLUMNS)
        if entries and required - branches:
            raise ValueError(f'Missing Nano branches in {filename}: {sorted(required - branches)}')
        stat = Path(filename).stat()
        return {'path': filename, 'uuid': str(handle.GetUUID().AsString()),
                'entries': entries, 'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns,
                'has_trigger_valid': 'HGCMetaData_trigValid' in branches}
    finally:
        handle.Close()


def resolve(config: AnalysisConfig, job: PreparationJob) -> dict[str, Any]:
    """Read only headers and constants, never event data, to identify this cache."""
    if job.conditions:
        from .conditions import validate
        validate(job.conditions,job.calibration_file,job.module_map)
    else:
        validate_calibration_run(job.calibration_file, job.pedestal_run)
    locator = _locator(job.module_map)
    calibration = json.loads(job.calibration_file.read_text())
    return _resolve_files(config, job, locator, calibration)


def _resolve_files(config, job, locator, calibration) -> dict:
    from .config import Dataset
    files = discover_files(Dataset(job.source.run, (job.source,), None, None))
    info = []
    offset = 0
    uuids = set()
    for index, filename in enumerate(files):
        item = _file_info(filename)
        if item['uuid'] in uuids:
            raise ValueError(f'Duplicate Nano UUID in preparation inputs: {filename}')
        uuids.add(item['uuid'])
        item.update(index=index, entry_offset=offset)
        offset += item['entries']
        info.append(item)
    populated = [item for item in info if item['entries']]
    if not populated:
        raise ValueError(f'No events in any Nano file for Run{job.source.run}')
    available = read_module_metadata(populated[0]['path'])
    modules = {}
    cell_areas=json.loads(config.dnn['cell_area_file'].read_text())
    for selection in job.selections:
        if isinstance(selection, str):
            names = [name for name in available if selection == 'all'
                     or selection == 'ld' and name.startswith('ML_')
                     or selection == 'hd' and name.startswith('MH_')]
        else:
            names = list(selection)
        if not names:
            raise ValueError(f'Empty module selection {selection} for Run{job.source.run}')
        for name in names:
            if name not in available or name not in locator:
                raise ValueError(f'Module {name} must occur in both Runs and module_map')
            identity = available[name]
            row = locator[name]
            if identity[:2] != (int(row['fedid']), int(row['econdidx'])):
                raise ValueError(f'Runs/module_map identity mismatch for {name}: {identity[:2]}')
            key = name.replace('_', '-')
            if key not in calibration:
                raise ValueError(f'Calibration lacks module {name}')
            constants = calibration[key]
            nch = 37 * identity[2]
            if constants.get('Channel') != list(range(nch)):
                raise ValueError(f'Calibration channel order/length mismatch for {name}')
            for field in ('ADC_ped', 'CM_ped', 'Noise', 'Valid'):
                values = constants.get(field)
                if (not isinstance(values, list) or len(values) != nch
                        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
                    raise ValueError(f'Invalid calibration {field} for {name}')
            if any(value < 0 for value in constants['Noise']) or any(value not in (0, 1) for value in constants['Valid']):
                raise ValueError(f'Invalid noise or validity constants for {name}')
            modules[name] = {'identity': identity, 'locator': row,
                             'calibration': {field: constants[field]
                                             for field in ('ADC_ped', 'CM_ped', 'Noise', 'Valid')}}
            from .preprocessing import absent_erx,cm_reference
            areas=cell_areas.get(name[:4],{}).get('SF')
            modules[name]['absent_erx']=absent_erx(constants,nch,name,areas)
            cm_reference(constants,nch,name,areas)  # Reject present but uncalibrated hardware early.
    for item in populated[1:]:
        other = read_module_metadata(item['path'])
        for name, module in modules.items():
            if other.get(name) != module['identity']:
                raise ValueError(f'Module identity changes for {name} in {item["path"]}')
    central = FRAMEWORK_ROOT.parent.parent / 'LocalCalibration/python/DigiAnalysisUtils.py'
    code = {name: _sha(FRAMEWORK_ROOT / name) for name in
            ('prepare.py', 'utils/prepared.py', 'utils/prepare_rdf.py',
             'utils/prepare_helpers.hxx', 'utils/nano.py','utils/preprocessing.py','utils/conditions.py','utils/datasets.py')}
    code['central/DigiAnalysisUtils.py'] = _sha(central)
    definition = {'schema': SCHEMA, 'campaign': job.campaign, 'run': job.source.run,
                  'pedestal_run': job.pedestal_run, 'files': info,
                  'module_order': list(modules),
                  'modules': {name: {'identity': row['identity'], 'locator': row['locator']}
                              for name, row in modules.items()},
                  'calibration_sha256': _sha(job.calibration_file),
                  'module_map_sha256': _sha(job.module_map),
                  'activity_thresholds': config.preparation['activity_thresholds'], 'code_sha256': code,
                  'cell_area_sha256':_sha(config.dnn['cell_area_file'])}
    if job.conditions:
        definition['conditions']=job.conditions
    encoded = json.dumps(definition, sort_keys=True, separators=(',', ':'))
    signature = hashlib.sha256(encoded.encode()).hexdigest()
    return {'job': job, 'definition': definition, 'signature': signature,
            'destination': cache_directory(job), 'modules': modules, 'files': info}


def select_modules(resolved, names):
    """A role-specific view of immutable prepared modules, independent of other roles."""
    from .analytic_artifacts import signature
    names = list(names)
    if not names or len(set(names)) != len(names) or set(names)-resolved['modules'].keys():
        raise ValueError('Invalid prepared module view')
    definition = dict(resolved['definition'], module_order=names,
                      modules={name: resolved['definition']['modules'][name] for name in names})
    return dict(resolved, modules={name: resolved['modules'][name] for name in names},
                definition=definition, signature=signature(definition))


def common_definition(resolved):
    return {key: value for key, value in resolved['definition'].items()
            if key not in {'modules', 'module_order'}}


def module_definition(resolved, module):
    value = dict(common_definition(resolved), module_order=[module],
                 modules={module: resolved['definition']['modules'][module]})
    return json.loads(json.dumps(value))


def module_signature(resolved, module):
    from .analytic_artifacts import signature
    return signature(module_definition(resolved, module))


def module_record(resolved, module):
    """Independent completion records make module additions/subsets harmless."""
    from .analytic_artifacts import signature
    directory = resolved['destination']/module
    if not directory.exists():
        return None
    path = directory/'manifest.json'
    if not path.is_file():
        raise ValueError(f'Incomplete prepared module: {directory}')
    record = json.loads(path.read_text())
    expected = module_definition(resolved, module)
    if (record.get('schema') != SCHEMA or record.get('complete') is not True
            or record.get('definition') != expected or record.get('signature') != signature(expected)):
        raise ValueError(f'Prepared module provenance mismatch: {directory}')
    outputs = record.get('outputs', [])
    expected_files = [str(Path(module)/Path(source['path']).name) for source in resolved['files']]
    if [row['file'] for row in outputs] != expected_files:
        raise ValueError(f'Incomplete prepared module file inventory: {directory}')
    for row, source in zip(outputs, resolved['files']):
        relative = Path(row['file'])
        if (relative.is_absolute() or '..' in relative.parts or row['module'] != module
                or row['source_file_index'] != source['index'] or row['entries'] != source['entries']):
            raise ValueError(f'Invalid prepared output metadata: {directory}')
        file = resolved['destination']/relative
        if not file.is_file():
            raise ValueError(f'Missing prepared file: {file}')
        stat = file.stat()
        if stat.st_size != row['size'] or stat.st_mtime_ns != row['mtime_ns']:
            raise ValueError(f'Modified prepared file: {file}')
    return record


def prepared_manifest(resolved):
    """Synthesize a stable selected view; no mutable global inventory enters IDs."""
    from .analytic_artifacts import signature
    directory = resolved['destination']
    if not directory.exists():
        return None
    path = directory/'manifest.json'
    if not path.is_file():
        raise ValueError(f'Incomplete prepared cache: {directory}')
    record = json.loads(path.read_text())
    expected = json.loads(json.dumps(common_definition(resolved)))
    if (record.get('schema') != SCHEMA or record.get('definition') != expected
            or record.get('signature') != signature(expected)):
        raise ValueError(f'Prepared cache provenance mismatch: {directory}')
    records = {module: module_record(resolved, module) for module in resolved['modules']}
    if any(row is None for row in records.values()):
        return None
    outputs = [next(row for row in records[module]['outputs'] if row['source_file_index']==source['index'])
               for source in resolved['files'] for module in resolved['modules']]
    return {'schema': SCHEMA, 'complete': True, 'signature': resolved['signature'],
            'definition': json.loads(json.dumps(resolved['definition'])), 'outputs': outputs,
            'module_calibrations': {name: row['calibration'] for name, row in resolved['modules'].items()}}


def cached_files(resolved):
    manifest = prepared_manifest(resolved)
    return None if manifest is None else [resolved['destination']/row['file'] for row in manifest['outputs']]


def open_module(config: AnalysisConfig, role: str, module: str, *, apply_selection: bool = True):
    """Reusable downstream entry point; missing caches require an explicit prepare."""
    from .prepare_rdf import analysis_view, initialize_root
    if role not in {'derive', 'apply'}:
        raise ValueError('role must be derive or apply')
    recipe = config.correction if role == 'derive' else config.target
    selection = recipe.modules
    if ((not isinstance(selection, str) and module not in selection)
            or selection == 'ld' and not module.startswith('ML_')
            or selection == 'hd' and not module.startswith('MH_')):
        raise ValueError(f'Module {module} is not selected for {role}')
    files = []
    for job in plan(config):
        if role not in job.roles:
            continue
        resolved = resolve(config, job)
        available = cached_files(resolved)
        if available is None:
            raise FileNotFoundError(f'Run prepare first for {resolved["destination"]}')
        if module not in resolved['modules']:
            raise ValueError(f'Module {module} was not selected for Run{job.source.run}')
        files.extend(str(path) for path in available if path.parent.name == module)
    ROOT = initialize_root()
    rdf = ROOT.RDataFrame('Events', files)
    recipe = config.correction if role == 'derive' else config.target
    return analysis_view(rdf, recipe) if apply_selection else rdf
