"""Prepare shared linear/DNN event membership and counts in one RDF pass."""
from __future__ import annotations
from utils.prepared import prepared_manifest, select_modules

import hashlib
import json
import os
from pathlib import Path
import tempfile

from utils.config import AnalysisConfig
from utils.nano import root_path
from utils.paths import resolve_paths
from utils.preprocessing import statistics_config
from utils.prepared import cached_files, plan, resolve
from utils.training import (SCHEMA, book_statistics, initialize_training_root,
                            make_feature_spec, preprocessing_payload, training_view)

ROOT_DIR = Path(__file__).resolve().parent


def describe(config: AnalysisConfig) -> dict:
    from utils.datasets import recipes
    return {'destination': resolve_paths(config)['dnn_training_inputs'],
            'derive_runs': ([{'campaign':row.campaign,'run':source.run} for row in recipes(config) for source in row.dataset.sources]
                            if config.derivations else [source.run for source in config.correction.dataset.sources]),
            'feature_groups': config.dnn['input_features'], 'max_channels': config.dnn['max_channels'],
            'linear_feature_groups': statistics_config(config).dnn['input_features'],
            'processing': 'one shared event-selection/counting pass per prepared run/file/module',
            'validation_split_fraction': config.dnn['validation_split_fraction'],
            'split_seed': config.dnn['split_seed'], 'independent_test_set': False,
            'train_event_fraction': config.dnn['train_event_fraction'],
            'validation_event_fraction': config.dnn['validation_event_fraction'],
            'stored_data': 'event identities, split/retention, target counts; shared measurements remain in prepared caches',
            'preprocessing_mode': config.dnn['preprocessing_mode'],
            'preprocessing': ('central calibration only; no fitted normalization' if config.dnn['preprocessing_mode']=='calibration_only'
                              else 'retained training events only; target-channel ADC masked before fitting'),
            'cell_area_file': str(config.dnn['cell_area_file'])}


def _code_hashes(config):
    return {name: _sha(ROOT_DIR/name) for name in ('training_inputs.py','utils/training.py',
            'utils/training_helpers.hxx','utils/module_registry.py','utils/preprocessing.py','utils/identity.py',
            'utils/datasets.py','utils/conditions.py')}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected(config, available):
    selection = config.correction.modules
    if not isinstance(selection, str):
        return [name for name in selection if name in available]
    return [name for name in available if selection == 'all'
            or selection == 'ld' and name.startswith('ML_')
            or selection == 'hd' and name.startswith('MH_')]


def resolve_inputs(config: AnalysisConfig, *, register_modules=False) -> dict:
    if config.derivations:
        return resolve_joint_inputs(config,register_modules=register_modules)
    if 'test' in config.correction.selection:
        raise ValueError('training_inputs requires derive_selection full or trigtime, not a held-out selection')
    if config.diagnostics['plot_dnninputs']:
        raise NotImplementedError('plot_dnninputs: true is not implemented yet; use false for input preparation')
    prepared, modules = [], {}
    for job in plan(config):
        if 'derive' not in job.roles:
            continue
        resolved = resolve(config, job)
        names = _selected(config, resolved['modules'])
        if not names:
            continue
        resolved = select_modules(resolved, names)
        if cached_files(resolved) is None:
            raise FileNotFoundError(f'Run prepare first: {resolved["destination"]}')
        for name in names:
            metadata = resolved['modules'][name]
            if name in modules and metadata['identity'] != modules[name]['identity']:
                raise ValueError(f'Training module identity changes between runs: {name}')
            modules[name] = metadata
        manifest = prepared_manifest(resolved)
        prepared.append({'resolved': resolved, 'names': names, 'outputs': manifest['outputs']})
    if not prepared or not modules:
        raise ValueError('No derivation modules are available for training inputs')
    if register_modules and config.dnn['preprocessing_mode']=='calibration_only':
        from utils.module_registry import ensure_modules
        ensure_modules(sorted(modules), path=str(config.dnn['module_registry_file']))
    spec = make_feature_spec(config, modules)
    # Features differ between estimators, but membership and target counts do not.
    # Record both contracts without a second event loop or duplicate ROOT indexes.
    linear_spec = make_feature_spec(statistics_config(config), modules)
    recipe = config.correction
    options = {name: config.dnn[name] for name in ('split_seed', 'validation_split_fraction',
               'train_event_fraction', 'validation_event_fraction', 'preprocess_inputs',
               'preprocess_targets')}
    definition = {'schema': SCHEMA, 'feature_spec': spec,
                  'linear_feature_spec': linear_spec, 'options': options,
                  'campaign': recipe.campaign, 'selection': recipe.selection,
                  'split_identity': spec['split_identity'],
                  'trigger_time_window': recipe.trigger_time_window,
                  'adc_min': recipe.dataset.adcmin, 'adc_max': recipe.dataset.adcmax,
                  'prepared': [{'run': row['resolved']['job'].source.run,
                                'directory': str(row['resolved']['destination']),
                                'signature': row['resolved']['signature'],
                                'preparation_definition': row['resolved']['definition'],
                                'files': row['resolved']['files'], 'modules': row['names'],
                                'outputs': row['outputs']} for row in prepared],
                  'cell_area_sha256': _sha(config.dnn['cell_area_file']),
                  'code_sha256': _code_hashes(config)}
    # Canonical JSON also normalizes tuples before later equality checks.
    definition = json.loads(json.dumps(definition))
    signature = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'destination': Path(resolve_paths(config)['dnn_training_inputs']),
            'signature': signature, 'definition': definition, 'prepared': prepared, 'feature_spec': spec}


def index_path(prepared, item):
    prefix=(Path(f'dataset{prepared["dataset_index"]}_{prepared["campaign"]}')
            if 'dataset_index' in prepared else Path())
    return prefix/f'Run{prepared["run"]}'/item['module']/Path(item['file']).name


def resolve_joint_inputs(config, *, register_modules=False):
    from utils.datasets import recipes,single,stream_table
    from utils.identity import digest
    rows=[resolve_inputs(single(config,r),register_modules=register_modules) for r in recipes(config)]
    modules={};prepared=[];definitions=[];seen=set()
    for index,(recipe,row) in enumerate(zip(recipes(config),rows)):
        for name,meta in row['feature_spec']['modules'].items():
            if name in modules and modules[name]!=meta:
                raise ValueError(f'Joint datasets disagree on physical geometry/module ID: {name}')
            modules[name]=meta
        for base,definition in zip(row['prepared'],row['definition']['prepared']):
            for source in definition['files']:
                if source['uuid'] in seen:
                    raise ValueError('Duplicate physical Nano UUID across derivation datasets: '+source['uuid'])
                seen.add(source['uuid'])
            prepared.append(dict(base,dataset_index=index))
            definitions.append(dict(definition,dataset_index=index,campaign=recipe.campaign))
    spec=dict(rows[0]['feature_spec'],modules=dict(sorted(modules.items())))
    linear_modules={name:dict(meta,module_id=i+1) for i,(name,meta) in enumerate(sorted(modules.items()))}
    linear_spec=dict(rows[0]['definition']['linear_feature_spec'],modules=linear_modules)
    for row in rows[1:]:
        if {k:v for k,v in row['feature_spec'].items() if k!='modules'}!={k:v for k,v in spec.items() if k!='modules'}:
            raise ValueError('Joint datasets have incompatible feature contracts')
    table=stream_table(definitions)
    for row in definitions:
        row['stream_id']=next(int(k) for k,v in table.items()
                              if v=={'dataset_index':row['dataset_index'],'run':row['run']})
    definition={'schema':SCHEMA,'feature_spec':spec,'linear_feature_spec':linear_spec,
        'options':rows[0]['definition']['options'],'campaign':'joint','selection':'per_dataset',
        'trigger_time_window':None,'adc_min':None,'adc_max':None,'split_identity':spec['split_identity'],
        'datasets':[{k:r['definition'][k] for k in ('campaign','selection','trigger_time_window','adc_min','adc_max')}
                    | {'modules':list(r['feature_spec']['modules'])} for r in rows],
        'stream_table':table,'prepared':definitions,'cell_area_sha256':rows[0]['definition']['cell_area_sha256'],
        'code_sha256':_code_hashes(config)}
    return {'destination':Path(resolve_paths(config)['dnn_training_inputs']),'signature':digest(definition),
            'definition':definition,'prepared':prepared,'feature_spec':spec}


def cached_inputs(resolved: dict) -> dict | None:
    directory = resolved['destination']
    if not directory.exists():
        return None
    path = directory / 'manifest.json'
    if not path.is_file():
        raise ValueError(f'Incomplete training-input products: {directory}')
    manifest = json.loads(path.read_text())
    if manifest.get('signature') != resolved['signature']:
        raise ValueError(f'Training-input provenance mismatch: {directory}. Preserve the old directory '
                         'under another name or choose a separate output_dir before producing a replacement.')
    if manifest.get('feature_spec') != resolved['feature_spec']:
        raise ValueError(f'Training-input feature definition changed: {directory}')
    if manifest.get('definition') != resolved['definition']:
        raise ValueError(f'Training-input manifest definition changed: {directory}')
    expected = ['preprocessing.json'] + [
        str(index_path(row,item))
        for row in resolved['definition']['prepared'] for item in row['outputs']
        if item['module'] in row['modules']]
    outputs = manifest.get('outputs', [])
    if [row['file'] for row in outputs] != expected:
        raise ValueError(f'Incomplete training-input output inventory: {directory}')
    for row in outputs:
        relative = Path(row['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid training output path: {relative}')
        output = directory / relative
        if not output.is_file():
            raise ValueError(f'Missing training-input output: {output}')
        stat = output.stat()
        if stat.st_size != row['size'] or stat.st_mtime_ns != row['mtime_ns']:
            raise ValueError(f'Modified training-input output: {output}')
    if _sha(directory / 'preprocessing.json') != manifest.get('preprocessing_sha256'):
        raise ValueError(f'Modified preprocessing constants: {directory}')
    return manifest


def load_inputs(config: AnalysisConfig, *, for_dnn=False) -> tuple[Path, dict]:
    """Validate one shared product and expose the estimator's feature contract.

    The on-disk manifest and its signature always describe the shared product.
    Linear consumers only replace the in-memory feature view; saved membership,
    counts and provenance remain the same as for the DNN.
    """
    resolved = resolve_inputs(config)
    manifest = cached_inputs(resolved)
    if manifest is None:
        raise FileNotFoundError('Run --steps training_inputs before consuming training products')
    if not for_dnn:
        manifest = dict(manifest, feature_spec=manifest['definition']['linear_feature_spec'])
    return resolved['destination'], manifest


def _moments_to_counts(stats, width):
    return {label: [int(stats.channel_counts[index*width+c]) for c in range(width)]
            for index, label in enumerate(('train', 'validation'))}


def _produce(config, resolved):
    ROOT = initialize_training_root()
    destination, spec = resolved['destination'], resolved['feature_spec']
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.work-', dir=destination.parent))
    global_stats, summary, counts_by_module_run, outputs = None, [], {}, []
    print(f'Preparing training inputs -> {destination}', flush=True)
    try:
        for row in resolved['prepared']:
            base = row['resolved']
            run = base['job'].source.run
            from utils.datasets import source_config
            current=source_config(config,row)
            defined=next(d for d in resolved['definition']['prepared'] if d['directory']==str(base['destination']))
            stream=defined.get('stream_id',run)
            for info in base['files']:
                for name in row['names']:
                    original = next(item for item in row['outputs'] if item['module'] == name
                                    and item['source_file_index'] == info['index'])
                    rdf = ROOT.RDataFrame('Events', root_path(str(base['destination'] / original['file'])))
                    ROOT.RDF.Experimental.AddProgressBar(ROOT.RDF.AsRNode(rdf))
                    input_count = rdf.Count()
                    node = training_view(rdf, current, spec, name, dict(info, run=run), statistics_only=True)
                    stats_result = book_statistics(node, config, spec, name)
                    relative = index_path(defined,original)
                    target = work / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    options = ROOT.RDF.RSnapshotOptions()
                    options.fLazy = True
                    options.fAutoFlush = config.resources['events_per_chunk']
                    index = node.Snapshot('TrainingEvents', str(target),
                        ['source_run', 'source_file_index', 'source_entry', 'event_id', 'event',
                         'training_split', 'training_retained', 'training_n_targets'], options)
                    # Snapshot, accounting and preprocessing share one RDF loop.
                    index.GetValue()
                    stats = ROOT.hgcal_cm_training.Statistics(stats_result.GetValue())
                    if int(input_count.GetValue()) != info['entries']:
                        raise ValueError(f'Prepared input event count changed: Run{run}/{name}')
                    if global_stats is None:
                        global_stats = ROOT.hgcal_cm_training.Statistics(stats)
                    else:
                        global_stats.merge(stats)
                    counts = _moments_to_counts(stats, spec['max_channels'])
                    by_run = counts_by_module_run.setdefault(name, {})
                    if str(stream) not in by_run:
                        by_run[str(stream)] = counts
                    else:
                        for label in counts:
                            by_run[str(stream)][label] = [a+b for a,b in zip(by_run[str(stream)][label], counts[label])]
                    summary.append({'run': run, 'dataset_index':row.get('dataset_index',0),
                                    'module': name, 'source_file_index': info['index'],
                                    'input_events': info['entries'],
                                    'eligible_events': [int(x) for x in stats.events],
                                    'retained_events': [int(x) for x in stats.kept_events],
                                    'retained_targets': [int(x) for x in stats.samples]})
                    outputs.append({'file': str(relative), 'entries': sum(int(x) for x in stats.events)})
                    print(f'Run{run}/{name}/file{info["index"]}: '
                          f'train {int(stats.kept_events[0])} events/{int(stats.samples[0])} targets; '
                          f'validation {int(stats.kept_events[1])} events/{int(stats.samples[1])} targets', flush=True)
        if global_stats is None or any(int(x) == 0 for x in global_stats.samples):
            raise ValueError('Training inputs require retained finite targets in both train and validation partitions')
        for name, by_run in counts_by_module_run.items():
            if not sum(sum(row['train']) for row in by_run.values()):
                raise ValueError(f'No retained training targets for module {name}')
        preprocessing = preprocessing_payload(global_stats, spec, config)
        (work / 'preprocessing.json').write_text(json.dumps(preprocessing, indent=2, allow_nan=False) + '\n')
        outputs.insert(0, {'file': 'preprocessing.json'})
        for item in outputs:
            stat = (work / item['file']).stat()
            item.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        scales = {}
        for name, by_run in counts_by_module_run.items():
            scales[name] = {}
            for label in ('train', 'validation'):
                values = [count for row in by_run.values() for count in row[label] if count > 0]
                scales[name][label] = sum(values)/len(values) if values else 1.
        manifest = {'signature': resolved['signature'], 'definition': resolved['definition'],
                    'feature_spec': spec, 'preprocessing_sha256': _sha(work / 'preprocessing.json'),
                    'counts_by_module_run': counts_by_module_run, 'inverse_frequency_scales': scales,
                    'summary': summary, 'outputs': outputs}
        if 'stream_table' in resolved['definition']:
            manifest['stream_table']=resolved['definition']['stream_table']
        (work / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if destination.exists():
            raise FileExistsError(f'Training-input destination appeared during production: {destination}')
        os.rename(work, destination)
    except Exception as exc:
        raise RuntimeError(f'{exc}. Retained training work directory: {work}') from exc
    print(f'Training inputs ready: {destination}', flush=True)


def run(config: AnalysisConfig) -> None:
    ROOT = initialize_training_root()
    if ROOT.IsImplicitMTEnabled():
        ROOT.DisableImplicitMT()
    ROOT.EnableImplicitMT(config.resources['cpus'])
    from utils.datasets import recipes
    print(f'Checking training-input provenance for {sum(len(r.dataset.sources) for r in recipes(config))} derivation runs', flush=True)
    resolved = resolve_inputs(config, register_modules=True)
    if cached_inputs(resolved) is not None:
        print(f'Reusing training inputs: {resolved["destination"]}', flush=True)
    else:
        _produce(config, resolved)
