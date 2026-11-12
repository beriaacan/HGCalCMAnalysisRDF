"""CPU/CUDA DNN application; shared prepared inputs and compact ROOT outputs."""
from __future__ import annotations

from dataclasses import replace
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile

from .analytic_artifacts import sha, signature
from .application import SCHEMA, TREE, COLUMNS
from .inference_config import method_directory
from .paths import resolve_paths

ROOT_DIR = Path(__file__).resolve().parents[1]
CODE = ('utils/dnn_application.py', 'utils/inference_inputs.py', 'utils/inference_predictor.py',
        'utils/inference_io.hxx', 'utils/inference_runtime.py', 'utils/inference_config.py',
        'utils/dnn_tensors.py', 'utils/training.py', 'utils/training_helpers.hxx',
        'utils/prepare_rdf.py', 'utils/prepare_helpers.hxx', 'utils/dnn_prefetch.py',
        'utils/dnn_training.py', 'utils/training_metadata.py', 'apply.py', 'utils/preprocessing.py', 'utils/module_registry.py', 'utils/dnn_internal_normalization.py')


def application_code_hashes(config):
    from .application_provenance import current_hashes
    return current_hashes(ROOT_DIR,CODE)


def describe(config):
    from .dnn_training import destination
    from .model_compatibility import find
    group = 'combined' if config.training['training_grouping'] == 'combined' else '<module>'
    selected=find(config,group,required=False) if group!='<module>' else None
    return {'destination': str(Path(resolve_paths(config)['target']['predictions']) / method_directory(config, 'dnn')),
            'model': str(selected or destination(config,group)),
            'resources': config.application, 'infer_batch': config.dnn['infer_batch'],
            'checkpoint': 'best model.pt; saved feature ordering and normalization',
            'selection': 'application event/channel selections only; no split, retention or training weights',
            'unknown_modules': config.dnn['allow_unknown_modules']}


def application_spec(config, saved, metadata):
    """Preserve training IDs; explicitly map permitted unseen modules to ID 0."""
    import math
    spec = copy.deepcopy(saved)
    areas = json.loads(config.dnn['cell_area_file'].read_text())
    registry = None
    if saved.get('module_id_convention') == 'stable_registry_v1':
        from .module_registry import load_registry
        registry = load_registry(str(config.dnn['module_registry_file']))
        # Appending unrelated modules is harmless; changing any saved ID is not.
        for name, old in saved['modules'].items():
            if registry['modules'].get(name) != old['module_id']:
                raise ValueError(f'Persistent module ID differs from the saved model for {name}')
    for name, info in metadata.items():
        nch = 37*info['identity'][2]
        values = areas.get(name[:4], {}).get('SF')
        if (nch > saved['max_channels'] or not isinstance(values, list) or len(values) != nch
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError(f'Invalid application channel width/central cell areas for {name}')
        if name in spec['modules']:
            old = spec['modules'][name]
            if old['nch'] != nch or old['cell_area_fraction'] != values:
                raise ValueError(f'Application/training geometry differs for {name}')
        else:
            if not config.dnn['allow_unknown_modules']:
                raise ValueError(f'{name} was not in training_inputs. Set allow_unknown_modules: true '
                                 'explicitly to apply the model outside its saved module vocabulary.')
            index = saved['unknown_module_id']
            if registry is not None:
                if name not in registry['modules']:
                    raise ValueError(f'{name} is not in the persistent module registry. Register it explicitly with '
                                     f'python3 -m utils.module_registry --registry {config.dnn["module_registry_file"]} --add {name}')
                index = registry['modules'][name]
                if index > 2**24:
                    raise ValueError('Module ID is not exactly representable as float32')
            spec['modules'][name] = {'module_id': index, 'nch': nch,
                                     'nerx': nch//37, 'cell_area_fraction': values}
    from .preprocessing import calibration_only, reference_from_file
    if calibration_only(spec):
        for name, info in metadata.items():
            reference_from_file(config.target.calibration_file, name, 37*info['identity'][2])
    return spec


def _preprocessing(pre, spec):
    from .preprocessing import validate_descriptor
    validate_descriptor(pre, spec)


def resolve_application(config):
    if config.target.selection not in {'full','trigtime'}:
        raise NotImplementedError('DNN application supports full or trigtime selections')
    from .model_selection import resolve_application as resolve_models
    return resolve_models(config)


def load_checkpoint(model):
    import torch
    folder, record = model['folder'], model['record']
    checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    expected = model['definition']
    pre = json.loads((folder/'preprocessing.json').read_text())
    if (checkpoint['schema'] != record['schema'] or checkpoint['signature'] != record['signature']
            or checkpoint['feature_spec'] != expected['feature_spec']
            or checkpoint['options'] != expected['options'] or checkpoint['modules'] != expected['modules']
            or checkpoint['preprocessing'] != pre or checkpoint['epoch'] != record['summary']['best_epoch']):
        raise ValueError(f'DNN checkpoint/manifest metadata mismatch: {folder}')
    _preprocessing(pre, checkpoint['feature_spec'])
    return checkpoint


def compute_module(config, resolved, module, work, checkpoint, device):
    import numpy as np
    from .training import initialize_training_root
    from .inference_inputs import event_batches
    from .inference_predictor import Predictor
    from .dnn_prefetch import prefetched_batches
    ROOT = initialize_training_root()
    if ROOT.IsImplicitMTEnabled(): ROOT.DisableImplicitMT()
    ROOT.EnableThreadSafety()
    if not ROOT.gInterpreter.Declare(f'#include {json.dumps(str(ROOT_DIR/"utils/inference_io.hxx"))}'):
        raise RuntimeError('Could not compile the prediction ROOT writer')
    row = resolved['modules'][module]
    spec = resolved.get('feature_specs', {}).get(module, resolved['feature_spec'])
    from .model_selection import activity_thresholds
    config = replace(config, preparation=dict(config.preparation,
                     activity_thresholds=tuple(activity_thresholds(checkpoint['feature_spec']))))
    predictor = Predictor(config, checkpoint, spec, module, device)
    outputs, count_targets = [], 0
    for source in row['inputs']:
        target = work/source['output_file']
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = ROOT.hgcal_cm_inference.Writer(str(target), spec['modules'][module]['nch'],
                                               config.application['apply_events_per_chunk'])
        events = 0
        try:
            with prefetched_batches(event_batches(config, spec, module, source),
                    depth=config.application['apply_prefetch_batches'], prepare=predictor.prepare) as batches:
                for batch in batches:
                    prediction, corrected = predictor.predict(batch)
                    writer.append(ROOT.VecOps.RVec['ULong64_t'](np.ascontiguousarray(batch['identities']).ravel()),
                                  ROOT.VecOps.RVec['float'](prediction.ravel()), ROOT.VecOps.RVec['float'](corrected.ravel()))
                    events += len(prediction)
                    count_targets += int(np.isfinite(prediction).sum())
            writer.finish()
        finally:
            del writer
        stat = target.stat()
        outputs.append({'file': source['output_file'], 'run': source['run'],
            'source_file_index': source['source_file_index'], 'input_events': source['input_entries'],
            'selected_events': events, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        print(f'{module}: Run{source["run"]}/file{source["source_file_index"]}: {events}/{source["input_entries"]} selected events', flush=True)
    return outputs, {'input_events': sum(r['input_events'] for r in outputs),
                     'selected_events': sum(r['selected_events'] for r in outputs),
                     'channels': spec['modules'][module]['nch'], 'predicted_targets': count_targets}


def run(config, resolved):
    import torch
    import apply as analytic
    from analytic_allinputs import _write_manifest
    from .inference_runtime import device as choose_device
    from .application_provenance import current_hashes
    chosen = choose_device(config)
    destination = resolved['destination']
    existing = analytic.cached_application(resolved)
    if existing and existing['complete']:
        print(f'Reusing applied DNN corrections: {destination}', flush=True)
        return
    checkpoints = {name: load_checkpoint(model) for name, model in resolved['models'].items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent/f'.{destination.name}.application-lock'
    try:
        lock.mkdir()
    except FileExistsError:
        owner = lock/'owner.json'
        details = owner.read_text() if owner.is_file() else 'unknown owner'
        raise ValueError(f'Application lock DIRECTORY exists: {lock}; recorded owner: {details}. '
                         'Check that process on its recorded host before removing only this lock.') from None
    try:
        from utils.workflow_registry import record_lock_owner
        record_lock_owner(lock, {'host': socket.gethostname(), 'pid': os.getpid()})
        existing = analytic.cached_application(resolved)
        if existing and existing['complete']:
            print(f'Reusing applied DNN corrections: {destination}', flush=True)
            return
        records = dict(existing['module_records']) if existing else {}
        destination.mkdir(exist_ok=True)
        _write_manifest(destination, analytic._overview(resolved, records, complete=False))
        print(f'DNN application: torch={torch.__version__}, device={chosen}; checkpoint=model.pt (best)', flush=True)
        for module, row in resolved['modules'].items():
            if module in records:
                print(f'Reusing applied DNN module: {destination/module}', flush=True)
                continue
            pre = checkpoints[row['group']]['preprocessing']
            from .preprocessing import calibration_only
            missing = (0 if calibration_only(pre) else
                       sum(n == 0 for n in pre['targets']['count'][:row['definition']['nch']]))
            if missing:
                print(f'{module}: {missing} target positions have no normalization observations; '
                      'using the saved zero-mean/unit-scale policy, not newly fitted statistics.', flush=True)
            work = Path(tempfile.mkdtemp(prefix=f'.{module}.work-', dir=destination))
            try:
                print(f'Applying dnn to {module}', flush=True)
                outputs, summary = compute_module(config, resolved, module, work, checkpoints[row['group']], chosen)
                record = {'schema': SCHEMA, 'signature': signature(resolved['definition']),
                    'definition': resolved['definition'], 'module': module, 'complete': True,
                    'summary': summary, 'outputs': outputs,
                    'implementation': {'torch': torch.__version__, 'device': chosen,
                        'resources': json.loads(json.dumps(config.application, default=str)),
                        'infer_batch': config.dnn['infer_batch'],
                        'source_sha256': current_hashes(ROOT_DIR, CODE)}}
                _write_manifest(work, record)
                analytic._check_outputs(work, outputs, [r['output_file'] for r in row['inputs']])
                if (destination/module).exists():
                    raise FileExistsError(f'Application module appeared during production: {destination/module}')
                os.rename(work, destination/module)
                records[module] = record
                _write_manifest(destination, analytic._overview(resolved, records, complete=False))
            except Exception as exc:
                raise RuntimeError(f'{exc}. Completed DNN modules remain in {destination}; unfinished work: {work}') from exc
        _write_manifest(destination, analytic._overview(resolved, records, complete=True))
        print(f'Applied DNN corrections ready: {destination}', flush=True)
    finally:
        shutil.rmtree(lock)
