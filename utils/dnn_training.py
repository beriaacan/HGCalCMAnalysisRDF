"""One normal-MSE trainer for local CPU/GPU and HTCondor execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import socket
import tempfile
import time

from analytic_allinputs import _write_manifest as _atomic_write_manifest
from .analytic_artifacts import sha, signature

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA = 'hgcal_cm_dnn_v4'
ARTIFACTS = ('model.pt', 'last.pt', 'preprocessing.json', 'model_spec.json', 'history.json')
CODE = ('train.py', 'utils/dnn_training.py', 'utils/dnn_batches.py',
        'utils/training.py', 'utils/training_helpers.hxx', 'utils/train_config.py',
        'utils/dnn_prefetch.py', 'utils/dnn_tensors.py', 'utils/training_cache.py', 'utils/training_metadata.py', 'utils/preprocessing.py', 'utils/module_registry.py', 'utils/dnn_internal_normalization.py')


def _write_manifest(directory, record):
    from .training_metadata import attach
    _atomic_write_manifest(directory, attach(directory, record))


def groups(config, spec):
    names = sorted(spec['modules'])
    return {'combined': names} if config.training['training_grouping'] == 'combined' else {m: [m] for m in names}


def destination(config, group):
    from .training_metadata import destination as model_destination
    return model_destination(config, group)


def definition(config, manifest, directory, modules):
    options = {name: config.dnn[name] for name in ('nodes', 'dropout', 'epochs', 'batch_events',
               'learning_rate', 'weight_decay', 'shuffle_buffer_chunks', 'sample_weighting')}
    options.update({name: config.training[name] for name in ('training_seed', 'training_deterministic',
        'optimizer_betas', 'optimizer_epsilon', 'early_stopping_patience', 'early_stopping_min_delta',
        'lr_scheduler_factor', 'lr_scheduler_patience', 'lr_scheduler_threshold', 'lr_scheduler_min_lr')})
    from .identity import model_id, data_id
    group = 'combined' if config.training['training_grouping']=='combined' else modules[0]
    result = {'schema': SCHEMA, 'recipe_id': model_id(config, group),
        'dataset_id': data_id(manifest['definition'], modules), 'training_inputs': str(directory), 'training_signature': manifest['signature'],
        'training_manifest_sha256': sha(directory / 'manifest.json'),
        'preprocessing_sha256': manifest['preprocessing_sha256'], 'feature_spec': manifest['feature_spec'],
        'modules': modules, 'options': options, 'events_per_chunk': config.training['training_events_per_chunk'],
        'batching': 'up to batch_events accepted module-event rows; all finite selected targets per event; partial batch at buffer end',
        'architecture': 'Frozen ADC/CM FixedStandardization buffers; Linear-ReLU-(optional Dropout) hidden layers; scalar linear output',
        'optimizer': 'AdamW, AMSGrad=false; ReduceLROnPlateau relative threshold, cooldown=0, eps=1e-8',
        'loss': 'mean(weight * squared ADC residual); epoch sum divided by sample count',
        'validation': 'saved retained validation; best checkpoint and early stopping, no normalization fitting',
        'source_loading': 'ROOT unfiltered RBatchGenerator; RDF acceptance flags; keep final partial batch',
        'code_sha256': scientific_code_hashes()}
    from .preprocessing import dnn_definition
    result['preprocessing'] = dnn_definition()
    result['dnn_internal_statistics'] = ('retained training event-target samples for this model group; '
        'own ADC replaced before moments; unweighted population variance; constants scale one; '
        'ADC/CM columns only; buffers stored in model state')
    return result


def scientific_code_hashes():
    from .dnn_provenance import current_hashes
    return current_hashes(ROOT_DIR,CODE)


def cached(directory, expected):
    path = directory / 'manifest.json'
    if not path.exists():
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f'DNN directory has no provenance manifest: {directory}')
        return None
    record = json.loads(path.read_text())
    if (record.get('schema') != SCHEMA or record.get('signature') != signature(expected)
            or record.get('definition') != json.loads(json.dumps(expected))):
        raise ValueError(f'DNN provenance mismatch: {directory}. The automatic variant already exists with different provenance; do not overwrite it.')
    if not record.get('complete'):
        return None
    if [row['file'] for row in record['outputs']] != list(ARTIFACTS):
        raise ValueError(f'Incomplete DNN output inventory: {directory}')
    for row in record['outputs']:
        relative = Path(row['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid DNN artifact path')
        if not (directory / relative).is_file() or sha(directory / relative) != row['sha256']:
            raise ValueError(f'Missing or modified DNN artifact: {directory / relative}')
    if 'training_data' in record:
        from .training_metadata import read
        read(directory, record)
    return record


def build_model(spec, options):
    import torch
    layers = []
    size = len(spec['feature_names'])
    for width in options['nodes']:
        layers.extend([torch.nn.Linear(size, width), torch.nn.ReLU()])
        if options['dropout'] > 0:
            layers.append(torch.nn.Dropout(options['dropout']))
        size = width
    layers.append(torch.nn.Linear(size, 1))
    network = torch.nn.Sequential(*layers)
    from collections import OrderedDict
    from .preprocessing import calibration_only
    from .dnn_internal_normalization import FixedStandardization, columns
    if (not calibration_only(spec) or spec.get('max_channels') != 444
            or 'module_fraction' not in spec.get('global_groups', [])
            or spec.get('module_fraction_denominator') != 444):
        raise ValueError('DNN requires physical ADC/CM features and module_fraction=Nchannels/444 before the model')
    columns(spec)  # Validate the fixed normalization-column mapping.
    return torch.nn.Sequential(OrderedDict([
        ('input_normalization', FixedStandardization(len(spec['feature_names']))),
        ('network', network)]))


def save_checkpoint(path, payload):
    import torch
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def run_epoch(model, optimizer, data, device, training):
    import torch
    def transfer(value):
        value=torch.as_tensor(value)
        return value.to(device,non_blocking=device=='cuda' and value.is_pinned())
    model.train(training)
    loss_sum = 0.
    samples = 0
    with torch.set_grad_enabled(training):
        for x, y, w in data:
            x, y, w = (transfer(value) for value in (x,y,w))
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x).squeeze(-1)
            loss = torch.mean(w * (prediction - y).square())
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite DNN loss')
            if training:
                loss.backward()
                if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                    raise ValueError('Nonfinite DNN gradient')
                optimizer.step()
                if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
                    raise ValueError('Nonfinite DNN parameter')
            samples += len(y)
            loss_sum += float(loss.detach()) * len(y)
    if not samples:
        raise ValueError('No retained samples in DNN partition')
    return loss_sum / samples, samples


def write_curves(config, target, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Training figures describe the derivation, independent of application cuts.
    base = ROOT_DIR / 'plots' / config.correction.campaign / 'training'
    setup = base / config.model_dir.name
    folder = setup / target.parent.name / target.name
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [('train_loss', 'Training'), ('validation_loss', 'Validation')]:
        ax.plot([r['epoch'] for r in history], [r[key] for r in history], label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean squared error [ADC²]')
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    for extension in config.plotting['plot_formats']:
        fig.savefig(folder / f'loss.{extension}')
    plt.close(fig)
    return str(folder)


def training_lock_message(lock, *, purpose='training'):
    """Explain the filesystem guard without assuming a remote PID is alive."""
    try:
        owner=json.loads((lock/'owner.json').read_text())
    except (OSError,ValueError):
        owner={}
    if not isinstance(owner,dict): owner={}
    host,pid=owner.get('host'),owner.get('pid')
    title,actor,artifacts=('Training','trainer','model and checkpoint files') if purpose=='training' else ('Cache build','cache builder','completed cache units')
    lines=[f'{title} blocked: a lock DIRECTORY already exists:\n  {lock}',
           f'This prevents two {actor}s from writing to the same destination.']
    if isinstance(host,str) and host and type(pid) is int and pid>0:
        lines.append(f'Recorded owner: PID {pid} on {host}.')
        if host.split('.')[0].lower()==socket.gethostname().split('.')[0].lower():
            try:
                os.kill(pid,0)
            except ProcessLookupError:
                lines.append('Status: STALE lock; the recorded PID no longer exists on this host.')
            except PermissionError:
                lines.append('Status: cannot verify the recorded process with current permissions.')
            else:
                lines.append('Status: the recorded PID still exists on this host; check its command.')
        else:
            lines.append('Status: not verified; the recorded owner is on another host.')
        lines.append(f'Check on {host}: ps -p {pid} -o pid,stat,etime,cmd')
    else:
        lines.append(f'Owner details are missing/unreadable; verify that no {actor} is using this destination.')
    lines.extend([f'An interrupted/crashed {actor} can leave this directory behind.',
                  f'After confirming the {actor} has exited, remove ONLY the lock directory above and retry.',
                  f'Keep the {artifacts}.'])
    return '\n'.join(lines)


def train_group(config, manifest, input_dir, preprocessing, group, modules, *, batch_factory=None, event_cache=None):
    from .dnn_batches import batches
    from .dnn_tensors import compact_event_batches
    if event_cache is not None and batch_factory is None:
        from functools import partial
        batches=partial(batches,chunk_reader=event_cache.source_chunks)
        compact_event_batches=partial(compact_event_batches,chunk_reader=event_cache.source_chunks)
    target = destination(config, group)
    expected = definition(config, manifest, input_dir, modules)
    completed = cached(target, expected)
    if completed:
        from .training_metadata import ensure
        ensure(config, target, expected)
        write_curves(config, target, json.loads((target / 'history.json').read_text()))
        print(f'Reusing DNN model: {target}', flush=True)
        return completed
    target.mkdir(parents=True, exist_ok=True)
    lock = target / '.training-lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(training_lock_message(lock)) from None
    try:
        (lock / 'owner.json').write_text(json.dumps({'host': socket.gethostname(), 'pid': os.getpid()}))
        return _train_locked(config, manifest, preprocessing, group, modules, target, expected,
                             batches if batch_factory is None else batch_factory,
                             compact_factory=compact_event_batches if batch_factory is None else None,
                             event_cache=event_cache)
    finally:
        shutil.rmtree(lock)


def run_partition(config,manifest,preprocessing,modules,partition,epoch,group,model,optimizer,device,
                  *,factory,expander=None):
    """Prefetch CPU payloads; perform any tensor expansion on the consumer."""
    from .training_progress import report_batches,sample_total
    from .dnn_prefetch import cpu_batch,prefetched_batches
    from .dnn_tensors import compact_cpu_batch
    prepare_batch=compact_cpu_batch if expander is not None else cpu_batch
    prepare=lambda batch:prepare_batch(batch,pin_memory=device=='cuda' and config.training['training_pin_memory'])
    source=factory(config,manifest,preprocessing,modules,partition,epoch)
    with prefetched_batches(source,depth=config.training['training_prefetch_batches'],prepare=prepare) as data:
        if expander is not None:data=(expander.expand(payload) for payload in data)
        progress=report_batches(data,sample_total(manifest,modules,partition),
            f'{group} epoch {epoch+1}/{config.dnn["epochs"]} {("training","validation")[partition]}')
        return run_epoch(model,optimizer,progress,device,partition==0)


def _train_locked(config, manifest, preprocessing, group, modules, target, expected, factory, *, compact_factory=None, event_cache=None):
    import numpy as np
    import torch
    from .training import initialize_training_root
    ROOT = initialize_training_root()
    from .dnn_provenance import current_hashes
    implementation = {'source_sha256':current_hashes(ROOT_DIR,CODE),
        'cuda_workspace_config':config.training['cublas_workspace_config'],
        'cublaslt_workspace_size_kib':config.training['cublaslt_workspace_size_kib'],
        'prefetch_batches':config.training['training_prefetch_batches'],
        'pin_memory':config.training['training_pin_memory']}
    # ROOT's loader has a single tensor writer, with a separate C++ I/O thread.
    if ROOT.IsImplicitMTEnabled():
        ROOT.DisableImplicitMT()
    ROOT.EnableThreadSafety()
    torch.set_num_threads(config.resources['cpus'])
    torch.use_deterministic_algorithms(config.training['training_deterministic'])
    torch.backends.cudnn.benchmark = False
    chosen = config.training['training_device']
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if chosen == 'auto' else chosen
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA training requested but no usable CUDA GPU is available')
    if os.environ.get('HGCAL_REQUIRE_CUDA') == '1' and device != 'cuda':
        raise ValueError('HTCondor requested a GPU but CUDA is unusable; refusing CPU fallback')
    use_compact=device=='cuda' and compact_factory is not None
    implementation['input_backend']='numeric_cache' if event_cache is not None else 'root'
    if event_cache is not None:
        implementation['event_cache']={'directory':str(event_cache.folder),'signature':event_cache.signature}
    implementation['input_expansion']='cuda_compact_events' if use_compact else 'cpu_expanded_inputs'
    runtime = {'torch': torch.__version__, 'ROOT': ROOT.gROOT.GetVersion(), 'numpy': np.__version__,
               'python': __import__('platform').python_version(), 'cuda_build': torch.version.cuda}
    options = expected['options']
    seed = options['training_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(manifest['feature_spec'], options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'],
        weight_decay=options['weight_decay'], betas=tuple(options['optimizer_betas']), eps=options['optimizer_epsilon'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=options['lr_scheduler_factor'], patience=options['lr_scheduler_patience'],
        threshold=options['lr_scheduler_threshold'], threshold_mode='rel', min_lr=options['lr_scheduler_min_lr'],
        cooldown=0, eps=1e-8)
    history = []
    best = float('inf')
    patience_best = float('inf')
    bad = start = 0
    best_checkpoint = None
    last = target / 'last.pt'
    if last.exists():
        if not config.training['training_resume']:
            raise ValueError(f'Incomplete model exists and training_resume is false: {target}')
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['signature'] != signature(expected) or state['runtime'] != runtime:
            raise ValueError('Checkpoint provenance or Python/ROOT/PyTorch runtime changed; refusing to resume this automatic model variant')
        model.load_state_dict(state['model_state'])
        from .dnn_internal_normalization import validate_model
        validate_model(model)
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        history = state['history']
        best, patience_best, bad, start = state['best_loss'], state['patience_best'], state['bad_epochs'], state['epoch']
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None and device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        best_checkpoint = state['best_checkpoint']
        if best_checkpoint is not None:
            save_checkpoint(target / 'model.pt', best_checkpoint)
        print(f'Resuming {group} after epoch {start}', flush=True)
    else:
        if (target / 'manifest.json').exists() and not config.training['training_resume']:
            raise ValueError('Incomplete training directory exists and training_resume is false')
        _write_manifest(target, {'schema': SCHEMA, 'definition': expected, 'signature': signature(expected),
                                 'complete': False, 'implementation':implementation})
        (target / 'preprocessing.json').write_text(json.dumps(preprocessing, indent=2, allow_nan=False) + '\n')
        (target / 'model_spec.json').write_text(json.dumps({'definition': expected,
            'output': 'pedestal-subtracted ADC; frozen ADC/CM z-scoring is inside the model'}, indent=2)+'\n')
    from .training_metadata import ensure
    ensure(config, target, expected, owns_training_lock=True)
    from .dnn_internal_normalization import layer, fit, validate_model
    normalizer = layer(model)
    if not last.exists():
        normalizer.initialize(*fit(config, manifest, modules, event_cache))
        # Save fixed buffers before epoch 1; resume never refits them.
        save_checkpoint(last, {'signature': signature(expected), 'runtime': runtime, 'epoch': 0,
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(), 'history': [], 'best_loss': best,
            'patience_best': patience_best, 'bad_epochs': bad, 'best_checkpoint': None,
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else None,
            'implementation': implementation})
    validate_model(model)
    print('Fixed DNN preprocessing: internal ADC/CM z-score; targets and MSE in ADC units', flush=True)
    from .dnn_tensors import TensorExpander
    expanders=({p:TensorExpander(config,manifest,preprocessing,device,p) for p in (0,1)} if use_compact else {})
    print('Event source: '+('numeric cache '+str(event_cache.folder) if event_cache is not None else 'prepared ROOT files'),flush=True)
    print(f'Training {group}: {len(modules)} modules, device={device}, parameters={sum(p.numel() for p in model.parameters()):,}', flush=True)
    print(f'Input expansion: {"CUDA from compact events" if use_compact else "CPU"}; '
          f'{len(manifest["feature_spec"]["feature_names"])} features',flush=True)
    print(f'Batches: up to {options["batch_events"]} full module-events; '
          f'read chunks={config.training["training_events_per_chunk"]} source events; '
          f'prefetch={config.training["training_prefetch_batches"]}; '
          f'pinned CPU memory={device=="cuda" and config.training["training_pin_memory"]}',flush=True)
    for epoch in range(start, options['epochs']):
        if bad >= options['early_stopping_patience']:
            break
        t0 = time.monotonic()
        lr = optimizer.param_groups[0]['lr']
        def partition_loss(partition):
            return run_partition(config,manifest,preprocessing,modules,partition,epoch,group,model,optimizer,device,
                factory=compact_factory if use_compact else factory,expander=expanders.get(partition))
        train_loss, nt = partition_loss(0)
        val_loss, nv = partition_loss(1)
        scheduler.step(val_loss)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'validation_loss': val_loss,
            'learning_rate': lr, 'train_samples': nt, 'validation_samples': nv, 'seconds': time.monotonic() - t0})
        if val_loss < best:
            best = val_loss
            best_checkpoint = {'schema': SCHEMA, 'signature': signature(expected), 'epoch': epoch + 1,
                'validation_loss': best, 'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'feature_spec': manifest['feature_spec'], 'preprocessing': preprocessing, 'options': options, 'modules': modules}
            save_checkpoint(target / 'model.pt', best_checkpoint)
        if val_loss < patience_best - options['early_stopping_min_delta']:
            patience_best, bad = val_loss, 0
        else:
            bad += 1
        state = {'signature': signature(expected), 'runtime': runtime, 'epoch': epoch + 1,
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
            'history': history, 'best_loss': best, 'patience_best': patience_best, 'bad_epochs': bad,
            'best_checkpoint': best_checkpoint, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else None,
            'implementation':implementation}
        save_checkpoint(last, state)
        _write_manifest(target, {'schema': SCHEMA, 'definition': expected, 'signature': signature(expected),
            'complete': False, 'last_epoch': epoch + 1, 'best_validation_loss': best,
            'runtime': runtime, 'implementation':implementation})
        (target / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
        write_curves(config, target, history)
        print(f'{group} epoch {epoch+1}/{options["epochs"]}: train={train_loss:.6g}, validation={val_loss:.6g}, lr={lr:.3g}, {history[-1]["seconds"]:.1f}s', flush=True)
    if best_checkpoint is None:
        raise ValueError('Training produced no valid best checkpoint')
    (target / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
    curve_dir = write_curves(config, target, history)
    files = ARTIFACTS
    record = {'schema': SCHEMA, 'definition': expected, 'signature': signature(expected), 'complete': True,
        'runtime': runtime, 'implementation':implementation,
        'summary': {'epochs': len(history), 'best_epoch': best_checkpoint['epoch'],
        'best_validation_loss': best, 'device': device, 'loss_plots': curve_dir},
        'outputs': [{'file': name, 'sha256': sha(target / name)} for name in files]}
    _write_manifest(target, record)
    print(f'DNN model ready: {target}', flush=True)
    return record
