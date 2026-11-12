"""Immutable, bounded event arrays for repeated CPU/CUDA training.

Uncompressed NumPy archives contain one array per original ROOT read chunk,
not one expanded row per channel. Keeping chunk boundaries preserves shuffling
and optimizer updates. ZIP CRC checks run when arrays are read; manifests pin
the input definition, unit metadata, file identity and creation-time SHA256.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import zipfile

import numpy as np

from analytic_allinputs import _write_manifest
from .analytic_artifacts import sha, signature
from .analytic_fit import _sources

SCHEMA = 'hgcal_cm_event_arrays_v2'
ROOT_DIR = Path(__file__).resolve().parents[1]
CODE = ('utils/training_cache.py', 'utils/dnn_batches.py',
        'utils/training.py', 'utils/training_helpers.hxx', 'utils/prepare_rdf.py',
        'utils/prepare_helpers.hxx', 'utils/preprocessing.py', 'utils/module_registry.py')
LABELS = ('train', 'validation')


def destination(config, input_dir):
    return Path(input_dir)/'event_arrays'/f'chunks_{config.training["training_events_per_chunk"]}'


def definition(config, input_dir, manifest):
    return {'schema': SCHEMA, 'training_inputs': str(input_dir),
            'training_manifest_sha256': sha(Path(input_dir)/'manifest.json'),
            'training_signature': manifest['signature'],
            'preprocessing_sha256': manifest['preprocessing_sha256'],
            'events_per_chunk': config.training['training_events_per_chunk'],
            'storage': 'uncompressed NumPy chunks; float32 raw globals and selected targets; uint64 source_entry',
            'code_sha256': _code_hashes(manifest)}


def _code_hashes(manifest):
    return {name:sha(ROOT_DIR/name) for name in (*CODE,'utils/identity.py')}


def unit_path(module, source):
    # Module names come from the validated feature specification, not YAML paths.
    if not module or Path(module).name != module or module in {'.', '..'}:
        raise ValueError(f'Invalid cache module name: {module}')
    return Path(f'Run{int(source["run"])}')/module/f'file{int(source["index"])}'


def unit_definition(parent_signature, module, path, source):
    return {'cache_signature': parent_signature, 'module': module,
            'prepared_file': str(path), 'source': source}


def file_record(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    stat = path.stat()
    return {'file': path.name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'sha256': digest.hexdigest()}


def check_unit(folder, expected, manifest_sha=None):
    path = folder/'manifest.json'
    if not path.is_file():
        raise ValueError(f'Incomplete event-array cache unit: {folder}')
    if manifest_sha is not None and sha(path) != manifest_sha:
        raise ValueError(f'Event-array unit manifest changed: {path}')
    record = json.loads(path.read_text())
    if (record.get('definition') != expected or record.get('signature') != signature(expected)
            or not record.get('complete')):
        raise ValueError(f'Event-array cache provenance mismatch: {folder}; keep it separate from changed inputs')
    if [row['file'] for row in record['outputs']] != [f'{label}.npz' for label in LABELS]:
        raise ValueError(f'Invalid event-array output inventory: {folder}')
    for row in record['outputs']:
        file = folder/row['file']
        if not file.is_file():
            raise ValueError(f'Missing event-array file: {file}')
        stat = file.stat()
        if stat.st_size != row['size'] or stat.st_mtime_ns != row['mtime_ns']:
            raise ValueError(f'Event-array file changed: {file}; refusing cache reuse')
    return record


def load(config, input_dir, manifest):
    """Validate once before training/submission; never build implicitly."""
    folder = destination(config, input_dir)
    path = folder/'manifest.json'
    command = f'./steer.py --config {config.config_file} --steps training_cache'
    if not path.is_file():
        raise ValueError(f'Numeric training cache is missing: {folder}\nBuild it once with: {command}')
    expected = definition(config, input_dir, manifest)
    record = json.loads(path.read_text())
    if record.get('definition') != expected or record.get('signature') != signature(expected):
        raise ValueError(f'Numeric training cache provenance mismatch: {folder}; do not mix different inputs')
    if not record.get('complete'):
        raise ValueError(f'Numeric training cache is incomplete: {folder}\nResume its build with: {command}')
    expected_units = {}
    for module in sorted(manifest['feature_spec']['modules']):
        for path, source in _sources(manifest, module):
            relative = str(unit_path(module, source))
            expected_units[relative] = unit_definition(record['signature'], module, path, source)
    if set(record['units']) != set(expected_units):
        raise ValueError(f'Numeric training cache source inventory differs: {folder}')
    units = {key: check_unit(folder/key, value, record['units'][key]['manifest_sha256'])
             for key, value in expected_units.items()}
    return EventCache(folder, record['signature'], units)


class EventCache:
    def __init__(self, folder, cache_signature, units):
        self.folder, self.signature, self.units = folder, cache_signature, units

    def source_chunks(self, config, manifest, module, path, source, partition):
        key = str(unit_path(module, source))
        record = self.units[key]
        if record['definition'] != unit_definition(self.signature, module, path, source):
            raise ValueError(f'Event-array source changed: {key}')
        # Recheck size/mtime before every source read, without rehashing GB files.
        check_unit(self.folder/key, record['definition'])
        nch = manifest['feature_spec']['modules'][module]['nch']
        width = manifest['feature_spec']['global_size']
        with np.load(self.folder/key/f'{LABELS[partition]}.npz', allow_pickle=False) as archive:
            for chunk in record['partitions'][partition]['chunks']:
                prefix = str(chunk['index'])
                features, targets = archive[prefix+'_features'], archive[prefix+'_targets']
                if (features.dtype != np.float32 or targets.dtype != np.float32
                        or features.shape != (chunk['events'], width)
                        or targets.shape != (chunk['events'], nch)
                        or int(np.isfinite(targets).sum()) != chunk['targets']):
                    raise ValueError(f'Invalid event-array chunk: {key}/{prefix}')
                yield module, source['run'], features, targets


def write_array(archive, name, value):
    with archive.open(name+'.npy', 'w', force_zip64=True) as output:
        np.lib.format.write_array(output, np.ascontiguousarray(value), allow_pickle=False)


def produce_unit(config, manifest, module, path, source, expected, folder):
    from .dnn_batches import source_chunks
    from .training import initialize_training_root
    ROOT = initialize_training_root()
    # training_inputs may have enabled RDF IMT earlier in this same workflow.
    # ROOT's batch generator requires the same single writer as the trainer.
    if ROOT.IsImplicitMTEnabled():
        ROOT.DisableImplicitMT()
    ROOT.EnableThreadSafety()
    folder.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.'+folder.name+'.work-', dir=folder.parent))
    nch = manifest['feature_spec']['modules'][module]['nch']
    parts = [{'chunks': [], 'events': 0, 'targets': 0, 'channel_counts': [0]*nch} for _ in LABELS]
    try:
        with ExitStack() as stack:
            archives = [stack.enter_context(zipfile.ZipFile(work/f'{label}.npz', 'w',
                        compression=zipfile.ZIP_STORED, allowZip64=True)) for label in LABELS]
            reader = source_chunks(config, manifest, module, path, source, None, include_entries=True)
            stack.callback(reader.close)
            for _, _, features, targets, entries, partitions in reader:
                for partition, archive in enumerate(archives):
                    mask = partitions == partition
                    if not mask.any():
                        continue
                    x, y, ids = features[mask], targets[mask], entries[mask]
                    if ids.dtype != np.uint64 or len(np.unique(ids)) != len(ids):
                        raise ValueError(f'Invalid/duplicate physical source entries: {path}')
                    part = parts[partition]
                    index = len(part['chunks'])
                    for suffix, value in [('features', x), ('targets', y), ('source_entry', ids)]:
                        write_array(archive, f'{index}_{suffix}', value)
                    counts = np.isfinite(y).sum(axis=0)
                    part['chunks'].append({'index': index, 'events': len(y), 'targets': int(counts.sum())})
                    part['events'] += len(y)
                    part['targets'] += int(counts.sum())
                    part['channel_counts'] = (np.asarray(part['channel_counts'])+counts).tolist()
        summary = next(row for row in manifest['summary'] if row['module'] == module
                       and row['run'] == source['run'] and row['source_file_index'] == source['index'])
        if [row['targets'] for row in parts] != summary['retained_targets']:
            raise ValueError(f'Cached target counts differ from training_inputs: {module}/Run{source["run"]}/file{source["index"]}')
        record = {'definition': expected, 'signature': signature(expected), 'complete': True,
                  'partitions': parts, 'outputs': [file_record(work/f'{label}.npz') for label in LABELS]}
        _write_manifest(work, record)
        if folder.exists():
            raise FileExistsError(f'Event-array destination already exists: {folder}')
        os.rename(work, folder)
        return record
    finally:
        # A completed unit is already published; failed temporary arrays are disposable.
        if work.exists():
            shutil.rmtree(work)


def build(config, input_dir, manifest):
    folder = destination(config, input_dir)
    expected = definition(config, input_dir, manifest)
    expected_signature = signature(expected)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder/'.build-lock'
    try:
        lock.mkdir()
    except FileExistsError:
        from .dnn_training import training_lock_message
        raise ValueError(training_lock_message(lock, purpose='cache')) from None
    try:
        (lock/'owner.json').write_text(json.dumps({'host': socket.gethostname(), 'pid': os.getpid()}))
        existing = folder/'manifest.json'
        if existing.exists():
            old = json.loads(existing.read_text())
            if old.get('definition') != expected or old.get('signature') != expected_signature:
                raise ValueError(f'Event-array cache provenance mismatch: {folder}; refusing to overwrite')
            if old.get('complete'):
                load(config, input_dir, manifest)
                print(f'Reusing numeric training cache: {folder}', flush=True)
                return
        records = {}
        overview = {'definition': expected, 'signature': expected_signature, 'complete': False, 'units': records}
        _write_manifest(folder, overview)
        total_counts = {}
        stored_bytes = 0
        sources = [(module, path, source) for module in sorted(manifest['feature_spec']['modules'])
                   for path, source in _sources(manifest, module)]
        for i, (module, path, source) in enumerate(sources, 1):
            key = str(unit_path(module, source))
            unit_expected = unit_definition(expected_signature, module, path, source)
            target = folder/key
            print(f'Event arrays {i}/{len(sources)}: {key}' + (' (reuse)' if target.exists() else ''), flush=True)
            record = (check_unit(target, unit_expected) if target.exists() else
                      produce_unit(config, manifest, module, path, source, unit_expected, target))
            stored_bytes += sum(row['size'] for row in record['outputs'])
            for partition, part in enumerate(record['partitions']):
                count_key = (module, str(source['run']), partition)
                total_counts[count_key] = total_counts.get(count_key, np.zeros(len(part['channel_counts']), dtype=np.int64)) + part['channel_counts']
            records[key] = {'manifest_sha256': sha(target/'manifest.json')}
            _write_manifest(folder, overview)
        for module, by_run in manifest['counts_by_module_run'].items():
            nch = manifest['feature_spec']['modules'][module]['nch']
            for run, counts in by_run.items():
                for partition, label in enumerate(LABELS):
                    if not np.array_equal(total_counts.get((module, str(run), partition)), counts[label][:nch]):
                        raise ValueError(f'Event-array channel counts differ: {module}/Run{run}/{label}')
        overview['complete'] = True
        overview['stored_bytes'] = stored_bytes
        _write_manifest(folder, overview)
        print(f'Numeric training cache ready: {folder} ({stored_bytes/1024**3:.2f} GiB)', flush=True)
    finally:
        shutil.rmtree(lock)
