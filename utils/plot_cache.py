"""Audited immutable file contributions and versioned derived plotting products."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import tempfile

from analytic_allinputs import _write_manifest
from .analytic_artifacts import sha, signature

SCHEMA = 'hgcal_cm_plot_cache_v1'


class Product:
    def __init__(self, directory, record):
        self.directory, self.record = Path(directory), record
        self.path = self.directory / record['generation']

    @property
    def token(self):
        return {'directory': str(self.directory), 'signature': self.record['signature'],
                'outputs': self.record['outputs']}

    def arrays(self):
        import numpy as np
        with np.load(self.path / 'data.npz', allow_pickle=False) as saved:
            return {name: saved[name] for name in saved.files}


def read(directory):
    directory = Path(directory)
    if not directory.exists():
        return None
    path = directory / 'manifest.json'
    if not path.is_file():
        # An interrupted first write may have only a retained work directory.
        if all(p.name.startswith('.') for p in directory.iterdir()):
            return None
        raise ValueError(f'Incomplete plotting cache: {directory}')
    record = json.loads(path.read_text())
    if (record.get('schema') != SCHEMA or record.get('complete') is not True
            or record.get('signature') != signature(record.get('definition', {}))):
        raise ValueError(f'Invalid plotting completion record: {path}')
    generation = record.get('generation', '')
    if not generation.startswith('generation_') or not generation.removeprefix('generation_').isdigit():
        raise ValueError(f'Invalid plotting generation: {path}')
    for row in record['outputs']:
        relative = Path(row['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid plotting cache path: {relative}')
        output = directory / generation / relative
        if not output.is_file() or sha(output) != row['sha256']:
            raise ValueError(f'Missing or modified plotting product: {output}')
    return Product(directory, record)


def matching(directory, definition, *, immutable=False):
    previous = read(directory)
    if previous is not None and previous.record['definition'] == json.loads(json.dumps(definition)):
        return previous
    if previous is not None and immutable:
        old = previous.record['definition']
        changed = [key for key in old.keys() | definition.keys() if old.get(key) != definition.get(key)]
        raise ValueError(f'Previously used source contribution changed: {directory}; fields: {sorted(changed)}. '
                         'Preserve the affected statistics cache under another name before explicitly rebuilding it, '
                         'or use a separate output root.')
    return None


def publish(directory, definition, build):
    """Build locally, copy closed files to storage, publish the manifest last."""
    directory = Path(directory)
    read(directory)
    directory.mkdir(parents=True, exist_ok=True)
    version = 1
    while (directory / f'generation_{version:04d}').exists():
        version += 1
    generation = f'generation_{version:04d}'
    work = Path(tempfile.mkdtemp(prefix='.work-', dir=directory))
    local = Path(tempfile.mkdtemp(prefix='hgcal_plot_cache-', dir='/tmp'))
    try:
        # NPZ/ZIP writers seek back to update local headers. Build on local
        # storage, then copy complete files sequentially to EOS/AFS instead.
        summary = build(local)
        files = sorted(p for p in local.rglob('*') if p.is_file())
        outputs = []
        for source in files:
            relative = source.relative_to(local)
            expected = sha(source)
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha(target) != expected:
                raise OSError(f'Plotting product changed during storage copy: {target}')
            outputs.append({'file': str(relative), 'sha256': expected})
        if not outputs:
            raise ValueError('Plot product builder produced no files')
        record = {'schema': SCHEMA, 'complete': True, 'generation': generation,
                  'definition': json.loads(json.dumps(definition)), 'signature': signature(definition),
                  'summary': summary, 'outputs': outputs}
        _write_manifest(work, record)
        os.rename(work, directory / generation)
        _write_manifest(directory, record)
    except Exception as exc:
        raise RuntimeError(f'{exc}. Retained plotting work: {work}; local build: {local}') from exc
    shutil.rmtree(local)
    return Product(directory, record)
