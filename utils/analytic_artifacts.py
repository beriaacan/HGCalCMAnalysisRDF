"""Per-module completion records and checked readers for analytic predictors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .analytic import SCHEMA

PUBLICATION = 'per_module_v1'
MODULE_FILES = ('predictor.npz', 'diagnostics.json')



def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(definition: dict) -> str:
    return hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def definitions_match(stored,requested):
    return json.loads(json.dumps(stored))==json.loads(json.dumps(requested))


def check_definition(record: dict, path: Path, definition: dict | None = None) -> None:
    stored = record.get('definition')
    if (record.get('schema') != SCHEMA or not isinstance(stored, dict)
            or record.get('signature') != signature(stored)):
        raise ValueError(f'Invalid analytic completion record: {path}')
    if definition is not None and not definitions_match(stored, definition):
        raise ValueError(f'Analytic predictor provenance mismatch: {path}. Preserve the previous '
                         'directory under another name or choose a separate model_dir.')


def check_outputs(directory: Path, outputs: list, expected: list[str]) -> None:
    if [row['file'] for row in outputs] != expected:
        raise ValueError(f'Incomplete analytic predictor inventory: {directory}')
    for row in outputs:
        path = directory / row['file']
        if not path.is_file() or sha(path) != row['sha256']:
            raise ValueError(f'Missing or modified analytic predictor output: {path}')


def checked_module(directory: Path, module: str, definition: dict | None = None) -> dict | None:
    target = directory / module
    if not target.exists():
        return None
    path = target / 'manifest.json'
    if not path.is_file():
        raise ValueError(f'Incomplete analytic module: {target}')
    record = json.loads(path.read_text())
    check_definition(record, path, definition)
    if (record.get('publication') != PUBLICATION or record.get('complete') is not True
            or record.get('module') != module or not isinstance(record.get('summary'), dict)):
        raise ValueError(f'Invalid analytic module completion record: {path}')
    check_outputs(target, record.get('outputs', []), list(MODULE_FILES))
    return record


def load_module(directory: Path, module: str) -> tuple[dict, dict]:
    """Read a completed module even while other modules are still being produced."""
    import numpy as np
    directory = Path(directory)
    if (directory / module / 'manifest.json').is_file():
        checked_module(directory, module)
    else:
        raise ValueError(f'No completed new-format analytic predictor for {module}: {directory}')
    with np.load(directory / module / 'predictor.npz', allow_pickle=False) as saved:
        model = {name: saved[name] for name in saved.files}
    return model, json.loads((directory / module / 'diagnostics.json').read_text())
