"""Atomic writer ownership and small receipts; safe to import on a scheduler AP.

Only the Python standard library is used here. Receipts audit files, never ROOT
entries. Locks live beside the shared outputs, so another checkout/state_dir
cannot accidentally create an independent writer for the same physical scope.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import uuid

WAIT = 75
SKIP = 99


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.write('\n')
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def evidence(paths):
    rows = []
    for path in sorted({str(Path(p).absolute()) for p in paths}):
        file = Path(path)
        stat = file.stat()
        row = {'path': path, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if file.suffix in {'.json', '.yaml', '.py', '.sh', '.hxx', '.txt'}:
            row['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
        rows.append(row)
    return rows


def intact(rows):
    try:
        for row in rows:
            path = Path(row['path'])
            stat = path.stat()
            if stat.st_size != row['size'] or stat.st_mtime_ns != row['mtime_ns']:
                return False
            if 'sha256' in row and hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                return False
        return bool(rows)
    except (OSError, KeyError, TypeError):
        return False


def owner_at(lock):
    path = Path(lock)/'owner.json'
    if not path.is_file():
        return {'status': 'unknown', 'detail': 'Owner publication pending or interrupted; do not remove by age'}
    return json.loads(path.read_text())


def lock_context(record):
    return {'token': record['token'], 'workflow_id': record['workflow_id'],
            'work': record['work'], 'node': record['node'],
            'scheduler': record['scheduler'],
            'lock_catalog': str(Path(record['receipt']).parents[1]/'lock_catalog')}


def record_lock_owner(lock, owner):
    """Publish normal ownership plus a persistent central index for recovery."""
    context = json.loads(os.environ.get('HGCAL_LOCK_CONTEXT', '{}'))
    owner = {**context, **owner}
    if not owner.get('token'):
        token = os.environ.get('HGCAL_LOCAL_WRITER_TOKEN')
        owner['token'] = token or f'local-{owner.get("host")}-{owner.get("pid")}'
    atomic_json(Path(lock)/'owner.json', owner)
    catalog = Path(owner.get('lock_catalog') or
                   Path(__file__).resolve().parents[1]/'.workflow_state'/'lock_catalog')
    path = str(Path(lock).resolve())
    atomic_json(catalog/(digest(path)+'.json'), {'path': path, 'owner': owner})


def describe_owner(lock, owner):
    return (f'Writer lock DIRECTORY exists: {lock}\n'
            f'Owner: {owner.get("status", "unknown")}, host={owner.get("host", "unknown")}, '
            f'pid={owner.get("pid", "unknown")}, job={owner.get("job_id", "not recorded")}, '
            f'workflow={owner.get("work", "local command")}, node={owner.get("node", "local")}. '
            'This is an ownership directory, not a running process. If its owner has stopped, '
            'use workflow.py --clear-locks to inspect stopped-owner cleanup, or --release <submission-directory> --node <node> --owner-stopped '
            'for a workflow lock; verify a local owner before removing its stale directory.')


def claim(locks, owner):
    acquired = []
    try:
        for value in sorted(set(locks)):
            lock = Path(value)
            lock.parent.mkdir(parents=True, exist_ok=True)
            try:
                lock.mkdir()
            except FileExistsError:
                previous = owner_at(lock)
                if previous.get('token') == owner['token']:
                    continue
                return False, (str(lock), previous)
            acquired.append(lock)
            record_lock_owner(lock, owner)
        acquired.clear()  # Ownership now belongs to caller, including inherited locks.
        return True, None
    finally:
        for lock in reversed(acquired):
            if (lock/'owner.json').is_file():
                (lock/'owner.json').unlink()
            lock.rmdir()


def update(locks, owner):
    for value in locks:
        lock = Path(value)
        if owner_at(lock).get('token') != owner['token']:
            raise RuntimeError(f'Writer ownership changed: {lock}')
        record_lock_owner(lock, owner)


def release(locks, token):
    for value in reversed(sorted(set(locks))):
        lock = Path(value)
        if not lock.exists():
            continue
        if owner_at(lock).get('token') != token:
            raise ValueError(f'Refusing to release another producer: {lock}')
        (lock/'owner.json').unlink()
        lock.rmdir()


def write_state(record, status, **extra):
    value = {'token': record['token'], 'work': record['work'], 'node': record['node'],
             'stage': record['stage'], 'status': status, 'updated': stamp(), **extra}
    atomic_json(record['state_file'], value)
    return value


def receipt_valid(record, *, metadata_only=False):
    path = Path(record['receipt'])
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    if metadata_only and any(row['path'].startswith(('/eos/', '/cvmfs/')) for row in value.get('evidence', [])):
        return False  # Worker validates storage; the scheduler need not mount EOS.
    return (value.get('request') == record['request'] and value.get('complete') is True
            and intact(value.get('evidence', [])))


def check_freeze(record, *, metadata_only=False):
    files = [row for row in record['frozen_files'] if not metadata_only
             or not row['path'].startswith(('/eos/', '/cvmfs/'))]
    if not intact(files):
        raise ValueError('Frozen code/config/runtime dependencies changed; prepare a new workflow')
    for row in record['inventories']:
        if metadata_only:
            continue
        import glob
        current = sorted(str(Path(p).resolve()) for p in glob.glob(row['pattern']) if Path(p).is_file())
        if current != row['files'] or not intact(row['evidence']):
            raise ValueError(f'Pinned Nano inventory changed: {row["pattern"]}')
    for assignment in record.get('module_ids', []):
        if metadata_only and assignment['path'].startswith(('/eos/', '/cvmfs/')):
            continue
        path = Path(assignment['path'])
        modules = json.loads(path.read_text())['modules'] if path.is_file() else {}
        if any(modules.get(name) != index for name, index in assignment['values'].items()):
            raise ValueError(f'Pinned module IDs changed: {path}')


def load_record(path):
    record = json.loads(Path(path).read_text())
    plan = json.loads((Path(record['work'])/'plan.json').read_text())
    expected = next((row for row in plan['nodes'] if row['node'] == record['node']), None)
    if expected != record:
        raise ValueError('Node job.json differs from the frozen workflow plan')
    return record


def pre(record):
    check_freeze(record, metadata_only=True)
    initial = record.get('initial_complete') and not any(
        row['path'].startswith(('/eos/', '/cvmfs/')) for row in record['initial_evidence'])
    if receipt_valid(record, metadata_only=True) or (initial and intact(record['initial_evidence'])):
        write_state(record, 'reused')
        return SKIP
    if initial:
        raise ValueError('A product verified complete during planning changed. Prepare a new workflow; dependencies were pruned.')
    owner = {'token': record['token'], 'work': record['work'], 'node': record['node'],
             'status': 'queued', 'host': socket.gethostname(), 'pid': None, 'updated': stamp(),
             'state_file': record['state_file'], **lock_context(record)}
    ok, busy = claim(record['registry_locks'], owner)
    if not ok:
        lock, other = busy
        write_state(record, 'waiting', owner=other, lock=lock)
        if other.get('status') == 'failed':
            raise ValueError(describe_owner(lock, other))
        print(describe_owner(lock, other), flush=True)
        return WAIT
    # A producer may have completed between the first receipt check and claim.
    if receipt_valid(record, metadata_only=True):
        release(record['registry_locks'], record['token'])
        write_state(record, 'reused')
        return SKIP
    update(record['registry_locks'], owner)
    write_state(record, 'queued')
    return 0


def post(record, result):
    # A POST script may run even when the PRE script failed. Never touch another owner.
    current = json.loads(Path(record['state_file']).read_text()) if Path(record['state_file']).is_file() else {}
    if result == 0 and current.get('status') in {'complete', 'reused'}:
        return 0
    if current.get('status') == 'waiting':
        return result or 1
    # The AP owns the AFS reservation. Physical output locks are worker/user-side.
    owned = [p for p in record['registry_locks'] if Path(p).exists()
             and owner_at(p).get('token') == record['token']]
    if owned:
        owner = owner_at(owned[0])
        owner.update(status='failed', updated=stamp(), returncode=result)
        update(owned, owner)
    write_state(record, 'failed', returncode=result,
                error=current.get('error'), job_id=current.get('job_id'),
                detail='Worker did not publish verified completion; inspect node logs before recovery')
    return result or 1


@contextmanager
def stage_guard(config, stage):
    """Protect local and batch stage writers, including Python runtime exec()."""
    from .workflow_products import lock_paths
    locks = lock_paths(config, stage)
    token = os.environ.get('HGCAL_WORKFLOW_TOKEN') or os.environ.get('HGCAL_LOCAL_WRITER_TOKEN')
    if token is None:
        token = f'local-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}'
        os.environ['HGCAL_LOCAL_WRITER_TOKEN'] = token
    external = bool(os.environ.get('HGCAL_WORKFLOW_TOKEN'))
    owner = {'token': token, 'status': 'running', 'host': socket.gethostname(),
             'pid': os.getpid(), 'stage': stage, 'updated': stamp()}
    if external:
        for lock in locks:
            if not Path(lock).exists() or owner_at(lock).get('token') != token:
                raise ValueError(f'Workflow does not own this stage output: {lock}')
    else:
        ok, busy = claim(locks, owner)
        if not ok:
            raise ValueError(describe_owner(*busy))
    try:
        yield
    finally:
        if not external:
            release(locks, token)
