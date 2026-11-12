"""Recover indexed abandoned writer locks without requiring job work directories."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess

from .workflow_registry import digest, owner_at, release

ROOT = Path(__file__).resolve().parents[1]


def state_directory(config=None):
    if config is None:
        return ROOT/'.workflow_state'
    import yaml
    from .config import UniqueKeyLoader, local_path
    path = Path(config).resolve()
    settings = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    if not isinstance(settings, dict):
        raise ValueError('Workflow config must be a mapping')
    return local_path(settings.get('state_dir', str(ROOT/'.workflow_state')),
                      path.parent, 'state_dir')


def indexed_locks(state):
    """Ignore completed/deleted locks; never infer that an unknown owner stopped."""
    result = {}
    for path in sorted((state/'lock_catalog').glob('*.json')):
        record = json.loads(path.read_text())
        lock = Path(record['path'])
        if not lock.exists():
            continue
        current = owner_at(lock)
        if current != record['owner']:
            raise ValueError(f'Lock owner changed since its index was written: {lock}; inspect before cleanup')
        result[str(lock.resolve())] = {'path': str(lock), 'owner': current, 'index': str(path), 'context': record.get('context', {})}
    # A claim interrupted before index publication must not silently disappear.
    for lock in sorted((state/'claims').glob('*')):
        if lock.is_dir() and str(lock.resolve()) not in result:
            raise ValueError(f'Unindexed workflow claim: {lock}; inspect its owner before cleanup')
    return list(result.values())


def ensure_stopped(rows):
    schedulers = {}
    for row in rows:
        owner = {**row.get('context', {}), **row['owner']}
        if not owner.get('token'):
            raise ValueError(f'Unknown lock owner: {row["path"]}')
        workflow = owner.get('workflow_id')
        if workflow:
            scheduler = owner.get('scheduler')
            if not scheduler:
                raise ValueError(f'No scheduler recorded for {row["path"]}; inspect before cleanup')
            schedulers.setdefault(json.dumps(scheduler, sort_keys=True), set()).add(workflow)
        elif owner.get('work'):
            raise ValueError(f'Incomplete workflow ownership: {row["path"]}')
        host, pid = owner.get('host'), owner.get('pid')
        if host and pid and host.split('.')[0] == socket.gethostname().split('.')[0]:
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                pass
            except PermissionError:
                raise ValueError(f'Cannot verify local PID {pid} stopped: {row["path"]}') from None
            else:
                raise ValueError(f'Local owner PID {pid} is still running: {row["path"]}')
    for encoded, workflows in schedulers.items():
        env = os.environ.copy()
        env.update({'_CONDOR_'+k:v for k,v in json.loads(encoded).items()})
        constraint = ' || '.join(f'HgcalWorkflow =?= {json.dumps(w)}' for w in sorted(workflows))
        query = subprocess.run(['condor_q', '-constraint', constraint, '-json', '-attributes',
                                'ClusterId,ProcId,JobStatus,HgcalWorkflow'],
                               text=True, capture_output=True, timeout=20, env=env)
        if query.returncode:
            raise RuntimeError('Cannot verify stopped DAGs/workers; no locks removed: '+query.stderr.strip())
        active = json.loads(query.stdout or '[]')
        if active:
            raise ValueError('DAGs/workers still exist in the queue; no locks removed: '
                             + ', '.join(f'{r["ClusterId"]}.{r["ProcId"]}' for r in active))


def clear_locks(config=None, *, owner_stopped=False, as_json=False):
    state = state_directory(config)
    rows = indexed_locks(state)
    if owner_stopped:
        ensure_stopped(rows)
        # Validate the complete set before removing any directory or owner record.
        for row in rows:
            lock = Path(row['path'])
            if owner_at(lock) != row['owner'] or {p.name for p in lock.iterdir()} != {'owner.json'}:
                raise ValueError(f'Lock changed or contains unexpected files: {lock}; no locks removed')
        for row in rows:
            # release checks the ownership token again immediately before removal.
            release([row['path']], row['owner']['token'])
            Path(row['index']).unlink(missing_ok=True)
    report = {'state_dir': str(state), 'removed': owner_stopped, 'count': len(rows),
              'locks': [{'path':r['path'], 'owner':r['owner']} for r in rows]}
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        for row in rows:
            owner = row['owner']
            print(f'{row["path"]}\n  owner: {owner.get("work", owner.get("host"))}; '
                  f'node={owner.get("node", "local")}; pid={owner.get("pid")}')
        print(f'{"Removed" if owner_stopped else "Found"} {len(rows)} lock directories.')
        if not owner_stopped and rows:
            print('Preview only. After stopping all affected owners, rerun with --owner-stopped.')
    return report
