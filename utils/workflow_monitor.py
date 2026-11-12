"""Read frozen submissions and native DAGMan status without reloading analysis data."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NODE_STATES = {0: 'pending', 1: 'ready', 2: 'preparing', 3: 'queued',
               4: 'finalizing', 5: 'complete', 6: 'failed', 7: 'blocked'}


def directory_for_config(filename):
    """Prefer the latest submitted request, even if analysis settings changed."""
    import yaml
    from .config import UniqueKeyLoader, local_path
    path = Path(filename).expanduser().resolve()
    try:
        settings = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid workflow YAML: {error}') from None
    if not isinstance(settings, dict):
        raise ValueError('Workflow config must be a YAML mapping')
    base = local_path(settings.get('work_dir', str(ROOT/'workdir_condor'/'workflows')),
                      path.parent, 'work_dir')
    choices, unreadable = [], []
    for manifest in base.glob('*/plan.json'):
        try:
            plan = json.loads(manifest.read_text())
            if plan['workflow']['source'] != str(path):
                continue
            marker = manifest.parent/'submission.json'
            record = json.loads(marker.read_text()) if marker.is_file() else {}
            choices.append(((marker.is_file(), record.get('time', plan['created']),
                             plan['created'], str(manifest.parent)), manifest.parent))
        except (OSError, ValueError, KeyError, TypeError):
            unreadable.append(str(manifest))
    if not choices:
        detail = f' Unreadable saved plans: {", ".join(unreadable[:3])}.' if unreadable else ''
        raise ValueError(f'No saved workflow for {path} under {base}. Prepare/submit it first.'+detail)
    return max(choices, key=lambda row: row[0])[1]


def _classad_scalars(text):
    """Only consume documented scalar fields from DAGMan's node-status format.

    HTCondor Python bindings are not installed in the CMSSW runtime. This reader
    deliberately does not evaluate general ClassAd expressions or arbitrary code.
    """
    records = []
    for block in re.finditer(r'(?ms)^\[\s*\n(.*?)^\]\s*$', text):
        row = {}
        for field in re.finditer(r'(?m)^\s*(\w+)\s*=\s*("(?:\\.|[^"\\])*"|[+-]?\d+)\s*;', block[1]):
            token = field[2]
            if token.startswith('"'):
                try:
                    value = json.loads(token)
                except ValueError:
                    value = token[1:-1]
            else:
                value = int(token)
            row[field[1]] = value
        if row:
            records.append(row)
    return records


def dagman_status(work, submission):
    """Use node failures and final metrics even after jobs leave condor_q."""
    work = Path(work)
    result = {'dag': {}, 'nodes': {}, 'metrics': {}, 'warnings': []}
    metric_path = work/'workflow.dag.metrics'
    if metric_path.is_file():
        try:
            metrics = json.loads(metric_path.read_text())
            # Rescue submissions use another manager ID. Ignore an old result.
            if str(metrics.get('dagman_id')) == str(submission.get('cluster_id')):
                result['metrics'] = metrics
        except (OSError, ValueError):
            result['warnings'].append('DAG metrics are incomplete or unreadable')
    status_path = work/'nodes.status'
    if status_path.is_file():
        try:
            records = _classad_scalars(status_path.read_text())
            if not any(row.get('Type') == 'StatusEnd' for row in records):
                result['warnings'].append('DAG node status is being updated; retry shortly')
            else:
                dag = next((row for row in records if row.get('Type') == 'DagStatus'), {})
                started = submission.get('time')
                current = True
                if started and not result['metrics']:
                    # Whole seconds in native snapshots; allow a small submit/clock boundary.
                    current = dag.get('Timestamp', 0) >= datetime.fromisoformat(started).timestamp()-2
                if current:
                    result['dag'] = dag
                    result['nodes'] = {row['Node']: row for row in records
                                       if row.get('Type') == 'NodeStatus' and 'Node' in row}
        except (OSError, ValueError, TypeError):
            result['warnings'].append('DAG node status is unreadable')
    return result


def merge_node_state(row, native):
    if native is None:
        return row
    row['dag_status'] = NODE_STATES.get(native.get('NodeStatus'), 'unknown')
    row['dag_details'] = native.get('StatusDetails', '')
    code = native.get('NodeStatus')
    if code in {6, 7}:
        row['status'] = row['dag_status']
        if code == 6 and not row.get('error'):
            row['error'] = row['dag_details']
    elif code == 5:
        row['status'] = 'reused' if row.get('status') == 'reused' else 'complete'
    elif code in NODE_STATES and not (code in {2, 3, 4} and row.get('status') in {'waiting', 'queued', 'running'}):
        row['status'] = row['dag_status']
    return row


def workflow_result(native, submission, jobs):
    metrics = native['metrics']
    if metrics:
        return 'complete' if metrics.get('exitcode') == 0 else 'failed'
    state = native['dag'].get('DagStatus')
    if state in {5, 6}:
        return NODE_STATES[state]
    if any(job.get('JobStatus') in {1, 2, 5, 6, 7} for job in jobs):
        return 'active'
    return 'prepared' if submission.get('status') == 'prepared' else 'unknown'


def display_node_state(row):
    if row['status'] in {'failed', 'blocked', 'complete', 'reused', 'released'}:
        return row['status']
    return row.get('scheduler_status', row['status'])
