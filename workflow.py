#!/usr/bin/env python3
"""Plan and submit config-driven HGCal workflows; ordinary steer.py stays local."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys

from utils.workflow_registry import (lock_context, atomic_json, check_freeze, evidence, receipt_valid,
    claim, update, release, owner_at, write_state, stamp, load_record, intact, describe_owner)

ROOT = Path(__file__).resolve().parent


def worker(path, job_id):
    from utils.config import load_config
    from utils.workflow_products import probe, completed_files, output_paths
    record = load_record(path)
    check_freeze(record)
    for key, expected in record['runtime'].items():
        if os.environ.get(key) != expected:
            raise ValueError(f'Worker runtime differs from frozen {key}')
    owner = {'token': record['token'], 'status': 'running', 'work': record['work'],
             'node': record['node'], 'state_file': record['state_file'],
             'host': __import__('socket').gethostname(), 'pid': os.getpid(), 'job_id': job_id,
             'updated': stamp(), **lock_context(record)}
    os.environ['HGCAL_LOCK_CONTEXT'] = json.dumps(lock_context(record))
    for lock in record['registry_locks']:
        if not Path(lock).exists() or owner_at(lock).get('token') != record['token']:
            raise ValueError('Worker has no producer claim; submit through the generated DAG')
    update(record['registry_locks'], owner)
    write_state(record, 'running', job_id=job_id, host=owner['host'], pid=owner['pid'])
    try:
        ok, busy = claim(record['locks'], owner)
        if not ok:
            raise ValueError(describe_owner(*busy))
        update(record['locks'], owner)
        config = load_config(record['config'], steps=[record['stage']])
        if record.get('source_model_check'):
            from utils.model_compatibility import find,find_linear
            from utils.model_selection import read_model
            from utils.workflow_products import artifact_files
            for folder in record['source_model_check']:
                method=record.get('source_model_method','dnn')
                selected=find(config,Path(folder).name) if method=='dnn' else find_linear(config,method)[0]
                if str(selected)!=folder:
                    raise ValueError('Frozen source-model selection changed; prepare a new workflow')
                if method=='dnn':
                    read_model(Path(folder))
            complete,files=True,sorted(artifact_files(record['source_model_check']))
        else:
            complete, files = probe(config, record['stage'])
        if record['stage'] in {'plots', 'multimodule_plot'}:
            if receipt_valid(record):
                saved = json.loads(Path(record['receipt']).read_text())
                complete, files = True, [row['path'] for row in saved['evidence']]
            elif record.get('initial_complete') and intact(record['initial_evidence']):
                complete, files = True, [row['path'] for row in record['initial_evidence']]
        if record.get('initial_complete') and not complete:
            raise ValueError('A product verified complete during planning changed; prepare a new workflow to restore its prerequisites')
        if not complete:
            env = os.environ.copy()
            env['HGCAL_WORKFLOW_TOKEN'] = record['token']
            command = [sys.executable, str(ROOT/'steer.py'), '--config', record['config'], '--steps', record['stage']]
            print(f'Workflow {record["node"]}: executing {record["stage"]}', flush=True)
            result = subprocess.run(command, env=env)
            if result.returncode:
                raise RuntimeError(f'{record["stage"]} exited with status {result.returncode}; see node logs')
            # Identities may have changed after module IDs/calibrations were created.
            config = load_config(record['config'], steps=[record['stage']])
            files = completed_files(config, record['stage'])
        else:
            print(f'Workflow {record["node"]}: reusing verified existing products', flush=True)
        outputs=record.get('source_model_check') or output_paths(config,record['stage'])
        atomic_json(record['receipt'], {'request': record['request'], 'complete': True,
            'stage': record['stage'], 'work': record['work'], 'node': record['node'],
            'evidence': evidence(files), 'completed': stamp(), 'outputs': outputs})
        write_state(record, 'reused' if complete else 'complete', job_id=job_id,
                    outputs=outputs)
        release([*record['locks'], *record['registry_locks']], record['token'])
    except BaseException as error:
        owner.update(status='failed', updated=stamp(), error=str(error))
        owned = [p for p in [*record['locks'], *record['registry_locks']] if Path(p).exists()
                 and owner_at(p).get('token') == record['token']]
        update(owned, owner)
        write_state(record, 'failed', job_id=job_id, error=str(error))
        raise


def status(work, *, as_json=False):
    from collections import Counter
    from utils.workflow_condor import queued, scheduler_environment
    from utils.workflow_monitor import dagman_status, merge_node_state, workflow_result, display_node_state
    plan = json.loads((work/'plan.json').read_text())
    marker = work/'submission.json'
    submission = json.loads(marker.read_text()) if marker.is_file() else {'status': 'prepared'}
    native = dagman_status(work, submission)
    try:
        jobs = queued(work)
        scheduler_error = None
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        jobs, scheduler_error = [], str(error)
    names = {1: 'idle', 2: 'running', 3: 'removed', 4: 'completed', 5: 'held', 6: 'transferring', 7: 'suspended'}
    active = {row.get('HgcalWorkflowNode'): row for row in jobs}
    rows = []
    for node in plan['nodes']:
        path = Path(node['state_file'])
        row = json.loads(path.read_text()) if path.is_file() else {'node': node['node'], 'status': 'pending'}
        row.update(stage=node['stage'], logs=str(work/node['node']), parents=node['parents'])
        if node['node'] in active:
            job = active[node['node']]
            row.update(scheduler_status=names.get(job['JobStatus'], 'unknown'),
                       job_id=f'{job["ClusterId"]}.{job["ProcId"]}')
            if job.get('HoldReason'):
                row['hold_reason'] = job['HoldReason']
        rows.append(merge_node_state(row, native['nodes'].get(node['node'])))
    result = {'work': str(work), 'nodes': rows, 'scheduler_error': scheduler_error}
    result['submission'] = submission
    if submission.get('cluster_id') is not None and scheduler_error is None:
        query = subprocess.run(['condor_q', str(submission['cluster_id']), '-json', '-attributes',
                                'ClusterId,ProcId,JobStatus,HoldReason'], text=True, capture_output=True,
                               timeout=15, env=scheduler_environment(plan))
        result['dag_jobs'] = json.loads(query.stdout or '[]') if not query.returncode else []
        if query.returncode:
            result['scheduler_error'] = query.stderr.strip()
    result['dagman'] = native
    result['workflow_status'] = workflow_result(native, submission, jobs+result.get('dag_jobs', []))
    result['counts'] = dict(Counter(display_node_state(row) for row in rows))
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print(f'Workflow: {work}')
    print(f'Scheduler: {plan["scheduler"]["SCHEDD_HOST"]}')
    print(f'DAG submission: {submission["status"]}; job={submission.get("cluster_id", "not submitted")}')
    print(f'Workflow result: {result["workflow_status"]}')
    print('Stages: '+', '.join(f'{name}={count}' for name,count in sorted(result['counts'].items())))
    if native['metrics']:
        metrics=native['metrics']
        print(f'DAGMan exit: {metrics.get("exitcode")}; workers submitted: {metrics.get("jobs_submitted", "unknown")}')
    for warning in native['warnings']:
        print(f'Status note: {warning}')
    for job in result.get('dag_jobs', []):
        print(f'  DAG scheduler status: {names.get(job["JobStatus"], "unknown")}; {job.get("HoldReason", "")}')
    if result['scheduler_error']:
        print(f'Scheduler unavailable; showing last recorded state: {result["scheduler_error"]}')
    for row in rows:
        state = display_node_state(row)
        print(f'{row["node"]}: {state}; job={row.get("job_id", "not recorded")}')
        if row.get('lock'):
            print(f'  waiting on {row["lock"]}: {row.get("owner")}')
        if row.get('error') or row.get('hold_reason'):
            print(f'  {row.get("error") or row.get("hold_reason")}')
        for name in ('pre.log', 'post.log'):
            path=Path(row['logs'])/name
            if state=='failed' and path.is_file():
                print(f'  Hook log: {path}')
    print('Node logs: <workflow-directory>/<node>/job.<cluster>.<proc>.out/.err')
    print(f'DAG log: {work/"workflow.dag.dagman.out"}')


def release_failed(work, node, owner_stopped):
    from utils.workflow_condor import queued, scheduler_environment
    if not owner_stopped:
        raise ValueError('--release requires --owner-stopped: first stop/check the DAG and its workers')
    plan = json.loads((work/'plan.json').read_text())
    matches = [row for row in plan['nodes'] if row['node'] == node]
    if len(matches) != 1:
        raise ValueError('--node must identify one node in this submission')
    # A still-active DAG may submit a node after the current condor_q snapshot.
    # Refuse release while either the manager or any worker is still present.
    active = queued(work)
    submission = work/'submission.json'
    if submission.is_file():
        cluster = json.loads(submission.read_text()).get('cluster_id')
        if cluster:
            result = subprocess.run(['condor_q', str(cluster), '-json', '-attributes', 'ClusterId,JobStatus'],
                                    text=True, capture_output=True, timeout=15, env=scheduler_environment(plan))
            if result.returncode:
                raise RuntimeError('Cannot verify that the owning DAG stopped; refusing lock release')
            active += json.loads(result.stdout or '[]')
    if active:
        raise ValueError('Owning DAG/workers still exist in the queue. Stop them before releasing writer ownership.')
    record = matches[0]
    owned = [p for p in [*record['locks'], *record['registry_locks']] if Path(p).exists()
             and owner_at(p).get('token') == record['token']]
    release(owned, record['token'])
    write_state(record, 'released', detail='User explicitly confirmed the owner stopped')
    print(f'Released workflow producer ownership for {node}. Existing stage-specific locks are unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Workflow YAML referring to ordinary analysis YAMLs')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--show', action='store_true', help='Inspect dependencies/reuse; no processing or submission (default)')
    actions.add_argument('--prepare', action='store_true', help='Freeze configs and generate reviewable DAG/job files; do not submit')
    actions.add_argument('--submit', action='store_true', help='Reuse matching prepared files, or prepare them, then submit one workflow DAG')
    actions.add_argument('--submit-directory', type=Path, help='Submit a previously prepared directory')
    actions.add_argument('--status', nargs='?', const=True, type=Path,
                         help='Monitor the latest saved workflow with --config, or provide an explicit directory')
    actions.add_argument('--resume', type=Path, help='Resume a stopped DAG with its frozen configs and rescue information')
    actions.add_argument('--release', type=Path, help='Release a stopped producer after explicit owner confirmation')
    actions.add_argument('--clear-locks', action='store_true',
                         help='List indexed locks; add --owner-stopped to clear them after queue/PID checks')
    actions.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--refresh-plots', action='store_true', help='Render figures again; numerical caches remain reusable')
    parser.add_argument('--json', action='store_true', help='Print machine-readable plan/status')
    parser.add_argument('--node', help='Node name for --release')
    parser.add_argument('--owner-stopped', action='store_true', help='Confirm all affected owners stopped before releasing or clearing locks')
    parser.add_argument('--job-id', default='unknown', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.clear_locks:
            if args.refresh_plots or args.node:
                parser.error('--clear-locks accepts --config, --owner-stopped and --json only')
            from utils.workflow_cleanup import clear_locks
            clear_locks(args.config, owner_stopped=args.owner_stopped, as_json=args.json)
            return 0
        if args.worker:
            worker(args.worker.resolve(), args.job_id)
            return 0
        if args.status is True:
            if not args.config:
                parser.error('--status without a directory requires --config <workflow.yaml>')
            if args.refresh_plots or args.node or args.owner_stopped:
                parser.error('--status accepts --config and --json, not processing/recovery options')
            from utils.workflow_monitor import directory_for_config
            status(directory_for_config(args.config), as_json=args.json)
            return 0
        directory = args.status or args.resume or args.submit_directory or args.release
        if directory:
            if args.config or args.refresh_plots:
                parser.error('Directory actions use the frozen config; omit --config and --refresh-plots')
            directory = directory.resolve()
            if args.status:
                status(directory, as_json=args.json)
            elif args.release:
                release_failed(directory, args.node, args.owner_stopped)
            else:
                from utils.workflow_condor import submit
                submit(directory, resume=bool(args.resume))
            return 0
        if not args.config:
            parser.error('--config is required for planning, preparation and submission')
        if args.node or args.owner_stopped:
            parser.error('--node/--owner-stopped are only supported with --release')
        from utils.workflow_config import load
        from utils.workflow_plan import build, show
        from utils.workflow_condor import prepare, submit, find_preparation, preparation_lock
        workflow = load(args.config)
        context = preparation_lock(workflow, args.refresh_plots) if args.prepare or args.submit else nullcontext()
        with context:
            work = find_preparation(workflow, args.refresh_plots) if args.submit else None
            if work is not None:
                print(json.dumps({'prepared_directory': str(work)}) if args.json
                      else f'Using matching prepared workflow: {work}', flush=True)
            else:
                plan = build(workflow, refresh_plots=args.refresh_plots)
                if args.json:
                    print(json.dumps(plan, indent=2, default=str))
                else:
                    show(plan)
                if args.prepare or args.submit:
                    work = prepare(plan)
            if args.submit:
                submit(work)
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
