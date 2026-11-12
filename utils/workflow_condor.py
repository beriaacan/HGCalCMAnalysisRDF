"""Freeze reviewable stage jobs and a native DAGMan graph; no nested submission."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import uuid

import yaml

from .config import load_config, local_path
from .workflow_products import from_settings
from .workflow_registry import atomic_json, digest, evidence, check_freeze, stamp, intact, load_record
from .training_batch import ROOT_DIR, WORKER_CODE, submit_path, _credential, stage_proxy, validate_proxy


def preparation_request(workflow, refresh_plots=False):
    return digest({'workflow': workflow, 'refresh_plots': refresh_plots,
                   'runtime': {key: os.environ.get(key) for key in ('CMSSW_BASE', 'SCRAM_ARCH')}})


def preparation_index(workflow, refresh_plots=False):
    return Path(workflow['work_dir'])/'.prepared'/(preparation_request(workflow, refresh_plots)+'.json')


@contextmanager
def preparation_lock(workflow, refresh_plots=False):
    """Serialize prepare/submit for one request, including callers in two shells."""
    import fcntl
    path = preparation_index(workflow, refresh_plots).with_suffix('.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Waiting for another prepare/submit operation for this request to finish...', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def remember_preparation(work, plan):
    """Publish the lookup only after every job and its plan are closed."""
    files = [work/'plan.json', work/'workflow.dag', work/'hook.py']
    for node in plan['nodes']:
        files.extend(work/node['name']/name for name in ('config.yaml', 'job.json', 'run.sh', 'job.sub'))
    rows = evidence(files)
    for row in rows:
        if 'sha256' not in row:
            row['sha256'] = hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
    atomic_json(preparation_index(plan['workflow'], plan['refresh_plots']), {
        'schema': 'hgcal_cm_preparation_index_v1', 'directory': str(work),
        'request': preparation_request(plan['workflow'], plan['refresh_plots']),
        'files': rows, 'created': stamp()})


def find_preparation(workflow, refresh_plots=False):
    """Find the last identical request without choosing an unrelated latest DAG."""
    index = preparation_index(workflow, refresh_plots)
    if not index.is_file():
        return None
    saved = json.loads(index.read_text())
    request = preparation_request(workflow, refresh_plots)
    if saved.get('schema') != 'hgcal_cm_preparation_index_v1' or saved.get('request') != request:
        raise ValueError(f'Invalid preparation index: {index}; inspect it and run --prepare explicitly')
    work = Path(saved['directory']).resolve()
    if work.parent != Path(workflow['work_dir']).resolve() or not (work/'plan.json').is_file():
        raise ValueError(f'Recorded preparation is missing or moved: {work}. Check its submission '
                         'status before creating another request with --prepare.')
    # submit() reports the existing attempt before any new scheduler call. Never
    # classify a submitted/uncertain request as stale and silently duplicate it.
    if (work/'submission.json').exists():
        return work
    try:
        if not intact(saved.get('files', [])):
            raise ValueError('generated submission files changed')
        frozen = json.loads((work/'plan.json').read_text())
        if (frozen.get('request') != request or frozen.get('workflow') != workflow
                or frozen.get('refresh_plots') != refresh_plots):
            raise ValueError('resolved request changed')
        current_code = {str(ROOT_DIR/name) for name in WORKER_CODE}
        for node in frozen['nodes']:
            if not current_code <= {row['path'] for row in node['frozen_files']}:
                raise ValueError('runtime source inventory changed')
            load_record(work/node['node']/'job.json')
            check_freeze(node)
            if node['initial_complete'] and not intact(node['initial_evidence']):
                raise ValueError('a product reused during planning changed')
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Previous preparation is no longer current: {work}\n  {error}\n'
              'Preparing a fresh request; the previous directory is retained.', flush=True)
        return None
    return work


def scheduler():
    """Pin the schedd used for submit/status/recovery, even from another shell."""
    result = subprocess.run(['condor_config_val', 'SCHEDD_HOST'], text=True,
                            capture_output=True, timeout=20)
    value = result.stdout.strip()
    if result.returncode or not value or any(c.isspace() for c in value):
        raise ValueError('Cannot resolve one HTCondor SCHEDD_HOST. Activate the normal CERN '
                         'batch setup or set _CONDOR_SCHEDD_HOST explicitly before preparation. '
                         +result.stderr.strip())
    return {'SCHEDD_HOST': value}


def scheduler_environment(plan):
    env = os.environ.copy()
    env.update({'_CONDOR_'+key: value for key, value in plan['scheduler'].items()})
    return env


def inventories(config, *, pedestal=False):
    rows = {}
    from .datasets import recipes
    for recipe in (*recipes(config), config.target):
        sources = recipe.pedestal.sources if pedestal else recipe.dataset.sources
        for source in sources:
            files = sorted(str(Path(p).resolve()) for p in glob.glob(source.input_pattern) if Path(p).is_file())
            if not files:
                raise FileNotFoundError(f'No pinned Nano inputs: {source.input_pattern}')
            if len({str(Path(p).parent) for p in files}) != 1:
                raise ValueError(f'Multiple Nano productions selected: {source.input_pattern}')
            rows[source.input_pattern] = {'pattern': source.input_pattern, 'files': files,
                                           'evidence': evidence(files)}
    return list(rows.values())


def fixed_files(config):
    files = {ROOT_DIR/name for name in WORKER_CODE}
    files.update((config.dnn['cell_area_file'], config.correction.module_map, config.target.module_map,
                  config.training['condor_setup_file']))
    from .datasets import recipes
    for row in (*recipes(config), config.target):
        files.add(row.module_map)
        if row.calibration_file.is_file():
            files.add(row.calibration_file)
    central = ROOT_DIR.parents[1]
    files.update((central/'LocalCalibration/python').glob('*.py'))
    files.update(config.multimodule['multimodule_geometry_dir'].glob('*.json'))
    environment = config.training['training_environment']
    files.update(p for p in (environment/'pyvenv.cfg', environment/'runtime.json',
        environment/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/torch/version.py') if p.is_file())
    return sorted(p for p in files if p.is_file())


def selected_ids(config):
    from .prepared import derivation_modules
    path = config.dnn['module_registry_file']
    values = json.loads(path.read_text())['modules'] if path.is_file() else {}
    names = derivation_modules(config)
    return [{'path': str(path), 'values': {name: values[name] for name in names if name in values}}]


def prepare(plan):
    workflow = plan['workflow']
    if not os.environ.get('CMSSW_BASE') or not os.environ.get('SCRAM_ARCH'):
        raise ValueError('Activate the CMSSW setup before preparing workflow jobs')
    selected_scheduler = scheduler()
    base = Path(workflow['work_dir'])
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-'), dir=base))
    # Native DAG scripts refer to absolute paths; forbid characters DAGMan cannot safely tokenize.
    if any(c in str(work) for c in ('\n', '\r', '\x00', '"')):
        raise ValueError('Workflow work_dir cannot contain quotes or control characters')
    workflow_id = uuid.uuid4().hex
    runtime = {key: os.environ[key] for key in ('CMSSW_BASE', 'SCRAM_ARCH')}
    code = evidence([ROOT_DIR/name for name in WORKER_CODE])
    code_key = digest(code)
    records = []
    staged = {}
    for node in plan['nodes']:
        config = from_settings(node['settings'])
        folder = work/node['name']
        folder.mkdir()
        settings = deepcopy(node['settings'])
        settings['steps'] = [node['stage']]
        settings['cpus'] = node['resources']['cpus']
        if node['stage'] == 'train':
            settings['training_device'] = 'cuda' if node['resources']['gpus'] else 'cpu'
        if node['stage'] == 'apply' and 'dnn' in config.methods:
            settings['application']['device'] = 'cuda' if node['resources']['gpus'] else 'cpu'
        if node['stage'] in {'train', 'apply'} and (node['stage'] == 'train' or 'dnn' in config.methods):
            from .training_runtime import environment
            environment(config)  # Validate interpreter existence; never log its environment.
        if not config.training['condor_setup_file'].is_file():
            raise FileNotFoundError(f'Missing batch setup: {config.training["condor_setup_file"]}')
        proxy = _credential(config)
        if proxy is not None:
            if str(proxy) not in staged:
                # One protected credential copy per distinct proxy.
                holder = work/f'credential_{len(staged)}'
                holder.mkdir()
                staged[str(proxy)] = stage_proxy(proxy, holder)
            proxy = staged[str(proxy)]
        path = folder/'config.yaml'
        path.write_text(yaml.safe_dump(settings, sort_keys=False))
        coverage = settings['derivation']['modules'] if node['stage'] == 'prepare' else None
        request = digest({'recipe': node['request'], 'coverage': coverage, 'code': code_key})
        if plan['refresh_plots'] and node['stage'] in {'plots', 'multimodule_plot'}:
            request = digest({'request': request, 'refresh': workflow_id})
        record = {'schema': plan['schema'], 'node': node['name'], 'stage': node['stage'],
                  'source_model_check':node.get('source_model_check'),
                  'source_model_method':node.get('source_model_method'),
                  'request': request, 'work': str(work), 'workflow_id': workflow_id,
                  'token': workflow_id+'/'+node['name'], 'config': str(path),
                  'state_file': str(work/node['name']/'state.json'),
                  'receipt': str(Path(workflow['state_dir'])/'products'/f'{request}.json'),
                  'locks': node['locks'], 'scheduler': selected_scheduler, 'runtime': runtime, 'resources': node['resources'],
                  'registry_locks': [str(Path(workflow['state_dir'])/'claims'/digest(lock)) for lock in node['locks']],
                  'frozen_files': evidence([*fixed_files(config), path]),
                  'inventories': ([] if node['stage'] == 'pedestals' and node['complete'] else
                                  inventories(config, pedestal=node['stage'] == 'pedestals')),
                  'module_ids': [] if node['stage'] == 'pedestals' else selected_ids(config),
                  'initial_complete': node['complete'],
                  'initial_evidence': evidence(node['files']) if node['complete'] else [],
                  'proxy_file': str(proxy) if proxy else None,
                  'parents': node['parents'], 'sources': node['sources'],
                  'outputs': node['outputs'],
                  'source_settings': node['settings']}
        atomic_json(folder/'job.json', record)
        write_worker(folder, record, config)
        write_submit(folder, record, config)
        records.append(record)
    write_hook(work)
    write_dag(work, records, workflow)
    atomic_json(work/'plan.json', {'schema': plan['schema'], 'workflow_id': workflow_id,
        'workflow': workflow, 'goals': plan['goals'], 'nodes': records, 'created': stamp(),
        'scheduler': selected_scheduler, 'refresh_plots': plan['refresh_plots'],
        'request': preparation_request(workflow, plan['refresh_plots'])})
    remember_preparation(work, plan)
    print(f'Workflow files ready: {work}', flush=True)
    print('No jobs submitted. Inspect workflow.dag, plan.json and each config.yaml/job.sub.', flush=True)
    return work


def write_dag(work, records, workflow):
    lines = ['# Generated by workflow.py; execute train inside the graph, never submit_train.',
             'NODE_STATUS_FILE nodes.status 30 ALWAYS-UPDATE']
    for record in records:
        node = record['node']
        # DAGMan SCRIPT arguments are not shell-quoted. Relative paths composed
        # of generated node names avoid literal quotes and whitespace splitting.
        lines += [f'JOB {node} {node}/job.sub',
                  f'SCRIPT DEFER 75 30 DEBUG {node}/pre.log ALL PRE {node} '
                  f'/usr/bin/python3 -B -E hook.py pre {node}/job.json',
                  f'PRE_SKIP {node} 99',
                  f'SCRIPT DEBUG {node}/post.log ALL POST {node} '
                  f'/usr/bin/python3 -B -E hook.py post {node}/job.json $RETURN',
                  f'RETRY {node} {workflow["retries"]}',
                  f'CATEGORY {node} {"GPU" if record["resources"]["gpus"] else "IO"}']
        if record['parents']:
            lines.append(f'PARENT {" ".join(record["parents"])} CHILD {node}')
    lines += [f'MAXJOBS IO {workflow["max_io_jobs"]}', f'MAXJOBS GPU {workflow["max_gpu_jobs"]}']
    (work/'workflow.dag').write_text('\n'.join(lines)+'\n')


def write_worker(folder, record, config):
    bootstrap = ROOT_DIR/'setup.sh'
    setup = config.training['condor_setup_file']
    runtime = record['runtime']
    lines = ['#!/usr/bin/env bash',
             f'source {shlex.quote(str(bootstrap))} {shlex.quote(runtime["CMSSW_BASE"])} '
             f'{shlex.quote(runtime["SCRAM_ARCH"])} || exit $?']
    if setup != bootstrap:
        lines.append(f'source {shlex.quote(str(setup))} || exit $?')
    lines += ['set -euo pipefail', 'export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HGCAL_BATCH=1']
    python = Path(sys.executable)
    uses_torch = record['stage'] == 'train' or record['stage'] == 'apply' and 'dnn' in config.methods
    if uses_torch:
        environment = config.training['training_environment']
        python = environment/'bin/python3'
        site = environment/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        lines += [f'export PYTHONPATH={shlex.quote(str(site))}:"${{PYTHONPATH:-}}"',
                  f'export HGCAL_TRAINING_ENV={shlex.quote(str(environment))}']
    if record['resources']['gpus']:
        if record['stage'] == 'train':
            lines.append('export HGCAL_REQUIRE_CUDA=1')
    else:
        lines.append('export CUDA_VISIBLE_DEVICES=""')
    lines += [f'exec {shlex.quote(str(python))} {shlex.quote(str(ROOT_DIR/"workflow.py"))} '
              f'--worker {shlex.quote(str(folder/"job.json"))} --job-id "$1"']
    target = folder/'run.sh'
    target.write_text('\n'.join(lines)+'\n')
    target.chmod(0o700)


def write_submit(folder, record, config):
    row = record['resources']
    lines = ['universe = vanilla', f'executable = {submit_path(folder/"run.sh")}',
             f'initialdir = {submit_path(folder)}', 'getenv = False', 'should_transfer_files = NO',
             'arguments = "$(ClusterId).$(ProcId)"',
             f'output = {submit_path(folder/"job.$(ClusterId).$(ProcId).out")}',
             f'error = {submit_path(folder/"job.$(ClusterId).$(ProcId).err")}',
             f'log = {submit_path(folder/"job.$(ClusterId).log")}',
             f'request_cpus = {row["cpus"]}', f'request_gpus = {row["gpus"]}',
             f'request_memory = {row["memory_gb"]} GB', f'+JobFlavour = {json.dumps(row["job_flavour"])}',
             '+SendCredential = True', 'max_retries = 0',
             f'+HgcalWorkflow = {json.dumps(record["workflow_id"])}',
             f'+HgcalWorkflowNode = {json.dumps(record["node"])}']
    if record['proxy_file']:
        lines += ['use_x509userproxy = True', f'x509userproxy = {submit_path(record["proxy_file"])}']
    if row['gpus']:
        lines.append(f'gpus_minimum_runtime = {config.training["condor_minimum_cuda_runtime"]}')
    (folder/'job.sub').write_text('\n'.join(lines+['queue 1'])+'\n')


def write_hook(work):
    # Runs on the AP with system Python: no CMSSW, YAML, torch or ROOT imports.
    source = '''import json,sys
from pathlib import Path
sys.path.insert(0, ROOT)
from utils.workflow_registry import pre,post,load_record
try:
    record=load_record(sys.argv[2])
    result=pre(record) if sys.argv[1]=='pre' else post(record,int(sys.argv[3]))
except Exception as error:
    print(str(error),file=sys.stderr,flush=True)
    result=1
raise SystemExit(result)
'''.replace('ROOT', repr(str(ROOT_DIR)))
    (work/'hook.py').write_text(source)


def queued(work):
    plan = json.loads((Path(work)/'plan.json').read_text())
    result = subprocess.run(['condor_q', '-json', '-constraint',
        f'HgcalWorkflow == "{plan["workflow_id"]}"', '-attributes',
        'ClusterId,ProcId,JobStatus,HgcalWorkflowNode,HoldReason'], capture_output=True, text=True,
        timeout=15, env=scheduler_environment(plan))
    if result.returncode:
        raise RuntimeError(f'Cannot query scheduler: {result.stderr.strip()}')
    return json.loads(result.stdout or '[]')


def submit(work, *, resume=False):
    work = Path(work).resolve()
    plan = json.loads((work/'plan.json').read_text())
    marker = work/'submission.json'
    marker_before = marker.read_bytes() if marker.is_file() else None
    if marker.exists() and not resume:
        raise ValueError('This directory already has a submission attempt; use --status and --resume, not a second submit')
    if resume:
        if not marker.is_file():
            raise ValueError('No previous submission to resume; use --submit-directory')
        previous = json.loads(marker.read_text())
        if previous.get('status') in {'submitting', 'unknown'}:
            raise ValueError('Previous submission outcome is unknown. Inspect condor_q and submission.json before recovery; refusing duplicate DAG')
        active = queued(work)
        cluster = previous.get('cluster_id')
        if cluster is not None:
            result = subprocess.run(['condor_q', str(cluster), '-json', '-attributes', 'ClusterId,JobStatus'],
                                    text=True, capture_output=True, timeout=15, env=scheduler_environment(plan))
            if result.returncode:
                raise RuntimeError('Cannot establish whether the previous DAG is still active')
            active += json.loads(result.stdout or '[]')
        if active:
            raise ValueError('The previous DAG or its workers are still queued/running/held; do not submit a second copy')
    for record in plan['nodes']:
        check_freeze(record)
        if record['proxy_file']:
            validate_proxy(Path(record['proxy_file']))
    command = ['condor_submit_dag', '-maxjobs', str(plan['workflow']['max_jobs']),
               '-maxpre', str(plan['workflow']['max_jobs']), '-append', '+SendCredential = True',
               '-append', f'+HgcalWorkflow = {json.dumps(plan["workflow_id"])}']
    if resume:
        command.append('-force')
    command.append('workflow.dag')
    with (work/'.submit-lock').open('a') as lock:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (marker.read_bytes() if marker.is_file() else None) != marker_before:
            raise ValueError('Submission state changed during preflight; inspect --status before retrying')
        # Recheck after the lock to prevent two callers submitting the same prepared DAG.
        if marker.exists() and not resume:
            raise ValueError('Another caller already attempted submission of this directory')
        atomic_json(marker, {'status': 'submitting', 'command': command, 'time': stamp()})
        result = subprocess.run(command, cwd=work, text=True, capture_output=True, env=scheduler_environment(plan))
        match = re.search(r'submitted to cluster\s+(\d+)', result.stdout, re.IGNORECASE)
        record = {'status': 'submitted' if result.returncode == 0 and match else 'unknown',
                  'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
                  'cluster_id': int(match.group(1)) if match else None, 'time': stamp()}
        atomic_json(marker, record)
    print(result.stdout, end='', flush=True)
    if record['status'] != 'submitted':
        raise RuntimeError(f'DAG submission did not return a confirmed job ID; inspect {marker} and condor_q before retrying. {result.stderr}')
    config=Path(plan['workflow']['source'])
    try:
        config=config.relative_to(ROOT_DIR)
    except ValueError:
        pass
    print(f'Status: ./workflow.py --config {shlex.quote(str(config))} --status', flush=True)
    return record
