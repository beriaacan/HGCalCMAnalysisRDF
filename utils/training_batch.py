"""Reviewable CERN HTCondor jobs invoking the same configured local trainer."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import shlex
import shutil
import subprocess
import tempfile

import yaml

from .analytic_artifacts import sha
from .config import local_path, read_settings
from .dnn_training import CODE, destination

ROOT_DIR = Path(__file__).resolve().parents[1]
WORKER_CODE = tuple(sorted({str(path.relative_to(ROOT_DIR))
    for path in [*ROOT_DIR.glob('*.py'), *ROOT_DIR.glob('*.sh'), *(ROOT_DIR/'utils').glob('*.py'),
                 *(ROOT_DIR/'utils').glob('*.hxx')] if path.is_file()}))


def submit_path(path):
    # These submit fields consume a filename, not a ClassAd string literal:
    # surrounding quotes become part of initialdir on the installed HTCondor.
    value=str(path)
    if any(c in value for c in ('\n','\r','\x00')):
        raise ValueError('HTCondor paths cannot contain control characters')
    return value


def frozen_settings(config, modules):
    _, settings = read_settings(config.config_file)
    # Freeze resolved public blocks. Production selectors are replaced by exact
    # absolute patterns, so a worker cannot resolve another production or inherit
    # paths from the other campaign.
    for role in ('derivation','application'):
        row=settings[role]
        for name in ('nano_root','nano_production','pedestal_nano_production'):
            row.pop(name,None)
    for name in ('condor_proxy',):
        if settings[name] is not None:
            settings[name]=str(local_path(settings[name],config.config_file.parent,name))
    settings['steps'] = ['train']
    settings['cpus'] = config.training['condor_cpus']
    gpu = config.training['condor_gpus'] > 0
    chosen = config.training['training_device']
    if (gpu and chosen=='cpu') or (not gpu and chosen=='cuda'):
        raise ValueError('training_device conflicts with condor_gpus; use auto or matching resource settings')
    settings['training_device'] = 'cuda' if gpu else 'cpu'
    return settings


def validate_proxy(path):
    """Use the standard VOMS validator, before any condor_submit call."""
    if not path.is_file() or not os.access(path,os.R_OK):
        raise FileNotFoundError(f'X.509 proxy missing or unreadable: {path}. Renew it before submitting.')
    validator=shutil.which('voms-proxy-info')
    if validator is None:
        raise FileNotFoundError('voms-proxy-info is unavailable; activate CMSSW before submitting.')
    check=subprocess.run([validator,'-file',str(path),'-exists'],text=True,capture_output=True)
    lifetime=subprocess.run([validator,'-file',str(path),'-timeleft'],text=True,capture_output=True)
    try:
        seconds=int(lifetime.stdout.strip()) if lifetime.returncode==0 else 0
    except ValueError:
        seconds=0
    if check.returncode or seconds<=0:
        raise ValueError(f'Invalid or expired X.509 proxy: {path}. Renew it before submitting.')
    return seconds


def _credential(config):
    if not config.training['condor_use_x509_proxy']:
        return None
    path = config.training['condor_proxy']
    if path is None:
        path = Path(os.environ.get('X509_USER_PROXY',f'/tmp/x509up_u{os.getuid()}'))
    validate_proxy(path)
    return path.absolute()


def stage_proxy(proxy, work):
    """Remote CERN schedulers need an AFS copy, not the submit host's /tmp.

    AFS ACLs control access independently of Unix modes. Restrict the newly
    created credential directory before copying any credential bytes into it.
    Each submission owns its copy; later submissions cannot overwrite it.
    """
    if proxy is None:
        return None
    folder=work/'.credentials'
    folder.mkdir(mode=0o700)
    owner=pwd.getpwuid(os.getuid()).pw_name
    result=subprocess.run(['fs','setacl','-dir',str(folder),'-acl',owner,'all','-clear'],
                          text=True,capture_output=True)
    if result.returncode:
        folder.rmdir()
        raise ValueError('Cannot protect the proxy directory with an AFS ACL. '
                         'Use an AFS condor_work_dir and activate valid AFS credentials. '
                         +result.stderr.strip())
    target=folder/'x509up'
    descriptor=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(descriptor,'wb') as output, proxy.open('rb') as source:
            shutil.copyfileobj(source,output)
        validate_proxy(target)
    except BaseException:
        target.unlink(missing_ok=True)
        folder.rmdir()
        raise
    return target


def make_jobs(config, selected):
    setup = config.training['condor_setup_file']
    if not setup.is_file():
        raise FileNotFoundError(f'Cannot find the CMSSW batch setup script: {setup}')
    if not os.environ.get('CMSSW_BASE') or not os.environ.get('SCRAM_ARCH'):
        raise ValueError('Activate the CMSSW environment before preparing Condor jobs')
    proxy = _credential(config)
    settings = {group:frozen_settings(config,modules) for group,modules in selected.items()}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base = config.training['condor_work_dir']/config.model_dir.name
    base.mkdir(parents=True,exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=stamp+'-',dir=base))
    proxy = stage_proxy(proxy,work)
    code = {str(ROOT_DIR/name):sha(ROOT_DIR/name) for name in WORKER_CODE}
    identity = {'CMSSW_BASE':os.environ['CMSSW_BASE'],'SCRAM_ARCH':os.environ['SCRAM_ARCH']}
    jobs = []
    for group, modules in selected.items():
        folder=work/group
        folder.mkdir()
        (folder/'config.yaml').write_text(yaml.safe_dump(settings[group],sort_keys=False))
        record={'code_sha256':code,'runtime':identity,'configuration_sha256':sha(folder/'config.yaml'),
                'modules':modules,'model_destination':str(destination(config,group)),
                'proxy_file':str(proxy) if proxy is not None else None,
                'source_config':str(config.config_file)}
        (folder/'job.json').write_text(json.dumps(record,indent=2)+'\n')
        # This worker is self-contained: verify the immutable request before
        # importing the shared checkout, then exec the standard entry point.
        worker = '''import hashlib,json,os,sys
from pathlib import Path
folder=Path(__file__).resolve().parent
record=json.loads((folder/'job.json').read_text())
for key,value in record['runtime'].items():
    if os.environ.get(key)!=value: raise RuntimeError(f"CMSSW runtime changed: {key}")
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if digest(folder/'config.yaml')!=record['configuration_sha256']:
    raise RuntimeError("Submitted configuration changed; prepare a new job")
for path,expected in record['code_sha256'].items():
    if digest(path)!=expected: raise RuntimeError(f"Training code changed: {path}; prepare a new job")
os.execv(sys.executable,[sys.executable,STEER,'--config',str(folder/'config.yaml'),'--steps','train','--training-group',GROUP])
'''.replace('STEER',repr(str(ROOT_DIR/'steer.py'))).replace('GROUP',repr(group))
        (folder/'worker.py').write_text(worker)
        envdir=config.training['training_environment']
        python=envdir/'bin/python3'
        # Resolve Python ABI from the activated CMSSW Python at submission time.
        import sys
        site=envdir/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        # Bootstrap from the recorded CMSSW release even when a custom setup
        # file only defines site helpers. No private shell function is required.
        bootstrap = ROOT_DIR/'setup.sh'
        wrapper=['#!/usr/bin/env bash',
                 f'source {shlex.quote(str(bootstrap))} {shlex.quote(identity["CMSSW_BASE"])} '
                 f'{shlex.quote(identity["SCRAM_ARCH"])} || exit $?']
        if setup != bootstrap:
            wrapper.append(f'source {shlex.quote(str(setup))} || exit $?')
        wrapper += ['set -euo pipefail',
                 f'export PYTHONPATH={shlex.quote(str(site))}:"${{PYTHONPATH:-}}"',
                 f'export HGCAL_TRAINING_ENV={shlex.quote(str(envdir))}',
                 'export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HGCAL_BATCH=1']
        if config.training['condor_gpus']:
            wrapper.append('export HGCAL_REQUIRE_CUDA=1')
        else:
            wrapper.append('export CUDA_VISIBLE_DEVICES=""')
        wrapper.append(f'exec {shlex.quote(str(python))} {shlex.quote(str(folder/"worker.py"))}')
        executable=folder/'run.sh'
        executable.write_text('\n'.join(wrapper)+'\n')
        executable.chmod(0o700)
        lines=['universe = vanilla',f'executable = {submit_path(executable)}',
            f'initialdir = {submit_path(folder)}','getenv = False','should_transfer_files = NO',
            f'output = {submit_path(folder/"job.$(ClusterId).$(ProcId).out")}',
            f'error = {submit_path(folder/"job.$(ClusterId).$(ProcId).err")}',
            f'log = {submit_path(folder/"job.$(ClusterId).log")}',
            f'request_cpus = {config.training["condor_cpus"]}',
            f'request_gpus = {config.training["condor_gpus"]}',
            f'request_memory = {config.training["condor_memory_gb"]} GB',
            f'+JobFlavour = {json.dumps(config.training["condor_job_flavour"])}',
            '+SendCredential = True', 'max_retries = 0']
        if proxy:
            lines += ['use_x509userproxy = True',f'x509userproxy = {submit_path(proxy)}']
        if config.training['condor_gpus']:
            lines += [f'gpus_minimum_runtime = {config.training["condor_minimum_cuda_runtime"]}']
        lines += ['queue 1']
        submit=folder/'job.sub'
        submit.write_text('\n'.join(lines)+'\n')
        jobs.append({'group':group,'directory':str(folder),'submit_file':str(submit)})
    (work/'manifest.json').write_text(json.dumps({'jobs':jobs,'submitted':False},indent=2)+'\n')
    return work,jobs


def submit_jobs(work,jobs):
    # Check every prepared job first: one bad proxy must prevent a partial
    # submission of the batch. Recheck immediately before each submit too.
    proxies=[]
    for job in jobs:
        folder=Path(job['directory'])
        record=json.loads((folder/'job.json').read_text())
        settings=yaml.safe_load((folder/'config.yaml').read_text())
        proxy=record.get('proxy_file')
        if settings['condor_use_x509_proxy']:
            if not isinstance(proxy,str) or not proxy:
                raise ValueError(f'Prepared job has no staged proxy path: {folder}. Prepare it again.')
            validate_proxy(Path(proxy))
        proxies.append(Path(proxy) if settings['condor_use_x509_proxy'] else None)
    records=[]
    for job,proxy in zip(jobs,proxies):
        if proxy is not None:
            validate_proxy(proxy)
        result=subprocess.run(['condor_submit','-terse',job['submit_file']],text=True,capture_output=True)
        row=dict(job,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        records.append(row)
        (work/'submission.json').write_text(json.dumps(records,indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'condor_submit failed for {job["group"]}: {result.stderr.strip()}; records: {work}')
        print(f'Submitted {job["group"]}: {result.stdout.strip()}',flush=True)
    (work/'manifest.json').write_text(json.dumps({'jobs':records,'submitted':True},indent=2)+'\n')
