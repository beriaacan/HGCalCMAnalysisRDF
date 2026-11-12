"""Small scheduler recipes referring to the ordinary scientific analysis YAMLs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from .config import load_config, read_settings, local_path, UniqueKeyLoader
from .steps import validate_steps

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_KEYS = {'cpus', 'gpus', 'memory_gb', 'job_flavour'}
WORKFLOW_KEYS = {'analyses', 'work_dir', 'state_dir', 'max_jobs', 'max_io_jobs',
                 'max_gpu_jobs', 'retries', 'resources'}


def positive(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def load(path):
    import yaml
    path = Path(path).resolve()
    try:
        value = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid workflow YAML {path}: {error}') from None
    if not isinstance(value, dict) or set(value) - WORKFLOW_KEYS:
        raise ValueError(f'Workflow must be a mapping; supported options: {sorted(WORKFLOW_KEYS)}')
    entries = value.get('analyses')
    if not isinstance(entries, list) or not entries:
        raise ValueError('analyses must be a non-empty list of config/steps mappings')
    analyses = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {'config', 'steps'} or 'config' not in entry:
            raise ValueError('Each analyses entry accepts config and optional steps only')
        config_path = local_path(entry['config'], path.parent, 'config')
        config = load_config(config_path, steps=entry.get('steps'))
        goals = validate_steps(list(config.steps))
        if 'submit_train' in goals:
            raise ValueError('Workflow goals use train, not submit_train; DAGMan submits the training job')
        _, settings = read_settings(config_path)
        if settings.get('condor_proxy') is not None:
            settings['condor_proxy'] = str(local_path(settings['condor_proxy'], config_path.parent, 'condor_proxy'))
        analyses.append({'source': str(config_path), 'settings': freeze(settings), 'steps': list(goals)})
    resources = value.get('resources', {})
    if not isinstance(resources, dict):
        raise ValueError('resources must be a mapping from step names to resource mappings')
    for stage, row in resources.items():
        validate_steps([stage])
        if stage == 'submit_train' or not isinstance(row, dict) or set(row) - RESOURCE_KEYS:
            raise ValueError(f'Invalid resource settings for {stage}')
        for key in ('cpus', 'gpus', 'memory_gb'):
            if key in row:
                positive(row[key], f'resources.{stage}.{key}', 0 if key == 'gpus' else 1)
        if 'job_flavour' in row and (not isinstance(row['job_flavour'], str)
                or not re.fullmatch(r'[A-Za-z0-9_]+', row['job_flavour'])):
            raise ValueError(f'Invalid job_flavour for {stage}')
    result = {'source': str(path), 'analyses': analyses, 'resources': resources,
              'work_dir': str(local_path(value.get('work_dir', str(ROOT/'workdir_condor'/'workflows')),
                                         path.parent, 'work_dir')),
              'state_dir': str(local_path(value.get('state_dir', str(ROOT/'.workflow_state')),
                                          path.parent, 'state_dir'))}
    for key, default in (('max_jobs', 8), ('max_io_jobs', 2), ('max_gpu_jobs', 1), ('retries', 0)):
        result[key] = positive(value.get(key, default), key, 0 if key == 'retries' else 1)
    if result['max_gpu_jobs'] > result['max_jobs'] or result['max_io_jobs'] > result['max_jobs']:
        raise ValueError('max_io_jobs and max_gpu_jobs must not exceed max_jobs')
    return result


def freeze(settings):
    """read_settings has already resolved defaults and YAML-relative paths."""
    value = deepcopy(settings)
    from .datasets import all_blocks
    for row in all_blocks(value):
        for key in ('nano_root', 'nano_production', 'pedestal_nano_production'):
            row.pop(key, None)
    if value.get('condor_proxy') is not None:
        # The caller replaces this with the resolved path before submission.
        value['condor_proxy'] = str(Path(value['condor_proxy']).expanduser())
    return value


def resources(workflow, stage, config):
    gpu = 0
    if stage == 'train':
        gpu = config.training['condor_gpus']
    elif stage == 'apply' and 'dnn' in config.methods and config.application['apply_device'] != 'cpu':
        gpu = config.training['condor_gpus']
    row = {'cpus': config.training['condor_cpus'], 'gpus': gpu,
           'memory_gb': config.training['condor_memory_gb'],
           'job_flavour': config.training['condor_job_flavour']}
    row.update(workflow['resources'].get(stage, {}))
    choice = config.training['training_device'] if stage == 'train' else config.application['apply_device']
    if stage in {'train', 'apply'} and (stage == 'train' or 'dnn' in config.methods):
        if (row['gpus'] and choice == 'cpu') or (not row['gpus'] and choice == 'cuda'):
            raise ValueError(f'{stage}: device setting conflicts with the requested GPUs')
    if row['gpus'] and stage not in {'train', 'apply'}:
        raise ValueError(f'{stage} does not use CUDA; request gpus: 0')
    return row
