"""Select the dedicated CUDA-capable environment without changing hgc."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def configure_cuda_workspace(config):
    # Apply explicit resource settings before torch creates CUDA handles.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = config.training['cublas_workspace_config']
    os.environ['CUBLASLT_WORKSPACE_SIZE'] = str(config.training['cublaslt_workspace_size_kib'])


def environment(config):
    folder = config.training['training_environment']
    python = folder / 'bin/python3'
    site = folder / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    if not python.is_file() or not site.is_dir():
        raise FileNotFoundError(f'Training environment missing: {folder}. Run setup_training.sh after hgc.')
    if not os.environ.get('CMSSW_BASE'):
        raise ValueError('CMSSW environment is not active; activate the current hgc setup first')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(site) + os.pathsep + env.get('PYTHONPATH', '')
    env['HGCAL_TRAINING_ENV'] = str(folder)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['CUBLAS_WORKSPACE_CONFIG'] = config.training['cublas_workspace_config']
    env['CUBLASLT_WORKSPACE_SIZE'] = str(config.training['cublaslt_workspace_size_kib'])
    return python, env


def launch_if_needed(config):
    folder = config.training['training_environment']
    if os.environ.get('HGCAL_TRAINING_ENV') == str(folder) and Path(sys.prefix) == folder:
        return False
    python, env = environment(config)
    script = Path(__file__).resolve().parents[1] / 'steer.py'
    # Replace the launcher: the shell tracks the actual trainer's PID and cannot
    # return while a child remains in ROOT cleanup after Ctrl-C. Errors/exit codes
    # propagate directly, without a second argparse usage/error block.
    # Earlier workflow steps have already run; preserve every remaining step.
    remaining=config.steps[config.steps.index('train'):]
    args=[str(python),str(script),'--config',str(config.config_file),'--steps',*remaining]
    if config.training.get('_selected_group') is not None:
        args += ['--training-group',config.training['_selected_group']]
    os.execve(str(python),args,env)
