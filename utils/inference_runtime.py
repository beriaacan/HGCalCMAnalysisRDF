"""Enter the existing dedicated PyTorch runtime at the application step."""
import os
from pathlib import Path
import sys

from .training_runtime import environment, configure_cuda_workspace


def launch_if_needed(config):
    configure_cuda_workspace(config)
    folder = config.training['training_environment']
    if os.environ.get('HGCAL_TRAINING_ENV') == str(folder) and Path(sys.prefix) == folder:
        return
    python, env = environment(config)
    script = Path(__file__).resolve().parents[1] / 'steer.py'
    remaining = config.steps[config.steps.index('apply'):]
    os.execve(str(python), [str(python), str(script), '--config', str(config.config_file),
                          '--steps', *remaining], env)


def device(config):
    import torch
    site = config.training['training_environment'] / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    if not Path(torch.__file__).is_relative_to(site):
        raise ValueError('DNN application loaded PyTorch outside the dedicated environment')
    choice = config.application['apply_device']
    chosen = ('cuda' if torch.cuda.is_available() else 'cpu') if choice == 'auto' else choice
    if chosen == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA application requested but no usable CUDA GPU is available')
    torch.set_num_threads(config.resources['cpus'])
    return chosen
