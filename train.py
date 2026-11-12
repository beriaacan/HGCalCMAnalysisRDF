"""Train DNN models locally."""
from __future__ import annotations

import json
from pathlib import Path
from utils.config import AnalysisConfig
from utils.paths import resolve_paths


def describe(config):
    from utils.preprocessing import dnn_definition
    from utils.training_cache import destination as cache_destination
    from utils.training_metadata import destination
    group = 'combined' if config.training['training_grouping'] == 'combined' else '<module>'
    return {'training_inputs': resolve_paths(config)['dnn_training_inputs'],
            'input_backend':config.training['training_input_backend'],
            'numeric_cache':str(cache_destination(config,resolve_paths(config)['dnn_training_inputs'])),
            'models': str(destination(config, group)), 'automatic_recipe': destination(config, group).parent.name,
            'grouping':config.training['training_grouping'],'modules':config.correction.modules,
            'device':config.training['training_device'],'environment':str(config.training['training_environment']),
            'batch_events':config.dnn['batch_events'],
            'training_events_per_chunk':config.training['training_events_per_chunk'],
            'prefetch_batches':config.training['training_prefetch_batches'],
            'pin_memory':config.training['training_pin_memory'],
            'input_expansion':'CUDA: compact event transfer and device expansion; CPU: existing NumPy expansion',
            'preprocessing': dnn_definition(),
            'loss': 'MSE in pedestal-subtracted ADC units',
            'validation':'existing retained validation events; best checkpoint and early stopping',
            'application':'not performed by this step', 'resume':config.training['training_resume']}


def resolve_training(config):
    from training_inputs import load_inputs
    from utils.dnn_training import groups,definition,cached,destination
    directory,manifest=load_inputs(config, for_dnn=True)
    preprocessing=json.loads((directory/'preprocessing.json').read_text())
    selected=groups(config,manifest['feature_spec'])
    requested=config.training.get('_selected_group')
    if requested is not None:
        if requested not in selected: raise ValueError(f'Unknown configured training group: {requested}')
        selected={requested:selected[requested]}
    for label,modules in selected.items():
        for partition in ('train','validation'):
            count=sum(sum(row[partition]) for name in modules
                      for row in manifest['counts_by_module_run'][name].values())
            if not count: raise ValueError(f'No retained {partition} targets in group {label}')
        cached(destination(config,label),definition(config,manifest,directory,modules))
    event_cache=None
    if config.training['training_input_backend']=='numeric_cache':
        from utils.training_cache import load
        event_cache=load(config,directory,manifest)
    return directory,manifest,preprocessing,selected,event_cache




def run(config: AnalysisConfig) -> None:
    from utils.training_runtime import launch_if_needed, configure_cuda_workspace
    configure_cuda_workspace(config)
    if launch_if_needed(config): return
    import torch
    expected_site=config.training['training_environment']/f'lib/python{__import__("sys").version_info.major}.{__import__("sys").version_info.minor}/site-packages'
    if not Path(torch.__file__).is_relative_to(expected_site):
        raise ValueError('PyTorch was loaded outside the dedicated training environment; check PYTHONPATH')
    if (config.training['training_device']=='cuda' or __import__('os').environ.get('HGCAL_REQUIRE_CUDA')=='1') and not torch.cuda.is_available():
        raise ValueError('CUDA training requested but no usable CUDA GPU is available in the dedicated environment')
    print(f'Training runtime: torch={torch.__version__}, CUDA build={torch.version.cuda}',flush=True)
    print('Checking training-input provenance for DNN training',flush=True)
    directory,manifest,preprocessing,selected,event_cache=resolve_training(config)
    from utils.dnn_training import train_group
    for group,modules in selected.items():
        train_group(config,manifest,directory,preprocessing,group,modules,event_cache=event_cache)
