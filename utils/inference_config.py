"""Explicit DNN application resources, independent of training settings."""
from pathlib import Path

KEYS = {'apply_device', 'apply_batch_events', 'apply_events_per_chunk',
        'apply_prefetch_batches', 'apply_pin_memory'}


def parse(settings, base, local_path):
    out = {key: settings[key] for key in KEYS}
    if out['apply_device'] not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('apply_device must be auto, cpu or cuda')
    for key, minimum in [('apply_batch_events', 1), ('apply_events_per_chunk', 1),
                         ('apply_prefetch_batches', 0)]:
        if type(out[key]) is not int or out[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if type(out['apply_pin_memory']) is not bool:
        raise ValueError('apply_pin_memory must be true or false')
    return out


def method_directory(config, method):
    from .identity import linear_id, model_id, variant, code_identity
    if method == 'dnn':
        from .model_compatibility import find
        group='combined' if config.training['training_grouping']=='combined' else None
        if group is None:
            from .prepared import derivation_modules
            models={name:find(config,name,required=False) for name in derivation_modules(config)}
            model={name:folder.parent.name if folder else model_id(config,name) for name,folder in models.items()}
        else:
            folder=find(config,group,required=False)
            model=folder.parent.name if folder else model_id(config)
        key = variant({'model':model, 'code':code_identity(('utils/dnn_application.py',
            'utils/model_selection.py','utils/inference_inputs.py','utils/inference_predictor.py',
            'utils/inference_io.hxx','utils/inference_runtime.py','utils/inference_config.py',
            'utils/model_compatibility.py','utils/conditions.py','utils/datasets.py',
            'utils/preprocessing.py','utils/training.py','apply.py'))})
        return Path('dnn')/key
    if method in {'analytic', 'analytic_allinputs'}:
        from .model_compatibility import find_linear
        saved=find_linear(config,method,required=False)
        model=saved[0].parent.name if saved else linear_id(config)
        return Path(method)/variant({'model':model, 'code':code_identity((
            'apply.py','utils/cm_application.py','utils/application.py','utils/apply_helpers.hxx',
            'utils/model_compatibility.py','utils/training.py','utils/preprocessing.py'))})
    return Path(method)
