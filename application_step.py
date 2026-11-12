"""Application dispatcher; preserve the existing analytic implementation verbatim."""
from dataclasses import replace
import apply as analytic

SUPPORTED = analytic.SUPPORTED_METHODS | {'analytic', 'dnn'}


def describe(config):
    result = analytic.describe(config)
    result['unsupported_methods'] = sorted(set(config.methods) - SUPPORTED)
    if 'analytic' in config.methods:
        from utils.cm_application import describe as cm_description
        result['analytic'] = cm_description(config)
    if 'dnn' in config.methods:
        from utils.dnn_application import describe as dnn_description
        result['dnn'] = dnn_description(config)
    return result


def run(config):
    unsupported = set(config.methods) - SUPPORTED
    if unsupported:
        raise NotImplementedError(f'Application methods not implemented: {sorted(unsupported)}')
    if 'dnn' in config.methods:
        from utils.inference_runtime import launch_if_needed, device
        launch_if_needed(config)
        device(config)  # Fail an unavailable explicit CUDA request before writes.
        from utils.dnn_application import resolve_application, run as run_dnn
        # Validate the DNN before starting writes for any requested method.
        resolved = resolve_application(config)
    if 'analytic' in config.methods:
        from utils.cm_application import resolve_application as resolve_cm, run as run_cm
        cm_resolved = resolve_cm(config)
    linear = tuple(m for m in config.methods if m in analytic.SUPPORTED_METHODS)
    if linear:
        analytic.run(replace(config, methods=linear))
    if 'analytic' in config.methods:
        run_cm(config, cm_resolved)
    if 'dnn' in config.methods:
        run_dnn(config, resolved)
