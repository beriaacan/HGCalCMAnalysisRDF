"""Materialize compact selected event arrays once, independently of model training."""
from utils.paths import resolve_paths
from utils.training_cache import build, destination


def describe(config):
    return {'directory': str(destination(config, resolve_paths(config)['dnn_training_inputs'])),
            'contents': 'raw global features, ADC-selected targets and physical source entries per module-event',
            'partitions': 'existing retained training and validation events; original ROOT chunk boundaries',
            'format': 'uncompressed NumPy archives; no expanded event-channel input table',
            'reuse': 'input/source/code provenance checked; completed source units reused after interruption'}


def run(config):
    from training_inputs import load_inputs
    print('Checking training-input provenance for numeric cache', flush=True)
    directory, manifest = load_inputs(config, for_dnn=True)
    build(config, directory, manifest)
