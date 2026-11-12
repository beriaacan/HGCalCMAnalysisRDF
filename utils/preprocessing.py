"""Fixed DNN physical features and internal linear-statistics adapter."""
from dataclasses import replace
from functools import lru_cache
import json
import math
from pathlib import Path

STATISTICS = 'training_statistics'
CALIBRATION = 'calibration_only'
CENTERED_CM = 'central average per eRx minus mean finite CM_ped with Valid=1 in that eRx'


def calibration_only(value):
    return value.get('preprocessing_mode', STATISTICS) == CALIBRATION


def statistics_config(config):
    """Internal linear feature recipe; it does not need separate preparation."""
    if not calibration_only(config.dnn):
        return config
    settings = dict(config.dnn, preprocessing_mode=STATISTICS,
                    preprocess_inputs=True, preprocess_targets=True)
    if 'module_fraction' in settings['input_features']:
        # Linear corrections retain their existing shared feature/cache contract.
        settings['input_features'] = tuple('module_nch' if name == 'module_fraction' else name
                                          for name in settings['input_features'])
    return replace(config, dnn=settings)


def dnn_definition():
    """The single supported DNN contract, also exposed by --show and saved models."""
    return {'adc': 'central per-channel ADC pedestal subtraction',
            'cm': CENTERED_CM,
            'module_fraction': 'Nchannels / 444 before the DNN',
            'internal_normalization': 'fixed training-only z-score for ADC and CM feature columns; own ADC masked before fitting',
            'other_features': 'identity', 'targets': 'pedestal-subtracted ADC units; no target normalization',
            'normalization_storage': 'frozen mean/scale buffers inside the model checkpoint'}


def cm_reference(calibration, nch, module):
    """One equal-weight reference per real eRx, excluding invalid placeholders."""
    if calibration.get('Channel') != list(range(nch)) or nch % 37:
        raise ValueError(f'Invalid calibration channel mapping for {module}')
    ped, valid = calibration.get('CM_ped', []), calibration.get('Valid', [])
    if len(ped) != nch or len(valid) != nch:
        raise ValueError(f'Incomplete CM calibration for {module}')
    result = []
    for start in range(0, nch, 37):
        values = [float(ped[c]) for c in range(start, start+37)
                  if valid[c] == 1 and isinstance(ped[c], (int, float)) and math.isfinite(ped[c])]
        if not values:
            raise ValueError(f'No finite CM_ped with Valid=1 for {module}/eRx{start//37}; no reference can be defined')
        result.append(math.fsum(values)/len(values))
    return result


@lru_cache(maxsize=16)
def _calibration(path, size, mtime):
    return json.loads(Path(path).read_text())


def reference_from_file(path, module, nch):
    path = Path(path)
    stat = path.stat()
    data = _calibration(str(path), stat.st_size, stat.st_mtime_ns)
    key = module.replace('_', '-')
    if key not in data:
        raise ValueError(f'Calibration lacks module {module}: {path}')
    return cm_reference(data[key], nch, module)


def descriptor(spec):
    """Calibration-only models carry a rule, not fitted mean/width arrays."""
    result = {'preprocessing_mode': CALIBRATION, 'feature_names': spec['feature_names'],
            'feature_transforms': spec['feature_transforms'],
            'inputs_enabled': False, 'targets_enabled': False,
            'fit_partition': 'none; no data-fitted normalization',
            'cm_reference': CENTERED_CM, 'target_units': 'pedestal-subtracted ADC',
            'missing_input_value': spec['missing_input_value'],
            'module_id_convention': spec['module_id_convention']}
    if 'module_fraction' in spec['global_groups']:
        result['module_fraction_denominator'] = spec['module_fraction_denominator']
    return result


def validate_descriptor(pre, spec):
    if pre != descriptor(spec):
        raise ValueError('Calibration-only preprocessing descriptor differs from the saved feature definition')
