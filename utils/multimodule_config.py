"""Explicit settings for cached, run-level noise summaries."""
from __future__ import annotations

import math

GROUPS = ('performance', 'detector_maps', 'canonical_maps')
KEYS = {'multimodule_plot_groups', 'multimodule_min_coherent_sigma',
        'multimodule_performance_range', 'multimodule_performance_bins',
        'multimodule_noise_range', 'multimodule_relative_range',
        'multimodule_ratio_range', 'multimodule_closure_range',
        'multimodule_geometry_dir'}


def parse(settings, base, local_path):
    out = {key: settings[key] for key in KEYS}
    groups = out['multimodule_plot_groups']
    if (not isinstance(groups, list) or not groups or any(g not in GROUPS for g in groups)
            or len(set(groups)) != len(groups)):
        raise ValueError(f'multimodule_plot_groups must be a distinct nonempty list from {GROUPS}')
    threshold = out['multimodule_min_coherent_sigma']
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold < 0):
        raise ValueError('multimodule_min_coherent_sigma must be finite and nonnegative')
    for key in ('multimodule_performance_range', 'multimodule_noise_range',
                'multimodule_relative_range', 'multimodule_ratio_range', 'multimodule_closure_range'):
        value = out[key]
        if (not isinstance(value, list) or len(value) != 2
                or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in value)
                or value[0] >= value[1]):
            raise ValueError(f'{key} must be [finite lower, finite upper]')
        out[key] = [float(x) for x in value]
    for key in ('multimodule_noise_range', 'multimodule_ratio_range'):
        if out[key][0] < 0:
            raise ValueError(f'{key} must have a nonnegative lower limit')
    for key, center in (('multimodule_relative_range', 0.), ('multimodule_ratio_range', 1.),
                        ('multimodule_closure_range', 0.)):
        if not out[key][0] < center < out[key][1]:
            raise ValueError(f'{key} must contain its color-scale center {center:g} strictly inside')
    bins = out['multimodule_performance_bins']
    if type(bins) is not int or bins < 1:
        raise ValueError('multimodule_performance_bins must be a positive integer')
    out['multimodule_geometry_dir'] = local_path(out['multimodule_geometry_dir'], base, 'multimodule_geometry_dir')
    return out
