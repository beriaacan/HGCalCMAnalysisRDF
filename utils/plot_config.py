"""Explicit, flat configuration for the legacy apply -p plot families."""
from __future__ import annotations

import math

GROUPS = ('covariance', 'noise_model', 'spectra', 'channel_summaries', 'cell_area')
KEYS = {'plot_noise_fit_variants', 'plot_groups', 'plot_formats', 'plot_adc_range', 'plot_adc_bins',
        'plot_mip_range', 'plot_mip_bins', 'plot_sum_range', 'plot_sum_bins',
        'plot_noise_window', 'plot_mip_fit', 'plot_mip_fit_range',
        'plot_gaussian_initial', 'plot_spectrum_maxfev', 'plot_noise_maxiter', 'plot_noise_gtol', 'plot_noise_ftol',
        'plot_noise_max_rank', 'plot_noise_min_improvement', 'plot_noise_localization_min',
        'plot_noise_variance_fraction_min', 'plot_noise_objective_atol', 'plot_noise_objective_rtol',
        'plot_covariance_range', 'plot_noise_area_range', 'plot_font_size',
        'plot_mean_summary_range','plot_mean_summary_bins','plot_sigma_summary_range',
        'plot_sigma_summary_bins','plot_ratio_summary_range','plot_ratio_summary_bins'}


def parse(settings):
    out = {}
    variants = settings['plot_noise_fit_variants']
    if (not isinstance(variants, list) or not variants or variants[0] != 'independent'
            or len(set(variants)) != len(variants)
            or any(v not in ('independent', 'fixedD_uncorrected') for v in variants)):
        raise ValueError('plot_noise_fit_variants must start with independent, optionally followed by fixedD_uncorrected')
    out['plot_noise_fit_variants'] = list(variants)
    for name, choices in (('plot_groups', GROUPS), ('plot_formats', ('pdf', 'png'))):
        values = settings[name]
        if (not isinstance(values, list) or not values or len(set(values)) != len(values)
                or any(v not in choices for v in values)):
            raise ValueError(f'{name} must be a nonempty distinct list from {choices}')
        out[name] = list(values)
    for name in ('plot_adc_range', 'plot_mip_range', 'plot_sum_range', 'plot_noise_window',
                 'plot_mip_fit_range', 'plot_covariance_range', 'plot_noise_area_range',
                 'plot_mean_summary_range','plot_sigma_summary_range','plot_ratio_summary_range'):
        value = settings[name]
        if (not isinstance(value, list) or len(value) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)
                or value[0] >= value[1]):
            raise ValueError(f'{name} must be [finite lower, finite upper]')
        out[name] = [float(x) for x in value]
    for name in ('plot_adc_bins', 'plot_mip_bins', 'plot_sum_bins', 'plot_spectrum_maxfev', 'plot_noise_maxiter', 'plot_font_size',
                 'plot_mean_summary_bins','plot_sigma_summary_bins','plot_ratio_summary_bins'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
        out[name] = settings[name]
    if type(settings['plot_mip_fit']) is not bool:
        raise ValueError('plot_mip_fit must be true or false')
    out['plot_mip_fit'] = settings['plot_mip_fit']
    initial = settings['plot_gaussian_initial']
    if (not isinstance(initial, list) or len(initial) != 3
            or any(isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) for x in initial)
            or initial[0] <= 0 or initial[2] <= 0):
        raise ValueError('plot_gaussian_initial must be [positive amplitude, mean, positive width]')
    out['plot_gaussian_initial'] = [float(x) for x in initial]
    for name in ('plot_noise_gtol', 'plot_noise_ftol'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        out[name] = float(value)
    maximum = settings['plot_noise_max_rank']
    if type(maximum) is not int or not 0 <= maximum <= 5:
        raise ValueError('plot_noise_max_rank must be an integer from 0 through 5')
    out['plot_noise_max_rank'] = maximum
    for name in ('plot_noise_min_improvement', 'plot_noise_localization_min', 'plot_noise_variance_fraction_min',
                 'plot_noise_objective_atol', 'plot_noise_objective_rtol'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
        if name == 'plot_noise_min_improvement' and value >= 1:
            raise ValueError('plot_noise_min_improvement must be in [0, 1)')
        if name in ('plot_noise_localization_min', 'plot_noise_variance_fraction_min') and not 0 < value < 1:
            raise ValueError(f'{name} must be in (0, 1)')
        out[name] = float(value)
    return out


def histogram_definition(options):
    return {key: options[key] for key in ('plot_adc_range','plot_adc_bins','plot_mip_range',
            'plot_mip_bins','plot_sum_range','plot_sum_bins','plot_noise_window')}


def histogram_label(options):
    def part(name):
        lo, hi = options[f'plot_{name}_range']
        return f'{name}{lo:g}_{hi:g}_{options[f"plot_{name}_bins"]}'
    lo, hi = options['plot_noise_window']
    return '__'.join([part('adc'),part('mip'),part('sum'),f'window{lo:g}_{hi:g}'])
