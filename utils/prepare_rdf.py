"""RDF columns using unchanged central selections and our cache-specific layout."""
from __future__ import annotations

import json
import math
from pathlib import Path

_READY = False


def initialize_root():
    global _READY
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    if not _READY:
        import os
        source = Path(os.environ['CMSSW_BASE']) / 'src'
        central = source / 'HGCalCommissioning/LocalCalibration/interface/helpers.h'
        local = Path(__file__).with_name('prepare_helpers.hxx')
        if not ROOT.gInterpreter.Declare(f'#include {json.dumps(str(central))}\n#include {json.dumps(str(local))}'):
            raise RuntimeError('Could not compile central/preparation RDF helpers')
        _READY = True
    return ROOT


def _vector(values) -> str:
    if any(not math.isfinite(float(v)) for v in values):
        raise ValueError('Non-finite calibration constant')
    literals = []
    for value in values:
        token = f'{float(value):.17g}'
        if '.' not in token and 'e' not in token:
            token += '.0'
        literals.append(token + 'f')
    return 'ROOT::RVec<float>{' + ','.join(literals) + '}'



def threshold_label(value: float) -> str:
    return f'{value:g}'.replace('.', 'p')


def module_columns(rdf, name: str, identity: tuple[int, int, int], calibration: dict,
                   activity_thresholds: tuple[float, ...]):
    """Keep every source event; unavailable channel values remain NaN."""
    initialize_root()
    from HGCalCommissioning.LocalCalibration import DigiAnalysisUtils as central
    fed, sequence, nerx = identity
    nch = 37 * nerx
    rdf = rdf.Define('modulemask', f'HGCDigi_fedId=={fed} && HGCDigi_fedReadoutSeq=={sequence}')
    rdf = rdf.Define('module_channels', 'HGCDigi_channel[modulemask]')
    rdf = rdf.Define('module_present', f'hgcal_cm_prepare::present(module_channels,{nch})')
    # nofilter=True deliberately preserves events with no measurements for a module.
    rdf = central.addStandardADCSelection(rdf, modulemask='modulemask', nofilter=True)
    rdf = central.addStandardTOTSelection(rdf, modulemask='modulemask', nofilter=True)
    rdf = central.addStandardTOASelection(rdf, modulemask='modulemask', nofilter=True)
    columns = ['module_present', 'nchadc', 'nchtot', 'nchtoa']
    rdf = rdf.Define('module_nch', str(nch)).Define('module_nerx', str(nerx))
    columns += ['module_nch', 'module_nerx']
    for output, channels, values in (
        ('adc_raw', 'chadc', 'adc'), ('adcm1_raw', 'chadc', 'adcm1'),
        ('tot_raw', 'chtot', 'tot'), ('toa_raw', 'chtoa', 'toa'),
        ('channel_type', 'module_channels', 'HGCDigi_chType[modulemask]'),
        ('channel_u', 'module_channels', 'HGCDigi_chI1[modulemask]'),
        ('channel_v', 'module_channels', 'HGCDigi_chI2[modulemask]'),
        ('digi_flags', 'module_channels', 'HGCDigi_flags[modulemask]'),
        ('digi_tctp', 'module_channels', 'HGCDigi_tctp[modulemask]'),
        ('adc_in_tot', 'chtot', 'HGCDigi_adc[masktot]'),
    ):
        rdf = rdf.Define(output, f'hgcal_cm_prepare::dense({channels},{values},{nch})')
        columns.append(output)
    rdf = rdf.Define('cm_erx', f'hgcal_cm_prepare::erx_cm(module_channels,HGCDigi_cm[modulemask],{nerx})')
    rdf = rdf.Define('cm_raw', 'hgcal_cm_prepare::expand_cm(cm_erx)')
    rdf = rdf.Define('adc_pedsub', f'static const auto ped = {_vector(calibration["ADC_ped"])}; return adc_raw - ped;')
    rdf = rdf.Define('cm_pedsub', f'static const auto ped = {_vector(calibration["CM_ped"])}; return cm_raw - ped;')
    columns += ['cm_erx', 'cm_raw', 'adc_pedsub', 'cm_pedsub']
    for output, expression in (
        ('adc_sum_pedsub', 'hgcal_cm_prepare::sum_finite(adc_pedsub)'),
        ('adc_max_pedsub', 'hgcal_cm_prepare::max_finite(adc_pedsub)'),
        ('adcm1_max', 'hgcal_cm_prepare::max_finite(adcm1_raw)'),
        ('erx_nchtoa', 'hgcal_cm_prepare::count_erx(toa_raw)'),
        ('erx_nchtot', 'hgcal_cm_prepare::count_erx(tot_raw)'),
        ('erx_hastoa', 'erx_nchtoa > 0'), ('erx_hastot', 'erx_nchtot > 0'),
    ):
        rdf = rdf.Define(output, expression)
        columns.append(output)
    for threshold in activity_thresholds:
        label = threshold_label(threshold)
        per_erx = f'erx_nchadcgt{label}'
        total = f'nchadcgt{label}'
        rdf = rdf.Define(per_erx, f'hgcal_cm_prepare::count_above(adc_pedsub,{threshold:.17g})')
        rdf = rdf.Define(total, f'Sum({per_erx})')
        columns += [per_erx, total]
    return rdf, columns


def analysis_view(rdf, recipe):
    """Apply cheap analysis masks and event selections to reusable prepared inputs."""
    initialize_root()
    low, high = recipe.dataset.adcmin, recipe.dataset.adcmax
    rdf = rdf.Define('adc_selected',
                     f'hgcal_cm_prepare::adc_window(adc_pedsub,{str(low is not None).lower()},'
                     f'{low if low is not None else 0.},{str(high is not None).lower()},'
                     f'{high if high is not None else 0.})')
    rdf = rdf.Define('adc_selected_sum', 'hgcal_cm_prepare::sum_finite(adc_selected)')
    if 'trigtime' in recipe.selection:
        lower, upper = recipe.trigger_time_window
        rdf = rdf.Filter(f'source_is_pedestal || (trig_time>{lower:.17g} && trig_time<{upper:.17g})',
                         'trigger-time selection (pedestal source retained)')
    if 'test' in recipe.selection:
        if 'is_test' not in {str(c) for c in rdf.GetColumnNames()}:
            raise ValueError('Test-event selection requires training split metadata; preparation does not assign splits')
        rdf = rdf.Filter('is_test', 'test-event selection')
    return rdf
