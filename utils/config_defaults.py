"""Strict derivation/application blocks and documented workflow defaults."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from .calibrations import default_calibration_file

FRAMEWORK_ROOT=Path(__file__).resolve().parents[1]
PACKAGE_ROOT=FRAMEWORK_ROOT.parents[1]
REQUIRED=frozenset({'output_dir','derivation','application'})
DATA_REQUIRED=frozenset({'campaign','runs'})
CAMPAIGN_PATHS = {'Jun2026TB': ('/eos/cms/store/group/dpg_hgcal/tb_hgcal/2026/TestBeam',
               'Calibrations/TB2026/maps/modulelocator_v4.txt',
               'Calibrations/TB2026/calib'),
 'Cassette': ('/eos/cms/store/group/dpg_hgcal/tb_hgcal/2025/Test',
              'Calibrations/P5/maps/modulelocator_P5v9.txt',
              'Calibrations/P5/calib')}

DEFAULTS = {'steps': ['prepare'],
 'pedestal_optimize_binning': False,
 'methods': ['uncorrected', 'analytic', 'analytic_allinputs', 'dnn'],
 'loss': 'mse',
 'nodes': [256, 256, 256, 32],
 'dropout': 0.0,
 'input_features': ['cm_erx',
                    'activity_fractions',
                    'module_id',
                    'module_fraction',
                    'adc_allchannels',
                    'channel_in_erx',
                    'erx_index',
                    'cell_area_fraction'],
 'max_channels': 444,
 'missing_input_value': 0.0,
 'input_zs_threshold': None,
 'input_unsuppressed_fraction': None,
 'epochs': 200,
 'batch_events': 1024,
 'learning_rate': 0.001,
 'weight_decay': 0.0,
 'shuffle_buffer_chunks': 10,
 'sample_weighting': 'none',
 'activity_thresholds': [10, 50, 200, 500],
 'cpus': 4,
 'events_per_chunk': 5000,
 'plot_dnninputs': False,
 'n_coherent_noise_model': 3,
 'plot_noise_fit_variants': ['independent', 'fixedD_uncorrected'],
 'analytic_rcond': 1e-10,
 'plot_groups': ['covariance', 'noise_model', 'spectra', 'channel_summaries', 'cell_area'],
 'plot_formats': ['pdf'],
 'plot_adc_range': [-100, 200],
 'plot_adc_bins': 400,
 'plot_mip_range': [-10, 40],
 'plot_mip_bins': 200,
 'plot_sum_range': [-1000, 10000],
 'plot_sum_bins': 200,
 'plot_noise_window': [-10, 10],
 'plot_mip_fit': True,
 'plot_mip_fit_range': [-5, 35],
 'plot_gaussian_initial': [1000000, 0, 3],
 'plot_spectrum_maxfev': 20000,
 'plot_noise_maxiter': 2000,
 'plot_noise_gtol': 1e-05,
 'plot_noise_ftol': 2.220446049250313e-09,
 'plot_noise_max_rank': 5,
 'plot_noise_min_improvement': 0.03,
 'plot_noise_localization_min': 0.8,
 'plot_noise_variance_fraction_min': 0.1,
 'plot_noise_objective_atol': 1e-12,
 'plot_noise_objective_rtol': 1e-06,
 'plot_covariance_range': [-5, 5],
 'plot_noise_area_range': [0, 4],
 'plot_font_size': 12,
 'plot_mean_summary_range': [-5, 5],
 'plot_mean_summary_bins': 49,
 'plot_sigma_summary_range': [0, 10],
 'plot_sigma_summary_bins': 49,
 'plot_ratio_summary_range': [0, 2.5],
 'plot_ratio_summary_bins': 49,
 'training_grouping': 'combined',
 'training_device': 'auto',
 'training_seed': 6789,
 'training_resume': True,
 'training_deterministic': True,
 'training_input_backend': 'numeric_cache',
 'training_events_per_chunk': 50000,
 'training_prefetch_batches': 1,
 'training_pin_memory': True,
 'cublas_workspace_config': ':4096:8',
 'cublaslt_workspace_size_kib': 1024,
 'optimizer_betas': [0.9, 0.999],
 'optimizer_epsilon': 1e-08,
 'early_stopping_patience': 15,
 'early_stopping_min_delta': 0.0,
 'lr_scheduler_factor': 0.5,
 'lr_scheduler_patience': 5,
 'lr_scheduler_threshold': 0.0001,
 'lr_scheduler_min_lr': 1e-06,
 'condor_cpus': 4,
 'condor_gpus': 1,
 'condor_minimum_cuda_runtime': '12.6',
 'condor_memory_gb': 16,
 'condor_job_flavour': 'tomorrow',
 'condor_submit': True,
 'condor_use_x509_proxy': True,
 'condor_proxy': None,
 'multimodule_plot_groups': ['performance', 'detector_maps', 'canonical_maps'],
 'multimodule_min_coherent_sigma': 0.05,
 'multimodule_performance_range': [0.5, 1.1],
 'multimodule_performance_bins': 35,
 'multimodule_noise_range': [0, 2.5],
 'multimodule_relative_range': [-1, 1],
 'multimodule_ratio_range': [0, 2],
 'multimodule_closure_range': [-0.1, 0.1]}

DATA_DEFAULTS = {'modules': 'all',
 'pedestal_run': None,
 'conditions': None,
 'nano_root': None,
 'nano_production': None,
 'input_pattern': None,
 'module_map': None,
 'pedestal_nano_production': None,
 'pedestal_input_pattern': None,
 'calibration_file': None,
 'selection': 'full',
 'trigger_time_min': -2,
 'trigger_time_max': 5,
 'adc_min': None,
 'adc_max': None}

DERIVATION_DEFAULTS = {'split_seed': 6789,
 'validation_split_fraction': 0.2,
 'train_event_fraction': 1.0,
 'validation_event_fraction': 1.0}

APPLICATION_DEFAULTS = {'allow_unknown_modules': False,
 'device': 'auto',
 'batch_events': 1024,
 'events_per_chunk': 50000,
 'prefetch_batches': 1,
 'pin_memory': True,
 'infer_batch': 8192}

PATH_OPTIONS={'model_dir','cell_area_file','module_registry_file','training_environment',
              'condor_work_dir','condor_setup_file','multimodule_geometry_dir'}
SETTINGS=frozenset(DEFAULTS)|REQUIRED|PATH_OPTIONS
BLOCK_SETTINGS={role:DATA_REQUIRED|DATA_DEFAULTS.keys()|extra.keys()
                for role,extra in [('derivation',DERIVATION_DEFAULTS),('application',APPLICATION_DEFAULTS)]}


def _block(role,written,base,local_path):
    if not isinstance(written,dict):
        raise ValueError(f'{role} must be a YAML mapping')
    unknown=written.keys()-BLOCK_SETTINGS[role]
    if unknown:
        raise ValueError(f'Unknown {role} settings: {sorted(unknown)}')
    missing=DATA_REQUIRED-written.keys()
    if missing:
        raise ValueError(f'Missing {role} settings: {sorted(missing)}')
    campaign=written['campaign']
    if not isinstance(campaign,str) or campaign not in CAMPAIGN_PATHS:
        raise ValueError(f'{role}.campaign must be Jun2026TB or Cassette')
    run=written.get('pedestal_run')
    if written.get('conditions') is None:
        if type(run) is not int or run<=0:
            raise ValueError(f'{role}.pedestal_run must be a positive integer')
    elif run is not None:
        raise ValueError(f'{role}: choose conditions or pedestal_run; an IOV is not a pedestal acquisition run')
    nano_root,locator,calibrations=CAMPAIGN_PATHS[campaign]
    row=deepcopy(DATA_DEFAULTS)
    row.update(deepcopy(DERIVATION_DEFAULTS if role=='derivation' else APPLICATION_DEFAULTS))
    row.update(written)
    row['nano_root']=str(local_path(row['nano_root'] if row['nano_root'] is not None else nano_root,base,f'{role}.nano_root'))
    def pattern(production):
        if (not isinstance(production,str) or not production or production in {'.','..'}
                or any(c in production for c in ('/','{','}'))):
            raise ValueError(f'{role}.nano_production must name one production directory or an explicit glob')
        return str(Path(row['nano_root'])/'Run{run}'/production/'NANO_{run}_*.root')
    if row['input_pattern'] is not None:
        if row['nano_production'] is not None or written.get('nano_root') is not None:
            raise ValueError(f'{role}: choose input_pattern or nano_root/nano_production, not both')
        row['input_pattern']=str(local_path(row['input_pattern'],base,f'{role}.input_pattern'))
    else:
        if row['nano_production'] is None:
            raise ValueError(f'{role}: specify nano_production or input_pattern; no production is guessed')
        row['input_pattern']=pattern(row['nano_production'])
    if row['pedestal_input_pattern'] is not None:
        if row['pedestal_nano_production'] is not None:
            raise ValueError(f'{role}: choose pedestal_input_pattern or pedestal_nano_production')
        row['pedestal_input_pattern']=str(local_path(row['pedestal_input_pattern'],base,f'{role}.pedestal_input_pattern'))
    else:
        row['pedestal_input_pattern']=(pattern(row['pedestal_nano_production'])
            if row['pedestal_nano_production'] is not None else row['input_pattern'])
    calibration=row['calibration_file']
    if row['conditions'] is not None:
        from .conditions import resolve
        record,calibration,locator_file=resolve(row['conditions'],
            local_path(calibration,base,f'{role}.calibration_file') if calibration is not None else None,
            local_path(row['module_map'],base,f'{role}.module_map') if row['module_map'] is not None else None,
            PACKAGE_ROOT,campaign,row['runs'] if isinstance(row['runs'],list) else [row['runs']])
        row.update(conditions=record,module_map=locator_file)
    else:
        row['module_map']=str(local_path(row['module_map'] if row['module_map'] is not None
                                        else str(PACKAGE_ROOT/locator),base,f'{role}.module_map'))
    if calibration is None:
        calibration=str(default_calibration_file(PACKAGE_ROOT/calibrations,
            FRAMEWORK_ROOT/'calibrations'/campaign,run))
    row['calibration_file']=str(local_path(calibration,base,f'{role}.calibration_file'))
    nullable={'nano_production','pedestal_nano_production','adc_min','adc_max','conditions'}
    if row['conditions'] is not None:
        nullable.add('pedestal_run')
    nulls=[key for key,value in row.items() if value is None and key not in nullable]
    if nulls:
        raise ValueError(f'{role}: these settings cannot be null: {sorted(nulls)}')
    return row


def complete_settings(written,base,local_path):
    unknown=written.keys()-SETTINGS
    if unknown:
        raise ValueError(f'Unknown settings: {sorted(unknown)}')
    missing=REQUIRED-written.keys()
    if missing:
        raise ValueError(f'Missing required settings: {sorted(missing)}')
    result=deepcopy(DEFAULTS);result.update(written)
    derivation=written['derivation']
    if isinstance(derivation,dict) and 'datasets' in derivation:
        if set(derivation) != {'datasets'} or not isinstance(derivation['datasets'],list) or not derivation['datasets']:
            raise ValueError('derivation.datasets must be a nonempty list; put source settings inside each dataset')
        result['derivation']={'datasets':[_block('derivation',row,base,local_path) for row in derivation['datasets']]}
        if result['training_grouping']!='combined' or result['sample_weighting']!='none':
            raise ValueError('Multi-dataset derivation requires combined training and sample_weighting: none')
        campaigns={row['campaign'] for row in result['derivation']['datasets']}
        if len(campaigns)>1 and set(result['methods'])-{'uncorrected','dnn'}:
            raise ValueError('Cross-campaign derivation supports methods: [uncorrected, dnn]; linear pooling requires one campaign')
        source_campaign=next(iter(campaigns)) if len(campaigns)==1 else 'joint'
    else:
        result['derivation']=_block('derivation',derivation,base,local_path)
        source_campaign=result['derivation']['campaign']
    result['application']=_block('application',written['application'],base,local_path)
    output=local_path(written['output_dir'],base,'output_dir');result['output_dir']=str(output)
    paths={'model_dir':output/'models'/source_campaign,
           'cell_area_file':PACKAGE_ROOT/'LocalCalibration/data/cellareas.json',
           'module_registry_file':FRAMEWORK_ROOT/'data/module_registry.json',
           'training_environment':FRAMEWORK_ROOT/'.venv_training',
           'condor_work_dir':FRAMEWORK_ROOT/'workdir_condor',
           'condor_setup_file':FRAMEWORK_ROOT/'setup.sh',
           'multimodule_geometry_dir':PACKAGE_ROOT/'DQM/data'}
    for name,default in paths.items():
        result[name]=str(local_path(written.get(name) if written.get(name) is not None else str(default),base,name))
    return result
