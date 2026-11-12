"""Application-only discovery by physical recipe and supported saved contract.

Historical hashes remain provenance. They are never rewritten or used to resume
training under a different implementation. Ambiguous historical models fail.
"""
import json
from dataclasses import replace
from pathlib import Path

from .analytic_artifacts import sha
from .datasets import recipes

OPTION_KEYS = ('nodes','dropout','epochs','batch_events','learning_rate','weight_decay',
    'shuffle_buffer_chunks','sample_weighting','training_seed','training_deterministic',
    'optimizer_betas','optimizer_epsilon','early_stopping_patience','early_stopping_min_delta',
    'lr_scheduler_factor','lr_scheduler_patience','lr_scheduler_threshold','lr_scheduler_min_lr')
SPLIT_KEYS = ('split_seed','validation_split_fraction','train_event_fraction','validation_event_fraction')
_verified = {}


def verified_model(folder):
    """Hash a closed bundle once per unchanged file inventory in this process."""
    from .model_selection import read_model
    files=('manifest.json','model.pt','last.pt','preprocessing.json','model_spec.json',
           'history.json','training_data.json')
    fingerprint=tuple((name,(folder/name).stat().st_size,(folder/name).stat().st_mtime_ns) for name in files)
    previous=_verified.get(folder)
    if previous is None or previous[0]!=fingerprint:
        previous=(fingerprint,read_model(folder))
        if len(_verified)>=64:
            _verified.clear()
        _verified[folder]=previous
    return previous[1]


def canonical(value):
    return json.loads(json.dumps(value, default=str, sort_keys=True, allow_nan=False))


def options(config):
    return canonical({key:(config.dnn if key in config.dnn else config.training)[key] for key in OPTION_KEYS})


def population_request(config):
    from .identity import inventory,file_identity
    from .prepared import derivation_modules
    populations=[]
    for recipe in recipes(config):
        populations.append({'campaign':recipe.campaign,'selection':recipe.selection,
            'trigger_time_window':recipe.trigger_time_window,'adc_min':recipe.dataset.adcmin,
            'adc_max':recipe.dataset.adcmax,
            'sources':[{'campaign':recipe.campaign,'run':source.run,
                'files':inventory(source)['files'],'pedestal_run':recipe.pedestal.run,
                'calibration_sha256':file_identity(recipe.calibration_file)['sha256'],
                'module_map_sha256':file_identity(recipe.module_map)['sha256'],
                'activity_thresholds':config.preparation['activity_thresholds'],
                **({'conditions':recipe.conditions} if recipe.conditions else {})}
                for source in recipe.dataset.sources],
            'modules':list(derivation_modules(replace(config,correction=recipe,derivations=())))})
    return canonical({'datasets':populations,'split':{k:config.dnn[k] for k in SPLIT_KEYS}})


def request(config, group):
    from .preprocessing import dnn_definition
    return canonical({'schema':'application_source_request_v1','population':population_request(config),
        'options':options(config),'group':group,'events_per_chunk':config.training['training_events_per_chunk'],
        'input_features':config.dnn['input_features'],'max_channels':config.dnn['max_channels'],
        'missing_input_value':config.dnn['missing_input_value'],'preprocessing':dnn_definition()})


def legacy_request(folder, record):
    """Interpret existing signed v4 bundles; no hash aliases or bundle migration."""
    from .training_metadata import read
    data=read(folder,record)['definition']['training_recipe']
    definition=record['definition'];spec=definition['feature_spec']
    sources=[]
    for row in data['prepared']:
        old=row['preparation_definition']
        sources.append({k:old[k] for k in ('campaign','run','pedestal_run','calibration_sha256',
                                        'module_map_sha256','activity_thresholds')})
        sources[-1]['files']=[{k:f[k] for k in ('path','size','mtime_ns')} for f in old['files']]
    return canonical({'schema':'application_source_request_v1','population':{
        'datasets':[{k:data[k] for k in ('campaign','selection','trigger_time_window','adc_min','adc_max')}
                    | {'sources':sources,'modules':list(spec['modules'])}],
        'split':{k:data['options'][k] for k in SPLIT_KEYS}},
        'options':definition['options'],'group':Path(folder).name,
        'events_per_chunk':definition['events_per_chunk'],
        'input_features':[*spec['global_groups'],*spec['local_features']],
        'max_channels':spec['max_channels'],'missing_input_value':spec['missing_input_value'],
        'preprocessing':definition['preprocessing']})


def normalized(value):
    value=canonical(value)
    # Module enumeration order is not a feature: permanent physical IDs are.
    for row in value['population']['datasets']:
        row['modules']=sorted(row['modules'])
    return value


def find(config, group='combined', *, required=True):
    from .identity import model_directory
    from .model_selection import read_model
    exact=model_directory(config,group)
    if (exact/'manifest.json').is_file():
        state=json.loads((exact/'manifest.json').read_text())
        if state.get('complete') is False:
            if required:
                raise FileNotFoundError(f'Current DNN training is incomplete; resume train before application: {exact}')
            return None
        verified_model(exact)
        return exact
    if exact.exists() and any(exact.iterdir()):
        raise ValueError(f'Current DNN bundle is incomplete: {exact}')
    if 'dnn' in config.training.get('_workflow_current_models',()):
        if required:
            raise FileNotFoundError('Workflow explicitly requests current DNN training; wait for that producer')
        return None
    if any(not r.calibration_file.is_file() or not r.module_map.is_file() for r in recipes(config)):
        if required:
            raise FileNotFoundError('Source conditions are unavailable; cannot verify a historical DNN recipe')
        return None
    expected=normalized(request(config,group))
    candidates=[]
    for path in sorted((config.model_dir/'dnn').glob(f'rdf2_*/{group}/manifest.json')):
        record=json.loads(path.read_text())
        if record.get('complete') is not True:
            continue
        declared=record.get('definition',{}).get('application_source_request')
        if declared is None:
            # Fully check old metadata before using it to select a model.
            if record.get('schema')!='hgcal_cm_dnn_v4':
                continue
            declared=legacy_request(path.parent,record)
        if normalized(declared)!=expected:
            continue
        checked=verified_model(path.parent)
        # Validate the saved physical module vocabulary against the current registry.
        registry=json.loads(config.dnn['module_registry_file'].read_text())['modules']
        areas=json.loads(config.dnn['cell_area_file'].read_text())
        for name,meta in checked['definition']['feature_spec']['modules'].items():
            if registry.get(name)!=meta['module_id']:
                raise ValueError(f'Saved module ID changed for {name}: {path.parent}')
            if areas.get(name[:4],{}).get('SF')!=meta['cell_area_fraction']:
                raise ValueError(f'Saved training geometry changed for {name}: {path.parent}')
        candidates.append(path.parent)
    if len(candidates)>1:
        raise ValueError('Ambiguous compatible DNNs for this source recipe; no newest-model selection:\n'
                         +'\n'.join(map(str,candidates)))
    if candidates:
        return candidates[0]
    if required:
        raise FileNotFoundError(f'No completed compatible DNN for the configured source recipe: {exact}')
    return None


def find_linear(config,method, *, required=True):
    """Equivalent application-only selection for existing June linear predictors."""
    from .identity import linear_id
    from .preprocessing import CENTERED_CM
    from analytic_allinputs import cached_predictors
    suffix='cm' if method=='analytic' else 'allinputs'
    exact=config.model_dir/'analytic'/linear_id(config)/suffix
    if any(not r.calibration_file.is_file() or not r.module_map.is_file() for r in recipes(config)):
        if required:
            raise FileNotFoundError('Source conditions unavailable for linear model verification')
        return None
    wanted=population_request(config)
    candidates=[]
    paths=[exact/'manifest.json'] if (exact/'manifest.json').is_file() else (
        [] if method in config.training.get('_workflow_current_models',()) else sorted(
            (config.model_dir/'analytic').glob(f'rdf2_*/{suffix}/manifest.json')))
    for path in paths:
        stored=json.loads(path.read_text())
        if stored.get('complete') is not True:
            continue
        definition=stored['definition'];spec=definition['feature_spec']
        from .preprocessing import statistics_config
        local={'channel_in_erx','erx_index','cell_area_fraction'}
        features=statistics_config(config).dnn['input_features']
        groups=['cm_erx'] if method=='analytic' else [k for k in features if k not in local]
        locals_expected=[] if method=='analytic' else [k for k in features if k in local]
        if (spec.get('cm_convention')!=CENTERED_CM
                or spec.get('global_groups')!=groups or spec.get('local_features')!=locals_expected
                or spec.get('max_channels')!=config.dnn['max_channels']
                or spec.get('missing_input_value')!=config.dnn['missing_input_value']
                or definition.get('analytic_rcond')!=config.diagnostics['analytic_rcond']):
            continue
        original=Path(definition['training_inputs'])/'manifest.json'
        if not original.is_file() or sha(original)!=definition['training_manifest_sha256']:
            raise ValueError(f'Missing/modified saved linear training manifest: {original}')
        manifest=json.loads(original.read_text());data=manifest['definition']
        from .analytic_artifacts import signature
        if manifest['signature']!=definition['training_signature'] or signature(data)!=manifest['signature']:
            raise ValueError('Saved linear training recipe signature differs')
        sources=[]
        for row in data['prepared']:
            old=row['preparation_definition']
            source={k:old[k] for k in ('campaign','run','pedestal_run','calibration_sha256',
                                      'module_map_sha256','activity_thresholds')}
            source['files']=[{k:f[k] for k in ('path','size','mtime_ns')} for f in old['files']]
            if old.get('conditions'):
                source['conditions']=old['conditions']
            sources.append(source)
        population=canonical(definition.get('application_population_request')) or {'datasets':[{k:data[k] for k in
            ('campaign','selection','trigger_time_window','adc_min','adc_max')}
            | {'sources':sources,'modules':list(spec['modules'])}],
            'split':{k:data['options'][k] for k in SPLIT_KEYS}}
        # Normalize module enumeration order without changing feature/channel order.
        for value in (population,wanted):
            for row in value['datasets']:
                row['modules']=sorted(row['modules'])
        if canonical(population)!=canonical(wanted):
            continue
        checked=cached_predictors(path.parent,definition)
        if checked is None or not checked.get('complete'):
            raise ValueError(f'Incomplete saved linear predictor: {path.parent}')
        candidates.append((path.parent,checked))
    if len(candidates)>1:
        raise ValueError('Ambiguous compatible linear predictors; no newest-model selection')
    if candidates:
        return candidates[0]
    if required:
        raise FileNotFoundError(f'No completed compatible {method} predictor for the source recipe: {exact}')
    return None
