"""Workflow adapters to existing recipes/readers; no numerical algorithms."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile

from .config import load_config
from .workflow_registry import digest

ROOT = Path(__file__).resolve().parents[1]


def output_paths(config, stage):
    from .paths import resolve_paths
    paths = resolve_paths(config)
    if stage == 'pedestals':
        return sorted({str(r.calibration_file) for r in (config.correction, config.target)})
    if stage == 'prepare':
        from .prepared import plan, cache_directory
        return [str(cache_directory(job)) for job in plan(config)]
    if stage == 'training_inputs':
        return [paths['training_inputs']]
    if stage == 'training_cache':
        from .training_cache import destination
        return [str(destination(config, paths['training_inputs']))]
    if stage in {'analytic', 'analytic_allinputs'}:
        return [str(Path(paths['analytic_predictors'])/('cm' if stage == 'analytic' else 'allinputs'))]
    if stage == 'train':
        from .identity import model_directory
        from .prepared import derivation_modules
        names = ['combined'] if config.training['training_grouping'] == 'combined' else derivation_modules(config)
        return [str(model_directory(config, name)) for name in names]
    if stage == 'apply':
        from .inference_config import method_directory
        return [str(Path(paths['target']['predictions'])/method_directory(config, method))
                for method in config.methods if method != 'uncorrected']
    if stage in {'plots', 'multimodule_plot'}:
        return [paths['target']['plots']]
    if stage == 'export':
        from .nano_export import destination
        return [str(destination(config))]
    raise ValueError(stage)


def from_settings(settings):
    import yaml
    with tempfile.TemporaryDirectory(prefix='hgcal-workflow-config-') as temporary:
        path = Path(temporary)/'config.yaml'
        path.write_text(yaml.safe_dump(settings, sort_keys=False))
        return load_config(path)


def lock_paths(config, stage):
    """Conservative physical write scopes, stable while upstream IDs are pending.

    Different variants may serialize unnecessarily; they can never write the same
    destination through different registries. Preparation coverage is serialized
    per run/calibration; per-module standalone training jobs keep separate scopes.
    """
    out = config.output_dir
    locks = []
    def add(scope, base=None):
        root = Path(base) if base is not None else out
        locks.append(str(root/'.workflow-locks'/digest(str(Path(scope).resolve()))))
    correction, target = config.correction, config.target
    source = (out/correction.campaign/'training'/str(correction.dataset.run)
              /f'pedestals_{correction.pedestal.run}'/correction.selection)
    evaluation = (out/target.campaign/'evaluation'/str(target.dataset.run)
                  /f'pedestals_{target.pedestal.run}'/target.selection/config.model_dir.name)
    if stage == 'pedestals':
        for row in (correction, target):
            if not row.calibration_file.is_file():
                add(row.calibration_file, row.calibration_file.parent)
    elif stage == 'prepare':
        for row in (correction, target):
            for run in row.dataset.sources:
                add(out/row.campaign/'prepared'/f'Run{run.run}'/f'pedestals_{row.pedestal.run}')
    elif stage in {'training_inputs', 'training_cache'}:
        add(source/stage)
        if stage == 'training_inputs':
            path = config.dnn['module_registry_file']
            add(path, path.parent)
    elif stage in {'analytic', 'analytic_allinputs'}:
        add(config.model_dir/'analytic'/stage)
    elif stage == 'train':
        from .prepared import derivation_modules
        group = config.training.get('_selected_group')
        groups = ([group] if group else ['combined'] if config.training['training_grouping'] == 'combined'
                  else derivation_modules(config))
        for group in groups:
            add(config.model_dir/'dnn'/group)
    elif stage == 'apply':
        add(evaluation/'predictions')
    elif stage in {'plots', 'multimodule_plot'}:
        add(evaluation/'statistics-and-fits')
        add(ROOT/'plots'/target.campaign/f'Run{target.dataset.run}', ROOT/'plots')
    elif stage == 'export':
        add(evaluation/'nano_exports')
    elif stage != 'submit_train':
        raise ValueError(f'No workflow writer scope for {stage}')
    return sorted(set(locks))


def request_key(config, stage):
    """Logical job identity; final scientific paths are re-resolved by each worker."""
    from .identity import input_id, linear_id, model_id, population, variant
    from .paths import resolve_paths
    if stage == 'prepare':
        from .prepared import plan, cache_directory
        return digest({'stage': stage, 'destinations': [str(cache_directory(job)) for job in plan(config)]})
    if stage == 'pedestals':
        return digest({'stage': stage, 'targets': [str(config.correction.calibration_file), str(config.target.calibration_file)],
                       'campaign': config.correction.campaign, 'run': config.correction.pedestal.run})
    if stage in {'training_inputs', 'training_cache'}:
        return digest({'stage': stage, 'inputs': resolve_paths(config)['training_inputs'],
                       'chunk': config.training['training_events_per_chunk'] if stage == 'training_cache' else None})
    if stage in {'analytic', 'analytic_allinputs'}:
        return digest({'stage': stage, 'directory': str(config.model_dir), 'recipe': linear_id(config)})
    if stage == 'train':
        from .prepared import derivation_modules
        groups = ['combined'] if config.training['training_grouping'] == 'combined' else derivation_modules(config)
        return digest({'stage': stage, 'directory': str(config.model_dir),
                       'models': {g: model_id(config, g) for g in groups}})
    target = {'target': population(config, target=True), 'modules': config.target.modules,
              'methods': list(config.methods), 'model_dir': str(config.model_dir),
              'models': {m: model_id(config) if m == 'dnn' else linear_id(config)
                         for m in config.methods if m != 'uncorrected'},
              'output_dir': str(config.output_dir), 'unknown': config.dnn['allow_unknown_modules']}
    if stage in {'plots', 'multimodule_plot'}:
        target.update(plotting=config.plotting, multimodule=config.multimodule)
    return digest(json.loads(json.dumps({'stage': stage, **target}, default=str)))


def artifact_files(directories):
    """Record metadata for published files only, never retained temporary work."""
    files = set()
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in directory.rglob('*'):
            relative = path.relative_to(directory)
            if path.is_file() and not any(p.startswith('.') for p in relative.parts):
                files.add(path)
    return files


def dependencies_files(config):
    files = {config.dnn['cell_area_file']}
    for recipe in (config.correction, config.target):
        files.update((recipe.calibration_file, recipe.module_map))
        for source in recipe.dataset.sources:
            from .paths import discover_files
            files.update(map(Path, discover_files(replace(recipe.dataset, sources=(source,)))))
    return {p for p in files if p.is_file()}


def training_references(directory, manifest):
    files = artifact_files([directory])
    for row in manifest['definition']['prepared']:
        root = Path(row['directory'])
        files.add(root/'manifest.json')
        for output in row['outputs']:
            file = root/output['file']
            files.update((file, file.parent/'manifest.json'))
    return files


def application_references(resolved):
    files = set()
    for row in resolved['modules'].values():
        predictor = Path(row['definition']['predictor'])
        files.update(artifact_files([predictor.parent]))
        if (predictor.parent.parent/'manifest.json').is_file():
            files.add(predictor.parent.parent/'manifest.json')
        for item in row['inputs']:
            file = Path(item['input_file'])
            files.update((file, file.parent/'manifest.json', file.parent.parent/'manifest.json'))
    return files


def probe(config, stage):
    """Validate with current stage readers. Return completion and audited files.

    Incomplete, resumable overall publications remain missing work. Invalid
    published modules are errors. Plot rendering is intentionally repeated unless
    a matching workflow receipt proves that exact render request already finished.
    """
    directories, extra_files = [], set()
    if stage == 'pedestals':
        from .calibrations import validate_calibrations
        if not all(r.calibration_file.is_file() for r in (config.correction, config.target)):
            return False, []
        validate_calibrations(config, require_files=True)
        return True, sorted({r.calibration_file for r in (config.correction, config.target)})
    if not all(r.calibration_file.is_file() for r in (config.correction, config.target)):
        return False, []
    try:
        if stage == 'prepare':
            from .prepared import plan, resolve, cached_files, cache_directory
            jobs = plan(config)
            if any(not cache_directory(job).exists() for job in jobs):
                return False, []
            for job in jobs:
                resolved = resolve(config, job)
                if cached_files(resolved) is None:
                    return False, []
                directories.extend(resolved['destination']/name for name in resolved['modules'])
                directories.append(resolved['destination'])
        elif stage == 'training_inputs':
            from .paths import resolve_paths
            if not Path(resolve_paths(config)['training_inputs']).exists():
                return False, []
            from training_inputs import resolve_inputs, cached_inputs
            resolved = resolve_inputs(config)
            if cached_inputs(resolved) is None:
                return False, []
            directories.append(resolved['destination'])
            extra_files.update(training_references(resolved['destination'], cached_inputs(resolved)))
        elif stage == 'training_cache':
            from training_inputs import load_inputs
            from .training_cache import destination, load
            from .paths import resolve_paths
            if not destination(config, resolve_paths(config)['training_inputs']).exists():
                return False, []
            directory, manifest = load_inputs(config, for_dnn=True)
            target = destination(config, directory)
            if not (target/'manifest.json').is_file():
                return False, []
            record = json.loads((target/'manifest.json').read_text())
            if not record.get('complete'):
                return False, []
            load(config, directory, manifest)
            directories.append(target)
            extra_files.update(training_references(directory, manifest))
        elif stage in {'analytic', 'analytic_allinputs'}:
            from training_inputs import load_inputs
            from analytic_allinputs import cached_predictors
            import analytic, analytic_allinputs
            module = analytic if stage == 'analytic' else analytic_allinputs
            target = Path(module.describe(config)['destination'])
            if not target.exists():
                return False, []
            directory, manifest = load_inputs(config)
            expected = module.definition(config, directory, manifest) if stage == 'analytic' else module._definition(config, directory, manifest)
            target = Path(module.describe(config)['destination'])
            record = cached_predictors(target, expected)
            if record is None or len(record['module_records']) != len(expected['feature_spec']['modules']):
                return False, []
            directories.append(target)
            extra_files.update(training_references(directory, manifest))
        elif stage == 'train':
            from .identity import model_directory
            from .model_selection import read_model
            from .prepared import derivation_modules
            groups = ['combined'] if config.training['training_grouping'] == 'combined' else derivation_modules(config)
            for group in groups:
                target = model_directory(config, group)
                path = target/'manifest.json'
                if not path.is_file():
                    if target.is_dir() and any(target.iterdir()):
                        raise ValueError(f'DNN directory has no provenance manifest: {target}')
                    return False, []
                if not json.loads(path.read_text()).get('complete'):
                    return False, []
                read_model(target)
                directories.append(target)
        elif stage == 'apply':
            import apply as linear
            for method in config.methods:
                if method != 'uncorrected' and not probe(config, 'train' if method == 'dnn' else method)[0]:
                    return False, []
            for method in config.methods:
                if method == 'uncorrected':
                    continue
                if method == 'dnn':
                    from .dnn_application import resolve_application
                elif method == 'analytic':
                    from .cm_application import resolve_application
                else:
                    resolve_application = linear.resolve_application
                resolved = resolve_application(replace(config, methods=(method,)))
                result = linear.cached_application(resolved)
                if result is None or not result['complete']:
                    return False, []
                directories.append(resolved['destination'])
                extra_files.update(application_references(resolved))
            if not directories:
                from .prepared import plan, resolve, cached_files
                for job in plan(config):
                    if 'apply' in job.roles:
                        resolved = resolve(config, job)
                        if cached_files(resolved) is None:
                            return False, []
                        directories.append(resolved['destination'])
        elif stage == 'export':
            applied, applied_files = probe(config, 'apply')
            if not applied:
                return False, []
            extra_files.update(applied_files)
            from .nano_export import resolve, cached_unit
            resolved = resolve(config)
            for key, unit in resolved['files'].items():
                if cached_unit(resolved['destination']/Path(key).parent, unit['definition']) is None:
                    return False, []
            directories.append(resolved['destination'])
        elif stage in {'plots', 'multimodule_plot'}:
            return False, []
        else:
            raise ValueError(f'No workflow product probe for {stage}')
    except FileNotFoundError:
        return False, []
    return True, sorted(artifact_files(directories) | dependencies_files(config) | extra_files)


def completed_files(config, stage):
    """Called only after the existing stage returned successfully."""
    if stage in {'plots', 'multimodule_plot'}:
        from .paths import resolve_paths, module_plot_directory
        import plots, multimodule_plot
        modules = plots.resolve_inputs(config)
        directories = ([module_plot_directory(config, name) for name in modules] if stage == 'plots'
                       else [multimodule_plot.destination(config)])
        paths = resolve_paths(config)['target']
        directories += [Path(paths['covariances']), Path(paths['noise_model_fits']),
                        Path(paths['covariances']).parent/'statistics']
        files = artifact_files(directories) | dependencies_files(config)
        for info in modules.values():
            for source in info['sources']:
                file = Path(source['input_file'])
                files.update((file, file.parent/'manifest.json', file.parent.parent/'manifest.json'))
                for correction in source['corrections'].values():
                    files.add(Path(correction['path']))
                    predictor = Path(correction['predictor'])
                    files.update(artifact_files([predictor.parent]))
        if not files or not all((Path(p)/'manifest.json').is_file() for p in directories[:len(modules) if stage == 'plots' else 1]):
            raise ValueError('Plot stage returned without its completed figure manifests')
        return sorted(files)
    complete, files = probe(config, stage)
    if not complete:
        raise ValueError(f'{stage} returned without complete verified outputs')
    return files
