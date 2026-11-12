"""Expand scientific goals, prune checked products and combine shared producers."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .workflow_config import resources
from .workflow_products import from_settings, request_key, probe, lock_paths, output_paths, ROOT


class Planner:
    def __init__(self, workflow, *, refresh_plots=False):
        self.workflow = workflow
        self.nodes = {}
        self.refresh_plots = refresh_plots
        self.goals = []
        from .training_batch import WORKER_CODE
        from .workflow_registry import digest, evidence
        self.code_key = digest(evidence([ROOT/name for name in WORKER_CODE]))

    def build(self):
        for entry in self.workflow['analyses']:
            selected = []
            for stage in entry['steps']:
                selected.extend(self.ensure(entry['settings'], stage, entry['source']))
            # Both stages can write covariance/fit caches. Order them if both requested.
            for name in selected:
                node = self.nodes[name]
                if node['stage'] == 'multimodule_plot' and not node['complete']:
                    node['parents'] = sorted(set(node['parents']) | {
                        n for n in selected if self.nodes[n]['stage'] == 'plots'})
            self.goals.append({'config': entry['source'], 'steps': entry['steps'], 'nodes': selected})
        for node in self.nodes.values():
            if node['stage'] == 'multimodule_plot' and not node['complete']:
                node['parents'] = sorted(set(node['parents']) | {p['name'] for p in self.nodes.values()
                    if p['stage'] == 'plots' and set(p['locks']) & set(node['locks'])})
        for goal in self.goals:
            pending, seen = list(goal['nodes']), set()
            while pending:
                name = pending.pop()
                if name in seen:
                    continue
                seen.add(name)
                node = self.nodes[name]
                if goal['config'] not in node['sources']:
                    node['sources'].append(goal['config'])
                pending.extend(node['parents'])
        # Merged coverage keeps its dependencies. Show producers before consumers.
        from graphlib import TopologicalSorter
        ordered = TopologicalSorter({n: row['parents'] for n, row in self.nodes.items()}).static_order()
        for node in self.nodes.values():
            if node['complete']:
                node['resources']['gpus'] = 0  # At most a CPU metadata verification; never recompute a pruned producer.
        return {'schema': 'hgcal_cm_workflow_v1', 'workflow': self.workflow,
                'nodes': [self.nodes[name] for name in ordered], 'goals': self.goals,
                'refresh_plots': self.refresh_plots}

    def ensure(self, settings, stage, source, *, calibration=False):
        if stage == 'prepare':
            return self.preparations(settings, source)
        if stage == 'pedestals' and not calibration:
            return self.calibrations(settings, source)
        settings = deepcopy(settings)
        settings['steps'] = [stage]
        if stage in {'training_inputs', 'analytic', 'analytic_allinputs', 'training_cache', 'train'}:
            from .config_defaults import DATA_DEFAULTS, DATA_REQUIRED
            row = settings['derivation']
            settings['application'].update({key: deepcopy(row[key]) for key in
                DATA_REQUIRED | DATA_DEFAULTS.keys() if key in row})
        config = from_settings(settings)
        key = request_key(config, stage)
        name = stage+'_'+key[:16]
        if name in self.nodes:
            if stage == 'pedestals' and not self.nodes[name]['complete']:
                previous = from_settings(self.nodes[name]['settings'])
                if (previous.correction.pedestal != config.correction.pedestal
                        or previous.pedestal_options != config.pedestal_options):
                    raise ValueError('Conflicting pedestal production recipes target one calibration file')
                left, right = previous.correction.modules, config.correction.modules
                if left != right:
                    if left == 'all' or right == 'all' or {s for s in (left, right) if isinstance(s, str)} == {'ld', 'hd'}:
                        merged = 'all'
                    elif isinstance(left, str) or isinstance(right, str):
                        from types import SimpleNamespace
                        merged = self.modules(SimpleNamespace(selections=(left, right),
                            source=config.correction.pedestal.sources[0]))
                    else:
                        merged = sorted(set(left) | set(right))
                    for role in ('derivation', 'application'):
                        self.nodes[name]['settings'][role]['modules'] = merged
            if self.nodes[name]['resources'] != resources(self.workflow, stage, config):
                raise ValueError(f'Conflicting resources for shared stage {name}; use one workflow resource profile')
            if source not in self.nodes[name]['sources']:
                self.nodes[name]['sources'].append(source)
            return [name]
        complete, files = probe(config, stage)
        if stage in {'plots', 'multimodule_plot'} and not self.refresh_plots:
            import json
            from .workflow_registry import digest, intact
            request = digest({'recipe': key, 'coverage': None, 'code': self.code_key})
            receipt = Path(self.workflow['state_dir'])/'products'/f'{request}.json'
            if receipt.is_file():
                saved = json.loads(receipt.read_text())
                if saved.get('request') == request and saved.get('complete') and intact(saved.get('evidence', [])):
                    complete, files = True, [p['path'] for p in saved['evidence']]
        node = {'name': name, 'stage': stage, 'request': key, 'settings': settings,
                'sources': [source], 'complete': complete, 'files': [str(p) for p in files],
                'parents': [], 'resources': resources(self.workflow, stage, config),
                'locks': lock_paths(config, stage), 'outputs': output_paths(config, stage)}
        self.nodes[name] = node
        if stage != 'pedestals':
            node['parents'] = self.calibrations(settings, source)
        if complete:
            return [name]
        parents = list(node['parents'])
        if stage == 'pedestals':
            import pedestals
            pedestals.plan(config, for_execution=True)  # Existing central/overwrite guards.
        elif stage == 'training_inputs':
            parents += self.preparations(settings, source, role='derive')
        elif stage in {'analytic', 'analytic_allinputs', 'training_cache'}:
            parents += self.ensure(settings, 'training_inputs', source)
        elif stage == 'train':
            parents += self.ensure(settings, 'training_inputs', source)
            if config.training['training_input_backend'] == 'numeric_cache':
                parents += self.ensure(settings, 'training_cache', source)
        elif stage == 'apply':
            parents += self.preparations(settings, source, role='apply')
            for method in config.methods:
                if method != 'uncorrected':
                    parents += self.ensure(settings, 'train' if method == 'dnn' else method, source)
        elif stage in {'plots', 'multimodule_plot', 'export'}:
            parents += self.ensure(settings, 'apply', source)
        else:
            raise ValueError(f'Unsupported workflow stage: {stage}')
        node['parents'] = sorted(set(parents))
        return [name]

    def preparations(self, settings, source, role=None):
        from dataclasses import replace
        from .prepared import plan, cache_directory, derivation_modules
        from .paths import discover_files
        from .config import Dataset
        config = from_settings(settings)
        result = []
        for job in plan(config):
            if role is not None and role not in job.roles:
                continue
            discover_files(Dataset(job.source.run, (job.source,), None, None))
            chosen = 'derivation' if 'derive' in job.roles else 'application'
            if role is not None:
                chosen = 'derivation' if role == 'derive' else 'application'
                selection = config.correction.modules if role == 'derive' else config.target.modules
                if role == 'apply' and selection == 'all':
                    selection = derivation_modules(config)
                job = replace(job, selections=(selection,))
            selected = self.modules(job)
            value = self.preparation_settings(settings, chosen, job.source.run, selected)
            single = from_settings(value)
            key = request_key(single, 'prepare')
            name = 'prepare_'+key[:16]
            if name in self.nodes:
                node = self.nodes[name]
                if node['resources'] != resources(self.workflow, 'prepare', single):
                    raise ValueError(f'Conflicting resources for shared preparation {name}')
                combined = sorted(set(node['settings']['derivation']['modules']) | set(selected))
                node['settings']['derivation']['modules'] = combined
                node['settings']['application']['modules'] = combined
                single = from_settings(node['settings'])
                complete, files = probe(single, 'prepare')
                node.update(complete=complete, files=[str(p) for p in files])
                if source not in node['sources']:
                    node['sources'].append(source)
            else:
                complete, files = probe(single, 'prepare')
                node = {'name': name, 'stage': 'prepare', 'request': key, 'settings': value,
                        'sources': [source], 'complete': complete, 'files': [str(p) for p in files],
                        'parents': [], 'resources': resources(self.workflow, 'prepare', single),
                        'locks': lock_paths(single, 'prepare'), 'directory': str(cache_directory(plan(single)[0]))}
                node['outputs'] = [node['directory']]
                self.nodes[name] = node
            node['parents'] = self.calibrations(node['settings'], source)
            result.append(name)
        return list(dict.fromkeys(result))

    def calibrations(self, settings, source):
        """Always retain one check/production prerequisite per configured file."""
        result = []
        for role in ('derivation', 'application'):
            row = settings[role]
            selection = row['modules']
            if role == 'application' and selection == 'all':
                selection = settings['derivation']['modules']
            value = self.preparation_settings(settings, role, row['pedestal_run'], selection)
            for block in ('derivation', 'application'):
                value[block]['input_pattern'] = row['pedestal_input_pattern']
            result += self.ensure(value, 'pedestals', source, calibration=True)
        return list(dict.fromkeys(result))

    @staticmethod
    def modules(job):
        explicit = {name for selection in job.selections if not isinstance(selection, str) for name in selection}
        selectors = [selection for selection in job.selections if isinstance(selection, str)]
        if selectors:
            from .paths import discover_files
            from .config import Dataset
            from .prepared import _metadata_modules
            dataset = Dataset(job.source.run, (job.source,), None, None)
            for file in discover_files(dataset):
                stat = Path(file).stat()
                for name in _metadata_modules(file, stat.st_size, stat.st_mtime_ns):
                    if any(s == 'all' or s == 'ld' and name.startswith('ML_')
                           or s == 'hd' and name.startswith('MH_') for s in selectors):
                        explicit.add(name)
        if not explicit:
            raise ValueError(f'No preparation modules selected for Run{job.source.run}')
        return sorted(explicit)

    @staticmethod
    def preparation_settings(settings, role, run, modules):
        from .config_defaults import DATA_DEFAULTS, DATA_REQUIRED
        value = deepcopy(settings)
        row = settings[role]
        common = {key: deepcopy(row[key]) for key in DATA_REQUIRED | DATA_DEFAULTS.keys() if key in row}
        common.update(runs=[run], modules=modules if isinstance(modules, str) else list(modules))
        for block in ('derivation', 'application'):
            value[block].update(common)
        value['steps'] = ['prepare']
        return value


def build(workflow, *, refresh_plots=False):
    return Planner(workflow, refresh_plots=refresh_plots).build()


def show(plan):
    from .workflow_registry import owner_at, describe_owner
    print('Workflow plan (metadata only; no processing or submission)')
    for goal in plan['goals']:
        print(f'  {goal["config"]}: {", ".join(goal["steps"])}')
    for node in plan['nodes']:
        action = 'reuse verified product' if node['complete'] else 'render/update using numerical caches' if node['stage'] in {'plots', 'multimodule_plot'} else 'run/resume if still needed'
        print(f'\n{node["name"]}: {action}')
        if len(node['sources']) > 1:
            print(f'  shared by {len(node["sources"])} configs')
        print(f'  parents: {", ".join(node["parents"]) or "none"}')
        print(f'  resources: {node["resources"]}')
        for output in node['outputs']:
            print(f'  output: {output}')
        for lock in node['locks']:
            if Path(lock).exists():
                print('  '+describe_owner(lock, owner_at(lock)))
    print(f'\n{len(plan["nodes"])} distinct stages; '
          f'{sum(n["complete"] for n in plan["nodes"])} verified complete. '
          'Missing upstream products may leave final artifact IDs pending until execution.')
