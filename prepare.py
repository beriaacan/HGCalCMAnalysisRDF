"""Persist reusable uncut, calibrated module inputs with ROOT RDataFrame."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from utils.config import AnalysisConfig
from utils.prepared import (cached_files, describe, plan, resolve, common_definition,
                            module_definition, module_record)
from utils.analytic_artifacts import signature
from analytic_allinputs import _write_manifest
from utils.nano import root_path, empty_dataframe
from utils.prepare_rdf import initialize_root, module_columns


def _produce(config: AnalysisConfig, resolved: dict) -> None:
    ROOT = initialize_root()
    destination = resolved['destination']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=True)
    common = common_definition(resolved)
    if not (destination/'manifest.json').exists():
        _write_manifest(destination, {'schema': common['schema'], 'definition': common,
                                     'signature': signature(common)})
    missing = {name: metadata for name, metadata in resolved['modules'].items()
               if module_record(resolved, name) is None}
    if not missing:
        return
    work = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.work-', dir=destination.parent))
    outputs = []
    job = resolved['job']
    print(f'Preparing Run{job.source.run}: {len(missing)} modules -> {destination}', flush=True)
    for info in resolved['files']:
        if info['entries']:
            rdf = ROOT.RDataFrame('Events', root_path(info['path']))
        else:
            print(f'Preserving empty source file: {info["path"]}', flush=True)
            rdf = empty_dataframe(ROOT, trigger_valid=info['has_trigger_valid'])
        rdf = rdf.Define('source_run', f'unsigned({job.source.run})')
        rdf = rdf.Define('source_file_index', str(info['index']))
        rdf = rdf.Define('source_entry', 'ULong64_t(rdfentry_)')
        rdf = rdf.Define('event_id', f'ULong64_t(rdfentry_)+ULong64_t({info["entry_offset"]})')
        rdf = rdf.Define('source_is_pedestal', str(job.source.run == job.pedestal_run).lower())
        rdf = rdf.Alias('trig_time', 'HGCMetaData_trigTime')
        rdf = rdf.Alias('orbit', 'orbitNumber').Alias('bx', 'bunchCrossing')
        base_columns = ['source_run', 'source_file_index', 'source_entry', 'event_id',
                        'source_is_pedestal', 'event', 'trig_time', 'orbit', 'bx']
        if info['has_trigger_valid']:
            rdf = rdf.Alias('trig_valid', 'HGCMetaData_trigValid')
            base_columns.append('trig_valid')
        if info['entries']:
            ROOT.RDF.Experimental.AddProgressBar(ROOT.RDF.AsRNode(rdf))
        snapshots = []
        for name, metadata in missing.items():
            node, columns = module_columns(rdf, name, metadata['identity'], metadata['calibration'],
                                           config.preparation['activity_thresholds'],metadata['absent_erx'])
            relative = Path(name) / Path(info['path']).name
            target = work / relative
            target.parent.mkdir(exist_ok=True)
            options = ROOT.RDF.RSnapshotOptions()
            options.fLazy = True
            options.fAutoFlush = config.resources['events_per_chunk']
            # Book all module snapshots/counts on the same graph before execution.
            snapshots.append((name, relative, node.Count(),
                              node.Snapshot('Events', str(target), base_columns + columns, options)))
        if snapshots:
            snapshots[0][3].GetValue()
        for name, relative, count, snapshot in snapshots:
            snapshot.GetValue()
            entries = int(count.GetValue())
            if entries != info['entries']:
                raise RuntimeError(f'Preparation lost source events for {name}: {entries} != {info["entries"]}')
            stat = (work / relative).stat()
            outputs.append({'module': name, 'source_file_index': info['index'],
                            'file': str(relative), 'entries': entries,
                            'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        print(f'Prepared source file {info["index"]+1}/{len(resolved["files"])}: {info["entries"]} events', flush=True)
    for name in missing:
        definition = module_definition(resolved, name)
        record = {'schema': definition['schema'], 'complete': True, 'definition': definition,
                  'signature': signature(definition),
                  'outputs': [row for row in outputs if row['module']==name]}
        _write_manifest(work/name, record)
        if (destination/name).exists():
            raise FileExistsError(f'Prepared module appeared during production; retained {work}')
        os.rename(work/name, destination/name)
        print(f'Prepared module ready: {destination/name}', flush=True)
    work.rmdir()
    print(f'Prepared cache ready: {destination}', flush=True)


def run(config: AnalysisConfig) -> None:
    ROOT = initialize_root()
    if ROOT.IsImplicitMTEnabled():
        ROOT.DisableImplicitMT()
    ROOT.EnableImplicitMT(config.resources['cpus'])
    # Preflight every source and calibration before starting any event processing.
    jobs = [resolve(config, job) for job in plan(config)]
    for resolved in jobs:
        if cached_files(resolved) is not None:
            print(f'Reusing prepared cache: {resolved["destination"]}', flush=True)
        else:
            import shutil, socket
            destination = resolved['destination']
            destination.parent.mkdir(parents=True, exist_ok=True)
            lock = destination.parent/f'.{destination.name}.preparation-lock'
            try:
                lock.mkdir()
            except FileExistsError:
                raise ValueError(f'Preparation lock DIRECTORY exists: {lock}; verify its recorded owner before cleanup') from None
            try:
                from utils.workflow_registry import record_lock_owner
                record_lock_owner(lock, {'host': socket.gethostname(), 'pid': os.getpid()})
                if cached_files(resolved) is None:
                    _produce(config, resolved)
            finally:
                shutil.rmtree(lock)
