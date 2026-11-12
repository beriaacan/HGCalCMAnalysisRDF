"""Checked, resumable Nano copy-and-append export; no inference or event selection."""
from __future__ import annotations

import inspect
import json
import os
import shutil
import socket
from pathlib import Path
import tempfile

from analytic_allinputs import _write_manifest
from .analytic_artifacts import sha, signature
from .inference_config import method_directory
from .nano import root_path
from .paths import resolve_paths
from .prepare_rdf import initialize_root

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA = 'hgcal_cm_augmented_nano_v2'
PROVENANCE_KEY = 'HGCalCMAnalysisExport'
METHODS = ('analytic', 'analytic_allinputs', 'dnn')
_READY = False


def methods(config):
    unsupported = set(config.methods)-{'uncorrected', *METHODS}
    if unsupported:
        raise NotImplementedError(f'Nano export methods not implemented: {sorted(unsupported)}')
    selected = tuple(m for m in METHODS if m in config.methods)
    if not selected:
        raise ValueError('Nano export needs at least one correction method; original ADCs are already in the Nano')
    return selected


def destination(config):
    tag = '__'.join('-'.join(method_directory(config, m).parts) for m in methods(config))
    return Path(resolve_paths(config)['target']['nano_exports'])/tag


def branches(method):
    return {'correction': f'HGCDigi_adcCorrection_{method}',
            'corrected': f'HGCDigi_adcCorrected_{method}',
            'valid': f'HGCDigi_adcCorrectionValid_{method}'}


def describe(config):
    return {'destination': str(destination(config)), 'methods': methods(config),
            'branches': {m: branches(m) for m in methods(config)},
            'source': 'original Nanos; existing checked application predictions; no new inference',
            'layout': 'Run<run>/<original Nano filename>; <stem>.manifest.json beside each ROOT file',
            'preservation': 'byte-copy original file; append branches only; retain all original events, branches and metadata',
            'corrected_adc': 'float32(HGCDigi_adc - saved predicted correction); Nano pedestal offset retained',
            'alignment': 'unique uint64 Nano event within the pinned original source; module FED/readout-sequence/channel; cache IDs are not physical entry numbers',
            'invalid': 'NaN correction/corrected ADC and zero validity for unprocessed, rejected or non-ADC digis',
            'valid_dtype': 'vector<unsigned char>, 0 or 1', 'provenance_key': PROVENANCE_KEY,
            'publication': 'one original Nano at a time; matching completed exports reused; incompatible exports refused'}


def initialize():
    global _READY
    ROOT = initialize_root()
    if ROOT.IsImplicitMTEnabled():
        ROOT.DisableImplicitMT()
    if not _READY:
        header = Path(__file__).with_name('nano_export.hxx')
        if not ROOT.gInterpreter.Declare(f'#include {json.dumps(str(header))}'):
            raise RuntimeError('Could not compile Nano export helper; activate the current hgc environment')
        _READY = True
    return ROOT


def _code():
    import plots
    central = Path(os.environ['CMSSW_BASE'])/'src/HGCalCommissioning/LocalCalibration/python/DigiAnalysisUtils.py'
    import hashlib
    result = {name: sha(ROOT_DIR/name) for name in ('export.py', 'utils/nano_export.py', 'utils/nano_export.hxx', 'utils/nano.py')}
    result['plots.py:resolve_inputs'] = hashlib.sha256(inspect.getsource(plots.resolve_inputs).encode()).hexdigest()
    result['central/DigiAnalysisUtils.py'] = sha(central)
    return result


def resolve(config):
    """Reuse the application/source preflight, without collecting plotting statistics."""
    import plots
    selected = methods(config)
    modules = plots.resolve_inputs(config)
    files, preparations, applications = {}, {}, {}
    for name, info in modules.items():
        for source in info['sources']:
            manifest_path = Path(source['input_file']).parent/'manifest.json'
            if manifest_path not in preparations:
                preparations[manifest_path] = json.loads(manifest_path.read_text())
            prepared = preparations[manifest_path]
            recipe = prepared['definition']
            original = next(r for r in recipe['files'] if r['index']==source['source_file_index'])
            if (original['uuid'] != source['source_uuid'] or original['entries'] != source['input_entries']
                    or recipe['run'] != source['run'] or prepared['signature'] != source['prepared_signature']):
                raise ValueError(f'Nano/prepared physical source identity differs for {name}')
            common_recipe = {k:v for k,v in recipe.items() if k not in {'modules','module_order'}}
            common_signature = signature(common_recipe)
            key = f'Run{source["run"]}/{Path(original["path"]).name}'
            if key not in files:
                files[key] = {'original': original, 'run': source['run'], 'modules': {},
                              'prepared': {'signature': common_signature, 'definition': common_recipe},
                              'applications': {}, 'output_file': Path(original['path']).name}
            unit = files[key]
            if unit['original'] != original or unit['prepared']['signature'] != common_signature:
                raise ValueError(f'Conflicting original Nano/preparation identities for {key}')
            if name in unit['modules']:
                raise ValueError(f'Duplicate export module/source: {name}/{key}')
            identity = list(source['module_identity']['identity'])
            if len(identity)!=3 or info['nch']!=37*identity[2]:
                raise ValueError(f'Invalid export module/channel mapping: {name}')
            corrections = {}
            for method in selected:
                correction = source['corrections'][method]
                corrections[method] = correction
                path = Path(correction['path']).parents[1]/'manifest.json'
                if path not in applications:
                    record = json.loads(path.read_text())
                    definition = {k:v for k,v in record['definition'].items() if k!='modules'}
                    applications[path] = {'signature': signature(definition), 'definition': definition}
                current = applications[path]
                if method in unit['applications'] and unit['applications'][method] != current:
                    raise ValueError(f'Mixed application definitions for {method}/{key}')
                unit['applications'][method] = current
            unit['modules'][name] = {'identity': identity, 'nch': info['nch'], 'corrections': corrections}
    if not files:
        raise ValueError('No original Nano files selected for export')
    code = _code()
    for key, unit in files.items():
        pairs = [tuple(v['identity'][:2]) for v in unit['modules'].values()]
        if len(pairs)!=len(set(pairs)):
            raise ValueError(f'Multiple export modules map to the same FED/readout sequence: {key}')
        definition = {'schema': SCHEMA, 'source': unit['original'], 'run': unit['run'],
            'modules': unit['modules'], 'prepared': unit['prepared'], 'applications': unit['applications'],
            'methods': list(selected), 'branches': {m: branches(m) for m in selected},
            'formula': 'float32(original HGCDigi_adc - float32 saved predicted correction); retain pedestal',
            'alignment': 'unique uint64 Nano event within source run/file/UUID; independently validate internal cache entry/event-ID consistency',
            'invalid': 'NaN and validity=0; no extension of saved application coverage',
            'provenance_key': PROVENANCE_KEY, 'code_sha256': code}
        unit['definition'] = json.loads(json.dumps(definition))
    return {'destination': destination(config), 'methods': selected, 'files': files}


def _record_path(directory, filename):
    return Path(directory)/(Path(filename).stem+'.manifest.json')


def cached_unit(directory, definition):
    """Check one flat ROOT file and its neighboring completion record."""
    directory = Path(directory)
    filename = Path(definition['source']['path']).name
    file = directory/filename
    path = _record_path(directory, filename)
    if not path.is_file():
        if file.exists():
            raise ValueError(f'Incomplete Nano export: {file}; completion record missing: {path}. '
                             'Preserve the unfinished output before rebuilding')
        return None
    record = json.loads(path.read_text())
    if (record.get('schema')!=SCHEMA or record.get('complete') is not True
            or record.get('signature')!=signature(definition) or record.get('definition')!=definition):
        raise ValueError(f'Nano export provenance mismatch: {path}; preserve the existing export under another name')
    output = record.get('output', {})
    if output.get('file')!=filename:
        raise ValueError(f'Invalid Nano export output inventory: {path}')
    if not file.is_file():
        raise ValueError(f'Missing exported Nano: {file}')
    stat = file.stat()
    if (stat.st_size!=output.get('size') or stat.st_mtime_ns!=output.get('mtime_ns')
            or output.get('events')!=definition['source']['entries']):
        raise ValueError(f'Exported Nano was modified or is incomplete: {file}')
    return record


def _stat(path, size, mtime):
    value = Path(path).stat()
    if value.st_size!=size or value.st_mtime_ns!=mtime:
        raise ValueError(f'Export source changed: {path}')


def compute(config, unit, work, selected):
    ROOT = initialize()
    from HGCalCommissioning.LocalCalibration import DigiAnalysisUtils as central
    source = unit['original']
    _stat(source['path'], source['size'], source['mtime_ns'])
    for info in unit['modules'].values():
        for item in info['corrections'].values():
            _stat(item['path'], item['size'], item['mtime_ns'])
    output = work/unit['output_file']
    print(f'Copying original Nano: {source["path"]}', flush=True)
    if not ROOT.TFile.Cp(root_path(source['path']), str(output), False):
        raise OSError(f'Could not copy original Nano to {output}')
    _stat(source['path'], source['size'], source['mtime_ns'])
    if output.stat().st_size!=source['size']:
        raise OSError('Copied Nano size differs from the original before augmentation')
    print(f'Adding correction branches to {source["entries"]:,} original events', flush=True)
    writer = ROOT.hgcal_cm_export.Writer(str(output), source['uuid'], source['entries'],
        unit['run'], source['index'], source['entry_offset'], ROOT.std.vector['string'](list(selected)),
        config.resources['events_per_chunk'])
    try:
        for name, info in unit['modules'].items():
            fed, sequence, _ = info['identity']
            for method in selected:
                item = info['corrections'][method]
                writer.add(method, name, fed, sequence, info['nch'], root_path(item['path']), item['selected_events'])
        node = ROOT.RDataFrame('Events', root_path(source['path']))
        node = central.addStandardADCSelection(node, nofilter=True)
        node = node.Define('hgcal_export_payload', 'hgcal_cm_export::Payload{ULong64_t(rdfentry_),ULong64_t(event),'
            'hgcal_cm_export::copy_unsigned(HGCDigi_fedId),hgcal_cm_export::copy_unsigned(HGCDigi_fedReadoutSeq),'
            'hgcal_cm_export::copy_unsigned(HGCDigi_channel),hgcal_cm_export::copy_unsigned(HGCDigi_adc),'
            'hgcal_cm_export::copy_unsigned(good_digiadc)}')
        writer.consume(ROOT.RDF.AsRNode(node))
        provenance = {'schema': SCHEMA, 'signature': signature(unit['definition']), 'definition': unit['definition']}
        writer.finish(json.dumps(provenance, sort_keys=True, allow_nan=False))
        summary = {'events': int(writer.events), 'valid_digis': {m: int(writer.valid(m)) for m in selected}}
    finally:
        del writer
    for info in unit['modules'].values():
        for item in info['corrections'].values():
            _stat(item['path'], item['size'], item['mtime_ns'])
    stat = output.stat()
    return dict(summary, file=unit['output_file'], size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def run(config):
    print('Checking original Nano and saved application provenance for export', flush=True)
    resolved = resolve(config)
    destination = resolved['destination']
    overview_path = destination/'manifest.json'
    if overview_path.is_file() and json.loads(overview_path.read_text()).get('schema')!=SCHEMA:
        raise ValueError(f'Export directory uses a different format: {destination}. '
                         'Remove or move the old exports before starting a fresh export')
    # Check every completed destination before creating any new copies.
    existing = {key: cached_unit(destination/Path(key).parent, unit['definition'])
                for key, unit in resolved['files'].items()}
    destination.mkdir(parents=True, exist_ok=True)
    records = dict(existing)
    def overview(complete):
        _write_manifest(destination, {'schema': SCHEMA, 'complete': complete, 'methods': list(resolved['methods']),
            'files': {key: {'signature': rec['signature'], 'output': rec['output']}
                      for key, rec in records.items() if rec is not None}})
    overview(False)
    for key, unit in resolved['files'].items():
        directory = destination/Path(key).parent
        target = directory/unit['output_file']
        record_path = _record_path(directory, unit['output_file'])
        if existing[key] is not None:
            print(f'Reusing augmented Nano: {target}', flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        # A file rename may replace an existing file. Serialize producers before
        # checking/publishing it, so a second exporter cannot overwrite a result.
        lock = directory/f'.{target.stem}.export-lock'
        try:
            lock.mkdir()
        except FileExistsError:
            raise ValueError(f'Nano export lock DIRECTORY exists: {lock}; '
                             'verify its recorded owner before cleanup') from None
        work = None
        try:
            (lock/'owner.json').write_text(json.dumps({'host': socket.gethostname(), 'pid': os.getpid()}))
            current = cached_unit(directory, unit['definition'])
            if current is not None:
                records[key] = current
                overview(False)
                print(f'Reusing augmented Nano: {target}', flush=True)
                continue
            work = Path(tempfile.mkdtemp(prefix=f'.{target.stem}.work-', dir=directory))
            output = compute(config, unit, work, resolved['methods'])
            record = {'schema': SCHEMA, 'complete': True, 'signature': signature(unit['definition']),
                      'definition': unit['definition'], 'output': output}
            _write_manifest(work, record)
            if target.exists() or record_path.exists():
                raise FileExistsError(f'Nano export appeared during production: {target}')
            # Close the ROOT file in compute, publish it, then publish its
            # completion record last. An interruption cannot claim an incomplete
            # Nano is reusable; an orphan file is refused by cached_unit.
            os.rename(work/unit['output_file'], target)
            os.rename(work/'manifest.json', record_path)
            work.rmdir()
            records[key] = record
            overview(False)
            print(f'Augmented Nano ready: {target}', flush=True)
        except KeyboardInterrupt:
            print(f'Export interrupted/failed; completed files remain reusable. Unfinished work: {work}', flush=True)
            raise
        except Exception as exc:
            raise RuntimeError(f'{exc}. Completed exports remain reusable; unfinished work: {work}') from exc
        finally:
            shutil.rmtree(lock)
    overview(True)
    print(f'Nano export ready: {destination}', flush=True)
