"""Source-level RDF statistics, keyed ROOT joins and mergeable sufficient statistics."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

from .prepare_rdf import analysis_view, initialize_root
from .nano import root_path

_READY = False


def initialize():
    global _READY
    ROOT = initialize_root()
    if not _READY:
        header = Path(__file__).with_name('plot_helpers.hxx')
        if not ROOT.gInterpreter.Declare('#ifndef EIGEN_DONT_PARALLELIZE\n#define EIGEN_DONT_PARALLELIZE\n#endif\n'
                                         f'#include {json.dumps(str(header))}'):
            raise RuntimeError('Could not compile plotting statistics helpers')
        _READY = True
    return ROOT


@contextmanager
def source_view(config, source):
    ROOT = initialize()
    handles = []
    try:
        base = ROOT.TFile.Open(root_path(source['input_file']), 'READ'); handles.append(base)
        if not base or base.IsZombie():
            raise OSError(f'Cannot open prepared plotting input: {source["input_file"]}')
        tree = base.Get('Events')
        if not tree or int(tree.GetEntries()) != source['input_entries']:
            raise ValueError('Prepared plotting entry count changed')
        correction = source.get('correction')
        if correction:
            handle = ROOT.TFile.Open(root_path(correction['path']), 'READ'); handles.append(handle)
            if not handle or handle.IsZombie():
                raise OSError(f'Cannot open applied predictions: {correction["path"]}')
            main = handle.Get('Predictions')
            if not main or int(main.GetEntries()) != correction['selected_events']:
                raise ValueError('Applied prediction entry count changed')
            if source['input_entries']:
                tree.BuildIndex('source_entry')
                ROOT.hgcal_cm_plot.check_index(tree, source['input_entries'])
            main.AddFriend(tree, 'prepared')
            node = ROOT.RDataFrame(main)
            # Unique prepared branches are already available unqualified through
            # ROOT friends. Re-aliasing those existing column names is invalid.
            other = ','.join('prepared.'+column for column in
                             ('source_run','source_file_index','source_entry','event_id','event'))
        else:
            node = ROOT.RDataFrame(tree)
            other = 'source_run,source_file_index,source_entry,event_id,event'
        node = node.Filter('hgcal_cm_plot::identity(source_run,source_file_index,source_entry,event_id,event,'
                           f'{other},{source["run"]}U,{source["source_file_index"]},{source["input_entries"]}ULL)',
                           'matching physical source identities')
        node = analysis_view(node, config.target).Filter('module_present', 'module present for plotting')
        yield node
    finally:
        node = None # release RDF readers before their caller-owned TTrees/TFiles
        for handle in reversed(handles):
            if handle: handle.Close()


def collect(config, source, nch, nerx, requested):
    """Collect all missing contributions from one source in a single RDF loop."""
    import numpy as np
    ROOT = initialize()
    flags = [('covariance','uncorrected'),('covariance','analytic_allinputs'),('cm',None),
             ('histograms','uncorrected'),('histograms','analytic_allinputs')]
    cr,cp,cc,hr,hp = [key in requested for key in flags]
    options = config.plotting
    ranges = sum([options[f'plot_{name}_range'] for name in ('adc','mip','sum')], []) + options['plot_noise_window']
    bins = [options[f'plot_{name}_bins'] for name in ('adc','mip','sum')]
    collector = ROOT.hgcal_cm_plot.Collector(nch,nerx,config.resources['events_per_chunk'],cr,cp,cc,hr,hp,
                  ROOT.VecOps.RVec['double'](ranges),ROOT.VecOps.RVec['int'](bins))
    with source_view(config, source) as node:
        raw = 'adc_selected' if cr or hr else 'ROOT::RVec<float>{}'
        corrected = 'adc_corrected' if cp or hp else 'ROOT::RVec<float>{}'
        cm = 'cm_erx' if cr or cp or cc else 'ROOT::RVec<float>{}'
        try:
            node = node.Define('plot_payload',f'hgcal_cm_plot::Row{{{raw},{corrected},{cm},source_entry}}')
            collector.consume(ROOT.RDF.AsRNode(node))
        finally:
            del node
    events = int(collector.events)
    if source.get('correction') and events != source['correction']['selected_events']:
        raise ValueError('Applied events no longer match the configured plotting selection')
    result = {}
    for key, enabled, pair in [(flags[0],cr,collector.raw_cov),(flags[1],cp,collector.corrected_cov),
                                (flags[2],cc,collector.cm_cov)]:
        if enabled:
            result[key] = {name: np.asarray(getattr(pair,name),dtype=np.float64).copy().reshape(pair.rows,pair.cols)
                           for name in ('n','mx','my','m2')}
            result[key]['events'] = np.asarray(events,dtype=np.int64)
    for key, enabled, hist in [(flags[3],hr,collector.raw_hist),(flags[4],hp,collector.corrected_hist)]:
        if enabled:
            data = {name: np.asarray(getattr(hist,name),dtype=np.float64).copy() for name in ('adc','mip','sum','flow')}
            for mode in ('full','window'):
                for name in ('n','mean','m2'):
                    data[f'{mode}_{name}'] = np.asarray(getattr(getattr(hist,mode),name),dtype=np.float64).copy()
            for name in ('adc','mip','sum'):
                data[name+'_edges'] = np.linspace(*options[f'plot_{name}_range'],options[f'plot_{name}_bins']+1)
            data['events'] = np.asarray(events,dtype=np.int64)
            result[key] = data
    return result


def merge(parts, kind):
    import numpy as np
    if not parts: raise ValueError('No source contributions to combine')
    out = {key: value.copy() for key,value in parts[0].items()}
    for other in parts[1:]:
        if kind in {'cm','covariance'}:
            n,m = out['n'],other['n']; total=n+m
            ratio = np.divide(m,total,out=np.zeros_like(m),where=total>0)
            factor = n*ratio; dx=other['mx']-out['mx'];dy=other['my']-out['my']
            out['m2'] += other['m2']+dx*dy*factor
            out['mx'] += dx*ratio;out['my'] += dy*ratio;out['n']=total
        else:
            for mode in ('full','window'):
                n,m=out[mode+'_n'],other[mode+'_n'];total=n+m
                ratio=np.divide(m,total,out=np.zeros_like(m),where=total>0)
                delta=other[mode+'_mean']-out[mode+'_mean']
                out[mode+'_m2']+=other[mode+'_m2']+delta*delta*n*ratio
                out[mode+'_mean']+=delta*ratio;out[mode+'_n']=total
            for name in ('adc','mip','sum','flow'): out[name]+=other[name]
            for name in ('adc','mip','sum'):
                if not np.array_equal(out[name+'_edges'],other[name+'_edges']):
                    raise ValueError('Histogram definitions differ between sources')
        out['events']+=other['events']
    return out


def covariance(pair):
    import numpy as np
    return np.divide(pair['m2'],pair['n'],out=np.full_like(pair['m2'],np.nan),where=pair['n']>0)


def joint_covariance(pair, cm):
    import numpy as np
    block,cc = covariance(pair),covariance(cm)
    a = block.shape[0]
    matrix = np.block([[block[:,:a],block[:,a:]],[block[:,a:].T,cc]])
    counts = np.block([[pair['n'][:,:a],pair['n'][:,a:]],[pair['n'][:,a:].T,cm['n']]])
    return {'covariance':matrix,'counts':counts}


def channel_statistics(hist, mode):
    import numpy as np
    n=hist[mode+'_n']
    mean=np.where(n>0,hist[mode+'_mean'],np.nan)
    variance=np.divide(hist[mode+'_m2'],n,out=np.full_like(n,np.nan),where=n>0)
    return mean,np.sqrt(np.maximum(variance,0)),n
