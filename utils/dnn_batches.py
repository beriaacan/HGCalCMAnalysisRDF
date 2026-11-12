"""Bounded ROOT/TMVA event loading and reuse of the agreed sample expansion."""
from __future__ import annotations
from dataclasses import replace
import atexit
import numpy as np

from .analytic_fit import _sources
from .nano import root_path
from .prepare_rdf import analysis_view
from .training import (initialize_training_root,global_expression,source_salt,
                       iter_sample_batches,sample_weights)


def source_chunks(config,manifest,module,path,source,partition, *, include_entries=False):
    """Use ROOT's unfiltered loader; filtered vector loading is broken in 6.36."""
    from .datasets import source_config,stream_id
    config=source_config(config,source)
    ROOT=initialize_training_root()
    from ROOT._pythonization._tmva._batchgenerator import CreateNumPyGenerators
    spec=manifest['feature_spec'];meta=spec['modules'][module]
    node=ROOT.RDataFrame('Events',root_path(str(path)))
    # Reuse the ADC mask without a Filter: ROOT's filtered RBatchGenerator
    # incorrectly strides expanded vector columns in this installed release.
    node=analysis_view(node,replace(config.correction,selection='full'))
    node=node.Define('training_global_features',global_expression(config,spec,module))
    salt=source_salt(config.correction.campaign,source['run'],source['uuid'])
    seed=config.dnn['split_seed']
    split=f'hgcal_cm_training::split(ULong64_t(event),{salt}ULL,{seed}ULL,{config.dnn["validation_split_fraction"]:.17g})'
    node=node.Define('dnn_partition',split)
    retained=(f'hgcal_cm_training::retained(ULong64_t(event),{salt}ULL,{seed}ULL,dnn_partition,'
              f'{config.dnn["train_event_fraction"]:.17g},{config.dnn["validation_event_fraction"]:.17g})')
    keep=f'module_present && {retained}'
    if partition is not None: keep+=f' && dnn_partition=={partition}'
    if config.correction.selection=='trigtime':
        lo,hi=config.correction.trigger_time_window
        keep+=f' && (source_is_pedestal || (trig_time>{lo:.17g} && trig_time<{hi:.17g}))'
    if config.correction.selection not in {'full','trigtime'}:
        raise NotImplementedError('DNN training requires full or trigtime derivation selection')
    node=node.Define('dnn_keep',f'float({keep})')
    # The ROOT loader transports scalar columns as float32. Split physical
    # uint64 entries into exact 16-bit pieces rather than losing ID precision.
    extras=[]
    if include_entries:
        for i in range(4):
            name=f'dnn_entry_{i}'
            node=node.Define(name,f'float((source_entry >> {16*i}) & 65535ULL)')
            extras.append(name)
        node=node.Define('dnn_cache_partition','float(dnn_partition)')
        extras.append('dnn_cache_partition')
    size=config.training['training_events_per_chunk']
    def selected(batch):
        chosen=batch[batch[:,-1]==1].copy()
        if not len(chosen): return
        features=chosen[:,:spec['global_size']]
        targets=chosen[:,spec['global_size']:spec['global_size']+meta['nch']]
        valid=np.isfinite(targets).any(axis=1)
        if valid.any():
            result=(module,stream_id(source),features[valid],targets[valid])
            if include_entries:
                start=spec['global_size']+meta['nch']
                pieces=chosen[valid,start:start+4].astype(np.uint64)
                entries=np.bitwise_or.reduce(pieces << np.arange(0,64,16,dtype=np.uint64),axis=1)
                result+=entries,chosen[valid,start+4].astype(np.int8)
            yield result
    entries=source['entries']
    full_chunks=entries//size
    columns=['training_global_features','adc_selected']+extras+['dnn_keep']
    if full_chunks:
        generator,_=CreateNumPyGenerators(node,batch_size=size,chunk_size=size,columns=columns,
            max_vec_sizes={'training_global_features':spec['global_size'],'adc_selected':meta['nch']},
            validation_split=0.,max_chunks=full_chunks,shuffle=False,drop_remainder=False)
        # ROOT's default binding holds Python's GIL while waiting for its C++
        # loader. Release it so our producer cannot block GPU submission in the
        # main thread. These graphs contain C++ expressions, no Python callbacks.
        generator.base_generator.generator.GetTrainBatch.__release_gil__=True
        generator.base_generator.generator.DeActivate.__release_gil__=True
        try:
            for batch in generator:
                yield from selected(batch)
        finally:
            generator.DeActivate()
            atexit.unregister(generator.base_generator.DeActivate)
    if entries%size:
        # The central loader requests start+size even at EOF. Reuse it for all
        # full chunks, then use its native RDF range mechanism for an exact tail.
        # No ROOT warnings/errors are suppressed, and the tail remains bounded.
        ROOT.Internal.RDF.ChangeBeginAndEndEntries(ROOT.RDF.AsRNode(node),full_chunks*size,entries)
        # Book all central AsNumpy actions, then run the graph through a public
        # Count result with the GIL released. All booked actions run in that same
        # pass; GetValue below only converts already-ready results to NumPy.
        result=node.AsNumpy(columns,lazy=True)
        ready=node.Count()
        ready.GetValue.__release_gil__=True
        ready.GetValue()
        tail=result.GetValue()
        batch=np.column_stack([np.stack(tail['training_global_features']),
                               np.stack(tail['adc_selected'])]+[tail[name] for name in extras]+[tail['dnn_keep']])
        yield from selected(batch)


def event_chunks(config,manifest,modules,partition,rng, *, chunk_reader=None):
    reader=source_chunks if chunk_reader is None else chunk_reader
    sources=[(module,path,source) for module in modules for path,source in _sources(manifest,module)]
    if partition==0: rng.shuffle(sources)
    counts={}
    for module,path,source in sources:
        for chunk in reader(config,manifest,module,path,source,partition):
            from .datasets import stream_id
            key=(module,stream_id(source))
            count=np.isfinite(chunk[3]).sum(axis=0)
            counts[key]=counts.get(key,np.zeros_like(count))+count
            yield chunk
    label=('train','validation')[partition]
    for module in modules:
        for run,row in manifest['counts_by_module_run'][module].items():
            expected=np.asarray(row[label][:manifest['feature_spec']['modules'][module]['nch']])
            actual=counts.get((module,int(run)),np.zeros_like(expected))
            if not np.array_equal(actual,expected):
                raise ValueError(f'Reconstructed DNN {label} counts differ from training_inputs: {module}/Run{run}')


def expand_buffer(config,manifest,preprocessing,chunks,partition,rng):
    spec=manifest['feature_spec'];width=spec['max_channels']
    names=sorted(spec['modules']);ids={name:i for i,name in enumerate(names)}
    areas=np.asarray([spec['modules'][name]['cell_area_fraction']+
                     [0.]*(width-spec['modules'][name]['nch']) for name in names],dtype=np.float32)
    features=np.concatenate([row[2] for row in chunks])
    # Target storage only needs the largest actual module in this buffer. This
    # does NOT shrink the common 467-feature input representation or its padding.
    target_width=max(row[3].shape[1] for row in chunks)
    targets=np.concatenate([np.pad(row[3],((0,0),(0,target_width-row[3].shape[1])),constant_values=np.nan) for row in chunks])
    module_ids=np.concatenate([np.full(len(row[2]),ids[row[0]],dtype=np.int32) for row in chunks])
    runs=np.concatenate([np.full(len(row[2]),row[1],dtype=np.int64) for row in chunks])
    # Shuffle indices, preserving the identical event order without duplicating
    # the entire feature/target buffers just to permute their rows.
    order=rng.permutation(len(features)) if partition==0 else np.arange(len(features))
    # Target masking, local channel/eRx features and normalization come from the
    # existing bounded expander. Replace its synthetic area with each real module's.
    mixed=dict(spec,modules={'__mixed__':{'nch':target_width,'cell_area_fraction':[0.]*target_width}})
    area_column=spec['feature_names'].index('cell_area_fraction') if 'cell_area_fraction' in spec['local_features'] else None
    # An optimizer batch consists of full module-event rows, followed by ALL
    # accepted channel targets from those rows. Never split an event's targets
    # across optimizer steps. Reuse the agreed masking/normalization expander.
    batch_events=config.dnn['batch_events']
    for start in range(0,len(features),batch_events):
        stop=min(start+batch_events,len(features))
        events=order[start:stop]
        selected_targets=targets[events]
        samples=int(np.isfinite(selected_targets).sum())
        if not samples: continue
        inputs=np.empty((samples,len(spec['feature_names'])),dtype=np.float32)
        values=np.empty(samples,dtype=np.float32)
        weights=np.ones(samples,dtype=np.float32)
        offset=0
        # Reuse the existing exact masker/normalizer in small CPU pieces. Assemble
        # ONE full-event optimizer batch; no intermediate optimizer updates or
        # gradient accumulation. Bound gather/normalization temporaries to 8192
        # channel-target rows instead of several full hundreds-of-MiB matrices.
        for batch in iter_sample_batches(features[events],selected_targets,mixed,'__mixed__',8192,preprocessing):
            rr,cc=events[batch['event_rows']],batch['channels'];mm=module_ids[rr];run=runs[rr]
            if area_column is not None:
                batch['inputs'][:,area_column]=areas[mm,cc]
            batch_weights=weights[offset:offset+len(cc)]
            if config.dnn['sample_weighting']!='none':
                for mi in np.unique(mm):
                    for ri in np.unique(run[mm==mi]):
                        mask=(mm==mi)&(run==ri)
                        batch_weights[mask]=sample_weights(manifest,names[mi],int(ri),('train','validation')[partition],
                                                    cc[mask],weighting=config.dnn['sample_weighting'])
            inputs[offset:offset+len(cc)]=batch['inputs']
            values[offset:offset+len(cc)]=batch['targets']
            offset+=len(cc)
        if offset!=samples: raise RuntimeError('Full-event batch expansion lost channel targets')
        if 'stream_table' in manifest and partition==1:
            mapping={int(k):v['dataset_index'] for k,v in manifest['stream_table'].items()}
            rr,_cc=np.nonzero(np.isfinite(selected_targets))
            datasets=np.asarray([mapping[int(run)] for run in runs[events[rr]]],dtype=np.int64)
            yield inputs,values,weights,datasets
        else:
            yield inputs,values,weights
        del inputs,values,weights


def batches(config,manifest,preprocessing,modules,partition,epoch, *, chunk_reader=None):
    rng=np.random.default_rng(np.random.SeedSequence([config.training['training_seed'],epoch,partition]))
    buffer=[]
    limit=config.dnn['shuffle_buffer_chunks'] if partition==0 else 1
    for chunk in event_chunks(config,manifest,modules,partition,rng,chunk_reader=chunk_reader):
        buffer.append(chunk)
        if len(buffer)==limit:
            yield from expand_buffer(config,manifest,preprocessing,buffer,partition,rng)
            buffer=[]
    if buffer: yield from expand_buffer(config,manifest,preprocessing,buffer,partition,rng)
