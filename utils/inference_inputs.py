"""Bounded central ROOT loading of application events, with exact physical IDs."""
from dataclasses import replace
import atexit
import numpy as np

from .nano import root_path
from .prepare_rdf import analysis_view
from .training import initialize_training_root, global_expression

IDENTITIES = ('source_run', 'source_file_index', 'source_entry', 'event_id', 'event')


def source_chunks(config, spec, module, source):
    ROOT = initialize_training_root()
    from ROOT._pythonization._tmva._batchgenerator import CreateNumPyGenerators
    if ROOT.IsImplicitMTEnabled():
        ROOT.DisableImplicitMT()
    node = ROOT.RDataFrame('Events', root_path(source['input_file']))
    # ROOT 6.36's filtered vector batch loader has a stride bug. Transport the
    # selection flag unfiltered, then select whole rows in the bounded batch.
    node = analysis_view(node, replace(config.target, selection='full'))
    node = node.Define('application_features', global_expression(config, spec, module, recipe=config.target))
    keep = 'module_present'
    if config.target.selection == 'trigtime':
        lo, hi = config.target.trigger_time_window
        keep += f' && (source_is_pedestal || (trig_time>{lo:.17g} && trig_time<{hi:.17g}))'
    elif config.target.selection != 'full':
        raise NotImplementedError('DNN application supports full or trigtime selections')
    node = node.Define('application_keep', f'float({keep})')
    pieces = []
    # RBatchGenerator transports scalar columns as float32. All five IDs are
    # encoded losslessly, including uint64 values above 2**53.
    for name in IDENTITIES:
        for i in range(4):
            col = f'application_{name}_{i}'
            node = node.Define(col, f'float((ULong64_t({name}) >> {16*i}) & 65535ULL)')
            pieces.append(col)
    columns = ['application_features', 'adc_selected', *pieces, 'application_keep']
    nch, width = spec['modules'][module]['nch'], spec['global_size']
    size = config.application['apply_events_per_chunk']
    entries, seen = source['input_entries'], 0

    def unpack(batch):
        selected = batch[batch[:, -1] == 1]
        if not len(selected):
            return None
        split = width+nch
        encoded = selected[:, split:split+20].astype(np.uint64).reshape(-1, 5, 4)
        ids = np.bitwise_or.reduce(encoded << np.arange(0, 64, 16, dtype=np.uint64), axis=2)
        if np.any(ids[:, 0] != source['run']) or np.any(ids[:, 1] != source['source_file_index']):
            raise ValueError('Application source identity differs from its manifest')
        if np.any(ids[:, 2] >= entries):
            raise ValueError('Application source entry outside the physical input range')
        return {'features': selected[:, :width].copy(), 'targets': selected[:, width:split].copy(),
                'identities': ids}

    full = entries//size
    if full:
        generator, _ = CreateNumPyGenerators(node, batch_size=size, chunk_size=size, columns=columns,
            max_vec_sizes={'application_features': width, 'adc_selected': nch},
            validation_split=0., max_chunks=full, shuffle=False, drop_remainder=False)
        generator.base_generator.generator.GetTrainBatch.__release_gil__ = True
        generator.base_generator.generator.DeActivate.__release_gil__ = True
        try:
            for batch in generator:
                seen += len(batch)
                result = unpack(batch)
                if result is not None:
                    yield result
        finally:
            generator.DeActivate()
            atexit.unregister(generator.base_generator.DeActivate)
    if entries % size:
        ROOT.Internal.RDF.ChangeBeginAndEndEntries(ROOT.RDF.AsRNode(node), full*size, entries)
        values = node.AsNumpy(columns, lazy=True)
        count = node.Count()
        count.GetValue.__release_gil__ = True
        count.GetValue()
        arrays = values.GetValue()
        batch = np.column_stack([np.stack(arrays['application_features']), np.stack(arrays['adc_selected'])]
                                + [arrays[name] for name in pieces] + [arrays['application_keep']])
        seen += len(batch)
        result = unpack(batch)
        if result is not None:
            yield result
    if seen != entries:
        raise ValueError(f'Application reader processed {seen} of {entries} expected input rows')


def event_batches(config, spec, module, source):
    stream = source_chunks(config, spec, module, source)
    # Bound expansion as well as network forward calls by infer_batch. A single
    # event may exceed that channel-row limit; forward calls still respect it.
    size = min(config.application['apply_batch_events'],
               max(1, config.dnn['infer_batch']//spec['modules'][module]['nch']))
    try:
        for chunk in stream:
            for begin in range(0, len(chunk['targets']), size):
                yield {key: value[begin:begin+size] for key, value in chunk.items()}
    finally:
        stream.close()
