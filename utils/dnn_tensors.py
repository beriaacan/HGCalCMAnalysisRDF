"""Compact event streaming and the numerically verified tensor input expander.

Ported from the standalone CUDA prototype validated on the June 20-batch sample.
ROOT selection/count checks and the sample-weight definition remain shared with
the existing CPU path. Never construct expanded CPU inputs in this CUDA path.
"""
from __future__ import annotations

import numpy as np


def compact_batches(config, manifest, chunks, partition, rng):
    """Exactly the production buffer order, without channel-expanded inputs."""
    names=sorted(manifest['feature_spec']['modules'])
    ids={name:i for i,name in enumerate(names)}
    width=max(row[3].shape[1] for row in chunks)
    features=np.concatenate([row[2] for row in chunks])
    targets=np.concatenate([np.pad(row[3],((0,0),(0,width-row[3].shape[1])),constant_values=np.nan) for row in chunks])
    modules=np.concatenate([np.full(len(row[2]),ids[row[0]],dtype=np.int64) for row in chunks])
    runs=np.concatenate([np.full(len(row[2]),row[1],dtype=np.int64) for row in chunks])
    order=rng.permutation(len(features)) if partition==0 else np.arange(len(features))
    size=config.dnn['batch_events']
    for start in range(0,len(features),size):
        events=order[start:start+size]
        y=targets[events]
        samples=int(np.isfinite(y).sum())
        if samples:
            yield {'features':features[events], 'targets':y, 'modules':modules[events],
                   'runs':runs[events], 'samples':samples}


def compact_event_batches(config,manifest,preprocessing,modules,partition,epoch, *, chunk_reader=None):
    """Stream the same event order/boundaries and run the same count checks."""
    from .dnn_batches import event_chunks
    rng=np.random.default_rng(np.random.SeedSequence([config.training['training_seed'],epoch,partition]))
    source=event_chunks(config,manifest,modules,partition,rng,chunk_reader=chunk_reader)
    buffer=[]
    limit=config.dnn['shuffle_buffer_chunks'] if partition==0 else 1
    try:
        for chunk in source:
            buffer.append(chunk)
            if len(buffer)==limit:
                yield from compact_batches(config,manifest,buffer,partition,rng)
                buffer=[]
        if buffer:yield from compact_batches(config,manifest,buffer,partition,rng)
    finally:
        source.close()


def compact_cpu_batch(payload, *, pin_memory):
    """Stage only event arrays in the CPU producer; no CUDA kernels here."""
    import torch
    tensors={name:torch.as_tensor(payload[name]) for name in ('features','targets','modules')}
    if pin_memory:tensors={name:value.pin_memory() for name,value in tensors.items()}
    # Runs are only needed for the small optional weight lookup on the consumer.
    return dict(tensors,runs=payload['runs'],samples=payload['samples'])


class TensorExpander:
    def __init__(self, config, manifest, preprocessing, device, partition=0):
        import torch
        self.torch=torch;self.device=torch.device(device)
        self.spec=manifest['feature_spec'];self.pre=preprocessing
        self.names=sorted(self.spec['modules'])
        from .preprocessing import validate_descriptor
        validate_descriptor(preprocessing,self.spec)
        self.adc=torch.tensor(self.spec['adc_columns'],dtype=torch.int64,device=self.device)
        self.all_own=all(col>=0 for col in self.spec['adc_columns'])
        self.any_own=any(col>=0 for col in self.spec['adc_columns'])
        width=self.spec['max_channels']
        area=[self.spec['modules'][n]['cell_area_fraction']+[0.]*(width-self.spec['modules'][n]['nch']) for n in self.names]
        self.area=torch.tensor(area,dtype=torch.float32,device=self.device)
        self.weight_table=None
        self.dataset_ids=({int(k):v['dataset_index'] for k,v in manifest['stream_table'].items()}
                          if 'stream_table' in manifest and partition==1 else None)
        if config.dnn['sample_weighting']!='none':
            from .training import sample_weights
            self.runs=sorted({int(run) for n in self.names for run in manifest['counts_by_module_run'][n]})
            self.run_ids={run:i for i,run in enumerate(self.runs)}
            table=np.full((len(self.names),len(self.runs),width),np.nan,dtype=np.float32)
            label=('train','validation')[partition]
            for mi,name in enumerate(self.names):
                for run,row in manifest['counts_by_module_run'][name].items():
                    counts=np.asarray(row[label]);channels=np.flatnonzero(counts>0)
                    if len(channels):
                        table[mi,self.run_ids[int(run)],channels]=sample_weights(manifest,name,int(run),partition=label,
                            channels=channels,weighting=config.dnn['sample_weighting'])
            self.weight_table=torch.tensor(table,device=self.device)

    def _transfer(self,value):
        value=self.torch.as_tensor(value)
        return value.to(self.device,non_blocking=value.device.type=='cpu' and value.is_pinned())

    def expand(self, payload):
        torch=self.torch
        with torch.no_grad():
            g=self._transfer(payload['features'])
            target=self._transfer(payload['targets'])
            modules=self._transfer(payload['modules'])
            # CUDA nonzero synchronizes to obtain its output size; include that
            # cost in the measured prototype instead of hiding it in setup.
            rows,channels=torch.nonzero(torch.isfinite(target),as_tuple=True)
            n=rows.numel()
            if n!=payload['samples']: raise ValueError('Tensor expansion changed accepted target count')
            x=torch.empty((n,len(self.spec['feature_names'])),dtype=torch.float32,device=self.device)
            x[:,:self.spec['global_size']]=g.index_select(0,rows)
            # Mask the target's own raw ADC BEFORE applying normalization.
            if self.all_own:
                x[torch.arange(n,device=self.device),self.adc[channels]]=self.spec['missing_input_value']
            elif self.any_own:
                own=self.adc[channels];valid=own>=0
                x[torch.arange(n,device=self.device)[valid],own[valid]]=self.spec['missing_input_value']
            area_column=None
            for offset,feature in enumerate(self.spec['local_features'],self.spec['global_size']):
                if feature=='channel_in_erx': x[:,offset]=channels%37
                elif feature=='erx_index': x[:,offset]=torch.div(channels,37,rounding_mode='floor')
                elif feature=='cell_area_fraction':
                    # Match the production helper's synthetic area followed by
                    # its real mixed-module area override after normalization.
                    x[:,offset]=0.;area_column=offset
                else: raise ValueError(f'Unsupported local feature: {feature}')
            if area_column is not None:
                area=self.area[modules[rows],channels]
                x[:,area_column]=area
            y=target[rows,channels]
            if self.weight_table is None:
                weights=torch.ones(n,dtype=torch.float32,device=self.device)
            else:
                run_ids=np.asarray([self.run_ids[int(run)] for run in payload['runs'].tolist()],dtype=np.int64)
                run_ids=self._transfer(run_ids)
                weights=self.weight_table[modules[rows],run_ids[rows],channels]
            if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(weights).all()):
                raise ValueError('Nonfinite tensor-expanded inputs, targets or weights')
            if self.dataset_ids is not None:
                ids=np.asarray([self.dataset_ids[int(run)] for run in payload['runs']],dtype=np.int64)
                return x,y,weights,self._transfer(ids)[rows]
            return x,y,weights
