"""Fixed normalization inside the DNN, serialized only in its model state.

Moments use the same weighted Welford/Chan population definition as
training_helpers.hxx. The NumPy adapter consumes existing compact event arrays;
it never materializes an events-times-targets-times-features training table.
"""
from __future__ import annotations

import numpy as np
import torch


class FixedStandardization(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('mean', torch.zeros(width, dtype=torch.float32))
        self.register_buffer('scale', torch.ones(width, dtype=torch.float32))
        self.register_buffer('fit_samples', torch.tensor(0, dtype=torch.int64))

    def forward(self, inputs):
        return (inputs - self.mean) / self.scale

    def initialize(self, mean, scale, samples):
        with torch.no_grad():
            self.mean.copy_(torch.as_tensor(mean, dtype=self.mean.dtype, device=self.mean.device))
            self.scale.copy_(torch.as_tensor(scale, dtype=self.scale.dtype, device=self.scale.device))
            self.fit_samples.fill_(int(samples))
        self.validate()

    def validate(self):
        if (self.fit_samples.item() <= 0 or not torch.isfinite(self.mean).all().item()
                or not torch.isfinite(self.scale).all().item() or not (self.scale > 0).all().item()):
            raise ValueError('Missing or invalid fitted DNN-internal normalization buffers')


def layer(model):
    return model.input_normalization if hasattr(model, 'input_normalization') else None


def validate_model(model):
    normalizer = layer(model)
    if normalizer is None:
        raise ValueError('DNN is missing its required fixed ADC/CM normalization layer')
    normalizer.validate()

def columns(spec):
    """Only measured ADC/CM columns are standardized; all other inputs pass through."""
    selected=[]
    if 'cm_erx' in spec['global_groups']:
        selected.extend(spec['feature_names'].index(f'cm_erx{i:02d}') for i in range(spec['max_channels']//37))
    if 'adc_allchannels' in spec['global_groups']:
        selected.extend(col for col in spec['adc_columns'] if col >= 0)
    return sorted(selected)


def fit_chunks(spec, chunks):
    """Population moments of the actual own-ADC-masked event/target inputs.

An event contributes n accepted targets. An ADC predictor has weight n-1
when it is itself an accepted target, plus one explicit fill observation.
Compute centered moments directly and merge with Chan's formula, rather than
subtracting large uncentered squared sums. Identity features stay mean 0/scale 1.
"""
    selected = columns(spec)
    own_channel = {col: channel for channel, col in enumerate(spec['adc_columns']) if col >= 0}
    mean = np.zeros(len(spec['feature_names']), dtype=np.float64)
    m2 = np.zeros_like(mean)
    count = 0
    fill = spec['missing_input_value']
    for module, _run, features, targets in chunks:
        if (features.ndim != 2 or features.shape[1] != spec['global_size']
                or targets.shape != (len(features), spec['modules'][module]['nch'])):
            raise ValueError('Invalid compact arrays while fitting DNN-internal normalization')
        valid = np.isfinite(targets)
        multiplicity = valid.sum(axis=1).astype(np.float64)
        total = int(multiplicity.sum())
        if not total:
            continue
        chunk_mean, chunk_m2 = np.zeros_like(mean), np.zeros_like(mean)
        for col in selected:
            values = features[:, col].astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite filled feature in DNN-internal normalization')
            channel = own_channel.get(col)
            hidden = valid[:, channel] if channel is not None and channel < targets.shape[1] else None
            weights = multiplicity if hidden is None else multiplicity - hidden
            observed = float(weights.sum())
            replaced = 0 if hidden is None else int(hidden.sum())
            mu = float(np.sum(weights * values) / observed) if observed else 0.
            variance_sum = float(np.sum(weights * (values - mu)**2)) if observed else 0.
            # Merge the fill observations without cancellation in the variance.
            delta = fill - mu
            chunk_mean[col] = mu + delta * replaced / total
            chunk_m2[col] = variance_sum + delta**2 * observed * replaced / total
        combined = count + total
        delta = chunk_mean - mean
        m2 += chunk_m2 + delta**2 * (count * total / combined)
        mean += delta * (total / combined)
        count = combined
    if not count:
        raise ValueError('No retained training targets for DNN-internal normalization')
    scale = np.ones_like(mean)
    variance = np.maximum(m2 / count, 0.)
    for col in selected:
        if variance[col] > 0:
            scale[col] = np.sqrt(variance[col])
    return mean, scale, count


def fit(config, manifest, modules, event_cache=None):
    from .dnn_batches import event_chunks
    from .training_progress import sample_total
    # A private RNG leaves optimizer initialization and epoch shuffles untouched.
    rng = np.random.default_rng(np.random.SeedSequence([config.training['training_seed'], 0x4E4F524D]))
    reader = event_cache.source_chunks if event_cache is not None else None
    expected = sample_total(manifest, modules, 0)
    print(f'Fitting fixed DNN-internal normalization: {expected:,} training targets; no validation', flush=True)
    source = event_chunks(config, manifest, modules, 0, rng, chunk_reader=reader)
    def progress():
        previous = None
        try:
            for chunk in source:
                key = chunk[:2]
                if key != previous:
                    print(f'  Normalization {key[0]}/Run{key[1]}', flush=True)
                    previous = key
                yield chunk
        finally:
            source.close()
    stream = progress()
    try:
        result = fit_chunks(manifest['feature_spec'], stream)
    finally:
        stream.close()
    if result[2] != expected:
        raise ValueError('DNN-internal normalization population differs from saved training counts')
    print('Fixed DNN-internal normalization fitted; buffers will be saved with model weights', flush=True)
    return result
