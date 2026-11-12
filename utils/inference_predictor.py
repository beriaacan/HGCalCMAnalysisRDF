"""Reuse the verified expanders for inference, without training weights or splits."""
from dataclasses import replace
import numpy as np


class Predictor:
    def __init__(self, config, checkpoint, spec, module, device):
        import torch
        from .dnn_training import build_model
        self.torch, self.device = torch, device
        self.config, self.spec, self.module = config, spec, module
        self.pre = checkpoint['preprocessing']
        self.model = build_model(checkpoint['feature_spec'], checkpoint['options']).to(device)
        self.model.load_state_dict(checkpoint['model_state'], strict=True)
        from .dnn_internal_normalization import validate_model
        validate_model(self.model)
        self.model.eval()
        self.expander = None
        if device == 'cuda':
            from .dnn_tensors import TensorExpander
            # Existing expander also returns training y/weights. Supply a
            # dummy target mask and disable optional weights; only the input
            # construction is consumed during application.
            no_weights = replace(config, dnn=dict(config.dnn, sample_weighting='none'))
            self.expander = TensorExpander(no_weights, {'feature_spec': spec}, self.pre, device)
            self.module_index = sorted(spec['modules']).index(module)

    def prepare(self, batch):
        if self.expander is None:
            return batch
        import torch
        mask = np.where(np.isfinite(batch['targets']), np.float32(0), np.float32(np.nan))
        payload = {'features': torch.as_tensor(batch['features']), 'targets': torch.as_tensor(mask),
                   'modules': torch.full((len(mask),), self.module_index, dtype=torch.int64),
                   'runs': batch['identities'][:, 0], 'samples': int(np.isfinite(mask).sum())}
        if self.config.application['apply_pin_memory']:
            for key in ('features', 'targets', 'modules'):
                payload[key] = payload[key].pin_memory()
        return dict(batch, payload=payload)

    def predict(self, batch):
        from .training import iter_sample_batches
        torch = self.torch
        targets = batch['targets']
        predictions = np.full_like(targets, np.nan)
        limit = self.config.dnn['infer_batch']

        def forward(x, rows, channels):
            if not len(channels):
                return
            x = torch.as_tensor(x, device=self.device)
            for begin in range(0, len(x), limit):
                end = begin+limit
                values = self.model(x[begin:end]).squeeze(-1).cpu().numpy()
                cc = channels[begin:end]
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite DNN prediction in physical ADC units')
                predictions[rows[begin:end], cc] = values

        with torch.inference_mode():
            if self.expander is not None:
                if batch['payload']['samples']:
                    x, _, _ = self.expander.expand(batch['payload'])
                    rows, channels = np.nonzero(np.isfinite(targets))
                    forward(x, rows, channels)
            else:
                for part in iter_sample_batches(batch['features'], targets, self.spec, self.module, limit, self.pre):
                    forward(part['inputs'], part['event_rows'], part['channels'])
        corrected = targets-predictions
        if not np.array_equal(np.isfinite(predictions), np.isfinite(targets)):
            raise ValueError('DNN application changed the accepted channel mask')
        if not np.isfinite(corrected[np.isfinite(targets)]).all():
            raise ValueError('Nonfinite corrected application ADC')
        return predictions, corrected
