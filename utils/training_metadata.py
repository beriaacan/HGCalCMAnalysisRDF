"""Self-contained automatic data identity, owned by each newly trained model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import tempfile
import shutil

from .analytic_artifacts import sha, signature
from .identity import data_id, model_directory

SCHEMA = 'hgcal_cm_training_data_v2'
FILENAME = 'training_data.json'


def destination(config, group):
    return model_directory(config, group)


def _verify(data, model):
    try:
        expected, payload = model['definition'], data['definition']
        recipe = payload['training_recipe']
        if (data['schema'] != SCHEMA or data['signature'] != signature(payload)
                or model['signature'] != signature(expected)
                or payload['model_signature'] != model['signature']
                or payload['recipe_id'] != expected['recipe_id']
                or payload['dataset_id'] != data_id(recipe, expected['modules'])
                or payload['dataset_id'] != expected['dataset_id']
                or payload['training_signature'] != expected['training_signature']
                or signature(recipe) != expected['training_signature']
                or payload['training_manifest_sha256'] != expected['training_manifest_sha256']
                or payload['trained_modules'] != expected['modules']
                or recipe['feature_spec'] != expected['feature_spec']):
            raise ValueError('Training data does not match the saved model recipe')
        for source in recipe['prepared']:
            definition = source['preparation_definition']
            if (signature(definition) != source['signature']
                    or definition['run'] != source['run'] or definition['files'] != source['files']):
                raise ValueError('Invalid signed preparation record in training metadata')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed training-data record') from error
    return data


def declaration(path, data):
    return {'dataset_id': data['definition']['dataset_id'], 'file': FILENAME,
            'signature': data['signature'], 'sha256': sha(path)}


def read(folder, model, *, require_declaration=True):
    path = Path(folder)/FILENAME
    if not path.is_file():
        raise ValueError(f'Missing new-format training-data record: {folder}; train a new model')
    data = _verify(json.loads(path.read_text()), model)
    if require_declaration and model.get('training_data') != declaration(path, data):
        raise ValueError(f'Training-data declaration/checksum mismatch: {folder}')
    return data


def check_model_identity(folder, data):
    folder = Path(folder)
    if folder.parent.name.startswith('rdf2_') and folder.parent.name != data['definition']['recipe_id']:
        raise ValueError(f'Model directory and saved automatic recipe ID differ: {folder}')


def build(expected):
    path = Path(expected['training_inputs'])/'manifest.json'
    if not path.is_file() or sha(path) != expected['training_manifest_sha256']:
        raise ValueError(f'Missing or changed training manifest: {path}')
    manifest = json.loads(path.read_text())
    if (manifest['signature'] != expected['training_signature']
            or signature(manifest['definition']) != expected['training_signature']
            or manifest['preprocessing_sha256'] != expected['preprocessing_sha256']):
        raise ValueError('Training manifest differs from the model definition')
    payload = {'dataset_id': expected['dataset_id'], 'recipe_id': expected['recipe_id'],
               'model_signature': signature(expected), 'training_signature': expected['training_signature'],
               'training_manifest_sha256': expected['training_manifest_sha256'],
               'trained_modules': expected['modules'], 'training_recipe': manifest['definition']}
    data = {'schema': SCHEMA, 'definition': payload, 'signature': signature(payload)}
    return _verify(data, {'definition': expected, 'signature': signature(expected)})


def _write_data(folder, data):
    descriptor, name = tempfile.mkstemp(prefix='.training-data-',suffix='.tmp',dir=folder)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
        os.replace(name, Path(folder)/FILENAME)
    finally:
        Path(name).unlink(missing_ok=True)


def ensure(config, folder, expected, *, owns_training_lock=False):
    from analytic_allinputs import _write_manifest
    folder = Path(folder)
    if (folder/'.training-lock').exists() and not owns_training_lock:
        raise ValueError(f'Training-data update blocked by training lock: {folder}')
    lock = folder/'.training-data-lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f'Training-data lock DIRECTORY exists: {lock}') from None
    try:
        (lock/'owner.json').write_text(json.dumps({'host':socket.gethostname(),'pid':os.getpid()}))
        model = json.loads((folder/'manifest.json').read_text())
        if (model.get('signature') != signature(expected)
                or model.get('definition') != json.loads(json.dumps(expected))):
            raise ValueError(f'Model changed before metadata publication: {folder}')
        data = read(folder,model,require_declaration=False) if (folder/FILENAME).exists() else build(expected)
        if not (folder/FILENAME).exists():
            _write_data(folder,data)
        model['training_data'] = declaration(folder/FILENAME,data)
        _write_manifest(folder,model)
        return data
    finally:
        shutil.rmtree(lock)


def attach(folder, model):
    path = Path(folder)/FILENAME
    if path.exists():
        data = read(folder,model,require_declaration=False)
        model = dict(model,training_data=declaration(path,data))
    elif model.get('complete'):
        raise ValueError('Completed models require their automatic training-data record')
    return model


def summary(data):
    payload = data['definition']; recipe = payload['training_recipe']
    calibrations = sorted({(row['preparation_definition']['pedestal_run'],
                           row['preparation_definition']['calibration_sha256']) for row in recipe['prepared']})
    return {'dataset_id':payload['dataset_id'],'recipe_id':payload['recipe_id'],
            'campaign':recipe['campaign'],'runs':[row['run'] for row in recipe['prepared']],
            'selection':recipe['selection'],'trigger_time_window':recipe['trigger_time_window'],
            'adc_min':recipe['adc_min'],'adc_max':recipe['adc_max'],
            'split':recipe['options'],'calibrations':calibrations}
