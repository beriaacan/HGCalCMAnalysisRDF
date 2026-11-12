"""Inspect saved DNN metadata; application always uses the derivation recipe."""
from __future__ import annotations
import json
from pathlib import Path
from .analytic_artifacts import signature

def catalogue(root,model_dir=None):
    from .dnn_training import SCHEMA
    from .training_metadata import read,summary,check_model_identity
    root=Path(root);rows=[]
    paths=set(root.glob('*/dnn/rdf2_*/*/manifest.json'))
    if model_dir is not None:
        paths.update(Path(model_dir).glob('dnn/rdf2_*/*/manifest.json'))
    for path in sorted(paths):
        recipe,group=path.parent.parent.name,path.parent.name
        row={'model_id':recipe,'group':group,'directory':str(path.parent),'status':'invalid','trained_modules':[]}
        try:
            model=json.loads(path.read_text());definition=model['definition']
            if (model.get('schema')!=SCHEMA or model.get('signature')!=signature(definition)
                    or definition.get('recipe_id')!=recipe or type(model.get('complete')) is not bool):
                raise ValueError('Invalid new-format model definition')
            data=read(path.parent,model);check_model_identity(path.parent,data)
            row.update(status='complete' if model['complete'] else 'incomplete',
                       trained_modules=definition['modules'],training_data=summary(data),
                       preprocessing=definition.get('preprocessing'),options=definition['options'],
                       best_epoch=model.get('summary',{}).get('best_epoch'))
        except (OSError,ValueError,KeyError,TypeError) as error:
            row['error']=str(error)
        rows.append(row)
    return rows


def print_catalogue(config):
    root=config.output_dir/'models';print(f'Models directory: {root}')
    rows=catalogue(root,config.model_dir)
    if not rows: print('No new-format models found; create them with the training workflow.')
    for row in rows:
        print(f'\n{row["model_id"]}/{row["group"]} [{row["status"]}]')
        print('  Directory: '+row['directory'])
        print('  Trained modules: '+', '.join(row['trained_modules']))
        if 'training_data' in row:
            data=row['training_data']
            print('  Source campaign: '+data['campaign'])
            print('  Automatic data ID: '+data['dataset_id'])
            print('  Runs: '+', '.join(map(str,data['runs'])))
            print(f'  Selection: {data["selection"]}; target ADC bounds: {data["adc_min"]}, {data["adc_max"]}')
            options=data['split']
            print(f'  Validation split: {options["validation_split_fraction"]}; retained '
                  f'train={options["train_event_fraction"]}, validation={options["validation_event_fraction"]}')
        if row.get('preprocessing'):
            print('  Preprocessing: ADC/CM internal z-score; Nchannels/444 input; targets in ADC units')
            options=row['options']
            print(f'  Training: nodes={options["nodes"]}; batch_events={options["batch_events"]}; learning_rate={options["learning_rate"]}')
        if row.get('best_epoch') is not None: print(f'  Best epoch: {row["best_epoch"]}')
        if row.get('error'): print('  '+row['error'])
