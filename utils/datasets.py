"""Resolve each source's own conditions without merging physical campaigns."""
from dataclasses import replace


def recipes(config):
    return config.derivations or (config.correction,)


def single(config, recipe):
    return replace(config, correction=recipe, derivations=())


def source_config(config, source):
    rows = recipes(config)
    return single(config, rows[source.get('dataset_index', 0)])


def stream_id(source):
    return source.get('stream_id', source['run'])


def calibration_label(recipe):
    if recipe.conditions:
        return 'conditions_' + recipe.conditions['era'].replace('/', '') + '_Run' + str(recipe.conditions['run'])
    return f'pedestals_{recipe.pedestal.run}'


def blocks(settings):
    row = settings['derivation']
    return row.get('datasets', [row])


def all_blocks(settings):
    return [*blocks(settings), settings['application']]


def source_settings(settings, index):
    import copy
    value = copy.deepcopy(settings)
    value['derivation'] = copy.deepcopy(blocks(settings)[index])
    return value


def stream_table(prepared):
    """Small stable integer transport IDs; keys and real runs remain in metadata."""
    keys = list(dict.fromkeys((row['dataset_index'], row['run']) for row in prepared))
    return {str(i): {'dataset_index': dataset, 'run': run} for i, (dataset, run) in enumerate(keys)}
