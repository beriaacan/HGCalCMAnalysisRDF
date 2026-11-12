"""Actual DNN application implementation hashes, without compatibility fallbacks."""
from .analytic_artifacts import sha
def current_hashes(root,code):
    return {name:sha(root/name) for name in dict.fromkeys((*code,'utils/model_selection.py',
            'utils/model_catalogue.py','utils/identity.py'))}
