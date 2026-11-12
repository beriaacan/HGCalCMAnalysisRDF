"""Actual implementation hashes only; no historical compatibility aliases."""
from .analytic_artifacts import sha
EXTRA=('utils/training_progress.py','utils/training_runtime.py','utils/identity.py')
def current_hashes(root,code):
    return {name:sha(root/name) for name in dict.fromkeys((*code,*EXTRA))}
