"""Actual CM-only application sources for the fresh artifact format."""
from .analytic_artifacts import sha
def code_hashes(root):
    names=('utils/cm_application.py','apply.py','utils/application.py','utils/apply_helpers.hxx',
           'utils/training.py','utils/training_helpers.hxx','utils/prepare_rdf.py',
           'utils/prepare_helpers.hxx','utils/identity.py')
    return {name:sha(root/name) for name in names}
