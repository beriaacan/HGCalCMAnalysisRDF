"""Submit DNN training to Condor."""
from __future__ import annotations

from utils.config import AnalysisConfig


def describe(config):
    return {'trainer':'same train step and dedicated runtime as local execution',
            'work_directory':str(config.training['condor_work_dir']/config.model_dir.name),
            'grouping':config.training['training_grouping'],'modules':config.correction.modules,
            'cpus':config.training['condor_cpus'],'gpus':config.training['condor_gpus'],
            'memory_gb':config.training['condor_memory_gb'],'job_flavour':config.training['condor_job_flavour'],
            'minimum_cuda_runtime':config.training['condor_minimum_cuda_runtime'],
            'submit':config.training['condor_submit'],'setup_file':str(config.training['condor_setup_file']),
            'environment':str(config.training['training_environment']),
            'completion':'non-blocking; submission does not imply a trained model exists'}


def run(config: AnalysisConfig) -> None:
    from train import resolve_training
    from utils.dnn_training import cached,definition,destination
    from utils.training_runtime import environment
    from utils.training_batch import make_jobs,submit_jobs
    environment(config) # Check the dedicated interpreter exists before writing job files.
    print('Checking training-input provenance for HTCondor training',flush=True)
    directory,manifest,_,selected,_=resolve_training(config)
    pending={group:modules for group,modules in selected.items()
             if cached(destination(config,group),definition(config,manifest,directory,modules)) is None}
    if not pending:
        print('All requested DNN models are complete; no jobs submitted',flush=True)
        return
    work,jobs=make_jobs(config,pending)
    print(f'Training job files ready: {work}',flush=True)
    if config.training['condor_submit']:
        submit_jobs(work,jobs)
    else:
        print('condor_submit: false; no jobs submitted',flush=True)
