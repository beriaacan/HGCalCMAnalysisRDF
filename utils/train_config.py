"""Explicit local/HTCondor training choices, separate from input provenance."""
from __future__ import annotations
import math
import re

KEYS={'training_grouping','training_device','training_seed','training_environment',
      'training_resume','training_deterministic','early_stopping_patience','early_stopping_min_delta',
      'lr_scheduler_factor','lr_scheduler_patience','lr_scheduler_threshold','lr_scheduler_min_lr',
      'optimizer_betas','optimizer_epsilon','condor_cpus','condor_gpus','condor_memory_gb',
      'condor_job_flavour','condor_submit','condor_use_x509_proxy','condor_proxy',
      'condor_work_dir','condor_setup_file','condor_minimum_cuda_runtime',
      'cublas_workspace_config','cublaslt_workspace_size_kib',
      'training_events_per_chunk','training_prefetch_batches','training_pin_memory','training_input_backend'}


def parse(settings,base,local_path):
    out={key:settings[key] for key in KEYS}
    for key in ('condor_job_flavour',):
        if not isinstance(out[key],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',out[key]):
            raise ValueError(f'{key} must be a simple nonempty name')
    for key,choices in [('training_grouping',{'combined','per_module'}),
                        ('training_device',{'cpu','cuda','auto'}),
                        ('training_input_backend',{'root','numeric_cache'})]:
        if out[key] not in choices: raise ValueError(f'{key} must be one of {sorted(choices)}')
    for key in ('training_resume','training_deterministic','condor_submit','condor_use_x509_proxy','training_pin_memory'):
        if type(out[key]) is not bool: raise ValueError(f'{key} must be true or false')
    for key,minimum in [('training_seed',0),('early_stopping_patience',1),('lr_scheduler_patience',0),
                        ('condor_cpus',1),('condor_gpus',0),('condor_memory_gb',1),
                        ('training_events_per_chunk',1),('training_prefetch_batches',0)]:
        if type(out[key]) is not int or out[key]<minimum: raise ValueError(f'{key} must be an integer >= {minimum}')
    if out['training_seed']>=2**32: raise ValueError('training_seed must be below 2**32')
    if out['condor_gpus']>1: raise ValueError('Only single-GPU training is implemented; condor_gpus must be 0 or 1')
    if not isinstance(out['condor_minimum_cuda_runtime'],str) or not re.fullmatch(r'\d+\.\d+',out['condor_minimum_cuda_runtime']):
        raise ValueError('condor_minimum_cuda_runtime must be a quoted CUDA version, e.g. "12.6"')
    for key in ('early_stopping_min_delta','lr_scheduler_factor','lr_scheduler_threshold',
                'lr_scheduler_min_lr','optimizer_epsilon'):
        value=out[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite numeric')
        if value<0 or (key=='optimizer_epsilon' and value==0): raise ValueError(f'Invalid {key}')
    if not 0<out['lr_scheduler_factor']<1: raise ValueError('lr_scheduler_factor must be between zero and one')
    workspace=out['cublas_workspace_config']
    if not isinstance(workspace,str) or not re.fullmatch(r'(?::[0-9]+:[0-9]+)+',workspace):
        raise ValueError('cublas_workspace_config must use :SIZE_KiB:COUNT pairs')
    lt=out['cublaslt_workspace_size_kib']
    if type(lt) is not int or lt<0: raise ValueError('cublaslt_workspace_size_kib must be a nonnegative integer')
    capacity=sum(int(size)*int(count) for size,count in re.findall(r':([0-9]+):([0-9]+)',workspace))
    if capacity<lt: raise ValueError('cuBLAS workspace must be at least cublaslt_workspace_size_kib')
    if out['training_deterministic'] and workspace not in {':4096:8',':16:8'}:
        raise ValueError('Deterministic training requires cublas_workspace_config :4096:8 or :16:8')
    beta=out['optimizer_betas']
    if (not isinstance(beta,list) or len(beta)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) or not 0<=v<1 for v in beta)):
        raise ValueError('optimizer_betas must contain two values in [0,1)')
    for key in ('condor_work_dir','condor_setup_file','training_environment'):
        out[key]=local_path(out[key],base,key)
    if out['condor_proxy'] is not None: out['condor_proxy']=local_path(out['condor_proxy'],base,'condor_proxy')
    return out
