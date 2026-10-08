"""Qualify frozen checkpoints with a repeatable full-size backward reference.

Primary timing workers retain their original default execution settings. Only
Torch correctness checks enable deterministic algorithms and a cuBLAS workspace.
Tensor correctness checks retain the existing numerical implementation.
"""
import hashlib
import os
from pathlib import Path
import sys

assert sys.argv[1]=='full_qualification'
backend=sys.argv[sys.argv.index('--backend')+1]
if backend=='torch':os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

import torch
import latent_checkpoint_sweep as sweep

torch.use_deterministic_algorithms(backend=='torch')
original_backward=sweep.backward
original_save=sweep.save
repeat_checks={}


def backward(model,x,y,policy):
    loss,gradients=original_backward(model,x,y,policy)
    if policy=='none':
        repeat_loss,repeat_gradients=original_backward(model,x,y,policy)
        metrics=sweep.gradient_metrics(repeat_gradients,gradients)
        assert repeat_loss==loss
        assert metrics['maximum_parameter_relative_l2']<1e-5,metrics
        repeat_checks[backend]=dict(loss=loss,gradient_error=metrics)
    return loss,gradients


def save(stem,out):
    out['qualification_controls']=dict(
        torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        repeat_uncheckpointed_checks=repeat_checks,
        primary_performance_settings_changed=False)
    source=Path(__file__)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot=sweep.OUT/'sources'/digest/source.name
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_bytes(source.read_bytes())
    out['qualification_driver_source']=dict(path=str(source.relative_to(sweep.base.ROOT)),sha256=digest)
    original_save(stem,out)


sweep.backward=backward
sweep.save=save
sweep.main()
