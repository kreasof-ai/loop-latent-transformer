"""Qualify frozen checkpoints with a repeatable full-size backward reference.

Primary timing workers retain their original default execution settings. Only
Torch correctness checks enable deterministic algorithms and a cuBLAS workspace.
Tensor correctness checks retain the existing numerical implementation.
"""

# Support both direct script execution and python -m experiments.l40s.<module>.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hashlib
import os

def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        from experiments.l40s import latent_checkpoint_sweep as sweep
        sweep.main()
        return
    assert len(sys.argv)>1 and sys.argv[1]=='full_qualification'
    backend=sys.argv[sys.argv.index('--backend')+1]
    if backend=='torch':os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

    import torch
    from experiments.l40s import latent_checkpoint_sweep as sweep

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
    try:
        sweep.main()
    finally:
        sweep.backward=original_backward
        sweep.save=original_save



if __name__ == '__main__':
    main()
