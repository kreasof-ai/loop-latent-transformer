# Inference helpers

The current inference code supports the measured prefill and supplied-token
cached-decode operations. The main study does not include sampling, beam search,
or a production generation service.

[ContextualLLT](../model/contextual_llt.py) defines current LLT decode, using the
common allocation/rewind helpers in [BackendTransformer](../model/tensor_backend.py).
LLT stores one latent history per physical layer, constructed during the first
loop and reused thereafter. Each new token appends once per layer, not once per
loop. Conventional loops and the refreshed latent control keep per-application
histories. Prefill, cache copies and weight preparation are measured separately
from cached decode as serving startup in the
[main experiment](../experiments/l40s/LOOP_SWEEP.md).

[prepared.py](prepared.py) provides `prepare(model)` for the Torch serving control.
It attaches a linear operation that caches BF16 copies of FP32 master parameters.
Embedding lookup retains FP32 masters. Copies are refreshed when a parameter's
version or storage changes. The prepared operation requires `torch.inference_mode()`.
Tensor uses its own inference-weight preparation in its backend operators.

See the [model package](../model/README.md) for configuration and the
[reproduction guide](../docs/REPRODUCIBILITY.md) for the measured serving protocol.
The archived nanoGPT study retains its original, separate serving adapter.
