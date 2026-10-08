# Inference helpers

The current inference code supports the measured prefill and supplied-token
cached-decode operations. The main study does not include sampling, beam search,
or a production generation service.

[BackendTransformer](../model/tensor_backend.py) owns the architecture-dependent
cache allocation and the `prefill`, `decode_token`, and `rewind` operations. LLT stores one shared latent
history. Conventional loop controls store independent K/V histories for each
block application. Prefill, cache copies, and weight preparation happen before
timed token decode; their latency and memory are reported separately as serving
startup in the [main experiment](../experiments/l40s/LOOP_SWEEP.md).

[prepared.py](prepared.py) provides `prepare(model)` for the Torch serving control.
It attaches a linear operation that caches BF16 copies of FP32 master parameters.
Embedding lookup retains FP32 masters. Copies are refreshed when a parameter's
version or storage changes. The prepared operation requires `torch.inference_mode()`.
Tensor uses its own inference-weight preparation in its backend operators.

See the [model package](../model/README.md) for configuration and the
[reproduction guide](../docs/REPRODUCIBILITY.md) for the measured serving protocol.
The archived nanoGPT study retains its original, separate serving adapter.
