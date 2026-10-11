# LLT Tensor

This package owns the LLT-specific Tensor backend previously housed in
`tensor-torch`. It supplies `Operators`, first-order autograd, `AdamW`, `KVCache`,
and `RecurrentOperators`, plus their Tensor kernel factories. It depends on core
`tensor-workspace` and Torch, and has no dependency on `tensor-torch`.

From the Loop Latent Transformer checkout, with Python 3.12 and a matching Tensor
checkout available:

```sh
python -m pip install -e ../tensor
python -m pip install -e packages/llt-tensor
```

```python
from llt_tensor import Operators, AdamW, KVCache, RecurrentOperators
```

`src/llt_tensor/llt.py` contains the workload API, backward implementations,
optimizer, and cache. `recurrent.py` extends that API. `templates/` owns all LLT
kernel factories. `bridge.py`, `launch.c`, and `executor.cpp` preserve the previous
launch implementation locally so changes to Tensor's compatibility binding
cannot break these workloads. The custom-op namespace is `llt_tensor`.

The portable submission extension builds when a host C compiler is available.
The optional C++ executor currently targets Torch 2.14 and is built using
`LLT_TENSOR_BUILD_NATIVE=1` with build isolation disabled. Set
`LLT_TENSOR_NATIVE=0` to disable loading it. Existing installed native extensions
from `tensor_torch` are not reused.

The default GEMM profile uses coalesced MMA and small-input GEMV schedules;
`gemm_profile="legacy"` retains the original schedule. Optional
`cache_inference_weights=True` prepares parameter casts under
`torch.inference_mode()` and refreshes them after versioned updates. Training
behavior remains unchanged. These contracts and their original evidence are
recorded in Tensor's `docs/research/llt-optimization.md` and
`docs/research/llt-inference-weight-cache.md`.

## Tests and benchmarks

From this repository root:

```sh
python -m pytest packages/llt-tensor/tests
TENSOR_LLT_CUDA=1 python -m pytest packages/llt-tensor/tests
python -m benchmarks.llt.qualify gradients
```

CUDA compilation requires Tensor's compiler dependencies and NVRTC bundle. Set
`TENSOR_NVRTC_HOME` to that bundle, `TENSOR_CHECKOUT` if the Tensor checkout is
not `../tensor`, and optionally `TENSOR_LLT_CACHE_DIR` for compiled artifacts.
Existing cache artifacts may be rebuilt because kernel module identities changed.
New benchmark results go under this repository's `docs/research/data` and `build`.
Historical result snapshots in both repositories keep their original bytes and
paths; the migration does not requalify their measurements.
