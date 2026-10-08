# L40S kernel and memory study

This study prioritizes CUDA kernel correctness, inference cache scaling and GPU
training memory. It does not establish trained model quality. The original Vulkan
and CPU reports are preserved.

The causal model has full residuals, loop-tied block weights, 4x GELU MLPs,
RMS normalization and learned absolute positions. LLT computes one global KV
latent from initial token/position embeddings. Query and output projections are
folded per head; the folds remain differentiable in training. The `layerwise`
control computes one such latent per layer, also from initial embeddings. This
is a geometry control, not a faithful DeepSeek/YOCO/LLA reproduction.

[Tensor](https://github.com/kreasof-ai/tensor) owns custom CUDA attention kernels, compiled via TileLang/TIRx and NVRTC
into sm_89 `.tbin` files and launched via `tensor-torch`. PyTorch owns linear
layers, MLPs, normalization, embeddings and all training backward/optimizer work.
The reported `tensor` decoder/prefill backend means **Tensor attention in an
otherwise PyTorch model**. The PyTorch control forces CUDA Flash SDPA.

## Setup and reproduction

Install [Tensor](https://github.com/kreasof-ai/tensor) at the revision recorded
in the raw reports. From the LLT repository root, the following creates a separate
checkout and installs the pinned compiler environment:

```bash
llt_project_root="$PWD"
export TENSOR_CHECKOUT="$(mktemp -d)/tensor"
git clone https://github.com/kreasof-ai/tensor "$TENSOR_CHECKOUT"
cd "$TENSOR_CHECKOUT"
git checkout caf0118d870739b176a88f487ae9e54de1d5d214
uv sync --locked
uv pip install --python .venv/bin/python --no-deps -e packages/tensor-torch
uv run --no-sync python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
export LLT_PYTHON="$TENSOR_CHECKOUT/.venv/bin/python"
export TENSOR_NVRTC_HOME="$TENSOR_CHECKOUT/build/nvrtc-12.9"
cd "$llt_project_root"
bash experiments/l40s/run.sh
python experiments/l40s/summarize.py --plots
```

The last command needs matplotlib; it is available in the machine's base Python.
Do not use a later `uv sync` to remove the separately installed Torch adapter.
`TENSOR_CHECKOUT`, `LLT_PYTHON`, and `TENSOR_NVRTC_HOME` configure the installation
used by the experiment. Individual phases
can be rerun using the commands in `run.sh`. Reports are overwritten; archive a
completed output directory before another independent run.

## Protocol

- Nine samples per measurement with all observations retained. Warmup, compilation,
  fixture allocation and initial optimizer-state creation are outside timing.
- Attention kernels use prepared output storage. CUDA graph replay isolates GPU
  work from Python launch gaps. Ordinary CUDA-event and synchronized wall times
  are reported separately for attention/decoder/prefill. Events surrounding
  ordinary Python submissions include gaps when the GPU waits for the host.
  Captured-graph allocator peaks are recorded with graph timings; graph-mode
  qualification uses those peaks, rather than mixing eager memory and graph time.
- One-token decode uses synthetic independent histories, includes the current
  token, and overwrites preallocated current-token slots. It avoids copying the
  historical cache during each token. Folded weights are prepared outside timing.
- Prefill executes the full causal model on random tokens and retains the actual
  prefix caches. Cached decode matches causal prefill on the correctness fixture.
- Memory is the PyTorch CUDA allocator's peak allocated bytes. It includes model,
  caches, folded weights, gradients, optimizer state, operator outputs and cuBLAS
  allocations visible to the allocator. Reserved bytes are separate. This is
  **not** whole-process VRAM; CUDA context/module and non-allocator native scratch
  are excluded. Inactive cuBLAS graph/side-stream workspaces are cleared between
  comparisons to avoid accumulation. Persistent cache bytes are counted directly.
- Training uses FP32 parameters/AdamW state, BF16 autocast and forced Flash SDPA.
  Both naive and LLT receive `none` and exact non-reentrant per-loop checkpoint
  policies. Full-token cross entropy and the optimizer update are included.
- The float64 algebra reference explicitly expands K/V and validates outputs and
  all parameter gradients. Exact checkpoint gradients are checked in float64 and
  actual CUDA BF16 Flash execution. Future-token perturbations check causality.
- Tensor kernels are compared with explicit FP32 matmul/softmax and/or forced
  PyTorch Flash SDPA. Checks include odd lengths, empty partitions, K/V aliasing,
  batches, 64K histories, and query magnitudes up to 10.

The serial reference, partitioned decode and fixed 8/16/32/64-partition sweep are
reported separately. Full decoder runs use **32 partitions**, not the best
partition count selected retrospectively for each case.

## Artifacts

[`benchmarks/results/l40s`](../../benchmarks/results/l40s) contains raw JSON,
derived comparisons and figures. Reports retain source hashes, exact packages,
hardware/driver information and binary hashes. Hash-addressed source snapshots
are retained in `sources/`. Compiled `.tbin` files, generated exports, logs and
optional quality checkpoints stay on this machine under ignored directories.
`summarize.py` checks report status and derives memory/latency qualification using
the existing >=50% saving / <=20% latency tax / <=25% non-loop overhead criteria.

The optional `quality.py` supplies small language and pointer-chasing runners;
only a language smoke run was performed while establishing the harness. It is
not part of the completed kernel/memory experiment or its performance claims.
