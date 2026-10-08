# Full-model Tensor L40S study

This extends the archived [attention-only study](README.md) using the qualified
[Tensor](https://github.com/kreasof-ai/tensor) implementation
`ab17948fe94dd7c7b857c8f067c465e901329b94`. The dependency readiness report is in
[Tensor's repository](https://github.com/kreasof-ai/tensor/blob/main/docs/research/llt-readiness.md).
LLT model, integration and experiment results live here; Tensor development
plans and defect reports remain in that repository.

The [frozen architecture](../../research/NATIVE_MODEL.md) defines global and layerwise
initial-input latents, tied blocks, batched projection folding, learned absolute
positions and exact checkpoints. Tensor performs the numerical model operators,
loss, backward kernels, clipping and optimizer update. PyTorch controls layouts,
autograd/checkpoint scheduling and shared-gradient accumulation. This is an
explicit backend, not automatic lowering of an arbitrary PyTorch model.

## Reproduction

Install the qualified Tensor compiler environment and matching native Torch
adapter using its [LLT guide](https://github.com/kreasof-ai/tensor/blob/main/docs/guides/llt.md).
The script requires the following environment variables:

- `LLT_PYTHON`: Python executable with Tensor and the native Torch adapter.
- `TENSOR_CHECKOUT`: Tensor installation used for revision/source verification.
- `TENSOR_NVRTC_HOME`: bootstrapped NVRTC 12.9 installation.

From the LLT repository root:

```bash
bash experiments/l40s/run_native.sh
python experiments/l40s/native_summary.py --plots
```

The plotting command requires matplotlib; the base Python on the experiment
machine supplies it. Individual phases are available as
`$LLT_PYTHON experiments/l40s/native_study.py correctness|training|inference|audit`.
The `--smoke` option limits training/inference to the first geometry. Use the
default nine samples for a qualifying run. Run phases sequentially on an idle
GPU. Existing reports are replaced, so archive them before an independent run.

## Protocol

- Correctness first: independent FP64 folded/unfolded algebra and all gradients;
  BF16 Tensor/PyTorch outputs and every parameter gradient; exact checkpoint
  gradients; 32 matched update steps per architecture; causality and four
  consecutive cached tokens from those updated weights. Cache addresses remain
  stable, and version-tracked parameter updates invalidate inference state.
  This short training fixture is backend validation, not model quality evidence.
- Nineteen training geometries cover widths 128/512/768, ranks 32/64/128,
  layers 2/12, loops 1/4/10/20, batches 1/4, odd sequences 257/1025/4097 and
  vocabularies 4096/50257. Each has naive/global LLT, no/exact-loop checkpointing,
  and Torch/Tensor backends: 152 methods. Every recurrence configuration has
  its matching one-loop control. Model/data seeds are separate; data match
  across architectures, policies and backends at a given geometry.
- Training uses FP32 master weights, residuals and AdamW moments; BF16 matrix
  operations; global norm clipping to 1; learning rate 0.001; full-token mean
  cross entropy and `foreach=False` Torch AdamW. Three warmup updates establish
  optimizer state. Peak allocation is measured on a further complete update,
  followed by nine timed updates without extra warmup. Training is eager;
  event timings include host submission gaps, and synchronized wall times are
  also retained. No graph training claim or streamed-loss substitution is made.
- Fourteen inference geometries use width 128/512/768, layers 2/4/12, loops
  1/4/10/20, ranks 32/64/128, batches 1/4, odd prefixes 257/1025/4097 and
  vocabulary 256. Naive/global/layerwise architectures each have both backends:
  84 methods. Projection weights are prepared in BF16 once; embeddings,
  normalization and residuals remain FP32. These models use synthetic inputs
  and random initial weights, not quality-selected trained checkpoints.
- The causal-forward prefill refreshes all folds and retains raw prefix tensors.
  Conversion to preallocated serving capacity and generation-fold preparation
  happen outside decode timing and are excluded from this causal-forward timing.
  Entire serving startup and first-token latency remain separate open measures.
  Generation initializes preallocated persistent caches and measures four
  consecutive appends through logits. Its final output is checked against full
  causal prefill at every method/geometry. Folding and prefix construction are
  outside decode timing. The fixture rewinds logical lengths before each sample,
  outside timed work and outside CUDA graph capture. No historical prefix is
  copied during decode. Graph replay describes a fixed four-token trajectory;
  it is not a variable-length serving scheduler or tail-latency guarantee.
- Nine individual observations are retained for synchronized eager wall/event
  timing and CUDA graph event timing. Compilation, allocation of fixtures,
  warmup and optimizer initialization are excluded. The warm Tensor cache was
  seeded with qualified dependency artifacts when identities matched; cache
  hits and all binaries actually used are recorded. No compile-time speedup is
  inferred. Samples always run sequentially without another GPU experiment.
- Peaks are PyTorch CUDA allocator allocated bytes, including model weights,
  gradients, optimizer, embeddings/classifier, folded weights, persistent cache,
  transient activations and visible cuBLAS workspaces. Reserved memory is
  recorded separately. CUDA context/driver/module allocations and on-chip
  resources are excluded. Unused cuBLAS workspaces are cleared between phases
  and methods; active Torch operations recreate any required workspace. Small
  cross-backend memory differences can reflect that workspace or fused optimizer
  temporaries, rather than LLT compression. Architecture comparisons use the
  same backend and checkpoint policy. Eager peaks accompany eager times;
  captured peaks accompany graph times.

`native_summary.py` applies the existing criteria to each matched comparison:
at least 50% less total peak allocation than naive, at most 20% latency tax,
and at most 25% more allocation than the corresponding one-loop LLT. It does
not equate constant cache size with constant total training memory. All
observations and unsuccessful regimes remain in the reports.

## Artifacts

Raw JSON, source snapshots, derived criteria, audit and exportable figures are
in [`benchmarks/results/l40s-native`](../../results/l40s-native).
Binary kernels, generated exports/compiler caches and execution logs remain
on the machine under ignored paths. Reports retain the exact Tensor pin,
package/GPU/driver versions, LLT source hashes, artifact identities and launch
coverage. The audit verifies completion, sources, binaries, sm_89 targets,
baseline/policy coverage and sample counts. The original FP16 reports are
preserved in `benchmarks/results/l40s` and are not overwritten by this study.
