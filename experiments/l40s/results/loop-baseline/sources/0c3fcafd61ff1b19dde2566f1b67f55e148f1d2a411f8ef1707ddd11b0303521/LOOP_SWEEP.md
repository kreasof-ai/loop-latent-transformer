# L40S four-model loop sweep

The requested grid is batch 4, context 1024, and loop count T=1..16. Numerical
kernels use [Tensor](https://github.com/kreasof-ai/tensor), with a matched PyTorch
control using Flash SDPA and fused AdamW. Each case runs in its own fresh process;
GPU jobs are sequential. No activation checkpointing or streamed loss is enabled.

All models have residual width 768, 12 heads (head dimension 64), learned absolute
position embeddings of capacity 1025, a 50,304-token vocabulary, independent input
embedding/output weights, exact GELU, and unweighted RMSNorm with epsilon 1e-5.
LLT uses a shared latent of rank 64. Master weights, residuals, embeddings, AdamW
states, and loss reductions are FP32; projection and attention arithmetic is BF16.
Serving prepares BF16 linear-weight copies on both backends while retaining FP32
master parameters. Those copies count toward peak memory.

| Architecture | Unique blocks | Applied blocks | MLP hidden width |
|---|---:|---:|---:|
| LLT | 12 | 12T | 3072 |
| Naive Loop | 12 | 12T | 3072 |
| Independent stack | 12T | 12T | 3072 |
| Fixed depth, matched parameters | 12 | 12 | 768(6T−2) |

The fixed-depth control is a conventional Transformer with independent blocks.
Its active MLP weights match the stack's total parameter count exactly at each T.
This keeps residual width, heads, vocabulary, and embeddings identical. It changes
the attention-to-MLP compute balance: equal parameter counts do not imply equal
attention work or equal activation memory. It never pads the model with unused
parameters. At T=1, the three conventional models are equivalent architectures,
but are measured in independent processes.

Training measures complete forward, full-token output projection/cross entropy,
backward, global gradient clipping (norm 1), and AdamW updates (learning rate
0.0006, betas 0.9/0.95, weight decay 0.1). Inputs and targets are seeded synthetic
CUDA token IDs. This is a performance study, not a trained-quality evaluation.

Inference reports three separate operations:

- Prompt: causal 1024-token forward and last-position logits, without persistent KV
  allocation.
- Serving startup: the same prompt, plus KV allocation/copy and LLT fold rebuild.
- Cached decode: one supplied token after a 1024-token history, with a 1025-slot KV
  capacity. This measures the model computation, without sampling or beam search.

All latencies are per batch of four. A training step consumes 4096 tokens; a
cached-decode call produces one token per sequence (four tokens total).

There are three warmups and nine timing samples for each operation. Eager results
include CUDA-event and synchronized wall time. CUDA graphs remove Python launch
cost; each sample times three replays and divides by three. Captured decode
includes a logical cache rewind to make every replay use the same history.
Training validates finite losses and every parameter's gradient, then verifies
GPU optimizer counters advance through all 42 actual updates; graph capture itself
records kernels without performing an update. Compilation and warmup are excluded.

Eager memory is PyTorch's maximum allocated memory during a measured call. Graph
memory is the maximum allocated during **capture**, including capture temporaries
and the private graph pool. Replay peaks and allocator reservations are also
retained in the raw JSON. These include model weights and live model state, not
just incremental activation bytes; they exclude driver/context allocations and
allocator reservations from the allocated metric. Models that exceed the GPU
budget are recorded as OOM at the exact failed operation. No smaller geometry is
substituted. A later operation skipped after OOM has no timing value.

The full grid has 256 performance cases (4 architectures × 16 loop counts × 2
backends × 2 phases), plus eight small correctness cases covering loop counts 4
and 16. Qualification uses width 128, two heads, two base layers, batch 2, and sequence
32. It compares Tensor against matched PyTorch outputs and every parameter
gradient, checks cache/full-forward equivalence, and verifies that stack and
fixed-depth parameter counts agree. Full-size training additionally checks every
parameter gradient for finiteness. Raw JSON pins source snapshots,
Tensor revision, runtime versions, and compiled artifact identities.

Run using the existing Tensor environment with its native Torch executor and
CUDA 12.9 NVRTC installation:

```bash
LLT_RESUME=1 experiments/l40s/run_loop_sweep.sh
LLT_RESUME=1 experiments/l40s/run_loop_sweep_recovery.sh
python experiments/l40s/loop_sweep_summary.py
python experiments/l40s/loop_sweep_manifest.py
python experiments/l40s/loop_sweep_report.py
```

Set `LLT_PYTHON`, `TENSOR_CHECKOUT`, and `TENSOR_NVRTC_HOME` for another installation.
The summary requires Matplotlib; it can run in a separate CPU Python environment.
Do not run another GPU benchmark concurrently. `LLT_RESUME=1` skips completed JSON
files, including recorded OOM cases. Remove a specific result file to rerun it.
Results and figures are written to `benchmarks/results/l40s-loop-sweep/`.

Capture failures can surface as chained allocation/graph-instantiation exceptions.
The CPU recorder requires explicit CUDA OOM evidence, retains the unmodified
exception JSON under `observed-capture-oom/`, and annotates its classification.
Other exceptions still stop the sweep.

A supplementary capture retry releases the eager gradients and allocator caches
after graph warmup, before capture. It preserves all numerical kernels, optimizer
settings, input geometry, and 42-update validation. These retries are stored in
`capture-recovery/` and do not replace the primary measurements. They distinguish
setup/pool reservations from a model that cannot complete an eager training step.
