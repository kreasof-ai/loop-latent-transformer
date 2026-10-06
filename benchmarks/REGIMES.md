# Memory and latency regimes

This follow-up searches **system-resource regimes**, not equal-quality trained
models. GPU inference uses the RX 6700 XT/Vulkan Tensor runtime. CPU training
uses a Ryzen 5 5600, four PyTorch threads, FP32 and Adam.

An operating point qualifies when all three measured conditions hold:

1. At least **50% less peak live tensor storage** than naive looping.
2. At most **20% extra median latency** versus naive looping at the same depth.
3. At most **25% more tensor storage** than the same LLT model with one loop.

Memory includes weights and workspaces; training also includes gradients, dense
Adam state and output tensors. It excludes driver/runtime overhead and native
operator scratch. These are allocation ledgers, not whole-process RSS/VRAM.

## Inference: a useful regime exists

Width 768, 12 layers, head width 64, batch 1. The decoder includes evolving
queries/residuals, RMSNorm, 4× GELU MLPs, current-token attention, and 256 output
logits. Prefixes are synthetic. LLT uses one shared global latent and folds
loop-tied K/V up-projections into query/output weights before execution.

At 4,096 historical tokens and ten loops, an independent nine-sample repeat measured:

| Method | Peak live tensor MiB | GPU ms |
|---|---:|---:|
| Naive MHA, separate caches, rank-32 control | 1,602.846 | 58.044 |
| Per-head LLA geometry, rank 32 | 257.696 | 46.267 |
| Folded LLT, rank 32 | 135.746 | 46.209 |
| Naive MHA, separate caches, rank-64 control | 1,602.846 | 57.820 |
| Per-head LLA geometry, rank 64 | 352.245 | 62.358 |
| Folded LLT, rank 64 | 163.071 | 60.227 |

Rank-32 LLT saves **91.5%** versus naive, is **20.4% faster**, and saves **47.3%** versus
the LLA geometry prototype at approximately equal latency. Rank-64 LLT saves
**89.8%** versus naive with **4.2%** extra latency, and saves **53.7%** versus
the corresponding LLA prototype. LLT allocation is exactly its non-loop value
at both ranks. Ordinary non-loop MHA uses 306.530 MiB at this context.

The later boundary run brackets the tested rank limit at this geometry:

| LLT rank, context 4K, ten loops | Peak MiB | GPU ms | Ratio to naive |
|---|---:|---:|---:|
| 32, independent repeat | 135.746 | 46.209 | 0.796× |
| 64, independent repeat | 163.071 | 60.227 | 1.042× |
| 96, boundary | 190.396 | 124.492 | 2.158× |
| 128, boundary | 217.721 | 142.129 | 2.456× |

Thus ranks **32–64** are tested candidates, while **96–128 fail the 20% latency
criterion**. This brackets a crossover; it does not establish its exact rank.

At context 512, the first >=50% storage point is measured at **seven loops for
rank 32**: six loops save 49.9%, seven save 53.1%. Rank 64 first qualifies at
**ten loops**: nine save 49.9%, ten save 52.6%. Both qualified cases meet the
latency condition. At longer contexts the memory criterion is met with fewer
loops. Tests at 8K and ten loops also qualify for ranks 32 and 64.

This differs from the earlier unfused decoder result because query/output
folding removes repeated projection work. It requires tied, fixed up-projections.
Folded GPU weights are FP32; other GPU weights/cache are FP16. This precision
choice is part of the implementation and memory/timing results.

## What the LLA comparison means

The [LLA paper](https://arxiv.org/html/2607.15456v2#S3) describes a post-training
codec with separate per-head K/V stores and loop-specific reconstruction maps.
Our local prototype implements that geometry with random zero-mean codecs,
absorbed historical attention, exact current-token KV and an encoder for the
completed new-token loop trajectory. Codec weights and encoding work are counted.
It is **not** a reproduction of the authors' trained serving system.

LLT and LLA here have different latent budgets. For rank 32 at ten loops, naive
cache is 1,440 MiB, per-head LLA cache is 72 MiB, and global LLT cache is 0.25
MiB. LLT compresses across layers and heads as well as loops. A storage win under
this more restrictive representation does not establish a quality win.

RoPE, learned codec fitting, trained-model prefill and generation quality are
not tested. The paper's low-latency reconstructed-cache fast path retains full
KV; its reported memory and speed paths must not be combined into one result.
[Source](https://arxiv.org/html/2607.15456v2#S7)

## CPU training: exact checkpointing can meet the target

LLT training retains the full residual state and a shared KV latent. Query/output
folding is differentiable and performed once outside the loops. Exact per-loop
PyTorch checkpointing stores full loop-boundary states and recomputes each
block once during backward. This avoids the earlier quadratic prefix replay.
It is ordinary checkpointing, not reconstruction of residuals from KV alone.
Boundary-state storage still grows with loops; it is small in the tested regime.

With width 512, two layers, eight 64D heads, rank 32, batch 1, sequence 128 and
a small 128-class output, an independent nine-round interleaved run measured:

| Training method | Loops | Peak live tensor MiB | Forward/backward/Adam ms |
|---|---:|---:|---:|
| Non-loop naive | 1 | 109.564 | 55.243 |
| Non-loop LLT | 1 | 94.814 | 47.339 |
| Naive, no checkpoint | 13 | 201.431 | 550.401 |
| Naive, same loop checkpoints | 13 | 116.073 | 715.045 |
| LLT, loop checkpoints | 13 | 99.793 | 512.485 |

LLT saves **50.5%** versus uncheckpointed naive, is **6.9% faster by median**,
and uses **5.3%** more storage than non-loop LLT. All nine paired latency ratios
are at most 1.073×. At 12 loops, it saves only 48.5%, so the measured >=50%
memory boundary for this small-vocabulary configuration lies between 12 and 13.

Against naive with the same checkpoint policy, the additional memory saving is
**14.0%**, and median time is **28.3% lower**. Most of the gross memory saving
comes from checkpointing; its availability to the baseline is explicitly counted.

### A 50K vocabulary shifts the threshold

Keeping width 512, two layers, rank 32 and batch 1, but increasing sequence
length to 1,024 and vocabulary to **50,257**, counts the dense classifier,
all-token cross-entropy and its Adam state. Three rotating timing rounds measured:

| Training method | Loops | Peak live tensor MiB | Forward/backward/Adam ms |
|---|---:|---:|---:|
| Non-loop naive | 1 | 1,219.830 | 1,592.131 |
| Non-loop LLT | 1 | 1,200.897 | 1,524.367 |
| Naive, no checkpoint | 16 | 2,181.002 | 6,834.270 |
| LLT, loop checkpoints | 16 | 1,180.818 | 6,635.593 |
| Naive, no checkpoint | 20 | 2,437.315 | 8,449.932 |
| Naive, same loop checkpoints | 20 | 1,195.752 | 10,205.507 |
| LLT, loop checkpoints | 20 | 1,188.818 | 7,701.249 |

At 16 loops, LLT saves **45.9%**, missing the 50% criterion. At **20 loops**
it saves **51.2%**, is **8.9% faster by median**, and uses slightly less tensor
storage than uncheckpointed non-loop LLT. All three paired time ratios at
20 loops are below 0.961×. Thus the measured training crossover is bracketed
between 16 and 20 loops in this configuration; the exact first qualifying
loop count was not measured. Three samples establish this operating point,
not a tail-latency guarantee.

Against naive with the **same checkpoints**, LLT saves only **0.58%** memory,
while median time is **24.5% lower**. The vocabulary/loss/optimizer floor makes
the architecture's additional training-memory benefit small here. The large
51.2% saving requires comparing checkpointed LLT with uncheckpointed naive.
Without checkpoints, LLT at 20 loops saves only 10.1% and remains 1.82× its
non-loop footprint, so it does not meet the memory target.

These results do not give a universal training threshold. Vocabulary, sequence,
layer count, optimizer and checkpoint policy determine the memory floor and
the crossover. The training prototype has two layers; it does not validate
the proposal's 12-layer GPU training configuration.

Checkpoint gradients match uncheckpointed autograd in every tested case. An
independent float64 full-K/V, unfolded-projection reference matches the folded
LLT output within 3.34e-16 and gradients within 2.17e-18 absolute error.

The original LLA codec is fitted after training a teacher; it supplies no native
pretraining activation-memory method. We therefore report its parent teacher
training as the naive graph, rather than assigning the inference compression
factor to training activations. Codec fitting is a different training objective.

## Scope, reports and reproduction

The coarse search contains 16 inference configurations across three methods and
16 CPU training configurations across two models/four checkpoint policies.
Boundary and repeat runs retain additional samples. GPU execution and host times
are separate. CPU timings include forward, cross-entropy, backward and Adam;
setup, clones and GC are outside timing. Memory-profile hooks are absent from
timing runs. GPU jobs and CPU timing jobs were run sequentially.

All seven completed reports passed correctness checks: 48 measured configurations
(including repeats), 231 method/policy measurements, and 229 recorded kernel
artifacts. Source/report hashes and kernel binary hashes were checked against
the local files; kernel source hashes use normalized newlines on Windows.

- [Coarse inference](results/regime-inference/report.json)
- [Inference boundary](results/regime-boundary/report.json)
- [Independent inference repeat](results/regime-repeat/report.json)
- [Coarse CPU training](results/regime-training/report.json)
- [Interleaved CPU training repeat](results/regime-training-repeat/report.json)
- [Realistic-vocabulary CPU training](results/regime-training-model/report.json)
- [Realistic-vocabulary training at 20 loops](results/regime-training-model-deep/report.json)
- [Machine-readable qualification results](results/regime-summary.json)

```powershell
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\regime_inference.py
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\regime_boundary.py
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\regime_boundary.py --phase repeat --samples 9
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\regime_training.py --samples 3
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\regime_training_repeat.py --loops 1 12 13 16
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\regime_training_model.py
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\regime_training_model.py --loops 20 --out benchmarks\results\regime-training-model-deep
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\regime_summary.py
```

The qualification script supports `--tax`, `--saving` and `--nonloop` to inspect
different acceptance criteria. Ratios use medians and are not tail-latency
guarantees. Reports retain observations, source hashes and GPU kernel hashes.
No H100, NVLink, GPU backward or equal-quality trained LLT/LLA result is claimed.
