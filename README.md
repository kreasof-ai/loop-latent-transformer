# Loop-Latent Transformer (LLT)

A proposed architecture combining **Multi-head Latent Attention (MLA)**,
**Looped Transformers**, and **YOCO / U-YOCO**, with a proposed training
strategy called **Latent Activation Checkpointing (LAC)**. The current
implementation evidence covers synthetic inference and CPU training prototypes;
the full combination and trained-model quality remain unvalidated.

## Core Idea

A naive looped transformer reuses one block T times, but stores a
separate KV cache and a separate set of activations for every loop.
Memory and communication scale linearly with T.

LLT observes that:

1. MLA already compresses KV into a low-rank latent.
2. Looped transformers produce highly redundant per-loop K/V trajectories.
3. YOCO / U-YOCO already share a single global KV cache across layers.

By **compressing along the loop axis** and **sharing the compressed latent
across loops and layers**, LLT targets an inference KV cache that is
**constant in T**. This has been demonstrated in the local prototype.
Constant total training memory and constant total communication have **not**
been demonstrated: exact loop checkpoints retain full residual boundary states,
and compute still grows with T.

## Provisional Evidence — 2026-10-06

The current [regime report](benchmarks/REGIMES.md) searches for at least **50%
less peak live tensor storage**, at most **20% extra median latency** versus
naive looping, and at most **25% more storage than non-loop LLT**. These are
synthetic resource comparisons, not equal-quality trained-model comparisons.

### GPU inference

RX 6700 XT, Vulkan via the local Tensor runtime; width 768, 12 layers,
12 heads of width 64, batch 1, 4,096 historical tokens, ten loops and
256 output logits. Nine-sample repeats of the folded implementation measured:

| Method | Peak live tensor MiB | Median GPU ms |
|---|---:|---:|
| Naive MHA, rank-32 control | 1,602.846 | 58.044 |
| Per-head LLA geometry prototype, rank 32 | 257.696 | 46.266 |
| LLT, rank 32 | 135.746 | 46.209 |
| Naive MHA, rank-64 control | 1,602.846 | 57.820 |
| Per-head LLA geometry prototype, rank 64 | 352.245 | 62.358 |
| LLT, rank 64 | 163.071 | 60.227 |

Rank 32 saves **91.5%** versus naive and is **20.4% faster**. Rank 64 saves
**89.8%** with a **4.2% latency tax**. Both use exactly their non-loop LLT
tensor storage; non-loop naive uses 306.530 MiB. Versus the corresponding LLA
prototype, LLT saves **47.3% / 53.7%** memory at rank 32 / 64.

This improvement requires folding tied, fixed K/V up-projections into query
and output weights. Folded weights are FP32; cache/base weights are FP16.
The earlier unfused decoder had a substantial latency tax. At the same 4K
context and ten loops, ranks **96 and 128 still fail** the latency target:
they take **2.158× and 2.456×** naive time. The exact rank crossover is unmeasured.
At context 512, the measured 50% memory boundary is between six and seven
loops for rank 32, and between nine and ten loops for rank 64.

The LLA comparison implements per-head cache-codec geometry with random
weights, including new-token encoding. It is not an author-system reproduction.
LLT and LLA use different latent budgets: rank-32 caches here occupy 0.25 MiB
and 72 MiB respectively, versus naive's 1,440 MiB. The stronger LLT compression
has no verified quality guarantee.

### CPU training

Ryzen 5 5600, four PyTorch threads, FP32 and Adam; width 512, two layers,
rank 32, batch 1, sequence 1,024 and vocabulary 50,257. Three rotating timing
rounds include forward, full-vocabulary cross-entropy, backward and Adam:

| Method | Loops | Peak live tensor MiB | Median step ms |
|---|---:|---:|---:|
| Non-loop LLT | 1 | 1,200.897 | 1,524.367 |
| Naive, no checkpoints | 16 | 2,181.002 | 6,834.270 |
| LLT, loop checkpoints | 16 | 1,180.818 | 6,635.593 |
| Naive, no checkpoints | 20 | 2,437.315 | 8,449.932 |
| Naive, same loop checkpoints | 20 | 1,195.752 | 10,205.507 |
| LLT, loop checkpoints | 20 | 1,188.818 | 7,701.249 |

At 20 loops, checkpointed LLT saves **51.2%** versus uncheckpointed naive,
is **8.9% faster**, and has a footprint close to non-loop LLT. At 16 loops,
it saves only **45.9%**, so the measured crossover lies between 16 and 20.

Against naive with the **same checkpoints**, the additional memory saving
is only **0.58%**, while median step time is **24.5% lower**. Most of the
gross training-memory saving comes from ordinary checkpointing. Without
checkpoints, LLT at 20 loops saves only **10.1%** and uses **1.82×** its
non-loop footprint. A smaller 128-class, sequence-128 test qualified at
13 loops, demonstrating that the threshold depends on the memory floor.

The verified training path uses differentiable projection folding and exact
loop checkpoints that retain full residual boundary states. Its gradients
match uncheckpointed autograd; a separate float64 unfolded reference also
passes. It does **not** validate reconstructing every residual activation
from a single KV latent. LLA is a post-training inference codec; no comparable
native LLA pretraining activation-memory graph was measured.

### Measurement limits

Memory is peak **live tensor storage**, including weights/workspaces and,
for training, gradients and dense Adam state. Driver/runtime overhead and
native operator scratch are excluded; these numbers are not process RSS/VRAM.
The seven completed regime reports cover 48 configurations including repeats
and 231 method/policy measurements, with correctness and artifact-hash checks.
RoPE, prefill, learned quality, GPU backward, H100/NVLink and distributed
communication remain unverified. Timings are medians, not tail-latency guarantees.
See [reports and reproduction commands](benchmarks/REGIMES.md#scope-reports-and-reproduction).

# Architecture

## Block Layout

LLT is a decoder-only transformer with three structural deviations
from a standard NanoGPT-style model:

1. **A single shared block** is applied T times (looped transformer).
2. **Attention uses MLA-style latent KV** (low-rank compressed cache).
3. **The latent is shared across layers and loops** (YOCO / U-YOCO style).

## Forward Pass (per token t)

The measured implementation uses tied up-projections and folded attention:

    Q_fold[l] = W_K[l]^T W_Q[l]         # prepared once; differentiable in training
    O_fold[l] = OutProj[l] W_V[l]
    C_t = Down_KV(Norm(x_t))            # low-rank latent, dim d_kv
    C_cache = append(C_cache, C_t)
    for i in 1..T:                       # loop index
        for l in 1..L:                   # layer index (shared block reused)
            q_lat = Q_fold[l] Norm(x_t)
            a_lat = softmax(q_lat C_cache^T / sqrt(d_head)) C_cache
            x_t = x_t + O_fold[l](a_lat)
            x_t = x_t + MLP^l(Norm(x_t))

Crucially:

- `C_t` is computed **once per token**.
- Attention reads the shared latent directly, avoiding full K/V expansion.
- No per-loop or per-layer KV is ever persisted.

This is schematic per-head algebra; CPU sequence training applies a causal
mask. Decoupled RoPE remains untested. The full residual state still evolves
and must be handled during backward.

## Why This Is Not Just MLA + YOCO

- **MLA alone** compresses per-layer KV, but each layer still has its own cache.
- **YOCO alone** shares one cache across layers, but does not compress it.
- **U-YOCO alone** iterates the self-decoder, but does not compress the state.
- **LLT** does all three simultaneously, and adds the loop axis as a
  first-class compression dimension.

## Parameters

- Shared block parameters: `W_Q, W_K, W_V, OutProj, MLP`
- Loop-conditioned (or loop-tied) up-projections: `W_K^{l,i}, W_V^{l,i}`
- Down-projection: `Down_KV`
- Optional gating (MELT-style): `g_i` per loop

The measured path ties up-projections across loops (`W_K^{l,i} = W_K^l`).
Loop-specific maps and optional gating remain design variants whose quality
and resource tradeoffs have not been measured here.

# Mathematical Foundations

## 1. MLA Matrix Absorption and Projection Folding

Standard MLA attention:

    Attn = softmax( Q (W_UK C)^T / sqrt(d) ) (W_UV C)

Absorption reorders the multiplications:

    Attn = softmax( (W_UK^T Q) C^T / sqrt(d) ) C W_UV^T

This eliminates materialization of K = W_UK C and V = W_UV C.
The compressed latent C is the persistent KV cache; residuals, weights and
workspaces remain separate allocations.

## 2. Measured Rank and Training Tradeoffs

The absorbed form materializes:

- `q_absorbed = W_UK^T q_nope` with shape (n_h, d_kv)
- a post-attention latent accumulator with shape (n_h, d_kv)

These intermediates grow with `d_kv` and can be larger than their per-head
counterparts when `d_kv > d_head`. The total memory/latency tradeoff depends
on the attention implementation, saved tensors and checkpoint policy.

At DeepSeek-V3 scale (n_h = 128, d_model = 7168, seq = 16384),
the absorbed form's peak activation memory exceeds the explicit
form by **20–34%**, up to **9.2 GB** on a single device.

**Current conclusion:** absorption is conditional, not an inference-only rule.
The local CPU prototype differentiates folded projections exactly and finds
useful small-rank training regimes. This does not validate large-rank absorbed
training or GPU backward. On this Vulkan implementation, even inference at
rank 96/128 fails the latency target despite cache savings.

## 3. LAGA (Latent All-Gather Attention)

LAGA keeps the explicit attention math but changes what crosses the wire:

    All-gather: C_t  (dim d_kv + d_rope = 576 at DS-V3 scale)
    Instead of: K, V (dim n_h (d_nope + d_v) = 32,768 at DS-V3 scale)

Per-token volume reduction: ~57×.
Measured total collective reduction: 1.98× (other collectives dominate).

Reconstruction is **local and transient**: expand K/V from C, run the
attention tile, free the expansion.

## 4. Loop-Axis Low-Rankness

For fixed (token, layer, head), the per-loop K/V vectors trace a
short, low-rank trajectory. This is the empirical finding of LLA:

- Recurrence is the **most compressible cache axis** in looped models.
- Per-head LLA outperforms head-axis MLA at matched cache budget.
- Compression up to 21.3× is near-lossless for generation quality.
- Collapsing to the final loop only (naive truncation) destroys GSM8K.

## 5. Latent Activation Checkpointing (LAC)

Define a low-rank codec:

    Z_t = Down(x_t)          # stored checkpoint, dim r
    x_t' = Up(Z_t)           # reconstructed during backward

Two regimes:

- **Exact:** the forward pass uses Z_t directly. Backward is exact for
  the modified model. This is "low-rank activations," not checkpointing.
- **Approximate:** the forward pass uses full-rank x_t, but backward
  reconstructs x_t' ≈ x_t from Z_t. Gradients can be biased. A residual
  or outlier path can improve reconstruction, but exactness requires
  recovering the necessary full states exactly or recomputing them.

The proposed loop-axis LAC would store one latent per token and reconstruct
per-loop activations on the fly. A KV latent alone does not recover the
full residual/query/MLP states of the measured model. Exact backward needs
additional checkpoints, replay or a different forward architecture.

# Training Strategy

## Measured Exact Loop Checkpointing

The verified CPU path uses a shared KV latent and ordinary exact checkpoints:

    C = Down_KV(Norm(initial_state))     # shared sequence latent
    folds = differentiable_fold(weights)
    state = initial_state
    for i in 1..T:
        state = checkpoint(loop_block, state, use_reentrant=False)
        # loop_block closes over C, folds and the shared block weights
    loss = cross_entropy(classifier(Norm(state)), targets)
    loss.backward()                     # autograd recomputes checkpointed blocks
    optimizer.step()

Checkpoints retain full residual states at loop boundaries. In the measured
50K-vocabulary configuration, LLT storage rises from **1,180.818 MiB at
16 loops to 1,188.818 MiB at 20 loops**: 2 MiB per added loop over that interval.
This yields near-non-loop memory at 20 loops, but does not eliminate T-dependent
training storage. Its **0.58%** extra saving over equally checkpointed naive
is the current architecture-specific memory result.

Earlier toy prefix replay reduced unique saved-tensor storage by **96.7–96.8%**
at 32 loops, but cost **5.73–6.45×** forward/backward time and 496 extra prefix
steps. The measured exact loop policy avoids that quadratic replay schedule.
The lossy rank-16 checkpoint control produced **7.5% input-gradient relative
L2 error**, with **96.4–154.8%** parameter-gradient errors in that fixture.
See [CPU checkpoint findings](benchmarks/README.md#cpu-checkpoint-findings).

## Remaining Training Target

Loop-axis LAC aims to replace full residual checkpoints with compact state
while retaining exact or acceptably accurate training. Reconstructing K/V
alone is insufficient for the measured full-residual model. A redesigned
forward graph, reversible state or additional retained information would
need to establish the memory/quality tradeoff. MELT-style chunking remains
an untested route toward this target; it does not by itself parallelize
dependent residual loops.

## Current Implementation Guidance

- **Checkpointing:** exact per-loop checkpoints are the verified CPU path;
  compare against naive using the same policy.
- **Projection folding:** tied up-projections remove repeated query/output
  projection work; differentiable folding is tested on CPU.
- **Rank:** 32 and 64 qualify for the tested 768D GPU decoder; rank 128,
  an earlier proposal setting, currently has excessive latency.
- **Residuals:** retain or recompute full states for exact gradients.
  A small residual or outlier path does not by itself guarantee exactness.
- **Training fusion and chunking:** further candidates to measure; neither establishes
  the complete proposed training-memory or communication claims.

# Inference

## KV Cache Behavior

| Model               | Cache size               |
|---------------------|--------------------------|
| Standard transformer| O(seq × L × d)           |
| MLA                 | O(seq × L × d_kv)        |
| YOCO                | O(seq × d_kv)            |
| U-YOCO              | O(seq × d_kv)            |
| **LLT**             | **O(seq × d_kv)**        |

LLT's cache is constant in both L and T. The persistent KV cache
is the per-token latent C_t (plus a decoupled RoPE key if implemented).
Weights, residuals and workspaces also occupy memory; only the cache's
independence of L and T follows from this design.

## Decoding

At each step:

1. Compute C_t = Down_KV(Norm(x_t)).
2. Append C_t to the latent cache.
3. For each layer l and loop i:
   - Project the evolving residual into a per-head query using folded weights.
   - Attend directly against cached latents, including the current token.
   - Project the latent accumulator back to the residual width and run the MLP.
4. Apply final normalization and the output classifier.

The measured decoder folds W_UK and W_UV into query/output projections once
before the loops. The CPU training prototype differentiates those folds.
This avoids the full historical K/V expansion used by the earlier unfused path.

## Batch Scaling

The [extended attention measurements](benchmarks/README.md#nine-follow-up-suites)
test batch sizes 2, 4 and 8. At batch 8, context 1,024 and rank 64, absorbed
attention takes **3.782 ms**, versus **3.990 ms** for cached attention: 5.2%
less time. This fixed-query microbenchmark has not received an independent
rerun and does not establish complete-decoder batch capacity. The current
folded decoder regime uses batch 1. Matched-quality serving capacity remains
a target to test, rather than an inherited LLA compression benefit.

# Parallelism and Communication

No LLT distributed run has been measured. Lower KV traffic remains part of
the main value proposition, with three implementation targets:

- **SP/CP:** transfer shared latents and reconstruct or attend locally,
  rather than transfer full expanded K/V.
- **TP:** shard the latent instead of duplicating its cache across ranks.
- **Loop reuse:** reuse transferred latent data across recurrence when the
  schedule permits it.

Attention outputs, residuals, MLPs and gradients can still communicate per
loop. Published LAGA/TPLA results motivate these targets but do not measure
their composition with LLT or justify a total LLT communication speedup.

# Value Proposition and Remaining Targets

LLT's main target is to make deeper recurrence practical by sharing a small
KV latent across layers and loops, while preserving useful model quality and
keeping latency acceptable. Current performance claims use the
[measured operating points above](#provisional-evidence--2026-10-06).

| Value proposition | Current evidence | Remaining target |
|---|---|---|
| Inference memory close to non-loop | Rank 32: 135.746 MiB at both one and ten loops; 91.5% below naive at ten loops | Preserve trained quality with the shared latent |
| Acceptable latency at a useful rank | Rank 32 is 20.4% faster; rank 64 has 4.2% tax in the tested GPU decoder | Verify RoPE, prefill and production workloads |
| Training memory close to non-loop | At 20 loops: 1,188.818 MiB with exact checkpoints, versus 1,200.897 MiB for non-loop LLT | Achieve a substantial additional saving over equally checkpointed naive; currently only 0.58% |
| Exact backward from compact state | Exact residual-boundary checkpointing passes gradient checks | Establish whether a redesigned forward or reconstruction method can eliminate full residual checkpoints |
| Lower distributed KV traffic | No local distributed measurement | Share/shard latents without repeated KV transfers; measure total collectives and synchronization |

Constant total training memory from a single latent and reduced distributed
communication remain core research targets. There is no measured H100/NVLink
throughput or whole-device memory figure. Results from LLA, LAGA or CompAct
provide related-work evidence; their gains cannot be multiplied into an LLT
performance prediction. The [benchmark history](benchmarks/README.md) retains
earlier implementation results and the [regime report](benchmarks/REGIMES.md)
contains current protocols and raw measurements.

# Measured Scaling with Loop Count

The [coarse inference sweep](benchmarks/results/regime-inference/report.json)
holds width 768, 12 layers, rank 32, batch 1 and context 4,096 fixed, using
five GPU timing samples per point:

| Loops | Naive peak MiB | LLT peak MiB | Naive GPU ms | LLT GPU ms |
|---|---:|---:|---:|---:|
| 1 | 306.530 | 135.746 | 5.379 | 4.409 |
| 4 | 738.635 | 135.746 | 21.664 | 17.278 |
| 10 | 1,602.846 | 135.746 | 56.707 | 45.965 |
| 16 | 2,467.057 | 135.746 | 90.885 | 72.959 |

LLT's tensor allocation stays fixed across these loop counts while latency
grows in both models. These are coarse-sweep timings; the independent
nine-sample repeat above is the preferred ten-loop comparison.

For 50K-vocabulary CPU training, increasing loops from 16 to 20 raises naive
uncheckpointed peak storage from **2,181.002 to 2,437.315 MiB** and checkpointed
LLT from **1,180.818 to 1,188.818 MiB**. Checkpointing suppresses the growth
substantially but does not make training memory constant. The equally
checkpointed naive model already reaches **1,195.752 MiB** at 20 loops.

The measured crossover depends on context, vocabulary, optimizer and checkpoint
policy. There is no measured universal threshold derived from width alone.
Useful reasoning quality and time-to-solution at deeper recurrence remain
targets for trained-model evaluation.

# Reasoning Evaluation Targets

The planned quality evaluation focuses on **ARC-AGI, Sudoku and iterative
reasoning tasks used by comparable looped models**. These are evaluation
targets; no trained-model scores have been measured yet.

| Evaluation | Primary outcome | Protocol to record |
|---|---|---|
| ARC-AGI | Exact output-grid match on held-out tasks | Dataset version, split, input representation and allowed prediction attempts |
| Sudoku | Whole-puzzle success: a valid completed grid preserving the givens | Puzzle source, split, difficulty and solution-validity checks; cell accuracy is a diagnostic |
| Other iterative reasoning suites | Task correctness versus loop count and difficulty | Named suites and splits aligned with the reference loop-model evaluation |

Compare non-loop naive, naive looped and LLT models across loop counts and
latent ranks. Match training data and track parameter count and training
compute. For LLA, apply the inference codec to a trained looped teacher and
report codec fitting separately from teacher training.

Report held-out accuracy alongside peak memory, end-to-end latency and
training cost. Include comparisons at the same loop count, the same latency
budget and matched accuracy where achievable. Memory comparisons use the
same checkpoint policy and state each method's latent/cache budget.

The main question is whether additional loops improve reasoning enough to
justify their compute, and whether LLT retains that improvement with its
smaller shared latent. A qualifying synthetic memory/latency point alone
does not establish reasoning quality.

The [2026-10-07 literature check](research/LITERATURE_2026-10-07.md) covers
23 recent papers and identifies direct puzzle baselines and cache-sharing
controls. A whole-grid solver can discard previous-loop activations during
inference, so its naive inference memory need not grow with loop count as
autoregressive historical KV does. Measure the puzzle solver directly.

The current prototype's fixed input cache also lacks a write path for other
positions to read newly inferred solver states. Test an uncompressed static
cache control and a mutable compressed workspace to distinguish rank loss
from communication loss. The workspace variant is a proposed quality ablation,
not part of the current resource measurements.

# Terminology

## Established

- **MLA** — Multi-head Latent Attention (DeepSeek-V2)
- **YOCO** — You Only Cache Once
- **U-YOCO** — Universal YOCO (iterated self-decoder, shared cache)
- **MELT** — Memory-Efficient Looped Transformer (gated constant-size state)
- **LLA** — Looped Latent Attention (post-training loop-axis cache codec)
- **LAGA** — Latent All-Gather Attention (explicit-form latent SP)
- **CompAct** — Compressed Activations (low-rank activation storage)
- **Adacc** — Adaptive Compression + Activation Checkpointing
- **TPLA** — Tensor Parallel Latent Attention

## Proposed Here

- **LLT** — Loop-Latent Transformer (the merged architecture)
- **LAC** — Latent Activation Checkpointing
  (general technique: low-rank latent as the checkpoint format)
- **Loop-LAC** — LAC applied to the loop axis of a looped transformer
  (store one latent per token, expand per layer per loop during backward)

## Rationale for "Latent"

"Latent" is chosen deliberately:

- It connects to MLA's **latent KV cache**.
- It connects to LLA's **latent loop-axis compression**.
- It signals that the checkpoint is a **learned, structured compression**,
  not merely a truncated or quantized activation.

## Not Recommended

- "Compressed checkpointing" — too generic, collides with Adacc.
- "Low-rank checkpointing" — accurate but loses the loop-axis specificity.
- "Latent replay" — ambiguous with latent replay in RL.

# Related Work

## Recent Looped Models — Literature Check, 2026-10-07

See the [complete 23-paper review](research/LITERATURE_2026-10-07.md) for
version-pinned sources, reported evidence and experiment priorities.

- [**LPT**](https://arxiv.org/html/2610.02383v1) reuses first-loop context KV
  per layer while retaining later-loop local windows.
- [**Rethinking at Fixed Points**](https://arxiv.org/html/2610.06833v1)
  studies terminal KV sharing and truncated-gradient training.
- [**Gated Recurrent Transformers**](https://arxiv.org/html/2608.15062v4)
  tests first, last and averaged recurrent KV with evolving gated state.
- [**Thinking with Looped Flows**](https://arxiv.org/html/2609.11801v1) and
  [**GRAM**](https://arxiv.org/html/2605.19376v2) are direct ARC-AGI/Sudoku
  baselines with evolving solver states and different training objectives.

Sharing KV across loops is established prior work. LLT's remaining target is
trained reasoning quality with a smaller cache shared across layers and loops,
alongside a measured memory/latency advantage. Exact finite-depth checkpointing
and truncated/local-objective training require separate comparisons.

## Compression of KV Cache

- **MLA (DeepSeek-V2)** — low-rank KV latent, decoupled RoPE.
  Its cited training implementation disables absorption; this is not a
  general impossibility result for differentiable folding at smaller ranks.
- **YOCO** — one global KV cache, cross-decoder layers reuse it.
- **U-YOCO** — iterated self-decoder, constant cache, no compression.
- **LLA** — loop-axis codec, 21.3× compression, batch 32 → 768 on H200.
  Post-training, inference-only.
- **FlashLoop** — cross-loop KV sharing, residual quantization,
  token-sparse updates.

## Memory-Efficient Looped Training

- **MELT** — gated shared KV state per layer, chunk-wise training,
  constant memory in loop count.
- **CompAct** — low-rank projected activations for weight gradients.
  25–30% memory reduction pretraining, 50% fine-tuning.
- **Adacc** — adaptive compression + checkpointing, MILP scheduler.
- **BOOST** — low-rank activation checkpointing for bottleneck
  architectures.

## Communication Reduction

- **LAGA** — latent all-gather, explicit attention, 1.98× measured
  collective reduction, ≤0.5% memory overhead vs explicit form.
- **TPLA** — tensor-parallel latent attention, no KV duplication.

## Reversibility

- **RevNets** — analytically invertible blocks, no stored activations.
  Not trivially compatible with MLA / looped / YOCO.

## Intended Combination

The proposal aims to combine:

1. Loop-axis latent compression (LLA),
2. Cross-layer cache sharing (YOCO / U-YOCO),
3. Loop-decoupled training (MELT),
4. Low-rank activation checkpointing (CompAct / LAC),
5. Latent-only communication (LAGA / TPLA).

The full five-part composition around a shared latent C_t remains a research
target. The local prototypes validate selected resource properties of a
modified forward model and exact checkpoints, not the complete combination
or a literature-wide novelty claim.

# Open Questions

## 1. Is the loop-axis latent exact or approximate?

If the forward pass uses C_t directly, training is exact for the
modified model. If the forward pass uses full-rank activations and
only backward reconstructs from C_t, gradients are biased.

**Question:** which regime preserves downstream quality at T = 16+?

## 2. How does LAC interact with chunk-wise training?

MELT's gated state introduces a sequential dependency across chunks.
LAC introduces a reconstruction step in backward. Do these compose
cleanly, or does the gated state need its own checkpoint?

## 3. Does absorption ever help in training?

The CPU prototype demonstrates exact differentiable folding and useful
small-rank timing regimes. The relevant open questions are the GPU-backward
crossover, larger ranks, RoPE and quality at useful training scales.

## 4. How does the latent behave under TP sharding?

TPLA shards the latent. Does reconstruction remain exact per shard?
Does the all-reduce of attention outputs preserve the LAGA guarantees?

## 5. What is the right loop-conditioning for W_K^{l,i}, W_V^{l,i}?

- Tied across loops (U-YOCO regime): minimal parameters, maximum sharing.
- Loop-specific: retains one shared KV latent, but adds loop-specific maps
  and may increase weight storage/folding work.
- Low-rank loop embedding: middle ground.

## 6. Does the LLA compression ratio hold under training?

LLA is a **post-training** codec. Its 21.3× figure is measured on
already-trained models. Training with a low-rank bottleneck from
scratch may produce different (possibly better) loop-axis structure.

## 7. What is the wall-clock cost of transient expansion?

The earlier rank-128, 4K decoder's absorbed path took **134.966 ms** versus
**53.088 ms** for its matched naive-cache path; tile-fused expansion took
**812.852 ms**. The current folded rank-32/rank-64 regimes qualify. These
measurements show that expansion/fusion and rank must be evaluated in their
actual schedule. GPU backward and optimized matrix-kernel performance remain
unmeasured; see the [decoder results](benchmarks/README.md#complete-decoder-and-allocation-protocol).

## 8. Can RevNet-style reversibility be combined with MLA?

If the shared block were invertible, backward could reconstruct exact
activations from the output alone—no latent checkpoint needed. But
MLA's down/up projections are not trivially invertible. Is there a
reversible MLA variant?

## 9. How does LLT behave under multi-node training?

The 1.98× LAGA figure is single-node (8× Ascend 910B). Cross-node
collectives have different latency/bandwidth trade-offs. Does the
latent all-gather remain the dominant win at 64+ GPUs?

## 10. Does deeper recurrence improve reasoning within a useful resource budget?

The [evaluation targets](#reasoning-evaluation-targets) are ARC-AGI, Sudoku
and comparable iterative reasoning suites. Does accuracy improve with more
loops on held-out tasks, and at what latent rank does LLT retain the naive
looped model's accuracy? Compare accuracy versus latency and memory, with
matched checkpoint policies, explicit cache budgets and training costs.
