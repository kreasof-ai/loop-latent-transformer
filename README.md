# Loop-Latent Transformer (LLT)

A merged architecture combining **Multi-head Latent Attention (MLA)**,
**Looped Transformers**, and **YOCO / U-YOCO**, with a novel training
strategy called **Latent Activation Checkpointing (LAC)**.

## Core Idea

A naive looped transformer reuses one block T times, but stores a
separate KV cache and a separate set of activations for every loop.
Memory and communication scale linearly with T.

LLT observes that:

1. MLA already compresses KV into a low-rank latent.
2. Looped transformers produce highly redundant per-loop K/V trajectories.
3. YOCO / U-YOCO already share a single global KV cache across layers.

By **compressing along the loop axis** and **sharing the compressed latent
across loops and layers**, LLT makes inference cache, training activation
memory, and inter-device communication all **constant in T**.

# Architecture

## Block Layout

LLT is a decoder-only transformer with three structural deviations
from a standard NanoGPT-style model:

1. **A single shared block** is applied T times (looped transformer).
2. **Attention uses MLA-style latent KV** (low-rank compressed cache).
3. **The latent is shared across layers and loops** (YOCO / U-YOCO style).

## Forward Pass (per token t)

    C_t   = Down_KV(x_t)                # low-rank latent, dim d_kv
    q_t   = W_Q x_t                     # per-head query, decoupled RoPE
    for i in 1..T:                       # loop index
        for l in 1..L:                   # layer index (shared block reused)
            K_{t,l}^{(i)} = W_K^{l,i} C_t
            V_{t,l}^{(i)} = W_V^{l,i} C_t
            a = softmax(q_t K^T / sqrt(d)) V
            x_t = x_t + OutProj(a)

Crucially:

- `C_t` is computed **once per token**.
- `K, V` are expanded **per layer, per loop, on the fly**.
- No per-loop or per-layer KV is ever persisted.

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

The up-projections can be tied across loops (`W_K^{l,i} = W_K^l`) to
recover the U-YOCO regime, or untied to allow loop-specific expansion.

# Mathematical Foundations

## 1. MLA Matrix Absorption (Inference)

Standard MLA attention:

    Attn = softmax( Q (W_UK C)^T / sqrt(d) ) (W_UV C)

Absorption reorders the multiplications:

    Attn = softmax( (W_UK^T Q) C^T / sqrt(d) ) C W_UV^T

This eliminates materialization of K = W_UK C and V = W_UV C.
The compressed latent C is the only persistent state.

## 2. Why Absorption Is a Memory Trap in Training

The absorbed form materializes:

- `q_absorbed = W_UK^T q_nope` with shape (n_h, d_kv)
- a post-attention latent accumulator with shape (n_h, d_kv)

These intermediates are larger than the per-head K/V they replace
whenever `d_kv > d_head`.

At DeepSeek-V3 scale (n_h = 128, d_model = 7168, seq = 16384),
the absorbed form's peak activation memory exceeds the explicit
form by **20–34%**, up to **9.2 GB** on a single device.

**Conclusion:** use absorption for inference, but never for training.

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
  reconstructs x_t' ≈ x_t from Z_t. Gradients are biased unless a
  residual or outlier path is added.

For looped transformers, LAC is applied **along the loop axis**:
store one latent per token, reconstruct per-loop activations on the fly.

# Training Strategy

## Chunk-Wise Training (MELT-style)

Problem: a gated latent state updated across loops introduces a
sequential dependency that prevents fully parallel training.

Solution: split the sequence into chunks. Within each chunk, run the
loops in parallel. Across chunks, carry the gated latent forward.

This gives:

- Constant activation memory in T (loop count)
- Bounded sequential dependency (chunk length, not sequence length)
- Compatibility with activation checkpointing

## Latent Activation Checkpointing (LAC) for the Loop Axis

Forward:

    for each token t:
        C_t = Down_KV(x_t)        # store ONLY this
        for i in 1..T:
            for l in 1..L:
                K, V = expand(C_t, l, i)
                x_t = attention(x_t, K, V)

Backward:

    for each token t:
        load C_t                  # from checkpoint
        grad_C = 0
        for i in 1..T:
            for l in 1..L:
                K, V = expand(C_t, l, i)   # recompute, transient
                grad_C += backward_through_expand(K, V)
        grad_Down = backward_through_down(grad_C)

Memory: O(seq × r) instead of O(seq × L × T × d).
Compute: + one up-projection per (layer, loop) during backward.

## Recommended Settings

- **Activation checkpointing granularity:** per chunk, not per layer.
- **Reconstruction:** explicit (LAGA-style), never absorbed.
- **Residual path:** small full-rank residual for near-exact gradients
  if loss curve diverges.
- **Outlier handling:** quantization-aware or sparse-outlier path
  (Adacc-style) if the latent distribution is heavy-tailed.
- **Fusion:** fuse expand + attention into one kernel so K/V never
  persist in HBM.

# Inference

## KV Cache Behavior

| Model               | Cache size               |
|---------------------|--------------------------|
| Standard transformer| O(seq × L × d)           |
| MLA                 | O(seq × L × d_kv)        |
| YOCO                | O(seq × d_kv)            |
| U-YOCO              | O(seq × d_kv)            |
| **LLT**             | **O(seq × d_kv)**        |

LLT's cache is constant in both L and T. The only persistent state
is the per-token latent C_t (plus the decoupled RoPE key).

## Decoding

At each step:

1. Compute C_t = Down_KV(x_t).
2. Append C_t to the latent cache.
3. For each layer l and loop i:
   - Expand K, V from C_t locally.
   - Run attention against the cached latents.
   - Free the expansion.

Absorption can be used here (it is an inference-only optimization):
fold W_UK and W_UV into the query and output projections.

## Batch Scaling

LLA reports that 21.3× loop-axis compression increases batch capacity
from 32 to 768 sequences on a single H200 at matched quality.
LLT inherits this and adds cross-layer sharing on top.

# Parallelism and Communication

## Sequence Parallelism (SP) / Context Parallelism (CP)

Standard explicit MLA SP communicates:

    n_h × (d_nope + d_v) = 128 × 256 = 32,768 elements / token

LLT communicates:

    d_kv + d_rope = 512 + 64 = 576 elements / token

Per-token reduction: ~57×.
Measured total collective reduction (LAGA): 1.98× on 8× Ascend 910B.

For CP, the same substitution applies. NVIDIA's Megatron-LM has a
PR that overlaps the compressed-KV CP all-gather with independent
compute, confirming this is an active engineering direction.

## Tensor Parallelism (TP)

Standard MLA TP duplicates the KV cache across ranks.
TPLA partitions the latent and each head's input dimension, runs
attention independently per shard, and all-reduces attention outputs.

LLT inherits TPLA's structure:

- Latent is partitioned, not duplicated.
- All-reduce volume scales with query dim, not KV dim.

## Loop Axis as a Free Dimension

Because C_t is shared across loops, communication happens **once per
token per chunk**, not once per loop. Adding loops adds no wire traffic.

| Axis | Naive looped | LLT |
|------|--------------|-----|
| SP/CP| O(T × n_h × d) | O(d_kv) |
| TP   | O(T × d_kv) duplicated | O(d_kv / tp) partitioned |
| Loop | O(T × d_kv) | 0 |

# Estimated Performance (8×H100, NanoGPT-style)

These are **estimates** synthesized from published results for the
individual techniques (MELT, LLA, LAGA, CompAct) and standard NanoGPT
speedrun performance. They are not measured for this exact combination.

A [local RX 6700 XT Vulkan and CPU benchmark report](benchmarks/README.md) now
measures attention, a synthetic complete decoder forward, live tensor allocations,
and toy checkpoint gradients. It verifies cache savings with latency tradeoffs,
but does not validate the H100 throughput or total-memory estimates below.

## Setup

- Model: NanoGPT-style decoder
- Shared block: L = 12 layers, d_model = 768, n_h = 12
- Loops: T = 10
- Latent: d_kv = 128 (≈ 6× compression vs d_model)
- Hardware: 8×H100 SXM, NVLink

## Naive Looped Transformer

| Metric | Value |
|--------|-------|
| Throughput | ~15,000 – 20,000 tokens/sec |
| Peak memory | ~65 – 75 GB / GPU |
| KV cache | O(seq × L × T × d) |
| Activations | O(seq × L × T × d_model) |

## Optimized Loop-Latent Transformer

| Metric | Value |
|--------|-------|
| Throughput | ~40,000 – 55,000 tokens/sec |
| Peak memory | ~25 – 35 GB / GPU |
| KV cache | O(seq × d_kv) |
| Activations | O(seq × r) |

## Relative Improvement

| Axis | Ratio |
|------|-------|
| Throughput | 2 – 3× |
| Peak memory | 2 – 3× |
| SP/CP wire volume | ~57× per token, ~2× total |
| TP KV duplication | eliminated |
| Loop-axis cost | T× → 1× |

## Caveats

- The 21.3× LLA figure is **inference-only**. Training gains require
  baking the low-rank bottleneck into the forward graph.
- The 1.98× LAGA figure is **measured on 8× Ascend 910B**, not H100.
- CompAct's 25–50% activation savings are for **general activations**,
  not loop-axis latents specifically.
- The proposed loop-axis LAC training system is **novel and unvalidated**.
  Local toy replay passes exact gradient checks but exposes recomputation cost;
  lossy full-state reconstruction produces biased gradients.

# Scaling with Loop Count

Let T = number of loops.

## Naive Looped Transformer

    Memory(T)     ∝ T
    Compute(T)    ∝ T
    Comm(T)       ∝ T

## Loop-Latent Transformer

    Memory(T)     ≈ C_0          (constant)
    Compute(T)    ≈ T × c_expand (expansion is cheap)
    Comm(T)       ≈ C_comm       (constant)

## Relative Gain

    Gain(T) ≈ T / (1 + (ε/C_0) T)

As T → ∞, Gain(T) → C_0/ε, the compression ratio of the latent.

At small T: super-linear improvement.
At large T: saturates at the latent's compression ceiling.

## Empirical Reference Points

- YOCO-U shows consistent gains scaling loops 1 → 5 with a constant
  global KV cache.
- FlashLoop reports that gains from cross-loop sharing **grow with
  deeper recurrence**.
- LLA shows 21.3× compression is achievable at matched quality on
  the loop axis.

## Practical Implication

The loop axis is the **one axis where adding compute costs the naive
model linearly but the optimized model essentially nothing**. This
makes LLT attractive for reasoning-heavy tasks where T is large
(e.g., T = 16 or 32 loops for multi-step reasoning).

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

## Compression of KV Cache

- **MLA (DeepSeek-V2)** — low-rank KV latent, decoupled RoPE.
  Inference-only absorption; training explicitly disables absorption.
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

## Gap LLT Fills

No existing work combines:

1. Loop-axis latent compression (LLA),
2. Cross-layer cache sharing (YOCO / U-YOCO),
3. Loop-decoupled training (MELT),
4. Low-rank activation checkpointing (CompAct / LAC),
5. Latent-only communication (LAGA / TPLA).

LLT is the union of these five, applied simultaneously to a single
shared latent C_t.

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

Megatron-Core hard-asserts against it. But for very small d_kv / d_head
ratios, the absorbed form might be cheaper. Is there a threshold?

## 4. How does the latent behave under TP sharding?

TPLA shards the latent. Does reconstruction remain exact per shard?
Does the all-reduce of attention outputs preserve the LAGA guarantees?

## 5. What is the right loop-conditioning for W_K^{l,i}, W_V^{l,i}?

- Tied across loops (U-YOCO regime): minimal parameters, maximum sharing.
- Loop-specific: more expressive, but breaks the "one latent" claim.
- Low-rank loop embedding: middle ground.

## 6. Does the LLA compression ratio hold under training?

LLA is a **post-training** codec. Its 21.3× figure is measured on
already-trained models. Training with a low-rank bottleneck from
scratch may produce different (possibly better) loop-axis structure.

## 7. What is the wall-clock cost of transient expansion?

LAGA matches explicit-form memory within 0.5%. But the expand-and-free
pattern in backward adds kernel launches. Is the net throughput gain
positive at NanoGPT scale, or only at frontier scale?

## 8. Can RevNet-style reversibility be combined with MLA?

If the shared block were invertible, backward could reconstruct exact
activations from the output alone—no latent checkpoint needed. But
MLA's down/up projections are not trivially invertible. Is there a
reversible MLA variant?

## 9. How does LLT behave under multi-node training?

The 1.98× LAGA figure is single-node (8× Ascend 910B). Cross-node
collectives have different latency/bandwidth trade-offs. Does the
latent all-gather remain the dominant win at 64+ GPUs?

## 10. What is the right benchmark for looped reasoning?

GSM8K, MATH, and code generation stress different loop depths.
LLT's advantage grows with T, so the benchmark choice determines
how large the measured gain appears.
