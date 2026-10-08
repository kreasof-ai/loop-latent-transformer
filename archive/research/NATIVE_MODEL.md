# Architecture frozen for the qualified Tensor L40S study

> Historical study. The current main result is [LOOP_SWEEP.md](../../experiments/l40s/LOOP_SWEEP.md); see the [archive index](../README.md) for context.

The experiment in `experiments/l40s/native_study.py` implements a causal,
initial-input static-cache model. It is a simplified LLT composition, not an
implementation of the YOCO/U-YOCO self-decoder or an iterative mutable workspace.
Its resource results do not establish learned reasoning quality.

Let width be W, heads H, head width D = W/H, latent rank R, layers L and
recurrence count T. A block's parameters are reused on every recurrence. The
global model has one down-projection A shared by every layer and recurrence.
The layerwise control has L down-projections, also reused across recurrences.
There is no learned loop embedding or loop-dependent conditioning. An untied
parameter option exists in the implementation but is outside this study.

The input residual is token embedding plus learned absolute position embedding.
Training master weights and residuals are FP32. Projections and attention are
BF16, with FP32 accumulation. Serving prepares BF16 projection weights once;
embeddings, normalization and residuals remain FP32. RMS normalization is
unweighted with epsilon 1e-5.
Every attention and 4W-wide MLP sublayer has a full residual connection. Exact
GELU is used by both new backends; the archived attention-only study used the
tanh approximation. Position encoding and normalization have no new trainable
parameters in this integration. The classifier is untied from token embeddings.

For global LLT, compute C = norm(input) A^T once. Each block has a query
projection Q_h (D by W), key up-projection U_kh (D by R), value up-projection
U_vh (D by R), and output projection O (W by HD). Absorb the up-projections:

```text
folded_query_h = U_kh^T Q_h                  [R, W]
folded_output_h = O[:, hD:(h+1)D] U_vh       [W, R]
query_h = norm(residual) folded_query_h^T
attention_h = softmax(query_h C^T / sqrt(D) + causal_mask) C
attention_delta = concat(attention_h) folded_output^T
residual = residual + attention_delta
residual = residual + W2 GELU(W1 norm(residual))
```

The scale remains 1/sqrt(D), rather than 1/sqrt(R). Batched GEMMs construct all
head folds together. The folds are differentiable and refreshed on every
training/prefill forward. In generation they are prepared once and invalidated
when a parameter's PyTorch version changes. Inference assumes parameters are
updated through normal version-tracked operations, rather than `.data` writes.

The global cache stores C once, physically aliasing keys and values, with one
stored KV head. The layerwise control stores L such latents. Naive looping
computes ordinary MHA keys and values from the current residual at each block
invocation and stores LT cache pairs. With BF16 and fixed serving capacity S:

| Model | Cache payload bytes |
|---|---:|
| Global LLT | 2 B S R |
| Layerwise LLT | 2 B S R L |
| Naive loop | 4 B S W L T |

Tensor adds two int32 control vectors per cache (8B bytes). The Torch reference
uses Python lengths. Payload and control bytes are counted separately from
folded weights and total allocator peaks. Capacity, not current logical length,
determines physical cache allocation. Decode appends into preallocated storage;
it never recopies the historical prefix. The timed four-token trajectory includes
current-token cache writes, attention, residuals, MLPs and classifier. Fixture
rewind happens before each measurement and outside graph capture. Folding and
raw prefix construction are reported in the causal-forward prefill. Conversion
to preallocated serving capacity and generation-fold preparation are outside
the reported timing. CUDA graph replay uses a fixed four-token trajectory from
the same prefix, not a dynamic production serving scheduler.

The static initial-input cache cannot transmit newly inferred states from one
position to another. This is an architectural limitation to investigate with
uncompressed static and mutable compressed controls in future quality studies.
Learned absolute positions bound supported lengths to the configured embedding
table. RoPE, variable per-sequence lengths, dropout and arbitrary attention masks
are outside this LLT experiment, even where the Tensor dependency supports them.

Training compares no checkpointing with exact non-reentrant checkpointing at
each loop boundary. Checkpoint arguments include shared latents and folds, so
all gradient paths survive recomputation. Full residual boundary states remain
necessary. No reconstruction of residual/query/MLP activations from C is claimed,
and ordinary loop checkpointing is not latent activation checkpointing. Constant
cache bytes in T do not imply constant total training memory or computation.

The separate [rank/AC/LAC sweep](../../experiments/l40s/LOOP_SWEEP.md) adds exact
per-block AC and LLT's native latent attention/output checkpoint region. LAC
preserves projected queries Q_r, shared C and the folded output weight as inputs;
query formation, residual and MLP paths stay outside that region. It introduces
no residual codec or approximate reconstruction. Full-size checkpoint gradient
checks use deterministic Torch execution and unchanged Tensor kernels; the
primary latency/memory profiles keep their original execution settings.

Tensor executes the numerical projections, batched folds, embeddings, RMSNorm,
GELU, residual adds, attention and its backward, full cross entropy, clipping and
AdamW. PyTorch controls storage/layout, autograd, checkpoint scheduling and tied
gradient accumulation. The reference forces Flash SDPA and uses PyTorch AdamW
with `foreach=False`. Both have FP32 optimizer state and clip global gradient
norm to 1.0. The optional Tensor streamed classifier is not used in the matched
full-loss sweep. Its separate dependency measurements are a memory/time tradeoff.

Correctness uses the original FP64 folded/unfolded model as an independent
algebra reference, then tests BF16 outputs, every parameter gradient, exact
checkpoint gradients, short matched training, causality, four consecutive cached
tokens from trained weights, stable cache addresses and stale-weight rejection.
The larger systems sweep uses synthetic tokens and initial random weights for
inference; the measured training steps update weights but are not a quality study.
