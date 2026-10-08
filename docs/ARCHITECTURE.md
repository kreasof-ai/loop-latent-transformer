# Architecture

The implemented LLT shares one input-derived KV latent across all blocks and
loops. It repeats L parameterized blocks T times, carrying a full-width residual
state between applications. This reduces cached KV storage without reducing the
residual width or making the residual stream reconstructible from the KV latent.

## Computation

For batch B, sequence S, width W, heads H, head dimension D=W/H, and latent rank R,
let X₀ be token embeddings plus learned absolute position embeddings. LLT forms
one causal memory source:

```text
C = RMSNorm(X₀) W_downᵀ                    [B, S, R]
```

At block i, queries come from the evolving residual X. Each head has latent
expansion weights U_K and U_V of shape [D,R]. With the usual Torch linear-weight
convention, folding gives:

```text
F_Q[h] = U_K[h]ᵀ W_Q[h]                  [R, W]
F_O[:,h,:] = W_O[:,h,:] U_V[h]            [W, R]
Q_r = RMSNorm(X) F_Qᵀ                    [B, H, S, R]
A = softmax(causal(Q_r Cᵀ / sqrt(D))) C   [B, H, S, R]
X' = X + flatten_heads(A) F_Oᵀ
X_next = X' + W₂ GELU(W₁ RMSNorm(X'))
```

The attention scale stays **1/√D**, even when attention runs at latent rank R.
This preserves equivalence with explicitly expanded K=C U_Kᵀ and V=C U_Vᵀ.
Folds remain differentiable during training. Their reuse avoids full-width K/V
materialization; the separate per-block expansion weights remain trainable.

C is fixed across all block applications for a given token sequence. Historical
KV values do not acquire later loop states. Queries can change what each token
reads, but this implementation does not exchange evolving full-width states
through a refreshed KV memory. That distinction matters for future quality tests.

The measured model has no learned normalization scale or biases. It uses
unweighted RMSNorm (epsilon 1e-5), exact GELU, independent token/output weights,
and learned absolute positions. FP32 master weights, embeddings, and residuals
are retained; projections and attention run in BF16. See the main report for
optimizer and timing details.

## Controls in the main experiment

| Variant | Blocks stored | Blocks applied | KV source |
|---|---:|---:|---|
| LLT | L | LT | One shared input-derived C |
| Naive Loop | L | LT | Full-width K/V from each application's current residual |
| Independent stack | LT | LT | Full-width K/V at each independent block |
| Fixed depth / matched parameters | L | L | Conventional K/V; MLP widened to match the stack |

The fixed-depth control has one pass through L blocks. Its MLP hidden width is
F=W(6T−2), exactly matching the independent stack with F=4W while holding width,
heads, vocabulary, and embeddings fixed. Equal parameters do not imply equal
attention cost, expressiveness, or activation memory.

The code also retains an earlier `layerwise` latent control for historical
comparisons. It is outside the main sweep and is not a faithful reproduction of
YOCO or DeepSeek. Its earlier specification is in the archive.

## Parameter and cache scaling

Let V be vocabulary size, P position capacity, and E=2VW+PW the common embedding
and output parameters. The measured bias-free models have:

| Variant | Trainable parameters |
|---|---|
| LLT | E + L(2WF + 2W² + 2WR) + WR |
| Naive Loop | E + L(2WF + 4W²) |
| Independent stack | E + LT(2W·4W + 4W²) |
| Fixed depth | E + L(2W·W(6T−2) + 4W²) |

For BF16 cache payload at a historical length S:

| Variant | Cache bytes |
|---|---:|
| LLT | 2BSR |
| Naive Loop | 4BSWLT |
| Independent stack | 4BSWLT |
| Fixed depth | 4BSWL |

Actual serving caches allocate a capacity of 1025 slots in this study. Tensor
cache counters are small additional state; prepared projection weights and model
parameters are also resident. The formulas describe KV payload, not total GPU
memory. LLT's full residual and MLP activations still grow with applied depth in
uncheckpointed training. The shared KV latent cannot recover those activations.

## Code map

- [reference.py](../model/reference.py): configuration, folding, full forward, and the small Torch reference.
- [tensor_backend.py](../model/tensor_backend.py): explicit numerical backend, losses, KV allocation, and cached-token decode.
- [checkpointing.py](../model/checkpointing.py): block AC and the current experimental LAC boundary.
- [prepared.py](../inference/prepared.py): BF16 serving copies for the Torch control.
- [loop_sweep.py](../experiments/l40s/loop_sweep.py): measured geometry and exact parameter matching.

[Tensor](https://github.com/kreasof-ai/tensor) owns its CUDA kernels and runtime.
This repository owns LLT's architecture and experiment orchestration.
