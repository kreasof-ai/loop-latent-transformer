# Architecture

The current LLT uses **one contextual rank-R latent per physical layer**, built
on the first loop and reused on later loops. There are L cache banks, independent
of T. Blocks and projections are tied across loops. The full-width residual
continues to evolve through all LT block applications.

This replaces the historical globally shared embedding-derived latent. Its
measurements remain in the [historical report](../archive/reports/L40S_GLOBAL_LATENT_SWEEP.md).
The separate per-layer latent control refreshes memory at every layer/loop
application and therefore keeps LT banks; it is a different architecture.

## Computation

Let X be the evolving residual, width W, heads H, head dimension D=W/H, layers L,
loops T, and latent rank R. X starts from token plus absolute position embeddings.
Each physical layer i has its own down projection A_i and K/V expansion weights.
Using Torch linear-weight conventions:

```text
F_Q[i,h] = U_K[i,h]^T W_Q[i,h]                 [R,W]
F_O[i,:,h,:] = W_O[i,:,h,:] U_V[i,h]           [W,R]

for loop t = 0 .. T-1:
    for layer i = 0 .. L-1:
        Z = RMSNorm(X)
        if t == 0:
            C_i = Z A_i^T                    [B,S,R]
        Q_r = Z F_Q[i]^T                     [B,H,S,R]
        A = softmax(causal(Q_r C_i^T / sqrt(D))) C_i
        X = X + flatten_heads(A) F_O[i]^T
        X = X + W_2[i] GELU(W_1[i] RMSNorm(X))
```

The first layer's C_0 comes from embeddings. Deeper C_i include preceding
first-loop computations. Later loops reuse these first-loop representations;
they do not update historical memory with later-loop states. This makes the
cache constant in T while preserving contextual first-loop memory per layer.
It does not establish trained quality or equivalence with a conventional model.

Folding is equivalent to expanding K_h=C_i U_K[i,h]^T and
V_h=C_i U_V[i,h]^T. Attention scaling remains 1/sqrt(D), not 1/sqrt(R).
Folds and C_i remain differentiable; later-loop gradients flow back into the
first-loop memory construction. No latent is detached.

## Causal serving

Prefill constructs all L memories during the first pass. Each bank stores a
single latent tensor physically shared by K and V. On a new token, the first
loop constructs and appends its C_i at each layer, using that token's evolving
first-loop state. Later loops read the same banks without appending again.
Prompt/startup currently computes all query positions on every loop. Once the
first-loop memories exist, later loops could evaluate only the requested last
position; this optimization is not used in the measured implementation.

This is causally consistent with full-sequence execution. Historical bank
contents and allocated addresses remain stable across decode steps.

## Parameters and cache scaling

The current study uses B4/S1024, W768, H12, L12, V50304, position capacity P1025,
MLP width F3072, exact GELU and unweighted RMSNorm (epsilon 1e-5). Masters,
embeddings, residuals, optimizer state and loss are FP32; projections and
attention are BF16. Serving retains BF16 prepared projection copies in addition
to FP32 masters; these count toward total peak memory.

With E=2VW+PW, LLT has:

```text
Parameters = E + L(2WF + 2W^2 + 3WR)
BF16 KV payload = 2BSRL bytes
```

| Architecture | Cache banks | BF16 cache payload |
|---|---:|---:|
| Current LLT | L | 2BSRL |
| Per-layer latent control | LT | 2BSRLT |
| Naive Loop / independent stack | LT full K/V pairs | 4BSWLT |
| Fixed depth / matched parameters | L full K/V pairs | 4BSWL |
| Historical global input-latent prototype | 1 | 2BSR |

Cache grows linearly with physical layer count L and latent rank R. Increasing
loop count T does not add banks. The memories stay fixed after their first-loop
construction; this sharing changes the model compared with refreshed loop states.

Serving allocates 1025 slots. Tensor adds 8B bytes of counters per bank.
At rank 64, current LLT has approximately 6 MiB of KV payload at every T;
the refreshed per-layer control reaches approximately 96 MiB at T16.
Payload is separate from weights, prepared copies, folds, temporaries and
activation storage. Uncheckpointed training memory still grows with T.

The fixed-depth control keeps L blocks and widens its MLP to W(6T-2), matching
the independent stack's parameters. Equal parameter counts do not match compute.

## Checkpointing and code

The current LLT supports none and exact full-block AC. On the first loop,
AC recomputes latent construction and the complete block, returning both the
updated residual and C_i. On later loops its inputs include the retained C_i.
LAC is excluded; low-rank KV does not suffice to reconstruct full residuals.
See [checkpoint boundaries](CHECKPOINTING.md).

- [contextual_llt.py](../model/contextual_llt.py): current architecture, AC, and cached decode.
- [tensor_backend.py](../model/tensor_backend.py): common Torch/Tensor operators, losses, prefill allocation and historical model.
- [reference.py](../model/reference.py): configuration and frozen historical global-latent algebra reference.
- [research_baselines.py](../model/research_baselines.py): refreshed per-application latent control and other families.
- [contextual_llt_sweep.py](../experiments/l40s/contextual_llt_sweep.py): current profiling/qualification harness.

[Tensor](https://github.com/kreasof-ai/tensor) owns numerical kernels and runtime;
this repository owns the architecture, orchestration, and results.
