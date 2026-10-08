# Additional architecture-family kernel profiles

These are random-weight **kernel and memory profiles**, with no quality evaluation.
They use the main study's B4/S1024, width 768, 12 heads, vocabulary 50,304,
independent token/output weights, FP32 masters and BF16 projections. Residuals are FP32 in the original controls
and four added families. GRT's recurrence projection emits BF16 core residuals;
its prelude/coda and retention blend use FP32 residuals. This follows the
projection/blend execution in the reference topology and is recorded in the CSV. T sweeps 1..16.
All model adaptations are explicit; these rows are not reproductions of trained
paper checkpoints or published quality numbers.

## Architecture and work

| Profile | Physical blocks | Applied attention blocks | FFN applications | Persistent KV |
|---|---:|---:|---:|---|
| U-YOCO / SWA | 6 self + 6 cross | 6T + 6 | 6T + 6 | One global full-width bank; local windows per self-layer application |
| LPT cache layout | 12 | 12T | 12T | First-loop full KV per layer, later-loop local KV |
| GRT / full KV (BF16 core) | 2 prelude + 8 core + 2 coda | 4 + 8T | 4 + 8T | Full per-application KV; no averaged-cache substitution |
| Per-layer latent loop (MLA-style) | 12 | 12T | 12T | Rank-64 latent per layer application |
| Attention-only loop / FFN once | 12 | 12T | 12 | Full per-attention-application KV |

FFN counts refer to the Transformer feed-forward updates. GRT additionally runs
one gate MLP and one recurrence projection per loop; both are timed and recorded
separately in the joined CSV.

The same T therefore represents different work allocations. Parameter counts,
applied attention/FFN counts, cache bytes, prepared weights, latency and peak memory
are recorded individually. These controls do not impose an additional iso-parameter
or iso-FLOP match. The existing independent-stack and fixed-depth matched-parameter
controls remain available in the main report.

## Source fidelity and controlled changes

**U-YOCO:** [Universal YOCO, 2604.01220v1](https://arxiv.org/html/2604.01220v1).
The 6-layer self-decoder alone loops, using causal SWA with 512 slots including
the current token, RoPE base 10,000, weighted RMSNorm with epsilon 1e-5,
and SwiGLU hidden width 3W. Its final
output forms one full-width global K/V bank. The 6-layer cross-decoder runs once,
uses NoPE, and shares that bank. Self-decoder windows retain separate histories
for every layer/loop application; this storage is counted. It is not constant in T.
The study uses 12 KV heads rather than the paper's grouped-KV configuration and
its fixed nanoGPT-scale geometry. Token/output weights are independent. The final
cross-decoder can evaluate only the last prompt position because it reads a fixed
memory and has no cross-position writes; both prompt and serving-startup profiles
use this valid last-position optimization. Training computes every token's logits.
The preserved LLT prompt path computes all query positions at every application.
Its fixed embedding-derived latent also permits last-query-only prompt execution;
that serving optimization is absent from the original measurements. Prompt latency
therefore reflects current execution choices as well as architectural work.

**LPT:** [Shared Memory in Looped Transformers, 2610.02383v1](https://arxiv.org/html/2610.02383v1).
The first recursion builds full context K/V independently for each physical layer.
Later recursions read that same layer's first-loop memory plus a local bank with
64 preceding tokens and the current token. **One softmax** normalizes their union;
two independently normalized attentions are not substituted. Training retains
all gradient paths to first-loop memory. To isolate the cache layout, the adapter
uses this project's unweighted RMSNorm, exact GELU, and absolute positions rather
than reproducing the paper's complete Ouro-derived blocks or training objective.
Shared and local decode banks stay physically separate on Tensor. Torch's supplied-
token control concatenates the two banks for Flash SDPA; that layout work is timed.

**GRT:** [Gated Recurrent Transformers, 2608.15062v4](https://arxiv.org/html/2608.15062v4),
with the authors' [reference code](https://github.com/Amr-Hegazy1/gated-recurrent-transformer).
The prelude anchor, recurrent 2W→W projection, separate gate normalizations,
2W→W→W SiLU gate MLP, +4 gate bias, and elementwise retention/proposal blend are
included. All Transformer blocks use learned LayerNorm and exact GELU. The gate
normalization over the concatenated features follows the released code. Training
includes resampled state noise and per-token broadcast gate noise with std 0.1;
Torch CUDA RNG is explicit common control work on both backends and is timed.
Inference and numerical qualification disable noise for deterministic cache/full-
forward equivalence; the released code's unconditional noisy forward is not used
as an inference reference. The split is 2+8T+2 at width 768, with separate Q/K/V
projections and independent token/output weights to match the local harness.
The profile keeps full loop caches. Averaging KV from different loops changes
cached execution relative to this finite-depth forward, so that transform is not
silently treated as exact cached equivalence.

**Per-layer latent loop:** an [MLA-inspired](https://arxiv.org/abs/2405.04434v5)
control, not DeepSeek-V2. Each physical block has its own rank-64 down projection;
C is refreshed from the evolving block input on every application. Head-specific
linear expansion weights are folded into queries/output exactly as in LLT. The
cache stores one rank-64 latent per applied block. The control omits DeepSeek's
query-compression branch, decoupled rotary-key stream, and MoE. It isolates
cross-layer/loop memory sharing from per-application latent storage.

**Attention-only loop:** an FFN-frequency control inspired by
[MixerLoop, 2608.18230v1](https://arxiv.org/html/2608.18230v1). Each physical block
repeats its standard causal attention update T times, then applies its FFN once.
It uses softmax attention, not the paper's Gated DeltaNet mixer. All cache/state
storage for those T attention applications is counted. It tests compute allocation
without attributing published DeltaNet speedups or quality to this implementation.

## Kernels, checkpointing, and cache qualification

The model is in [research_baselines.py](../model/research_baselines.py).
[Tensor](https://github.com/kreasof-ai/tensor) owns the native window/shared-bank,
SwiGLU/gating, projection, normalization, loss, optimizer and decode operations.
Torch uses compiled FlexAttention for window/union masks, Flash SDPA for ordinary
and cached attention, and its fused optimizer. Compilation is excluded from timing.
Tensor does not materialize a dense attention mask or score matrix.

Training profiles **none** and **AC**. AC checkpoints whole Transformer blocks;
attention-only loops checkpoint each attention update and the once-per-block FFN
separately. GRT's recurrent projection/gate sits outside block AC. There is no
new LAC policy in this extension. These boundaries and retained states matter
when interpreting memory. Inference has one configuration per model/T/backend.

Small checks at T=4/16 compare every parameter gradient across Torch/Tensor,
exact none/AC gradients within each backend, and supplied-token cache/full-forward
logits. Full-geometry T=16 controls repeat uncheckpointed backward before checking
AC, enabling deterministic algorithms only for correctness. GRT qualification
uses fixed, noise-free eval execution; training timings retain stochastic noise.
A separate seeded CPU check verifies noisy GRT training outputs, every parameter
gradient, and RNG consumption across none/AC.

Persistent states in this profile support one supplied token after a fixed prefix,
then logical rewind for repeated timing, matching the existing measured operation.
They are not a general generation service. Local caches allocate the retained
window plus its current-token slot; global caches allocate 1025 slots at S=1024.
No shared full-context history is replicated into every LPT local bank.

## Reproduce the added-family sweep

Use the Torch/Tensor environment from the main reproduction guide. Choose a fresh
output directory; the GPU runner executes cases sequentially in fresh processes.

```bash
LLT_RESULTS_ROOT=/path/to/fresh-results bash experiments/l40s/run_research_sweep.sh
LLT_RESULTS_ROOT=/path/to/fresh-results python -m experiments.l40s.research_summary
```

The grid has 480 primary records: five families, 16 loop counts, two backends,
and training none/AC plus inference. Ten small and ten full-size qualification
records are separate from performance measurements. `LLT_RESUME=1` resumes completed
cases; the auditor rejects mixed source or runtime revisions. Compilation and
warmup are excluded, matching the original timing harness.
