# Local Vulkan and CPU measurements

Measured on **AMD Radeon RX 6700 XT**, Windows, Vulkan, using the adjacent
[Tensor checkout](https://github.com/kreasof-ai/tensor) at commit
`fa006eda9f01b5b2539ad1ca7eda03bc41f253c4`. Measurements were collected on
October 6, 2026. No changes to the Tensor checkout are needed.

The original run below is a synthetic **single-token, single-layer attention microbenchmark**,
with batch 1, 12 heads and head dimension 64. It does not implement a trained
LLT, a complete transformer stack, MLPs, RoPE, backward, or distributed training.
Its results do not establish H100 throughput or whole-device training memory.

The follow-up measurements also cover an evolving complete decoder forward,
isolated live tensor allocations, and CPU gradient replay. These remain synthetic
prototypes; they do not test trained-model quality or the proposed training system.

A later [regime search](REGIMES.md) adds folded query/output projections, a
per-head LLA geometry comparison, and exact CPU training checkpoints. It finds
lower-rank points that meet explicit memory and latency criteria. Its results
use a different, optimized implementation from the unfused decoder below.

## Measured latency

Context 4,096; ten repeated attention calls. These are medians of **GPU execution
time in milliseconds**, including the full ordered kernel sequence for each
method. All methods use the same C, projection weights and query for a given
rank. Queries stay fixed across repetitions; these repetitions are not ten
iterations of an evolving recurrent model.

| Method | Rank 64 | Rank 128 (proposal) |
|---|---:|---:|
| Cached explicit K/V | 2.527 | 2.529 |
| Expand full historical K/V before every call | 6.205 | 9.868 |
| Expand once, reuse across ten calls | 2.891 | 3.249 |
| Absorbed latent attention, including query/output transforms | 2.547 | 7.587 |

For rank 128, re-expansion takes **3.90×** the cached-attention time; absorption
takes **3.00×**. Expanding once reduces that overhead to **1.28×**, while retaining
the full expanded K/V scratch buffer across the repeated calls. Reuse is valid
only with fixed C and loop-tied up-projections. These ratios describe this
implementation and workload, not universal architectural bounds.

An independent repeat of the 4,096-context, ten-call cases agreed within 1.1%
for every method. Rank-128 repeat timings were 2.551 / 9.893 / 3.258 / 7.604 ms
in the table's method order.

At rank 64, absorbed attention is approximately equal in latency to cached K/V
for this long-context case. Rank 64 and 128 use independently generated synthetic
models, so this comparison says nothing about trained-model quality at either rank.

The portable SIMT kernels partition context across 16 workgroups per head and
merge their online-softmax statistics without creating a global score matrix.
An independently checked [sequential control](results/rx6700xt-sequential/report.json)
uses one workgroup per head. At rank 128 and ten calls it takes 33.124 ms for
cached K/V and 68.025 ms for absorbed attention. This large schedule effect is
another reason not to extrapolate these absolute timings to optimized vendor
kernels or H100s.

## Cache and scratch sizes

Sizes below are the native tensor buffers' logical `nbytes`, not driver-reported
whole-process VRAM peaks. The harness keeps buffers for all methods resident so
that allocation and transfer do not contaminate timing.

| Buffer at context 4,096 | Rank 64 | Rank 128 |
|---|---:|---:|
| One explicit FP16 K/V tensor, all 12 heads | 12 MiB | 12 MiB |
| Shared FP16 latent cache | 0.5 MiB | 1 MiB |
| Full expansion scratch, when used | 12 MiB | 12 MiB |
| Absorbed query and accumulator, FP32 | 6 KiB | 12 KiB |
| Explicit attention partition scratch | 49.5 KiB | 49.5 KiB |
| Absorbed attention partition scratch | 49.5 KiB | 97.5 KiB |
| Tied K/V up-projection weights, FP16 | 192 KiB | 384 KiB |

The proposed rank 128 therefore compresses one explicit cache by **12×**, but
unfused expansion still needs a **12 MiB temporary buffer**. Absorption removes
that buffer at the measured latency cost.

A naive 12-layer, ten-loop stack retaining separate FP16 caches would require
`12 MiB × 12 × 10 = 1,440 MiB` for this context, versus a globally shared
rank-128 cache of 1 MiB, excluding positional keys. The later
[4K decoder run](#complete-decoder-and-allocation-protocol) allocated these
cache sizes. This original microbenchmark allocates only one representative
explicit cache; the **1,440×** ratio describes raw cache storage, not whole-model
peak memory.

## Protocol and correctness

- Four contexts: 128, 512, 2,048 and 4,096; two ranks: 64 and 128; three repetition
  counts: 1, 4 and 10. **24 configurations, 96 timed method sequences.**
- FP16 latent, weights and expanded K/V; FP32 query, accumulation and outputs.
- Independent float64 NumPy oracle. Explicit attention has an FP16 expansion
  rounding boundary; absorbed attention does not. Each path is checked against
  its own oracle, then the paths are compared with a rounding tolerance.
- **20 correctness replays per path and shape** passed. Largest explicit-oracle
  error: `2.97e-4`; absorbed-oracle error: `1.85e-7`; cross-path difference: `2.37e-4`.
- Explicit oracle gate: `atol=2e-4, rtol=.002`; absorbed uses the same gate.
  Cross-path gate: `atol=.002, rtol=.02`. Partition merging is additionally checked
  against float64 combination of the GPU partials.
- A shared softmax normalizer has a single writer, with a workgroup barrier
  between tiles. An earlier experimental schedule with a race was rejected;
  the retained reports were produced after this fix.
- Main sweep: 0.4-second device warmup per shape, three warmups per timing path,
  nine samples of twenty repetitions. GPU timestamp period **10 ns**, queried
  from `vulkaninfo`, not assumed. Whole-plan GPU timing excludes CPU encoding.
- JSON also records host submission plus completion timing. Compilation,
  pipeline creation, allocation, uploads, downloads and CPU reference are
  outside both timed regions. Pipeline creation is recorded separately.
- Method order rotates across repetition cases. Short workloads show clock and
  fixed-overhead sensitivity; use the long-context results for the comparisons
  above. No locked GPU-clock or background-workload control is claimed.

## Reproduce

Use the existing Tensor Python 3.12 environment with NumPy, TileLang and wgpu:

```powershell
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' `
  benchmarks\vulkan_attention.py --samples 9 --repeats 20 `
  --out benchmarks\results\rx6700xt-partitioned
```

Run a sequential scheduling control:

```powershell
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' `
  benchmarks\vulkan_attention.py --partitions 1 --contexts 4096 `
  --ranks 64 128 --loops 1 10 --samples 5 --repeats 3 `
  --out benchmarks\results\rx6700xt-sequential
```

Representative independent repeat:

```powershell
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' `
  benchmarks\vulkan_attention.py --contexts 4096 --ranks 64 128 --loops 10 `
  --samples 9 --repeats 20 --out benchmarks\results\rx6700xt-recheck
```

Use `--tensor-root` and `--device` for another checkout/adapter. The harness
requires a physical discrete Vulkan GPU and a single Vulkan device for timestamp
calibration. It generates portable `.tbin` kernels in the result directory and
executes them with Tensor's prepared-plan runtime. Runtime buffer binding and
timestamps use existing Tensor/wgpu interfaces; no NVIDIA dependencies are used.

The [full report](results/rx6700xt-partitioned/report.json),
[sequential control](results/rx6700xt-sequential/report.json), and
[repeat report](results/rx6700xt-recheck/report.json) retain samples, adapter
identity, versions, buffer sizes, correctness errors and kernel/source hashes.
Generated artifacts, build caches and exact per-run source snapshots are retained
locally but excluded from Git. A rerun overwrites its report and source snapshot;
use another `--out` directory to preserve earlier measurements.

## Implications for LLT

Keep compressed cache storage separate from whole-model peak memory and latency
claims. With these kernels, expanding all historical K/V on every loop is costly;
tied projections and reuse help. Rank 128 absorption also costs more than cached
attention, while rank 64 is worth investigating if model quality survives.
The follow-up below measures tile-fused attention and CPU checkpoint replay.
Distributed communication, production GPU training and quality preservation
remain unmeasured.

## Nine follow-up suites

All nine locally feasible suites have been exercised:

| Suite | Coverage |
|---|---|
| Latent rank | 32, 64, 96, 128 at context 4,096 |
| Longer context | 8,192 and 16,384, ranks 64/128 |
| Deeper repetition | 16, 32, 64 calls, ranks 64/128 |
| Batch size | 2, 4, 8 at context 1,024 |
| Tile-fused expansion/attention | 16-token workgroup tile, no global expanded KV |
| Precision | FP16/FP32 weights and cache, FP32 accumulation |
| Complete decoder forward | Width 768, 12 layers, 1/4/10 loops, ranks 64/128 |
| Peak live tensor allocations | Each decoder method allocated separately, weights/cache/workspace included |
| CPU LAC gradients | Full/latent state, 1/4/16/32 loops, exact replay and lossy reconstruction |

The [extended attention report](results/extended/report.json) retains 24 cases
across five methods. Ten correctness replays per path/shape compare against
independent float64 references. Timing uses five GPU timestamp samples, each
containing three sequence repetitions, plus separately recorded host timings.

At context 4,096 and ten calls:

| Rank | Cached KV (ms) | Absorbed (ms) | Tile-fused (ms) |
|---|---:|---:|---:|
| 32 | 2.542 | 1.584 | 22.377 |
| 64 | 2.500 | 2.533 | 34.806 |
| 96 | 2.515 | 6.273 | 48.766 |
| 128 | 2.501 | 7.525 | 61.288 |

Rank 32 absorption reduces this attention time by 37.7% (1.60× throughput for
this fixed workload). Rank 128 absorption costs 3.01× cached attention; fused
expansion costs 24.50×. The fused implementation is a portable generic SIMT
GEMM, not an optimized hardware matrix kernel. These negative results measure
this implementation, not a lower bound on fused attention performance.

At context 16,384, rank 64 absorption takes 9.280 ms versus 9.897 ms cached
(6.2% less); rank 128 absorption takes 60.348 versus 9.898 ms (6.10× slower).
At batch 8/context 1,024, rank 64 absorption takes 3.782 versus 3.990 ms (5.2%
less). These small gains have not received independent reruns. Depth 16→64
increases latency approximately linearly while the latent/cache buffer ledger
stays fixed: constant cache memory does not eliminate loop computation.

FP32 cached attention happens to be faster here: 1.644/1.649 ms for ranks
64/128 versus 2.502/2.498 ms with FP16. FP16 still halves cache storage. Storage
precision, scalar conversion costs and kernel scheduling must be measured
together; a smaller cache does not by itself guarantee faster execution.

## Complete decoder and allocation protocol

[decoder_forward.py](decoder_forward.py) runs one input embedding through
RMSNorm, attention, residual additions, a 4× GELU MLP in each layer/loop, and
final RMSNorm plus 256 vocabulary logits. Queries and residual state evolve;
weights are tied across loops. The context has 512 positions and includes the
current token, whose cache entry is written by the GPU. Synthetic prefix caches
are supplied directly. This is one decoder step, not autoregressive generation
or trained-model prefill. RoPE is not implemented.

The primary baseline, `naive_cache`, evaluates the same latent-cache model with
distinct expanded KV buffers for every layer/loop. `reuse_cache` keeps only one
expanded cache per layer, valid because this model ties up-projections across
loops and shares the token latent. `absorbed` avoids expanded KV; `fused` expands
tiles in workgroup memory. An `mha` control projects the current KV from the
evolving hidden state and therefore evaluates different model math. Its latency
cannot establish equal-quality LLT speedup.

Each method allocates its own buffers, runs five correctness replays against a
CPU float64 forward reference, warms up, records five GPU samples with three
repetitions each, and releases its buffers before the next method. The
[decoder report](results/decoder/report.json) contains actual allocated buffer
counts and `nbytes` by category. All tensor buffers are simultaneously live,
so their sum is the peak live **tensor allocation** for this harness. It excludes
driver, pipeline, allocator overhead and on-chip workgroup storage; it is not a
driver-measured whole-process VRAM peak. Timings exclude setup, transfers,
compilation, readback, token lookup and sampling. Method order is fixed and GPU
clocks are not locked, so small timing differences need repeated confirmation.

At context 512 and ten loops, the final decoder medians and live allocations are:

| Method | Rank 64 time (ms) | Rank 64 MiB | Rank 128 time (ms) | Rank 128 MiB |
|---|---:|---:|---:|---:|
| Matched naive cache | 28.657 | 317.824 | 28.781 | 320.169 |
| Reuse expanded cache per layer | 28.602 | 155.824 | 28.519 | 158.169 |
| Absorbed shared latent | 32.877 | 137.935 | 46.492 | 140.395 |
| Tile-fused shared latent | 90.758 | 137.881 | 139.287 | 140.288 |

Absorption reduces total live allocations versus naive by 56.6% at rank 64 and
56.1% at rank 128 (2.30×/2.28× smaller), but latency rises by 14.7%/61.5%.
Against the explicit-cache reuse baseline, absorption saves only 11.5%/11.2%
of total allocations, while latency rises by 14.9%/63.0%. Reuse alone removes
51.0%/50.6% of naive allocations. This distinguishes latent compression from
eliminating redundant caches under the shared-latent, tied-projection model.

The MHA control takes 36.139/29.450 ms and 342.481 MiB at ranks 64/128.
It is a different forward graph, has synthetic history, and receives no quality
test; these numbers do not establish an LLT improvement over a trained naive
looped transformer. The shorter-context decoder and attention-only 4K timings
are different workloads and must not be compared as whole-model speedups.

An additional [4K decoder run](results/decoder-4k/report.json) uses the proposal's
width 768, 12 layers, rank 128 and ten loops, with three samples and one sequence
per sample. Every method passes five CPU-reference correctness replays:

| Method | GPU time (ms) | Peak live tensor MiB |
|---|---:|---:|
| Matched naive cache | 53.088 | 1,580.169 |
| Reuse expanded cache per layer | 53.330 | 284.169 |
| MHA control (different model math) | 54.524 | 1,602.481 |
| Absorbed shared latent | 134.966 | 141.270 |
| Tile-fused shared latent | 812.852 | 141.163 |

At this context, absorbed latency is 2.54× naive, while total live tensor
allocation is 91.1% lower (11.2× smaller). Against cache reuse, the allocation
reduction is 50.3% and latency is 2.53×. The naive/cache-reuse/shared-latent
history buffers are exactly 1,440/144/1 MiB respectively; model weights and
workspace explain why total savings are smaller than the raw 1,440× cache ratio.
Fused latency is 15.31× naive. Larger context strengthens memory savings but
does not establish a decoder speedup in these kernels.

## CPU checkpoint findings

The [CPU report](results/lac-cpu/report.json) measures a causal-attention,
RMSNorm and GELU MLP toy model with width 64, rank 16 and sequence length 16,
using float64 PyTorch on four CPU threads. Both full-state replay and a model
whose forward state is actually latent match ordinary autograd gradients at
every tested depth. A separate directional finite difference agrees to
6.04×10⁻¹³ absolute error.

At 32 loops, full-state replay reduces peak unique autograd-saved storage from
4,284,416 to 141,824 bytes (96.7%, 30.2× smaller), while forward plus backward
increases from 28.811 to 165.167 ms (5.73×). Latent-state replay reduces
4,605,952 to 145,920 bytes (96.8%, 31.6× smaller), while time increases from
34.278 to 221.233 ms (6.45×). These are saved-tensor measurements, excluding
parameters and other CPU workspace, not total training memory.

Exact replay reconstructs each backward step by rerunning its forward prefix:
32 loops require 496 extra prefix steps. Saving a 4× smaller initial latent
checkpoint (2,048 versus 8,192 bytes) does not reduce the entire live backward
tape 4×; reconstructed full-width operations remain necessary.

The lossy checkpoint control retains the original full-state forward and
reconstructs it through a rank-16 codec during backward. Its input gradient has
7.5% relative L2 error; parameter-gradient errors range from 96.4% to 154.8% in
this synthetic fixture. This confirms the expected bias, not a trained codec's
quality limit. Storing only KV latent does not reconstruct evolving query,
residual and MLP states. The later [exact loop checkpoint path](REGIMES.md)
retains full residual boundary states; KV-only residual reconstruction remains
an unresolved research target.

## Reproduce the follow-up

```powershell
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\extended_attention.py
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\decoder_forward.py
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\decoder_forward.py --context 4096 --ranks 128 --loops 10 --samples 3 --repeats 1 --out benchmarks\results\decoder-4k
& 'D:\Workspace\kreasof-ai\github\tensor\.venv\Scripts\python.exe' benchmarks\lac_cpu.py
```

Run GPU commands sequentially. The two GPU scripts support `--smoke` and
`--out`; decoder options include `--context`, `--ranks` and `--loops`. Reports
record source and kernel hashes. No H100/NVLink throughput, distributed
collectives, GPU backward, loss curves or benchmark quality results have been
verified by these local suites.
