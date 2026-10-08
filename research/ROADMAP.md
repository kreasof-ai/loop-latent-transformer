# LLT research roadmap

Date: 2026-10-08. The required kernel and training capabilities in
[Tensor](https://github.com/kreasof-ai/tensor) passed the
[Tensor readiness gate](https://github.com/kreasof-ai/tensor/blob/main/docs/research/llt-readiness.md)
at implementation revision `ab17948fe94dd7c7b857c8f067c465e901329b94`.
Tensor development plans, defect reports, and acceptance evidence belong in that
repository. This roadmap tracks LLT architecture, quality, and experiments.

The original [L40S study](../benchmarks/L40S.md) establishes synthetic forward
kernel and memory results. The [full-model Tensor study](../benchmarks/L40S_NATIVE.md)
adds numerical model/backward/optimizer kernels and real persistent generation,
with 152 training methods, 84 inference methods and 706 audited CUDA artifacts.
Trained quality and constant total training memory remain unvalidated.

The subsequent [nanoGPT-scale report](../benchmarks/L40S_NANOGPT.md) adds the
optimized Tensor implementation (`385124e` projections, `9d03a69` serving cache),
actual pinned nanoGPT, 66 performance methods, full-size gradient checks,
serving startup, explicit prepared weights, fused Torch AdamW controls, and
complete nanoGPT training CUDA graphs. This resolves the requested kernel-first
and nanoGPT-scale profiling phase; trained quality and distributed evidence
remain separate research work.

## LLT requirements after Tensor readiness

### L01 — Freeze the model and prove the implemented algebra

- [x] Specify global versus layerwise caches, tied block layout, latent ranks,
  head geometry, projection sharing, normalization, loop conditioning, and
  position encoding. Specify whether the YOCO/U-YOCO self-decoder is actually
  implemented; the current simplified composition does not validate all variants.
  The [frozen model](MODEL.md) specifies the supported composition and limitations.
- [x] Verify causal full-sequence, prefill, and cached-decode equivalence with
  trained weights. Separate cache/parameter complexity from compute and total
  training-memory complexity.
  Verified on short trained-weight fixtures (32 updates), with larger random-weight
  resource checks; this does not establish trained quality or long-context quality.
- [ ] For iterative reasoning, compare a static initial-input cache with an
  uncompressed static-cache control and a mutable compressed workspace. Other
  positions currently cannot read newly inferred states through the fixed cache.

### L02 — Resolve the activation-checkpointing claim

- [ ] Establish what state is necessary to recover residual, query, and MLP
  activations. A KV latent alone does not establish exact reconstruction.
- [x] Keep ordinary exact loop checkpointing as the validated baseline.
  Independent all-parameter gradient comparisons pass in FP64 and BF16.
- [ ] Explore
  reversible blocks, residual-state storage, or extra recomputation if needed.
- [ ] If using lossy activation compression, quantify reconstruction error,
  gradient bias, convergence, quality, peak memory, and recomputation time.
- [x] Test matched checkpoint policies. Do not attribute ordinary checkpoint
  savings to latent compression or claim constant total training memory from
  a cache whose size is independent of loop count.

### L03 — Train real models and evaluate quality

- [ ] Fix tokenizer, datasets, licenses, train/validation/test splits, seeds,
  parameter budgets, and training-compute budgets; prevent evaluation leakage.
- [ ] Train stable models beyond smoke tests, with multiple seeds and checkpoint
  selection based only on validation data.
- [ ] Compare non-loop, tied naive-loop, global LLT, and layerwise LLT controls;
  sweep rank and loops. Add MLA/YOCO and codec controls where their implementations
  support a meaningful comparison. Match latent budgets when comparing codecs.
- [ ] Measure held-out LM loss/perplexity plus ARC-AGI exact grid match, Sudoku
  whole-puzzle validity, and other chosen iterative tasks. Report task difficulty,
  attempts, loop count, and out-of-distribution behavior.
- [ ] Compare memory and time at matched quality, as well as at matched model
  geometry. A smaller cache can sacrifice useful information.

### L04 — Re-run complete systems and memory studies

- [x] Profile LLT, naive looping, independent stacks, and an exact stack-parameter
  matched fixed-depth control at B4/S1024 for every loop count 1–16. Retain
  Tensor/PyTorch eager and graph latency/peak allocation, OOM stages, and
  controlled gradient-release capture retries. See [the sweep report](../benchmarks/L40S_LOOP_SWEEP.md).

- [ ] Benchmark trained-model prefill, first-token latency, multi-token decode,
  throughput, and latency distributions; include folding/setup and cache updates.
- [x] Sweep batch, context, rank, layers, width, vocabulary, and loop count. Test
  positional correctness before interpreting long-context quality.
- [x] Account for residual checkpoints, gradients, optimizer/classifier floors,
  workspaces, and device overhead. Plot measured scaling rather than cache size alone.
  The allocator accounting and exclusions are explicit; driver/context storage
  remains outside measured peaks, so these are not whole-process VRAM results.
- [x] Re-evaluate the current targets: at least 50% less peak memory versus naive,
  at most 20% extra median latency, and at most 25% more memory than non-loop LLT.
  State the baseline checkpoint policy and execution mode for every result.
- [ ] Measure whole-grid reasoning solvers separately: they may discard per-loop
  inference states and need not retain autoregressive per-loop KV histories.

### L05 — Distributed evidence, when hardware is available

- [ ] Validate gradient reductions for shared/tied state, sharding, distributed
  checkpoint/resume, and actual communicated bytes per training step.
- [ ] Include residuals, parameter gradients, collectives, synchronization, and
  repeated loop computation in communication claims.
- [ ] Treat H100/NVLink or multi-GPU results as a separate hardware gate. One
  L40S cannot validate these claims; single-GPU completion may defer this scope
  with the limitation stated explicitly.

### L06 — Reproducible completion and claims

- [ ] Publish the selected architecture/configurations, dataset procedure,
  trained checkpoints where permitted, Tensor pin, and one-command evaluations.
- [ ] Retain raw correctness, quality, timing, memory, and provenance reports.
- [ ] Rewrite project claims around measured behavior and supported modes.
  Separate demonstrated constant KV-cache size from open total-memory and
  communication claims, and report unsuccessful regimes.
