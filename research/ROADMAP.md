# LLT research roadmap

Date: 2026-10-08. Development proceeds after the required kernel and training
capabilities in [Tensor](https://github.com/kreasof-ai/tensor) pass the
[Tensor readiness gate](https://github.com/kreasof-ai/tensor/blob/main/docs/plan/llt-readiness.md).
Tensor development plans, defect reports, and acceptance evidence belong in that
repository. This roadmap tracks LLT architecture, quality, and experiments.

The current [L40S study](../benchmarks/L40S.md) establishes synthetic forward
kernel and memory results. Trained quality and constant total training memory
remain unvalidated.

## LLT requirements after Tensor readiness

### L01 — Freeze the model and prove the implemented algebra

- [ ] Specify global versus layerwise caches, tied block layout, latent ranks,
  head geometry, projection sharing, normalization, loop conditioning, and
  position encoding. Specify whether the YOCO/U-YOCO self-decoder is actually
  implemented; the current simplified composition does not validate all variants.
- [ ] Verify causal full-sequence, prefill, and cached-decode equivalence with
  trained weights. Separate cache/parameter complexity from compute and total
  training-memory complexity.
- [ ] For iterative reasoning, compare a static initial-input cache with an
  uncompressed static-cache control and a mutable compressed workspace. Other
  positions currently cannot read newly inferred states through the fixed cache.

### L02 — Resolve the activation-checkpointing claim

- [ ] Establish what state is necessary to recover residual, query, and MLP
  activations. A KV latent alone does not establish exact reconstruction.
- [ ] Keep ordinary exact loop checkpointing as the validated baseline. Explore
  reversible blocks, residual-state storage, or extra recomputation if needed.
- [ ] If using lossy activation compression, quantify reconstruction error,
  gradient bias, convergence, quality, peak memory, and recomputation time.
- [ ] Test matched checkpoint policies. Do not attribute ordinary checkpoint
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

- [ ] Benchmark trained-model prefill, first-token latency, multi-token decode,
  throughput, and latency distributions; include folding/setup and cache updates.
- [ ] Sweep batch, context, rank, layers, width, vocabulary, and loop count. Test
  positional correctness before interpreting long-context quality.
- [ ] Account for residual checkpoints, gradients, optimizer/classifier floors,
  workspaces, and device overhead. Plot measured scaling rather than cache size alone.
- [ ] Re-evaluate the current targets: at least 50% less peak memory versus naive,
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

