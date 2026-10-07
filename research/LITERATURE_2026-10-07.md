# Literature check for Loop-Latent Transformer

Reviewed **2026-10-07**. Covers all **23 supplied papers**, with version-pinned primary-source links below. Titles/abstracts were checked for every paper; relevant architecture, training and evaluation sections were inspected in arXiv HTML. This is a targeted literature review, not a reproduction, exhaustive novelty search, code audit, or independent verification of the authors' measurements. Three additional puzzle-baseline abstracts are identified separately.

**Conclusion:** LLT still has a useful research question, but sharing memory across loops is established prior work. Its candidate contribution is the quality–memory–latency tradeoff of a smaller memory shared across both layers and loops, with folded projections. Training-memory claims must specify the gradient objective. For ARC-AGI and Sudoku, the most urgent question is whether the current immutable input cache leaves enough communication between evolving solver states.

Paper numbers below are **author-reported**, under each paper's own protocol. LLT's local measurements remain in [the regime report](../benchmarks/REGIMES.md); no paper result replaces those measurements or establishes LLT's reasoning quality.

## 1. What kind of recurrence is being compared?

| Family | Computation being repeated | Relevance to LLT |
|---|---|---|
| Depth recurrence | Shared blocks revisit the current token or sequence state | Closest architectural and KV-sharing comparisons |
| Temporal recurrence | Hidden computation carries into the next token/environment step | Alternative compute allocation; a different recurrence axis |
| Structured recursive solver | A whole puzzle's hidden state or candidate solution evolves | Direct ARC-AGI/Sudoku quality baselines |
| Latent-thought generator | Continuous thought positions/trajectories condition a decoder | Includes decoder, latent positions and sampling cost |
| Serving/scheduling method | Drafting, verification, step control or early exit | Potential latency complement after a model is trained |
| Expert reuse | Expert weights are shared between layers or loops | Parameter efficiency does not establish cache or activation savings |

Here, a **KV latent** is compressed attention storage; a **solver latent** is evolving computation. They need not be the same tensor. The two papers called LRT below also describe different models.

## 2. Catalogue of all 23 supplied papers

Priority: **A** = immediate baseline or design constraint; **B** = useful follow-up ablation; **C** = adjacent application or architecture. The priority reflects LLT's current ARC-AGI/Sudoku and resource goals, rather than paper quality.

| # | Paper / inspected version | Mechanism and evidence relevant to LLT | Priority / implication |
|---:|---|---|---|
| 1 | [The Surprising Effectiveness of Shared Memory in Looped Transformers — 2610.02383v1](https://arxiv.org/html/2610.02383v1) | LPT reuses first-recursion context KV **per physical layer**, retaining later-loop local windows. Jointly trained language models. | **A:** closest first-loop cache baseline; sharing across loops alone is insufficient as a novelty claim. |
| 2 | [Towards Looped Models Done Right, Part II: Rethinking at Fixed Points — 2610.06833v1](https://arxiv.org/html/2610.06833v1) | Terminal KV sharing, learned depth priors and truncated backpropagation; equilibrium-gradient justification requires fixed-point/contractivity assumptions. | **A:** compare terminal sharing and gradient horizons; this is not exact full finite-depth BPTT. |
| 3 | [Scheduling Recursive Reasoning in Looped Transformers — 2609.36653v1](https://arxiv.org/html/2609.36653v1) | TAPS adjusts updates from persistent versus fluctuating progress; reports up to **1.56×** speed at baseline accuracy across tested reasoners. | **A:** scheduler ablation; include its EMA state and control overhead in memory/timing. |
| 4 | [DeepLoop: Depth Scaling for Looped Transformers — 2607.13491v2](https://arxiv.org/html/2607.13491v2) | Depth-aware residual/initialization scaling. HRM ARC-AGI-1 two-vote accuracy **36.50→39.75%** under its augmentation protocol. | **A:** stability control before attributing quality loss to compression; normalization assumptions matter. |
| 5 | [Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts — 2610.01153v1](https://arxiv.org/html/2610.01153v1) | LOOM shares attention/experts but uses loop-specific routers, residual scaling and input reinjection. Near-iso-FLOP 700M experiment favors five loops. | **B:** stability ideas; router parameters still grow with loops. MoE is a separate architectural variable. |
| 6 | [MoRE: Mixture of Reused Experts — 2609.18176v2](https://arxiv.org/html/2609.18176v2) | Adjacent layers access a shared enlarged expert pool; layer-specific attention/routing remain. Matched-budget experiments preserve total unique expert count. | **C:** expert reuse is not one shared attention cache or automatically fewer total parameters. |
| 7 | [Looped Transformers as Optimizers — 2609.37379v1](https://arxiv.org/html/2609.37379v1) | OperLoop aligns read/write maps with an optimizer-like state update. Experiments use four residual streams and unchanged effective KV layouts. | **B:** compression changes state geometry; this framework does not itself compress KV. Count expanded states. |
| 8 | [Temporal Recurrence Favors Fewer Layers — 2609.12531v1](https://arxiv.org/html/2609.12531v1) | Shared state carries across external steps; Sokoban/FineWeb sweeps favor fewer sequential layers at approximately matched per-step work. | **B:** sweep physical depth and width; temporal results do not establish the optimum for depth-loop LLT. |
| 9 | [LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers — 2609.17184v1](https://arxiv.org/html/2609.17184v1) | Early loops draft tokens and later loops verify; reports up to **6.83×** decoding speed. Correctness preserves the target decoder's output distribution. | **B:** optional autoregressive serving baseline; draft branches have cache costs and do not solve training activation storage. |
| 10 | [Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs — 2609.01117v1](https://arxiv.org/html/2609.01117v1) | Task encoder proposes soft tokens; a TRM-like refiner updates scratch/integrator states with truncated gradients. **11.2M trainable parameters plus frozen Qwen3-8B**; includes Sudoku. | **B:** fixed conditioning plus mutable reasoning state. Frozen decoder weights and differentiated activations still cost memory. |
| 11 | [Trading Depth for Time in Recurrent Transformers — 2609.21605v1](https://arxiv.org/html/2609.21605v1) | Temporal LRT inserts thought positions; one thought nearly matches double physical depth with about **48% fewer parameters**. Every thought writes KV. | **B:** equal block executions are not equal attention FLOPs, cache storage or latency. |
| 12 | [Thinking with Looped Flows — 2609.11801v1](https://arxiv.org/html/2609.11801v1) | Persistent solver states learned with local denoising objectives across decreasing noise levels. Direct Sudoku/Maze/ARC-AGI evaluations; no full-trajectory BPTT. | **A:** essential quality and alternative-training baseline; its latent is evolving solver state. |
| 13 | [The Recurrent Transformer: Greater Effective Depth and Efficient Decoding — 2604.21215v1](https://arxiv.org/html/2604.21215v1) | Each layer publishes KV from its **output** to later token positions. Exact tiled training/prefill reduces asymptotic HBM traffic from quadratic to N log N. | **B:** temporal recurrence with per-layer memory; traffic reduction is not the same as storage or arithmetic reduction. |
| 14 | [T²MLR: Transformer with Temporal Middle-Layer Recurrence — 2607.15178v2](https://arxiv.org/html/2607.15178v2) | Prior-token middle-layer state feeds an earlier current layer. Default training approximates recurrence with **16 forward / 4 backward** refinement steps; localized recurrence helps. | **B:** tests where to allocate recurrence; low inference overhead comes with extra approximate training computation. |
| 15 | [Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation — 2608.15062v4](https://arxiv.org/html/2608.15062v4) | Fixed prelude conditions evolving gated state. Tests first/last/averaged loop KV. Large iso-FLOP model reports **59% less peak decode memory with 10% compiled latency overhead** versus dense. | **A:** cache-sharing and gating baseline; reported savings include fewer weights, not just KV compression. |
| 16 | [Hyperloop Transformers — 2604.21254v3](https://arxiv.org/html/2604.21254v3) | Middle-layer looping plus hyperconnections and **four parallel residual streams**; approximately half the parameters of depth-matched untied models. | **B:** quality-oriented comparison; expanded residual/checkpoint state can oppose the memory goal. |
| 17 | [HRM-Text: Efficient Pretraining Beyond Scaling — 2605.20613v1](https://arxiv.org/html/2605.20613v1) | Slow/fast modules, MagicNorm, backward-horizon warmup **2→5**, instruction-response training and PrefixLM masking. Its ARC-C result is not ARC-AGI. | **B:** training-objective/data changes must be isolated from architecture; do not import its estimated compute ratios into LLT. |
| 18 | [Allocating Recurrent Compute in Looped Language Models — 2608.18230v1](https://arxiv.org/html/2608.18230v1) | MixerLoop repeats a Gated DeltaNet mixer while running FFN once. **45.9% backbone projection-FLOP saving**; batch-one prefill speed **1.11–1.12×** in measured models. | **A:** FFN-once ablation; FLOPs are not latency. At 110M, it retains only **41.5%** of FullLoop's CORE improvement. |
| 19 | [Full-bandwidth Transformer — 2608.08888v2](https://arxiv.org/html/2608.08888v2) | Previous top-layer state feeds the next token via gated fusion. Scheduled multipass training; retains standard KV histories and language-model objective. | **B:** low-overhead temporal feedback, not overwriteable shared KV or loop-axis compression. |
| 20 | [Generative Recursive Reasoning — 2605.19376v2](https://arxiv.org/html/2605.19376v2) | GRAM evolves stochastic high/low solver states and samples multiple paths. Gradients traverse only the final transition per supervision step, a biased ELBO surrogate. | **A:** direct puzzle baseline; count sampling, augmentation, selection and supervision budgets. |
| 21 | [Latent Thought Flow: Efficient Latent Reasoning in Large Language Models — 2606.16222v1](https://arxiv.org/html/2606.16222v1) | Continuous GFlowNet allocates variable-length latent reasoning by answer quality and computation cost; tested on math/data tasks with LoRA finetuning. | **B:** adaptive reasoning objective; fewer reasoning steps alone do not verify peak-memory or wall-clock improvements. |
| 22 | [LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation — 2602.11451v1](https://arxiv.org/html/2602.11451v1) | Time/step-conditioned blocks train long and short trajectories, using a stop-gradient consistency teacher. Short inference routes are explicitly trained. | **A:** fewer-loop quality baseline; include both training routes. Teacher detachment does not make the long route constant-memory. |
| 23 | [Looped World Models — 2606.18208v1](https://arxiv.org/html/2606.18208v1) | Inner latent dynamics loops plus state propagation across environment steps; stochastic depth, truncated backward horizon and learned exit. | **C:** adjacent world-simulation application; its parameter-efficiency claim is not an ARC/Sudoku memory–latency result. |

## 3. Closest prior work and the remaining contribution

### Shared KV is already a baseline

LPT cache positions scale as `L × [N + (T−1)w]`, versus naive `L × T × N`, omitting common feature factors: approximately **4.7×** reduction at five loops, N=4,096, w=64. Its random-weight H100 timing shows **8–12%** batch-one latency overhead; peak-throughput gains use larger batches. See the [architecture and inference appendix](https://arxiv.org/html/2610.02383v1).

The fixed-point paper retains terminal KV per physical block, not one low-rank cache across all layers. At five recursions its illustrated prelude/core/coda layout changes **12 logical cache banks to four**. Merely enabling sharing on its full-BPTT teacher damages GSM8K substantially; learned depth priors improve robustness. This is evidence that training for reuse matters, rather than evidence that arbitrarily discarding loop caches preserves quality. See [KV sharing and experiments](https://arxiv.org/html/2610.06833v1).

GRT's averaged-loop cache is another inexpensive baseline. In its medium checkpoint at batch 32, weights-plus-KV storage is **3.32 GiB with full loop caches versus 1.44 GiB averaged**; HellaSwag is **33.65 versus 33.90%**. That is a narrow empirical result, not a guarantee across tasks. Its dense comparison includes a different parameter budget. See [GRT cache ablation and Tables 3–4](https://arxiv.org/html/2608.15062v4).

Across the supplied set, these inspected designs do not establish LLT's exact proposed global, low-rank, cross-layer-and-loop cache plus folded maps. That leaves a **candidate combination to evaluate**, not a proven literature-wide first. MLA/YOCO/U-YOCO/LLA and the other inherited README references require their own broader novelty audit.

### Exact checkpointing and short-gradient training solve different problems

| Training regime | What is differentiated? | What a memory comparison must say |
|---|---|---|
| Full finite-depth BPTT | Every transition on the selected forward trajectory | Full gradient reference; tape usually grows with depth |
| Exact checkpointing | The same forward trajectory and gradient, with recomputation | Include retained boundaries and replay latency |
| Truncated BPTT | Only a chosen suffix; earlier states are detached | Report backward horizon; do not claim full-gradient equivalence |
| Equilibrium-gradient approximation | A fixed-point sensitivity or finite approximation to it | Report convergence/contractivity assumptions and approximation |
| Local denoising or surrogate objective | A separately specified training objective | Compare achieved quality and total training cost, not gradient identity |

The differences are explicit in [fixed-point training](https://arxiv.org/html/2610.06833v1), [GRAM §2.2](https://arxiv.org/html/2605.19376v2), [frozen-LRT §3.3](https://arxiv.org/html/2609.01117v1), and [HRM-Text §2.1](https://arxiv.org/html/2605.20613v1). They make low training memory plausible without exact full-history credit assignment. None establishes that LLT can reconstruct all full-width residual boundaries from its input KV latent.

The local LLT result is **exact loop checkpointing**, with only **0.58%** additional peak-storage saving over equally checkpointed naive at the measured realistic-vocabulary point. Large savings versus uncheckpointed naive are primarily ordinary checkpointing. See [local evidence](../benchmarks/REGIMES.md). A new training claim must outperform this fair control or explicitly change the training objective.

## 4. The main ARC-AGI/Sudoku design risk

**Repository analysis, not a reported paper finding:** the current [CPU training prototype](../benchmarks/regime_training.py) constructs `C = Down(Norm(initial_state))` once. Attention reads C through evolving queries, and its remaining MLP/norm operations are positionwise. Conditional on C, one position's later residual update does not write new values that other positions can read. Evolving queries can make increasingly sophisticated use of the original problem, but cannot read another position's newly inferred candidate through this fixed cache.

This does **not** prove failure on puzzles. A sufficiently informative initial encoder and per-position computation may still solve them. It does mean that rank sweeps alone cannot distinguish a compression bottleneck from a missing communication path. The tested causal synthetic decoder is also not yet a whole-grid puzzle model.

**A separate inference-memory distinction:** an autoregressive looped decoder needs historical KV for each loop so it can generate later tokens. A direct whole-grid solver can recompute K/V from its current grid state and discard prior-loop activations after each update. Its ordinary inference working memory can already be independent of loop count, even without LLT. Training tape and simultaneous trajectories still add memory. Consequently, the measured 4K-context decoder saving cannot be assumed for an 81-cell Sudoku solver or direct ARC grid prediction; compare actual solver peaks and include input/output grid states.

In contrast, [GRAM](https://arxiv.org/html/2605.19376v2) and [looped flows](https://arxiv.org/html/2609.11801v1) evolve solver states, while [frozen LRT](https://arxiv.org/html/2609.01117v1) keeps fixed conditioning alongside mutable scratch states. This motivates testing **immutable encoded problem memory plus one mutable solver workspace**. This is an unimplemented hypothesis, not an already-validated replacement for LLT.

Suggested workspace variant:

```text
C_input = encode_and_compress(problem)        # held fixed
Z = initialize_solver_state(problem)
for step in schedule:
    workspace = compress(Z)                  # refreshed from current state
    Z = update(Z, read(C_input), read(workspace), step)
answer = decode(Z)
```

At inference, keeping only the current workspace can avoid a per-loop history, although it adds writes and attention reads. Exact training still needs reconstruction/checkpointing of the evolving states. Two memories, or a full-width solver state, must be counted in the actual peak; this variant changes the forward model and its current measured regime.

## 5. Reasoning evidence and fair baselines

The most relevant supplied papers for **quality** are looped flows, GRAM and DeepLoop; LPT/GRT are strongest for **cache architecture**. ARC-Easy/ARC-Challenge in language-model tables are different benchmarks from ARC-AGI grid transformations.

The following are useful reference points, not a pooled leaderboard or equal-budget LLT comparisons:

| Source / protocol | Sudoku-Extreme solution accuracy | ARC-AGI-1 | ARC-AGI-2 |
|---|---:|---:|---:|
| [Looped flows, Table 1; single trajectory, three seeds](https://arxiv.org/html/2609.11801v1) | 97.9 ± 0.4% | 58.8 ± 1.8% pass@2 | 12.2 ± 1.9% pass@2 |
| [Looped flows; five-trajectory ensemble](https://arxiv.org/html/2609.11801v1) | 99.3 ± 0.2% | 59.5 ± 1.9% pass@2 | 12.2 ± 0.9% pass@2 |
| [DeepLoop HRM scaling ablation; two-vote ARC-1](https://arxiv.org/html/2607.13491v2) | — | 36.50→39.75% | — |

The flow paper follows TRM preprocessing and uses task-specific 5–7M models; its table imports comparator results from publications. A single flow trajectory is **not synonymous with one allowed ARC prediction or no augmentation**. Record task transformations, puzzle embeddings, voting, trajectories, solver calls and prediction attempts explicitly. GRAM's appendix separates sampling from augmentation, making the same distinction necessary for an LLT comparison. [Flow evaluation §5.1](https://arxiv.org/html/2609.11801v1), [GRAM Appendix D.2](https://arxiv.org/html/2605.19376v2).

Three **additional direct baselines**, outside the supplied 23, were checked at title/abstract level:

| Baseline / checked version | Why include it |
|---|---|
| [Hierarchical Reasoning Model (HRM), 2506.21734v4](https://arxiv.org/abs/2506.21734v4) | Original high/low-timescale puzzle solver; 27M parameters. Pin v4 or the earlier version used by a reproduced comparator. |
| [Less is More: Recursive Reasoning with Tiny Networks (TRM), 2510.04871v1](https://arxiv.org/abs/2510.04871v1) | Small two-layer recursive baseline with direct ARC-AGI/Sudoku relevance; useful starting point for compression ablations. |
| [Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers (FPRM), 2606.18206v1](https://arxiv.org/abs/2606.18206v1) | Fixed-point halting and residual scaling on Sudoku, Maze, state tracking and ARC-AGI. Distinct from paper #2. |

Use a trained naive recurrent solver and a non-loop control first, then static-cache LLT, LLT with mutable workspace, and a trained cache-sharing control. Add TRM/FPRM for deterministic quality comparisons and flows/GRAM when studying alternative objectives or sampling. LLA remains a post-training decoder-codec comparison, not a substitute for a native puzzle-solver training baseline.

## 6. Which additions help or oppose the resource target?

| Candidate | Expected resource effect — hypothesis unless stated otherwise | Condition for keeping it |
|---|---|---|
| Folded fixed linear K/V maps | Already beneficial in the local low-rank inference regime | Preserve trained quality and count folded-weight storage |
| Mutable compressed workspace | Adds per-loop writes/read cost; may enable needed state communication | Better matched-quality Pareto point than static C |
| FFN once, recurrent mixer only | Fewer FFN executions; MixerLoop supports testing this allocation, using a different mixer | Verify quality and measured speed on LLT, rather than inheriting DeltaNet numbers |
| Step control, trained shortcuts, adaptive exit | May reduce loops needed for correctness; EMA/gates/training routes add cost | Compare time-to-solution at matched accuracy, including control overhead |
| Residual scaling / input reinjection | Small local operations may stabilize deeper computation | Validate with LLT's actual normalization and objective |
| Four-stream hyperconnections | Expanded persistent state and checkpoint boundaries | Keep only if quality gains compensate for measured peak/storage cost |
| Parallel trajectories or speculative branches | Lower sequential depth can require more live states/caches | Report sequential and parallel execution separately, with total work and peak memory |
| Loop-specific routers/adapters/maps | More specialized capacity, but loop-dependent parameter/fold storage | Account for storage and refolding; preserve the assumptions enabling absorption |
| Temporal thought positions | Fewer unique layers, but more attention positions | Measure full KV and attention costs; parameter savings alone are insufficient |

These conclusions follow the inspected designs of [MixerLoop](https://arxiv.org/html/2608.18230v1), [TAPS](https://arxiv.org/html/2609.36653v1), [LoopFormer](https://arxiv.org/html/2602.11451v1), [Hyperloop](https://arxiv.org/html/2604.21254v3), [LoopSpec](https://arxiv.org/html/2609.17184v1), [LOOM](https://arxiv.org/html/2610.01153v1), and [temporal LRT](https://arxiv.org/html/2609.21605v1). A state gate need not invalidate fixed-map folding; input-dependent or nonlinear K/V maps require a fresh algebra/implementation check.

## 7. Prioritized experiment plan

1. **Establish a small Sudoku quality reference.** Use one pinned task representation/split and a trained naive recurrent solver, TRM-like reference and non-loop control. Use bidirectional grid access or explicitly justify a causal serialization. Record whole-puzzle validity and preservation of givens.
2. **Separate static memory from compressed working memory.** Compare full evolving attention, first-loop shared cache with local access, static low-rank C, and fixed input plus refreshed workspace. Include an uncompressed static-cache control to isolate write-path loss from rank loss. Sweep rank and loop count; do not assume rank 32 retains quality because it passes synthetic resource thresholds.
3. **Compare training strategies independently.** Exact BPTT and exact checkpointing share a full-gradient reference. Test short backward horizons and local/surrogate objectives as separate training recipes. Record forward depth, backward horizon, total examples/tokens, training wall time, optimizer memory and seed variation.
4. **Test latency allocation after quality works.** Sweep physical layers × loops, FFN frequency, scaling/gates and stopping/shortcuts. Use the same hardware, kernel path and batch when isolating architecture effects; report capacity-enabled throughput separately.
5. **Move to ARC-AGI with a pinned protocol.** State dataset edition and public/private split, training examples, augmentation and task-specific adaptation, maximum attempts, voting/selection and total inference calls. Add Maze/state tracking, then N-Queens or graph coloring if multi-solution reasoning is a goal.

For each viable configuration, report accuracy versus measured end-to-end latency and peak memory; compare both at equal compute budgets and at matched accuracy. Include training cost. Evaluate the existing target (at least 50% peak saving, at most 20% latency tax, near non-loop storage) **only after specifying an acceptable quality gap**. Published percentages are reference points, not an automatic quality threshold for a different protocol.

## 8. Claim boundaries after this review

- **Supported locally:** selected synthetic folded inference regimes and exact checkpointed CPU training, as qualified in the benchmark reports.
- **Supported by these papers:** multiple ways to share recurrent KV, stabilize recurrence, change credit assignment and reduce required inference steps; their results belong to their own architectures/protocols.
- **Still unverified for LLT:** learned rank/loop redundancy, ARC-AGI/Sudoku quality, preserving iterative communication with static C, exact training memory independent of depth, a large memory advantage over equally checkpointed naive, GPU backward and distributed communication benefits.
- **Candidate contribution:** a trained model that preserves useful recursive reasoning while sharing a small memory across layers and loops and improves the measured quality–memory–latency frontier. This is the proposition to test; this review does not certify novelty.
