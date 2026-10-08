# Checkpoint boundaries

The current LLT sweep measures **none and standard AC**. LAC is excluded.
The low-rank KV memories are not sufficient to reconstruct full-width residual
states, so per-layer contextual memory alone does not establish useful LAC.

## Standard AC

[ContextualLLT](../model/contextual_llt.py) uses non-reentrant Torch checkpointing
around each full Transformer block. On the first loop the region constructs
C_i from its block input and returns both C_i and the updated residual. Later
loops receive C_i as an explicit input and reuse it without refresh or detachment.
Gradients through shared memory and folded projection weights remain intact.

The block's residual input and first-loop memories remain live checkpoint state;
AC recomputes normalization, queries, attention, projections, and MLP intermediates.
Folds, embeddings, classifier/loss and optimizer state sit outside block AC.
None/AC numerical checks compare every parameter gradient and causal cached logits.
Full-size correctness uses deterministic controls separately from performance.
The architecture has no stochastic blocks, so RNG preservation is disabled.

Standard AC also remains measured for all conventional and research controls.
See [baseline contracts](RESEARCH_BASELINES.md) for GRT's gates/projection outside
AC and attention-only loop's separate attention/FFN checkpoint regions.

## Historical LAC

The globally shared embedding-latent prototype used an exact but narrow
attention/output checkpoint. It retained projected queries, the shared latent
and folded output weight, leaving residual and MLP activations outside the region.
That policy's numerical correctness and weak memory savings are historical;
it is not implemented or profiled for the current contextual LLT.

The frozen implementation is [checkpointing.py](../model/checkpointing.py), and
its evidence remains in the [historical report](../archive/reports/L40S_GLOBAL_LATENT_SWEEP.md).
Designing a broader latent checkpoint boundary remains an open research question.
