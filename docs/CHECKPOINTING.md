# Checkpoint boundaries

LAC is still preliminary in this implementation. The current experiment tests an
**exact checkpoint around LLT's existing latent attention/output branch**, not a
method for recovering a full loop's residual state from its KV latent.

## Implemented policies

| Policy | Boundary inputs | Recomputed region | State outside the boundary |
|---|---|---|---|
| None | Ordinary autograd | None | All normal saved intermediates |
| AC | Full-width block input; shared latent/fold weights for LLT | Entire Transformer block | Block boundaries, shared folds/latent, embeddings, loss, optimizer state |
| LAC, exploratory | Projected queries Q_r, existing C, folded output weight | Latent attention and folded output projection | Query formation, residual path, full MLP path, folds, loss, optimizer state |

Both AC and LAC use non-reentrant Torch checkpointing. Numerical operations are
routed to the chosen Torch or Tensor backend during recomputation. The current
blocks contain no stochastic operation; these checkpoints disable RNG-state
preservation. Neither policy adds a codec, parameters, or an approximate residual
reconstruction. Tests compare every parameter gradient with uncheckpointed
backward, in addition to checking forward values.

## Why current LAC uses much more memory than AC

At rank 64, T=16, B4/S1024 on L40S, captured Tensor training peaks at **4.79 GiB
with AC, 21.70 GiB with LAC, and 25.11 GiB without checkpoints**. AC discards
intermediates from an entire block and recomputes them during backward. The
current LAC discards only its attention/output branch. Its query construction,
full-width residual computations, and wide MLP activations retain their normal
autograd state over 192 block applications.

C is an attention memory derived from initial embeddings. It is not a complete
representation of each block's evolving residual X. Recovering X from C alone
would change the model, require additional state, or require replaying an earlier
forward prefix. The implemented branch boundary therefore does not establish
constant total training memory. Designing a broader, useful latent checkpoint
boundary is an open architecture question.

## Applicability

Among the four architectures in this sweep, only LLT exposes this particular
native Q_r/C boundary. Conventional Naive Loop, stack, and fixed-depth controls
show **LAC: N/A**. Standard AC applies to all four.

LAC is not inherently exclusive to the LLT name: another architecture with a
suitable native latent state could expose an exact boundary too. Its inputs must
be sufficient to recompute the selected region. Compressing an arbitrary
full-width checkpoint and calling it LAC is outside the intended design.

## Qualification and limits

The [main sweep](../experiments/l40s/LOOP_SWEEP.md#numerical-qualification) contains
small and full-geometry checks, including repeated uncheckpointed backward.
Torch BF16 full-size backward varied even when rerunning an unchanged model, so
Torch correctness checks use deterministic algorithms and a cuBLAS workspace.
Performance measurements keep their original default settings. Tensor checks
retain the original numerical kernels. This separates numerical variability from
checkpoint correctness without substituting controlled timings for measured ones.

The relevant implementation is [checkpointing.py](../model/checkpointing.py).
The current results qualify this branch checkpoint, without demonstrating the
intended broader memory benefit or trained quality.

## Added architecture controls

The [added-family study](RESEARCH_BASELINES.md) profiles training none/AC only.
Its per-layer latent control also exposes a native low-rank attention boundary,
so that type of checkpointing is not inherently exclusive to LLT. No LAC result
is measured for the added families, and no full-residual compression is implied.
