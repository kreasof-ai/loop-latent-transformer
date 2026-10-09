# Open research questions

The [main L40S sweep](../../experiments/l40s/LOOP_SWEEP.md) measures resource cost
for the current first-loop contextual per-layer memory architecture. This roadmap identifies what
those measurements leave unresolved; it does not report additional experiments.

## Architectural quality

Train matched LLT, Naive Loop, and independent-stack controls to determine how a
fixed first-loop contextual KV memory affects language modeling. Sweep loop count and
latent rank at comparable training budgets. Compare quality as well as parameters,
attention work, latency, and memory; equal parameters alone do not match compute.
The current synthetic-token profiles establish no trained-quality result.

## Checkpoint boundary

The historical LAC branch passed exactness checks but retained most residual/query/MLP
activation state. LAC is excluded from the current experiment. Find a native latent boundary that supports a larger useful
recomputation region while preserving the intended model. Specify which state is
sufficient to recover the evolving residual, and measure the extra recomputation
and storage. Block AC is the qualified exact baseline. LAC remains preliminary.

## Kernel and serving scale

Use the existing rank/loop profiles to identify launch, attention, folding, and
MLP costs. Assess cached-token execution over longer histories and different
batch sizes. The current decode experiment computes supplied tokens without
sampling or beam search. Any broader serving study needs its own explicit
protocol and raw records.

## Evidence management

Keep [LOOP_SWEEP.md](../../experiments/l40s/LOOP_SWEEP.md) as the current result
until a new study answers a distinct research question. Preserve source snapshots,
failed cases, and run provenance. Tensor kernel/runtime development is documented
in [Tensor](https://github.com/kreasof-ai/tensor); LLT documents architecture,
experiment definitions, and resulting evidence.
