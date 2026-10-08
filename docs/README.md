# Documentation map

Read the [architecture](ARCHITECTURE.md), then the
[main L40S experiment](../experiments/l40s/LOOP_SWEEP.md). The experiment is the
current resource-cost result; the architecture describes what was implemented.

## Design and implementation

| Document | Question it answers |
|---|---|
| [Architecture](ARCHITECTURE.md) | What does LLT compute, and what scales with loop count? |
| [Additional architecture profiles](RESEARCH_BASELINES.md) | Which source designs and adaptations are profiled? |
| [Checkpointing](CHECKPOINTING.md) | Which states do AC and the current LAC discard or retain? |
| [Model package](../model/README.md) | Where are the reference, backend adapter, and checkpoint implementations? |
| [Inference helpers](../inference/README.md) | How are serving weights prepared and cached decode measured? |
| [Experiment harness](../experiments/l40s/README.md) | Which scripts measure, audit, and publish the sweep? |
| [Reproducibility](REPRODUCIBILITY.md) | How are runs reproduced and published payloads verified? |

## Research record

- [Main result: L40S loop sweep](../experiments/l40s/LOOP_SWEEP.md).
- [Open research questions](research/ROADMAP.md).
- [Literature notes, 2026-10-07](research/LITERATURE_2026-10-07.md): dated context, not implementation requirements.
- [Archive index](../archive/README.md): original proposal, earlier measurements, and retired prototypes.

Raw measured records and hash-addressed source snapshots are authoritative for
individual runs. The current report, tables, and figures are derived views.
Their [result layout](../experiments/l40s/results/README.md) distinguishes original
measurement provenance from the tools that regenerate views after reorganization.

The layout follows the architecture/code/docs/archive separation used in
[ATMA](https://github.com/kreasof-ai/atma). LLT does not yet have a manuscript or
a trained-quality study.
