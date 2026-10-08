# Additional architecture-family measurements

Read the canonical [LOOP_SWEEP.md](../../LOOP_SWEEP.md) for results and the
[architecture contract](../../../../docs/RESEARCH_BASELINES.md) for primary sources
and explicit adaptations. These random-weight kernel profiles include no quality evaluation.

- `training-*.json` and `inference-*.json`: 480 primary records across five families,
  T=1..16, Tensor/Torch, training none/AC and three inference operations.
- `qualification-*.json` and `full_qualification-*.json`: ten small and ten full-size checks.
- [Measurement manifest](run-manifest.json): numerical source/runtime identities and immutable payload hashes.
- `sources/` and `tensor-sources/`: hash-addressed measured source snapshots.
- [Joined CSV](views/all-architectures.csv): all 1152 performance records, including the original 672.
- [Added-family CSV](views/summary.csv), [audit](views/audit.json), and [view manifest](views/run-manifest.json).
- `views/*.png` and `views/*.pdf`: standalone plots; [all-loop tables](views/tables.md) are included in the canonical report.

GPU compilation artifacts remain machine caches outside Git. Raw records retain
those absolute paths; strict auditing requires their bytes. The manifest in `views/`
records the tools that regenerate reports separately from numerical measurement revisions.
