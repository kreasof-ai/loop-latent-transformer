# Published result payloads

| Directory | Contents |
|---|---|
| [loop-baseline/](loop-baseline/) | Original 256 performance cases, eight backend checks, and supplementary capture retries |
| [loop-sweep/](loop-sweep/) | 416 additional rank/checkpoint cases, 12 small and 12 full-size qualifications, and numerical diagnostics |
| `*/sources/<sha256>/` | Exact source snapshots attached to the measured runs |
| `*/views/` | Current CPU-generated audits, tables, figures, CSVs, and view-generator manifests |

The report is [LOOP_SWEEP.md](../LOOP_SWEEP.md). The combined
[CSV](loop-sweep/views/combined.csv) and [audit](loop-sweep/views/audit.json)
include all 672 primary records. Original summaries and figures remain beside
the measured JSON files for historical provenance; current navigation uses `views/`.

## Preservation and path relocation

The original payloads were moved from `benchmarks/results/`. Their contents,
original source snapshots, original audits, and original measurement manifests
were preserved byte for byte. Earlier studies moved to [archive/results/](../../../archive/results/).
The [relocation map](relocation.json) records the prior commit, old/new prefixes,
and original SHA256 for 1023 tracked result/source files.

Recorded absolute artifact paths and source paths deliberately retain the
locations used during measurement. [runtime.py](../runtime.py) resolves relocated
artifact paths while auditing. CUDA compilation artifacts are machine cache files
and remain ignored by Git; auditing their bytes requires this machine's artifacts
or the corresponding retrieved cache. A fresh checkout can inspect all committed
raw records and source snapshots but cannot strictly audit absent CUDA artifacts.

The immutable measurement manifests pin the original numerical sources and
Tensor revision. The manifests in `views/` describe the current report generators
and hash the original manifests separately. Moving or regenerating documentation
does not claim that the original measurements were produced by current HEAD.

New numerical runs require a separate `LLT_RESULTS_ROOT`. See
[reproducibility](../../../docs/REPRODUCIBILITY.md).
