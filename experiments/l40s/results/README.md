# Published result payloads

| Directory | Contents |
|---|---|
| [contextual-llt/](contextual-llt/README.md) | Current LLT: 288 fresh contextual per-layer profiles, 12 qualifications, no LAC |
| [loop-baseline/](loop-baseline/) | Original 256 performance cases, eight backend checks, and supplementary capture retries |
| [loop-sweep/](loop-sweep/) | 416 additional rank/checkpoint cases, 12 small and 12 full-size qualifications, and numerical diagnostics |
| [research-baselines/](research-baselines/README.md) | 480 additional architecture profiles, ten small and ten full-size qualifications |
| `*/sources/<sha256>/` | Exact source snapshots attached to the measured runs |
| `*/views/` | Current CPU-generated audits, tables, figures, CSVs, and view-generator manifests |

The current report is [LOOP_SWEEP.md](../LOOP_SWEEP.md). Its active
[joined CSV](contextual-llt/views/all-architectures.csv) replaces historical LLT
rows with contextual per-layer memory, retains 768 non-LLT measurements and
excludes LAC. The current [audit](contextual-llt/views/audit.json) verifies the
replacement grid and source/artifact provenance.

The original [672-record CSV](loop-sweep/views/combined.csv) and
[1152-record historical join](research-baselines/views/all-architectures.csv)
remain unchanged as evidence for their historical source families. Their LLT
rows describe a single embedding-derived global latent, not current LLT.

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
