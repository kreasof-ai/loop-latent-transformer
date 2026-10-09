# Contextual per-layer LLT replacement study

This campaign replaces the globally shared embedding-latent rows in the
[main report](../../LOOP_SWEEP.md). LLT builds one latent per physical layer on
its first loop and reuses it across later loops. The 12-bank cache is constant
in T. Rank is swept over 32/64/128; no LAC is run.

The grid has 288 performance records: 16 loop counts, 3 ranks, 2 backends,
and training none/AC plus inference. Each inference record contains prompt,
serving startup and supplied-token decode. Six small and six full-size numerical
qualifications check all-parameter AC gradients, backend agreement and cached logits.

Raw JSON and hash-addressed source snapshots are frozen measured payloads.
[views/summary.csv](views/summary.csv) contains the replacement LLT records;
[views/all-architectures.csv](views/all-architectures.csv) contains 1056 active
records after joining 768 unchanged non-LLT measurements. Historical LLT and
LAC rows are excluded from this active join and retained in their original data.
[views/audit.json](views/audit.json) verifies the new grid and CUDA artifacts.
[run-manifest.json](run-manifest.json) hashes immutable payloads and pins measured
source/runtime revisions. [views/run-manifest.json](views/run-manifest.json)
pins the report generator separately.

CUDA artifacts and logs remain ignored machine cache files. Raw records retain
those original absolute paths; reproducing a strict artifact audit requires them.
Use [the runner](../../run_contextual_llt_sweep.sh) with a fresh LLT_RESULTS_ROOT;
never add new numerical revisions to this published campaign.
