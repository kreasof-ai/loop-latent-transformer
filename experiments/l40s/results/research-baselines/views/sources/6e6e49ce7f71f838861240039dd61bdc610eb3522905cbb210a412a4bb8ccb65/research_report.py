"""Render the audited architecture-family extension into the canonical report."""

import json
from experiments.l40s.runtime import PUBLISHED_RESULTS, results_root
from experiments.l40s.research_summary import LABELS, FAMILIES, cell


def section():
    root = results_root() / "research-baselines"
    views = root / "views"
    if not (views / "audit.json").exists():
        return ""
    audit = json.loads((views / "audit.json").read_text())
    assert audit["status"] == "passed" and audit["case_count"] == 480
    rows = json.loads((views / "summary.json").read_text())
    records = {
        (r["phase"], r["model"], r["backend"], r["loops"], r["checkpoint"]): r
        for r in rows
    }
    original_rows = json.loads(
        (PUBLISHED_RESULTS / "loop-sweep/views/combined.json").read_text()
    )
    original = {
        (
            r["phase"],
            r["model"],
            r["kv_rank"] or 64,
            r["backend"],
            r["loops"],
            r["checkpoint"],
        ): r
        for r in original_rows
    }
    text = f"""\n## Additional architecture families: kernel profiles

The extension adds **480 performance records: {audit['passed']} passed and
{audit['out_of_memory']} recorded OOM**, bringing the joined study to **1152 records**.
The original 672 measurements are retained. These are random-weight kernel
adaptations, without trained-quality evaluation. The [architecture contract](../../docs/RESEARCH_BASELINES.md)
describes primary sources and every material adaptation.

B4/S1024, width 768, 12 heads, vocabulary 50,304, BF16 projections, FP32 masters,
optimizer, timing and memory protocols match the original study. Training measures
none and exact AC; inference measures prompt, serving startup and supplied-token
decode. These families have no additional LAC policy. U-YOCO uses RoPE,
weighted RMSNorm and SwiGLU; GRT uses learned LayerNorm and recurrent gates.
Other added controls retain the original normalization, positions and GELU.
Residuals are FP32 except inside GRT's projected core, which uses BF16; its
prelude, coda and retention blend use FP32. The CSV records this precision detail.
The inherited generic serving-description string in raw GRT records describes
FP32 residuals; this statement applies outside its recurrent core.

| Profile | Applied attention blocks | FFN applications | Cache boundary |
|---|---:|---:|---|
| U-YOCO / SWA | 6T + 6 | 6T + 6 | Looped self-decoder windows; one global bank shared by the cross-decoder |
| LPT cache layout | 12T | 12T | First-loop per-layer full KV plus later-loop local KV, with one joint softmax |
| GRT / full KV (BF16 core) | 8T + 4 | 8T + 4 | Full KV for prelude, every recurrent application, and coda |
| Per-layer latent loop (MLA-style) | 12T | 12T | Rank-64 latent refreshed and stored per application |
| Attention-only loop / FFN once | 12T | 12 | Full KV for every attention application |

GRT additionally executes its gate MLP and recurrence projection T times; those
costs are included and their application counts are in the joined CSV.

**T does not match compute across these architectures.** The existing stack and
matched-parameter controls remain in the original tables. Parameter counts,
attention/FFN application counts, prepared weights and cache storage accompany
latency and peak memory in the joined CSV.

U-YOCO is a 12-KV-head adaptation. LPT preserves the shared/local cache topology
rather than the paper's complete Ouro block. GRT includes training state/gate noise,
but inference and correctness checks disable it; averaged-loop KV is not substituted.
The per-layer latent control omits DeepSeek's query compression, rotary-key stream
and MoE. The attention-only control uses softmax attention rather than MixerLoop's
DeltaNet mixer. These names must not be read as published checkpoint reproductions.

Torch window/union attention uses compiled FlexAttention; Tensor uses native masked
tiles and a two-bank decode kernel. Torch LPT decode concatenates its banks for
Flash SDPA, and that copy is included in timing. U-YOCO prompt/startup computes
only the last cross-decoder query, which is valid because its shared memory is fixed.
The preserved LLT prompt path computes all query positions. Its fixed latent also
permits last-query-only execution, which those measurements do not yet use. Prompt
latency reflects these serving implementation choices as well as architecture.
Local-window cache storage still grows with T. GRT recurrence projections/gates sit
outside block AC; attention-only AC checkpoints each attention update and FFN separately.

### All architectures at T=16

Cells are **CUDA graph ms / capture peak allocated GiB**, per batch of four.

"""
    for backend in ("tensor", "torch"):
        text += f"**{backend.capitalize()}**\n\n| Architecture | Parameters (M) | None | AC | Prompt | Startup | Cached decode | Cache (MiB) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
        from experiments.l40s.latent_checkpoint_summary import (
            VARIANTS,
            LABELS as ORIGINAL_LABELS,
        )

        comparison = []
        for model, rank in VARIANTS:
            inference = original[("inference", model, rank, backend, 16, "none")]
            values = [
                cell(
                    original[("training", model, rank, backend, 16, p)],
                    "training_graph",
                )
                for p in ("none", "ac")
            ]
            values += [
                cell(inference, m)
                for m in ("prompt_graph", "startup_graph", "decode_graph")
            ]
            comparison.append((ORIGINAL_LABELS[(model, rank)], inference, values))
        for family in FAMILIES:
            inference = records[("inference", family, backend, 16, "none")]
            values = [
                cell(records[("training", family, backend, 16, p)], "training_graph")
                for p in ("none", "ac")
            ]
            values += [
                cell(inference, m)
                for m in ("prompt_graph", "startup_graph", "decode_graph")
            ]
            comparison.append((LABELS[family], inference, values))
        for label, inference, values in comparison:
            text += (
                "| "
                + label
                + f" | {inference['parameter_count']/1e6:.2f} | "
                + " | ".join(values)
                + f" | {inference['cache_mib']:.2f} |\n"
            )
        text += "\n"
    text += "### Timing continuity check\n\nAt T=1, LPT has the same full causal attention and FFN equations as Naive Loop.\nIts independently measured training serves as a continuity check for the preserved\noriginal timings. The native runtime binaries have the same SHA256 as the original study.\n\n| Backend | Original Naive T=1 (ms / GiB) | New LPT T=1 (ms / GiB) | Latency change |\n|---|---:|---:|---:|\n"
    for backend in ("tensor", "torch"):
        previous = original[("training", "naive_loop", 64, backend, 1, "none")]
        current = records[("training", "lpt", backend, 1, "none")]
        change = 100 * (
            current["training_graph_gpu_ms"] / previous["training_graph_gpu_ms"] - 1
        )
        text += f"| {backend.capitalize()} | {cell(previous, 'training_graph')} | {cell(current, 'training_graph')} | {change:+.2f}% |\n"
    text += "\nThis checks the first-loop regime; it does not replace a contemporaneous rerun of every original case.\n\n"
    text += f"""### Qualification and records

The added-family audit verifies {audit['qualification_count']} small qualifications
(T=4/16), {audit['full_qualification_count']} full B4/S1024 T=16 qualifications,
{audit['unique_artifact_count']} hashed CUDA artifacts and zero implicit Tensor
numerical fallback. Small tests compare every parameter gradient across backends;
within-backend none/AC gradients have relative L2 below 1e-5. Full-size tests first
repeat uncheckpointed backward under deterministic correctness controls. Cached
logits must have relative L2 below 0.03 and maximum error below 0.05 against full
forward. Correctness controls do not alter primary performance settings.

- [Joined CSV: all 1152 performance records](results/research-baselines/views/all-architectures.csv).
- [Added-family audit](results/research-baselines/views/audit.json) and [manifest](results/research-baselines/views/run-manifest.json).
- [Added-family CSV](results/research-baselines/views/summary.csv), including eager/wall timings and cache bytes.
- [Model implementation](../../model/research_baselines.py) and [sweep harness](research_sweep.py).

![Additional families: training](results/research-baselines/views/training_graph.png)

![Additional families: prompt](results/research-baselines/views/prompt_graph.png)

![Additional families: cached decoding](results/research-baselines/views/decode_graph.png)

### Added families at every loop count

"""
    return text + (views / "tables.md").read_text()
