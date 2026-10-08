"""Audit replacement LLT records and build the active comparison without LAC.

Historical raw records and historical views are never overwritten. Only the
canonical report and this campaign's derived views are regenerated.
"""

import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.l40s.runtime import (
    ROOT,
    PUBLISHED_RESULTS,
    results_root,
    recorded_path,
    record_label,
)
from experiments.l40s.research_summary import METRICS, LABELS as RESEARCH_LABELS

OUT = results_root() / "contextual-llt"
VIEWS = OUT / "views"
RANKS = (32, 64, 128)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return (
        row["phase"],
        row["model"],
        row["kv_rank"] or 64,
        row["backend"],
        row["loops"],
        row["checkpoint"],
    )


def audit_new():
    rows, checks, inputs, artifacts = [], [], {}, {}
    profiles, commits, definitions = set(), set(), set()
    for path in sorted(OUT.glob("*.json")):
        if not path.name.startswith(
            ("training-", "inference-", "qualification-", "full_qualification-")
        ):
            continue
        d = json.loads(path.read_text())
        a, provenance = d["arguments"], d["provenance"]
        assert d["status"] in ("passed", "out_of_memory"), (path, d.get("error"))
        assert a["model"] == "llt" and a["policy"] in ("none", "ac")
        assert provenance["gpu"] == "NVIDIA L40S" and provenance["native_cpp_executor"]
        inputs[record_label(path)] = sha(path)
        profiles.add(tuple(sorted(provenance["sources"].items())))
        commits.add((provenance["llt_commit"], provenance["tensor_commit"]))
        definitions.add(json.dumps(d["architecture"], sort_keys=True))
        for name, digest in provenance["sources"].items():
            assert sha(OUT / "sources" / digest / Path(name).name) == digest
        for name, digest in provenance["runtime_binary_sha256"].items():
            assert sha(Path(name)) == digest
        coverage = d.get("coverage")
        if coverage:
            assert not coverage["fallbacks"]
            for artifact in coverage["artifacts"]:
                if artifact["sha256"] not in artifacts:
                    assert sha(recorded_path(artifact["path"])) == artifact["sha256"]
                    artifacts[artifact["sha256"]] = artifact
        if a["phase"].endswith("qualification"):
            assert d["status"] == "passed"
            for name, check in d["policy_checks"].items():
                if name.endswith("_cache"):
                    assert check["max_abs_error"] < 0.05 and check["relative_l2"] < 0.03
                else:
                    threshold = 0.15 if name == "cross_backend" else 1e-5
                    assert (
                        check["gradient_error"]["maximum_parameter_relative_l2"]
                        < threshold
                    )
            checks.append((path, d))
            continue
        c = d["config"]
        assert (
            c["width"],
            c["heads"],
            c["layers"],
            c["rank"],
            c["vocab"],
            c["max_seq"],
        ) == (768, 12, 12, a["rank"], 50304, 1025)
        assert (d["batch_size"], d["sequence_length"]) == (4, 1024)
        assert d["checkpoint_policy"] == a["policy"]
        row = dict(
            phase=a["phase"],
            model="llt",
            kv_rank=a["rank"],
            backend=a["backend"],
            loops=a["loops"],
            checkpoint=a["policy"],
            status=d["status"],
            parameter_count=d["parameter_count"],
            effective_depth=12 * a["loops"],
            unique_layers=12,
            attention_applications=12 * a["loops"],
            mlp_applications=12 * a["loops"],
            measurement_origin="contextual-llt",
            core_residual_dtype="float32",
            failed_stage=d.get("stage", ""),
            raw_record=record_label(path),
        )
        for metric in METRICS:
            if metric not in d:
                continue
            m = d[metric]
            assert len(m["gpu_samples_ms"]) == a["samples"] == 9
            assert all(math.isfinite(v) and v > 0 for v in m["gpu_samples_ms"])
            assert math.isclose(
                m["gpu_median_ms"],
                statistics.median(m["gpu_samples_ms"]),
                rel_tol=1e-12,
            )
            row[metric + "_gpu_ms"] = m["gpu_median_ms"]
            row[metric + "_peak_gib"] = (
                m.get("capture_peak_allocated_bytes", m.get("peak_allocated_bytes"))
                / 2**30
            )
            if metric.endswith("_graph"):
                assert m["repeats_per_sample"] == a["repeats"] == 3
            if "wall_median_ms" in m:
                row[metric + "_wall_ms"] = m["wall_median_ms"]
        if "training_graph" in d:
            counts = d["graph_step_counter"]
            assert counts["min"] == counts["max"] == counts["expected"] == 42
            assert d["all_parameter_gradients_finite"] and math.isfinite(
                d["graph_last_loss"]
            )
        if a["phase"] == "inference":
            for field, source in (
                ("cache_mib", "cache_bytes"),
                ("fold_mib", "fold_bytes"),
                ("prepared_weight_mib", "prepared_weight_bytes"),
            ):
                row[field] = d[source] / 2**20
            row["cache_banks"] = d["cache_banks"]
            assert d["cache_bytes"] == d["expected_cache_bytes"]
        rows.append(row)
    expected = {
        (phase, "llt", rank, backend, t, policy)
        for rank, backend, t in itertools.product(
            RANKS, ("tensor", "torch"), range(1, 17)
        )
        for phase, policy in (
            ("training", "none"),
            ("training", "ac"),
            ("inference", "none"),
        )
    }
    assert len(rows) == 288 and {key(r) for r in rows} == expected
    assert len(profiles) == len(commits) == len(definitions) == 1
    assert {
        (d["arguments"]["rank"], d["arguments"]["loops"])
        for _, d in checks
        if d["arguments"]["phase"] == "qualification"
    } == set(itertools.product(RANKS, (4, 16)))
    assert {
        (d["arguments"]["rank"], d["arguments"]["backend"])
        for _, d in checks
        if d["arguments"]["phase"] == "full_qualification"
    } == set(itertools.product(RANKS, ("tensor", "torch")))
    audit = dict(
        status="passed",
        case_count=len(rows),
        qualification_count=len(checks),
        passed=sum(r["status"] == "passed" for r in rows),
        out_of_memory=sum(r["status"] == "out_of_memory" for r in rows),
        architecture=json.loads(next(iter(definitions))),
        numerical_sources=dict(next(iter(profiles))),
        commits=list(next(iter(commits))),
        input_sha256=inputs,
        artifacts=list(artifacts.values()),
        unique_artifact_count=len(artifacts),
    )
    return rows, audit


def write_rows(path, rows):
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def label(model, rank):
    return (
        f"LLT rank {rank}"
        if model == "llt"
        else (
            RESEARCH_LABELS.get(model)
            or {
                "naive_loop": "Naive Loop",
                "stacked": "Independent stack",
                "fixed_depth": "Fixed depth / matched params",
            }[model]
        )
    )


def cell(row, metric):
    if metric + "_gpu_ms" not in row:
        return "OOM" if row["failed_stage"] == metric else "— (earlier OOM)"
    return f"{row[metric+'_gpu_ms']:.2f} / {row[metric+'_peak_gib']:.2f}"


def table(records, variants, backend, loops, include_loop=False):
    tcol = " T |" if include_loop else ""
    text = f"|{tcol} Architecture | Parameters (M) | Train: none | Train: AC | Prompt | Startup | Cached decode | Cache (MiB) |\n"
    text += (
        "|"
        + ("---:|" if include_loop else "")
        + "---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for t in loops:
        for model, rank in variants:
            inference = records[("inference", model, rank, backend, t, "none")]
            values = [
                cell(
                    records[("training", model, rank, backend, t, p)], "training_graph"
                )
                for p in ("none", "ac")
            ]
            values += [
                cell(inference, metric)
                for metric in ("prompt_graph", "startup_graph", "decode_graph")
            ]
            text += (
                "| "
                + (f"{t} | " if include_loop else "")
                + label(model, rank)
                + f" | {inference['parameter_count']/1e6:.2f} | "
                + " | ".join(values)
                + f" | {inference['cache_mib']:.2f} |\n"
            )
    return text + "\n"


def main():
    VIEWS.mkdir(parents=True, exist_ok=True)
    new, audit = audit_new()
    old_path = PUBLISHED_RESULTS / "research-baselines/views/all-architectures.json"
    old = json.loads(old_path.read_text())
    preserved = [r for r in old if r["model"] != "llt"]
    assert len(preserved) == 768 and all(r["checkpoint"] != "lac" for r in preserved)
    rows = preserved + new
    assert len(rows) == len({key(r) for r in rows}) == 1056
    audit.update(
        active_case_count=len(rows),
        preserved_case_count=len(preserved),
        historical_join_sha256=sha(old_path),
        active_passed=sum(r["status"] == "passed" for r in rows),
        active_out_of_memory=sum(r["status"] == "out_of_memory" for r in rows),
    )
    write_rows(VIEWS / "summary", new)
    write_rows(VIEWS / "all-architectures", rows)
    (VIEWS / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    records = {key(r): r for r in rows}
    variants = [("llt", r) for r in RANKS] + [
        (m, 64)
        for m in (
            "naive_loop",
            "stacked",
            "fixed_depth",
            "uyoco",
            "lpt",
            "grt",
            "per_layer_latent",
            "attention_only",
        )
    ]
    text = "# L40S loop and rank sweep: contextual LLT\n\n"
    text += "This is the current kernel latency and peak-memory study: **B4/S1024, T=1–16**, LLT ranks **32/64/128**, and training with **none or standard AC**. LAC is excluded. Tensor and matched Torch backends are profiled. These synthetic-token measurements do not establish trained language-model quality.\n\n"
    text += (
        "## Corrected LLT architecture\n\n"
        + audit["architecture"]["description"]
        + "\n\n"
    )
    text += "Each physical layer has its own down projection and head-specific K/V expansion weights. Expansion is folded into queries and output projections; cache keys and values share one physical latent tensor. The full-width residual evolves through all 12T applications. See [architecture](../../docs/ARCHITECTURE.md) and [implementation](../../model/contextual_llt.py).\n\n"
    text += "The previous globally shared embedding-derived LLT rows are **superseded**, not relabeled or reused. Their raw records remain unchanged; the [historical report](../../archive/reports/L40S_GLOBAL_LATENT_SWEEP.md) retains their results. All non-LLT rows below retain their original measurements, dates, and source provenance.\n\n"
    text += "## Measurement protocol\n\nWidth 768, 12 heads, head dimension 64, 12 tied blocks, vocabulary 50,304, position capacity 1025, MLP width 3072, unweighted RMSNorm epsilon 1e-5, and exact GELU. FP32 masters, embeddings, residuals, losses, and optimizer state; BF16 projections and attention. Attention scaling stays 1/sqrt(64).\n\n"
    text += "Training includes full-token logits/loss, backward, gradient clipping at 1, and AdamW (lr 0.0006, betas 0.9/0.95, weight decay 0.1). Standard non-reentrant AC recomputes full block regions and preserves all latent-memory gradient paths. Three warmups, nine samples, three graph replays per sample; 42 optimizer updates are checked. Each case runs in a fresh process, sequentially on NVIDIA L40S. Compilation is excluded.\n\n"
    text += "Inference measures causal prompt with last-position logits, serving startup including persistent-cache allocation/copies and fold rebuild, and one supplied cached token after 1024 tokens. Capacity is 1025; captured decode includes logical rewind. BF16 serving weight copies are counted. No sampling or beam search is included. Graph memory is peak allocated during capture, including graph pool and temporaries; allocator reservations and driver memory are excluded. Eager timing and memory remain in CSVs.\n\n"
    text += "Controls retain their original adaptations: U-YOCO has 6T+6 attention/FFN applications, GRT 8T+4 with BF16 core residuals, and attention-only loop 12T attention updates but 12 FFNs. Other variants apply 12T blocks, except fixed depth with 12 blocks and widened MLPs matching the independent stack's parameters. Thus T and parameters do not imply matched compute. [Baseline contracts](../../docs/RESEARCH_BASELINES.md) document source fidelity and checkpoint boundaries.\n\n"
    text += f"## Results\n\n**{len(rows)} active records: {audit['active_passed']} passed, {audit['active_out_of_memory']} OOM.** This replaces 384 historical LLT records (including LAC) with 288 new LLT records and preserves 768 non-LLT records. New LLT: {audit['passed']} passed, {audit['out_of_memory']} OOM.\n\nCells are **CUDA graph milliseconds / capture peak allocated GiB**, per batch of four. Cache is persistent payload plus Tensor counters, separately from total GPU memory.\n\n## All architectures at T=16\n\n"
    for backend in ("tensor", "torch"):
        text += f"### {backend.capitalize()}\n\n" + table(
            records, variants, backend, (16,)
        )
    text += f"## Numerical qualification and provenance\n\n{audit['qualification_count']} qualifications cover all ranks at small T=4/16 and full B4/S1024 T=16 on both backends. AC loss and every parameter gradient must match none (relative L2 <1e-5). Small cross-backend gradients must be below 0.15 relative L2. Cached logits must match full causal execution (max absolute error <0.05; relative L2 <0.03). Full-size checks use deterministic controls separately from performance timings. Cache bank counts and byte formulas are checked. The audit verifies {audit['unique_artifact_count']} hashed CUDA artifacts and zero implicit Tensor numerical fallback.\n\n"
    text += "- [Active joined CSV](results/contextual-llt/views/all-architectures.csv).\n- [New LLT CSV](results/contextual-llt/views/summary.csv) and [audit](results/contextual-llt/views/audit.json).\n- [Reproduction harness](run_contextual_llt_sweep.sh).\n\n"
    text += "## Every loop count\n\n"
    for backend in ("tensor", "torch"):
        text += f"### {backend.capitalize()} — T=1–16\n\n" + table(
            records, variants, backend, range(1, 17), True
        )
    (ROOT / "experiments/l40s/LOOP_SWEEP.md").write_text(text)
    for metric, title in (
        ("training_graph", "Training"),
        ("decode_graph", "Cached decode"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for j, rank in enumerate(RANKS):
            for i, (suffix, unit) in enumerate(
                (("gpu_ms", "Latency (ms)"), ("peak_gib", "Peak allocated (GiB)"))
            ):
                ax = axes[i, j]
                for backend in ("tensor", "torch"):
                    for policy in (
                        ("none", "ac") if metric == "training_graph" else ("none",)
                    ):
                        phase = (
                            "training" if metric == "training_graph" else "inference"
                        )
                        values = [
                            records[(phase, "llt", rank, backend, t, policy)].get(
                                metric + "_" + suffix, float("nan")
                            )
                            for t in range(1, 17)
                        ]
                        ax.plot(range(1, 17), values, label=f"{backend} {policy}")
                ax.set_title(f"LLT rank {rank}")
                ax.set_ylabel(unit)
                ax.set_xlabel("Loops")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
        fig.suptitle(title + ": contextual per-layer LLT")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(VIEWS / f"{metric}.{ext}", dpi=160)
        plt.close(fig)
    print("audit passed:", len(new), "new LLT records;", len(rows), "active records")


if __name__ == "__main__":
    main()
