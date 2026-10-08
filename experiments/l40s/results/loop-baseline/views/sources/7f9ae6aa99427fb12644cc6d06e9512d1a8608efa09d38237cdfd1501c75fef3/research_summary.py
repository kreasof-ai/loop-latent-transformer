"""Audit added architecture profiles and join their measured rows with the main study."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import csv
import hashlib
import itertools
import json
import math
import subprocess
import statistics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.l40s.runtime import (
    ROOT,
    PUBLISHED_RESULTS,
    results_root,
    record_label,
    recorded_path,
)

FAMILIES = ("uyoco", "lpt", "grt", "per_layer_latent", "attention_only")
LABELS = {
    "uyoco": "U-YOCO / SWA",
    "lpt": "LPT cache layout",
    "grt": "GRT / full KV (BF16 core)",
    "per_layer_latent": "Per-layer latent loop (MLA-style)",
    "attention_only": "Attention-only loop / FFN once",
}
METRICS = (
    "training_eager",
    "training_graph",
    "prompt_eager",
    "prompt_graph",
    "startup_eager",
    "startup_graph",
    "decode_eager",
    "decode_graph",
)
OUT = results_root() / "research-baselines"
VIEWS = OUT / "views"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows():
    records = {}
    inputs = {}
    artifacts = {}
    commits = set()
    runtimes = set()
    profiles = set()
    tensor_profiles = set()
    checks = []
    full_checks = []
    verified = set()
    for path in sorted(OUT.glob("*.json")):
        if not path.name.startswith(
            ("training-", "inference-", "qualification-", "full_qualification-")
        ):
            continue
        data = json.loads(path.read_text())
        a = data["arguments"]
        p = data["provenance"]
        assert data["status"] in ("passed", "out_of_memory"), (
            path,
            data.get("traceback"),
        )
        assert p["gpu"] == "NVIDIA L40S" and p["native_cpp_executor"]
        commits.add((p["llt_commit"], p["tensor_commit"]))
        runtimes.add(tuple(sorted(p["runtime_binary_sha256"].items())))
        profiles.add(tuple(sorted(p["sources"].items())))
        tensor_profiles.add(tuple(sorted(p["tensor_sources"].items())))
        inputs[record_label(path)] = sha(path)
        for key, folder in (
            ("sources", "sources"),
            ("tensor_sources", "tensor-sources"),
        ):
            for name, digest in p[key].items():
                assert sha(OUT / folder / digest / Path(name).name) == digest
        coverage = data.get("coverage")
        if coverage:
            assert not coverage["fallbacks"]
            for artifact in coverage["artifacts"]:
                assert artifact["target"] == "sm_89"
                identity = (artifact["path"], artifact["sha256"])
                if identity not in verified:
                    assert sha(recorded_path(artifact["path"])) == artifact["sha256"]
                    verified.add(identity)
                artifacts[artifact["sha256"]] = artifact
        if a["phase"].endswith("qualification"):
            full = a["phase"] == "full_qualification"
            assert data["qualification_controls"]["deterministic_algorithms"] == full
            assert (
                data["qualification_controls"]["primary_performance_settings_changed"]
                is False
            )
            assert data["qualification_controls"]["grt_noise_disabled"] is True
            if full:
                assert (
                    data["qualification_controls"]["cublas_workspace_config"]
                    == ":4096:8"
                )
            assert (data["batch_size"], data["sequence_length"]) == (
                (4, 1024) if full else (2, 32)
            )
            assert data["status"] == "passed", (path, data.get("error"))
            (full_checks if a["phase"] == "full_qualification" else checks).append(path)
            for name, check in data["policy_checks"].items():
                if name.endswith("_cache"):
                    assert check["relative_l2"] < 0.03 and check["max_abs_error"] < 0.05
                elif name == "cross_backend":
                    assert (
                        check["gradient_error"]["maximum_parameter_relative_l2"] < 0.15
                    )
                else:
                    assert (
                        check["gradient_error"]["maximum_parameter_relative_l2"] < 1e-5
                    )
            continue
        c = data["config"]
        t = a["loops"]
        model = a["model"]
        assert data["batch_size"] == 4 and data["sequence_length"] == 1024
        assert (c["width"], c["heads"], c["vocab"], c["max_seq"], c["layers"]) == (
            768,
            12,
            50304,
            1025,
            12,
        )
        assert c["family"] == model and c["loops"] == t
        expected_depth = (
            6 * t + 6 if model == "uyoco" else 8 * t + 4 if model == "grt" else 12 * t
        )
        assert data["effective_depth"] == expected_depth and data["unique_layers"] == 12
        assert data["mlp_applications"] == (
            12 if model == "attention_only" else expected_depth
        )
        assert data["checkpoint_policy"] == a["policy"]
        key = (a["phase"], model, a["backend"], t, a["policy"])
        assert key not in records
        row = dict(
            phase=a["phase"],
            model=model,
            kv_rank=64 if model == "per_layer_latent" else "",
            backend=a["backend"],
            loops=t,
            checkpoint=a["policy"],
            status=data["status"],
            failed_stage=data.get("stage", "") if data["status"] != "passed" else "",
            parameter_count=data["parameter_count"],
            effective_depth=data["effective_depth"],
            unique_layers=12,
            attention_applications=data["attention_applications"],
            mlp_applications=data["mlp_applications"],
            gate_mlp_applications=t if model == "grt" else 0,
            recurrence_projection_applications=t if model == "grt" else 0,
            core_residual_dtype="bfloat16" if model == "grt" else "float32",
            measurement_origin="research-baseline",
            raw_record=record_label(path),
        )
        for metric in METRICS:
            if metric not in data:
                continue
            value = data[metric]
            assert len(value["gpu_samples_ms"]) == a["samples"] == 9
            assert all(math.isfinite(v) and v > 0 for v in value["gpu_samples_ms"])
            assert math.isclose(
                value["gpu_median_ms"],
                statistics.median(value["gpu_samples_ms"]),
                rel_tol=1e-12,
            )
            if metric.endswith("_graph"):
                assert value["repeats_per_sample"] == a["repeats"] == 3
            row[metric + "_gpu_ms"] = value["gpu_median_ms"]
            row[metric + "_peak_gib"] = (
                value.get(
                    "capture_peak_allocated_bytes", value.get("peak_allocated_bytes")
                )
                / 2**30
            )
            if "wall_median_ms" in value:
                row[metric + "_wall_ms"] = value["wall_median_ms"]
            if "capture_peak_reserved_bytes" in value:
                row[metric + "_reserved_gib"] = (
                    value["capture_peak_reserved_bytes"] / 2**30
                )
        if "training_graph" in data:
            counts = data["graph_step_counter"]
            assert counts["min"] == counts["max"] == counts["expected"] == 42
            assert data["all_parameter_gradients_finite"]
        row["cache_mib"] = data.get("cache_bytes", 0) / 2**20
        row["prepared_weight_mib"] = data.get("prepared_weight_bytes", 0) / 2**20
        row["fold_mib"] = data.get("fold_bytes", 0) / 2**20
        records[key] = row
    expected = {
        (phase, m, b, t, p)
        for m, b, t in itertools.product(FAMILIES, ("tensor", "torch"), range(1, 17))
        for phase, p in (
            ("training", "none"),
            ("training", "ac"),
            ("inference", "none"),
        )
    }
    assert set(records) == expected, sorted(expected - set(records))
    assert len(checks) == 10 and len(full_checks) == 10

    def qualification_key(path):
        arguments = json.loads(path.read_text())["arguments"]
        return (arguments["model"], arguments["loops"], arguments["backend"])

    assert {qualification_key(p) for p in checks} == {
        (m, t, "tensor") for m, t in itertools.product(FAMILIES, (4, 16))
    }
    assert {qualification_key(p) for p in full_checks} == {
        (m, 16, b) for m, b in itertools.product(FAMILIES, ("tensor", "torch"))
    }
    assert (
        len(profiles) == len(tensor_profiles) == len(commits) == len(runtimes) == 1
    ), "source/runtime changes during the run"
    original_runtime = json.loads(
        (PUBLISHED_RESULTS / "loop-sweep/run-manifest.json").read_text()
    )["runtime_binary_sha256"]
    assert {Path(k).name: v for k, v in next(iter(runtimes))} == original_runtime
    audit = dict(
        status="passed",
        case_count=len(records),
        passed=sum(r["status"] == "passed" for r in records.values()),
        out_of_memory=sum(r["status"] == "out_of_memory" for r in records.values()),
        qualification_count=len(checks),
        full_qualification_count=len(full_checks),
        unique_artifact_count=len(artifacts),
        numerical_source_commit=next(iter(commits))[0],
        tensor_commit=next(iter(commits))[1],
        runtime_binary_sha256=dict(next(iter(runtimes))),
        sources=dict(next(iter(profiles))),
        tensor_sources=dict(next(iter(tensor_profiles))),
        input_sha256=inputs,
        artifacts=list(artifacts.values()),
    )
    return records, audit


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cell(record, metric):
    if metric + "_gpu_ms" not in record:
        return "OOM" if record["failed_stage"] == metric else "— (earlier OOM)"
    return f"{record[metric+'_gpu_ms']:.2f} / {record[metric+'_peak_gib']:.2f}"


def tables(records):
    text = "Cells are **CUDA graph ms / capture peak allocated GiB**, per batch of four.\n\n"
    for backend in ("tensor", "torch"):
        text += f"### {backend.capitalize()} — additional families, all loop counts\n\n"
        text += "| T | Architecture | Train: none | Train: AC | Prompt | Startup | Cached decode |\n|---:|---|---:|---:|---:|---:|---:|\n"
        for loops in range(1, 17):
            for family in FAMILIES:
                cells = [
                    cell(
                        records[("training", family, backend, loops, p)],
                        "training_graph",
                    )
                    for p in ("none", "ac")
                ]
                cells += [
                    cell(records[("inference", family, backend, loops, "none")], m)
                    for m in ("prompt_graph", "startup_graph", "decode_graph")
                ]
                text += (
                    "| "
                    + str(loops)
                    + " | "
                    + LABELS[family]
                    + " | "
                    + " | ".join(cells)
                    + " |\n"
                )
        text += "\n"
    return text


def plots(records):
    for metric in ("training_graph", "prompt_graph", "decode_graph"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        for row, backend in enumerate(("tensor", "torch")):
            for index, family in enumerate(FAMILIES):
                policies = ("none", "ac") if metric == "training_graph" else ("none",)
                for policy in policies:
                    phase = "training" if metric == "training_graph" else "inference"
                    for ax, suffix in zip(axes[row], ("gpu_ms", "peak_gib")):
                        values = [
                            records[(phase, family, backend, t, policy)].get(
                                metric + "_" + suffix, float("nan")
                            )
                            for t in range(1, 17)
                        ]
                        ax.plot(
                            range(1, 17),
                            values,
                            color=f"C{index}",
                            linestyle="--" if policy == "ac" else "-",
                            marker="." if policy == "ac" else None,
                            label=LABELS[family]
                            + (" / " + policy if metric == "training_graph" else ""),
                        )
            axes[row, 0].set_ylabel(backend.capitalize() + " GPU latency (ms)")
            axes[row, 1].set_ylabel(backend.capitalize() + " capture peak GiB")
            axes[row, 0].legend(fontsize=7, ncol=2)
        for ax in axes.flat:
            ax.set_xlabel("T (architecture-specific work)")
            ax.set_xticks((1, 4, 8, 12, 16))
            ax.grid(alpha=0.2)
        fig.suptitle(
            "L40S · B4/S1024 · "
            + metric.replace("_", " ")
            + " · architecture kernel adaptations"
        )
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(VIEWS / (metric + "." + extension), dpi=170)
        plt.close(fig)


def main():
    records, audit = read_rows()
    measurement_manifest = OUT / "run-manifest.json"
    if measurement_manifest.exists():
        measured = json.loads(measurement_manifest.read_text())
        for key in (
            "case_count",
            "numerical_source_commit",
            "tensor_commit",
            "sources",
            "tensor_sources",
            "runtime_binary_sha256",
        ):
            assert measured[key] == audit[key]
        for name, digest in measured["immutable_files_sha256"].items():
            assert sha(OUT / name) == digest, name
        audit["measurement_manifest_sha256"] = sha(measurement_manifest)
    VIEWS.mkdir(parents=True, exist_ok=True)
    rows = [records[key] for key in sorted(records)]
    write_csv(VIEWS / "summary.csv", rows)
    (VIEWS / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (VIEWS / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (VIEWS / "tables.md").write_text(tables(records).rstrip() + "\n")
    original = PUBLISHED_RESULTS / "loop-sweep/views/combined.json"
    original_rows = json.loads(original.read_text())
    assert len(original_rows) == 672
    enriched_original = [
        dict(
            r,
            attention_applications=r["effective_depth"],
            mlp_applications=r["effective_depth"],
            gate_mlp_applications=0,
            recurrence_projection_applications=0,
            core_residual_dtype="float32",
        )
        for r in original_rows
    ]
    combined = enriched_original + rows
    write_csv(VIEWS / "all-architectures.csv", combined)
    (VIEWS / "all-architectures.json").write_text(json.dumps(combined, indent=2) + "\n")
    plots(records)
    view_sources = {}
    for relative in (
        "experiments/l40s/research_summary.py",
        "experiments/l40s/research_report.py",
        "experiments/l40s/latent_checkpoint_report.py",
        "experiments/l40s/runtime.py",
    ):
        source = ROOT / relative
        digest = sha(source)
        destination = VIEWS / "sources" / digest / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        view_sources[relative] = digest
    manifest = dict(
        kind="regenerated architecture-family view",
        view_generator_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        view_generator_worktree_dirty=bool(
            set(
                subprocess.check_output(
                    ["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True
                ).splitlines()
            )
            & set(view_sources)
        ),
        view_generator_sources=view_sources,
        status="passed",
        case_count=len(rows),
        joined_case_count=len(combined),
        tensor_repository="https://github.com/kreasof-ai/tensor",
        numerical_source_commit=audit["numerical_source_commit"],
        tensor_commit=audit["tensor_commit"],
        runtime_binary_sha256=audit["runtime_binary_sha256"],
        sources=audit["sources"],
        tensor_sources=audit["tensor_sources"],
        audit_sha256=sha(VIEWS / "audit.json"),
        measurement_manifest_sha256=audit.get("measurement_manifest_sha256"),
        original_combined_sha256=sha(original),
        original_measurement_manifest_sha256={
            name: sha(PUBLISHED_RESULTS / name / "run-manifest.json")
            for name in ("loop-baseline", "loop-sweep")
        },
    )
    (VIEWS / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "research audit passed:",
        len(rows),
        "new cases;",
        len(combined),
        "joined;",
        len(audit["artifacts"]),
        "artifacts",
    )


if __name__ == "__main__":
    main()
