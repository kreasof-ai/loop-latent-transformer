"""Fresh contextual LLT profiles: ranks 32/64/128, none/AC, no LAC."""

import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "full_qualification":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from tensor_torch.llt import Operators
from model.contextual_llt import ContextualLLT
from experiments.l40s import loop_sweep as base
from experiments.l40s.latent_checkpoint_sweep import gradient_metrics
from experiments.l40s.runtime import ROOT, results_root, require_run_directory

OUT = results_root() / "contextual-llt"
ORIGINAL_CONFIG, ORIGINAL_PARAMETERS = base.config, base.parameters
SOURCES = (
    "model/contextual_llt.py",
    "model/reference.py",
    "model/tensor_backend.py",
    "model/__init__.py",
    "inference/prepared.py",
    "experiments/l40s/loop_sweep.py",
    "experiments/l40s/contextual_llt_sweep.py",
    "experiments/l40s/timing.py",
    "experiments/l40s/runtime.py",
    "experiments/l40s/latent_checkpoint_sweep.py",
    "model/checkpointing.py",
)


def config(model, loops, small=False, rank=64):
    c = ORIGINAL_CONFIG(model, loops, small)
    if model == "llt":
        c.kind = "layerwise"
        c.rank = rank
    return c


def parameters(c):
    if c.kind != "layerwise":
        return ORIGINAL_PARAMETERS(c)
    return (
        2 * c.vocab * c.width
        + c.max_seq * c.width
        + c.layers
        * (
            2 * c.width * (c.mlp_width or 4 * c.width)
            + 2 * c.width**2
            + 3 * c.width * c.rank
        )
    )


def backward(model, x, y, policy):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(
        SDPBackend.FLASH_ATTENTION
    ):
        loss = model.loss(x, y, policy=policy)
    loss.backward()
    assert torch.isfinite(loss).item()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )
    gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return loss.item(), gradients


def qualify(a, full=False):
    torch.use_deterministic_algorithms(full)
    c = config("llt", a.loops, not full, a.rank)
    batch, seq = (4, 1024) if full else (2, 32)
    x = torch.randint(c.vocab, (batch, seq), device="cuda")
    y = torch.randint(c.vocab, x.shape, device="cuda")
    checks, references = {}, {}
    weights, coverage = None, None
    for backend in ((a.backend,) if full else ("torch", "tensor")):
        ops = Operators(OUT / "artifacts") if backend == "tensor" else None
        model = ContextualLLT(c, ops).cuda()
        if weights is None:
            weights = {name: p.detach().cpu() for name, p in model.state_dict().items()}
        else:
            model.load_state_dict(weights)
        assert sum(p.numel() for p in model.parameters()) == parameters(c)
        loss, reference = backward(model, x, y, "none")
        if full:
            value, repeated = backward(model, x, y, "none")
            metrics = gradient_metrics(repeated, reference)
            assert (
                value == loss and metrics["maximum_parameter_relative_l2"] < 1e-5
            ), metrics
            checks[backend + "_repeat_none"] = dict(loss=value, gradient_error=metrics)
            del repeated
        value, actual = backward(model, x, y, "ac")
        metrics = gradient_metrics(actual, reference)
        assert (
            value == loss and metrics["maximum_parameter_relative_l2"] < 1e-5
        ), metrics
        checks[backend + "_ac"] = dict(loss=value, gradient_error=metrics)
        del actual
        references[backend] = reference
        model.eval()
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            logits = model(x, last_logits=True)
            _, state = model.prefill(x[:, :-1], capacity=c.max_seq, last_logits=True)
            cached = model.decode_token(x[:, -1:], state)
            error = cached.float() - logits.float()
            maximum = error.abs().max().item()
            relative = (error.norm() / logits.float().norm().clamp_min(1e-12)).item()
            assert maximum < 0.05 and relative < 0.03, (maximum, relative)
            assert len(state["caches"]) == model.cache_bank_count
            checks[backend + "_cache"] = dict(
                max_abs_error=maximum,
                relative_l2=relative,
                cache_banks=len(state["caches"]),
            )
        if ops:
            coverage = ops.report
            assert not coverage["fallbacks"]
        del model
        torch.cuda.empty_cache()
    if not full:
        metrics = gradient_metrics(references["tensor"], references["torch"])
        assert metrics["maximum_parameter_relative_l2"] < 0.15, metrics
        checks["cross_backend"] = dict(gradient_error=metrics)
    return dict(
        status="passed",
        config=base.asdict(c),
        parameter_count=parameters(c),
        batch_size=batch,
        sequence_length=seq,
        policy_checks=checks,
        coverage=coverage,
        qualification_controls=dict(
            deterministic_algorithms=full,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            primary_performance_settings_changed=False,
        ),
    )


def run(a, out):
    base.OUT = OUT
    base.config = lambda model, loops, small=False: config(model, loops, small, a.rank)
    base.parameters = parameters

    class ProfileModel(ContextualLLT):
        def loss(self, tokens, targets, policy=None, chunk_size=0):
            return super().loss(
                tokens, targets, a.policy if policy is None else policy, chunk_size
            )

    base.BackendTransformer = ProfileModel
    try:
        base.run(a, out)
    finally:
        out["checkpoint_policy"] = a.policy
    out["checkpoint_boundary"] = (
        "full Transformer block including latent construction when required"
        if a.policy == "ac"
        else None
    )
    out["training_protocol"] = (
        "Full-token loss/backward, clip=1, AdamW FP32 states; policy=" + a.policy
    )
    if a.phase == "inference":
        c = config("llt", a.loops, rank=a.rank)
        banks = ContextualLLT.bank_count(c)
        expected = 2 * 4 * 1025 * a.rank * banks + (
            8 * 4 * banks if a.backend == "tensor" else 0
        )
        assert out["cache_bytes"] == expected, (out["cache_bytes"], expected)
        out.update(cache_banks=banks, expected_cache_bytes=expected)


def save(stem, out):
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in SOURCES:
        path = ROOT / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot = OUT / "sources" / digest / path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources[name] = digest
    runtime = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (base.TENSOR / "packages/tensor-torch/src/tensor_torch").glob(
            "*.so"
        )
    }
    out["architecture"] = ContextualLLT.architecture()
    out["provenance"] = dict(
        llt_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        tensor_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=base.TENSOR, text=True
        ).strip(),
        tensor_repository="https://github.com/kreasof-ai/tensor",
        sources=sources,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        native_cpp_executor=base._bridge._executor is not None,
        runtime_binary_sha256=runtime,
        nvrtc=os.environ.get("TENSOR_NVRTC_HOME"),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
    )
    temporary = OUT / (stem + ".json.tmp")
    temporary.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUT / (stem + ".json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "phase",
        choices=("training", "inference", "qualification", "full_qualification"),
    )
    p.add_argument("--rank", type=int, choices=(32, 64, 128), required=True)
    p.add_argument("--loops", type=int, required=True)
    p.add_argument("--backend", choices=("tensor", "torch"), default="tensor")
    p.add_argument("--policy", choices=("none", "ac"), default="none")
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    a.model = "llt"
    require_run_directory()
    assert 1 <= a.loops <= 16 and a.samples >= 3 and a.repeats >= 1
    assert a.phase != "inference" or a.policy == "none"
    assert base._bridge._executor is not None
    torch.manual_seed(9505)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    stem = f"{a.phase}-llt-{a.backend}-r{a.rank:03}-t{a.loops:02}-{a.policy}"
    out = dict(
        status="running",
        arguments=vars(a),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    start = time.monotonic()
    try:
        if a.phase.endswith("qualification"):
            out.update(qualify(a, a.phase == "full_qualification"))
        else:
            run(a, out)
    except torch.cuda.OutOfMemoryError as error:
        out.update(
            status="out_of_memory",
            error=str(error),
            failed_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    except Exception as error:
        out.update(status="error", error=str(error), traceback=traceback.format_exc())
    out["duration_seconds"] = time.monotonic() - start
    save(stem, out)
    print(
        stem,
        out["status"],
        out.get("stage", ""),
        f"{out['duration_seconds']:.1f}s",
        flush=True,
    )
    if out["status"] == "error" or (
        a.phase.endswith("qualification") and out["status"] != "passed"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
