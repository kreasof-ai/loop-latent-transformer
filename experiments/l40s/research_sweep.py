"""Additional architecture-family profiles under the existing L40S protocol."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Deterministic correctness controls are separate from primary timings.
import os

if len(sys.argv) > 1 and sys.argv[1] == "full_qualification":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
from dataclasses import asdict
import hashlib
import json
import subprocess
import time
import traceback
import torch
from tensor_torch.recurrent import RecurrentOperators
from model.research_baselines import (
    FAMILIES,
    ResearchTransformer,
    ResearchConfig,
    make_config,
)
from experiments.l40s import loop_sweep as base
from experiments.l40s.latent_checkpoint_sweep import gradient_metrics
from experiments.l40s.runtime import ROOT, results_root, require_run_directory

OUT = results_root() / "research-baselines"
ORIGINAL_CONFIG = base.config
ORIGINAL_PARAMETERS = base.parameters
SOURCES = (
    "model/research_baselines.py",
    "experiments/l40s/research_sweep.py",
    "model/reference.py",
    "model/tensor_backend.py",
    "inference/prepared.py",
    "experiments/l40s/loop_sweep.py",
    "experiments/l40s/timing.py",
    "experiments/l40s/runtime.py",
)


def parameters(c):
    if not isinstance(c, ResearchConfig):
        return ORIGINAL_PARAMETERS(c)
    w, l, f = c.width, c.layers, c.mlp_width
    embedding = 2 * c.vocab * w + (0 if c.family == "uyoco" else c.max_seq * w)
    if c.family == "uyoco":
        return embedding + (3 * l + 2) * w * w + 3 * l * w * f + (2 * l + 2) * w
    if c.family == "grt":
        return embedding + l * (2 * w * f + 4 * w * w) + 5 * w * w + (2 * l + 6) * w
    if c.family == "per_layer_latent":
        return embedding + l * (2 * w * f + 2 * w * w + 3 * w * c.rank)
    return embedding + l * (2 * w * f + 4 * w * w)


def config(family, loops, small=False):
    return (
        make_config(family, loops, small)
        if family in FAMILIES
        else ORIGINAL_CONFIG(family, loops, small)
    )


def backward(model, tokens, targets, policy):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model.loss(tokens, targets, policy=policy)
    loss.backward()
    gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters()}
    assert all(torch.isfinite(grad).all().item() for grad in gradients.values())
    value = loss.item()
    model.zero_grad(set_to_none=True)
    return value, gradients


def qualify(a, full=False):
    torch.use_deterministic_algorithms(full)
    c = config(a.model, a.loops, small=not full)
    batch, seq = (4, 1024) if full else (2, 32)
    x = torch.randint(c.vocab, (batch, seq), device="cuda")
    y = torch.randint(c.vocab, x.shape, device="cuda")
    backends = (a.backend,) if full else ("torch", "tensor")
    results = {}
    references = {}
    coverage = None
    weights = None
    for backend in backends:
        ops = RecurrentOperators(OUT / "artifacts") if backend == "tensor" else None
        model = ResearchTransformer(c, ops).cuda().eval()
        if weights is None:
            weights = {n: p.detach().cpu() for n, p in model.state_dict().items()}
        else:
            model.load_state_dict(weights)
        assert sum(p.numel() for p in model.parameters()) == parameters(c)
        loss, reference = backward(model, x, y, "none")
        if full:
            repeat_loss, repeat = backward(model, x, y, "none")
            metrics = gradient_metrics(repeat, reference)
            assert (
                repeat_loss == loss and metrics["maximum_parameter_relative_l2"] < 1e-5
            ), metrics
            results[backend + "_repeat_none"] = dict(
                loss=repeat_loss, gradient_error=metrics
            )
            del repeat
        ac_loss, actual = backward(model, x, y, "ac")
        metrics = gradient_metrics(actual, reference)
        assert (
            ac_loss == loss and metrics["maximum_parameter_relative_l2"] < 1e-5
        ), metrics
        results[backend + "_ac"] = dict(loss=ac_loss, gradient_error=metrics)
        del actual
        references[backend] = (loss, reference)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            full_logits = model(x, last_logits=True)
            _, state = model.prefill(x[:, :-1], capacity=c.max_seq, last_logits=True)
            cached = model.decode_token(x[:, -1:], state)
            rel = (
                (cached.float() - full_logits.float()).norm()
                / full_logits.float().norm().clamp_min(1e-12)
            ).item()
            maximum = (cached.float() - full_logits.float()).abs().max().item()
            assert rel < 0.03 and maximum < 0.05, (rel, maximum)
            results[backend + "_cache"] = dict(relative_l2=rel, max_abs_error=maximum)
        if ops:
            coverage = ops.report
            assert not coverage["fallbacks"]
        del model
        torch.cuda.empty_cache()
    if not full:
        cross = gradient_metrics(references["tensor"][1], references["torch"][1])
        assert cross["maximum_parameter_relative_l2"] < 0.15, cross
        assert abs(references["tensor"][0] - references["torch"][0]) < 0.15
        results["cross_backend"] = dict(gradient_error=cross)
    return dict(
        status="passed",
        config=asdict(c),
        batch_size=batch,
        sequence_length=seq,
        parameter_count=parameters(c),
        policy_checks=results,
        coverage=coverage,
        qualification_controls=dict(
            deterministic_algorithms=full,
            grt_noise_disabled=True,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            primary_performance_settings_changed=False,
        ),
    )


def run(a, out):
    base.OUT = OUT
    base.config = config
    base.parameters = parameters
    base.Operators = RecurrentOperators

    class ProfileModel(ResearchTransformer):
        def __init__(self, c, ops=None):
            super().__init__(c, ops)
            out.update(
                effective_depth=self.effective_depth,
                unique_layers=c.layers,
                attention_applications=self.effective_depth,
                mlp_applications=self.mlp_applications,
            )

        def loss(self, tokens, targets, policy=None, chunk_size=0):
            return super().loss(
                tokens, targets, a.policy if policy is None else policy, chunk_size
            )

    base.BackendTransformer = ProfileModel
    try:
        base.run(a, out)
    finally:
        out["checkpoint_policy"] = a.policy
    out["checkpoint_region"] = (
        "each Transformer block; attention and FFN updates separately for attention_only; GRT recurrence gates remain outside block AC"
    )
    out["training_protocol"] = (
        "Full-token loss/backward, clip=1, AdamW FP32 state; policy=" + a.policy
    )
    out["implementation"] = dict(
        torch_window_attention="compiled FlexAttention",
        tensor_window_attention="native window/shared-local kernels",
        tensor_shared_decode="two-bank single-softmax kernel without KV concatenation",
        grt_training_rng="Torch CUDA normal RNG on both backends, included in timing; gate/state arithmetic uses the selected backend",
        grt_inference_noise=0.0,
        cache_profile="one supplied token after a fixed prefix; rewind before replay",
    )


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
    tensor_sources = {}
    for path in (base.TENSOR / "packages/tensor-torch/src/tensor_torch").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name not in (
            "recurrent.py",
            "llt_window.py",
            "recurrent_ops.py",
            "shared_decode.py",
        ):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot = OUT / "tensor-sources" / digest / path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        tensor_sources[str(path.relative_to(base.TENSOR))] = digest
    out["provenance"] = dict(
        llt_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        sources=sources,
        tensor_sources=tensor_sources,
        tensor_repository="https://github.com/kreasof-ai/tensor",
        tensor_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=base.TENSOR, text=True
        ).strip(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        native_cpp_executor=base._bridge._executor is not None,
        nvrtc=os.environ.get("TENSOR_NVRTC_HOME"),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        runtime_binary_sha256={
            str(path.relative_to(base.TENSOR)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (base.TENSOR / "packages/tensor-torch/src/tensor_torch").glob(
                "*.so"
            )
        },
    )
    temp = OUT / (stem + ".json.tmp")
    temp.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    temp.replace(OUT / (stem + ".json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("training", "inference", "qualification", "full_qualification"),
    )
    parser.add_argument("--model", choices=FAMILIES, required=True)
    parser.add_argument("--loops", type=int, required=True)
    parser.add_argument("--backend", choices=("tensor", "torch"), default="tensor")
    parser.add_argument("--policy", choices=("none", "ac"), default="none")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=3)
    a = parser.parse_args()
    require_run_directory()
    assert 1 <= a.loops <= 16 and (a.phase != "inference" or a.policy == "none")
    assert base._bridge._executor is not None
    torch.manual_seed(9505)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    stem = f"{a.phase}-{a.model}-{a.backend}-t{a.loops:02}-{a.policy}"
    out = dict(
        status="running",
        arguments=vars(a),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    started = time.monotonic()
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
            failed_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except Exception as error:
        out.update(status="error", error=str(error), traceback=traceback.format_exc())
    out["duration_seconds"] = time.monotonic() - started
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
