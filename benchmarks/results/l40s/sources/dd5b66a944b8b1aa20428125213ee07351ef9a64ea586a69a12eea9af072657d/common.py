"""Reproducibility and synchronized CUDA measurement helpers."""
import hashlib
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'benchmarks/results/l40s'
TENSOR = ROOT.parent / 'tensor'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    sources = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    for name, sha in sources.items():
        snapshot = OUT / 'sources' / sha / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists(): snapshot.write_bytes((Path(__file__).parent / name).read_bytes())
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                total_device_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                packages={name:importlib.metadata.version(name) for name in
                          ('torch','numpy','tensor-workspace','tensor-torch','tilelang','apache-tvm-ffi')},
                nvrtc_bootstrap_sha256=digest(TENSOR/'build/nvrtc-12.9/bootstrap.json'),
                python=platform.python_version(),
                project_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                tensor_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TENSOR, text=True).strip(),
                nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
                sources=sources)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def setup(seed=0):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def timing(fn, samples=9, repeats=10, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - before) * 1000 / repeats)
        gpu.append(start.elapsed_time(end) / repeats)
    return dict(gpu_median_ms=statistics.median(gpu), wall_median_ms=statistics.median(wall),
                gpu_samples_ms=gpu, wall_samples_ms=wall)


def graph_timing(fn, samples=9, repeats=20):
    # Captured replay removes Python submission gaps from CUDA kernel timing.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    measured = timing(graph.replay, samples, repeats)
    del result, graph
    torch.cuda.synchronize()
    gc.collect()
    # cuBLAS caches ~8 MiB per stream. Clear inactive graph/side-stream
    # workspaces so later methods do not inherit unrelated allocator storage.
    torch._C._cuda_clearCublasWorkspaces()
    return measured


def memory(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    return result, dict(baseline_allocated_bytes=base,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        incremental_peak_bytes=torch.cuda.max_memory_allocated() - base)
