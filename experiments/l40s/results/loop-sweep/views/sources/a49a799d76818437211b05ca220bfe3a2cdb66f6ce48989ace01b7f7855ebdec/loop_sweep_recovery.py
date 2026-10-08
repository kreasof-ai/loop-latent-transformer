"""Controlled retry: release eager gradients before creating a private graph pool.

Model, kernels, optimizer, warmup counts, precision and batch geometry are
unchanged. Keep these supplementary measurements separate from the primary grid.
"""

# Support both direct script execution and python -m experiments.l40s.<module>.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l40s.runtime import require_run_directory
import argparse
import gc
import hashlib
import inspect
import statistics
import time
import traceback
import torch
from experiments.l40s import loop_sweep as base


def graph_measure(fn,samples,repeats):
    opt=inspect.getclosurevars(fn).nonlocals['opt']
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result=fn()
            del result
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    before=torch.cuda.memory_allocated()
    opt.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    baseline=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        result=fn()
    capture_peak=torch.cuda.max_memory_allocated()
    capture_reserved=torch.cuda.max_memory_reserved()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(samples):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end)/repeats)
    out=dict(gpu_samples_ms=times,gpu_median_ms=statistics.median(times),
             baseline_before_gradient_release_bytes=before,baseline_allocated_bytes=baseline,
             released_gradient_bytes=before-baseline,capture_peak_allocated_bytes=capture_peak,
             capture_peak_reserved_bytes=capture_reserved,
             replay_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
             replay_peak_reserved_bytes=torch.cuda.max_memory_reserved(),repeats_per_sample=repeats)
    del graph,result
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=base.MODELS,required=True)
    parser.add_argument('--backend',choices=('tensor','torch'),required=True)
    parser.add_argument('--loops',type=int,required=True)
    a=parser.parse_args()
    require_run_directory()
    a.phase='training';a.samples=9;a.repeats=3
    assert 1<=a.loops<=16 and base._bridge._executor is not None
    original_out=base.OUT
    base.OUT=original_out/'capture-recovery'
    base.graph_measure=graph_measure
    torch.manual_seed(9505);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    stem=f'training-{a.model}-{a.backend}-t{a.loops:02}'
    source=Path(__file__)
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    snap=base.OUT/'sources'/sha/source.name
    snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(source.read_bytes())
    out=dict(status='running',arguments=vars(a),
             capture_setup=dict(protocol='Clear eager gradients, collect garbage, and empty cached allocations after graph warmup, before capture',
                                source_sha256=sha,source_snapshot=str(snap.relative_to(base.OUT))))
    start=time.monotonic()
    try:
        base.run(a,out)
    except torch.cuda.OutOfMemoryError as e:
        out.update(status='out_of_memory',error=str(e),failed_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   failed_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as e:
        out.update(status='error',error=str(e),traceback=traceback.format_exc())
    out['duration_seconds']=time.monotonic()-start
    base.save(stem,out)
    print('recovery',stem,out['status'],out.get('stage',''),flush=True)
    if out['status']=='error':
        raise SystemExit(1)


if __name__=='__main__':
    main()
