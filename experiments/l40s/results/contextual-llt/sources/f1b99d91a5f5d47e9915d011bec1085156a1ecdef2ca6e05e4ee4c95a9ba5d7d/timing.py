"""CUDA event, wall-clock and allocator sampling for the loop sweep."""
import statistics
import time
import torch


def measure(fn,samples=9,reset=None):
 for _ in range(3):
  if reset:reset()
  result=fn();del result
 torch.cuda.synchronize()
 gpu=[];wall=[];memory=[]
 for _ in range(samples):
  if reset:reset()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  t=time.perf_counter();a.record();result=fn();b.record();b.synchronize()
  gpu.append(a.elapsed_time(b));wall.append((time.perf_counter()-t)*1000)
  memory.append(dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),allocated_after_bytes=torch.cuda.memory_allocated()))
  del result
 return dict(gpu_samples_ms=gpu,wall_samples_ms=wall,gpu_median_ms=statistics.median(gpu),wall_median_ms=statistics.median(wall),
             memory_samples=memory,peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in memory))
