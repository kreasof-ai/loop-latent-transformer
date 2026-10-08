"""Four architectures at identical B4/S1024; fresh process per GPU case.

Fixed-depth MLP width W*(6*T-2) matches the independent 12*T-layer stack
exactly, including embeddings and output. All parameters participate in compute.
"""

# Support both direct script execution and python -m experiments.l40s.<module>.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l40s.runtime import results_root, require_run_directory
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from tensor_torch.llt import Operators, AdamW
import tensor_torch.bridge as _bridge
from model import Config
from model.tensor_backend import BackendTransformer
from inference.prepared import prepare
from experiments.l40s.timing import measure

ROOT = Path(__file__).resolve().parents[2]
OUT = results_root() / 'loop-baseline'
TENSOR = Path(os.environ.get('TENSOR_CHECKOUT', ROOT.parent / 'tensor'))
MODELS = ('llt', 'naive_loop', 'stacked', 'fixed_depth')


def config(model, loops, small=False):
    w, h, layers, vocab, seq = (128, 2, 2, 256, 33) if small else (768, 12, 12, 50304, 1025)
    return Config(kind='llt' if model == 'llt' else 'naive', width=w, heads=h,
                  layers=layers * loops if model == 'stacked' else layers,
                  loops=loops if model in ('llt', 'naive_loop') else 1,
                  rank=64, vocab=vocab, max_seq=seq, gelu='none',
                  mlp_width=w * (6 * loops - 2) if model == 'fixed_depth' else 4 * w)


def parameters(c):
    embedding = 2 * c.vocab * c.width + c.max_seq * c.width
    blocks = c.layers * (c.loops if c.untied else 1)
    block = 2 * c.width * (c.mlp_width or 4 * c.width) + 2 * c.width**2
    block += 2 * c.width**2 if c.kind == 'naive' else 2 * c.width * c.rank
    return embedding + blocks * block + (c.width * c.rank if c.kind == 'llt' else 0)


def save(stem, out):
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in ('experiments/l40s/loop_sweep.py', 'model/reference.py', 'model/tensor_backend.py',
                 'model/__init__.py', 'inference/prepared.py', 'experiments/l40s/timing.py', 'experiments/l40s/runtime.py'):
        p = ROOT / name
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        dest = OUT / 'sources' / sha / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        sources[str(p.relative_to(ROOT))] = sha
    out['provenance'] = dict(
        llt_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        tensor_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TENSOR, text=True).strip(),
        tensor_repository='https://github.com/kreasof-ai/tensor', sources=sources,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(), native_cpp_executor=_bridge._executor is not None,
        driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
        nvrtc=os.environ.get('TENSOR_NVRTC_HOME'))
    tmp = OUT / (stem + '.json.tmp')
    tmp.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    tmp.replace(OUT / (stem + '.json'))


def graph_measure(fn, samples, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = fn()
            del result
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = fn()
    capture_peak = torch.cuda.max_memory_allocated()
    capture_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / repeats)
    out = dict(gpu_samples_ms=times, gpu_median_ms=statistics.median(times),
               baseline_allocated_bytes=baseline, capture_peak_allocated_bytes=capture_peak,
               capture_peak_reserved_bytes=capture_reserved,
               replay_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               replay_peak_reserved_bytes=torch.cuda.max_memory_reserved(), repeats_per_sample=repeats)
    del graph, result
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    return out


def correctness(a):
    c = config(a.model, a.loops, small=True)
    ops = Operators(OUT / 'artifacts')
    m = BackendTransformer(c, ops).cuda()
    ref = BackendTransformer(c).cuda()
    ref.load_state_dict(m.state_dict())
    assert sum(p.numel() for p in m.parameters()) == parameters(c)
    if a.model == 'fixed_depth':
        assert parameters(c) == parameters(config('stacked', a.loops, small=True))
    x = torch.randint(c.vocab, (2, 32), device='cuda')
    y = torch.randint(c.vocab, x.shape, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out, expected = m(x), ref(x)
        loss = ops.cross_entropy(out.flatten(0, 1), y.flatten())
        reference_loss = torch.nn.functional.cross_entropy(expected.float().flatten(0, 1), y.flatten())
    loss.backward()
    reference_loss.backward()
    errors = {}
    for (name, p), (other, q) in zip(m.named_parameters(), ref.named_parameters()):
        assert name == other and p.grad is not None and q.grad is not None, name
        assert torch.isfinite(p.grad).all().item(), name
        rel = ((p.grad - q.grad).norm() / q.grad.norm().clamp_min(1e-12)).item()
        assert rel < .10, (name, rel)
        errors[name] = rel
    relative_output = ((out.float()-expected.float()).norm()/expected.float().norm()).item()
    assert relative_output < .04, relative_output
    assert abs(loss.item() - reference_loss.item()) < .15
    m.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        full = m(x)
        _, state = m.prefill(x[:, :-1], capacity=33, last_logits=True)
        cached = m.decode_token(x[:, -1:], state)
        cache_error = (cached.float() - full[:, -1:].float()).abs().max().item()
        assert cache_error < .05, cache_error
    assert not ops.report['fallbacks']
    return dict(status='passed',config=asdict(c),parameter_count=parameters(c),
                output_relative_l2=relative_output,gradient_relative_l2=errors,
                loss=loss.item(),reference_loss=reference_loss.item(),cached_max_error=cache_error,
                coverage=ops.report)


def run(a, out):
    c = config(a.model, a.loops)
    out.update(config=asdict(c), parameter_count=parameters(c),
               effective_depth=c.layers*c.loops, unique_layers=c.layers,
               stacked_parameter_target=parameters(config('stacked', a.loops)),
               sequence_length=1024,batch_size=4,checkpoint_policy='none')
    ops = Operators(OUT / 'artifacts', cache_inference_weights=a.phase=='inference') if a.backend=='tensor' else None
    out['stage'] = 'model_initialization'
    m = BackendTransformer(c, ops).cuda()
    assert sum(p.numel() for p in m.parameters()) == parameters(c)
    if a.model == 'fixed_depth':
        assert parameters(c) == out['stacked_parameter_target']
    x = torch.randint(c.vocab, (4, 1024), device='cuda')
    y = torch.randint(c.vocab, x.shape, device='cuda')
    out['parameter_bytes'] = parameters(c)*4
    if a.phase == 'training':
        m.train()
        groups = [dict(params=list(m.parameters()),weight_decay=.1)]
        opt = AdamW(groups,ops,lr=.0006,betas=(.9,.95),max_norm=1.) if ops else torch.optim.AdamW(
            groups,lr=.0006,betas=(.9,.95),fused=True,capturable=True)
        losses = []
        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                loss = m.loss(x,y)
            loss.backward()
            if not ops:
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step()
            losses.append(loss.detach())
            return loss.detach()
        out['stage'] = 'training_eager'
        out['training_eager'] = measure(step,a.samples)
        out['losses_eager'] = [float(loss.item()) for loss in losses]
        assert all(torch.isfinite(loss).item() for loss in losses)
        assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in m.parameters())
        out['all_parameter_gradients_finite'] = True
        out['stage'] = 'training_graph'
        out['training_graph'] = graph_measure(step,a.samples,a.repeats)
        counts = [int(state['step'].item()) for state in opt.state.values()]
        expected = 3+a.samples+3+a.samples*a.repeats
        assert min(counts) == max(counts) == expected, (min(counts),max(counts),expected)
        assert torch.isfinite(losses[-1]).item()
        out['graph_step_counter'] = dict(min=min(counts),max=max(counts),expected=expected)
        out['graph_last_loss'] = losses[-1].item()
        out['training_protocol'] = 'Full logits/loss, backward, global gradient clip=1, AdamW FP32 states; no activation checkpointing; Torch fused optimizer.'
    else:
        m.eval()
        if not ops:
            prepare(m)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            def prompt():
                return m(x,last_logits=True)
            out['stage'] = 'prompt_eager'
            out['prompt_eager'] = measure(prompt,a.samples)
            out['stage'] = 'prompt_graph'
            out['prompt_graph'] = graph_measure(prompt,a.samples,a.repeats)
            def startup():
                return m.prefill(x,capacity=1025,last_logits=True)
            out['stage'] = 'startup_eager'
            out['startup_eager'] = measure(startup,a.samples)
            out['stage'] = 'startup_graph'
            out['startup_graph'] = graph_measure(startup,a.samples,a.repeats)
            _, state = startup()
            token = torch.randint(c.vocab,(4,1),device='cuda')
            out['cache_bytes'] = sum(cache.nbytes for cache in state['caches'])
            out['fold_bytes'] = sum(t.numel()*t.element_size() for pair in state['folds'] for t in pair)
            def reset():
                m.rewind(state,1024)
            def decode():
                return m.decode_token(token,state)
            out['stage'] = 'decode_eager'
            out['decode_eager'] = measure(decode,a.samples,reset)
            def captured_decode():
                reset()
                return decode()
            out['stage'] = 'decode_graph'
            out['decode_graph'] = graph_measure(captured_decode,a.samples,a.repeats)
            assert torch.isfinite(decode_reset(m,token,state)).all().item()
            entries = ops.inference_weight_casts.values() if ops else m.prepared_weight_casts.values()
            out['prepared_weight_bytes'] = sum(entry[2 if ops else 1].numel()*entry[2 if ops else 1].element_size() for entry in entries)
        out['inference_protocol'] = 'Prepared BF16 linear weights, FP32 master/embedding/residual; prompt=last logits without KV stores; startup includes KV allocation/copy and LLT fold rebuild; decode=one supplied token with 1024-token history; captured rewind included.'
    if ops:
        assert not ops.report['fallbacks']
        out['coverage'] = ops.report
    out['status'] = 'passed'
    out.pop('stage',None)


def decode_reset(m,token,state):
    m.rewind(state,1024)
    return m.decode_token(token,state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=('training','inference','correctness'))
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--loops',type=int,required=True)
    p.add_argument('--backend',choices=('tensor','torch'),default='tensor')
    p.add_argument('--samples',type=int,default=9)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    require_run_directory()
    assert 1<=a.loops<=16 and a.samples>=3 and a.repeats>=1
    assert _bridge._executor is not None, 'qualified native Tensor executor must be installed'
    torch.manual_seed(9505)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    stem=f'{a.phase}-{a.model}-{a.backend}-t{a.loops:02}'
    out=dict(status='running',arguments=vars(a),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    start=time.monotonic()
    try:
        if a.phase=='correctness':
            out.update(correctness(a))
        else:
            run(a,out)
    except torch.cuda.OutOfMemoryError as e:
        out.update(status='out_of_memory',error=str(e),
                   failed_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   failed_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as e:
        out.update(status='error',error=str(e),traceback=traceback.format_exc())
    out['duration_seconds']=time.monotonic()-start
    save(stem,out)
    print(stem,out['status'],out.get('stage',''),f"{out['duration_seconds']:.1f}s",flush=True)
    if out['status']=='error':
        raise SystemExit(1)


if __name__=='__main__':
    main()
