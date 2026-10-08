"""Measure LLT attention tradeoffs using the adjacent Tensor Vulkan runtime.

Run with Tensor's Python environment. This is a single-token attention
microbenchmark, not a trained LLT model or an end-to-end generation benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np


def emit(args, body):
    declarations = [f'{name}: T.Tensor(({size},), "{dtype}")' for name, size, dtype in args]
    return ('import tilelang.language as T\n\n@T.prim_func\ndef kernel('
            + ', '.join(declarations) + '):\n'
            + '\n'.join('    ' + line for line in body.splitlines())
            + '\n\ndef tensor_export(): return {"kernel": kernel}\n')


def expansion_source(s, h, d, r):
    from tensor.compiler.webgpu_lowering import register_matmul_schedule
    body = register_matmul_schedule(
        s, r, 2*h*d,
        'value',
        f'T.cast(w[(bx*32+i)*{r}+tile*32+j], "float32")',
        tile_m=16, tile_n=32, tile_k=32, threads=128, dot_width=4)
    return emit([('x', s*r, 'float16'), ('w', 2*h*d*r, 'float16'),
                 ('out', s*2*h*d, 'float16')], body)


def attention_source(s, h, width, scale_dim, *, explicit=False, partitions=1):
    tile = 16
    # Same online-softmax schedule for explicit and absorbed attention. The
    # absorbed variant reads the shared C for both K and V, without expansion.
    stride = 2*h*width if explicit else width
    key = f'kv[(tile*{tile}+i)*{stride}+head*{width}+j]' if explicit else f'c[(tile*{tile}+i)*{width}+j]'
    val = f'kv[(tile*{tile}+i)*{stride}+{h*width}+head*{width}+j]' if explicit else key
    body = f'''with T.Kernel({h}, threads=128) as head:
    dots = T.alloc_fragment(({tile}, {width}), "float32")
    scores = T.alloc_fragment(({tile},), "float32")
    products = T.alloc_fragment(({tile}, {width}), "float32")
    partial = T.alloc_fragment(({width},), "float32")
    result = T.alloc_fragment(({width},), "float32")
    maximum = T.alloc_fragment((1,), "float32")
    previous = T.alloc_fragment((1,), "float32")
    normalizer = T.alloc_fragment((1,), "float32")
    total = T.alloc_fragment((1,), "float32")
    T.fill(maximum, -T.infinity("float32"))
    T.clear(result)
    T.clear(normalizer)
    for tile in T.serial({s//tile}):
        for i, j in T.Parallel({tile}, {width}):
            dots[i, j] = q[head*{width}+j] * T.cast({key}, "float32")
        T.reduce_sum(dots, scores, dim=1)
        T.copy(maximum, previous)
        for i in T.Parallel({tile}):
            scores[i] *= {scale_dim**-0.5}
        T.reduce_max(scores, maximum, dim=0, clear=False)
        for i in T.Parallel({tile}):
            scores[i] = T.exp(scores[i]-maximum[0])
        T.reduce_sum(scores, total, dim=0)
        for z in T.Parallel(1):
            normalizer[0] = normalizer[0]*T.exp(previous[0]-maximum[0])+total[0]
        for i, j in T.Parallel({tile}, {width}):
            products[i, j] = scores[i]*T.cast({val}, "float32")
        T.reduce_sum(products, partial, dim=0)
        for j in T.Parallel({width}):
            result[j] = result[j]*T.exp(previous[0]-maximum[0])+partial[j]
        T.sync_threads()
    for j in T.Parallel({width}):
        out[head*{width}+j] = result[j]/normalizer[0]'''
    cache = ('kv', s*2*h*width, 'float16') if explicit else ('c', s*width, 'float16')
    if partitions == 1:
        return emit([('q', h*width, 'float32'), cache, ('out', h*width, 'float32')], body)
    body = body.replace(f'with T.Kernel({h}, threads=128) as head:',
                        f'with T.Kernel({partitions}, {h}, threads=128) as (part, head):')
    body = body.replace(f'T.serial({s//tile})', f'T.serial({s//partitions//tile})')
    body = body.replace(f'tile*{tile}+i', f'part*{s//partitions}+tile*{tile}+i')
    body = body.replace(f'out[head*{width}+j] = result[j]/normalizer[0]',
                        f'out[(head*{partitions}+part)*{width}+j] = result[j]')
    body += f'''\n    for z in T.Parallel(1):
        stats[(head*{partitions}+part)*2] = maximum[0]
        stats[(head*{partitions}+part)*2+1] = normalizer[0]'''
    return emit([('q',h*width,'float32'),cache,('out',h*partitions*width,'float32'),
                 ('stats',h*partitions*2,'float32')],body)


def merge_source(h, width, partitions):
    return emit([('partial',h*partitions*width,'float32'),('stats',h*partitions*2,'float32'),
                 ('out',h*width,'float32')],f'''with T.Kernel({h}, threads=128) as head:
    maxima = T.alloc_fragment(({partitions},), "float32")
    factors = T.alloc_fragment(({partitions},), "float32")
    totals = T.alloc_fragment(({partitions},), "float32")
    products = T.alloc_fragment(({partitions}, {width}), "float32")
    result = T.alloc_fragment(({width},), "float32")
    maximum = T.alloc_fragment((1,), "float32")
    normalizer = T.alloc_fragment((1,), "float32")
    for i in T.Parallel({partitions}):
        maxima[i] = stats[(head*{partitions}+i)*2]
    T.reduce_max(maxima, maximum, dim=0)
    for i in T.Parallel({partitions}):
        factors[i] = T.exp(maxima[i]-maximum[0])
        totals[i] = factors[i]*stats[(head*{partitions}+i)*2+1]
    T.reduce_sum(totals, normalizer, dim=0)
    for i, j in T.Parallel({partitions}, {width}):
        products[i,j] = factors[i]*partial[(head*{partitions}+i)*{width}+j]
    T.reduce_sum(products, result, dim=0)
    for j in T.Parallel({width}):
        out[head*{width}+j] = result[j]/normalizer[0]''')


def query_source(h, d, r):
    return emit([('q', h*d, 'float32'), ('w', 2*h*d*r, 'float16'),
                 ('out', h*r, 'float32')], f'''with T.Kernel({h}, threads=128) as head:
    accum = T.alloc_fragment(({r},), "float32")
    T.clear(accum)
    for k in T.serial({d}):
        for j in T.Parallel({r}):
            accum[j] += q[head*{d}+k]*T.cast(w[(head*{d}+k)*{r}+j], "float32")
    for j in T.Parallel({r}):
        out[head*{r}+j] = accum[j]''')


def output_source(h, d, r):
    return emit([('latent', h*r, 'float32'), ('w', 2*h*d*r, 'float16'),
                 ('out', h*d, 'float32')], f'''with T.Kernel({h}, threads=128) as head:
    accum = T.alloc_fragment(({d},), "float32")
    T.clear(accum)
    for k in T.serial({r}):
        for j in T.Parallel({d}):
            accum[j] += latent[head*{r}+k]*T.cast(w[({h*d}+head*{d}+j)*{r}+k], "float32")
    for j in T.Parallel({d}):
        out[head*{d}+j] = accum[j]''')


def oracle(c, w, q, h, d):
    c64, w64, q64 = (x.astype(np.float64) for x in (c, w, q))
    full = c64 @ w64.T
    def attend(kv):
        k = kv[:, :h*d].reshape(-1, h, d).transpose(1, 0, 2)
        v = kv[:, h*d:].reshape(-1, h, d).transpose(1, 0, 2)
        scores = np.einsum('hd,hsd->hs', q64, k)*d**-0.5
        prob = np.exp(scores-scores.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        return np.einsum('hs,hsd->hd', prob, v)
    return full, attend(full.astype(np.float16).astype(np.float64)), attend(full)


def bound_call(device, kernel, args):
    from tensor.runtime.abi import BoundCall
    values, symbols, launch = kernel._bind(args, {}, include_outputs=True)
    return (kernel, BoundCall(device, kernel.manifest, values, symbols, launch, validated=True))


def measure(plan, device, samples, repeats):
    for _ in range(3):
        plan.launch(); device.synchronize()
    result = []
    for _ in range(samples):
        device.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            plan.launch()
        device.synchronize()
        result.append((time.perf_counter()-start)*1000/repeats)
    return {'median_ms': statistics.median(result), 'samples_ms': result,
            'min_ms': min(result), 'max_ms': max(result)}


class TimestampAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
    def __getattr__(self, name):
        return getattr(self.adapter, name)
    def request_device_sync(self, **kwargs):
        kwargs['required_features'] += ['timestamp-query', 'timestamp-query-inside-passes']
        return self.adapter.request_device_sync(**kwargs)


def measure_gpu(plan, device, samples, repeats, period_ns):
    import wgpu
    from wgpu.backends.wgpu_native.extras import write_timestamp
    gpu = device._gpu
    query = gpu.create_query_set(type='timestamp', count=2)
    resolved = gpu.create_buffer(size=16, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
    def sample():
        encoder = gpu.create_command_encoder()
        compute = encoder.begin_compute_pass()
        write_timestamp(compute, query, 0)
        for _ in range(repeats):
            for pipeline, group, grid in plan.nodes:
                compute.set_pipeline(pipeline); compute.set_bind_group(0, group)
                compute.dispatch_workgroups(*grid)
        write_timestamp(compute, query, 1); compute.end()
        encoder.resolve_query_set(query, 0, 2, resolved, 0)
        gpu.queue.submit([encoder.finish()])
        ticks = np.frombuffer(gpu.queue.read_buffer(resolved), dtype=np.uint64)
        return int(ticks[1]-ticks[0])*period_ns*1e-6/repeats
    try:
        for _ in range(3): sample()
        observations = [sample() for _ in range(samples)]
        if min(observations) <= 0:
            raise RuntimeError('nonpositive GPU timestamp measurement')
        return {'median_ms':statistics.median(observations), 'samples_ms':observations,
                'min_ms':min(observations), 'max_ms':max(observations)}
    finally:
        resolved.destroy(); query.destroy()


def warm_device(plan, device):
    deadline = time.perf_counter()+.4
    while time.perf_counter() < deadline:
        for _ in range(20): plan.launch()
        device.synchronize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tensor-root', type=Path, default=Path(__file__).resolve().parents[2]/'tensor')
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'results'/'rx6700xt-partitioned')
    p.add_argument('--contexts', nargs='+', type=int, default=[128, 512, 2048, 4096])
    p.add_argument('--ranks', nargs='+', type=int, default=[64, 128])
    p.add_argument('--loops', nargs='+', type=int, default=[1, 4, 10])
    p.add_argument('--samples', type=int, default=7)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--partitions', type=int, default=16,
                   help='split context across GPU workgroups; 1 reproduces the sequential control')
    a = p.parse_args()
    if any(s <= 0 or s % 16 for s in a.contexts) or any(r <= 0 or r % 32 for r in a.ranks):
        p.error('contexts must be positive multiples of 16; ranks positive multiples of 32')
    if any(t <= 0 for t in a.loops) or a.samples < 3 or a.repeats < 1:
        p.error('positive loops/repeats and at least three samples required')
    if a.partitions not in (1,2,4,8,16):
        p.error('partitions must be 1,2,4,8,16')
    os.environ.setdefault('WGPU_BACKEND_TYPE', 'Vulkan')
    sys.path.insert(0, str(a.tensor_root.resolve()/'src'))
    import tensor as tx
    from tensor.providers.webgpu import Device
    from tensor.artifacts.format import read_artifact
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'benchmark_source.py').write_bytes(Path(__file__).read_bytes())
    artifacts = a.out/'artifacts'; artifacts.mkdir(exist_ok=True)
    h, d = 12, 64
    vk = subprocess.run(['vulkaninfo'], capture_output=True, text=True, check=True)
    periods = re.findall(r'timestampPeriod\s*=\s*([0-9.]+)', vk.stdout)
    if len(periods) != 1:
        raise RuntimeError('timestamp calibration currently requires one Vulkan GPU')
    period_ns = float(periods[0])
    report = {'schema': 'llt.vulkan-attention.v1', 'status': 'running',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'tensor_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.tensor_root, text=True).strip(),
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'versions': {name: version(name) for name in ('numpy', 'wgpu', 'tilelang', 'tensor-workspace')},
              'configuration': {'batch': 1, 'heads': h, 'head_dim': d, 'layers_in_cache_estimate': 12,
                                'cache_dtype': 'float16', 'accumulator_dtype': 'float32', 'rope': False,
                                'max_context_partitions':a.partitions},
              'protocol': {'warmups': 3, 'samples': a.samples, 'repeats_per_sample': a.repeats,
                           'timing': 'prepared device-resident plan host submission + queue completion, amortized across repeats',
                           'excluded': ['build', 'pipeline creation', 'upload', 'readback', 'allocation', 'CPU oracle'],
                           'reference': 'independent float64; explicit FP16 K/V vs ideal absorbed K/V',
                           'not_measured': ['trained-model quality', 'full transformer', 'MLP', 'RoPE', 'backward',
                                            'actual whole-model peak VRAM', 'multi-GPU communication', 'H100 performance'],
                           'queries': 'fixed per-head query; repeated attention calls, not an evolving recurrent model',
                           'baseline_cache': 'one representative K/V cache resident; L*T scaling calculated, not allocated',
                           'gpu_timing': 'Vulkan timestamps around all ordered plan dispatches in one compute pass; host excluded',
                           'timestamp_period_ns':period_ns, 'timestamp_period_source':'vulkaninfo timestampPeriod',
                           'order': 'rotated method order across loop cases', 'seed': 61,
                           'device_warmup_seconds_per_shape':.4,
                           'correctness_replays_per_path':20},
              'cases': [], 'artifacts': []}
    report_path = a.out/'report.json'
    def save():
        report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    device = Device(a.device)
    device._adapter = TimestampAdapter(device._adapter)
    with device:
        if device.info['adapter']['adapter_type'] != 'DiscreteGPU' or device.info['adapter']['backend_type'] != 'Vulkan':
            raise RuntimeError('a physical discrete Vulkan adapter is required')
        report['device'] = device.info
        save()
        for r in a.ranks:
            for s in a.contexts:
                partitions = min(a.partitions,s//16)
                if s % (partitions*16):
                    raise ValueError('context must divide evenly into 16-token partition tiles')
                rng = np.random.default_rng(np.random.SeedSequence([61, r, s]))
                c = rng.normal(size=(s, r)).astype(np.float16)
                w = (rng.normal(size=(2*h*d, r))/r**0.5).astype(np.float16)
                q = (rng.normal(size=(h, d))*.5).astype(np.float32)
                full, expected_explicit, expected_absorbed = oracle(c, w, q, h, d)
                buffers = [device.from_numpy(c.ravel()), device.from_numpy(w.ravel()), device.from_numpy(q.ravel()),
                           device.full(s*2*h*d, np.nan, dtype='float16'), device.full(h*d, np.nan),
                           device.full(h*r, np.nan), device.full(h*r, np.nan)]
                cb, wb, qb, kvb, outb, aqb, latentb = buffers
                if partitions>1:
                    buffers += [device.full(h*partitions*d,np.nan),device.full(h*partitions*2,np.nan),
                                device.full(h*partitions*r,np.nan),device.full(h*partitions*2,np.nan)]
                    epb, esb, apb, asb = buffers[-4:]
                kernels = {}
                sources = {'expand': expansion_source(s,h,d,r),
                           'explicit': attention_source(s,h,d,d,explicit=True,partitions=partitions),
                           'query': query_source(h,d,r),
                           'latent': attention_source(s,h,r,d,partitions=partitions), 'output': output_source(h,d,r)}
                if partitions>1:
                    sources.update(explicit_merge=merge_source(h,d,partitions),latent_merge=merge_source(h,r,partitions))
                for name, source in sources.items():
                    digest = hashlib.sha256(source.encode()).hexdigest()
                    path = artifacts/f'{name}-{digest[:16]}.py'; artifact = path.with_suffix('.tbin')
                    path.write_text(source, encoding='utf-8')
                    if not artifact.exists():
                        tx.build(path, artifact, provider='webgpu', cache_dir=a.out/'cache')
                    start = time.perf_counter(); kernels[name] = device.load(artifact)
                    manifest, files = read_artifact(artifact)
                    report['artifacts'].append({'context':s,'rank':r,'name':name,
                        'path':str(artifact.relative_to(a.out)), 'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        'wgsl_sha256':hashlib.sha256(files['kernel.wgsl']).hexdigest(),
                        'pipeline_creation_ms':(time.perf_counter()-start)*1000,
                        'workgroup_storage_bytes':manifest['webgpu']['workgroup_storage_bytes']})
                arguments = [('expand',(cb,wb,kvb)),('query',(qb,wb,aqb)),('output',(latentb,wb,outb))]
                if partitions>1:
                    arguments += [('explicit',(qb,kvb,epb,esb)),('explicit_merge',(epb,esb,outb)),
                                  ('latent',(aqb,cb,apb,asb)),('latent_merge',(apb,asb,latentb))]
                else:
                    arguments += [('explicit',(qb,kvb,outb)),('latent',(aqb,cb,latentb))]
                calls = {name:bound_call(device,kernels[name],args) for name,args in arguments}
                explicit_calls = [calls['explicit']]+([calls['explicit_merge']] if partitions>1 else [])
                absorbed_calls = [calls['query'],calls['latent']]+([calls['latent_merge']] if partitions>1 else [])+[calls['output']]
                check_plan = device.prepare_plan([calls['expand']]+explicit_calls)
                explicit_errors = []
                for _ in range(20):
                    check_plan.launch()
                    actual_explicit = outb.to_numpy().reshape(h,d)
                    np.testing.assert_allclose(actual_explicit,expected_explicit,atol=2e-4,rtol=.002)
                    explicit_errors.append(float(np.max(np.abs(actual_explicit-expected_explicit))))
                actual_expansion = kvb.to_numpy().reshape(s,2*h*d)
                check_plan.close()
                np.testing.assert_allclose(actual_expansion,full.astype(np.float16),atol=.008,rtol=.003)
                if partitions>1:
                    partials = epb.to_numpy().reshape(h,partitions,d).astype(np.float64)
                    stats = esb.to_numpy().reshape(h,partitions,2).astype(np.float64)
                    factors = np.exp(stats[:,:,0]-stats[:,:,0].max(axis=1,keepdims=True))
                    combined = (partials*factors[:,:,None]).sum(axis=1)/(stats[:,:,1]*factors).sum(axis=1)[:,None]
                    np.testing.assert_allclose(actual_explicit,combined,atol=2e-6,rtol=2e-5,
                                               err_msg='partition merge differs from float64 combination of GPU partials')
                np.testing.assert_allclose(actual_explicit,expected_explicit,atol=2e-4,rtol=.002)
                check_plan = device.prepare_plan(absorbed_calls)
                absorbed_errors = []
                for _ in range(20):
                    check_plan.launch()
                    actual_absorbed = outb.to_numpy().reshape(h,d)
                    np.testing.assert_allclose(actual_absorbed,expected_absorbed,atol=2e-4,rtol=.002)
                    absorbed_errors.append(float(np.max(np.abs(actual_absorbed-expected_absorbed))))
                check_plan.close()
                np.testing.assert_allclose(actual_absorbed,expected_absorbed,atol=2e-4,rtol=.002)
                np.testing.assert_allclose(actual_explicit,actual_absorbed,atol=.002,rtol=.02)
                validation = {'passed':True,
                    'explicit_max_abs_error':max(explicit_errors),
                    'absorbed_max_abs_error':max(absorbed_errors),
                    'cross_path_max_abs_difference':float(np.max(np.abs(actual_explicit-actual_absorbed)))}
                print(f'validated context={s} rank={r}: {validation}',flush=True)
                warm = device.prepare_plan(explicit_calls); warm_device(warm,device); warm.close()
                methods = ['cached_explicit','reexpand_each_call','expand_once_reuse','absorbed']
                for case_index,t in enumerate(a.loops):
                    sequences = {'cached_explicit':explicit_calls*t,
                                 'reexpand_each_call':([calls['expand']]+explicit_calls)*t,
                                 'expand_once_reuse':[calls['expand']]+explicit_calls*t,
                                 'absorbed':absorbed_calls*t}
                    row = {'context':s,'rank':r,'loops':t,'context_partitions':partitions,'validation':validation,
                           'buffer_bytes': {'one_explicit_kv_cache':kvb.nbytes,'shared_latent_cache':cb.nbytes,
                                            'kv_expansion_scratch':kvb.nbytes,'absorbed_query_and_accumulator':aqb.nbytes+latentb.nbytes,
                                            'kv_projection_weights':wb.nbytes,
                                            'explicit_partition_scratch':epb.nbytes+esb.nbytes if partitions>1 else 0,
                                            'absorbed_partition_scratch':apb.nbytes+asb.nbytes if partitions>1 else 0,
                                            'naive_12layer_loop_cache_calculated':kvb.nbytes*12*t},
                           'timings':{}}
                    for method in methods[case_index%4:]+methods[:case_index%4]:
                        plan = device.prepare_plan(sequences[method])
                        row['timings'][method] = measure(plan,device,a.samples,a.repeats)
                        row['timings'][method]['gpu'] = measure_gpu(plan,device,a.samples,a.repeats,period_ns)
                        row['timings'][method]['dispatches_per_plan'] = len(sequences[method])
                        plan.close()
                    report['cases'].append(row); save()
                    print(f'S={s} r={r} T={t} '+', '.join(f'{m}={row["timings"][m]["gpu"]["median_ms"]:.3f}ms GPU' for m in methods),flush=True)
                for kernel in kernels.values(): kernel._dispose()
                for buffer in buffers: buffer.release()
        report['status']='passed';save()
    print(f'Saved {report_path}',flush=True)


if __name__ == '__main__':
    main()
