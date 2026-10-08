"""Qualified Tensor full-model LLT experiments, separate from the FP16 baseline."""
import argparse
import copy
import gc
import hashlib
import json
import importlib.metadata
import math
import os
import statistics
import subprocess
import time
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tensor_torch.llt import Operators, AdamW

from common import ROOT, TENSOR, digest, save, setup, timing, graph_timing, memory
from model import Config, Transformer
from tensor_model import BackendTransformer

OUT = ROOT / 'benchmarks/results/l40s-native'
PIN = 'ab17948fe94dd7c7b857c8f067c465e901329b94'
TOLERANCES = dict(logit_atol=0.025, logit_rtol=0.025, gradient_relative_l2=0.08,
                  training_loss_absolute=0.15, checkpoint_gradient_atol=2e-6,
                  checkpoint_gradient_rtol=2e-4)


def provenance():
    sources = {}
    for name in ('model.py', 'tensor_model.py', 'native_study.py', 'native_summary.py', 'common.py', 'run_native.sh'):
        path = Path(__file__).with_name(name)
        sha = digest(path)
        sources[name] = sha
        snapshot = OUT / 'sources' / sha / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    pin = subprocess.check_output(['git', 'rev-parse', PIN], cwd=TENSOR, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', pin, '--', 'packages/tensor-torch/src', 'src/tensor'], cwd=TENSOR, check=True, stdout=subprocess.DEVNULL)
    return dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                packages={name:importlib.metadata.version(name) for name in
                          ('torch','numpy','tensor-workspace','tensor-torch','tilelang','apache-tvm-ffi')},
                capability=list(torch.cuda.get_device_capability()),
                tensor_repository='https://github.com/kreasof-ai/tensor', tensor_implementation_commit=pin,
                tensor_checkout_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TENSOR, text=True).strip(),
                llt_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
                sources=sources, nvrtc_bootstrap_sha256=digest(TENSOR/'build/nvrtc-12.9/bootstrap.json'))


def clear():
    torch.cuda.synchronize()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()


def artifact_report(ops):
    report = copy.deepcopy(ops.report)
    for artifact in report['artifacts']:
        source = Path(artifact['path']).with_suffix('.py')
        if not source.exists():
            # A compiler-free dependency cache can contain binaries alone.
            # Recreate the identity-checked export for the experiment audit.
            from tensor.compiler.entry import export_source
            with zipfile.ZipFile(artifact['path']) as bundle:
                outputs=json.loads(bundle.read('manifest.json'))['outputs']
            module=artifact['module']
            exported=export_source(module,artifact['factory'],artifact['parameters'],
                dependencies=(module,'tensor.compiler.entry','tensor.runtime.abi'),outputs=outputs)
            assert hashlib.sha256((exported+artifact['target']).encode()).hexdigest()==source.stem
            source.write_text(exported)
        artifact['export_source_sha256'] = digest(source)
        artifact['path'] = str(Path(artifact['path']).relative_to(ROOT))
    return report


def finish(name, report, ops):
    report['coverage'] = artifact_report(ops)
    save(OUT / (name + '.json'), report)


def header(phase, samples):
    return dict(status='running', phase=phase, environment=provenance(), tolerances=TOLERANCES,
                protocol=dict(samples=samples, precision='FP32 master/residual; BF16 GEMM/attention',
                    seeds=dict(correctness=601,training_model=602,training_data=1602,inference_model=603,inference_data=1603),
                    positions='learned absolute', norm='unweighted RMSNorm epsilon 1e-5', gelu='exact',
                    architecture='initial-input global latent; loop-tied blocks; batched differentiable folds',
                    torch_attention='forced Flash SDPA, shared latent expanded as views',
                    tensor_control='PyTorch layouts, autograd and exact loop checkpoints, tied-gradient sums',
                    memory='CUDA allocator peak; includes weights, optimizer, activations and visible workspaces; excludes driver/context',
                    timing='one GPU process; compilation and warmup excluded; all observations retained'), cases=[])


def optimizer(model, ops):
    return AdamW(model.parameters(), ops, lr=1e-3, max_norm=1.0) if ops else torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)


def train_step(model, opt, tokens, targets, policy='none', chunk_size=0):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        loss = model.loss(tokens, targets, policy, chunk_size)
    loss.backward()
    if model.ops is None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.detach()


def relative_l2(x, y):
    return ((x-y).norm() / y.norm().clamp_min(1e-12)).item()


def prepared_timing(fn, reset, samples):
    """Fixture rewind is outside each measured serving trajectory."""
    gpu, wall = [], []
    for _ in range(samples):
        reset()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        gpu.append(start.elapsed_time(end))
        wall.append((time.perf_counter()-before)*1000)
        del result
    return dict(gpu_median_ms=statistics.median(gpu),wall_median_ms=statistics.median(wall),
                gpu_samples_ms=gpu,wall_samples_ms=wall)


def measure_decode(fn, reset, samples):
    for _ in range(3):
        reset(); fn()
    clear()
    reset()
    result, mem = memory(fn)
    del result
    eager = prepared_timing(fn,reset,samples)
    clear()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        reset(); fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    reset()
    clear()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with torch.cuda.graph(graph,stream=side):
        result = fn()
    captured = prepared_timing(graph.replay,reset,samples)
    captured['memory'] = dict(baseline_allocated_bytes=baseline,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del graph, result
    clear()
    return dict(memory=mem,timing=eager,graph_timing=captured)


def correctness(a, ops):
    report = header('correctness', a.samples)
    for kind in ('naive', 'llt', 'layerwise'):
        setup(601)
        c = Config(kind=kind, width=64, heads=2, layers=2, loops=3, rank=32, vocab=64, max_seq=64, gelu='none')
        model = BackendTransformer(c, ops).cuda()
        reference = BackendTransformer(c).cuda()
        reference.load_state_dict(model.state_dict())
        original = Transformer(c).double().cuda()
        original.load_state_dict(model.state_dict())
        batched = BackendTransformer(c).double().cuda()
        batched.load_state_dict(model.state_dict())
        tokens = torch.randint(c.vocab, (2, 17), device='cuda')
        targets = (tokens + 1) % c.vocab
        # Independent original folded/unfolded algebra, before low precision.
        gold = original(tokens)
        adapted = batched(tokens)
        torch.testing.assert_close(adapted, gold, atol=1e-11, rtol=1e-9)
        if kind != 'naive':
            torch.testing.assert_close(adapted, original(tokens, unfolded=True), atol=1e-11, rtol=1e-9)
        ga = torch.autograd.grad(adapted.square().mean(), tuple(batched.parameters()))
        gg = torch.autograd.grad(gold.square().mean(), tuple(original.parameters()))
        for x, y in zip(ga, gg):
            torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-9)
        row = dict(kind=kind, config=asdict(c), algebra_output_max_error=(adapted-gold).abs().max().item(),
                   algebra_gradient_max_error=max((x-y).abs().max().item() for x,y in zip(ga,gg)))
        del original, batched, gold, adapted, ga, gg
        with torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            actual = model(tokens)
            expected = reference(tokens)
            ck = model(tokens, policy='loop')
            la = model.loss(tokens, targets)
            lr = reference.loss(tokens, targets)
        torch.testing.assert_close(actual, expected, atol=TOLERANCES['logit_atol'], rtol=TOLERANCES['logit_rtol'])
        torch.testing.assert_close(actual, ck, atol=0, rtol=0)
        ag = torch.autograd.grad(la, tuple(model.parameters()))
        rg = torch.autograd.grad(lr, tuple(reference.parameters()))
        errors = {name: relative_l2(x, y) for (name,_), x,y in zip(model.named_parameters(),ag,rg)}
        assert max(errors.values()) < TOLERANCES['gradient_relative_l2'], errors
        g = torch.autograd.grad(actual.float().square().mean(), tuple(model.parameters()))
        cg = torch.autograd.grad(ck.float().square().mean(), tuple(model.parameters()))
        for x,y in zip(g,cg):
            torch.testing.assert_close(x,y,atol=TOLERANCES['checkpoint_gradient_atol'],rtol=TOLERANCES['checkpoint_gradient_rtol'])
        row.update(logit_max_error=(actual-expected).abs().max().item(), gradient_relative_l2=errors,
                   checkpoint_gradient_max_error=max((x-y).abs().max().item() for x,y in zip(g,cg)))
        del actual, expected, ck, la, lr, ag, rg, g, cg
        topt, ropt = optimizer(model, ops), optimizer(reference, None)
        drifts = []
        for _ in range(32):
            la = train_step(model,topt,tokens,targets,'loop').item()
            lr = train_step(reference,ropt,tokens,targets,'loop').item()
            drifts.append(abs(la-lr))
        assert max(drifts) < TOLERANCES['training_loss_absolute'], drifts
        row['training_steps'] = 32
        row['training_loss_absolute_drifts'] = drifts
        row['final_training_losses'] = dict(tensor=la, torch=lr)
        topt.zero_grad(set_to_none=True)
        ropt.zero_grad(set_to_none=True)
        del topt, ropt
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            altered = tokens.clone()
            altered[:, 9:] = (altered[:, 9:] + 7) % c.vocab
            full = model(tokens)
            torch.testing.assert_close(model(altered)[:, :9], full[:, :9], atol=0, rtol=0)
            _, state = model.prefill(tokens[:, :13], capacity=32)
            cache_ptrs = [(x.keys.data_ptr(),x.values.data_ptr()) for x in state['caches']]
            errors = []
            for length in range(14,18):
                decoded = model.decode_token(tokens[:, length-1:length], state)
                gold = model(tokens[:, :length])[:, -1:]
                torch.testing.assert_close(decoded,gold,atol=TOLERANCES['logit_atol'],rtol=TOLERANCES['logit_rtol'])
                errors.append((decoded-gold).abs().max().item())
            assert cache_ptrs == [(x.keys.data_ptr(),x.values.data_ptr()) for x in state['caches']]
            assert all(x.check()==17 for x in state['caches'])
            # Serving state must be invalidated if parameters change.
            next(model.parameters()).add_(0.01)
            try:
                model.decode_token(tokens[:,:1],state)
            except ValueError:
                pass
            else:
                raise AssertionError('stale folded cache accepted')
            row.update(trained_causal='passed', trained_decode_max_errors=errors,
                       cache_count=len(state['caches']), cache_bytes=sum(x.nbytes for x in state['caches']),
                       stable_cache_storage=True, stale_weights_rejected=True)
        report['cases'].append(row)
        finish('correctness',report,ops)
        print('Correctness passed:',kind,'max gradient relative L2',max(row['gradient_relative_l2'].values()),flush=True)
        del model, reference, state, full, gold, decoded
        clear()
    report['status'] = 'passed'
    finish('correctness',report,ops)


def training(a, ops):
    report = header('training', a.samples)
    configs = [(128,4,2,1,257,4096,t,r) for t,r in ((1,32),(4,32),(10,32),(20,32),(1,64),(10,64),(1,128),(10,128))]
    configs += [(512,8,2,b,s,v,t,r) for b,s,v,t,r in ((1,1025,4096,1,32),(1,1025,4096,10,32),
               (1,1025,4096,20,32),(1,4097,4096,1,64),(1,4097,4096,4,64),
               (4,257,4096,1,64),(4,257,4096,4,64),(1,1025,50257,1,64),(1,1025,50257,10,64))]
    configs += [(768,12,12,1,257,4096,t,64) for t in (1,10)]
    if a.smoke:
        configs = configs[:1]
    for w,h,l,b,s,v,t,r in configs:
        row = dict(width=w,heads=h,layers=l,batch=b,sequence=s,vocab=v,loops=t,rank=r,methods=[])
        for kind in ('naive','llt'):
            for policy in ('none','loop'):
                for backend in ('torch','tensor'):
                    setup(602)
                    clear()
                    c = Config(kind=kind,width=w,heads=h,layers=l,loops=t,rank=r,vocab=v,max_seq=max(s,128),gelu='none')
                    model = BackendTransformer(c,ops if backend=='tensor' else None).cuda()
                    tokens = torch.randint(v,(b,s),device='cuda',generator=torch.Generator(device='cuda').manual_seed(1602))
                    targets = (tokens+1)%v
                    opt = optimizer(model, model.ops)
                    fn = lambda: train_step(model,opt,tokens,targets,policy)
                    for _ in range(3): fn()
                    clear()
                    value, mem = memory(fn)
                    tm = timing(fn,a.samples,1,warmup=0)
                    row['methods'].append(dict(kind=kind,policy=policy,backend=backend,config=asdict(c),
                        loss=value.item(),memory=mem,timing=tm,
                        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                        gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None),
                        optimizer_bytes=sum(x.numel()*x.element_size() for st in opt.state.values() for x in st.values() if isinstance(x,torch.Tensor))))
                    print('Training',w,s,v,t,r,kind,policy,backend,round(mem['peak_allocated_bytes']/2**20,2),round(tm['wall_median_ms'],3),flush=True)
                    del fn, value, opt, model, tokens, targets
                    clear()
        report['cases'].append(row)
        finish('training-smoke' if a.smoke else 'training',report,ops)
    report['status']='passed'
    finish('training-smoke' if a.smoke else 'training',report,ops)


def inference(a, ops):
    report = header('inference', a.samples)
    report['protocol']['precision']='BF16 projection weights/compute; FP32 embeddings, normalization and residuals'
    report['protocol']['decode_fixture']='actual prefix plus four appends; rewind outside timing and graph capture'
    configs = [(128,4,2,1,257,t,32) for t in (1,4,10,20)]
    configs += [(512,8,4,b,s,t,r) for b,s,t,r in ((1,1025,1,32),(1,1025,4,32),
                (1,4097,1,64),(1,4097,10,64),(1,4097,1,128),(1,4097,10,128),
                (4,257,1,64),(4,257,4,64))]
    configs += [(768,12,12,1,4097,t,64) for t in (1,10)]
    if a.smoke:
        configs = configs[:1]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for w,h,l,b,s,t,r in configs:
            row=dict(width=w,heads=h,layers=l,batch=b,sequence=s,loops=t,rank=r,methods=[])
            for kind in ('naive','llt','layerwise'):
                for backend in ('torch','tensor'):
                    setup(603)
                    clear()
                    c=Config(kind=kind,width=w,heads=h,layers=l,loops=t,rank=r,vocab=256,max_seq=s+4,gelu='none')
                    # Parameters need ordinary version counters for stale-cache
                    # detection, even though all executions use inference mode.
                    with torch.inference_mode(False):
                        model=BackendTransformer(c,ops if backend=='tensor' else None).cuda().eval()
                        # Prepare BF16 weights once for tied-block serving.
                        for block in model.blocks:
                            block.to(dtype=torch.bfloat16)
                        if c.kind!='naive':
                            model.down.to(dtype=torch.bfloat16)
                        model.output.to(dtype=torch.bfloat16)
                    tokens=torch.randint(c.vocab,(b,s+4),device='cuda',generator=torch.Generator(device='cuda').manual_seed(1603))
                    # Full prefill includes refresh of differentiable folded projections.
                    def fn():
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            return model(tokens[:,:s],return_cache=True)
                    for _ in range(3):fn()
                    clear()
                    result,mem=memory(fn)
                    del result
                    prefill=dict(memory=mem,timing=timing(fn,a.samples,1,warmup=0),graph_timing=graph_timing(fn,a.samples,1))
                    del fn
                    clear()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        _,state=model.prefill(tokens[:,:s])
                    # Actual prefix and four consecutive single-token appends.
                    def trajectory():
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            for length in range(s+1,s+5):
                                result=model.decode_token(tokens[:,length-1:length],state)
                            return result
                    actual=trajectory()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        gold=model(tokens)[:,-1:]
                    torch.testing.assert_close(actual,gold,atol=TOLERANCES['logit_atol'],rtol=TOLERANCES['logit_rtol'])
                    decode_error=(actual-gold).abs().max().item()
                    del actual,gold
                    clear()
                    reset=lambda:model.rewind(state,s)
                    decoded=measure_decode(trajectory,reset,a.samples)
                    if model.ops:
                        assert all(cache.check()==s+4 for cache in state['caches'])
                    row['methods'].append(dict(kind=kind,backend=backend,config=asdict(c),prefill=prefill,
                        four_token_decode=decoded,cache_count=len(state['caches']),
                        cache_bytes=sum(cache.nbytes for cache in state['caches']),
                        folded_weight_bytes=sum(x.numel()*x.element_size() for pair in state['folds'] for x in pair),
                        decode_max_error=decode_error,validation='passed'))
                    print('Inference',w,s,t,r,kind,backend,'cache MiB',round(row['methods'][-1]['cache_bytes']/2**20,3),
                          'four-token graph ms',round(decoded['graph_timing']['gpu_median_ms'],3),flush=True)
                    del trajectory,reset,state,model,tokens
                    clear()
            report['cases'].append(row)
            finish('inference-smoke' if a.smoke else 'inference',report,ops)
    report['status']='passed'
    finish('inference-smoke' if a.smoke else 'inference',report,ops)


def audit(a, ops):
    binaries={}
    reports={}
    for name in ('correctness','training','inference'):
        path=OUT/(name+'.json')
        report=json.loads(path.read_text())
        assert report['status']=='passed',name
        reports[name]=digest(path)
        for filename,sha in report['environment']['sources'].items():
            assert digest(OUT/'sources'/sha/filename)==sha
        assert not report['coverage']['fallbacks']
        for artifact in report['coverage']['artifacts']:
            path=ROOT/artifact['path']
            assert digest(path)==artifact['sha256']
            assert digest(path.with_suffix('.py'))==artifact['export_source_sha256']
            with zipfile.ZipFile(path) as bundle:
                manifest=json.loads(bundle.read('manifest.json'))
                assert manifest['target']=='sm_89' and manifest['provider']=='cuda'
            binaries[artifact['path']]=artifact['sha256']
        for case in report['cases']:
            if name=='training':
                assert {(x['kind'],x['policy'],x['backend']) for x in case['methods']}=={
                    (k,p,b) for k in ('naive','llt') for p in ('none','loop') for b in ('torch','tensor')}
                for method in case['methods']:
                    assert math.isfinite(method['loss'])
                    assert len(method['timing']['gpu_samples_ms'])==9
            if name=='inference':
                assert len(case['methods'])==6
                for method in case['methods']:
                    assert method['validation']=='passed'
                    for phase in ('prefill','four_token_decode'):
                        assert len(method[phase]['timing']['gpu_samples_ms'])==9
                        assert len(method[phase]['graph_timing']['gpu_samples_ms'])==9
                        assert 'memory' in method[phase]['graph_timing']
    result=dict(status='passed',reports=reports,unique_binaries=len(binaries),binaries=binaries,
                checks=['source snapshots','binary hashes and sm_89','no semantic fallbacks','complete matched policies',
                        'nine timing observations','real prefix and consecutive cached decode','finite training loss'])
    save(OUT/'audit.json',result)
    print('Audit passed:',len(binaries),'Tensor artifacts')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('correctness','training','inference','audit'))
    parser.add_argument('--samples',type=int,default=9)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    setup(600)
    ops=Operators(OUT/'artifacts')
    globals()[args.phase](args,ops)
