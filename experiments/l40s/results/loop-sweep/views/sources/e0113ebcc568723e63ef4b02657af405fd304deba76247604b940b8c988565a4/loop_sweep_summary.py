"""Audit all requested cases and export standalone latency/memory figures."""

# Support both direct script execution and python -m experiments.l40s.<module>.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l40s.runtime import results_root, recorded_path
import csv
import hashlib
import itertools
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = results_root() / 'loop-baseline'
VIEWS = OUT / 'views'
MODELS = ('llt', 'naive_loop', 'stacked', 'fixed_depth')
LABELS = dict(llt='LLT',naive_loop='Naive Loop',stacked='Independent stack',fixed_depth='Fixed depth / matched params')
COLORS = dict(llt='#166f91',naive_loop='#d17d25',stacked='#8156a5',fixed_depth='#389361')
METRICS = ('training_eager','training_graph','prompt_eager','prompt_graph','startup_eager','startup_graph','decode_eager','decode_graph')


def main():
    VIEWS.mkdir(parents=True, exist_ok=True)
    inputs,artifacts,rows,checks = {},{},{},[]
    verified_artifact_files=set()
    numerical_profiles=set()
    for p in sorted(OUT.glob('*.json')):
        if not p.name.startswith(('training-','inference-','correctness-')):
            continue
        d=json.loads(p.read_text())
        assert d['status'] in ('passed','out_of_memory'), (p,d['status'],d.get('error'))
        a=d['arguments']
        inputs[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        assert d['provenance']['native_cpp_executor']
        assert d['provenance']['gpu']=='NVIDIA L40S'
        numerical_profiles.add(tuple(sorted(d['provenance']['sources'].items())))
        for name,sha in d['provenance']['sources'].items():
            snap=OUT/'sources'/sha/Path(name).name
            assert hashlib.sha256(snap.read_bytes()).hexdigest()==sha
        if 'coverage' in d:
            assert not d['coverage']['fallbacks']
            for ar in d['coverage']['artifacts']:
                assert ar['target']=='sm_89'
                identity=(ar['path'],ar['sha256'])
                if identity not in verified_artifact_files:
                    assert hashlib.sha256(recorded_path(ar['path']).read_bytes()).hexdigest()==ar['sha256']
                    verified_artifact_files.add(identity)
                artifacts[ar['sha256']]=ar
        if a['phase']=='correctness':
            assert d['status']=='passed'
            checks.append(p.name)
            continue
        key=(a['phase'],a['model'],a['backend'],a['loops'])
        assert key not in rows
        assert d['batch_size']==4 and d['sequence_length']==1024
        c=d['config']
        assert (c['width'],c['heads'],c['rank'],c['vocab'],c['max_seq'])==(768,12,64,50304,1025)
        assert d['checkpoint_policy']=='none'
        assert d['effective_depth']==(12 if a['model']=='fixed_depth' else 12*a['loops'])
        assert d['unique_layers']==(12*a['loops'] if a['model']=='stacked' else 12)
        if a['model']=='fixed_depth':
            assert d['parameter_count']==d['stacked_parameter_target']
        r=dict(phase=a['phase'],model=a['model'],backend=a['backend'],loops=a['loops'],
               status=d['status'],failed_stage=d.get('stage',''),parameter_count=d['parameter_count'],
               effective_depth=d['effective_depth'],unique_layers=d['unique_layers'])
        for name in METRICS:
            if name not in d:
                continue
            t=d[name]
            assert len(t['gpu_samples_ms'])==a['samples']==9
            assert all(math.isfinite(v) and v>0 for v in t['gpu_samples_ms'])
            r[name+'_gpu_ms']=t['gpu_median_ms']
            r[name+'_peak_gib']=t.get('capture_peak_allocated_bytes',t.get('peak_allocated_bytes'))/2**30
            if 'wall_median_ms' in t:
                r[name+'_wall_ms']=t['wall_median_ms']
            if 'capture_peak_reserved_bytes' in t:
                r[name+'_reserved_gib']=t['capture_peak_reserved_bytes']/2**30
        r['cache_mib']=d.get('cache_bytes',0)/2**20
        r['prepared_weight_mib']=d.get('prepared_weight_bytes',0)/2**20
        if 'training_graph' in d:
            counts=d['graph_step_counter']
            assert counts['min']==counts['max']==counts['expected']==42
            assert d['all_parameter_gradients_finite']
        rows[key]=r
    expected=set(itertools.product(('training','inference'),MODELS,('tensor','torch'),range(1,17)))
    assert set(rows)==expected, f'Missing cases: {sorted(expected-set(rows))}'
    assert len(checks)==8
    assert len(numerical_profiles)==1, 'numerical sources changed during the sweep'
    recovery=[]
    recovery_inputs={}
    for p in sorted((OUT/'capture-recovery').glob('training-*.json')):
        d=json.loads(p.read_text());a=d['arguments']
        assert d['status'] in ('passed','out_of_memory'),(p,d['status'])
        key=('training',a['model'],a['backend'],a['loops'])
        assert rows[key]['status']=='out_of_memory' and rows[key]['failed_stage']=='training_graph'
        assert tuple(sorted(d['provenance']['sources'].items())) in numerical_profiles
        source=OUT/'capture-recovery'/d['capture_setup']['source_snapshot']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==d['capture_setup']['source_sha256']
        for ar in d.get('coverage',{}).get('artifacts',[]):
            assert ar['target']=='sm_89'
            identity=(ar['path'],ar['sha256'])
            if identity not in verified_artifact_files:
                assert hashlib.sha256(recorded_path(ar['path']).read_bytes()).hexdigest()==ar['sha256']
                verified_artifact_files.add(identity)
            artifacts[ar['sha256']]=ar
        assert not d.get('coverage',{}).get('fallbacks',[])
        r=dict(model=a['model'],backend=a['backend'],loops=a['loops'],status=d['status'])
        if 'training_graph' in d:
            g=d['training_graph'];counts=d['graph_step_counter']
            assert counts['min']==counts['max']==counts['expected']==42
            assert g['released_gradient_bytes']>=d['parameter_bytes']
            assert d['all_parameter_gradients_finite']
            r.update(graph_ms=g['gpu_median_ms'],capture_peak_gib=g['capture_peak_allocated_bytes']/2**30,
                     capture_reserved_gib=g['capture_peak_reserved_bytes']/2**30,
                     released_gradient_gib=g['released_gradient_bytes']/2**30)
        recovery.append(r)
        recovery_inputs[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    required_recovery={(r['model'],r['backend'],r['loops']) for r in rows.values()
                       if r['status']=='out_of_memory' and r['failed_stage']=='training_graph'}
    assert {(r['model'],r['backend'],r['loops']) for r in recovery}==required_recovery
    ordered=[rows[k] for k in sorted(rows)]
    observations={}
    observed=list((OUT/'observed-capture-oom').glob('*.json'))+list((OUT/'capture-recovery/observed-capture-oom').glob('*.json'))
    for p in sorted(observed):
        original=json.loads(p.read_text())
        classified=json.loads((p.parent.parent/p.name).read_text())
        assert original['status']=='error' and classified['status']=='out_of_memory'
        assert classified['classification']['original_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
        assert ('torch.OutOfMemoryError: CUDA out of memory' in original['traceback']
                or 'CUDA_ERROR_OUT_OF_MEMORY' in original['traceback'])
        observations[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    audit=dict(status='passed',case_count=len(rows),correctness_checks=checks,inputs=inputs,observations=observations,recovery_inputs=recovery_inputs,
               numerical_sources=dict(next(iter(numerical_profiles))),
               tensor_artifact_count=len(artifacts),artifacts=artifacts)
    (VIEWS/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (VIEWS/'summary.json').write_text(json.dumps(dict(status='passed',rows=ordered,capture_recovery=recovery),indent=2)+'\n')
    fields=list(dict.fromkeys(k for r in ordered for k in r))
    with (VIEWS/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(ordered)
    if recovery:
        fields=list(dict.fromkeys(k for r in recovery for k in r))
        with (VIEWS/'capture-recovery.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
            writer.writeheader()
            writer.writerows(recovery)
    for phase,stem,label in [('training','training','Full training step'),('inference','prompt','Prompt / last logits'),('inference','startup','Serving startup / KV allocation'),('inference','decode','One cached token / 1024-token history')]:
        for mode in ('graph','eager'):
            fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharex=True)
            for col,backend in enumerate(('tensor','torch')):
                for model in MODELS:
                    rs=[rows[(phase,model,backend,t)] for t in range(1,17)]
                    metric=stem+'_'+mode
                    latency=metric+('_wall_ms' if mode=='eager' else '_gpu_ms')
                    axes[0,col].plot(range(1,17),[r.get(latency,float('nan')) for r in rs],color=COLORS[model],marker='.',label=LABELS[model])
                    axes[1,col].plot(range(1,17),[r.get(metric+'_peak_gib',float('nan')) for r in rs],color=COLORS[model],marker='.')
                    missing=[r['loops'] for r in rs if latency not in r]
                    if missing:
                        axes[1,col].scatter(missing,[45.0]*len(missing),marker='x',s=25,color=COLORS[model])
                axes[0,col].set_title('Tensor' if backend=='tensor' else 'PyTorch / fused AdamW' if phase=='training' else 'PyTorch')
                axes[0,col].set_ylabel('CUDA graph latency (ms)' if mode=='graph' else 'Eager wall latency (ms)')
                axes[1,col].set_ylabel('Capture peak allocated memory (GiB)' if mode=='graph' else 'Peak allocated memory (GiB)')
                axes[1,col].set_xlabel('Loop count T / independent stack depth = 12T')
                for ax in axes[:,col]:
                    ax.grid(alpha=.2)
                    ax.set_xticks(range(1,17))
                    ax.set_xlim(.7,16.3)
                axes[0,col].legend(fontsize=8)
            fig.suptitle(f'L40S / B4 / S1024 / W768 / {label} / {mode}\nCrosses at 45 GiB indicate unavailable measurements following OOM')
            fig.tight_layout()
            fig.savefig(VIEWS/f'{stem}-{mode}.png',dpi=180)
            fig.savefig(VIEWS/f'{stem}-{mode}.pdf')
            plt.close(fig)
    text=['Measured Tensor CUDA graph latency / capture peak allocation (ms / GiB). Blank methods following OOM are shown as OOM.','']
    for phase,stem in [('training','training'),('inference','prompt'),('inference','startup'),('inference','decode')]:
        text += [stem,'','| T | LLT | Naive Loop | Independent stack | Fixed depth, matched params |','|---:|---:|---:|---:|---:|']
        for t in range(1,17):
            cells=[]
            for model in MODELS:
                r=rows[(phase,model,'tensor',t)]
                metric=stem+'_graph'
                cells.append(f"{r[metric+'_gpu_ms']:.2f} / {r[metric+'_peak_gib']:.2f}" if metric+'_gpu_ms' in r else 'OOM')
            text.append('| '+str(t)+' | '+' | '.join(cells)+' |')
        text.append('')
    (VIEWS/'tables.md').write_text('\n'.join(text).rstrip()+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for model in MODELS:
        rs=[rows[('inference',model,'tensor',t)] for t in range(1,17)]
        style='--' if model=='fixed_depth' else ':' if model=='stacked' else '-'
        axes[0].plot(range(1,17),[r['parameter_count']/1e6 for r in rs],style,color=COLORS[model],label=LABELS[model])
        axes[1].plot(range(1,17),[r['cache_mib'] for r in rs],style,color=COLORS[model],label=LABELS[model])
    axes[0].set_ylabel('Unique active parameters (millions)')
    axes[1].set_ylabel('Persistent KV cache (MiB, logarithmic)')
    axes[1].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('T')
        ax.set_xticks(range(1,17))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('L40S / B4 / 1024-token history / capacity 1025')
    fig.tight_layout()
    fig.savefig(VIEWS/'parameters-and-cache.png',dpi=180)
    fig.savefig(VIEWS/'parameters-and-cache.pdf')
    plt.close(fig)
    print(json.dumps(dict(cases=len(rows),checks=len(checks),artifacts=len(artifacts),
                         passed=sum(r['status']=='passed' for r in ordered),
                         oom=sum(r['status']=='out_of_memory' for r in ordered)),indent=2))


if __name__=='__main__':
    main()
