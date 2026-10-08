"""Derive comparisons and publication figures from completed raw measurements."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'benchmarks/results/l40s'


def load(name):
    value=json.loads((OUT/(name+'.json')).read_text())
    assert value['status']=='passed',(name,value['status'])
    return value


def infer_key(row):
    return tuple(row[k] for k in ('width','layers','rank','batch','context'))


def train_key(row):
    return tuple(row[k] for k in ('width','layers','rank','batch','sequence','vocab'))


def summarize():
    reports={name:load(name) for name in ('attention','attention-serial','decode-partitions','inference','prefill','training','correctness')}
    result=dict(status='passed',raw_report_hashes={name:hashlib.sha256((OUT/(name+'.json')).read_bytes()).hexdigest() for name in reports},
                counts={name:len(r.get('cases',r.get('checks',[]))) for name,r in reports.items()},
                inference=[],training=[],attention=[])
    serial={tuple(r[k] for k in ('batch','heads','kv_heads','query','context','dim','causal')):r for r in reports['attention-serial']['cases']}
    for row in reports['attention']['cases']:
        key=tuple(row[k] for k in ('batch','heads','kv_heads','query','context','dim','causal'))
        tx=row['tensor_graph']['gpu_median_ms'];pt=row['pytorch_flash_graph']['gpu_median_ms']
        result['attention'].append(dict(zip(('batch','heads','kv_heads','query','context','dim','causal'),key),
            tensor_graph_ms=tx,pytorch_flash_graph_ms=pt,tensor_to_flash_ratio=tx/pt,
            serial_to_tensor_speedup=serial[key]['tensor_graph']['gpu_median_ms']/tx))
    nonloop={infer_key(row):row for row in reports['inference']['cases'] if row['loops']==1}
    for row in reports['inference']['cases']:
        methods={(m['kind'],m['backend']):m for m in row['methods']}
        for backend in ('tensor','pytorch_flash'):
            naive,llt,layerwise=[methods[kind,backend] for kind in ('naive','llt','layerwise')]
            base=next(m for m in nonloop[infer_key(row)]['methods'] if m['kind']=='llt' and m['backend']==backend)
            saving=1-llt['memory']['peak_allocated_bytes']/naive['memory']['peak_allocated_bytes']
            latency=llt['graph_timing']['gpu_median_ms']/naive['graph_timing']['gpu_median_ms']
            relative=llt['memory']['peak_allocated_bytes']/base['memory']['peak_allocated_bytes']
            result['inference'].append(dict(
                **{k:v for k,v in row.items() if k!='methods'},backend=backend,
                naive_peak_mib=naive['memory']['peak_allocated_bytes']/2**20,
                llt_peak_mib=llt['memory']['peak_allocated_bytes']/2**20,
                layerwise_peak_mib=layerwise['memory']['peak_allocated_bytes']/2**20,
                naive_cache_mib=naive['cache_bytes']/2**20,llt_cache_mib=llt['cache_bytes']/2**20,
                layerwise_cache_mib=layerwise['cache_bytes']/2**20,
                naive_graph_ms=naive['graph_timing']['gpu_median_ms'],llt_graph_ms=llt['graph_timing']['gpu_median_ms'],
                layerwise_graph_ms=layerwise['graph_timing']['gpu_median_ms'],
                memory_saving=saving,latency_ratio=latency,nonloop_memory_ratio=relative,
                versus_pytorch_naive_latency_ratio=llt['graph_timing']['gpu_median_ms']/methods['naive','pytorch_flash']['graph_timing']['gpu_median_ms'],
                qualifies=saving>=.5 and latency<=1.2 and relative<=1.25))
    nonloop={train_key(row):row for row in reports['training']['cases'] if row['loops']==1}
    for row in reports['training']['cases']:
        methods={(m['kind'],m['policy']):m for m in row['methods']}
        ll=methods['llt','loop'];nn=methods['naive','none'];nl=methods['naive','loop']
        base=next(m for m in nonloop[train_key(row)]['methods'] if m['kind']=='llt' and m['policy']=='none')
        saving=1-ll['memory']['peak_allocated_bytes']/nn['memory']['peak_allocated_bytes']
        ratio=ll['timing']['gpu_median_ms']/nn['timing']['gpu_median_ms']
        nonloop_ratio=ll['memory']['peak_allocated_bytes']/base['memory']['peak_allocated_bytes']
        result['training'].append(dict(**{k:v for k,v in row.items() if k!='methods'},
            naive_none_peak_mib=nn['memory']['peak_allocated_bytes']/2**20,naive_loop_peak_mib=nl['memory']['peak_allocated_bytes']/2**20,
            llt_loop_peak_mib=ll['memory']['peak_allocated_bytes']/2**20,
            naive_none_ms=nn['timing']['gpu_median_ms'],naive_loop_ms=nl['timing']['gpu_median_ms'],llt_loop_ms=ll['timing']['gpu_median_ms'],
            gross_memory_saving=saving,gross_latency_ratio=ratio,nonloop_memory_ratio=nonloop_ratio,
            same_policy_memory_saving=1-ll['memory']['peak_allocated_bytes']/nl['memory']['peak_allocated_bytes'],
            same_policy_latency_ratio=ll['timing']['gpu_median_ms']/nl['timing']['gpu_median_ms'],
            qualifies=saving>=.5 and ratio<=1.2 and nonloop_ratio<=1.25))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return reports,result


def plots(reports,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    colors={'naive':'#757575','llt':'#167d8d','layerwise':'#d58a33'}
    for kind in ('naive','llt','layerwise'):
        rows=[r for r in reports['inference']['cases'] if r['width']==512 and r['layers']==4 and r['rank']==64 and r['batch']==1 and r['context']==4096]
        rows=sorted(rows,key=lambda r:r['loops'])
        methods=[next(m for m in r['methods'] if m['kind']==kind and m['backend']=='tensor') for r in rows]
        axes[0,0].plot([r['loops'] for r in rows],[m['memory']['peak_allocated_bytes']/2**20 for m in methods],'-o',color=colors[kind],label=kind)
        axes[0,1].plot([r['loops'] for r in rows],[m['graph_timing']['gpu_median_ms'] for m in methods],'-o',color=colors[kind],label=kind)
    axes[0,0].set(xlabel='Loops',ylabel='Peak CUDA allocated (MiB)',title='Decode memory: W512 / L4 / S4096 / R64')
    axes[0,1].set(xlabel='Loops',ylabel='CUDA graph decode (ms)',title='Full Tensor decoder execution')
    rows=sorted([r for r in reports['training']['cases'] if r['width']==512 and r['layers']==2 and r['rank']==32 and r['sequence']==1024 and r['vocab']==4096],key=lambda r:r['loops'])
    for kind,policy,color,style in [('naive','none','#757575','-'),('naive','loop','#757575','--'),('llt','none','#167d8d','-'),('llt','loop','#167d8d','--')]:
        methods=[next(m for m in r['methods'] if m['kind']==kind and m['policy']==policy) for r in rows]
        axes[1,0].plot([r['loops'] for r in rows],[m['memory']['peak_allocated_bytes']/2**20 for m in methods],style+'o',color=color,label=kind+'/'+policy)
    axes[1,0].set(xlabel='Loops',ylabel='Peak CUDA allocated (MiB)',title='Training memory: W512 / L2 / S1024 / R32')
    for backend,color in [('tensor_graph','#167d8d'),('pytorch_flash_graph','#757575')]:
        rows=sorted([r for r in reports['attention']['cases'] if r['batch']==1 and r['kv_heads']==1 and r['query']==1],key=lambda r:r['dim'])
        axes[1,1].plot([r['dim'] for r in rows],[r[backend]['gpu_median_ms']*1000 for r in rows],'-o',color=color,label=backend.replace('_graph',''))
    axes[1,1].set(xlabel='Latent rank',ylabel='CUDA graph attention (µs)',title='Shared-cache decode kernel: B1 / H8 / S4097')
    for ax in axes.flat:
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'scaling.png',dpi=180);fig.savefig(OUT/'scaling.pdf')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plots',action='store_true');a=p.parse_args()
    reports,summary=summarize()
    if a.plots:plots(reports,summary)
    print(json.dumps(summary['counts']))
