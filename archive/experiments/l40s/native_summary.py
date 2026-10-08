"""Derive matched resource criteria and export figures for the BF16 study."""
import argparse
import json
from pathlib import Path

from common import ROOT, digest, save

OUT=ROOT/'benchmarks/results/l40s-native'


def geometry(row):
    return tuple(row[k] for k in ('width','heads','layers','batch','sequence','rank'))+(row.get('vocab',256),)


def run(plots=False):
    training=json.loads((OUT/'training.json').read_text())
    inference=json.loads((OUT/'inference.json').read_text())
    assert training['status']==inference['status']=='passed'
    source=Path(__file__)
    source_sha=digest(source)
    snapshot=OUT/'sources'/source_sha/source.name
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_bytes(source.read_bytes())
    result=dict(status='passed',summary_source_sha256=source_sha,
                report_sha256={name:digest(OUT/(name+'.json')) for name in ('correctness','training','inference')},
                criteria=dict(memory_ratio_max=0.5,latency_ratio_max=1.2,nonloop_memory_ratio_max=1.25),training=[],inference=[])
    nonloop={geometry(row):row for row in training['cases'] if row['loops']==1}
    for row in training['cases']:
        methods={(m['kind'],m['policy'],m['backend']):m for m in row['methods']}
        base={(m['kind'],m['policy'],m['backend']):m for m in nonloop[geometry(row)]['methods']}
        for backend in ('torch','tensor'):
            for policy in ('none','loop'):
                naive,llt=methods['naive',policy,backend],methods['llt',policy,backend]
                peak=llt['memory']['peak_allocated_bytes']
                mr=peak/naive['memory']['peak_allocated_bytes']
                tr=llt['timing']['wall_median_ms']/naive['timing']['wall_median_ms']
                nr=peak/base['llt',policy,backend]['memory']['peak_allocated_bytes']
                result['training'].append(dict(geometry=dict(row,methods=None),backend=backend,policy=policy,
                    memory_ratio=mr,wall_latency_ratio=tr,nonloop_memory_ratio=nr,qualifies=mr<=0.5 and tr<=1.2 and nr<=1.25))
    nonloop={geometry(row):row for row in inference['cases'] if row['loops']==1}
    for row in inference['cases']:
        methods={(m['kind'],m['backend']):m for m in row['methods']}
        base={(m['kind'],m['backend']):m for m in nonloop[geometry(row)]['methods']}
        for backend in ('torch','tensor'):
            naive,llt=methods['naive',backend],methods['llt',backend]
            for phase in ('prefill','four_token_decode'):
                for mode in ('eager','graph'):
                    def peak(method):
                        data=method[phase]
                        return (data['graph_timing']['memory'] if mode=='graph' else data['memory'])['peak_allocated_bytes']
                    def latency(method):
                        data=method[phase]
                        return data['graph_timing']['gpu_median_ms'] if mode=='graph' else data['timing']['wall_median_ms']
                    mr=peak(llt)/peak(naive)
                    tr=latency(llt)/latency(naive)
                    nr=peak(llt)/peak(base['llt',backend])
                    result['inference'].append(dict(geometry=dict(row,methods=None),backend=backend,phase=phase,mode=mode,
                        memory_ratio=mr,latency_ratio=tr,nonloop_memory_ratio=nr,qualifies=mr<=0.5 and tr<=1.2 and nr<=1.25))
    result['counts']=dict(training_comparisons=len(result['training']),training_qualifying=sum(x['qualifies'] for x in result['training']),
                         inference_comparisons=len(result['inference']),inference_qualifying=sum(x['qualifies'] for x in result['inference']))
    save(OUT/'summary.json',result)
    print(json.dumps(result['counts'],indent=2))
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        styles={'naive':('o','C1'),'llt':('s','C0')}
        for policy,ax in zip(('none','loop'),axes[0,:2]):
            for kind in styles:
                for backend,linestyle in (('tensor','-'),('torch','--')):
                    selected=[row for row in training['cases'] if row['width']==128 and row['rank']==32]
                    xs,ys=[],[]
                    for row in selected:
                        m=next(m for m in row['methods'] if (m['kind'],m['policy'],m['backend'])==(kind,policy,backend))
                        xs.append(row['loops']);ys.append(m['memory']['peak_allocated_bytes']/2**20)
                    marker,color=styles[kind]
                    ax.plot(xs,ys,marker=marker,color=color,linestyle=linestyle,label=f'{kind} {backend}')
            ax.set_title(f'Training memory: {policy} checkpoints')
            ax.set_ylabel('Peak allocated MiB');ax.set_xlabel('Loops');ax.grid(alpha=.2)
        for kind in styles:
            for backend,linestyle in (('tensor','-'),('torch','--')):
                selected=[row for row in training['cases'] if row['width']==128 and row['rank']==32]
                methods=[next(m for m in row['methods'] if (m['kind'],m['policy'],m['backend'])==(kind,'loop',backend)) for row in selected]
                marker,color=styles[kind]
                axes[0,2].plot([row['loops'] for row in selected],[m['timing']['wall_median_ms'] for m in methods],
                              marker=marker,color=color,linestyle=linestyle,label=f'{kind} {backend}')
        axes[0,2].set_title('Training latency: loop checkpoints');axes[0,2].set_ylabel('Median wall ms');axes[0,2].set_xlabel('Loops')
        for col,key,title in ((0,'cache','Physical cache'),(1,'peak','Decode peak allocation'),(2,'latency','Four-token decode graph latency')):
            ax=axes[1,col]
            for kind in ('naive','llt','layerwise'):
                selected=[row for row in inference['cases'] if row['width']==128]
                methods=[next(m for m in row['methods'] if (m['kind'],m['backend'])==(kind,'tensor')) for row in selected]
                values=[m['cache_bytes']/2**20 if key=='cache' else m['four_token_decode']['graph_timing']['memory']['peak_allocated_bytes']/2**20
                        if key=='peak' else m['four_token_decode']['graph_timing']['gpu_median_ms'] for m in methods]
                ax.plot([row['loops'] for row in selected],values,marker='o',label=kind)
            ax.set_title(title);ax.set_xlabel('Loops');ax.set_ylabel('MiB' if key!='latency' else 'Median GPU ms');ax.grid(alpha=.2)
            if key=='cache':
                ax.set_yscale('log')
                ax.set_ylabel('MiB (log scale)')
        axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
        fig.suptitle('L40S qualified Tensor backend — W128, L2, S257, R32 (synthetic resource study)')
        fig.tight_layout()
        fig.savefig(OUT/'scaling.png',dpi=160);fig.savefig(OUT/'scaling.pdf')
        result['figures']={name:digest(OUT/name) for name in ('scaling.png','scaling.pdf')}
        save(OUT/'summary.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plots',action='store_true')
    run(parser.parse_args().plots)
