"""Evaluate memory and latency criteria from completed regime measurements."""
import argparse, hashlib, json
from pathlib import Path


def assess(a):
    directory=Path(__file__).resolve().parent/'results'
    reports={name:json.loads((directory/name/'report.json').read_text()) for name in ('regime-inference','regime-training')}
    for name in ('regime-boundary','regime-repeat','regime-training-repeat','regime-training-model','regime-training-model-deep'):
        path=directory/name/'report.json'
        if path.exists(): reports[name]=json.loads(path.read_text())
    assert all(r['status']=='passed' for r in reports.values())
    summary={'schema':'llt.regime-summary.v1','criteria':{'max_latency_ratio':a.tax+1,'min_memory_saving':a.saving,
                    'max_nonloop_memory_ratio':a.nonloop},'source_report_sha256':{
                        name:hashlib.sha256((directory/name/'report.json').read_bytes()).hexdigest() for name in reports},
             'inference':[],'training':[]}
    inference=[];training=[]
    for name,report in reports.items():
        target=inference if report['schema']=='llt.regime-inference.v1' else training
        for original in report['cases']:
            c={**original,'source_run':name}
            if target is training: c['vocabulary']=c.get('vocabulary',128)
            target.append(c)
    base={}
    for c in inference:
        key=tuple(c[n] for n in ('width','heads','layers','rank','context_history'))
        if c['loops']==1: base[key]={m['method']:m for m in c['methods']}
    for c in inference:
        if c['loops']==1: continue
        key=tuple(c[n] for n in ('width','heads','layers','rank','context_history'))
        methods={m['method']:m for m in c['methods']};llt,naive,lla=[methods[n] for n in ('llt','naive','lla')]
        row={n:c[n] for n in ('source_run','width','heads','layers','rank','context_history','loops')}
        row.update({'llt_peak_mib':llt['peak_live_tensor_bytes']/2**20,'naive_peak_mib':naive['peak_live_tensor_bytes']/2**20,
            'lla_peak_mib':lla['peak_live_tensor_bytes']/2**20,
            'llt_ms':llt['timing']['median_ms'],'naive_ms':naive['timing']['median_ms'],'lla_ms':lla['timing']['median_ms'],
            'memory_saving_vs_naive':1-llt['peak_live_tensor_bytes']/naive['peak_live_tensor_bytes'],
            'memory_saving_vs_lla':1-llt['peak_live_tensor_bytes']/lla['peak_live_tensor_bytes'],
            'latency_ratio_vs_naive':llt['timing']['median_ms']/naive['timing']['median_ms'],
            'latency_ratio_vs_lla':llt['timing']['median_ms']/lla['timing']['median_ms'],
            'memory_ratio_vs_nonloop_llt':llt['peak_live_tensor_bytes']/base[key]['llt']['peak_live_tensor_bytes'],
            'memory_ratio_vs_nonloop_naive':llt['peak_live_tensor_bytes']/base[key]['naive']['peak_live_tensor_bytes']})
        row['qualifies']=row['memory_saving_vs_naive']>=a.saving and row['latency_ratio_vs_naive']<=a.tax+1 and row['memory_ratio_vs_nonloop_llt']<=a.nonloop
        summary['inference'].append(row)
    base={}
    for c in training:
        key=tuple(c[n] for n in ('width','heads','layers','rank','batch','sequence','vocabulary'))
        if c['loops']==1: base[key]={m['model']:m for m in c['methods'] if m['policy']=='none'}
    for c in training:
        if c['loops']==1: continue
        key=tuple(c[n] for n in ('width','heads','layers','rank','batch','sequence','vocabulary'))
        methods={(m['model'],m['policy']):m for m in c['methods']};naive=methods[('naive','none')]
        for policy in ('none','mlp','loop','pair'):
            if ('llt',policy) not in methods: continue
            llt=methods[('llt',policy)];fair=methods[('naive',policy)]
            eligible=[m for m in c['methods'] if m['model']=='naive' and m['peak_live_tensor_bytes']<=llt['peak_live_tensor_bytes']*1.05]
            fastest=min(eligible,key=lambda m:m['timing']['median_ms']) if eligible else None
            row={n:c[n] for n in ('source_run','width','heads','layers','rank','batch','sequence','vocabulary','loops')}
            row.update({'policy':policy,'llt_peak_mib':llt['peak_live_tensor_bytes']/2**20,
                'naive_peak_mib':naive['peak_live_tensor_bytes']/2**20,'naive_same_policy_peak_mib':fair['peak_live_tensor_bytes']/2**20,
                'nonloop_llt_peak_mib':base[key]['llt']['peak_live_tensor_bytes']/2**20,
                'nonloop_naive_peak_mib':base[key]['naive']['peak_live_tensor_bytes']/2**20,
                'llt_ms':llt['timing']['median_ms'],'naive_ms':naive['timing']['median_ms'],
                'naive_same_policy_ms':fair['timing']['median_ms'],
                'memory_saving_vs_naive':1-llt['peak_live_tensor_bytes']/naive['peak_live_tensor_bytes'],
                'memory_saving_vs_naive_same_policy':1-llt['peak_live_tensor_bytes']/fair['peak_live_tensor_bytes'],
                'latency_ratio_vs_naive':llt['timing']['median_ms']/naive['timing']['median_ms'],
                'latency_ratio_vs_naive_same_policy':llt['timing']['median_ms']/fair['timing']['median_ms'],
                'memory_ratio_vs_nonloop_llt':llt['peak_live_tensor_bytes']/base[key]['llt']['peak_live_tensor_bytes'],
                'memory_ratio_vs_nonloop_naive':llt['peak_live_tensor_bytes']/base[key]['naive']['peak_live_tensor_bytes'],
                'naive_at_similar_memory':None if fastest is None else {'policy':fastest['policy'],'ms':fastest['timing']['median_ms'],
                      'peak_mib':fastest['peak_live_tensor_bytes']/2**20}})
            row['qualifies']=row['memory_saving_vs_naive']>=a.saving and row['latency_ratio_vs_naive']<=a.tax+1 and row['memory_ratio_vs_nonloop_llt']<=a.nonloop
            summary['training'].append(row)
    for name in ('inference','training'):
        summary[name+'_qualifying_count']=sum(c['qualifies'] for c in summary[name])
        summary[name+'_counts_by_tax']={str(tax):sum(c['memory_saving_vs_naive']>=a.saving and c['memory_ratio_vs_nonloop_llt']<=a.nonloop and c['latency_ratio_vs_naive']<=1+tax for c in summary[name]) for tax in (0,.1,.2,.3,.5,1.)}
    out=directory/'regime-summary.json';out.write_text(json.dumps(summary,indent=2)+'\n')
    for name in ('inference','training'):
        print(name,summary[name+'_qualifying_count'],'qualifying; tax frontiers',summary[name+'_counts_by_tax'])
        for c in summary[name]:
            if c['qualifies']: print(c)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tax',type=float,default=.2)
    p.add_argument('--saving',type=float,default=.5);p.add_argument('--nonloop',type=float,default=1.25)
    assess(p.parse_args())
