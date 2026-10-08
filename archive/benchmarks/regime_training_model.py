"""Training regime with a realistic vocabulary memory floor; CPU, synthetic data."""
import argparse, hashlib, json, statistics
from datetime import datetime, timezone
from pathlib import Path
import regime_training as benchmark


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    template=json.loads((Path(__file__).resolve().parent/'results'/'regime-training'/'report.json').read_text())
    report={key:value for key,value in template.items() if key!='cases'}
    report.update({'status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('regime_training.py'),Path(__file__).with_name('lac_cpu.py'))},'cases':[]})
    report['protocol'].update({'samples':a.samples,'order':'method order rotates by one each paired round',
                               'vocabulary':a.vocab,'scope':'all tokens, full vocabulary cross-entropy and Adam; random targets, no quality claim'})
    def save(): (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    width,h,layers,r,b,s=a.width,a.width//64,a.layers,a.rank,1,a.seq
    common,extra,inp,_=benchmark.fixture(width,h,layers,r,b,s)
    generator=benchmark.torch.Generator().manual_seed(919)
    common[-1]=benchmark.torch.randn(a.vocab,width,generator=generator)*(.15/width**.5)
    target=benchmark.torch.randint(a.vocab,(b,s),generator=generator);f=common,extra,inp,target
    for t in a.loops:
        keys=[(model,policy) for model in ('naive','llt') for policy in ('none','loop')]
        measured={};observations={key:[] for key in keys}
        for model in ('naive','llt'):
            print('Checking',model,'T=',t,flush=True)
            gold=benchmark.trial(f,h,layers,r,t,model,'none',gradients=True,optimizer_step=False)
            for policy in ('none','loop'):
                checked=benchmark.trial(f,h,layers,r,t,model,policy,gradients=True,optimizer_step=False)
                errors=[]
                for actual,expected in zip(checked.pop('gradients'),gold['gradients']):
                    benchmark.torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-4)
                    errors.append(float((actual-expected).abs().max()))
                del checked
                print('Measuring memory',model,policy,'T=',t,flush=True)
                value=benchmark.trial(f,h,layers,r,t,model,policy,memory=True)
                value.pop('elapsed_ms');value.update({'model':model,'policy':policy,
                    'validation':{'passed':True,'gradient_max_absolute_error':max(errors)}})
                measured[model,policy]=value
                benchmark.trial(f,h,layers,r,t,model,policy)
            del gold
        for iteration in range(a.samples):
            order=keys[iteration%len(keys):]+keys[:iteration%len(keys)]
            for model,policy in order:
                result=benchmark.trial(f,h,layers,r,t,model,policy)
                observations[model,policy].append(result['elapsed_ms'])
            print('Timing round',iteration+1,'T=',t,flush=True)
        row={'width':width,'heads':h,'layers':layers,'rank':r,'batch':b,'sequence':s,
             'vocabulary':a.vocab,'loops':t,'methods':[]}
        for key in keys:
            value=measured[key];samples=observations[key]
            value['timing']={'median_ms':statistics.median(samples),'samples_ms':samples,'min_ms':min(samples),'max_ms':max(samples)}
            row['methods'].append(value)
            print(f'W={width} S={s} R={r} T={t} V={a.vocab} {key}',round(value['timing']['median_ms'],3),'ms',
                  round(value['peak_live_tensor_bytes']/2**20,3),'MiB',flush=True)
        ratios=[v/n for v,n in zip(observations['llt','loop'],observations['naive','none'])]
        row['paired_latency_ratios']={'llt_loop_vs_naive_none':ratios,'median':statistics.median(ratios),'max':max(ratios)}
        report['cases'].append(row);save()
    report['status']='passed';save();print('Saved',a.out/'report.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--width',type=int,default=512)
    p.add_argument('--layers',type=int,default=2);p.add_argument('--seq',type=int,default=1024)
    p.add_argument('--rank',type=int,default=32);p.add_argument('--vocab',type=int,default=50257)
    p.add_argument('--loops',nargs='+',type=int,default=[1,16]);p.add_argument('--samples',type=int,default=3)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results'/'regime-training-model')
    run(p.parse_args())
