"""Strict audit, combined tables and standalone figures for the exact sweep."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'benchmarks/results/l40s-loop-sweep'
OUT=ROOT/'benchmarks/results/l40s-latent-checkpoint-sweep'
VARIANTS=(('llt',32),('llt',64),('llt',128),('naive_loop',64),('stacked',64),('fixed_depth',64))
LABELS={('llt',r):f'LLT rank {r}' for r in (32,64,128)} | {
    ('naive_loop',64):'Naive Loop',('stacked',64):'Independent stack',('fixed_depth',64):'Fixed depth / matched params'}
METRICS=('training_eager','training_graph','prompt_eager','prompt_graph','startup_eager','startup_graph','decode_eager','decode_graph')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((BASE/'audit.json').read_text())['status']=='passed'
    records={};inputs={};artifacts={};artifact_files=set();profiles=set();extension_profiles=set();checks=[];full_checks=[]
    def verify(path,d,extension):
        directory=path.parent
        inputs[str(path.relative_to(ROOT))]=sha(path)
        assert d['provenance']['native_cpp_executor'] and d['provenance']['gpu']=='NVIDIA L40S'
        profiles.add(tuple(sorted(d['provenance']['sources'].items())))
        if extension:
            assert d['checkpoint_exact']
            extension_profiles.add(tuple(sorted(d['extension_sources'].items())))
        for name,digest in (d['provenance']['sources'] | d.get('extension_sources',{})).items():
            assert sha(directory/'sources'/digest/Path(name).name)==digest
        if 'classification' in d:
            assert sha(directory/d['classification']['original_record'])==d['classification']['original_sha256']
        coverage=d.get('coverage')
        if coverage:
            assert not coverage['fallbacks']
            for artifact in coverage['artifacts']:
                assert artifact['target']=='sm_89'
                identity=(artifact['path'],artifact['sha256'])
                if identity not in artifact_files:
                    assert sha(Path(artifact['path']))==artifact['sha256'];artifact_files.add(identity)
                artifacts[artifact['sha256']]=artifact
    for directory,extension in ((BASE,False),(OUT,True)):
        for path in sorted(directory.glob('*.json')):
            if not path.name.startswith(('training-','inference-','qualification-','full_qualification-')):continue
            d=json.loads(path.read_text());a=d['arguments']
            assert d['status'] in ('passed','out_of_memory'),(path,d.get('error'))
            verify(path,d,extension)
            if a['phase'] in ('qualification','full_qualification'):
                assert d['status']=='passed'
                for key,result in d['policy_checks'].items():
                    if key.endswith('_cache'):
                        assert result['max_abs_error']<.05 and result['relative_l2']<.03
                    else:
                        assert result['all_gradients_finite']
                        assert result['gradient_error']['maximum_parameter_relative_l2']<1e-5
                (full_checks if a['phase']=='full_qualification' else checks).append((path,d))
                continue
            rank=a.get('rank',64);policy=a.get('policy','none')
            key=(a['phase'],a['model'],rank,a['backend'],a['loops'],policy)
            assert key not in records,key
            assert d['batch_size']==4 and d['sequence_length']==1024 and d['checkpoint_policy']==policy
            c=d['config']
            assert (c['width'],c['heads'],c['rank'],c['vocab'],c['max_seq'])==(768,12,rank,50304,1025)
            assert d['effective_depth']==(12 if a['model']=='fixed_depth' else 12*a['loops'])
            assert d['unique_layers']==(12*a['loops'] if a['model']=='stacked' else 12)
            if a['model']=='fixed_depth':assert d['parameter_count']==d['stacked_parameter_target']
            assert policy!='lac' or a['model']=='llt'
            row=dict(phase=a['phase'],model=a['model'],kv_rank=rank if a['model']=='llt' else '',backend=a['backend'],
                     loops=a['loops'],checkpoint=policy,status=d['status'],failed_stage=d.get('stage',''),
                     parameter_count=d['parameter_count'],effective_depth=d['effective_depth'],unique_layers=d['unique_layers'],
                     measurement_origin='extension' if extension else 'original',raw_record=str(path.relative_to(ROOT)))
            for metric in METRICS:
                if metric not in d:continue
                m=d[metric]
                assert len(m['gpu_samples_ms'])==a['samples']==9
                assert all(math.isfinite(v) and v>0 for v in m['gpu_samples_ms'])
                row[metric+'_gpu_ms']=m['gpu_median_ms']
                row[metric+'_peak_gib']=m.get('capture_peak_allocated_bytes',m.get('peak_allocated_bytes'))/2**30
                if 'wall_median_ms' in m:row[metric+'_wall_ms']=m['wall_median_ms']
                if 'capture_peak_reserved_bytes' in m:row[metric+'_reserved_gib']=m['capture_peak_reserved_bytes']/2**30
            if 'training_graph' in d:
                counts=d['graph_step_counter']
                assert counts['min']==counts['max']==counts['expected']==42
                assert d['all_parameter_gradients_finite'] and math.isfinite(d['graph_last_loss'])
            if a['phase']=='inference':
                row['cache_mib']=d['cache_bytes']/2**20
                row['fold_mib']=d['fold_bytes']/2**20
                row['prepared_weight_mib']=d['prepared_weight_bytes']/2**20
            records[key]=row
    expected=set()
    for model,rank in VARIANTS:
        for loops,backend in itertools.product(range(1,17),('tensor','torch')):
            expected.add(('inference',model,rank,backend,loops,'none'))
            for policy in (('none','ac','lac') if model=='llt' else ('none','ac')):
                expected.add(('training',model,rank,backend,loops,policy))
    assert set(records)==expected,f'Missing {len(expected-set(records))} cases: {sorted(expected-set(records))[:8]}'
    assert len(profiles)==len(extension_profiles)==1,'numerical sources changed during the campaign'
    assert {(d['arguments']['model'],d['arguments']['rank'],d['arguments']['loops']) for _,d in checks}=={
        (m,r,t) for m,r in VARIANTS for t in (4,16)}
    assert {(d['arguments']['model'],d['arguments']['rank'],d['arguments']['backend'],d['arguments']['loops']) for _,d in full_checks}=={
        (m,r,b,16) for m,r in VARIANTS for b in ('tensor','torch')}
    rows=[records[key] for key in sorted(records)]
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/'combined.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (OUT/'combined.json').write_text(json.dumps(rows,indent=2)+'\n')
    diagnostics=[]
    for path,d in checks+full_checks:
        a=d['arguments']
        for name,result in d['policy_checks'].items():
            if name.endswith('_cache'):continue
            backend,policy=name.split('_',1)
            diagnostics.append(dict(model=a['model'],kv_rank=a['rank'] if a['model']=='llt' else '',loops=a['loops'],
                backend=backend,policy=policy,batch=d['batch_size'],seq=d['sequence_length'],width=d['config']['width'],
                gradient_global_relative_l2=result['gradient_error']['global_relative_l2'],
                maximum_parameter_relative_l2=result['gradient_error']['maximum_parameter_relative_l2'],
                gradient_cosine=result['gradient_error']['cosine'],raw_record=str(path.relative_to(ROOT))))
    with (OUT/'exact-gradient-checks.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(diagnostics[0]),lineterminator='\n');writer.writeheader();writer.writerows(diagnostics)
    plots(records)
    (OUT/'tables.md').write_text(tables(records))
    audit=dict(status='passed',case_count=len(rows),new_case_count=sum(r['measurement_origin']=='extension' for r in rows),
        reused_case_count=sum(r['measurement_origin']=='original' for r in rows),
        passed=sum(r['status']=='passed' for r in rows),out_of_memory=sum(r['status']=='out_of_memory' for r in rows),
        qualification_count=len(checks),full_qualification_count=len(full_checks),unique_artifact_count=len(artifacts),
        numerical_sources=dict(next(iter(profiles))),extension_sources=dict(next(iter(extension_profiles))),
        input_sha256=inputs,artifacts=list(artifacts.values()))
    assert audit['case_count']==672 and audit['new_case_count']==416 and audit['reused_case_count']==256
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('audit passed',audit['case_count'],'combined cases,',len(checks),'small +',len(full_checks),'full checks,',len(artifacts),'artifacts')


def cell(records,phase,model,rank,backend,loops,policy,metric):
    if policy=='lac' and model!='llt':return 'N/A'
    row=records[(phase,model,rank,backend,loops,policy)]
    if metric+'_gpu_ms' not in row:
        return 'OOM' if row['failed_stage']==metric else '— (earlier OOM)'
    return f"{row[metric+'_gpu_ms']:.2f} / {row[metric+'_peak_gib']:.2f}"


def tables(records):
    text='Cells are **CUDA graph ms / capture peak allocated GiB**, per batch of four. S=1024. All checkpoint policies are exact. LAC is N/A without an existing latent boundary. AC checkpoints full blocks; LAC checkpoints the latent attention/output region.\n\n'
    for backend in ('tensor','torch'):
        text+=f'### {backend.capitalize()} — all loop counts\n\n'
        text+='| T | Architecture | Train: none | Train: AC | Train: LAC | Prompt inference | Cached decode |\n|---:|---|---:|---:|---:|---:|---:|\n'
        for loops in range(1,17):
            for model,rank in VARIANTS:
                values=[cell(records,'training',model,rank,backend,loops,p,'training_graph') for p in ('none','ac','lac')]
                values += [cell(records,'inference',model,rank,backend,loops,'none',metric) for metric in ('prompt_graph','decode_graph')]
                text+=f'| {loops} | {LABELS[(model,rank)]} | '+' | '.join(values)+' |\n'
        text+='\n'
    return text


def plots(records):
    colors=dict(none='#333333',ac='#137ab2',lac='#b66b13')
    for metric,title in (('training_graph','Captured training'),('training_eager','Eager training')):
        for suffix,label in (('gpu_ms','GPU latency (ms)'),('peak_gib','Peak allocated memory (GiB)')):
            fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
            for ax,(model,rank) in zip(axes.flat,VARIANTS):
                for backend in ('tensor','torch'):
                    for policy in (('none','ac','lac') if model=='llt' else ('none','ac')):
                        values=[records[('training',model,rank,backend,t,policy)].get(metric+'_'+suffix,float('nan')) for t in range(1,17)]
                        ax.plot(range(1,17),values,color=colors[policy],linestyle='-' if backend=='tensor' else '--',
                            marker='.' if backend=='tensor' else None,label=backend+' '+policy.upper())
                ax.set_title(LABELS[(model,rank)]);ax.set_xlabel('Loop / depth multiplier T');ax.set_ylabel(label)
                ax.set_xticks((1,4,8,12,16));ax.grid(alpha=.2)
            axes.flat[0].legend(fontsize=8,ncol=2)
            fig.suptitle(title+' · L40S · B4/S1024 · exact AC and native-boundary LAC')
            fig.tight_layout()
            for extension in ('png','pdf'):fig.savefig(OUT/f'{metric}-{suffix}.{extension}',dpi=170)
            plt.close(fig)
    for metric in ('prompt_graph','startup_graph','decode_graph'):
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for index,(model,rank) in enumerate(VARIANTS):
            for backend in ('tensor','torch'):
                for ax,suffix in zip(axes,('gpu_ms','peak_gib')):
                    values=[records[('inference',model,rank,backend,t,'none')][metric+'_'+suffix] for t in range(1,17)]
                    ax.plot(range(1,17),values,color=f'C{index}',linestyle='-' if backend=='tensor' else '--',
                            label=LABELS[(model,rank)]+' ('+backend+')')
        for ax in axes:ax.set_xlabel('Loop / depth multiplier T');ax.grid(alpha=.2)
        axes[0].set_ylabel('GPU latency (ms)');axes[1].set_ylabel('Capture peak allocated memory (GiB)')
        axes[0].legend(fontsize=7,ncol=2)
        fig.suptitle(metric.replace('_',' ')+' · L40S · B4/S1024 · checkpoint policies do not apply')
        fig.tight_layout()
        for extension in ('png','pdf'):fig.savefig(OUT/f'{metric}.{extension}',dpi=170)
        plt.close(fig)


if __name__=='__main__':main()
