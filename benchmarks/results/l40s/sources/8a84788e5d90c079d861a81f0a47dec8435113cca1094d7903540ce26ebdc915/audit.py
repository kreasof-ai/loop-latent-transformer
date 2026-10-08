"""Audit completed reports, exact-SM binaries and preserved source snapshots."""
import hashlib
import json
import math
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'benchmarks/results/l40s'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    binaries={};sources=set();reports={}
    for name in ('attention','attention-serial','decode-partitions','inference','prefill','training','repeat','correctness'):
        path=OUT/(name+'.json');report=json.loads(path.read_text())
        assert report['status']=='passed',name
        reports[name]=sha(path)
        for filename,digest in report.get('environment',{}).get('sources',{}).items():
            snapshot=OUT/'sources'/digest/filename
            assert snapshot.exists() and sha(snapshot)==digest,str(snapshot)
            sources.add(digest)
        for artifact in report.get('artifacts',[]):
            path=ROOT/artifact['path']
            assert sha(path)==artifact['sha256'],str(path)
            assert sha(path.with_suffix('.py'))==artifact['source_sha256'],str(path)
            with zipfile.ZipFile(path) as bundle:
                manifest=json.loads(bundle.read('manifest.json'))
                assert manifest['provider']=='cuda' and manifest['target']=='sm_89'
            binaries[str(path.relative_to(ROOT))]=artifact['sha256']
        if name in ('inference','prefill'):
            for row in report['cases']:
                assert len(row['methods'])==6
                for method in row['methods']:
                    assert method['validation']=='passed'
                    assert len(method['graph_timing']['gpu_samples_ms'])==9
                    assert 'memory' in method['graph_timing']
        if name=='training':
            for row in report['cases']:
                assert {(m['kind'],m['policy']) for m in row['methods']}=={(k,p) for k in ('naive','llt','layerwise') for p in ('none','loop')}
                for method in row['methods']:
                    assert math.isfinite(method['loss'])
                    assert len(method['timing']['gpu_samples_ms'])==9
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['status']=='passed'
    for name,digest in summary['raw_report_hashes'].items():assert reports[name]==digest,name
    result=dict(status='passed',report_sha256=reports,unique_binaries=len(binaries),
                source_snapshots=len(sources),target='sm_89',binary_sha256=binaries,
                checks=['report completion','binary hashes','export source hashes','preserved dependency source hashes',
                        'CUDA sm_89 targets','complete baseline/checkpoint policies','finite losses','nine-sample graph and training observations',
                        'eager and captured-graph memory separately present','derived summary input hashes'])
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Audit passed:',len(reports),'reports;',len(binaries),'unique CUDA binaries;',len(sources),'source snapshots')


if __name__=='__main__':run()
