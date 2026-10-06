"""Boundary and isolated repeat cases for the inference regime search."""
import argparse, hashlib, json
from pathlib import Path
import regime_inference as benchmark


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tensor-root',type=Path,default=Path(__file__).resolve().parents[2]/'tensor')
    p.add_argument('--phase',choices=['boundary','repeat'],default='boundary')
    p.add_argument('--samples',type=int,default=7);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path);a=p.parse_args()
    if a.out is None: a.out=Path(__file__).resolve().parent/'results'/('regime-'+a.phase)
    specs=[(512,32,6),(512,32,7),(512,64,9)]+[(4096,r,t) for r in (96,128) for t in (1,10)] if a.phase=='boundary' else [(4096,r,10) for r in (32,64)]
    benchmark.configurations=lambda smoke:[(*spec,768,12,12) for spec in specs]
    a.smoke=False;benchmark.run(a)
    path=a.out/'report.json';report=json.loads(path.read_text())
    report['source_hashes'][Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['protocol']['configuration_selection']='regime_boundary.py: '+a.phase
    path.write_text(json.dumps(report,indent=2)+'\n')
