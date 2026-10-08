"""Classify chained CUDA graph OOM exceptions without changing measured code."""

# Support both direct script execution and python -m experiments.l40s.<module>.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('record',type=Path)
    a=parser.parse_args()
    original=a.record.read_bytes()
    data=json.loads(original)
    trace=data.get('traceback','')
    # Graph __exit__ may replace an allocation OOM with AcceleratorError during
    # instantiate. Require explicit CUDA OOM evidence, never generic failures.
    assert data['status']=='error',data['status']
    assert data.get('stage','').endswith('_graph'),data.get('stage')
    assert ('torch.OutOfMemoryError: CUDA out of memory' in trace
            or 'CUDA_ERROR_OUT_OF_MEMORY' in trace),trace
    archive=a.record.parent/'observed-capture-oom'/a.record.name
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():
        assert archive.read_bytes()==original
    else:
        archive.write_bytes(original)
    data['status']='out_of_memory'
    data['classification']=dict(reason='Explicit CUDA allocation OOM chained through graph capture/instantiate',
                                original_record=str(archive.relative_to(a.record.parent)),
                                original_sha256=hashlib.sha256(original).hexdigest(),
                                classifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.record.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(a.record.stem,'out_of_memory',data['stage'],'(original exception archived)')


if __name__=='__main__':
    main()
