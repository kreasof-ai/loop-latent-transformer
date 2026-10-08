"""Result locations and relocation of immutable historical record paths."""
from functools import lru_cache
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_RESULTS = ROOT / 'experiments/l40s/results'


def results_root():
    configured = os.environ.get('LLT_RESULTS_ROOT')
    return Path(configured).expanduser().resolve() if configured else PUBLISHED_RESULTS


BASE = results_root() / 'loop-baseline'
OUT = results_root() / 'loop-sweep'
VIEWS = OUT / 'views'


def require_run_directory():
    """Keep new numerical-source revisions out of the published measurements."""
    if results_root() == PUBLISHED_RESULTS:
        raise ValueError('Set LLT_RESULTS_ROOT to a fresh run directory; the published results are immutable.')


def recorded_path(value):
    """Resolve old report paths without rewriting their measured payloads."""
    path = Path(value)
    logical = str(value)
    marker = '/benchmarks/results/'
    if marker in logical:
        logical = 'benchmarks/results/' + logical.split(marker, 1)[1]
    elif path.is_absolute():
        try:
            logical = str(path.relative_to(ROOT))
        except ValueError:
            return path
    for old, new in relocation_moves():
        if logical == old or logical.startswith(old + '/'):
            return ROOT / (new + logical[len(old):])
    return ROOT / logical if not path.is_absolute() else path


@lru_cache(maxsize=1)
def relocation_moves():
    relocation = json.loads((PUBLISHED_RESULTS / 'relocation.json').read_text())
    return sorted(relocation['moves'].items(), key=lambda pair: len(pair[0]), reverse=True)


def record_label(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())
