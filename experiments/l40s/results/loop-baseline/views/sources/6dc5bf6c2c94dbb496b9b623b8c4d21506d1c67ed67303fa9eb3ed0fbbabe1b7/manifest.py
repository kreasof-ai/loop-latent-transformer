"""Describe regenerated views without relabeling historical measurements."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .runtime import ROOT, PUBLISHED_RESULTS, record_label


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(directory):
    views = directory / 'views'
    audit = json.loads((views / 'audit.json').read_text())
    assert audit['status'] == 'passed'
    sources = {}
    for package in ('model', 'inference', 'experiments/l40s'):
        for path in sorted((ROOT / package).glob('*')):
            if path.suffix not in ('.py', '.sh'):
                continue
            sha = digest(path)
            snapshot = views / 'sources' / sha / path.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
            sources[record_label(path)] = sha
    # Only current generator snapshots belong in this regenerated view.
    for directory_entry in (views / 'sources').iterdir():
        if directory_entry.is_dir() and directory_entry.name not in set(sources.values()):
            shutil.rmtree(directory_entry)
    measured = {}
    for name in ('loop-baseline', 'loop-sweep'):
        path = directory.parent / name / 'run-manifest.json'
        if path.exists():
            measured[record_label(path)] = dict(sha256=digest(path), manifest=json.loads(path.read_text()))
    relocation = PUBLISHED_RESULTS / 'relocation.json'
    manifest = dict(
        status='passed', kind='regenerated view; no new GPU measurements',
        case_count=audit['case_count'],
        view_generator_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        view_generator_worktree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()),
        view_generator_sources=sources,
        audit_sha256=digest(views / 'audit.json'),
        measured_sources={key: audit[key] for key in ('numerical_sources', 'extension_sources', 'qualification_driver_source') if key in audit},
        original_measurement_manifests=measured,
        published_relocation_sha256=digest(relocation),
        tensor_repository='https://github.com/kreasof-ai/tensor',
        notes='Source hashes under measured_sources describe frozen measurement snapshots. View generator sources describe the current layout and report tools. Original measurement manifests remain unchanged.')
    (views / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('view manifest passed:', record_label(views / 'run-manifest.json'))
