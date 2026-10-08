"""Record provenance for the regenerated combined views (CPU only)."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.l40s.runtime import OUT
from experiments.l40s.manifest import write_manifest


def main():
    write_manifest(OUT)


if __name__ == '__main__':
    main()
