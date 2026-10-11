"""The LLT backend must remain usable while tensor-torch is replaced."""
import subprocess
import sys
from pathlib import Path
import pytest

pytest.importorskip("torch")


def test_llt_imports_without_tensor_torch():
    source = Path(__file__).resolve().parents[1] / "src"
    script = '''
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class BlockBinding(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'tensor_torch':
            raise ImportError('LLT must not depend on tensor_torch')
sys.meta_path.insert(0, BlockBinding())
from llt_tensor import Operators, AdamW, KVCache, RecurrentOperators
assert not any(name.startswith('tensor_torch') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
