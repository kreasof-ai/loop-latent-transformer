"""Verify the provenance and numerical source invariants of the reorganization."""
import ast
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.l40s.runtime import ROOT, PUBLISHED_RESULTS, recorded_path, require_run_directory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_ast(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name), include_attributes=False)


class PreservationTests(unittest.TestCase):
    def test_all_relocated_payloads_are_byte_preserved(self):
        relocation = json.loads((PUBLISHED_RESULTS / 'relocation.json').read_text())
        self.assertEqual(len(relocation['immutable_files']), 1023)
        for item in relocation['immutable_files']:
            with self.subTest(path=item['current']):
                self.assertEqual(sha(ROOT / item['current']), item['sha256'])
                self.assertEqual(recorded_path(item['original']), ROOT / item['current'])

    def test_numerical_model_sources_are_unchanged(self):
        baseline = PUBLISHED_RESULTS / 'loop-baseline'
        sources = json.loads((baseline / 'run-manifest.json').read_text())['sources']
        for old, current in [('model.py', 'model/reference.py'), ('tensor_model.py', 'model/tensor_backend.py')]:
            self.assertEqual(sha(ROOT / current), sources['experiments/l40s/' + old])
        checkpoint = PUBLISHED_RESULTS / 'loop-sweep'
        digest = json.loads((checkpoint / 'audit.json').read_text())['extension_sources']['experiments/l40s/latent_checkpoint_model.py']
        original = ast.parse((checkpoint / 'sources' / digest / 'latent_checkpoint_model.py').read_text())
        current = ast.parse((ROOT / 'model/checkpointing.py').read_text())
        def cls(tree):
            return ast.dump(next(n for n in tree.body if isinstance(n, ast.ClassDef)), include_attributes=False)
        self.assertEqual(cls(original), cls(current))
        self.assertEqual(function_ast(ROOT / 'experiments/l40s/timing.py', 'measure'),
                         function_ast(ROOT / 'archive/experiments/l40s/nanogpt_scale.py', 'measure'))

    def test_architecture_and_parameter_selection_are_unchanged(self):
        baseline = PUBLISHED_RESULTS / 'loop-baseline'
        sources = json.loads((baseline / 'run-manifest.json').read_text())['sources']
        digest = sources['experiments/l40s/loop_sweep.py']
        snapshot = baseline / 'sources' / digest / 'loop_sweep.py'
        for name in ('config', 'parameters'):
            self.assertEqual(function_ast(snapshot, name), function_ast(ROOT / 'experiments/l40s/loop_sweep.py', name))

    def test_original_absolute_artifact_paths_resolve(self):
        original = '/home/sagemaker-user/loop-latent-transformer/benchmarks/results/l40s-loop-sweep/artifacts/example.json'
        self.assertEqual(recorded_path(original), PUBLISHED_RESULTS / 'loop-baseline/artifacts/example.json')

    def test_published_output_is_rejected_for_new_profiles(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'fresh run directory'):
                require_run_directory()
        with patch.dict(os.environ, {'LLT_RESULTS_ROOT': '/tmp/llt-new-test'}, clear=True):
            require_run_directory()


if __name__ == '__main__':
    unittest.main()
