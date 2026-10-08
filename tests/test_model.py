"""CPU numerical checks against frozen sources and exact checkpoint boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
from model import Config, Transformer
from model.checkpointing import CheckpointTransformer
from model.tensor_backend import BackendTransformer
from inference.prepared import prepare
from experiments.l40s.loop_sweep import config, parameters
from experiments.l40s.runtime import PUBLISHED_RESULTS


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(directory, source, field='sources'):
    manifest = json.loads((directory / 'run-manifest.json').read_text())
    digest = manifest[field][source]
    return directory / 'sources' / digest / Path(source).name


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        baseline = PUBLISHED_RESULTS / 'loop-baseline'
        sweep = PUBLISHED_RESULTS / 'loop-sweep'
        cls.old_backend = load_source('frozen_tensor_model', snapshot(baseline, 'experiments/l40s/tensor_model.py'))
        with patch.dict(sys.modules, {'tensor_model': cls.old_backend}):
            cls.old_checkpoint = load_source('frozen_checkpoint_model', snapshot(sweep, 'experiments/l40s/latent_checkpoint_model.py'))
        dummy = types.ModuleType('nanogpt_adapter')
        dummy.graph_compatible = lambda model: model
        with patch.dict(sys.modules, {'tensor_model': cls.old_backend, 'nanogpt_adapter': dummy}):
            cls.old_serving = load_source('frozen_prepared_inference', snapshot(baseline, 'experiments/l40s/prepared_inference.py'))

    def tiny_config(self, kind='llt', loops=3, layers=2):
        return Config(kind=kind, width=32, heads=2, layers=layers, loops=loops,
                      rank=8, vocab=31, max_seq=16, gelu='none')

    def test_folding_preserves_logits_and_every_gradient(self):
        torch.manual_seed(17)
        model = Transformer(self.tiny_config()).double()
        tokens = torch.randint(31, (2, 7))
        gradients = []
        outputs = []
        for unfolded in (True, False):
            model.zero_grad(set_to_none=True)
            output = model(tokens, unfolded=unfolded)
            output.square().sum().backward()
            outputs.append(output.detach())
            gradients.append({n: p.grad.clone() for n, p in model.named_parameters()})
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-10, atol=1e-12)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-9, atol=1e-11)

    def test_moved_checkpoints_match_frozen_sources_and_uncheckpointed_gradients(self):
        torch.manual_seed(21)
        tokens = torch.randint(31, (2, 7))
        for kind in ('llt', 'naive'):
            current = CheckpointTransformer(self.tiny_config(kind)).double()
            original = self.old_checkpoint.CheckpointTransformer(current.c).double()
            original.load_state_dict(current.state_dict())
            reference = None
            for policy in (('none', 'ac', 'lac') if kind == 'llt' else ('none', 'ac')):
                results = []
                for model in (original, current):
                    model.zero_grad(set_to_none=True)
                    output = model(tokens, policy=policy)
                    output.square().sum().backward()
                    results.append((output.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}))
                with self.subTest(kind=kind, policy=policy):
                    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
                    for name in results[0][1]:
                        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=0, atol=0)
                    if reference is None:
                        reference = results[1]
                    torch.testing.assert_close(reference[0], results[1][0], rtol=0, atol=0)
                    for name in reference[1]:
                        torch.testing.assert_close(reference[1][name], results[1][1][name], rtol=1e-9, atol=1e-11)

    def test_cached_decode_matches_full_forward_and_frozen_implementation(self):
        torch.manual_seed(22)
        tokens = torch.randint(31, (2, 10))
        for kind in ('llt', 'naive'):
            for loops in (1, 3):
                current = BackendTransformer(self.tiny_config(kind, loops)).double()
                original = self.old_backend.BackendTransformer(current.c).double()
                original.load_state_dict(current.state_dict())
                with self.subTest(kind=kind, loops=loops), torch.inference_mode():
                    actual = []
                    for model in (original, current):
                        _, state = model.prefill(tokens[:, :7], capacity=12)
                        for offset in range(7, 10):
                            decoded = model.decode_token(tokens[:, offset:offset+1], state)
                            full = model(tokens[:, :offset+1])[:, -1:]
                            torch.testing.assert_close(decoded, full, rtol=1e-10, atol=1e-12)
                            actual.append(decoded)
                        self.assertEqual(len(state['caches']), 1 if kind == 'llt' else 2 * loops)
                    for old, new in zip(actual[:3], actual[3:]):
                        torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_serving_copies_match_original_and_refresh_after_weight_update(self):
        torch.manual_seed(23)
        current = prepare(BackendTransformer(self.tiny_config()).float())
        original = self.old_serving.prepare(self.old_backend.BackendTransformer(current.c).float())
        original.load_state_dict(current.state_dict())
        x = torch.randn(2, 32).bfloat16()
        with self.assertRaisesRegex(ValueError, 'inference_mode'):
            current.linear(x, current.output.weight)
        with torch.inference_mode():
            for model in (original, current):
                first = model.linear(x, model.output.weight).clone()
                model.output.weight.add_(.125)
                second = model.linear(x, model.output.weight).clone()
                expected = torch.nn.functional.linear(x, model.output.weight.bfloat16())
                torch.testing.assert_close(second, expected, rtol=0, atol=0)
                self.assertFalse(torch.equal(first, second))
            torch.testing.assert_close(original.linear(x, original.output.weight), current.linear(x, current.output.weight), rtol=0, atol=0)

    def test_parameter_match_for_every_measured_loop_count(self):
        for loops in range(1, 17):
            with self.subTest(loops=loops):
                self.assertEqual(parameters(config('stacked', loops)), parameters(config('fixed_depth', loops)))
        for variant in ('llt', 'naive_loop', 'stacked', 'fixed_depth'):
            c = config(variant, 2, small=True)
            model = BackendTransformer(c)
            self.assertEqual(parameters(c), sum(p.numel() for p in model.parameters()))

    def test_conventional_lac_is_rejected(self):
        model = CheckpointTransformer(self.tiny_config('naive'))
        with self.assertRaisesRegex(ValueError, 'existing LLT latent boundary'):
            model(torch.zeros((1, 2), dtype=torch.long), policy='lac')


if __name__ == '__main__':
    unittest.main()
