"""First-loop memory semantics, folded algebra, AC gradients, and causal decode."""

import unittest

import torch
from model import Config
from model.contextual_llt import ContextualLLT


class ContextualLLTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(59)

    def model(self, loops=3):
        return ContextualLLT(
            Config(
                width=32,
                heads=2,
                layers=3,
                loops=loops,
                rank=8,
                vocab=31,
                max_seq=16,
                gelu="none",
            )
        ).double()

    def test_folding_and_full_block_ac_preserve_all_gradients(self):
        m = self.model()
        x = torch.randint(31, (2, 7))
        reference = None
        for unfolded, policy in ((True, "none"), (False, "none"), (False, "ac")):
            m.zero_grad(set_to_none=True)
            logits = m(x, policy=policy, unfolded=unfolded)
            logits.square().sum().backward()
            grads = {name: p.grad.clone() for name, p in m.named_parameters()}
            if reference is None:
                reference = logits.detach(), grads
            torch.testing.assert_close(logits, reference[0], rtol=1e-10, atol=1e-12)
            for name, grad in grads.items():
                torch.testing.assert_close(
                    grad, reference[1][name], rtol=1e-9, atol=1e-11
                )

    def test_first_loop_memory_is_contextual_and_reused_across_loops(self):
        m = self.model()
        x = torch.randint(31, (2, 7))
        calls = []
        original = m.block

        def record(state, index, latent, *args):
            result = original(state, index, latent, *args)
            calls.append((index, latent, result[1]))
            return result

        m.block = record
        _, caches = m(x, return_cache=True)
        self.assertEqual(len(caches), 3)
        for j, (index, incoming, outgoing) in enumerate(calls):
            if j < 3:
                self.assertIsNone(incoming)
            else:
                self.assertIs(incoming, caches[index])
            self.assertIs(outgoing, caches[index])
        altered = x.clone()
        altered[:, 0] = (altered[:, 0] + 1) % 31
        _, changed = m(altered, return_cache=True)
        torch.testing.assert_close(
            caches[0][:, :, 1:], changed[0][:, :, 1:], rtol=0, atol=0
        )
        self.assertGreater(
            (caches[1][:, :, 1:] - changed[1][:, :, 1:]).abs().max().item(), 1e-9
        )

    def test_multi_token_decode_and_causality(self):
        x = torch.randint(31, (2, 10))
        for loops in (1, 3):
            m = self.model(loops)
            with torch.inference_mode():
                _, state = m.prefill(x[:, :7], capacity=12)
                addresses = [cache.keys.data_ptr() for cache in state["caches"]]
                for offset in range(7, 10):
                    cached = m.decode_token(x[:, offset : offset + 1], state)
                    full = m(x[:, : offset + 1])[:, -1:]
                    torch.testing.assert_close(cached, full, rtol=1e-10, atol=1e-12)
                    self.assertEqual(
                        [cache.length for cache in state["caches"]], [offset + 1] * 3
                    )
                self.assertEqual(
                    addresses, [cache.keys.data_ptr() for cache in state["caches"]]
                )
                self.assertEqual(len(state["caches"]), 3)
                self.assertEqual(
                    sum(cache.nbytes for cache in state["caches"]), 2 * 12 * 8 * 8 * 3
                )
                torch.testing.assert_close(
                    m(x)[:, :7], m(x[:, :7]), rtol=1e-10, atol=1e-12
                )
            with self.assertRaisesRegex(ValueError, "none/AC"):
                m(x, policy="lac")


if __name__ == "__main__":
    unittest.main()
