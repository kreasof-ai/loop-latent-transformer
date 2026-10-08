"""Exact CPU checkpoint and cache boundaries for the added architecture profiles."""

from dataclasses import replace
import unittest
import torch
from model.research_baselines import FAMILIES, ResearchTransformer, make_config


class ResearchBaselinesTests(unittest.TestCase):
    def test_checkpoint_gradients_and_one_token_cache(self):
        torch.set_num_threads(2)
        for family in FAMILIES:
            with self.subTest(family=family):
                torch.manual_seed(10)
                c = replace(
                    make_config(family, 3, small=True),
                    width=32,
                    heads=2,
                    rank=16,
                    vocab=31,
                    max_seq=17,
                    mlp_width=96 if family == "uyoco" else 128,
                    window=4,
                )
                model = ResearchTransformer(c).double().eval()
                tokens = torch.randint(c.vocab, (2, 9))
                gradients = []
                outputs = []
                for policy in ("none", "ac"):
                    model.zero_grad(set_to_none=True)
                    output = model(tokens, policy=policy)
                    output.square().sum().backward()
                    outputs.append(output.detach())
                    gradients.append(
                        {n: p.grad.clone() for n, p in model.named_parameters()}
                    )
                torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
                for name in gradients[0]:
                    torch.testing.assert_close(
                        gradients[0][name], gradients[1][name], atol=1e-10, rtol=1e-8
                    )
                with torch.inference_mode():
                    full = model(tokens)
                    prompt, state = model.prefill(
                        tokens[:, :-1], capacity=17, last_logits=True
                    )
                    torch.testing.assert_close(
                        prompt,
                        model(tokens[:, :-1], last_logits=True),
                        atol=1e-10,
                        rtol=1e-8,
                    )
                    cached = model.decode_token(tokens[:, -1:], state)
                    torch.testing.assert_close(
                        cached, full[:, -1:], atol=1e-10, rtol=1e-8
                    )
                    model.rewind(state, 8)
                    torch.testing.assert_close(
                        model.decode_token(tokens[:, -1:], state),
                        cached,
                        atol=0,
                        rtol=0,
                    )

    def test_applied_depth_and_ffn_count(self):
        for family in FAMILIES:
            c = make_config(family, 16, small=True)
            m = ResearchTransformer(c)
            expected = 17 if family == "uyoco" else 17 if family == "grt" else 32
            self.assertEqual(m.effective_depth, expected)
            self.assertEqual(
                m.mlp_applications, 2 if family == "attention_only" else expected
            )

    def test_grt_stochastic_training_checkpoint(self):
        c = replace(
            make_config("grt", 3, small=True),
            width=32,
            heads=2,
            vocab=31,
            max_seq=17,
            mlp_width=128,
        )
        model = ResearchTransformer(c).double().train()
        tokens = torch.randint(c.vocab, (2, 9))
        results = []
        for policy in ("none", "ac"):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(23)
            output = model(tokens, policy=policy)
            output.square().mean().backward()
            results.append(
                (
                    output.detach(),
                    {n: p.grad.clone() for n, p in model.named_parameters()},
                    torch.get_rng_state(),
                )
            )
        torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
        torch.testing.assert_close(results[0][2], results[1][2], atol=0, rtol=0)
        for name in results[0][1]:
            torch.testing.assert_close(
                results[0][1][name], results[1][1][name], atol=1e-10, rtol=1e-8
            )


if __name__ == "__main__":
    unittest.main()
