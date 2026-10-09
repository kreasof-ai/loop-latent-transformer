"""Contextual per-layer latent memory, constructed on loop zero and reused.

The static global model is retained in reference.py for frozen historical tests.
This class defines LLT for the replacement experiment. It has L cache banks,
never L*T: historical memory is neither refreshed nor overwritten on later loops.
"""

from dataclasses import replace

import torch
from torch.utils.checkpoint import checkpoint

from .tensor_backend import BackendTransformer


class ContextualLLT(BackendTransformer):
    def __init__(self, c, ops=None):
        if c.untied or c.loops < 1 or c.layers < 1:
            raise ValueError("contextual LLT requires positive depth and tied loops")
        super().__init__(replace(c, kind="layerwise"), ops)

    @staticmethod
    def architecture():
        return dict(
            name="contextual_per_layer_loop_shared",
            latent_source="each layer's input residual during the first loop",
            loop_sharing="first-loop memory reused without refresh on later loops",
            layer_sharing=False,
            checkpoint_policies=["none", "ac"],
            description=(
                "LLT stores one contextual rank-R latent cache per physical layer. "
                "During the first loop, layer i constructs C_i from its normalized "
                "input residual and uses it as attention memory. Later loops reuse "
                "that same C_i while their queries and residual states evolve. "
                "The first layer's memory is embedding-derived; deeper layers' "
                "memories contain preceding first-loop computations. Historical "
                "memory is not refreshed on later loops. There are 12 banks, "
                "independent of T; BF16 KV payload is 2*B*S*R*12 bytes."
            ),
        )

    @staticmethod
    def bank_count(c):
        return c.layers

    @property
    def cache_bank_count(self):
        return self.bank_count(self.c)

    def block(self, x, index, latent, fq, fo, unfolded=False):
        c = self.c
        block = self.blocks[index]
        z = self.normalize(x)
        if latent is None:
            latent = self.linear(z, self.down[index].weight).unsqueeze(1).contiguous()
        if unfolded:
            if self.ops:
                raise ValueError("unfolded algebra reference uses Torch only")
            q = self.heads(self.linear(z, block.q.weight), c.width // c.heads)
            k = torch.einsum("bsr,hdr->bhsd", latent[:, 0], block.uk)
            v = torch.einsum("bsr,hdr->bhsd", latent[:, 0], block.uv)
            a = self.attend(q, k, v).transpose(1, 2).reshape(*z.shape[:2], c.width)
            delta = self.linear(a, block.o.weight)
        else:
            q = self.heads(self.linear(z, fq), c.rank)
            a = (
                self.attend(q, latent, latent)
                .transpose(1, 2)
                .reshape(*z.shape[:2], c.heads * c.rank)
            )
            delta = self.linear(a, fo)
        x = self.add(x, delta)
        x = self.add(
            x,
            self.linear(
                self.activate(self.linear(self.normalize(x), block.w1.weight)),
                block.w2.weight,
            ),
        )
        return x, latent

    def forward(
        self,
        tokens,
        policy="none",
        return_cache=False,
        return_hidden=False,
        last_logits=False,
        unfolded=False,
    ):
        if policy not in ("none", "ac") or (return_cache and policy != "none"):
            raise ValueError(
                "contextual LLT supports none/AC; cache creation uses none"
            )
        x = self.initial(tokens)
        latents = [None] * self.c.layers
        folds = [self.folds(block) for block in self.blocks]
        for loop in range(self.c.loops):
            for index in range(self.c.layers):
                latent = None if loop == 0 else latents[index]
                fq, fo = folds[index]
                fn = lambda state, memory, qweight, oweight, index=index: self.block(
                    state, index, memory, qweight, oweight, unfolded
                )
                if policy == "ac":
                    x, memory = checkpoint(
                        fn,
                        x,
                        latent,
                        fq,
                        fo,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    x, memory = fn(x, latent, fq, fo)
                if loop == 0:
                    latents[index] = memory
        hidden = self.normalize(x)
        result = (
            hidden
            if return_hidden
            else self.linear(
                hidden[:, -1:, :] if last_logits else hidden, self.output.weight
            )
        )
        return (result, latents) if return_cache else result

    @torch.no_grad()
    def decode_token(self, tokens, state):
        if tokens.shape[1] != 1 or state["versions"] != tuple(
            p._version for p in self.parameters()
        ):
            raise ValueError(
                "decode requires one token and unchanged inference weights"
            )
        c = self.c
        if state["length"] >= c.max_seq:
            raise ValueError("position capacity exceeded")
        x = self.initial(tokens, state["length"])
        for loop in range(c.loops):
            for index, block in enumerate(self.blocks):
                z = self.normalize(x)
                cache = state["caches"][index]
                if loop == 0:
                    cache.append(self.linear(z, self.down[index].weight).unsqueeze(1))
                fq, fo = state["folds"][index]
                q = self.heads(self.linear(z, fq), c.rank)
                if self.ops:
                    a = self.ops.decode(q, cache, scale=(c.width // c.heads) ** -0.5)
                else:
                    a = self.attend(
                        q,
                        cache.keys[:, :, : cache.length],
                        cache.values[:, :, : cache.length],
                        causal=False,
                    )
                a = a.transpose(1, 2).reshape(tokens.shape[0], 1, c.heads * c.rank)
                x = self.add(x, self.linear(a, fo))
                x = self.add(
                    x,
                    self.linear(
                        self.activate(self.linear(self.normalize(x), block.w1.weight)),
                        block.w2.weight,
                    ),
                )
        state["length"] += 1
        return self.linear(self.normalize(x), self.output.weight)
