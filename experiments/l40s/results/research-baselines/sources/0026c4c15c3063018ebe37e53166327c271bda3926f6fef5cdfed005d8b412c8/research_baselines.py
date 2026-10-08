"""Kernel-profile adaptations of recurrent and cache-sharing architecture families.

Paper topology, cache boundaries, and adaptation details are documented separately.
These randomly initialized models are not trained checkpoint reproductions.
"""

from dataclasses import dataclass
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tensor_torch.llt import KVCache
from .reference import Config
from .tensor_backend import BackendTransformer, TorchCache

FAMILIES = ("uyoco", "lpt", "grt", "per_layer_latent", "attention_only")
LABELS = {
    "uyoco": "U-YOCO / SWA",
    "lpt": "LPT cache layout",
    "grt": "GRT / full KV",
    "per_layer_latent": "Per-layer latent loop (MLA-style)",
    "attention_only": "Attention-only loop / FFN once",
}


@dataclass
class ResearchConfig(Config):
    family: str = "uyoco"
    window: int = 512
    prelude: int = 2
    coda: int = 2
    noise_std: float = 0.1


def make_config(family, loops, small=False):
    if family not in FAMILIES:
        raise ValueError(family)
    width, heads, layers, vocab, seq = (
        (128, 2, 2, 256, 33) if small else (768, 12, 12, 50304, 1025)
    )
    return ResearchConfig(
        kind="layerwise" if family == "per_layer_latent" else "naive",
        width=width,
        heads=heads,
        layers=layers,
        loops=loops,
        rank=64,
        vocab=vocab,
        max_seq=seq,
        gelu="none",
        mlp_width=3 * width if family == "uyoco" else 4 * width,
        family=family,
        window=512 if family == "uyoco" else 65,
        prelude=1 if small else 2,
        coda=0 if small else 2,
    )


@lru_cache(maxsize=32)
def _window_mask(m, n, window, prefix, device):
    from torch.nn.attention.flex_attention import create_block_mask

    def mask(batch, head, q, k):
        return ((k < prefix) & (k <= q)) | (
            (k >= prefix) & (k - prefix <= q) & (k - prefix > q - window)
        )

    return create_block_mask(
        mask, B=None, H=None, Q_LEN=m, KV_LEN=n, device=device, _compile=True
    )


@lru_cache(maxsize=1)
def _flex():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=False)


class ResearchTransformer(BackendTransformer):
    def __init__(self, c, ops=None):
        super().__init__(c, ops)
        if c.family == "uyoco":
            if c.layers % 2:
                raise ValueError("U-YOCO requires an even layer split")
            del self.position
            self.global_k = nn.Linear(c.width, c.width, bias=False)
            self.global_v = nn.Linear(c.width, c.width, bias=False)
            self.global_norm = nn.Parameter(torch.ones(c.width))
            self.final_norm = nn.Parameter(torch.ones(c.width))
            for index, block in enumerate(self.blocks):
                block.gate = nn.Linear(c.width, c.mlp_width, bias=False)
                block.norm_a = nn.Parameter(torch.ones(c.width))
                block.norm_f = nn.Parameter(torch.ones(c.width))
                if index >= c.layers // 2:
                    del block.k, block.v
            for module in (
                self.global_k,
                self.global_v,
                *(b.gate for b in self.blocks),
            ):
                nn.init.normal_(module.weight, std=0.02)
        elif c.family == "grt":
            if not c.prelude + c.coda < c.layers:
                raise ValueError("GRT needs a nonempty shared core")
            for block in self.blocks:
                block.norm_a = nn.LayerNorm(c.width, bias=False)
                block.norm_f = nn.LayerNorm(c.width, bias=False)
            self.final_norm = nn.LayerNorm(c.width, bias=False)
            self.gate_state_norm = nn.LayerNorm(c.width, bias=False)
            self.gate_anchor_norm = nn.LayerNorm(c.width, bias=False)
            self.gate_concat_norm = nn.LayerNorm(2 * c.width, bias=False)
            self.recurrent_projection = nn.Linear(2 * c.width, c.width, bias=False)
            self.gate_up = nn.Linear(2 * c.width, c.width, bias=False)
            self.gate_down = nn.Linear(c.width, c.width, bias=False)
            self.gate_bias = nn.Parameter(torch.full((c.width,), 4.0))
            self.register_buffer("zero_bias", torch.zeros(c.width), persistent=False)
            self.register_buffer(
                "zero_concat_bias", torch.zeros(2 * c.width), persistent=False
            )
            for module in (self.recurrent_projection, self.gate_up, self.gate_down):
                nn.init.normal_(module.weight, std=0.02)

    @property
    def effective_depth(self):
        c = self.c
        if c.family == "uyoco":
            return c.layers // 2 * (c.loops + 1)
        if c.family == "grt":
            return c.prelude + c.coda + (c.layers - c.prelude - c.coda) * c.loops
        return c.layers * c.loops

    @property
    def mlp_applications(self):
        return (
            self.c.layers if self.c.family == "attention_only" else self.effective_depth
        )

    def initial(self, tokens, offset=0):
        if self.c.family != "uyoco":
            return super().initial(tokens, offset)
        return (
            self.ops.embedding(tokens, self.token.weight)
            if self.ops
            else self.token(tokens)
        )

    def _norm(self, x, weight):
        if isinstance(weight, nn.LayerNorm):
            if self.ops:
                bias = (
                    self.zero_bias
                    if x.shape[-1] == self.c.width
                    else self.zero_concat_bias
                )
                return self.ops.layer_norm(x, weight.weight, bias, eps=weight.eps)
            return weight(x)
        return (
            self.ops.rms_norm(x, weight, eps=1e-5)
            if self.ops
            else self.normalize(x) * weight
        )

    def _silu(self, x):
        return self.ops.silu(x) if self.ops else F.silu(x)

    def _multiply(self, x, y):
        return self.ops.multiply(x, y) if self.ops else x * y

    def _rope(self, x, offset=0):
        if self.ops:
            return self.ops.rotary(x, offset=offset)
        half = x.shape[-1] // 2
        angle = (
            torch.arange(x.shape[2], device=x.device, dtype=torch.float32) + offset
        )[:, None] * 10000.0 ** (
            -torch.arange(half, device=x.device, dtype=torch.float32) / half
        )
        co = torch.cat((angle.cos(), angle.cos()), -1)[None, None]
        si = torch.cat((angle.sin(), angle.sin()), -1)[None, None]
        paired = torch.cat((-x[..., half:], x[..., :half]), -1)
        return (x.float() * co + paired.float() * si).to(x.dtype)

    def _window(self, q, k, v, window, prefix=0):
        if self.ops:
            return self.ops.window_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                window_size=window,
                shared_prefix=prefix,
                scale=(self.c.width // self.c.heads) ** -0.5,
            )
        if q.device.type == "cuda":
            mask = _window_mask(q.shape[2], k.shape[2], window, prefix, str(q.device))
            return _flex()(
                q, k, v, block_mask=mask, scale=(self.c.width // self.c.heads) ** -0.5
            )
        qi = torch.arange(q.shape[2], device=q.device)[:, None]
        kj = torch.arange(k.shape[2], device=q.device)[None, :]
        mask = ((kj < prefix) & (kj <= qi)) | (
            (kj >= prefix) & (kj - prefix <= qi) & (kj - prefix > qi - window)
        )
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=(self.c.width // self.c.heads) ** -0.5
        )

    def _mlp(self, x, block):
        z = (
            self._norm(x, block.norm_f)
            if self.c.family in ("uyoco", "grt")
            else self.normalize(x)
        )
        hidden = self.linear(z, block.w1.weight)
        if self.c.family == "uyoco":
            hidden = self._multiply(
                hidden, self._silu(self.linear(z, block.gate.weight))
            )
        else:
            hidden = self.activate(hidden)
        return self.add(x, self.linear(hidden, block.w2.weight))

    def _block(
        self,
        x,
        index,
        *,
        shared=None,
        fold=None,
        mlp=True,
        decode_cache=None,
        global_cache=None,
        offset=0
    ):
        c = self.c
        block = self.blocks[index]
        dim = c.width // c.heads
        z = (
            self._norm(x, block.norm_a)
            if c.family in ("uyoco", "grt")
            else self.normalize(x)
        )
        if c.family == "per_layer_latent":
            fq, fo = fold
            q = self.heads(self.linear(z, fq), c.rank)
            k = v = self.linear(z, self.down[index].weight).unsqueeze(1).contiguous()
        else:
            q = self.heads(self.linear(z, block.q.weight), dim)
            if shared is None:
                k, v = [
                    self.heads(self.linear(z, projection.weight), dim)
                    for projection in (block.k, block.v)
                ]
            else:
                k, v = shared
            if c.family == "uyoco" and index < c.layers // 2:
                q, k = self._rope(q, offset), self._rope(k, offset)
        if decode_cache is not None:
            decode_cache.append(k, v)
            if global_cache is not None:
                if self.ops:
                    a = self.ops.shared_decode(
                        q.contiguous(), global_cache, decode_cache, scale=dim**-0.5
                    )
                else:
                    kk = torch.cat(
                        (
                            global_cache.keys[:, :, : global_cache.length],
                            decode_cache.keys[:, :, : decode_cache.length],
                        ),
                        2,
                    )
                    vv = torch.cat(
                        (
                            global_cache.values[:, :, : global_cache.length],
                            decode_cache.values[:, :, : decode_cache.length],
                        ),
                        2,
                    )
                    a = self.attend(q, kk, vv, causal=False)
            else:
                a = self._cached_attention(q, decode_cache)
        elif global_cache is not None:
            a = self._cached_attention(q, global_cache)
        elif c.family == "lpt" and shared is not None:
            # Prediction recursion: local K/V come from the current state, not from shared memory.
            local_k, local_v = [
                self.heads(self.linear(z, projection.weight), dim)
                for projection in (block.k, block.v)
            ]
            a = self._window(
                q,
                torch.cat((k, local_k), 2),
                torch.cat((v, local_v), 2),
                c.window,
                k.shape[2],
            )
            k, v = local_k, local_v
        elif c.family == "uyoco" and index < c.layers // 2:
            a = self._window(q, k, v, c.window)
        else:
            a = self.attend(q, k, v, causal=(q.shape[2] == k.shape[2]))
        a = a.transpose(1, 2).reshape(*x.shape[:2], -1)
        delta = self.linear(
            a, fold[1] if c.family == "per_layer_latent" else block.o.weight
        )
        x = self.add(x, delta)
        return (self._mlp(x, block) if mlp else x), k, v

    def _cached_attention(self, q, cache):
        if self.ops:
            return self.ops.decode(
                q.contiguous(), cache, scale=(self.c.width // self.c.heads) ** -0.5
            )
        return self.attend(
            q,
            cache.keys[:, :, : cache.length],
            cache.values[:, :, : cache.length],
            causal=False,
        )

    def _call(self, fn, args, policy):
        if policy == "ac":
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
        return fn(*args)

    def _grt_features(self, x, anchor):
        features = torch.cat(
            (
                self._norm(x, self.gate_state_norm),
                self._norm(anchor, self.gate_anchor_norm),
            ),
            -1,
        )
        features = self._norm(features, self.gate_concat_norm)
        logits = self.linear(
            self._silu(self.linear(features, self.gate_up.weight)),
            self.gate_down.weight,
        ).float()
        logits = self.add(
            logits, self.gate_bias[None, None].expand_as(logits).contiguous()
        )
        source = x
        if self.training and self.c.noise_std:
            # Torch owns random-number generation on both backends; native math consumes explicit noise.
            noise = (
                torch.empty_like(logits[..., :1])
                .normal_(std=self.c.noise_std)
                .expand_as(logits)
                .contiguous()
            )
            logits = self.add(logits, noise)
            source = self.add(x, torch.empty_like(x).normal_(std=self.c.noise_std))
        gate = self.ops.sigmoid(logits) if self.ops else torch.sigmoid(logits)
        proposal = self.linear(
            torch.cat((source, anchor), -1), self.recurrent_projection.weight
        )
        return gate, proposal

    def _blend(self, g, x, z):
        return self.ops.blend(g, x, z) if self.ops else g * x + (1 - g) * z

    def _final(self, x, last_logits=False):
        x = x[:, -1:] if last_logits else x
        x = (
            self._norm(x, self.final_norm)
            if self.c.family in ("uyoco", "grt")
            else self.normalize(x)
        )
        return self.linear(x, self.output.weight)

    def forward(
        self,
        tokens,
        policy="none",
        return_cache=False,
        return_hidden=False,
        last_logits=False,
    ):
        if (
            policy not in ("none", "ac")
            or return_hidden
            or (return_cache and policy != "none")
        ):
            raise ValueError("unsupported profile policy")
        c = self.c
        x = self.initial(tokens)
        banks = {}
        folds = (
            [self.folds(b) for b in self.blocks]
            if c.family == "per_layer_latent"
            else []
        )

        def block(index, key, *, shared=None, mlp=True):
            nonlocal x
            args = (x,) if shared is None else (x, *shared)

            def fn(state, *kv):
                return self._block(
                    state,
                    index,
                    shared=kv or None,
                    fold=folds[index] if folds else None,
                    mlp=mlp,
                )

            if return_cache or (c.family == "lpt" and key[0] == "shared"):
                x, k, v = self._call(fn, args, policy)
                window = (
                    c.window - 1
                    if c.family == "uyoco" and index < c.layers // 2
                    else (
                        c.window - 1
                        if c.family == "lpt" and key[0] == "local"
                        else None
                    )
                )
                banks[key] = (k, v, window)
            else:
                x = self._call(lambda *a: fn(*a)[0], args, policy)

        if c.family == "uyoco":
            half = c.layers // 2
            for loop in range(c.loops):
                for index in range(half):
                    block(index, ("local", loop, index))
            memory = self._norm(x, self.global_norm)
            k, v = [
                self.heads(self.linear(memory, p.weight), c.width // c.heads)
                for p in (self.global_k, self.global_v)
            ]
            if return_cache:
                banks[("global",)] = (k, v, None)
            if last_logits:
                x = x[:, -1:]
            for index in range(half, c.layers):
                block(index, ("cross", index), shared=(k, v))
            # Cross blocks do not own caches: the single global bank is shared.
            for key in list(banks):
                if key[0] == "cross":
                    del banks[key]
        elif c.family == "lpt":
            for index in range(c.layers):
                block(index, ("shared", index))
            shared = {i: banks[("shared", i)][:2] for i in range(c.layers)}
            for loop in range(1, c.loops):
                for index in range(c.layers):
                    block(index, ("local", loop, index), shared=shared[index])
        elif c.family == "grt":
            core_end = c.layers - c.coda
            for index in range(c.prelude):
                block(index, ("prelude", index))
            anchor = x
            for loop in range(c.loops):
                previous = x
                gate, x = self._grt_features(x, anchor)
                for index in range(c.prelude, core_end):
                    block(index, ("core", loop, index))
                x = self._blend(gate, previous, x)
            for index in range(core_end, c.layers):
                block(index, ("coda", index))
        elif c.family == "attention_only":
            for index in range(c.layers):
                for loop in range(c.loops):
                    block(index, ("attention", index, loop), mlp=False)
                x = self._call(
                    lambda state, index=index: self._mlp(state, self.blocks[index]),
                    (x,),
                    policy,
                )
        else:
            for loop in range(c.loops):
                for index in range(c.layers):
                    block(index, ("latent", loop, index))
        result = self._final(x, last_logits)
        return (result, dict(banks=banks, folds=folds)) if return_cache else result

    @torch.no_grad()
    def prefill(self, tokens, capacity=None, last_logits=False):
        if self.training and self.c.family == "grt":
            raise ValueError("cache qualification requires eval mode (noise disabled)")
        logits, raw = self(tokens, return_cache=True, last_logits=last_logits)
        if capacity is not None and capacity < tokens.shape[1] + 1:
            raise ValueError("prefill needs a current-token slot")
        caches = {}
        for key, (k, v, window) in raw["banks"].items():
            if window is not None:
                k, v = (
                    k[:, :, -min(window, k.shape[2]) :].contiguous(),
                    v[:, :, -min(window, v.shape[2]) :].contiguous(),
                )
            shared = self.c.family == "per_layer_latent"
            if shared:
                v = k
            slots = k.shape[2] + 1
            if self.ops:
                cache = KVCache(
                    self.ops,
                    k.shape[0],
                    k.shape[1],
                    slots,
                    k.shape[-1],
                    dtype=k.dtype,
                    shared=shared,
                )
            else:
                keys = torch.empty(
                    (*k.shape[:2], slots, k.shape[-1]), device=k.device, dtype=k.dtype
                )
                cache = TorchCache(keys, keys if shared else torch.empty_like(keys))
            cache.append(k, v)
            caches[key] = cache
        state = dict(
            banks=caches,
            caches=list(caches.values()),
            folds=raw["folds"],
            length=tokens.shape[1],
            prefix_length=tokens.shape[1],
            prefix_lengths={key: cache.length for key, cache in caches.items()},
            versions=tuple(p._version for p in self.parameters()),
        )
        return logits, state

    @staticmethod
    def rewind(state, length):
        if length != state["prefix_length"]:
            raise ValueError(
                "this profile state replays one token after a fixed prefix"
            )
        state["length"] = length
        for key, cache in state["banks"].items():
            cache.length = state["prefix_lengths"][key]
            if isinstance(cache, KVCache):
                cache.lengths.fill_(cache.length)
                cache.overflow.zero_()
                cache.captured_mutation = False

    @torch.no_grad()
    def decode_token(self, tokens, state):
        if (
            tokens.shape[1] != 1
            or state["length"] != state["prefix_length"]
            or state["versions"] != tuple(p._version for p in self.parameters())
        ):
            raise ValueError(
                "decode profile requires unchanged weights and one token after the fixed prefix; rewind before replay"
            )
        c = self.c
        offset = state["length"]
        x = self.initial(tokens, offset)
        banks = state["banks"]

        def block(index, key, mlp=True, global_cache=None):
            nonlocal x
            x, _, _ = self._block(
                x,
                index,
                fold=state["folds"][index] if state["folds"] else None,
                mlp=mlp,
                decode_cache=banks[key],
                global_cache=global_cache,
                offset=offset,
            )

        if c.family == "uyoco":
            half = c.layers // 2
            for loop in range(c.loops):
                for index in range(half):
                    block(index, ("local", loop, index))
            z = self._norm(x, self.global_norm)
            k, v = [
                self.heads(self.linear(z, p.weight), c.width // c.heads)
                for p in (self.global_k, self.global_v)
            ]
            banks[("global",)].append(k, v)
            for index in range(half, c.layers):
                x, _, _ = self._block(
                    x, index, shared=(k, v), global_cache=banks[("global",)]
                )
        elif c.family == "lpt":
            for index in range(c.layers):
                block(index, ("shared", index))
            for loop in range(1, c.loops):
                for index in range(c.layers):
                    block(
                        index,
                        ("local", loop, index),
                        global_cache=banks[("shared", index)],
                    )
        elif c.family == "grt":
            core_end = c.layers - c.coda
            for index in range(c.prelude):
                block(index, ("prelude", index))
            anchor = x
            for loop in range(c.loops):
                previous = x
                gate, x = self._grt_features(x, anchor)
                for index in range(c.prelude, core_end):
                    block(index, ("core", loop, index))
                x = self._blend(gate, previous, x)
            for index in range(core_end, c.layers):
                block(index, ("coda", index))
        elif c.family == "attention_only":
            for index in range(c.layers):
                for loop in range(c.loops):
                    block(index, ("attention", index, loop), mlp=False)
                x = self._mlp(x, self.blocks[index])
        else:
            for loop in range(c.loops):
                for index in range(c.layers):
                    block(index, ("latent", loop, index))
        state["length"] += 1
        return self._final(x)
