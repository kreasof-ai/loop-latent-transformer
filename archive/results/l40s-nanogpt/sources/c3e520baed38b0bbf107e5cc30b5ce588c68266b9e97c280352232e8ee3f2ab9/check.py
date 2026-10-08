"""Independent algebra, exact checkpoint gradients, causality and decode tests."""
import json
import torch
from model import Config, Transformer
from common import setup, save, OUT
from torch.nn.attention import SDPBackend, sdpa_kernel


def run():
    setup()
    results = []
    for kind in ('naive', 'llt', 'layerwise'):
        c = Config(kind=kind, width=64, heads=2, rank=16, loops=3, max_seq=64, vocab=32)
        model = Transformer(c).double().cuda()
        tokens = torch.randint(c.vocab, (2, 13), device='cuda')
        y = model(tokens)
        active = tuple(model.parameters())
        g = torch.autograd.grad(y.square().mean(), active)
        ck = model(tokens, policy='loop')
        cg = torch.autograd.grad(ck.square().mean(), active)
        torch.testing.assert_close(y, ck, atol=1e-11, rtol=1e-9)
        for a, b in zip(g, cg):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)
        result = dict(kind=kind, checkpoint_gradient_max_error=max((a-b).abs().max().item() for a,b in zip(g,cg)))
        if kind != 'naive':
            explicit = model(tokens, unfolded=True)
            eg = torch.autograd.grad(explicit.square().mean(), active)
            torch.testing.assert_close(y, explicit, atol=1e-11, rtol=1e-9)
            for a, b in zip(g, eg):
                torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)
            result.update(folding_output_max_error=(y-explicit).abs().max().item(),
                          folding_gradient_max_error=max((a-b).abs().max().item() for a,b in zip(g,eg)))
        with torch.no_grad():
            _, caches = model(tokens[:, :-1], return_cache=True)
            decode = model.prepare_decode(caches, tokens.shape[1]-1)
            torch.testing.assert_close(decode(tokens[:, -1:]), y[:, -1:], atol=1e-11, rtol=1e-9)
            altered = tokens.clone()
            altered[:, 7:] = (altered[:, 7:] + 1) % c.vocab
            torch.testing.assert_close(model(altered)[:, :7], y[:, :7], atol=1e-11, rtol=1e-9)
        result['causal_and_cached_decode'] = 'passed'
        results.append(result)
    for kind in ('naive','llt','layerwise'):
        c=Config(kind=kind,width=128,heads=2,rank=32,loops=4,max_seq=129,vocab=64)
        model=Transformer(c).cuda()
        tokens=torch.randint(c.vocab,(2,129),device='cuda')
        active=tuple(model.parameters())
        with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            gold=model(tokens)
            ck=model(tokens,policy='loop')
        g=torch.autograd.grad(gold.float().square().mean(),active)
        cg=torch.autograd.grad(ck.float().square().mean(),active)
        torch.testing.assert_close(gold,ck,atol=0,rtol=0)
        for actual,expected in zip(cg,g):
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-4)
        results.append(dict(kind=kind,dtype='bfloat16 autocast',checkpoint_gradient_max_error=max((a-b).abs().max().item() for a,b in zip(g,cg))))
    save(OUT / 'correctness.json', dict(status='passed', dtype='float64 reference and CUDA BF16 autocast', checks=results))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    run()
