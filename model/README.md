# Model implementation

Start with the [architecture](../docs/ARCHITECTURE.md).
The package exports `Config`, `Transformer`, and `norm` without loading the
Tensor adapter. The reference can run on CPU with Torch alone.

| File | Responsibility |
|---|---|
| [reference.py](reference.py) | Configuration, Torch reference, projection folding, and a simple decode reference |
| [tensor_backend.py](tensor_backend.py) | `BackendTransformer`, explicit backend operations, training loss, prefill, and cache-aware decode |
| [checkpointing.py](checkpointing.py) | `CheckpointTransformer`, block AC, and exploratory latent-region LAC |

Minimal reference forward:

```python
import torch
from model import Config, Transformer

config = Config(kind="llt", width=128, heads=4, layers=2, loops=4,
                rank=32, vocab=65, max_seq=64, gelu="none")
model = Transformer(config)
logits = model(torch.randint(config.vocab, (2, 32)))
```

`Config` retains the earlier small-fixture defaults, including tanh GELU.
The main study explicitly selects exact GELU (`gelu="none"`) and its full measured
geometry. It constructs `CheckpointTransformer` for policy comparisons.

`BackendTransformer(config, ops=None)` uses Torch operations. Passing an
`Operators` instance from [Tensor](https://github.com/kreasof-ai/tensor) selects
its explicit numerical implementation. Importing the backend module requires
Tensor's Torch adapter. Layout transformations, checkpoint scheduling, and model
orchestration remain Torch work; successful measured Tensor cases have no implicit
numerical fallback.

The experiment's public variant names map to configurations: `naive_loop` uses
`kind="naive"`; `stacked` uses a single pass through an enlarged independent block stack;
`fixed_depth` uses a single pass and a wider active MLP. See
[loop_sweep.py](../experiments/l40s/loop_sweep.py) for geometry and parameter counts.
Kernel implementations and runtime development belong to the Tensor repository.
