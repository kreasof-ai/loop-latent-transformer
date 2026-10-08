# Reproduce the optimized nanoGPT-scale study

See [results and measurement limits](../../reports/L40S_NANOGPT.md).

Use [Tensor](https://github.com/kreasof-ai/tensor) implementation
`9d03a692b26e44f23348547263a97b7f18155c7f`, its qualified CUDA environment and the
Torch-versioned C++ executor. The tested machine uses NVIDIA L40S/sm89,
Torch 2.14.0+cu130, Python 3.12, TileLang 0.1.14 and NVRTC 12.9. Select the GPU
Python through `LLT_PYTHON`, and the compiler bundle through `TENSOR_NVRTC_HOME`.
The runner has defaults for this machine. `TENSOR_CHECKOUT` selects the source
checkout whose revision is recorded in reports.

From the LLT repository root:

```sh
experiments/l40s/run_nanogpt_scale.sh
# Continue a partially completed run without replacing passed method files:
LLT_RESUME=1 experiments/l40s/run_nanogpt_scale.sh
python experiments/l40s/nanogpt_scale_summary.py
"$LLT_PYTHON" experiments/l40s/nanogpt_scale_manifest.py
```

The plotting Python needs Matplotlib. It can be separate from the GPU Python.
The script runs GPU processes sequentially. Never run competing GPU benchmarks
while collecting samples. A full run contains five correctness checks, 20
inference methods, 28 scalar/Tensor optimizer training controls, 14 fused Torch
training controls and four actual nanoGPT full-step graph methods. Each graph
method also retains an eager calibration in the same process.

For one case:

```sh
"$LLT_PYTHON" experiments/l40s/nanogpt_scale.py inference \
  --model llt --backend tensor --batch 4 --loops 4
"$LLT_PYTHON" experiments/l40s/nanogpt_scale.py training \
  --model nanogpt --backend torch --torch-optimizer fused --training-graph
```

The vendored upstream nanoGPT source is unchanged and includes its MIT license.
`nanogpt_adapter.py` supplies explicit Tensor numerical replacements.
`prepared_inference.py` supplies Torch's equivalent BF16 projection preparation
while preserving FP32 masters and nanoGPT's tied embedding. Both use fixed-shape
serving graphs. The upstream CPU-list final-token index is replaced by an
equivalent basic slice solely in the inference adapter for capture compatibility.

Training graphs capture full loss, backward, clipping and AdamW. The optimizer
has runtime device step counters; the test expects 105 updates per parameter for
the default sample count. FP32 masters change on every replay. Parameter/optimizer
initialization and compilation are warmed outside timing. Initial eager losses,
final graph loss and counter checks are retained. The graph uses repeated fixed
microbatches and does not represent a corpus training run.

Inference compares full-prefix last-position logits consistently. LLT/naive also
measure complete capacity-cache startup and four supplied-token cached steps.
Actual nanoGPT measures four greedy full-prefix recomputations, cropped to its
1024-token context; it has no upstream persistent KV-cache API. These continuation
protocols are named separately rather than compared as equivalent solvers.

The primary inference memory measure is eager peak allocation. Older inference
graph records expose replay allocation, which excludes capture peak; the report
does not use those values as graph memory limits. Actual nanoGPT training graph
records additionally include capture peak and reserved memory. Allocator-reserved
memory includes cached blocks and graph pools; driver/context allocations remain
excluded. No number is whole-process VRAM.

`benchmarks/results/l40s-nanogpt` retains raw samples, exact source snapshots,
Torch/Tensor/upstream revisions and sm89 artifact hashes. The summary verifies
artifact contents before plotting. Binary/compiler caches and process logs are
ignored. Initial unprepared Torch controls and graph results collected before
capture-memory instrumentation remain in separate observation folders; they are
excluded from selected comparisons. Historical source snapshots remain so that
each run can be reconstructed independently of later harness edits.

The report compares model geometry. LLT, naive and nanoGPT differ in parameter
counts, normalization, projection layout and input/output weight tying. No
trained quality, constant total training memory or distributed performance claim
is inferred from this study.
