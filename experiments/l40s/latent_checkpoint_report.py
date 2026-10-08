"""Build the single current L40S report from audited combined records."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
from experiments.l40s.latent_checkpoint_summary import OUT, VIEWS, VARIANTS, LABELS
from experiments.l40s.runtime import ROOT, PUBLISHED_RESULTS, results_root

PROTOCOL = '''# L40S loop, rank, and checkpoint sweep

This is the project's **main measured experiment**: loop counts T=1..16,
batch 4, sequence length 1024, and LLT KV ranks 32, 64, and 128 on one NVIDIA L40S.
It compares [Tensor](https://github.com/kreasof-ai/tensor) numerical kernels with
matched PyTorch controls using Flash SDPA and fused AdamW. Inputs are synthetic;
this study measures execution cost and numerical consistency, without a trained
language-model quality evaluation.

The current **LAC is an exploratory, exact attention-region checkpoint policy**.
It is narrower than block AC and is not a completed strategy for checkpointing
an entire loop through its latent state. See [checkpoint boundaries](../../docs/CHECKPOINTING.md)
and the [architecture](../../docs/ARCHITECTURE.md).

## Model and measurement protocol

The original LLT/four-control study uses residual width 768, 12 heads (head dimension 64), independent
input embedding/output weights, vocabulary 50,304, learned absolute positions
with capacity 1025, exact GELU, and unweighted RMSNorm with epsilon 1e-5.
Master weights, embeddings, residuals, AdamW states, and loss reductions are FP32;
projection and attention arithmetic is BF16. Serving prepares BF16 linear-weight
copies while retaining FP32 masters; these copies count toward peak memory.

| Architecture | Unique blocks | Applied blocks | MLP hidden width |
|---|---:|---:|---:|
| LLT, ranks 32/64/128 | 12 | 12T | 3072 |
| Naive Loop | 12 | 12T | 3072 |
| Independent stack | 12T | 12T | 3072 |
| Fixed depth, matched parameters | 12 | 12 | 768(6T−2) |

The fixed-depth control matches the stack's parameter count exactly at each T
using active MLP weights. Residual width, heads, vocabulary, and embeddings stay
fixed. This changes its attention-to-MLP compute balance; parameter matching does
not match attention work or activation memory. At T=1 the three conventional
controls have equivalent architectures and are measured independently.

Training measures forward, full-token output projection/cross entropy, backward,
global gradient clipping (norm 1), and AdamW updates (learning rate 0.0006,
betas 0.9/0.95, weight decay 0.1). Input and target CUDA token IDs use seed 9505.
A step processes 4096 tokens. AC checkpoints full Transformer blocks with
non-reentrant recomputation. LAC checkpoints only LLT's latent attention and
folded output projection, with Q_r, C, and the folded output weight as explicit
inputs. LAC uses the existing model rank and adds no compression codec or parameters.
Query formation, full-width residuals, and MLP activations remain outside LAC.
Conventional controls have no matching native latent boundary, so LAC is N/A.

Inference has no backward checkpoints and reports three operations:

- **Prompt:** causal 1024-token forward and last-position logits, without persistent KV allocation.
- **Serving startup:** prompt plus KV allocation/copy and LLT fold rebuild.
- **Cached decode:** one supplied token after a 1024-token history, using 1025 cache slots.

Latency is per batch of four. Cached decode computes four supplied tokens; it
includes no token sampling or beam search. Historical cache copies and prepared
weights are outside cached-token timing and included in live memory.

Cases run sequentially in fresh processes. Compilation and warmup are excluded.
There are three warmups and nine timing samples per operation. Eager results
retain CUDA-event latency and synchronized wall time. Each graph sample times
three replays and divides by three. Decode rewinds the logical cache before
replay, keeping the same history. Successful training cases validate finite loss
and every parameter gradient, and check all 42 actual optimizer updates.

Eager memory is peak allocated memory during a call. Graph memory is peak
allocated memory during **capture**, including temporaries and the graph pool.
Both include weights and live model state; allocated memory excludes allocator
reservations and driver/context memory. Replay peaks and reservations remain in
raw records. OOM is recorded at the failed operation with unchanged geometry;
operations skipped after an earlier OOM have no latency. Supplementary retries
that release eager gradients before capture remain separate from primary cases.

## Results

'''


def main():
    audit = json.loads((VIEWS / 'audit.json').read_text())
    assert audit['status'] == 'passed' and audit['case_count'] == 672
    rows = json.loads((VIEWS / 'combined.json').read_text())
    records = {(r['phase'], r['model'], r['kv_rank'] or 64, r['backend'], r['loops'], r['checkpoint']): r for r in rows}
    text = PROTOCOL
    text += (f"**{audit['case_count']} performance records: {audit['passed']} passed and "
             f"{audit['out_of_memory']} recorded OOM.** The combined study reuses all "
             f"{audit['reused_case_count']} original records and adds {audit['new_case_count']} "
             "rank/checkpoint cases. Every new performance case passed.\n\n")
    text += '### T=16 training endpoint\n\nCells are **CUDA graph ms / capture peak allocated GiB**, per batch of four.\n\n'
    for backend in ('tensor', 'torch'):
        text += f'**{backend.capitalize()}**\n\n| Architecture | None | AC | LAC (exploratory) |\n|---|---:|---:|---:|\n'
        for model, rank in VARIANTS:
            values = []
            for policy in ('none', 'ac', 'lac'):
                if policy == 'lac' and model != 'llt':
                    values.append('N/A'); continue
                r = records[('training', model, rank, backend, 16, policy)]
                if 'training_graph_gpu_ms' not in r:
                    values.append('OOM' if r['failed_stage'] == 'training_graph' else '— (eager OOM)')
                else:
                    values.append(f"{r['training_graph_gpu_ms']:.2f} / {r['training_graph_peak_gib']:.2f}")
            text += '| ' + LABELS[(model, rank)] + ' | ' + ' | '.join(values) + ' |\n'
        text += '\n'
    text += '''At rank 64 and T=16, Tensor AC uses 4.79 GiB versus 21.70 GiB for
LAC and 25.11 GiB without checkpoints. These policies recompute different regions:
AC discards full-block intermediates; LAC retains the residual/query/MLP paths.
This result motivates further checkpoint design rather than a claim that the
current LAC achieves constant total training memory.

![Training GPU latency](results/loop-sweep/views/training_graph-gpu_ms.png)

![Training peak allocated memory](results/loop-sweep/views/training_graph-peak_gib.png)

![Cached inference](results/loop-sweep/views/decode_graph.png)

## Numerical qualification

'''
    maximum = max(result['gradient_error']['maximum_parameter_relative_l2']
                  for path in OUT.glob('*qualification-*.json')
                  for result in json.loads(path.read_text())['policy_checks'].values()
                  if 'gradient_error' in result)
    text += (f"The audit verifies {audit['qualification_count']} small and "
             f"{audit['full_qualification_count']} full-size T=16 checkpoint qualifications, "
             f"plus eight original backend correctness cases. It verifies "
             f"{audit['unique_artifact_count']} unique hashed sm89 CUDA artifacts and zero "
             "implicit Tensor numerical fallback.\n\n")
    text += '''Small checks cover every variant at T=4 and T=16, width 128, two
base blocks, batch 2, and sequence 32. Full-size checks use the measured geometry
at T=16 with unchanged initial weights, no optimizer between policies, and
reference gradients streamed to CPU. AC and applicable LAC gradients are compared
with uncheckpointed gradients; full checks first repeat uncheckpointed backward.
Within-backend parameter relative L2 must be below 1e-5 and forward losses must
agree exactly. Cached decode is compared with full-forward supplied-token logits
(maximum error <0.05; relative L2 <0.03). These checks establish numerical behavior
for this implementation, without evaluating trained quality.

Default Torch BF16 backward is not a bitwise full-size reference. Two unchanged
uncheckpointed rank-32 runs differed by about 0.6% in the worst parameter, with
comparable AC/LAC variation. Disabling the autocast cache did not remove it.
Torch correctness checks therefore use deterministic algorithms and
CUBLAS_WORKSPACE_CONFIG=:4096:8; repeated uncheckpointed backward, AC, and LAC then
agree exactly. Tensor qualification retains the original kernels. These controls
are limited to correctness checks; primary performance settings are unchanged.
The [diagnostics](results/loop-sweep/diagnostics/) preserve the original strict
failure and repeat/control results.

'''
    text += f'Observed worst within-backend checkpoint gradient relative L2: **{maximum:.6g}**. See the [gradient checks CSV](results/loop-sweep/views/exact-gradient-checks.csv).\n\n'
    text += '''## Records and reproduction

- [Combined CSV](results/loop-sweep/views/combined.csv): all eager/graph metrics, parameters, caches, origins, and raw-record paths.
- [Combined audit](results/loop-sweep/views/audit.json) and [view manifest](results/loop-sweep/views/run-manifest.json).
- Immutable measurement manifests: [baseline](results/loop-baseline/run-manifest.json) and [extension](results/loop-sweep/run-manifest.json).
- [Result layout and relocation map](results/README.md); [reproduction guide](../../docs/REPRODUCIBILITY.md).
- [Supplementary capture retries](results/loop-baseline/views/capture-recovery.csv), separate from primary OOMs.

The plots also have PDF exports. Eager training, prompt, and serving-startup plots
are beside the plotted graph results. Original payloads and source snapshots
remain byte-preserved; regenerated views describe the current file locations.
Earlier exploratory studies are indexed in the [archive](../../archive/README.md).

## Combined tables for every loop count

'''
    text += (VIEWS / 'tables.md').read_text()
    from experiments.l40s.research_report import section
    text += section()
    if results_root() == PUBLISHED_RESULTS:
        destination = ROOT / 'experiments/l40s/LOOP_SWEEP.md'
    else:
        destination = VIEWS / 'LOOP_SWEEP.md'
        # Reports for fresh runs link to their own results, with guides in this checkout.
        text = text.replace('results/loop-sweep/views/', '')
        text = text.replace('results/loop-sweep/', '../')
        text = text.replace('results/loop-baseline/', '../../loop-baseline/')
        text = text.replace('results/README.md', str(PUBLISHED_RESULTS / 'README.md'))
        text = text.replace('../../docs/', str(ROOT / 'docs') + '/')
        text = text.replace('../../archive/', str(ROOT / 'archive') + '/')
    destination.write_text(text.rstrip() + '\n')
    print('published:', destination)


if __name__ == '__main__':
    main()
