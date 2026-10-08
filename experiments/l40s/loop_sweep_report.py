"""Generate the public experiment report from the audited sweep, using CPU only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'benchmarks/results/l40s-loop-sweep'
MODELS=('llt','naive_loop','stacked','fixed_depth')
LABELS=dict(llt='LLT',naive_loop='Naive Loop',stacked='Independent stack',fixed_depth='Fixed depth, matched params')


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text())
    assert summary['status']==audit['status']=='passed' and len(summary['rows'])==256
    rows={(r['phase'],r['model'],r['backend'],r['loops']):r for r in summary['rows']}
    def cell(r,metric):
        return f"{r[metric+'_gpu_ms']:.2f} / {r[metric+'_peak_gib']:.2f}" if metric+'_gpu_ms' in r else 'OOM'
    text=['# L40S four-model loop sweep — 2026-10-08','',
          'Batch **4**, sequence length **1024**, and loop counts **1 through 16**. '
          'The complete grid covers LLT, Naive Loop, an independent deeper stack, '
          'and a fixed-depth model whose active parameter count exactly matches that stack. '
          '[Tensor](https://github.com/kreasof-ai/tensor) supplies numerical kernels, '
          'with matched PyTorch / Flash SDPA / fused AdamW controls.','',
          f"All **256 requested cases** were attempted: **{sum(r['status']=='passed' for r in rows.values())} passed**, "
          f"**{sum(r['status']=='out_of_memory' for r in rows.values())} recorded OOMs**. "
          f"Eight small-geometry output/all-parameter-gradient/cache checks passed, and **{audit['tensor_artifact_count']} CUDA artifacts** "
          'were audited against their SHA-256 hashes and `sm_89` target. '
          'No numerical fallback was reported by any successful Tensor case.','',
          'These are seeded synthetic performance measurements. Model quality is not compared. '
          'All latencies are per batch, and cached decoding produces one token for each of the four sequences.','',
          '## Model definitions','',
          '| Architecture | Unique blocks | Applied blocks | MLP hidden width |',
          '|---|---:|---:|---:|',
          '| LLT | 12 | 12T | 3072 |','| Naive Loop | 12 | 12T | 3072 |',
          '| Independent stack | 12T | 12T | 3072 |','| Fixed depth, matched parameters | 12 | 12 | 768(6T−2) |','',
          'Every model uses width 768, 12 heads, head dimension 64, vocabulary 50,304, '
          'untied input/output embeddings, exact GELU, unweighted RMSNorm, and learned absolute positions '
          'with capacity 1025. LLT shares a rank-64 latent across every block and pass. '
          'Residuals/master weights/embeddings/AdamW states are FP32; projections and attention are BF16. '
          'Inference retains explicit BF16 linear-weight copies on both backends.','',
          'The fixed-depth model matches the **stack**, not the recurrent models. '
          'It uses all matching parameters in its MLPs. Its attention depth stays at 12; '
          'equal parameter counts therefore do not imply equal attention work or activation memory. '
          'At T=1, the three conventional architectures coincide and were measured independently.','',
          '| T | LLT parameters | Naive Loop parameters | Stack = fixed-depth parameters |',
          '|---:|---:|---:|---:|']
    for t in (1,4,8,16):
        text.append('| '+str(t)+' | '+' | '.join(f"{rows[('training',m,'tensor',t)]['parameter_count']:,}" for m in ('llt','naive_loop','stacked'))+' |')
    text += ['', '## Sixteen-loop results','',
             'Full training step CUDA graph latency / capture peak allocated memory (**ms / GiB**). '
             'This includes full-token logits/loss, backward, global gradient clipping, and AdamW. '
             'No activation checkpointing or streamed classifier is enabled.','',
             '| Architecture | Tensor | PyTorch / fused AdamW |','|---|---:|---:|']
    for m in MODELS:
        text.append(f"| {LABELS[m]} | {cell(rows[('training',m,'tensor',16)],'training_graph')} | {cell(rows[('training',m,'torch',16)],'training_graph')} |")
    text += ['', 'Cached single-token inference after a 1024-token history, with persistent KV stores. '
             'Latency / capture peak allocated memory (**ms / GiB**). Logical rewind is included in graph replay; '
             'sampling and beam search are outside this experiment.','',
             '| Architecture | Tensor | PyTorch | KV cache, Tensor (MiB) |','|---|---:|---:|---:|']
    for m in MODELS:
        r=rows[('inference',m,'tensor',16)]
        text.append(f"| {LABELS[m]} | {cell(r,'decode_graph')} | {cell(rows[('inference',m,'torch',16)],'decode_graph')} | {r['cache_mib']:.2f} |")
    text += ['', 'Prompt processing (causal 1024-token forward, last logits, no persistent KV allocation) '
             'and serving startup (including KV allocation/copy and LLT fold rebuild) are separate measurements. '
             'Tensor CUDA graph latency / capture peak allocated memory (**ms / GiB**):','',
             '| Architecture | Prompt | Serving startup |','|---|---:|---:|']
    for m in MODELS:
        r=rows[('inference',m,'tensor',16)]
        text.append(f"| {LABELS[m]} | {cell(r,'prompt_graph')} | {cell(r,'startup_graph')} |")
    text += ['', '## Scaling and memory limits','',
             'LLT’s persistent KV cache stays at approximately **0.50 MiB** across T. '
             'Naive Loop and the independent stack retain approximately **144.14 MiB × T**. '
             'The fixed-depth model keeps 12 KV stores (approximately 144.14 MiB), '
             'while its parameter memory grows to match the deeper stack. '
             'Prompt-only inference can discard intermediate K/V; its memory does not imply the same saving as persistent serving.','',
             'Total LLT training memory grows with T. The backend speed and memory comparisons also change with T; '
             'the figures retain every requested loop count and show eager execution separately from CUDA graph replay.','',
             'Largest measured T with a successful operation (all cases through T=16 were attempted):','',
             '| Architecture | Tensor eager training | Tensor graph training | Torch eager training | Torch graph training |',
             '|---|---:|---:|---:|---:|']
    for m in MODELS:
        maxima=[]
        for backend,metric in (('tensor','training_eager'),('tensor','training_graph'),('torch','training_eager'),('torch','training_graph')):
            valid=[t for t in range(1,17) if metric+'_gpu_ms' in rows[('training',m,backend,t)]]
            maxima.append(str(max(valid)) if valid else 'none')
        text.append('| '+LABELS[m]+' | '+' | '.join(maxima)+' |')
    oom=[r for r in rows.values() if r['status']=='out_of_memory']
    if oom:
        text += ['','Recorded failures:']
        text += [f"- {LABELS[r['model']]} / {r['backend']} / T={r['loops']} / `{r['failed_stage']}`." for r in oom]
        text += ['','A recorded OOM does not substitute a smaller batch, shorter sequence, or checkpointed model. '
                 'Operations completed before an OOM remain in the CSV; later skipped operations have no timing value.']
    if summary.get('capture_recovery'):
        text += ['', '## Controlled capture retries','',
                 'These additional measurements release eager gradients, collect garbage, and empty cached allocations after '
                 'graph warmup and before capture. Batch, sequence, model parameters, numerical kernels, optimizer, '
                 'and the 42-update counter check stay identical. They do not replace the original grid. '
                 'A capture OOM is therefore distinguished from an eager-training OOM and from avoidable setup storage.','',
                 '| Architecture | Backend | T | Retry graph ms / peak GiB | Released gradients (GiB) |',
                 '|---|---|---:|---:|---:|']
        for r in summary['capture_recovery']:
            value=f"{r['graph_ms']:.2f} / {r['capture_peak_gib']:.2f}" if 'graph_ms' in r else 'OOM'
            released=f"{r['released_gradient_gib']:.2f}" if 'released_gradient_gib' in r else '—'
            text.append(f"| {LABELS[r['model']]} | {r['backend']} | {r['loops']} | {value} | {released} |")
    text += ['', '## Figures and complete data','',
             '![Training CUDA graph latency and memory](results/l40s-loop-sweep/training-graph.png)','',
             '![Cached inference latency and memory](results/l40s-loop-sweep/decode-graph.png)','',
             '![Parameter counts and KV cache scaling](results/l40s-loop-sweep/parameters-and-cache.png)','',
             '- [Every case and metric as CSV](results/l40s-loop-sweep/summary.csv).',
             '- [All 16 Tensor loop counts as latency/memory tables](results/l40s-loop-sweep/tables.md).',
             '- [Prompt graphs](results/l40s-loop-sweep/prompt-graph.png) and [serving-startup graphs](results/l40s-loop-sweep/startup-graph.png).',
             '- [Eager training](results/l40s-loop-sweep/training-eager.png), [prompt](results/l40s-loop-sweep/prompt-eager.png), '
             '[startup](results/l40s-loop-sweep/startup-eager.png), and [decode](results/l40s-loop-sweep/decode-eager.png).',
             '- Every PNG has a matching standalone PDF in [the results directory](results/l40s-loop-sweep).',
             '- [Raw artifact/input audit](results/l40s-loop-sweep/audit.json) and [runtime/source manifest](results/l40s-loop-sweep/run-manifest.json).','',
             '## Measurement and reproduction','',
             'Each GPU case uses a fresh process; all jobs run sequentially. There are three warmups and nine timing samples. '
             'Eager measurements retain CUDA-event and synchronized wall time. Each CUDA graph sample averages three replays. '
             'Training checks finite losses and every parameter gradient, and verifies that GPU optimizer counters reach exactly 42 updates. '
             'Graph capture records kernels without executing an update. Compilation/warmup are excluded from latency.','',
             'Eager peak memory is the maximum **allocated** during the measured call. Graph peak memory is the maximum allocated '
             'during **capture**, including temporaries and private graph storage. These include weights, prepared inference copies, '
             'gradients, optimizer state, and live KV stores where applicable. Replay allocation and allocator reservation peaks '
             'are retained separately in JSON. Driver/context allocations and allocator reservations are outside the plotted allocated metric.','',
             'See the [complete reproduction protocol](../experiments/l40s/LOOP_SWEEP.md). Run:', '', '```bash',
             'LLT_RESUME=1 experiments/l40s/run_loop_sweep.sh',
             'LLT_RESUME=1 experiments/l40s/run_loop_sweep_recovery.sh',
             'python experiments/l40s/loop_sweep_summary.py',
             'python experiments/l40s/loop_sweep_manifest.py',
             'python experiments/l40s/loop_sweep_report.py','```','',
             'The Tensor dependency is pinned in each raw report. Tensor development records remain in '
             'the [Tensor repository](https://github.com/kreasof-ai/tensor). '
             'The earlier [actual nanoGPT profile](L40S_NANOGPT.md) is retained as a separate study.','']
    (ROOT/'benchmarks/L40S_LOOP_SWEEP.md').write_text('\n'.join(text))
    print('report written')


if __name__=='__main__':
    main()
