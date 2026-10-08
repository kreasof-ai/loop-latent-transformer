"""Publish the audited exact rank/AC/LAC tables into LOOP_SWEEP.md."""
import json
from latent_checkpoint_summary import ROOT,OUT,VARIANTS,LABELS


def main():
    audit=json.loads((OUT/'audit.json').read_text())
    assert audit['status']=='passed' and audit['case_count']==672
    rows=json.loads((OUT/'combined.json').read_text())
    records={(r['phase'],r['model'],r['kv_rank'] or 64,r['backend'],r['loops'],r['checkpoint']):r for r in rows}
    def cell(model,rank,backend,policy):
        if policy=='lac' and model!='llt':return 'N/A'
        r=records[('training',model,rank,backend,16,policy)]
        if 'training_graph_gpu_ms' not in r:
            return 'OOM' if r['failed_stage']=='training_graph' else '— (eager OOM)'
        return f"{r['training_graph_gpu_ms']:.2f} / {r['training_graph_peak_gib']:.2f}"
    intro='## Exact rank, AC, and native-boundary LAC results\n\n'
    intro+=f'**{audit["case_count"]} primary performance records:** {audit["new_case_count"]} new cases and {audit["reused_case_count"]} unchanged original cases. **{audit["passed"]} passed; {audit["out_of_memory"]} recorded OOM.** There are {audit["qualification_count"]} small qualification cases and {audit["full_qualification_count"]} full-size T=16 qualification cases, with {audit["unique_artifact_count"]} independently hashed sm89 CUDA artifacts.\n\n'
    intro+='All runs use B4/S1024, width 768, 12 heads, 12 base blocks, vocabulary 50,304, and T=1..16. LLT KV ranks are 32, 64, and 128. Conventional model geometry, the exact stack/fixed-depth parameter match, precision, seeded tokens, optimizer, timing samples, and primary graph-capture setup follow the original protocol above.\n\n'
    intro+='**AC** checkpoints one full Transformer block with non-reentrant recomputation. **LAC** checkpoints only LLT’s existing latent attention and folded output projection, using the original projected queries Q_r, shared KV latent C, and folded output weight as explicit inputs. It adds no codec or trainable parameters and preserves all gradient paths. LAC uses the model’s existing rank; there is no separate checkpoint-compression rank. Naive Loop and both conventional stacked controls have no such latent boundary and show **LAC: N/A**.\n\n'
    intro+='The policies cover different regions: AC recomputes a whole block; LAC recomputes the latent attention/output branch. Query formation, residual and MLP paths remain outside LAC and contribute to peak memory. The shared latent alone does not reconstruct those states or establish constant total training memory. This is exact checkpointing of the unchanged forward model.\n\n'
    intro+='Inference stores no backward activations, so AC/LAC do not create separate inference configurations. Rank-32/rank-128 LLT inference is newly measured; rank-64 and baseline inference is reused. The original 256 records are preserved, and supplementary capture-setup retries are not substituted for primary OOMs. A skipped graph after eager OOM is labelled separately. Compilation/warmup are excluded; every successful full training profile checks finite loss/gradients and all 42 optimizer updates.\n\n'
    intro+='### Exactness and cache qualification\n\n'
    intro+='The small checks cover every variant at T=4 and T=16 with width 128, two base blocks, batch 2, and sequence 32. Both backends compare AC and applicable LAC gradients against uncheckpointed gradients, and Tensor against Torch within BF16 tolerances. Full-size checks use width 768, 12 base blocks, B4/S1024 at T=16, unchanged initial weights, no optimizer between policies, and CPU reference gradients to bound GPU residency. Each full-size check repeats uncheckpointed backward before comparing policies. Torch correctness checks enable deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8; Tensor checks use the unchanged kernels. These correctness controls do not replace or alter the primary performance settings. All parameter relative L2 errors must be below 1e-5; forward losses must agree exactly. Every check also compares supplied-token cached decode after a prefix with matching full-forward logits (maximum error <0.05; relative L2 <0.03). These qualify numerical behavior and do not establish trained language-model quality.\n\n'
    intro+='Default Torch BF16 backward is not a bitwise reference at full size. In the rank-32 diagnostic, two unchanged uncheckpointed runs differ by about 0.6% in the worst parameter, with comparable AC/LAC differences. Disabling the autocast cache does not remove this variation. Deterministic execution gives zero error for repeated uncheckpointed backward, AC and LAC. This is floating-point execution variability, without an approximate checkpoint codec. The [diagnostics](../../benchmarks/results/l40s-latent-checkpoint-sweep/diagnostics/) retain both the original failed strict check and the repeat/control results.\n\n'
    maximum=max(result['gradient_error']['maximum_parameter_relative_l2'] for path in OUT.glob('*qualification-*.json')
                for result in json.loads(path.read_text())['policy_checks'].values() if 'gradient_error' in result)
    intro+=f'Observed worst within-backend checkpoint gradient relative L2: **{maximum:.6g}**. Full details: [exact-gradient CSV](../../benchmarks/results/l40s-latent-checkpoint-sweep/exact-gradient-checks.csv).\n\n'
    intro+='### T=16 training endpoint\n\nCells are **CUDA graph ms / capture peak allocated GiB**, per batch of four.\n\n'
    for backend in ('tensor','torch'):
        intro+=f'**{backend.capitalize()}**\n\n| Architecture | None | AC | LAC |\n|---|---:|---:|---:|\n'
        for model,rank in VARIANTS:intro+='| '+LABELS[(model,rank)]+' | '+' | '.join(cell(model,rank,backend,p) for p in ('none','ac','lac'))+' |\n'
        intro+='\n'
    intro+='### Data, figures, and reproduction\n\n'
    intro+='- [Complete combined CSV](../../benchmarks/results/l40s-latent-checkpoint-sweep/combined.csv), including eager GPU/wall latency, serving startup, capture peak allocation, origin, parameters, cache bytes, and raw-record paths. Raw records also retain replay allocation.\n'
    intro+='- [Audit](../../benchmarks/results/l40s-latent-checkpoint-sweep/audit.json) and [manifest](../../benchmarks/results/l40s-latent-checkpoint-sweep/run-manifest.json).\n'
    intro+='- Numerical kernels use [Tensor](https://github.com/kreasof-ai/tensor); successful Tensor cases have zero implicit numerical fallback. GPU jobs run sequentially in fresh processes.\n\n'
    intro+='```bash\nLLT_RESUME=1 experiments/l40s/run_latent_checkpoint_sweep.sh\npython experiments/l40s/latent_checkpoint_summary.py\npython experiments/l40s/latent_checkpoint_report.py\npython experiments/l40s/latent_checkpoint_manifest.py\n```\n\n'
    intro+='![Training GPU latency](../../benchmarks/results/l40s-latent-checkpoint-sweep/training_graph-gpu_ms.png)\n\n'
    intro+='![Training peak allocated memory](../../benchmarks/results/l40s-latent-checkpoint-sweep/training_graph-peak_gib.png)\n\n'
    intro+='![Cached inference](../../benchmarks/results/l40s-latent-checkpoint-sweep/decode_graph.png)\n\n'
    intro+='PNG/PDF exports for eager training, prompt inference, serving startup and cached decode are beside the captured training figures. Raw samples, runtime versions, compiled artifact hashes and source snapshots remain available.\n\n'
    intro+='### Combined tables for every loop count\n\n'
    tables=(OUT/'tables.md').read_text()
    protocol=ROOT/'experiments/l40s/LOOP_SWEEP.md'
    original=protocol.read_text().split('\n## Next rank and checkpoint comparison')[0].split('\n## Exact rank, AC, and native-boundary LAC results')[0].rstrip()
    original=original.replace('# L40S four-model loop sweep','# L40S loop, rank, and checkpoint sweep',1)
    original=original.replace('GPU jobs are sequential. No activation checkpointing or streamed loss is enabled.',
        'GPU jobs are sequential. The original 256-case study below uses no activation checkpointing or streamed loss. The completed [rank/AC/LAC comparison](#exact-rank-ac-and-native-boundary-lac-results) follows it.')
    protocol.write_text((original+'\n\n'+intro+tables).rstrip()+'\n')
    # Repo-native second entry point uses paths relative to benchmarks/.
    report='# L40S exact native-latent checkpoint and rank sweep\n\n'+intro+tables
    report=report.replace('../../benchmarks/','').replace('follow the original protocol above','follow the [original protocol](../experiments/l40s/LOOP_SWEEP.md)')
    (ROOT/'benchmarks/L40S_LATENT_CHECKPOINT.md').write_text(report.rstrip()+'\n')
    print('combined tables published in experiments/l40s/LOOP_SWEEP.md')


if __name__=='__main__':main()
