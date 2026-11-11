"""Build a report from all fixed pairs, retaining negative and ambiguous evidence."""
import json
from run import OUT


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    review=json.loads((OUT/'pilot-review.json').read_text())
    assert json.loads((OUT/'audit.json').read_text())['passed']
    assert len(summary)==10 and all(v['count']==40 for v in summary.values())
    n4,n8=[summary[f'{c}/naive'] for c in ('4x128','8x128')]
    f4,f8=[summary[f'{c}/full'] for c in ('4x128','8x128')]
    a4,a8=[summary[f'{c}/hkvd_16'] for c in ('4x128','8x128')]
    b4,b8=[summary[f'{c}/hkvd_32'] for c in ('4x128','8x128')]
    lines=['# CacheBlend mechanism and quality reproduction','',
      'Internal learning work. The edit-repair direction remains closed in its [separate final archive](../../archives/kvrepair-final-2026-11-09/README.md). No novelty or serving speedup is claimed.','',
      'Completed all 40 fixed items × two configurations × five methods: 400 generated outputs, with no dropped items or configurations. The implementation demonstrates actual selective recomputation and a work–divergence tradeoff, but gold-quality gains are mixed and full-model quality is weak. It does not establish the paper’s quality-preservation result on this model.','',
      f"At 16% document selection, actual QKV work is {100*min(a4['qkv_fraction'],a8['qkv_fraction']):.1f}–{100*max(a4['qkv_fraction'],a8['qkv_fraction']):.1f}% of full. Word F1 is {a4['word_f1']:.3f}/{a8['word_f1']:.3f} for 4×128/8×128, versus naive reuse’s {n4['word_f1']:.3f}/{n8['word_f1']:.3f}. The 32% setting uses {100*min(b4['qkv_fraction'],b8['qkv_fraction']):.1f}–{100*max(b4['qkv_fraction'],b8['qkv_fraction']):.1f}% QKV work and reaches {b4['word_f1']:.3f}/{b8['word_f1']:.3f} F1, versus full recomputation’s {f4['word_f1']:.3f}/{f8['word_f1']:.3f}. Mean initial next-token KL falls from {n4['kl_full_to_method']:.2f}/{n8['kl_full_to_method']:.2f} under naive reuse to {a4['kl_full_to_method']:.2f}/{a8['kl_full_to_method']:.2f} at 16% and {b4['kl_full_to_method']:.2f}/{b8['kl_full_to_method']:.2f} at 32%. Even at 32%, normalized answer agreement with full is only {100*b4['normalized_answer_agrees_full']:.0f}%/{100*b8['normalized_answer_agrees_full']:.0f}%. These results show approximation effects, not reliable preservation of full behavior.",'',
      'Paper: [arXiv 2405.16444v3](https://arxiv.org/html/2405.16444v3). Reference: [YaoJiayi/CacheBlend, 55ad026](https://github.com/YaoJiayi/CacheBlend/tree/55ad02675939f783a38d579393527d218a7fd581). Cached Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; MPS float32/eager, deterministic greedy decoding, 32-token cap. [Protocol and source mapping](../../experiments/cacheblend/protocol.md).','',
      'The fixed pilot contains the first 20 MuSiQue and first 20 2WikiMQA records from the pinned release, with both first-4 and first-8 retrieved context configurations, capped at 128 Qwen tokens per context plus labels/separators. No item or configuration was dropped or selected by outcome. All 80 input assemblies, raw source records and truncations are in [pilot.json](../../experiments/cacheblend/pilot.json). Dataset access and downloads were confirmed by the user before fetching.','',
      'Composition uses independent chunk prefills and native-coefficient `R(global) R(local)^-1` correction of post-RoPE K. V is unchanged. Every method uses the same assembled token IDs, one native chat wrapper, and computes the entire question/assistant-prefix suffix fresh at every layer. Source identity, token identity, immutable cache ownership and independent generation branches are enforced.','',
      'The [milestone](milestone.json) validates every chunk/layer correction against captured pre-RoPE keys, first-layer K/V, first-prefix logits, native full/cache replay, native naive suffix execution, custom full logits/KV and independent storage on both configurations at atol=rtol=2e-4. The initial offset-only rotation failed; its [failed run](failed-milestone-1/milestone.json) is preserved. Five deep prefix V tensors still fail the same mixed tolerance in two native executions with different total attention shapes. First-prefix logits pass. These raw tensors are not claimed numerically equivalent. The numerical collection gate was refined after diagnosis; see the protocol. No CPU–MPS equivalence claim is made.','',
      'Reference V-HKVD selects once at zero-based layer 1 using summed squared V deviation from independently cached values. Layer 0 runs fully; layer 1 refreshes all QKV but only selected attention/MLP rows survive. Deeper layers recompute selected rows and splice them into owned stale KV. Ratios .16 and .32 were fixed before inference. This follows the release schedule, with a corrected absolute-position causal mask and full fresh suffix. It does not implement the paper’s underspecified gradual filtering. [Full-selection endpoint checks](selection-endpoint.json) compare the ratio-1 reference path to native full.','',
      '**Zero selection is separate from naive reuse.** It still refreshes document QKV in layers 0 and 1 and document attention/MLP in layer 0. Naive reuses all document KV on every layer. Mandatory fresh suffix work is included in every fraction.','',
      '| Contexts | Method | Gold EM | Gold word F1 | Answer agreement | Next-token agreement | KL(full‖method) | TV | QKV rows / full | Attention+MLP rows / full | Truncated |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for config in ['4x128','8x128']:
      for method in ['full','naive','zero','hkvd_16','hkvd_32']:
        v=summary.get(f'{config}/{method}')
        if not v: continue
        p=lambda k:f'{100*v[k]:.1f}%'
        lines.append(f"| {config} | {method} | {p('exact_match')} | {v['word_f1']:.3f} | {p('normalized_answer_agrees_full')} | {p('top1_agreement')} | {v['kl_full_to_method']:.4f} | {v['total_variation']:.4f} | {p('qkv_fraction')} | {p('attention_mlp_fraction')} | {v['truncated']}/{v['count']} |")
    lines += ['',
      'Each row includes all 40 items; scores, divergence and work fractions are means over those items. Gold scores use normalized whole-output EM and word-overlap F1 (best gold alternative); this differs from the release’s tokenizer-token F1 and yes/no parsing. Whole-output scoring penalizes elaboration; EM losses are score transitions, not manually adjudicated factual errors. Inspect the raw answers. Answer agreement means normalized agreement with full; exact generated-ID agreement is also saved. KL/TV compare the initial next-token distributions on the identical prompt, not later divergent generation prefixes. Truncated outputs remain scored and flagged.','',
      f"Before selection, full recomputation matched gold exactly on {review['full_exact_correct_pairs']}/80 pairs. Naive changed {review['naive_answer_changes']}/80 normalized answers. Paired EM losses: {review['naive_paired_gold_losses']}; paired EM gains: {review['naive_paired_gold_gains']} relative to full. The [pilot review](pilot-review.json) was recorded before the selection run. No workload or ratio was tuned after it.",'']
    for config in ['4x128','8x128']:
      vals=[r for r in review['coverage'] if r['config']==config]
      lines.append(f"For {config}, a normalized gold string occurs in {sum(x['gold_string_present'] for x in vals)}/40 assembled inputs; {sum(x['truncated_contexts'] for x in vals)} retained contexts were truncated and {sum(x['omitted_contexts'] for x in vals)} source contexts omitted. String presence is only a coverage heuristic, not proof of answerability. Missing evidence and small-model errors limit gold-quality interpretation.")
    lines += ['',
      'The results characterize this shortened retrieval pilot on Qwen3-1.7B, not the paper’s model/workload quality. Improvement in KL or agreement alone is not gold-answer improvement. Full recomputation is the agreement reference, not an oracle gold solver.','',
      'Work fractions count projected token rows and attention/MLP rows separately; they are not measured FLOPs. A native full reference is computed for evaluation; neither its cache nor its logits feed selection. That evaluation overhead is excluded from method work and timing. Prefill timings in the raw records also exclude independent chunk prefill, model loading, validation and generation, and include the custom executor’s cache copies. These local fixed-order observations do not reproduce CUDA serving TTFT, throughput, SSD retrieval or overlap speedups.','',
      'Evidence: [baseline outputs](baseline.jsonl), [selection outputs](selection.jsonl), [summary](summary.json), [completeness and token/work audit](audit.json), [source manifest](source-manifest.json), [archive verification](edit-repair-archive-verification.json). The implementation is [core.py](../../experiments/cacheblend/core.py); meaningful tiny-model mechanism tests are [test_cacheblend.py](../../tests/test_cacheblend.py). All failures, items and configurations remain available.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    print(OUT/'report.md')

if __name__=='__main__': main()
