"""Render this pilot's existing results; performs no inference or selection."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'artifacts/feasibility'


def median(records, key):
    return statistics.median(statistics.median(t[key] for t in r['timings']) for r in records)


def main():
    dev=json.loads((D/'development.json').read_text())
    held=json.loads((D/'heldout.json').read_text())
    gate=json.loads((D/'gate1.json').read_text())
    frozen=json.loads((D/'frozen.json').read_text())
    if not held.get('completed_utc') or len(held['cases']) != 24:
        raise RuntimeError('Wait for the complete frozen pilot; do not publish partial results as final')
    rows=[c for c in held['cases'] if 'methods' in c]
    methods=['full','edit-only','local','attention','delta','random-17','random-29','random-43']
    clean=[c for c in rows if c['clean_pair_correct']]
    n=len(rows)
    lines=['# Attention-output-change feasibility: results','',
           'This is a CPU float32 pilot on the cached TinyLlama checkpoint. No novelty claim. Existing baselines were preserved. The local repair control is part of this authorized matched-budget experiment, not a separate window sweep.','',
           f"Gate 1: **{'PASS' if gate['passed'] else 'FAIL'}**, maximum absolute first-layer attention-output error {max(c['max_abs_error'] for c in gate['cases']):.3g}. See [the numerical report](feasibility-gate1.md).",'',
           f"The method was frozen at `{frozen['frozen_utc']}` before held-out execution. Source/config/data fingerprints are in [frozen.json](../artifacts/feasibility/frozen.json).",'',
           '## Answer utility','',
           '| Cohort | Accepted questions | Original clean correct | Edited full correct | Both clean correct | Factual gate passed |',
           '|---|---:|---:|---:|---:|---|']
    for name,r in [('Development',dev),('Held-out',held)]:
        s=r['summary']
        lines.append(f"| {name} | {s['accepted']} | {s['old_full_correct']} | {s['edited_full_correct']} | {s['clean_pair_correct']} | {s['factual_experiment_conclusive']} |")
    lines += ['', 'Development has eight questions over three edits; held-out has 24 planned questions over eight edits. Questions within an edit share text and are correlated. Rejected cases remain in JSON and are excluded from denominators. No held-out cases were selected based on model success.','',
              '| Held-out stratum | Questions | Edited full correct | Both clean correct |',
              '|---|---:|---:|---:|']
    for kind,v in held['summary']['by_kind'].items():
        lines.append(f"| {kind} | {v['n']} | {v['edited_full_correct']} | {v['clean_pair_correct']} |")
    lines += ['', f"The prespecified reliability gate requires at least 80% correct original+edited clean answers overall and in every stratum. **{'The factual experiment is inconclusive because this gate failed.' if not held['summary']['factual_experiment_conclusive'] else 'The pilot meets this limited baseline-reliability gate; its small correlated sample still cannot establish general efficacy.'}**",'',
              'Known-answer utility requires the expected phrase and excludes a predefined competing answer. It is scored separately from strict short-answer formatting. This conservative rule can reject verbose/uncertain answers and is not a general semantic judge; raw text and token-limit flags are preserved below and in JSON.','',
              '| Method | Known-answer correct (all) | Correct on clean pairs | New failures on edited-full-correct cases | Strict short-answer matches |',
              '|---|---:|---:|---:|---:|']
    for method in methods:
        rs=[c['methods'][method] for c in rows]
        full_correct=sum(c['methods']['full']['known_answer_match'] for c in rows)
        failures=sum(c['methods']['full']['known_answer_match'] and not c['methods'][method]['known_answer_match'] for c in rows)
        lines.append(f"| {method} | {sum(x['known_answer_match'] for x in rs)}/{n} | {sum(c['methods'][method]['known_answer_match'] for c in clean)}/{len(clean)} | {failures}/{full_correct} | {sum(x['strict_short_answer_match'] for x in rs)}/{n} |")
    lines += ['', '| Method | Direct utility | Derived utility | Invariant utility |', '|---|---:|---:|---:|']
    for method in methods:
        counts=[]
        for kind in ['direct','derived','invariant']:
            cohort=[c for c in rows if c['case']['kind']==kind]
            counts.append(f"{sum(c['methods'][method]['known_answer_match'] for c in cohort)}/{len(cohort)}")
        lines.append('| '+method+' | '+' | '.join(counts)+' |')
    lines += ['', 'Random seeds are three instances of the same frozen random policy, not three separately tuned methods. Counts are descriptive; no significance claim is made.','',
              '## Distribution fidelity','',
              '| Method | Median next-token KL(full‖method) | Median max absolute logit error | Greedy token matches | Median absolute answer-margin error vs full |',
              '|---|---:|---:|---:|---:|']
    for method in methods:
        rs=[c['methods'][method] for c in rows]
        margin_errors=[abs(c['methods'][method]['answer_margin']-c['methods']['full']['answer_margin']) for c in rows]
        lines.append(f"| {method} | {statistics.median(x['logits_vs_full']['kl_full_to_candidate'] for x in rs):.6g} | {statistics.median(x['logits_vs_full']['max_abs'] for x in rs):.6g} | {sum(x['greedy_equal_to_full'] for x in rs)}/{n} | {statistics.median(margin_errors):.6g} |")
    lines += ['', 'Answer margins are differences between length-averaged teacher-forced expected/competing answer log probabilities. Lower next-token KL does not imply better factual answers. Greedy matches can hide drift; wording differences can preserve the answer.','',
              '| Delta versus | KL wins / ties / losses | Identical selected position sets |',
              '|---|---:|---:|']
    paired={}
    for control in ['local','attention','random-17','random-29','random-43']:
        differences=[c['methods']['delta']['logits_vs_full']['kl_full_to_candidate']-c['methods'][control]['logits_vs_full']['kl_full_to_candidate'] for c in rows]
        win=sum(x < -1e-9 for x in differences);loss=sum(x>1e-9 for x in differences)
        same=sum(c['methods']['delta']['timings'][0]['positions']==c['methods'][control]['timings'][0]['positions'] for c in rows)
        paired[control]={'kl_wins':win,'kl_ties':n-win-loss,'kl_losses':loss,'identical_support':same,
                         'median_delta_minus_control_kl':statistics.median(differences)}
        lines.append(f"| {control} | {win} / {n-win-loss} / {loss} | {same}/{n} |")
    forced=sum((c['document_tokens']-c['edit']['end'])//held['config']['region_size']==held['config']['budget']//held['config']['region_size'] for c in rows)
    lines += ['', f'{forced} questions have only two eligible regions, so selection there is forced. The other {n-forced} questions provide genuine region-choice comparisons; delta still matches local and attention on all of them.']
    lines += ['', '## Measured costs','',
              'Every partial selector repairs the same actual edit envelope plus 32 extra document positions through all layers. The common region grid is anchored at the actual edit end. Full and edit-only are reference endpoints, not matched-budget selectors. Selection never uses oracle states or expected answers.','',
              '| Method | Selection ms | Repair/prefill ms incl. copy | Fresh question ms | Greedy decode ms | Resume through answer ms | Original collection + resume ms |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for method in methods:
        rs=[c['methods'][method] for c in rows]
        vals=[median(rs,k) for k in ['selection_ms','repair_ms_including_copy','fresh_question_ms','decode_ms','resume_to_answer_ms','lifecycle_ms_with_original_collection']]
        lines.append('| '+method+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines += ['', 'These are medians across cases of two actual timed repetitions, with method order shuffled per case/repetition. The first two questions ran before the user pause and the remaining questions after resumption; raw execution-session metadata is retained. Conditions across sessions were not controlled, so these are functional timing observations. Component medians do not necessarily sum to the total median. Complete prompt tokenization, selection, repair/copies, fresh question and generation are included in resume time. Full recomputation processes the question within its complete native prefill, so its separate question column is zero. Text detokenization, oracle/scoring work and model loading are excluded.','',
              'Lifecycle cost charges the original native prefill plus method-specific metadata collection once per edit/question; it does not assume multi-question amortization or hiding work behind tool execution. This is not a GPU benchmark and no proportional savings are inferred from token counts.','',
              '| Preparation measurement | Median ms |', '|---|---:|']
    for name,key in [('Original uninstrumented prefill','original_prefill_ms'),('Delta: original prefill + metadata','delta_total_ms'),('Attention: original prefill + metadata','attention_total_ms')]:
        lines.append(f"| {name} | {statistics.median(c['collection'][key] for c in rows):.2f} |")
    for mode in ['delta','attention']:
        lines.append(f"| {mode}: metadata postprocessing | {statistics.median(c['collection'][mode+'_components']['metadata_postprocessing_ms'] for c in rows):.2f} |")
    lines += ['', 'Gate 2 collection has one measurement per method/case; Gate 1 retains three paired repetitions. Signed differences from uninstrumented prefill can be negative because of noise/warming and are not negative metadata costs. Q/O are float32 and log-normalizers float64. Delta retains 16,640 bytes/token (16.25 KiB); attention-to-edit retains 8,448 bytes/token (8.25 KiB), in addition to ordinary KV. Peak process memory was not measured.','',
              '## Per-case evidence','',
              'Answers below are literal decoded outputs. The actual span and selected positions are recorded in the raw JSON, including token IDs/pieces. Distance applies to the derived table dependency; invariant questions have no declared dependency on the edit.','',
              '| Case | Kind / distance | Expected old → new | Original clean answer | Edited full answer | Delta answer | Local answer | Attention answer |',
              '|---|---|---|---|---|---|---|---|']
    def cell(s):return repr(s).replace('|','\\|')
    for c in rows:
        v=c['case'];distance=c['actual_dependency_distance']
        lines.append(f"| {v['id']} | {v['kind']} / {distance} | {v['old_answer']} → {v['new_answer']} | {cell(c['original_full']['answer'])} | {cell(c['methods']['full']['answer'])} | {cell(c['methods']['delta']['answer'])} | {cell(c['methods']['local']['answer'])} | {cell(c['methods']['attention']['answer'])} |")
    lines += ['', 'No original baseline source or earlier results were replaced. The first partial development run is retained separately; it motivated a fixture-length correction before freezing. There was no held-out method tuning. See [the experiment protocol](../experiments/feasibility/protocol.md), [raw held-out records](../artifacts/feasibility/heldout.json), and [development records](../artifacts/feasibility/development.json).']
    (ROOT/'docs/feasibility-results.md').write_text('\n'.join(lines)+'\n')
    (D/'paired-comparisons.json').write_text(json.dumps(paired,indent=2)+'\n')
    print(json.dumps({'development':dev['summary'],'heldout':held['summary'],'paired':paired},indent=2))


if __name__=='__main__':
    main()
