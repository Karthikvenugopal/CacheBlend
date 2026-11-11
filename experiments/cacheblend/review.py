"""Describe the frozen pilot without filtering or changing it."""
import json
from datetime import datetime, timezone
from pathlib import Path
from run import HERE, OUT, normalize, save


def main():
    manifest=json.loads((HERE/'pilot.json').read_text())
    rows=[json.loads(s) for s in (OUT/'baseline.jsonl').read_text().splitlines()]
    expected={(x['id'],x['config']) for x in manifest['runs']}
    actual=[(x['id'],x['config']) for x in rows]
    assert len(actual)==len(set(actual))==len(expected) and set(actual)==expected
    coverage=[]
    for run in manifest['runs']:
        text=normalize(run['decoded_input'])
        coverage.append({'id':run['id'],'config':run['config'],
              'gold_string_present':any(normalize(g) in text for g in run['answers'] if normalize(g)),
              'retained_contexts':len(run['evidence']),
              'omitted_contexts':run['source_context_count']-len(run['evidence']),
              'truncated_contexts':sum(x['truncated'] for x in run['evidence'])})
    full_correct=sum(r['methods']['full']['exact_match'] for r in rows)
    changed=sum(not r['methods']['naive']['normalized_answer_agrees_full'] for r in rows)
    paired_losses=sum(r['methods']['full']['exact_match'] and not r['methods']['naive']['exact_match'] for r in rows)
    paired_gains=sum(not r['methods']['full']['exact_match'] and r['methods']['naive']['exact_match'] for r in rows)
    review={'recorded_utc':datetime.now(timezone.utc).isoformat(),
            'complete':True,'items':len(manifest['records']),'config_pairs':len(rows),
            'full_exact_correct_pairs':full_correct,'naive_answer_changes':changed,
            'naive_paired_gold_losses':paired_losses,'naive_paired_gold_gains':paired_gains,
            'coverage':coverage,
            'interpretation':'Gold scores characterize the fixed truncated retrieval contexts on Qwen3-1.7B. Exact string presence is a coverage heuristic, not proof of answerability. Preserve all items, including absent-answer and full-wrong cases. Changes in agreement or KL alone do not establish gold-answer improvement.',
            'interpretable':bool(full_correct and changed)}
    save(OUT/'pilot-review.json',review)
    print(json.dumps({k:v for k,v in review.items() if k!='coverage'},indent=2))

if __name__=='__main__': main()
