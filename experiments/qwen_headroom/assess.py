"""Post-run application of the frozen decision rubric; never imported by search."""
import hashlib
import json
from pathlib import Path
from statistics import median

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/qwen_headroom'


def main():
    run = json.loads((OUT/'run.json').read_text())
    assert run['status'] in ['completed', 'failed'], 'Never label while search is running'
    assert not (OUT/'assessment.json').exists(), 'Retain previous assessment'
    fixed = json.loads((HERE/'fixed_cases.json').read_text())
    tok = AutoTokenizer.from_pretrained(ROOT/'.models/qwen3-1.7b', local_files_only=True, trust_remote_code=False)
    report = {'search_completed': run.get('completed', False),
              'oracle_attempts': run['oracle_attempts'], 'control_attempts': run['control_attempts'],
              'assessment_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'rubric': 'Frozen sole decision word, case/terminal punctuation insensitive; cutoff always inconclusive. Bank is intended-label/mechanical recovery only.',
              'cases': []}
    for case in fixed['cases']:
        path = OUT/case['id']/'outcomes.json'
        if not path.exists():
            report['cases'].append({'id': case['id'], 'status': 'not_run'})
            continue
        data = json.loads(path.read_text())
        def grade(x):
            if x['status'] != 'completed' or x['hit_limit']:
                return 'inconclusive'
            word = x['text'].strip().casefold().rstrip('.?!')
            return ('correct_intended_label' if word == case['expected_new_decision'] else
                    'wrong_old_label' if word == case['wrong_old_decision'] else 'inconclusive')
        regions = [{'index': i, 'positions': ab, 'text': tok.decode(case['new_ids'][ab[0]:ab[1]])}
                   for i, ab in enumerate(case['regions'])]
        item = {'id': case['id'], 'status': data['status'],
                'factual_semantics': 'policy decision valid, but old cache uses artificial noncanonical token padding' if case['id']=='airline_cancel' else 'malformed padded monetary field; intended-label and full-output recovery only',
                'regions': regions, 'controls': [], 'budgets': {}}
        report['cases'].append(item)
        for row in data['controls']:
            item['controls'].append({'kind': row['kind'], 'text': row.get('text'),
                 'grade': grade(row), 'timings_ms': row.get('timings_ms'), 'work': row.get('work')})
        completed = [x for x in data['trials'] if x['status']=='completed']
        singles = {x['region_indices'][0]: grade(x) for x in completed if x['budget']==16}
        for budget in [16, 32]:
            rows = [x for x in completed if x['budget']==budget]
            local_support = [0] if budget==16 else [0, 1]
            local = next((x for x in rows if x['region_indices']==local_support), None)
            correct = [x for x in rows if grade(x)=='correct_intended_label']
            item['budgets'][str(budget)] = {
                'attempted_completed': len(rows),
                'correct_supports': [x['region_indices'] for x in correct],
                'correct_count': len(correct),
                'wrong_count': sum(grade(x)=='wrong_old_label' for x in rows),
                'inconclusive_count': sum(grade(x)=='inconclusive' for x in rows),
                'truncations': sum(x['hit_limit'] for x in rows),
                'local_support': local_support,
                'local_text': local['text'] if local else None,
                'local_grade': grade(local) if local else None,
                'oracle_answer_advantage_over_local': bool(correct) and local is not None and grade(local)!='correct_intended_label',
                'full_generated_ids_equal_count': sum(x['full_generated_ids_equal'] for x in rows),
                'all_oracle_successes': [{'support': x['region_indices'], 'text': x['text'],
                    'ranges': [case['regions'][i] for i in x['region_indices']],
                    'next_token_KL': x['logits_vs_full']['kl_full_to_candidate']} for x in correct],
                'median_cost_ms': {name: median(x['timings_ms'][name] for x in rows) for name in
                    ['diagnostic_readout_including_copy','generic_metrics','generation_including_readout_and_copy',
                     'trial_compute_wall','first_checkpoint_write','trial_wall_including_first_checkpoint']} if rows else {},
                'median_repair_ms': median(x['work']['repair_ms_including_copy'] for x in rows) if rows else None,
                'median_cache_copy_ms': median(x['work']['cache_copy_ms'] for x in rows) if rows else None,
                'median_forward_ms': median(x['work']['sparse_forward_ms'] for x in rows) if rows else None,
                'median_next_token_KL': median(x['logits_vs_full']['kl_full_to_candidate'] for x in rows) if rows else None,
                'local_next_token_KL': local['logits_vs_full']['kl_full_to_candidate'] if local else None,
                'two_region_complementarity': [x['region_indices'] for x in correct if budget==32 and
                     all(singles.get(i)!='correct_intended_label' for i in x['region_indices'])],
                'adding_region_loses_correct_answer': [x['region_indices'] for x in rows if budget==32 and
                     grade(x)!='correct_intended_label' and any(singles.get(i)=='correct_intended_label' for i in x['region_indices'])]}
    (OUT/'assessment.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'completed': report['search_completed'], 'attempts': report['oracle_attempts'],
          'cases': [{'id': x['id'], 'budgets': {k:{a:v[a] for a in ['attempted_completed','correct_count','local_text','oracle_answer_advantage_over_local']} for k,v in x.get('budgets',{}).items()}} for x in report['cases']]},indent=2))


if __name__ == '__main__':
    main()
