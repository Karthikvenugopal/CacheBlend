"""One bounded screen and, conditionally, exhaustive actual token repair."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import torch

from kvrepair.engine import CacheEngine, Edit, UnsupportedEdit, logit_metrics
from kvrepair.models import configure, load, diagnostics, MODEL, REVISION
from kvrepair.repair_probe import repair, fresh_question

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
D=ROOT/'artifacts/oracle_bound'
CONFIG={'seed':17,'threads':4,'max_new_tokens':24,'max_suffix_tokens':16,'max_oracle_cases':2,'budgets':[1,2]}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write(path,value):
    temp=Path(path).with_suffix('.next.json')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sources():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [HERE/'run.py',HERE/'candidates.json',HERE/'protocol.md',*sorted((ROOT/'src/kvrepair').glob('*.py'))]}


def render(case,tok):
    def make(doc):
        content='Use the record to answer. Return only the requested answer.\nRecord:\n'+doc+'\nQuestion: '+case['question']
        return tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    texts=[make(case['old']),make(case['new'])];ids=[];splits=[]
    for text in texts:
        data=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        end=text.index('\nQuestion: ')
        split=next(i for i,(_,b) in enumerate(data['offset_mapping']) if b>end)
        ids.append(tuple(data['input_ids']));splits.append(split)
    a,b=ids;u,v=splits
    if u!=v or a[u:]!=b[v:]:raise UnsupportedEdit('Fresh query seam or length mismatch')
    edit=Edit.between(a[:u],b[:v])
    if edit.prefix==edit.length:raise UnsupportedEdit('No actual token change')
    suffix=edit.length-edit.end
    if suffix<2 or suffix>CONFIG['max_suffix_tokens']:
        raise UnsupportedEdit(f'Suffix has {suffix} tokens; frozen exhaustive limit is 2..16')
    return {'old_prompt':texts[0],'new_prompt':texts[1],'old_ids':a,'new_ids':b,
            'old_document':a[:u],'new_document':b[:v],'question_ids':b[v:],'edit':asdict(edit),
            'suffix_positions':list(range(edit.end,edit.length))}


def mention(text,value):
    return re.search(r'(?<!\w)'+re.escape(value.casefold())+r'(?!\w)',text.casefold()) is not None


def grade(text,expected,other):
    return 'expected_only' if mention(text,expected) and not mention(text,other) else 'competing_only' if mention(text,other) and not mention(text,expected) else 'both' if mention(text,expected) and mention(text,other) else 'neither'


def decode(engine,tok,state):
    tokens=engine.generate(state,CONFIG['max_new_tokens'],tok.eos_token_id)
    return {'answer':tok.decode(tokens,skip_special_tokens=True),'generated_ids':tokens,
            'hit_token_limit':len(tokens)==CONFIG['max_new_tokens'] and tokens[-1]!=tok.eos_token_id}


def evaluate_screen_case(engine,tok,case):
    result={'case':case}
    try: view=render(case,tok)
    except (ValueError,UnsupportedEdit) as exc:
        result['rejected']=str(exc);return result
    result.update(view)
    old=engine.run(view['old_document']).state
    old_full=engine.run(view['old_ids']).state
    full=engine.run(view['new_ids']).state
    edit=view['edit'];mandatory=list(range(edit['prefix'],edit['end']))
    kv,work=repair(engine,old,view['new_document'],mandatory)
    reused=fresh_question(engine,view['new_document'],kv,view['question_ids'])
    result['original_full']=decode(engine,tok,old_full)
    result['edited_full']=decode(engine,tok,full)
    result['unrepaired_suffix']=decode(engine,tok,reused)
    result['mandatory_work']=work
    result['unrepaired_logits_vs_full']=logit_metrics(full.logits,reused.logits)
    for mode in ['original_full','edited_full','unrepaired_suffix']:
        expected=case['old_answer'] if mode=='original_full' else case['new_answer']
        other=case['new_answer'] if mode=='original_full' else case['old_answer']
        result[mode]['phrase_grade']=grade(result[mode]['answer'],expected,other)
    result['provisional_qualifier']=(result['edited_full']['phrase_grade']=='expected_only'
                                      and result['unrepaired_suffix']['phrase_grade']=='competing_only')
    result['affected_tokens']=[{'position':i,'old_id':view['old_document'][i],'new_id':view['new_document'][i],
                               'old_piece':tok.decode([view['old_document'][i]]),'new_piece':tok.decode([view['new_document'][i]])}
                              for i in mandatory]
    return result


def screen(engine,tok):
    path=D/'screen.json'
    if path.exists():raise RuntimeError('Screen already exists; preserve the fixed candidate outcomes')
    candidates=json.loads((HERE/'candidates.json').read_text())
    assert len(candidates)==12
    report={'started_utc':stamp(),'config':CONFIG,'candidate_sha256':digest(candidates),'source_sha256':sources(),
            'model':MODEL,'revision':REVISION,'environment':diagnostics(),'records':[]}
    write(D/'preregistered.json',{k:v for k,v in report.items() if k!='records'})
    for case in candidates:
        item=evaluate_screen_case(engine,tok,case);report['records'].append(item);write(path,report)
        print(case['id'],item.get('rejected',''),
              'full=',repr(item.get('edited_full',{}).get('answer')),
              'reuse=',repr(item.get('unrepaired_suffix',{}).get('answer')),
              'provisional=',item.get('provisional_qualifier',False),flush=True)
    report['completed_utc']=stamp()
    report['summary']={'candidates_screened':len(report['records']),
                       'unsupported':sum('rejected' in r for r in report['records']),
                       'edited_full_expected_only':sum(r.get('edited_full',{}).get('phrase_grade')=='expected_only' for r in report['records']),
                       'provisional_qualifiers':[r['case']['id'] for r in report['records'] if r.get('provisional_qualifier')]}
    write(path,report)


def oracle(engine,tok):
    screening=json.loads((D/'screen.json').read_text())
    audit=json.loads((D/'screen-audit.json').read_text())
    assert audit['screen_sha256']==digest(screening)
    assert screening['source_sha256']==sources(),'Sources/candidates changed since screening'
    confirmed=audit['confirmed_qualifiers']
    records=[r for r in screening['records'] if r['case']['id'] in confirmed]
    assert [r['case']['id'] for r in records]==confirmed
    assert all(r['provisional_qualifier'] for r in records)
    selected=records[:CONFIG['max_oracle_cases']]
    report={'started_utc':stamp(),'config':CONFIG,'screen_sha256':digest(screening),
            'oracle_scope':'All size-1 and size-2 token subsets of the complete short suffix; mandatory edit and fresh question common. Actual recomputation, no clean-state transplant.',
            'selected_ids':[r['case']['id'] for r in selected],
            'qualifiers_not_searched':[r['case']['id'] for r in records[len(selected):]],'cases':[]}
    path=D/'oracle.json'
    if path.exists():raise RuntimeError('Oracle report exists; do not replace outcomes')
    if not selected:
        report['stopped_reason']='No confirmed full-correct/reuse-wrong pair; factual repair evaluation unsuitable in this screened setup.'
        report['completed_utc']=stamp();write(path,report);return
    for saved in selected:
        case=saved['case'];view=render(case,tok);old=engine.run(view['old_document']).state
        full=engine.run(view['new_ids']).state;edit=view['edit'];mandatory=list(range(edit['prefix'],edit['end']))
        kv,_=repair(engine,old,view['new_document'],mandatory)
        baseline=fresh_question(engine,view['new_document'],kv,view['question_ids'])
        assert decode(engine,tok,full)['generated_ids']==saved['edited_full']['generated_ids']
        assert decode(engine,tok,baseline)['generated_ids']==saved['unrepaired_suffix']['generated_ids']
        kv,_=repair(engine,old,view['new_document'],range(edit['prefix'],edit['length']))
        endpoint=fresh_question(engine,view['new_document'],kv,view['question_ids'])
        assert logit_metrics(full.logits,endpoint.logits)['within_tolerance']
        assert decode(engine,tok,endpoint)['generated_ids']==saved['edited_full']['generated_ids']
        item={'id':case['id'],'view':view,'dependency':case['dependency'],'endpoint_passed':True,'budgets':{}}
        report['cases'].append(item)
        for budget in CONFIG['budgets']:
            supports=list(combinations(view['suffix_positions'],budget))
            outcomes=[];item['budgets'][str(budget)]={'expected_combinations':len(supports),'outcomes':outcomes}
            for support in supports:
                kv,work=repair(engine,old,view['new_document'],mandatory+list(support))
                state=fresh_question(engine,view['new_document'],kv,view['question_ids'])
                outcome={'support':support,'decoded_tokens':[tok.decode([view['new_document'][p]]) for p in support],
                         **decode(engine,tok,state),'logits_vs_full':logit_metrics(full.logits,state.logits),
                         'token_layers_recomputed':work['token_layers_recomputed']}
                outcomes.append(outcome)
                write(path,report)
            # Labels enter only after EVERY support at this budget was generated.
            for outcome in outcomes:
                outcome['phrase_grade']=grade(outcome['answer'],case['new_answer'],case['old_answer'])
            local=list(view['suffix_positions'][:budget])
            local_out=next(o for o in outcomes if list(o['support'])==local)
            correct=[list(o['support']) for o in outcomes if o['phrase_grade']=='expected_only']
            item['budgets'][str(budget)]['summary']={'recovering_supports':correct,'any_correct':bool(correct),
                                                    'local_support':local,'local_correct':local_out['phrase_grade']=='expected_only',
                                                    'oracle_outperforms_local':bool(correct) and local_out['phrase_grade']!='expected_only'}
            write(path,report)
            print(case['id'],'budget',budget,'choices',len(supports),'correct',len(correct),'local',local_out['phrase_grade'],flush=True)
    report['completed_utc']=stamp();write(path,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['screen','oracle']);args=parser.parse_args()
    configure(CONFIG['seed'],CONFIG['threads']);model,tok=load();engine=CacheEngine(model)
    if args.stage=='screen':screen(engine,tok)
    else:oracle(engine,tok)


if __name__=='__main__':main()
