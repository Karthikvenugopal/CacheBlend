"""Two-gate offline pilot. Run with python -m kvrepair.feasibility --help."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
from time import perf_counter

import torch

from .engine import CacheEngine, Edit, UnsupportedEdit, logit_metrics
from .models import configure, diagnostics, load, MODEL, REVISION
from .repair_probe import collect, changed_output, direct_attention, fresh_question, repair, scores, select_regions
from .feasibility_cases import cases, gate1_cases

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/feasibility'
CONFIG = {'seed':17, 'threads':4, 'budget':32, 'region_size':16, 'fallback_threshold':1e-4,
          'atol':2e-4, 'rtol':2e-4, 'max_new_tokens':12, 'repeats':2,
          'random_seeds':[17,29,43], 'max_prompt_tokens':1024,
          'clean_reliability_threshold':0.8}


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'src/kvrepair').glob('*.py'))}


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def timed(engine, fn):
    engine.sync(); begin = perf_counter(); value = fn(); engine.sync()
    return value, (perf_counter()-begin)*1000


def environment():
    return {'utc':now(), 'model':MODEL, 'revision':REVISION, 'dtype':'float32',
            'device':'cpu', 'diagnostics':diagnostics(), 'config':CONFIG}


def summary_times(values):
    return {'median_ms':statistics.median(values), 'samples_ms':values}


def gate1(engine, tokenizer, path):
    report = {'gate':1, **environment(), 'source_hashes':hashes(), 'cases':[],
              'assumptions':['eval float32 native Llama eager attention', 'batch one, unpadded full causal attention',
                             'unchanged weights and equal token lengths with unchanged suffix positions',
                             'original clean cache and metadata; first-layer unchanged suffix Q only',
                             'all prompts retokenized completely; changed queries recomputed, not updated by formula'],
              'metadata_note':'Q and pre-output-projection attention output float32; log-normalizers float64. Native eager does not expose LSE: collect charges an extra chunked QK pass. Original KV remains separate.',
              'timing_note':'Three alternating-order paired repetitions, CPU synchronization boundaries; native first-layer time includes QKV/RoPE/output projection but no MLP. Collection time includes full original prefill, hooks, copies and LSE pass.'}
    engine.run(tokenizer.encode('Warmup.', add_special_tokens=False))
    for case in gate1_cases():
        a = tuple(tokenizer.encode(case['old'], add_special_tokens=False))
        b = tuple(tokenizer.encode(case['new'], add_special_tokens=False))
        item = {'case':case, 'old_ids':a, 'new_ids':b}
        try:
            edit = Edit.between(a,b)
        except UnsupportedEdit as exc:
            item.update(passed=False, error=str(exc)); report['cases'].append(item)
            write(path,report); print(case['id'], 'FAIL',exc,flush=True)
            report['passed']=False; write(path,report); return False
        item['edit'] = asdict(edit)
        base_times, collection_times, updates, directs, overheads, post_times, hook_times = [], [], [], [], [], [], []
        max_abs, relative, fallback_rows, compared_rows = [], [], [], []
        all_close = True
        for repeat in range(3):
            if repeat % 2 == 0:
                baseline = engine.run(a)
                old, meta, collection_ms = collect(engine,a)
            else:
                old, meta, collection_ms = collect(engine,a)
                baseline = engine.run(a)
            # Capturing metadata must not perturb ordinary native execution.
            baseline_match = logit_metrics(baseline.state.logits,old.logits)
            (updated, info), update_ms = timed(engine,lambda:changed_output(engine,old,meta,b))
            direct, direct_ms = timed(engine,lambda:direct_attention(engine,b))
            reference = direct[...,info['suffix_start']:, :]
            delta = (updated-reference).abs()
            error = float(delta.max()) if delta.numel() else 0.0
            rel = float((updated-reference).double().norm()/reference.double().norm().clamp_min(1e-30)) if delta.numel() else 0.0
            okay = bool(torch.allclose(updated,reference,atol=CONFIG['atol'],rtol=CONFIG['rtol']))
            all_close &= okay and baseline_match['within_tolerance']
            max_abs.append(error);relative.append(rel)
            fallback_rows.append(info['fallback_token_rows']); compared_rows.append(reference.shape[-2])
            base_times.append(baseline.elapsed_ms);collection_times.append(collection_ms)
            overheads.append(collection_ms-baseline.elapsed_ms);updates.append(update_ms);directs.append(direct_ms)
            post_times.append(meta.collection['metadata_postprocessing_ms']);hook_times.append(meta.collection['hook_copy_host_ms'])
        item.update(passed=all_close, max_abs_error=max(max_abs),relative_l2_error=max(relative),
                    metadata_bytes=meta.nbytes, metadata_tensor_dtypes={'Q':str(meta.q.dtype),'O':str(meta.output.dtype),'LSE':str(meta.lse.dtype)},
                    fallback_token_rows=fallback_rows, compared_token_rows=compared_rows,
                    original_prefill=summary_times(base_times), collection_including_original_prefill=summary_times(collection_times),
                    paired_incremental_collection=summary_times(overheads), metadata_postprocessing=summary_times(post_times), hook_copy=summary_times(hook_times), attention_update=summary_times(updates), direct_first_attention=summary_times(directs))
        report['cases'].append(item)
        write(path,report)
        print(case['id'],f'passed={all_close} max_abs={max(max_abs):.3g} metadata={meta.nbytes} B fallback={fallback_rows}',flush=True)
        if not all_close:
            report['passed']=False;write(path,report);return False
    total_rows=sum(sum(c['compared_token_rows']) for c in report['cases'])
    fallback=sum(sum(c['fallback_token_rows']) for c in report['cases'])
    report['fallback_fraction']=fallback/max(1,total_rows)
    report['passed']=all(c['passed'] for c in report['cases']) and report['fallback_fraction']<=0.05
    report['completed_utc']=now();write(path,report)
    return report['passed']


def render(case, tokenizer):
    def text(document):
        content = 'Answer using only the document. Give a short answer without explanation.\nDocument:\n'+document+'\nQuestion: '+case['question']
        return tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    before, after = text(case['old']),text(case['new'])
    encoded=[]; splits=[]
    for rendered in (before,after):
        result=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
        boundary=rendered.index('\nQuestion: ')
        split=next((i for i,(start,end) in enumerate(result['offset_mapping']) if end>boundary),len(result['input_ids']))
        encoded.append(tuple(result['input_ids']));splits.append(split)
    a,b=encoded;pa,pb=splits
    if pa!=pb or a[pa:]!=b[pb:]:
        raise UnsupportedEdit('Document/query seam must align with identical fresh-query tokens')
    edit=Edit.between(a[:pa],b[:pb])
    if len(b)+CONFIG['max_new_tokens']>CONFIG['max_prompt_tokens']:
        raise UnsupportedEdit('Pilot exceeds frozen prompt length cap')
    # Record actual evidence start using complete prompt offsets, never estimated word counts.
    evidence_char=after.index(case['evidence'])
    offsets=tokenizer(after,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    evidence_token=next(i for i,(start,end) in enumerate(offsets) if end>evidence_char)
    select_regions(edit,None,'local',CONFIG['budget'],CONFIG['region_size'])  # Preflight the matched budget before inference.
    return {'old_prompt':before,'new_prompt':after,'old_ids':a,'new_ids':b,'document_tokens':pb,
            'old_document':a[:pa],'new_document':b[:pb],'question_ids':b[pb:], 'edit':edit,
            'evidence_token':evidence_token,'table_distance_tokens':evidence_token-edit.end,
            'actual_dependency_distance':(evidence_token-edit.end if case['kind']=='derived' and case['should_change'] else 0 if case['kind']=='direct' else None)}


def answer_score(text, expected, forbidden):
    def normalize(s):
        return re.sub(r'[^\w\s]','',s.casefold()).strip()
    def contains(s,term):
        return re.search(r'(?<!\w)'+re.escape(term.casefold())+r'(?!\w)',s.casefold()) is not None
    return {'known_answer_match':contains(text,expected) and not contains(text,forbidden),
            'strict_short_answer_match':normalize(text)==normalize(expected)}


@torch.inference_mode()
def answer_logprob(engine, tokenizer, state, prompt, answer):
    combined=tuple(tokenizer.encode(prompt+answer,add_special_tokens=False))
    if combined[:len(state.ids)]!=state.ids or len(combined)==len(state.ids):
        raise ValueError('Teacher-forced answer tokenization crosses prompt boundary')
    tokens=combined[len(state.ids):]
    kv=state.kv;logits=state.logits;total=0.0
    for i,token in enumerate(tokens):
        total+=float(logits.float().log_softmax(-1)[token])
        if i+1<len(tokens):
            kv,logits=engine._forward((token,),kv)
    return total/len(tokens)


def adapter_gate(engine, tokenizer, views):
    report={'checks':[],'passed':True}
    for view in ([views[0], views[-1]] if len(views)>1 else views):
        a,b=view['old_document'],view['new_document']
        old=engine.run(a).state
        kv,_=repair(engine,old,b,range(len(b)))
        reference=engine.run(b).state
        cache_errors=[float((x-y).abs().max()) for pair,rpair in zip(kv,reference.kv) for x,y in zip(pair,rpair)]
        cache_ok=all(torch.allclose(x,y,atol=CONFIG['atol'],rtol=CONFIG['rtol']) for pair,rpair in zip(kv,reference.kv) for x,y in zip(pair,rpair))
        fresh=fresh_question(engine,b,kv,view['question_ids'])
        full=engine.run(view['new_ids']).state
        metrics=logit_metrics(full.logits,fresh.logits)
        greedy=engine.generate(full,4)==engine.generate(fresh,4)
        check={'max_cache_error':max(cache_errors),'cache_within_tolerance':cache_ok,'question_logits':metrics,'greedy_equal':greedy}
        report['checks'].append(check)
        report['passed'] &= cache_ok and metrics['within_tolerance'] and greedy
    return report


def evaluate(engine, tokenizer, split, destination, frozen=None):
    source=cases(split)
    report={**environment(),'split':split,'cases':[], 'source_hashes':hashes(),
            'dataset_sha256':fingerprint(source),'frozen_manifest_sha256':fingerprint(frozen) if frozen else None,
            'timing_note':'Per-method randomized paired repeats. Resume includes complete tokenization, selection/repair/fresh-question/decode and copies. Lifecycle adds original cache+method-specific metadata collection. Native loading, evaluation scoring and oracle runs excluded. Collection measured separately, not hidden/amortized.',
            'scoring_note':'Known-answer phrase must occur without its predefined competing answer; strict format score reported separately. This conservative rule is not a general semantic judge. Token-limit flags and raw text retained.'}
    prepared=[]
    for case in source:
        try:
            view, tokenization_ms=timed(engine,lambda:render(case,tokenizer))
            prepared.append((case,view,tokenization_ms))
        except (UnsupportedEdit,ValueError) as exc:
            report['cases'].append({'case':case,'rejected':str(exc)})
    if split=='development':
        report['adapter_gate']=adapter_gate(engine,tokenizer,[v for _,v,_ in prepared])
        if not report['adapter_gate']['passed']:
            write(destination,report);raise RuntimeError('Sparse operator endpoint gate failed; investigate before continuing')
    methods=['full','edit-only','local','attention','delta']+[f'random-{s}' for s in CONFIG['random_seeds']]
    for case,view,tokenization_ms in prepared:
        old_doc,new_doc=view['old_document'],view['new_document']
        edit=view['edit'];query=view['question_ids']
        item={'case':case, **{k:v for k,v in view.items() if k!='edit'}, 'edit':asdict(edit),
              'tokenization_ms':tokenization_ms,
              'affected_tokens':[{'position':i,'old_id':old_doc[i],'new_id':new_doc[i],
                                  'old_piece':tokenizer.decode([old_doc[i]]),'new_piece':tokenizer.decode([new_doc[i]])}
                                 for i in range(edit.prefix,edit.end)],'methods':{}}
        plain=engine.run(old_doc)
        old_delta,meta_delta,delta_collection=collect(engine,old_doc,True)
        old_attention,meta_attention,attention_collection=collect(engine,old_doc,False)
        # Every comparison starts from the same clean original, with independently owned caches.
        if not all(logit_metrics(plain.state.logits,x.logits)['within_tolerance'] for x in (old_delta,old_attention)):
            raise RuntimeError('Metadata collection perturbed original logits')
        item['collection']={'original_prefill_ms':plain.elapsed_ms,
                            'delta_total_ms':delta_collection,'attention_total_ms':attention_collection,
                            'delta_incremental_ms':delta_collection-plain.elapsed_ms,
                            'attention_incremental_ms':attention_collection-plain.elapsed_ms,
                            'delta_metadata_bytes':meta_delta.nbytes,'attention_metadata_bytes':meta_attention.nbytes,
                            'delta_components':meta_delta.collection,'attention_components':meta_attention.collection,
                            'kv_snapshot_bytes':plain.state.nbytes}
        original_state=fresh_question(engine,old_doc,plain.state.kv,query)
        original_tokens=engine.generate(original_state,CONFIG['max_new_tokens'],tokenizer.eos_token_id)
        original_answer=tokenizer.decode(original_tokens,skip_special_tokens=True)
        old_forbidden=case['new_answer'] if case['should_change'] else case['alternative']
        item['original_full']={'answer':original_answer,**answer_score(original_answer,case['old_answer'],old_forbidden),
                               'generated_ids':original_tokens,'hit_token_limit':len(original_tokens)==CONFIG['max_new_tokens'] and original_tokens[-1]!=tokenizer.eos_token_id}
        reference=engine.run(view['new_ids']).state
        for repeat in range(CONFIG['repeats']):
            order=methods.copy();random.Random(f"{CONFIG['seed']}:{case['id']}:{repeat}").shuffle(order)
            item.setdefault('method_orders',[]).append(order)
            for method in order:
                engine.sync(); begin=perf_counter()
                diagnostics_={};selection_ms=0.0;question_ms=0.0;repaired_positions=[]
                if method=='full':
                    run=engine.run(view['new_ids']);state=run.state
                    repair_info={'repair_ms_including_copy':run.elapsed_ms,'cache_copy_ms':None,
                                 'token_layers_recomputed':len(view['new_ids'])*len(state.kv),'positions':list(range(len(new_doc)))}
                else:
                    signal_mode=method if method in {'delta','attention'} else None
                    meta=meta_delta if method=='delta' else meta_attention
                    def select():
                        signal,details=scores(engine,plain.state,meta,new_doc,signal_mode) if signal_mode else (None, {})
                        if method=='edit-only':
                            chosen=[]
                        else:
                            chosen=select_regions(edit,signal,'random' if method.startswith('random-') else method,
                                                  CONFIG['budget'],CONFIG['region_size'],
                                                  int(method.split('-')[1]) if method.startswith('random-') else CONFIG['seed'])
                        return chosen,details
                    (selected,diagnostics_),selection_ms=timed(engine,select)
                    repaired_positions=list(range(edit.prefix,edit.end))+selected
                    kv,repair_info=repair(engine,plain.state,new_doc,repaired_positions)
                    state,question_ms=timed(engine,lambda:fresh_question(engine,new_doc,kv,query))
                tokens,decode_ms=timed(engine,lambda:engine.generate(state,CONFIG['max_new_tokens'],tokenizer.eos_token_id))
                engine.sync();online_ms=(perf_counter()-begin)*1000+tokenization_ms
                answer=tokenizer.decode(tokens,skip_special_tokens=True)
                collection_ms=delta_collection if method=='delta' else attention_collection if method=='attention' else plain.elapsed_ms
                sample={'selection_ms':selection_ms,**repair_info,'fresh_question_ms':question_ms,
                        'decode_ms':decode_ms,'resume_to_answer_ms':online_ms,
                        'lifecycle_ms_with_original_collection':collection_ms+online_ms}
                if repeat==0:
                    lp=answer_logprob(engine,tokenizer,state,view['new_prompt'],case['new_answer'])
                    alt=answer_logprob(engine,tokenizer,state,view['new_prompt'],case['alternative'])
                    item['methods'][method]={'answer':answer,'generated_ids':tokens,
                                            'hit_token_limit':len(tokens)==CONFIG['max_new_tokens'] and tokens[-1]!=tokenizer.eos_token_id,
                                            **answer_score(answer,case['new_answer'],case['alternative']),
                                            'expected_answer_mean_logprob':lp,'alternative_mean_logprob':alt,'answer_margin':lp-alt,
                                            'logits_vs_full':logit_metrics(reference.logits,state.logits),
                                            'additional_repaired_tokens':0 if method in {'full','edit-only'} else len(repaired_positions)-(edit.end-edit.prefix),
                                            'selection_diagnostics':diagnostics_,'timings':[]}
                else:
                    if tokens!=item['methods'][method]['generated_ids']:
                        raise RuntimeError('Repeated deterministic generation diverged; investigate')
                item['methods'][method]['timings'].append(sample)
        item['clean_pair_correct']=item['original_full']['known_answer_match'] and item['methods']['full']['known_answer_match']
        for method,result in item['methods'].items():
            result['greedy_equal_to_full']=result['generated_ids']==item['methods']['full']['generated_ids']
            result['reuse_introduced_known_answer_failure']=(not result['known_answer_match']) if item['methods']['full']['known_answer_match'] else None
        report['cases'].append(item)
        report['summary']=summarize(report)
        write(destination,report)
        print(case['id'], 'old/full correct=',item['original_full']['known_answer_match'],item['methods']['full']['known_answer_match'],
              'full=',repr(item['methods']['full']['answer']), 'delta=',repr(item['methods']['delta']['answer']),flush=True)
        # Release evaluation states before the next document; no persistent mixed-cache history.
        del meta_delta,meta_attention,old_delta,old_attention,reference,original_state,plain,state
    report['summary']=summarize(report);report['completed_utc']=now();write(destination,report)
    return report


def summarize(report):
    accepted=[c for c in report['cases'] if 'methods' in c]
    if not accepted:
        return {'accepted':0,'rejected':len(report['cases'])}
    clean=[c for c in accepted if c['clean_pair_correct']]
    summary={'accepted':len(accepted),'rejected':len(report['cases'])-len(accepted),
             'old_full_correct':sum(c['original_full']['known_answer_match'] for c in accepted),
             'edited_full_correct':sum(c['methods']['full']['known_answer_match'] for c in accepted),
             'clean_pair_correct':len(clean), 'by_kind':{},'methods':{}}
    for kind in ['direct','derived','invariant']:
        cohort=[c for c in accepted if c['case']['kind']==kind]
        summary['by_kind'][kind]={'n':len(cohort),'clean_pair_correct':sum(c['clean_pair_correct'] for c in cohort),
                                 'edited_full_correct':sum(c['methods']['full']['known_answer_match'] for c in cohort)}
    summary['factual_experiment_conclusive']=(len(clean)/len(accepted)>=CONFIG['clean_reliability_threshold']
        and all(v['n']>0 and v['clean_pair_correct']/v['n']>=CONFIG['clean_reliability_threshold'] for v in summary['by_kind'].values()))
    for method in accepted[0]['methods']:
        r=[c['methods'][method] for c in accepted]
        cost=lambda key: statistics.median(statistics.median(t[key] for t in x['timings']) for x in r)
        summary['methods'][method]={'known_answer_correct':sum(x['known_answer_match'] for x in r),
                                   'known_answer_correct_on_clean_pairs':sum(c['methods'][method]['known_answer_match'] for c in clean),
                                   'greedy_matches_full':sum(x['greedy_equal_to_full'] for x in r),
                                   'median_kl':statistics.median(x['logits_vs_full']['kl_full_to_candidate'] for x in r),
                                   'median_answer_margin':statistics.median(x['answer_margin'] for x in r),
                                   **{'median_'+key:cost(key) for key in ['selection_ms','repair_ms_including_copy','decode_ms','resume_to_answer_ms','lifecycle_ms_with_original_collection']}}
    return summary


def require_gate1(path):
    report=json.loads(Path(path).read_text())
    if not report.get('passed'):
        raise RuntimeError('Gate 1 did not pass')
    # Mechanisms must be the same code that passed numerical validation.
    for name in ['src/kvrepair/repair_probe.py','src/kvrepair/engine.py','src/kvrepair/models.py']:
        if hashes()[name]!=report['source_hashes'][name]:
            raise RuntimeError('Gate 1 mechanism source changed; rerun Gate 1')
    return report


def freeze(development_path, gate1_path, destination):
    g=require_gate1(gate1_path)
    development=json.loads(Path(development_path).read_text())
    if development['split']!='development' or development['summary']['accepted']!=8 or not development['adapter_gate']['passed']:
        raise RuntimeError('Development and sparse-operator gate must complete before freezing')
    if development['source_hashes']!=hashes():
        raise RuntimeError('Code changed after development; rerun development before freezing')
    frozen={'frozen_utc':now(),'config':CONFIG,'source_hashes':hashes(),
            'gate1_sha256':fingerprint(g),'development_sha256':fingerprint(development),
            'development_dataset_sha256':fingerprint(cases('development')),
            'heldout_dataset_sha256':fingerprint(cases('heldout')),
            'selection':'16-token disjoint regions anchored at actual edit end; sum token scores; budget 32; ascending-position tie break',
            'utility':'Expected answer phrase present and predefined competing phrase absent; retain old/new baseline failures and clean-pair subset',
            'decision_rule':'Arithmetic gate must pass. Factual efficacy inconclusive below 80% clean-pair reliability overall or in any case kind. Distribution gains alone do not establish useful repair; selector and metadata costs must be counted.',
            'heldout_inference_performed_before_freeze':False}
    if Path(destination).exists():
        raise RuntimeError('Freeze manifest already exists; do not overwrite after looking at held-out results')
    write(destination,frozen)
    print('Frozen:',destination,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['gate1','development','freeze','heldout'])
    parser.add_argument('--directory',type=Path,default=OUT)
    args=parser.parse_args();directory=args.directory;directory.mkdir(parents=True,exist_ok=True)
    if args.stage=='freeze':
        freeze(directory/'development.json',directory/'gate1.json',directory/'frozen.json');return
    frozen=None
    if args.stage!='gate1':
        require_gate1(directory/'gate1.json')
    if args.stage=='heldout':
        frozen=json.loads((directory/'frozen.json').read_text())
        if frozen['source_hashes']!=hashes() or frozen['config']!=CONFIG or frozen['heldout_dataset_sha256']!=fingerprint(cases('heldout')):
            raise RuntimeError('Frozen code/config/data changed: do not run the held-out pilot')
        if (directory/'heldout.json').exists():
            raise RuntimeError('Held-out artifact exists; keep the first pilot rather than overwriting it')
    configure(CONFIG['seed'],CONFIG['threads'])
    model,tokenizer=load();engine=CacheEngine(model)
    if args.stage=='gate1':
        if not gate1(engine,tokenizer,directory/'gate1.json'):
            raise SystemExit('Gate 1 failed; stop and investigate')
    else:
        evaluate(engine,tokenizer,'development' if args.stage=='development' else 'heldout',directory/(args.stage+'.json'),frozen)


if __name__=='__main__':
    main()
