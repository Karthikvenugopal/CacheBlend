"""Offline MPS runner. Append-only item results, resumable without dropping failures."""
import argparse
import collections
import gc
import hashlib
import json
import platform
import re
import string
import sys
from pathlib import Path
from time import perf_counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import check, clone, compose, divergence, execute, generate, pairs, sync

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/cacheblend'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def load():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    assert torch.backends.mps.is_available(), 'MPS required; no CPU fallback'
    path = ROOT/'.models/qwen3-1.7b'
    manifest = json.loads((path/'download-manifest.json').read_text())
    assert manifest['revision'] == '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
    for name, meta in manifest['files'].items():
        with (path/name).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest() == meta['sha256'], name
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
              dtype=torch.float32, attn_implementation='eager').to('mps').eval()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return model, tok


@torch.inference_mode()
def milestone(model, runs):
    results = []
    for run in runs[:2]:
        ids, d = run['input_ids'], run['document_length']
        source = compose(model, run['chunks'], validate=True)
        frozen = [(k.clone(),v.clone()) for k,v in pairs(source)]
        first_len = len(run['chunks'][0])
        native = model(torch.tensor([ids], device='mps'), use_cache=True,
                       logits_to_keep=torch.tensor([first_len-1,len(ids)-1],device='mps'))
        checks = {}
        tensor_diagnostics = [check(a[:,:,:first_len],b[:,:,:first_len])
            for ref, candidate in zip(pairs(native.past_key_values), pairs(source))
            for a,b in zip(ref,candidate)]
        checks['first_chunk_next_token_logits'] = check(native.logits[0,0],source._cacheblend_first_logits)
        checks['position_correction_all_chunks_all_layers'] = source._cacheblend_rotation_checks
        checks['first_layer_all_documents'] = [check(a[:,:,:d],b)
             for a,b in zip(pairs(native.past_key_values)[0],pairs(source)[0])]
        custom, logits, work = execute(model,ids,d,source,'full')
        checks['custom_full_logits'] = check(native.logits[0,-1],logits)
        checks['custom_full_all_kv'] = [check(a,b) for ref,candidate in zip(pairs(native.past_key_values),pairs(custom)) for a,b in zip(ref,candidate)]
        replay = clone(native.past_key_values)
        token = int(native.logits[0,-1].argmax())
        continuation = model(torch.tensor([[token]],device='mps'),past_key_values=replay,use_cache=True,logits_to_keep=1)
        direct = model(torch.tensor([ids+[token]],device='mps'),use_cache=True,logits_to_keep=1)
        checks['native_cached_replay'] = check(direct.logits[0,-1],continuation.logits[0,-1])
        # Compare custom naive path to native suffix prefill over composed document cache.
        naive, nl, _ = execute(model,ids,d,source,'naive')
        suffix = model(torch.tensor([ids[d:]],device='mps'),past_key_values=clone(source),use_cache=True,logits_to_keep=1)
        checks['native_naive_logits'] = check(suffix.logits[0,-1],nl)
        checks['native_naive_all_kv'] = [check(a,b) for ref,candidate in zip(pairs(suffix.past_key_values),pairs(naive)) for a,b in zip(ref,candidate)]
        checks['source_unchanged'] = all(torch.equal(a,b) for ref,candidate in zip(frozen,pairs(source)) for a,b in zip(ref,candidate))
        checks['independent_storage'] = all(a.data_ptr()!=b.data_ptr() for ref,candidate in zip(pairs(naive),pairs(source)) for a,b in zip(ref,candidate))
        values = [v for x in checks.values() for v in (x if isinstance(x,list) else [x])]
        passed = all(v['passed'] if isinstance(v,dict) else v for v in values)
        results.append({'id':run['id'],'config':run['config'],'checks':checks,'passed':passed,
                        'native_prefix_tensor_diagnostics':tensor_diagnostics,
                        'naive_next_token_divergence':divergence(native.logits[0,-1],nl)})
        save(OUT/'milestone.json', {'results':results,'complete':len(results)==2,
                                  'passed':len(results)==2 and all(x['passed'] for x in results)})
        print('milestone',run['config'],passed,flush=True)
        if not passed:
            raise RuntimeError('Numerical gate failed; see milestone.json')
        del native,custom,naive,source,frozen,replay,continuation,direct,suffix
        gc.collect(); torch.mps.empty_cache()


def normalize(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())


def quality(pred, golds):
    p = normalize(pred).split()
    fs=[]
    for gold in golds:
        g=normalize(gold).split()
        common=sum((collections.Counter(p)&collections.Counter(g)).values())
        fs.append(2*common/(len(p)+len(g)) if p or g else 1.)
    return {'exact_match':max(int(normalize(pred)==normalize(g)) for g in golds),'word_f1':max(fs)}


@torch.inference_mode()
def selection_endpoint(model, runs):
    results=[]
    for run in runs[:2]:
        ids,d=run['input_ids'],run['document_length']
        source=compose(model,run['chunks'])
        native=model(torch.tensor([ids],device='mps'),use_cache=True,logits_to_keep=1)
        candidate,logits,_=execute(model,ids,d,source,'reference',1.)
        checks=[check(native.logits[0,-1],logits)]
        checks += [check(a,b) for ref,alt in zip(pairs(native.past_key_values),pairs(candidate)) for a,b in zip(ref,alt)]
        results.append({'id':run['id'],'config':run['config'],'checks':checks,
                        'passed':all(x['passed'] for x in checks)})
        save(OUT/'selection-endpoint.json',{'results':results,
             'passed':len(results)==2 and all(x['passed'] for x in results)})
        assert results[-1]['passed'], 'reference full-selection endpoint failed'
        del native,source,candidate,logits
        gc.collect(); torch.mps.empty_cache()


@torch.inference_mode()
def pilot(model,tok,runs,stage):
    path=OUT/f'{stage}.jsonl'
    previous=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done={(x['id'],x['config']) for x in previous}
    for run in runs:
        if (run['id'],run['config']) in done: continue
        ids,d=run['input_ids'],run['document_length']
        source=compose(model,run['chunks'])
        methods = [('full','full',1.),('naive','naive',0.)] if stage=='baseline' else [
                   ('zero','reference',0.),('hkvd_16','reference',.16),('hkvd_32','reference',.32)]
        baseline = None
        if stage != 'baseline':
            baseline = next(x for x in (json.loads(s) for s in (OUT/'baseline.jsonl').read_text().splitlines()) if x['id']==run['id'] and x['config']==run['config'])
        results={}
        full_logits=None
        # Native full reference is always measured; custom full was gated separately.
        native=model(torch.tensor([ids],device='mps'),use_cache=True,logits_to_keep=1)
        full_logits=native.logits[0,-1].clone()
        del native
        for label,mode,ratio in methods:
            sync(model); start=perf_counter()
            cache,logits,work=execute(model,ids,d,source,mode,ratio)
            sync(model); prefill_ms=(perf_counter()-start)*1000
            gen,eos=generate(model,cache,logits)
            text=tok.decode(gen,skip_special_tokens=True)
            results[label]={'answer':text,'generated_ids':gen,'ended_eos':eos,
                            **quality(text,run['answers']),**divergence(full_logits,logits),
                            'prefill_ms_including_cache_copy':prefill_ms,'work':work,
                            'qkv_rows':sum(x['qkv_rows'] for x in work),
                            'attention_mlp_rows':sum(x['attention_mlp_rows'] for x in work)}
            if mode=='full':
                results[label]['endpoint_check']=check(full_logits,logits)
                assert results[label]['endpoint_check']['passed']
            del cache,logits
        full=(baseline or {'methods':results})['methods']['full']
        for value in results.values():
            value['generated_ids_agree_full']=value['generated_ids']==full['generated_ids']
            value['normalized_answer_agrees_full']=normalize(value['answer'])==normalize(full['answer'])
            value['qkv_fraction']=value['qkv_rows']/(len(ids)*len(model.model.layers))
            value['attention_mlp_fraction']=value['attention_mlp_rows']/(len(ids)*len(model.model.layers))
        record={'id':run['id'],'config':run['config'],'answers':run['answers'],
                'input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                'tokens':len(ids),'document_tokens':d,'fresh_tokens':len(ids)-d,'methods':results}
        with path.open('a') as stream: stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(stage,run['id'],run['config'],{k:round(v['word_f1'],3) for k,v in results.items()},flush=True)
        del source,full_logits
        gc.collect(); torch.mps.empty_cache()


def summarize():
    summary={}
    for stage in ('baseline','selection'):
        path=OUT/f'{stage}.jsonl'
        if not path.exists(): continue
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        for config in ('4x128','8x128'):
            subset=[r for r in rows if r['config']==config]
            if not subset: continue
            for name in subset[0]['methods']:
                vals=[r['methods'][name] for r in subset]
                metrics=['exact_match','word_f1','generated_ids_agree_full','normalized_answer_agrees_full',
                         'kl_full_to_method','total_variation','top1_agreement','qkv_fraction','attention_mlp_fraction']
                summary[f'{config}/{name}']={'count':len(vals), **{m:sum(v[m] for v in vals)/len(vals) for m in metrics},
                                              'truncated':sum(not v['ended_eos'] for v in vals)}
    save(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['milestone','baseline','selection','summary'])
    args=parser.parse_args()
    if args.stage=='summary': return summarize()
    runs=json.loads((HERE/'pilot.json').read_text())['runs']
    if args.stage=='selection':
        assert json.loads((OUT/'pilot-review.json').read_text())['interpretable']
    if args.stage!='milestone':
        milestone_result=json.loads((OUT/'milestone.json').read_text())
        assert milestone_result['passed'] and len(milestone_result['results'])==2
    model,tok=load()
    save(OUT/'environment.json',{'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),
         'device':'mps','dtype':'float32','attention':'eager','seed':17,'deterministic':True,
         'pilot_sha256':hashlib.sha256((HERE/'pilot.json').read_bytes()).hexdigest()})
    if args.stage=='milestone': milestone(model,runs)
    else:
        if args.stage=='selection': selection_endpoint(model,runs)
        pilot(model,tok,runs,args.stage)

if __name__=='__main__': main()
