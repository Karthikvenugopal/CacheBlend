"""Bounded, post-pilot diagnosis. No generation, tuning or benchmark additions.

Replays only original first-layer scores for 11 distinct original document
edits. Synthetic cases test attention algebra and the existing score/selector
paths with controlled Q/K/V, independently of language-model answers.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import torch

from kvrepair.engine import CacheEngine, Edit, Snapshot
from kvrepair.models import configure, load
from kvrepair import repair_probe as p

ROOT = Path(__file__).resolve().parents[2]
D = ROOT/'artifacts/feasibility'
OUT = D/'diagnostic'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def reference_attention(q,k,v,groups,scale):
    """Independent scalar float64 softmax, with explicit head/causal indices."""
    b,h,n,d = q.shape
    assert b==1
    output=torch.zeros_like(q,dtype=torch.float64)
    probabilities=torch.zeros((1,h,n,n),dtype=torch.float64)
    lse=torch.zeros((1,h,n),dtype=torch.float64)
    for head in range(h):
        kh=head//groups
        for i in range(n):
            scores=[math.fsum(float(q[0,head,i,x])*float(k[0,kh,j,x]) for x in range(d))*scale for j in range(i+1)]
            peak=max(scores);z=math.fsum(math.exp(s-peak) for s in scores)
            lse[0,head,i]=peak+math.log(z)
            for j,s in enumerate(scores):
                probabilities[0,head,i,j]=math.exp(s-peak)/z
            for x in range(d):
                output[0,head,i,x]=math.fsum(float(probabilities[0,head,i,j])*float(v[0,kh,j,x]) for j in range(i+1))
    return output,probabilities,lse


def controlled_case(engine,name,q,k,v,new_k,new_v):
    n=q.shape[-2];a=engine.model.model.layers[0].self_attn
    old_o,old_p,lse=reference_attention(q,k,v,a.num_key_value_groups,a.scaling)
    complete_k,complete_v=k.clone(),v.clone()
    complete_k[...,0:1,:]=new_k;complete_v[...,0:1,:]=new_v
    new_o,_,_=reference_attention(q,complete_k,complete_v,a.num_key_value_groups,a.scaling)
    ids=tuple(range(1,n+1));changed=(100,)+ids[1:]
    old=Snapshot(ids,((k,v),),torch.empty(0),engine.owner)
    meta=p.AttentionMetadata(ids,engine.owner,q,old_o.float(),lse)
    # This controlled attention snapshot is used for first-layer score tests
    # only; it is not a valid whole-model cache and is never used for inference.
    with patch.object(p,'project',return_value=(q[...,0:1,:],new_k,new_v)):
        delta,_=p.scores(engine,old,meta,changed,'delta')
        attention,_=p.scores(engine,old,meta,changed,'attention')
    expected_delta=(new_o-old_o).transpose(1,2).reshape(n,-1).norm(dim=-1)
    expected_delta[0]=0  # Edited queries are excluded from the fixed-Q formula.
    expected_attention=old_p[...,0].mean(1)[0];expected_attention[0]=0
    assert torch.allclose(delta.double(),expected_delta,atol=2e-6,rtol=2e-5)
    assert torch.allclose(attention.double(),expected_attention,atol=2e-6,rtol=2e-5)
    edit=Edit.between(ids,changed)
    delta_choice=p.select_regions(edit,delta,'delta',budget=1,region_size=1)
    attention_choice=p.select_regions(edit,attention,'attention',budget=1,region_size=1)
    return {'name':name,'queries':q.tolist(),'old_keys':k.tolist(),'old_values':v.tolist(),
            'new_edited_key':new_k.tolist(),'new_edited_value':new_v.tolist(),
            'attention_scores':attention.tolist(),'delta_scores':delta.tolist(),
            'delta_choice':delta_choice,'attention_choice':attention_choice,
            'independent_delta_max_error':float((delta.double()-expected_delta).abs().max())}


def controlled_examples():
    configure(17,1)
    model,_=load(toy=True);engine=CacheEngine(model)
    a=model.model.layers[0].self_attn
    h=model.config.num_attention_heads;hk=model.config.num_key_value_heads;d=a.head_dim
    q=torch.zeros((1,h,4,d));q[:,:,3,0]=1
    k=torch.zeros((1,hk,4,d));v=k.clone()
    nk=k[...,0:1,:].clone();nv=v[...,0:1,:].clone();nv[...,0]=1
    proportional=controlled_case(engine,'value-only, aligned heads',q,k,v,nk,nv)
    assert proportional['delta_choice']==proportional['attention_choice']==[1]
    for i in [1,2,3]:
        assert abs(proportional['delta_scores'][i]-math.sqrt(h)*proportional['attention_scores'][i])<2e-6
    nk[...,0]=math.log(12)/a.scaling
    reversal=controlled_case(engine,'key and value change, position ranking reversal',q,k,v,nk,nv)
    assert reversal['attention_choice']==[1] and reversal['delta_choice']==[3]
    # Value-only edit with different query-head attention distributions.
    q=torch.zeros((1,h,3,d));k=torch.zeros((1,hk,3,d));v=k.clone();k[:,:,0,0]=1
    q[:,:,1,0]=math.log(.6/.4)/a.scaling
    q[:,:h//2,2,0]=math.log(2*.99/.01)/a.scaling
    q[:,h//2:,2,0]=math.log(2*.05/.95)/a.scaling
    nk=k[...,0:1,:].clone();nv=v[...,0:1,:].clone();nv[...,0]=1
    heads=controlled_case(engine,'value-only, head aggregation ranking reversal',q,k,v,nk,nv)
    assert heads['attention_choice']==[1] and heads['delta_choice']==[2]
    return [proportional,reversal,heads]


def selector_controls():
    edit=Edit(7,8,77)  # Four full regions and an excluded five-token tail.
    a=torch.zeros(edit.length);b=a.clone()
    a[40:72]=1;b[24:40]=2;b[56:72]=1
    choices={mode:p.select_regions(edit,signal,mode,32,16) for mode,signal in [('delta',a),('attention',b)]}
    assert choices['delta']==list(range(40,72))
    assert choices['attention']==list(range(24,40))+list(range(56,72))
    for signal in [a,b]:
        independent=sorted([(start,float(sum(signal[start:start+16].tolist()))) for start in [8,24,40,56]],key=lambda x:(-x[1],x[0]))[:2]
        expected=sorted(i for start,_ in independent for i in range(start,start+16))
        for mode in ['delta','attention']:
            assert p.select_regions(edit,signal,mode,32,16)==expected
    ones=torch.ones(edit.length)
    assert p.select_regions(edit,ones,'delta',32,16)==list(range(8,40))
    assert p.select_regions(edit,ones,'attention',32,16)==list(range(8,40))
    return {'independent_signals_choose_distinct_regions':True,'global_offsets_checked':True,
            'same_signal_same_result_across_modes':True,'exact_ties_choose_earliest_regions':True,
            'delta_support':choices['delta'],'attention_support':choices['attention'],
            'excluded_tail_positions':list(range(72,77))}


def ranks(xs):
    order=sorted(range(len(xs)),key=lambda i:xs[i]);out=[0.0]*len(xs);a=0
    while a<len(order):
        b=a+1
        while b<len(order) and xs[order[b]]==xs[order[a]]:b+=1
        for j in range(a,b):out[order[j]]=(a+b-1)/2
        a=b
    return out


def corr(a,b):
    ma,mb=mean(a),mean(b)
    x=[v-ma for v in a];y=[v-mb for v in b]
    den=math.sqrt(math.fsum(v*v for v in x)*math.fsum(v*v for v in y))
    return math.fsum(u*v for u,v in zip(x,y))/den if den else None


def rank_summary(attention,delta):
    a,d=attention.double(),delta.double()
    coefficient=float((a@d)/(a@a).clamp_min(1e-30))
    return {'pearson':corr(a.tolist(),d.tolist()),'spearman':corr(ranks(a.tolist()),ranks(d.tolist())),
            'best_proportional_fit_relative_residual':float((d-coefficient*a).norm()/d.norm().clamp_min(1e-30)),
            'best_proportional_coefficient':coefficient,
            'attention_order':sorted(range(len(a)),key=lambda i:(-float(a[i]),i)),
            'delta_order':sorted(range(len(d)),key=lambda i:(-float(d[i]),i))}


@torch.inference_mode()
def replay_scores(engine,record,group_id):
    old_ids,new_ids=tuple(record['old_document']),tuple(record['new_document'])
    old,meta,_=p.collect(engine,old_ids,True)
    k_backup=old.kv[0][0].clone();v_backup=old.kv[0][1].clone()
    attention,_=p.scores(engine,old,meta,new_ids,'attention');saved_attention=attention.clone()
    delta,info=p.scores(engine,old,meta,new_ids,'delta')
    assert attention.data_ptr()!=delta.data_ptr() and torch.equal(attention,saved_attention)
    assert torch.equal(k_backup,old.kv[0][0]) and torch.equal(v_backup,old.kv[0][1])
    edit=Edit.between(old_ids,new_ids);e,n=edit.end,edit.length
    # Independent global-row reference: recompute native new first attention,
    # then explicitly concatenate heads at each global position.
    direct_new=p.direct_attention(engine,new_ids)
    independent_delta=torch.zeros(n)
    for i in range(e,n):
        independent_delta[i]=(direct_new[0,:,i,:].double()-meta.output[0,:,i,:].double()).norm()
    assert torch.allclose(delta,independent_delta,atol=2e-6,rtol=2e-4)
    # Independent full softmax, rather than the production edited-column/LSE path.
    from transformers.models.llama.modeling_llama import repeat_kv
    a=engine.model.model.layers[0].self_attn
    raw=(meta.q@repeat_kv(old.kv[0][0],a.num_key_value_groups).transpose(-1,-2))*a.scaling
    mask=torch.arange(n)[None,:]>torch.arange(n)[:,None]
    raw.masked_fill_(mask,-torch.inf)
    independent_attention=raw.softmax(-1)[...,edit.prefix:e].sum(-1).mean(1)[0]
    independent_attention[:e]=0
    assert torch.allclose(attention,independent_attention,atol=2e-6,rtol=2e-4)
    blocks=[(i,i+16) for i in range(e,n-15,16)]
    block_a=torch.tensor([float(attention[x:y].sum()) for x,y in blocks],dtype=torch.float64)
    block_d=torch.tensor([float(delta[x:y].sum()) for x,y in blocks],dtype=torch.float64)
    independent_a=[math.fsum(float(independent_attention[i]) for i in range(x,y)) for x,y in blocks]
    independent_d=[math.fsum(float(independent_delta[i]) for i in range(x,y)) for x,y in blocks]
    block_stats=rank_summary(block_a,block_d)
    ranks_={};selected={};margins={};reverse_tie_support={}
    for mode,values,signal,ind_values in [('attention',block_a,attention,independent_a),('delta',block_d,delta,independent_d)]:
        order=sorted(range(len(blocks)),key=lambda j:(-float(values[j]),blocks[j][0]))
        ranks_[mode]=order;selected[mode]=p.select_regions(edit,signal,mode,32,16)
        independent_order=sorted(range(len(blocks)),key=lambda j:(-ind_values[j],blocks[j][0]))
        independent_choice=sorted(i for j in independent_order[:2] for i in range(*blocks[j]))
        assert selected[mode]==independent_choice
        reverse=sorted(range(len(blocks)),key=lambda j:(-float(values[j]),-blocks[j][0]))
        reverse_tie_support[mode]=sorted(i for j in reverse[:2] for i in range(*blocks[j]))
        margins[mode]={'second_minus_third':float(values[order[1]]-values[order[2]]) if len(blocks)>2 else None,
                       'exact_cutoff_tie':bool(values[order[1]]==values[order[2]]) if len(blocks)>2 else None,
                       'relative_second_third_gap':float((values[order[1]]-values[order[2]])/values[order[1]].clamp_min(1e-30)) if len(blocks)>2 else None}
        assert selected[mode]==record['methods'][mode]['timings'][0]['positions'][edit.end-edit.prefix:]
    return {'group_id':group_id,'split':record['case']['split'],'edit':asdict(edit),
            'case_ids':[], 'attention_token_scores':attention.tolist(),'delta_token_scores':delta.tolist(),
            'token_rank_statistics':rank_summary(attention[e:],delta[e:]),
            'region_rank_statistics':block_stats,'regions':[{'start':x,'end':y,'attention':float(block_a[i]),'delta':float(block_d[i])} for i,(x,y) in enumerate(blocks)],
            'region_rank_orders':ranks_,'selected_positions':selected,'cutoff_margins':margins,
            'reverse_tie_break_changes_support':{mode:reverse_tie_support[mode]!=selected[mode] for mode in selected},
            'score_arrays_alias':False,'original_first_layer_cache_unchanged':True,
            'independent_global_row_delta_max_error':float((delta-independent_delta).abs().max()),
            'independent_full_softmax_mass_max_error':float((attention-independent_attention).abs().max()),
            'excluded_tail_positions':list(range(e+len(blocks)*16,n)), 'fallback':info}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((D/'frozen.json').read_text())
    before={path:sha(ROOT/path) for path in frozen['source_hashes']}
    assert before==frozen['source_hashes'],'Frozen implementation changed'
    input_hashes={name:sha(D/name) for name in ['frozen.json','development.json','heldout.json']}
    report={'scope':'Original score replay and synthetic attention diagnostics only; no generation, new benchmark cases, tuning or implementation changes.',
            'input_sha256':input_hashes,'selector_controls':selector_controls(),
            'controlled_attention':controlled_examples(),'original_document_groups':[]}
    configure(17,4)
    model,_=load();engine=CacheEngine(model)
    groups=defaultdict(list)
    for filename in ['development.json','heldout.json']:
        for record in json.loads((D/filename).read_text())['cases']:
            if 'methods' in record:
                key=(tuple(record['old_document']),tuple(record['new_document']))
                groups[key].append(record)
    assert len(groups)==11,'Bounded audit expects exactly 3 development + 8 held-out edits'
    for number,records in enumerate(groups.values()):
        result=replay_scores(engine,records[0],number)
        result['case_ids']=[x['case']['id'] for x in records]
        for record in records:
            for mode in ['attention','delta']:
                assert result['selected_positions'][mode]==record['methods'][mode]['timings'][0]['positions'][result['edit']['end']-result['edit']['prefix']:]
        report['original_document_groups'].append(result)
        dump(OUT/'diagnosis.json',report)
        print(records[0]['case']['id'], 'regions=',len(result['regions']),
              'token Spearman=',result['token_rank_statistics']['spearman'],
              'region Spearman=',result['region_rank_statistics']['spearman'],
              'all-region orders equal=',result['region_rank_orders']['attention']==result['region_rank_orders']['delta'],flush=True)
    assert before=={path:sha(ROOT/path) for path in before}
    assert input_hashes=={name:sha(D/name) for name in input_hashes}
    report['frozen_source_and_original_results_preserved']=True
    dump(OUT/'diagnosis.json',report)


if __name__=='__main__':
    main()
