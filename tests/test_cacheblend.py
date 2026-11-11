"""Mechanism checks with a tiny random Qwen3; no factual claims or assets."""
import importlib.util
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

spec=importlib.util.spec_from_file_location('blend_core',Path(__file__).resolve().parents[1]/'experiments/cacheblend/core.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


@torch.inference_mode()
def test_composition_endpoints_and_ownership():
    torch.manual_seed(17)
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    config._attn_implementation='eager'
    model=Qwen3ForCausalLM(config).float().eval()
    chunks=[[1,2,3],[4,5,6,7]]; ids=sum(chunks,[])+[8,9,10]; d=7
    source=c.compose(model,chunks,validate=True)
    assert all(x['passed'] for x in source._cacheblend_rotation_checks)
    frozen=[(k.clone(),v.clone()) for k,v in c.pairs(source)]
    native=model(torch.tensor([ids]),use_cache=True)
    for a,b in zip(c.pairs(source)[0],c.pairs(native.past_key_values)[0]):
        torch.testing.assert_close(a,b[:,:,:d],atol=2e-6,rtol=2e-6)
    for mode,ratio in [('full',1.),('reference',1.)]:
        cache,logits,work=c.execute(model,ids,d,source,mode,ratio)
        torch.testing.assert_close(logits,native.logits[0,-1],atol=2e-6,rtol=2e-6)
        for ref,candidate in zip(c.pairs(native.past_key_values),c.pairs(cache)):
            for a,b in zip(ref,candidate): torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    naive,nl,nw=c.execute(model,ids,d,source,'naive')
    suffix=model(torch.tensor([ids[d:]]),past_key_values=c.clone(source),use_cache=True)
    torch.testing.assert_close(nl,suffix.logits[0,-1],atol=2e-6,rtol=2e-6)
    zero,zl,zw=c.execute(model,ids,d,source,'reference',0.)
    assert [w['qkv_rows'] for w in zw]==[10,10,3,3]
    assert [w['attention_mlp_rows'] for w in zw]==[10,3,3,3]
    assert all(w['positions'][-3:]==[7,8,9] for w in zw)
    selective,sl,sw=c.execute(model,ids,d,source,'reference',.5)
    deviation=(native.past_key_values.layers[1].values[:,:,:d]-source.layers[1].values).square().sum((0,1,3))
    expected=torch.topk(deviation,int(d*.5)).indices.sort().values.tolist()+[7,8,9]
    assert sw[1]['positions']==expected
    unselected=sorted(set(range(d))-set(expected))
    for a,b in zip(c.pairs(source)[3],c.pairs(selective)[3]):
        assert torch.equal(a[:,:,unselected],b[:,:,unselected])
    for cache in (naive,zero):
        assert all(a.data_ptr()!=b.data_ptr() for ref,candidate in zip(c.pairs(source),c.pairs(cache)) for a,b in zip(ref,candidate))
    assert all(torch.equal(a,b) for ref,candidate in zip(frozen,c.pairs(source)) for a,b in zip(ref,candidate))
    import pytest
    with pytest.raises(AssertionError, match='token IDs'):
        c.execute(model,[11]+ids[1:],d,source,'naive')
    with pytest.raises(AssertionError, match='another model'):
        other=Qwen3ForCausalLM(config).float().eval()
        c.execute(other,ids,d,source,'naive')
