"""Compare storage-only refactor against preserved original arithmetic on CPU."""
import importlib.util
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module('old_core','experiments/cacheblend/core.py')
new=module('new_core','experiments/cacheblend_overnight/memory_v3/core.py')
@torch.inference_mode()
def test_storage_changes_preserve_arithmetic_and_isolation():
 torch.manual_seed(17)
 config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
 config._attn_implementation='eager';model=Qwen3ForCausalLM(config).float().eval()
 chunks=[[1,2,3],[4,5,6,7]];ids=sum(chunks,[])+[8,9,10];d=7
 a=old.compose(model,chunks,validate=True);b=new.compose(model,chunks,validate=True)
 assert all(c['passed'] for c in b._cacheblend_rotation_checks)
 frozen=[(k.clone(),v.clone()) for k,v in new.pairs(b)]
 for p,q in zip(old.pairs(a),new.pairs(b)):
  for x,y in zip(p,q):assert torch.equal(x,y)
 for mode,ratio in [('full',1.),('naive',0.),('reference',0.),('reference',.16),('reference',.32),('reference',1.)]:
  ac,al,aw=old.execute(model,ids,d,a,mode,ratio)
  bc,bl,bw=new.execute(model,ids,d,None if mode=='full' else b,mode,ratio)
  assert aw==bw and torch.equal(al,bl)
  for p,q in zip(old.pairs(ac),new.pairs(bc)):
   for x,y in zip(p,q):assert torch.equal(x,y)
  assert all(x.untyped_storage().data_ptr()!=y.untyped_storage().data_ptr() for p,q in zip(new.pairs(b),new.pairs(bc)) for x,y in zip(p,q))
  # Native decode may mutate this method cache, never the immutable source.
  model(torch.tensor([[11]]),past_key_values=bc,use_cache=True)
  assert all(torch.equal(x,y) for p,q in zip(frozen,new.pairs(b)) for x,y in zip(p,q))
