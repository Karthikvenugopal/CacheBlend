"""Bounded numerical investigation: same first case, same MPS/BF16, no generation."""
import importlib.util,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'experiments/qwen_reproduction/run.py';spec=importlib.util.spec_from_file_location('bounded_reproduction',p);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
torch.manual_seed(17);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
model_path=ROOT/'.models/qwen3-1.7b';tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
m=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').eval()
assert torch.equal(m.lm_head.weight,m.model.embed_tokens.weight)
m.lm_head.weight=m.model.embed_tokens.weight;m.to('mps')
c=json.loads((ROOT/'experiments/qwen_reproduction/cases.json').read_text())[0]
a,b,e=r.aligned(tok,r.render(tok,c['old_body']),r.render(tok,c['new_body']))
def metrics(x,y):
 x=x[0,-1].float().cpu();y=y[0,-1].float().cpu()
 ci=tok(c['correct'],add_special_tokens=False)['input_ids'][0];si=tok(c['stale'],add_special_tokens=False)['input_ids'][0]
 return {'max_abs':float((x-y).abs().max()),'relative_l2':float((x-y).norm()/x.norm()),'within_frozen_tolerance':bool(torch.allclose(x,y,atol=.02,rtol=.02)),'vocabulary_argmax_ids':[int(x.argmax()),int(y.argmax())],'vocabulary_argmax_pieces':[tok.decode([int(x.argmax())]),tok.decode([int(y.argmax())])],'restricted_margins':[float(x[ci]-x[si]),float(y[ci]-y[si])],'restricted_choice_agrees':bool((x[ci]>=x[si])==(y[ci]>=y[si]))}
with torch.inference_mode():
 native=r.forward(m,b);repeat=r.forward(m,b)
 replay=r.last_step(m,native.past_key_values,b)
 prefix=r.forward(m,b[:-1]);incremental=r.forward(m,[b[-1]],r.clone(prefix.past_key_values),len(b)-1)
 explicit=m(input_ids=torch.tensor([[b[-1]]],device='mps'),past_key_values=r.clone(native.past_key_values,len(b)-1),cache_position=torch.tensor([len(b)-1],device='mps'),position_ids=torch.tensor([[len(b)-1]],device='mps'),attention_mask=torch.ones((1,len(b)),dtype=torch.long,device='mps'),use_cache=True,logits_to_keep=1,output_attentions=True)
 cache_drift=[]
 for i,(x,y) in enumerate(zip(native.past_key_values.layers,prefix.past_key_values.layers)):
  cache_drift.append({'layer':i,'k_max_abs':float((x.keys[:,:,:-1].float()-y.keys.float()).abs().max().cpu()),'v_max_abs':float((x.values[:,:,:-1].float()-y.values.float()).abs().max().cpu())})
 report={'case':c['id'],'scope':'First fixed case only, unchanged MPS BF16/eager/seed17. Numerical controls, no generated answers, no repair experiments.','length':len(b),'edit':e,'repeat_full':metrics(native.logits,repeat.logits),'full_vs_replayed_last':metrics(native.logits,replay.logits),'full_vs_prefix_then_last':metrics(native.logits,incremental.logits),'replayed_vs_explicit_mask_positions_and_attention_output':metrics(replay.logits,explicit.logits),'prefix_cache_drift_from_full_prefill':cache_drift}
out=ROOT/'artifacts/qwen_reproduction/endpoint-diagnosis.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='prefix_cache_drift_from_full_prefill'},indent=2))
