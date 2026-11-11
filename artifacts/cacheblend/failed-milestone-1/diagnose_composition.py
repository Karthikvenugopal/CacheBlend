"""Investigate MPS shape vs rotary correction errors at unchanged tolerance."""
import json
import gc
from pathlib import Path
import torch
from run import load, save, OUT, HERE
from core import pairs, check, compose
from transformers.models.qwen3.modeling_qwen3 import rotate_half

@torch.inference_mode()
def main():
 model,tok=load()
 run=json.loads((HERE/'pilot.json').read_text())['runs'][0]
 chunks=run['chunks']; ids=run['input_ids']; d=run['document_length']
 native=model(torch.tensor([ids],device='mps'),use_cache=True,logits_to_keep=1)
 # Retain CPU reference arrays only, release large native cache.
 full=[(k.cpu(),v.cpu()) for k,v in pairs(native.past_key_values)]
 del native; gc.collect(); torch.mps.empty_cache()
 result=[]; offset=0
 for ci,chunk in enumerate(chunks):
  x=torch.tensor([chunk],device='mps')
  local=model(x,use_cache=True,logits_to_keep=1).past_key_values
  shifted=model(x,position_ids=torch.arange(offset,offset+len(chunk),device='mps')[None],use_cache=True,logits_to_keep=1).past_key_values
  dummy=model.model.embed_tokens(x)
  cos0,sin0=model.model.rotary_emb(dummy,torch.arange(len(chunk),device='mps')[None]); cos0,sin0=cos0[:,None],sin0[:,None]
  cos1,sin1=model.model.rotary_emb(dummy,torch.arange(offset,offset+len(chunk),device='mps')[None]); cos1,sin1=cos1[:,None],sin1[:,None]
  cd,sd=model.model.rotary_emb(dummy,torch.full((1,len(chunk)),offset,device='mps')); cd,sd=cd[:,None],sd[:,None]
  # Exact numerical inverse of the actual local rotation, then target rotation.
  denom=cos0.square()+sin0.square()
  cr=(cos1*cos0+sin1*sin0)/denom
  sr=(sin1*cos0-cos1*sin0)/denom
  item={'chunk':ci,'offset':offset,'tokens':len(chunk),'layers':[]}
  for li,((kl,vl),(ks,vs)) in enumerate(zip(pairs(local),pairs(shifted))):
   delta=kl*cd+rotate_half(kl)*sd
   rephase=kl*cr+rotate_half(kl)*sr
   fkey,fval=full[li]
   item['layers'].append({'layer':li,'delta_vs_native_shift_K':check(ks,delta),
       'rephase_vs_native_shift_K':check(ks,rephase),
       'native_shift_vs_full_K':check(fkey[:,:,offset:offset+len(chunk)],ks),
       'local_vs_shift_V':check(vs,vl),
       'local_vs_full_V':check(fval[:,:,offset:offset+len(chunk)],vl),
       'key_abs_max':float(kl.abs().max()),'value_abs_max':float(vl.abs().max())})
  result.append(item); save(OUT/'composition-diagnosis.json',result)
  print('diagnosis chunk',ci,'layer0',item['layers'][0],flush=True)
  offset+=len(chunk)
  del local,shifted; gc.collect(); torch.mps.empty_cache()

if __name__=='__main__': main()
