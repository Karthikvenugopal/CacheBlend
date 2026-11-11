"""Offline, serialized checkpoint placement; no full CPU float32 model."""
import gc, hashlib, json
from pathlib import Path
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
ROOT=Path(__file__).resolve().parents[3]

def load(mark, guard):
 torch.set_num_threads(4);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
 assert torch.backends.mps.is_available()
 path=ROOT/'.models/qwen3-1.7b'
 manifest=json.loads((path/'download-manifest.json').read_text())
 assert manifest['revision']=='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
 mark('verify-checkpoint')
 for name,meta in manifest['files'].items():
  guard()
  with (path/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==meta['sha256']
 config=AutoConfig.from_pretrained(path,local_files_only=True)
 # Original runtime is untied and preserves BOTH stored matrices.
 tied=config.tie_word_embeddings;config.tie_word_embeddings=False
 with init_empty_weights():
  model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32,attn_implementation='eager')
 model.config.tie_word_embeddings=tied
 index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
 assert set(index)==set(dict(model.named_parameters()))
 records=[]
 # Bounded slices avoid a full-sized float32 cast of either large matrix.
 for name in sorted(index,key=lambda n:dict(model.named_parameters())[n].numel(),reverse=True):
  guard();mark('load:'+name)
  parent,leaf=name.rsplit('.',1);module=model.get_submodule(parent)
  old=module._parameters[leaf]
  dst=torch.empty(old.shape,dtype=torch.float32,device='mps')
  with safe_open(path/index[name],framework='pt',device='cpu') as f:
   view=f.get_slice(name)
   for start in range(0,old.shape[0],1024):
    guard();chunk=view[start:start+1024].float()
    dst[start:start+1024].copy_(chunk)
    # Exact float32 stored-value equality, with bounded host scratch.
    assert torch.equal(dst[start:start+1024].cpu(),chunk),name
    del chunk
  module._parameters[leaf]=torch.nn.Parameter(dst,requires_grad=False)
  records.append({'name':name,'shape':list(dst.shape),'exact_stored_float32_equality':True})
  del dst,old;gc.collect();torch.mps.empty_cache()
 for module in model.modules():
  for name,buf in list(module._buffers.items()):
   if buf is not None:module._buffers[name]=buf.to('mps')
 # Rotary has a nonregistered alias that must follow the live device buffer.
 model.model.rotary_emb.original_inv_freq=model.model.rotary_emb.inv_freq
 assert all(p.device.type=='mps' and p.dtype==torch.float32 for p in model.parameters())
 assert model.lm_head.weight.data_ptr()!=model.model.embed_tokens.weight.data_ptr()
 model.generation_config=GenerationConfig.from_pretrained(path,local_files_only=True)
 model.eval();mark('loaded')
 return model,AutoTokenizer.from_pretrained(path,local_files_only=True),records
