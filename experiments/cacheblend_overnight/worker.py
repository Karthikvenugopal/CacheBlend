"""One offline model instance. Parent watchdog enforces wall time and OS pressure."""
import gc,json,os,resource,signal,sys,time,traceback
from pathlib import Path
import torch
from common import ROOT,HERE,OUT,FROZEN,sha,save,append,score,norm,result_path
sys.path.insert(0,str(ROOT/'experiments/cacheblend'))
from core import compose,execute,clone,pairs,check,divergence,sync
from run import load

class Stop(RuntimeError):pass
ACTIVE={};PARTIAL=[];MODEL=None;START=time.time();STOP=False

def meters():
 z={'elapsed_s':time.time()-START,'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 if MODEL is not None:
  sync(MODEL)
  z.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory(),mps_recommended_bytes=torch.mps.recommended_max_memory())
 save(OUT/'worker-memory.json',z);append(OUT/'worker-memory.jsonl',z)
 return z

def guard(*args):
 if STOP or (OUT/'stop-request.json').exists() or time.time()>=DEADLINE or os.getppid()!=PARENT:
  raise Stop('Stop requested, deadline reached, or supervisor exited')

def finite(t):
 if not bool(torch.isfinite(t).all()):raise RuntimeError('nonfinite output')
def finite_cache(cache):
 if not bool(torch.stack([torch.isfinite(x).all() for p in pairs(cache) for x in p]).all()):raise RuntimeError('nonfinite cache')
def cache_digest(cache):
 import hashlib
 h=hashlib.sha256()
 for p in pairs(cache):
  for t in p:h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def require(checks,label):
 save(OUT/f'validation-{label}.json',checks)
 if not all(c['passed'] for c in checks):raise RuntimeError('numerical validation failed: '+label)

@torch.inference_mode()
def decode(cache,logits,cap,eos):
 global PARTIAL
 PARTIAL=[];cache=clone(cache)
 for _ in range(cap):
  guard();finite(logits);token=int(logits.argmax());PARTIAL.append(token)
  if token in eos:break
  out=MODEL(torch.tensor([[token]],device='mps'),past_key_values=cache,use_cache=True,logits_to_keep=1)
  logits=out.logits[0,-1];cache=out.past_key_values
 finite_cache(cache)
 return list(PARTIAL),PARTIAL[-1] in eos

@torch.inference_mode()
def item_run(item,preflight,protocol,phash,tok):
 global ACTIVE
 ids=item['input_ids'];d=item['document_length'];key=item['id'];methods=protocol['methods']
 pending=[m for m in methods if not result_path(key,m[0]).exists()]
 if not pending:return
 PARTIAL.clear();ACTIVE={'item':key,'phase':'composition'};guard();t=time.monotonic()
 source=compose(MODEL,item['chunks'],validate=preflight);finite_cache(source);sync(MODEL)
 compose_s=time.monotonic()-t;source_hash=cache_digest(source);meters()
 refpath=OUT/'references'/f"{key.replace(':','_')}.pt"
 gate_names=['position-correction','prefix-logits-layer0','full-endpoint','reference-endpoint','native-replay','naive-endpoint']
 gates_complete=all((OUT/f'validation-{g}.json').exists() and all(c['passed'] for c in json.loads((OUT/f'validation-{g}.json').read_text())) for g in gate_names)
 if not refpath.exists() or (preflight and not gates_complete):
  t=time.monotonic()
  positions=torch.tensor([len(item['chunks'][0])-1,len(ids)-1],device='mps') if preflight else 1
  native=MODEL(torch.tensor([ids],device='mps'),use_cache=True,logits_to_keep=positions)
  finite(native.logits);finite_cache(native.past_key_values)
  reference=native.logits[0,-1].clone();sync(MODEL)
  reference_s=time.monotonic()-t
  torch.save(reference.cpu(),refpath)
  if preflight:
   require(source._cacheblend_rotation_checks,'position-correction')
   require([check(native.logits[0,0],source._cacheblend_first_logits)]+[check(a[:,:,:d],b) for a,b in zip(pairs(native.past_key_values)[0],pairs(source)[0])],'prefix-logits-layer0')
   save(OUT/'native-prefix-tensor-diagnostics.json',[check(a[:,:,:len(item['chunks'][0])],b[:,:,:len(item['chunks'][0])]) for p,q in zip(pairs(native.past_key_values),pairs(source)) for a,b in zip(p,q)])
   for mode in ('full','reference'):
    guard();cache,logits,_=execute(MODEL,ids,d,source,mode,1.)
    require([check(reference,logits)]+[check(a,b) for p,q in zip(pairs(native.past_key_values),pairs(cache)) for a,b in zip(p,q)],mode+'-endpoint')
    del cache,logits;gc.collect();torch.mps.empty_cache();meters()
   token=int(reference.argmax()); replay=MODEL(torch.tensor([[token]],device='mps'),past_key_values=clone(native.past_key_values),use_cache=True,logits_to_keep=1)
   replay_logits=replay.logits[0,-1].clone();del replay
   del native;gc.collect();torch.mps.empty_cache()
   direct=MODEL(torch.tensor([ids+[token]],device='mps'),use_cache=True,logits_to_keep=1)
   require([check(direct.logits[0,-1],replay_logits)],'native-replay');del direct,replay_logits
   cache,logits,_=execute(MODEL,ids,d,source,'naive')
   native_suffix=MODEL(torch.tensor([ids[d:]],device='mps'),past_key_values=clone(source),use_cache=True,logits_to_keep=1)
   require([check(logits,native_suffix.logits[0,-1])]+[check(a,b) for p,q in zip(pairs(cache),pairs(native_suffix.past_key_values)) for a,b in zip(p,q)],'naive-endpoint')
   del cache,logits,native_suffix
  else:del native
 else:
  reference=torch.load(refpath,weights_only=True).to('mps');reference_s=0.
 finite(reference);gc.collect();torch.mps.empty_cache()
 save(OUT/'item-status.json',{'item':key,'composition_s':compose_s,'reference_s':reference_s,'preflight':preflight})
 eos=MODEL.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
 for label,mode,ratio in methods:
  path=result_path(key,label)
  if path.exists():
   assert json.loads(path.read_text())['protocol_sha256']==phash;continue
  PARTIAL.clear();ACTIVE={'item':key,'method':label,'phase':'execution'};guard();meters();sync(MODEL);t=time.monotonic()
  cache,logits,work=execute(MODEL,ids,d,source,mode,ratio);sync(MODEL);prefill_s=time.monotonic()-t
  finite(logits);finite_cache(cache)
  assert all(a.data_ptr()!=b.data_ptr() for p,q in zip(pairs(cache),pairs(source)) for a,b in zip(p,q))
  endpoint=check(reference,logits) if label=='full' else None
  if endpoint and not endpoint['passed']:raise RuntimeError('per-item full endpoint failed')
  stats=divergence(reference,logits);meters();t=time.monotonic()
  gen,ended=decode(cache,logits,protocol['max_new_tokens'],eos);sync(MODEL);generation_s=time.monotonic()-t
  text=tok.decode(gen,skip_special_tokens=True)
  result={'status':'complete','protocol_sha256':phash,'id':key,'method':label,'answers':item['answers'],'input_ids_sha256':__import__('hashlib').sha256(json.dumps(ids).encode()).hexdigest(),'answer':text,'generated_ids':gen,'ended_eos':ended,'truncated':not ended,**score(text,item['answers'],tok),**stats,'endpoint_check':endpoint,'prefill_s_including_cache_copy_and_position_logging':prefill_s,'generation_s_including_branch_clone':generation_s,'work':work,'qkv_rows':sum(w['qkv_rows'] for w in work),'attention_mlp_rows':sum(w['attention_mlp_rows'] for w in work),'full_token_rows':len(ids)*len(MODEL.model.layers),'memory':meters()}
  if label=='full':full=result
  else:full=json.loads(result_path(key,'full').read_text())
  result['normalized_answer_agrees_full']=norm(text)==norm(full['answer']);result['generated_ids_agree_full']=gen==full['generated_ids']
  save(path,result) # Commit each completed item-method before any next method.
  append(OUT/'progress.jsonl',{'elapsed_s':time.time()-START,'id':key,'method':label,'completed':len(list((OUT/'results').glob('*.json')))})
  print('completed',key,label,'EM',result['reference_em'],'F1',result['reference_qwen_token_f1'],flush=True)
  del cache,logits;gc.collect();torch.mps.empty_cache()
  if cache_digest(source)!=source_hash:raise RuntimeError('source cache mutated')
  guard()
 del source,reference;gc.collect();torch.mps.empty_cache();meters()


def main():
 global MODEL,START,PARENT,DEADLINE,STOP
 PARENT=os.getppid();state=json.loads((OUT/'run-state.json').read_text());START=state['start_epoch'];DEADLINE=START+21540
 signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True))
 protocol=json.loads(FROZEN.read_text());phash=sha(FROZEN)
 assert phash==state['protocol_sha256']
 assert sha(ROOT/'experiments/cacheblend/core.py')==protocol['implementation_sha256']
 for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest
 try:
  MODEL,tok=load()
  for layer in MODEL.model.layers:layer.input_layernorm.register_forward_pre_hook(guard)
  meters();guard();items=protocol['items'];pre=next(r for r in items if r['id']==protocol['preflight_item'])
  if not (OUT/'preflight.json').exists():
   begin=time.time();item_run(pre,True,protocol,phash,tok)
   save(OUT/'preflight.json',{'status':'complete','item':pre['id'],'item_wall_s_including_validation':time.time()-begin,'elapsed_s_including_load':time.time()-START,'memory':meters()})
  while not (OUT/'decision.json').exists():guard();time.sleep(1)
  decision=json.loads((OUT/'decision.json').read_text())
  if not decision['continue']:raise Stop('Preflight did not fit comfortably: '+str(decision['reasons']))
  for item in items:
   if item['id']!=pre['id']:item_run(item,False,protocol,phash,tok)
  save(OUT/'worker-status.json',{'status':'complete','elapsed_s':time.time()-START,'completed_results':len(list((OUT/'results').glob('*.json'))),'memory':meters()})
 except BaseException as e:
  save(OUT/'worker-status.json',{'status':'stopped' if isinstance(e,Stop) else 'failed','active':ACTIVE,'partial_generated_ids':PARTIAL,'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.time()-START,'completed_results':len(list((OUT/'results').glob('*.json')))})
  print(traceback.format_exc(),flush=True);sys.exit(2)
if __name__=='__main__':main()
