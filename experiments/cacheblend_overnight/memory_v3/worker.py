"""Single frozen-v2 longest-item preflight. No batch path exists."""
import gc,json,os,resource,signal,sys,time,traceback,threading
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import ROOT,save,append,sha,score,norm
from core import compose,execute,pairs,clone,check,divergence,sync
from loader import load
OUT=ROOT/'artifacts/cacheblend_overnight/memory_v3'
FROZEN=HERE.parent/'review_v2/frozen-v2.json'
MODEL=None;ACTIVE={};PARTIAL=[];STOP=False;START=time.time();PARENT=os.getppid()
def guard(*_):
 if STOP or (OUT/'stop-request.json').exists() or os.getppid()!=PARENT or time.time()-START>=21540:raise RuntimeError('Preflight stop requested/deadline/parent exit')
def mark(phase):
 ACTIVE['phase']=phase;guard()
 z={'elapsed_s':time.time()-START,'phase':phase,'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 if torch.backends.mps.is_available():
  z.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory(),mps_recommended_bytes=torch.mps.recommended_max_memory())
 append(OUT/'worker-memory.jsonl',z)
def release():
 gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
def finite(t):
 assert bool(torch.isfinite(t).all()),'nonfinite tensor'
def finite_cache(c):
 for p in pairs(c):
  for t in p:finite(t)
def digest(c):
 import hashlib
 h=hashlib.sha256()
 for p in pairs(c):
  for t in p:h.update(t.cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def require(label,checks):
 save(OUT/f'validation-{label}.json',checks)
 assert all(x['passed'] for x in checks),'numerical check failed: '+label
def dump(c,label):
 directory=OUT/'references'/label;directory.mkdir()
 for i,p in enumerate(pairs(c)):
  guard();torch.save(tuple(t.cpu() for t in p),directory/f'{i}.pt')
def compare(c,label,reference,logits):
 checks=[check(reference,logits)]
 for i,p in enumerate(pairs(c)):
  guard();expected=torch.load(OUT/'references'/label/f'{i}.pt',weights_only=True)
  checks.extend(check(a,b) for a,b in zip(expected,p));del expected
 return checks
@torch.inference_mode()
def run():
 global MODEL,PARTIAL
 protocol=json.loads(FROZEN.read_text());phash=sha(FROZEN)
 for name,h in protocol['source_sha256'].items():assert sha(ROOT/name)==h
 item=max(protocol['items'],key=lambda x:len(x['input_ids']))
 assert item['id']==protocol['preflight_item']
 save(OUT/'scope.json',{'frozen_sha256':phash,'item':item['id'],'tokens':len(item['input_ids']),'methods':protocol['methods'],'authorization':'one preflight only','original_core_sha256':sha(ROOT/'experiments/cacheblend/core.py')})
 ACTIVE['item']=item['id'];MODEL,tok,weights=load(mark,guard)
 save(OUT/'loader-equality.json',weights)
 for layer in MODEL.model.layers:layer.input_layernorm.register_forward_pre_hook(guard)
 ids=item['input_ids'];d=item['document_length'];first=len(item['chunks'][0])
 mark('native-reference')
 native=MODEL(torch.tensor([ids],device='mps'),use_cache=True,logits_to_keep=torch.tensor([first-1,len(ids)-1],device='mps'))
 finite(native.logits);finite_cache(native.past_key_values)
 reference=native.logits[0,-1].clone();prefix=native.logits[0,0].cpu()
 dump(native.past_key_values,'native')
 token=int(reference.argmax())
 # Replay owns native cache after its reference has been serialized.
 replay=MODEL(torch.tensor([[token]],device='mps'),past_key_values=native.past_key_values,use_cache=True,logits_to_keep=1)
 replay_logits=replay.logits[0,-1].cpu();del native,replay;release();mark('native-direct')
 direct=MODEL(torch.tensor([ids+[token]],device='mps'),use_cache=True,logits_to_keep=1)
 require('native-replay',[check(replay_logits,direct.logits[0,-1])]);del direct,replay_logits;release()
 mark('custom-full-no-source')
 cache,logits,_=execute(MODEL,ids,d,None,'full',1.)
 require('full-endpoint',compare(cache,'native',reference,logits));del cache,logits;release()
 mark('composition');source=compose(MODEL,item['chunks'],validate=True);finite_cache(source)
 source_hash=digest(source);require('position-correction',source._cacheblend_rotation_checks)
 layer0=torch.load(OUT/'references/native/0.pt',weights_only=True)
 require('prefix-logits-layer0',[check(prefix,source._cacheblend_first_logits)]+[check(a[:,:,:d],b) for a,b in zip(layer0,pairs(source)[0])]);del layer0
 diagnostics=[]
 for i,p in enumerate(pairs(source)):
  expected=torch.load(OUT/f'references/native/{i}.pt',weights_only=True)
  diagnostics.extend(check(a[:,:,:first],b[:,:,:first]) for a,b in zip(expected,p));del expected
 save(OUT/'native-prefix-tensor-diagnostics.json',diagnostics)
 mark('reference-full-selection')
 cache,logits,_=execute(MODEL,ids,d,source,'reference',1.)
 require('reference-endpoint',compare(cache,'native',reference,logits));del cache,logits;release()
 mark('native-naive-reference')
 suffix=MODEL(torch.tensor([ids[d:]],device='mps'),past_key_values=clone(source),use_cache=True,logits_to_keep=1)
 suffix_logits=suffix.logits[0,-1].cpu();dump(suffix.past_key_values,'naive');del suffix;release()
 cache,logits,_=execute(MODEL,ids,d,source,'naive')
 require('naive-endpoint',compare(cache,'naive',suffix_logits,logits));del cache,logits,suffix_logits;release()
 assert digest(source)==source_hash
 eos=MODEL.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
 full=None
 for label,mode,ratio in protocol['methods']:
  ACTIVE['method']=label;PARTIAL=[];mark('method-prefill');t=time.monotonic()
  cache,logits,work=execute(MODEL,ids,d,source,mode,ratio);sync(MODEL);prefill=time.monotonic()-t
  finite(logits);finite_cache(cache)
  assert all(a.untyped_storage().data_ptr()!=b.untyped_storage().data_ptr() for p,q in zip(pairs(cache),pairs(source)) for a,b in zip(p,q))
  if label=='full':require('method-full',[check(reference,logits)])
  stats=divergence(reference,logits);mark('method-decode');t=time.monotonic()
  for step in range(protocol['max_new_tokens']):
   guard();finite(logits);token=int(logits.argmax());PARTIAL.append(token)
   if token in eos:break
   out=MODEL(torch.tensor([[token]],device='mps'),past_key_values=cache,use_cache=True,logits_to_keep=1)
   logits=out.logits[0,-1];cache=out.past_key_values;del out
   mark('decode:'+str(step))
  finite_cache(cache);sync(MODEL);generation=time.monotonic()-t
  text=tok.decode(PARTIAL,skip_special_tokens=True)
  result={'id':item['id'],'method':label,'protocol_sha256':phash,'generated_ids':list(PARTIAL),'answer':text,'ended_eos':PARTIAL[-1] in eos,'truncated':PARTIAL[-1] not in eos,'work':work,'prefill_s':prefill,'generation_s':generation,**stats,**score(text,item['answers'],tok)}
  if full is None:full=result
  result.update(generated_ids_agree_full=result['generated_ids']==full['generated_ids'],normalized_answer_agrees_full=norm(text)==norm(full['answer']))
  del cache,logits;release()
  assert digest(source)==source_hash,'source mutated during method/decode'
  result['source_unchanged']=True;result['independent_storage']=True
  save(OUT/'results'/f'{label}.json',result);mark('method-released');print('completed',label,flush=True)
 del source,reference;release()
 save(OUT/'preflight.json',{'status':'complete','item':item['id'],'methods_completed':5,'elapsed_s':time.time()-START})
def sampler(done):
 while not done.wait(1):
  try:
   z={'elapsed_s':time.time()-START,'phase':ACTIVE.get('phase'),'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_allocated_bytes':torch.mps.current_allocated_memory(),'mps_driver_bytes':torch.mps.driver_allocated_memory(),'mps_recommended_bytes':torch.mps.recommended_max_memory()}
   append(OUT/'sampled-worker-memory.jsonl',z)
  except Exception as e:
   save(OUT/'sampler-error.json',{'error':repr(e)});globals()['STOP']=True;return

def main():
 global MODEL
 signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True))
 done=threading.Event();thread=threading.Thread(target=sampler,args=(done,),daemon=True);thread.start()
 try:
  run();save(OUT/'worker-status.json',{'status':'complete','elapsed_s':time.time()-START})
 except BaseException as e:
  save(OUT/'worker-status.json',{'status':'failed_or_stopped','error':repr(e),'traceback':traceback.format_exc(),'active':ACTIVE,'partial_generated_ids':PARTIAL,'elapsed_s':time.time()-START})
  traceback.print_exc();sys.exit(2)
 finally:
  done.set();thread.join(timeout=3)
  MODEL=None;gc.collect()
  if torch.backends.mps.is_available():torch.mps.empty_cache()
if __name__=='__main__':main()
