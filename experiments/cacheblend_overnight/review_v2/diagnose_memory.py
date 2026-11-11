"""Static accounting from retained logs/config and safetensors headers. No model load."""
import hashlib,json,math,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/cacheblend_overnight/review_v2'
G=1024**3

def main():
 config=json.loads((ROOT/'.models/qwen3-1.7b/config.json').read_text())
 params=0;weights={};checkpoint_dtypes={}
 for path in sorted((ROOT/'.models/qwen3-1.7b').glob('*.safetensors')):
  with path.open('rb') as f:
   length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length))
  for name,entry in header.items():
   if name=='__metadata__':continue
   n=math.prod(entry['shape']);params+=n
   checkpoint_dtypes[entry['dtype']]=checkpoint_dtypes.get(entry['dtype'],0)+n
   if name in ['model.embed_tokens.weight','lm_head.weight']:
    weights[name]={'shape':entry['shape'],'float32_bytes':n*4,'float32_GiB':n*4/G}
 frozen=json.loads((ROOT/'experiments/cacheblend_overnight/frozen.json').read_text())
 item=next(r for r in frozen['items'] if r['id']==frozen['preflight_item'])
 n,d=len(item['input_ids']),item['document_length']
 per=2*config['num_hidden_layers']*config['num_key_value_heads']*config['head_dim']*4
 events=[json.loads(l) for l in (ROOT/'artifacts/cacheblend_overnight/memory.jsonl').read_text().splitlines()]
 wm=[json.loads(l) for l in (ROOT/'artifacts/cacheblend_overnight/worker-memory.jsonl').read_text().splitlines()]
 origin=events[0]['epoch'];timeline=[]
 for e in events:
  timeline.append({'elapsed_s':e['epoch']-origin,'kernel_pressure':e['pressure_level'],'os_free_percentage':e['os_free_percentage'],'new_swapouts_bytes':(e['swapouts_pages']-events[0]['swapouts_pages'])*e['page_size'],'new_swapins_bytes':(e['swapins_pages']-events[0]['swapins_pages'])*e['page_size'],'sampled_process_rss_bytes':e['process_rss_bytes']})
 result={'analysis_type':'static log/source/header analysis; no inference or allocation experiment','parameter_elements_including_both_checkpoint_embedding_and_head':params,'checkpoint_elements_by_dtype':checkpoint_dtypes,'float32_parameter_payload_bytes':params*4,'float32_parameter_payload_GiB':params*4/G,'large_parameters':weights,'sequence_tokens':n,'document_tokens':d,'fresh_tokens':n-d,'float32_KV_bytes_per_token':per,'full_KV_bytes':n*per,'document_KV_bytes':d*per,'three_simultaneously_live_cache_payload_bytes':(2*n+d)*per,'128_generated_tokens_extra_KV_bytes':128*per,'one_full_attention_matrix_bytes':config['num_attention_heads']*n*n*4,'worker_memory_samples':wm,'os_timeline':timeline,'attribution_limits':['Counters are system-wide; baseline swap usage was already nonzero. Cumulative swap I/O is churn, not simultaneously resident bytes.','Pressure reached critical at10s, before the first post-load sample at18.3s. The evidence does not isolate checkpoint reading, asynchronous CPU conversion, device transfer or unrelated OS load.','The worker retains source and native caches while creating a third candidate cache for validation. It releases candidate but retains native before the second endpoint. Full endpoint passed; reference ratio1 endpoint was interrupted.','Source label ACTIVE.phase remained composition while validation was executing. Traceback and completed gate files, rather than this stale label, locate the stop.','Observed driver minus tensor allocations includes allocator/backend resources, not a measured leak. Sparse samples cannot identify every temporary or true MPS peak.','RSS and MPS counters overlap and must not be added. There was no recorded OOM or nonfinite error.'], 'proposed_plan_status':'proposal only; no loader, executor, precision or guard has been changed; no run authorized'}
 (OUT/'memory-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 print('float32 weights GiB',params*4/G,'KV MiB',n*per/1024**2,d*per/1024**2,'three cache MiB',(2*n+d)*per/1024**2,'attention MiB',config['num_attention_heads']*n*n*4/1024**2)
if __name__=='__main__':main()
