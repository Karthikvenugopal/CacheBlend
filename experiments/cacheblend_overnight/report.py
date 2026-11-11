"""Report this single pass, including incomplete/stopped runs. No model execution."""
import json,random,statistics
from datetime import datetime,timezone
from common import ROOT,HERE,OUT,FROZEN,sha,save,norm
METRICS=['existing_em','existing_word_f1','reference_em','reference_qwen_token_f1','normalized_answer_agrees_full','generated_ids_agree_full','kl_full_to_method','total_variation','top1_agreement']

def interval(values):
 rng=random.Random(17)
 if not values:return None
 means=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(10000))
 return {'n':len(values),'mean':statistics.mean(values),'paired_bootstrap_percentile_95':[means[249],means[9749]],'interpretation':'Exploratory resampling of these fixed, manually selected items, not population coverage or evidence of equivalence.'}

def main():
 p=json.loads(FROZEN.read_text());ph=sha(FROZEN);methods=[m[0] for m in p['methods']]
 files=sorted((OUT/'results').glob('*.json'));rows=[json.loads(f.read_text()) for f in files]
 lookup={(r['id'],r['method']):r for r in rows};assert len(lookup)==len(rows)
 for r in rows:
  assert r['protocol_sha256']==ph and r['status']=='complete'
  item=next(i for i in p['items'] if i['id']==r['id']);d=item['document_length'];n=len(item['input_ids'])
  assert __import__('hashlib').sha256(json.dumps(item['input_ids']).encode()).hexdigest()==r['input_ids_sha256']
  assert all(w['positions'][-(n-d):]==list(range(d,n)) for w in r['work'])
  if r['method']=='full':assert r['endpoint_check']['passed']
 summary={}
 for method in methods:
  rr=[r for r in rows if r['method']==method]
  if not rr:continue
  summary[method]={'completed':len(rr),'truncations':sum(r['truncated'] for r in rr),**{k:statistics.mean(float(r[k]) for r in rr) for k in METRICS},'qkv_rows':sum(r['qkv_rows'] for r in rr),'attention_mlp_rows':sum(r['attention_mlp_rows'] for r in rr),'qkv_work_proxy_fraction':statistics.mean(r['qkv_rows']/r['full_token_rows'] for r in rr),'attention_mlp_work_proxy_fraction':statistics.mean(r['attention_mlp_rows']/r['full_token_rows'] for r in rr),'mean_prefill_s':statistics.mean(r['prefill_s_including_cache_copy_and_position_logging'] for r in rr),'mean_generation_s':statistics.mean(r['generation_s_including_branch_clone'] for r in rr)}
 pairs={}
 for method in methods[1:]:
  for reference in ['full','naive']:
   for metric in ['existing_em','existing_word_f1','reference_em','reference_qwen_token_f1']:
    values=[lookup[(i['id'],method)][metric]-lookup[(i['id'],reference)][metric] for i in p['items'] if (i['id'],method) in lookup and (i['id'],reference) in lookup]
    pairs[f'{method}-minus-{reference}/{metric}']=interval(values)
 status=json.loads((OUT/'status.json').read_text())
 missing=[{'id':i['id'],'method':m} for i in p['items'] for m in methods if (i['id'],m) not in lookup]
 out={'protocol_sha256':ph,'status':status,'completed_item_methods':len(rows),'expected_item_methods':5*len(p['items']),'missing':missing,'summary':summary,'paired_differences':pairs,'truncations':[{'id':r['id'],'method':r['method'],'ids':r['generated_ids']} for r in rows if r['truncated']]}
 save(OUT/'summary.json',out)
 lines=['# Bounded evidence-controlled CacheBlend diagnostic','',f"Status: **{status['status']}**. Completed {len(rows)}/{5*len(p['items'])} item-method results across {len(p['items'])} fixed items. All completed results are included; missing results are listed in summary.json. No silent exclusions or retries.",'',
 'This is a manually annotated evidence-controlled diagnostic, not a normal retrieval benchmark. Cached release schemas contain no official supporting evidence, retrieval scores, ranks, or original passage IDs. Supplied windows overlap and concatenate unrelated text, and sometimes end mid-sentence or contain wordpiece markers. Their actual ranked-retrieval provenance cannot be established from the available assets. Two complete cached windows are retained per item in supplied order, without new truncation. This preserves annotated reasoning claims, not necessarily original source articles. Full reasoning-chain quotes and character offsets, every candidate exclusion, source hashes and exact assembled IDs are in frozen.json. Selection was disjoint from the earlier pilot and did not use model success.','',
 'The pinned single-check-layer V-HKVD implementation is unchanged. Zero selection still refreshes early layers; it is distinct from naive. Every question token is fresh on every layer. Native-coefficient RoPE correction, model ownership and independent caches are retained. Native full is an agreement reference, not a gold oracle. Prior MPS deep-prefix raw-tensor flags remain diagnostics; the same numerical tolerance is used, without relaxation. Numerical validation failures stop the run.','',
 'Reference scoring uses the release’s first-line and yes/no parser, normalization and token-overlap F1, adapted to Qwen by retaining the first real token because this tokenizer adds no BOS. Reference EM is an added normalized parsed-answer metric; the release supplies F1 rather than EM. Exact legacy Qwen `[1:]` F1 is saved only as a bug diagnostic. The parser itself can mistake words beginning with No/Yes for binary answers, and is preserved explicitly. Empty outputs are safely scored rather than crashing. Existing metrics use raw whole-output normalized EM and word F1. None is semantic adjudication; elaboration, aliases and transliterations can be penalized. The reference’s Mistral tokenizer/precision and serving environment are not reproduced.','',
 '| Method | n | Existing EM | Existing word F1 | Ref EM | Ref Qwen-token F1 | Answer agreement | Initial KL | TV | QKV proxy | Attn/MLP proxy | Truncated |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for m,s in summary.items():lines.append(f"| {m} | {s['completed']} | {s['existing_em']:.3f} | {s['existing_word_f1']:.3f} | {s['reference_em']:.3f} | {s['reference_qwen_token_f1']:.3f} | {s['normalized_answer_agrees_full']:.3f} | {s['kl_full_to_method']:.3f} | {s['total_variation']:.3f} | {s['qkv_work_proxy_fraction']:.3f} | {s['attention_mlp_work_proxy_fraction']:.3f} | {s['truncations']} |")
 lines += ['', 'QKV and attention/MLP row counts are work proxies, not FLOPs or speedups. Exact counts are saved per layer and method. Initial KL/TV use the same native full prompt distribution. Agreement includes normalized answers in the table and exact generated IDs in summary.json.','',
 'All paired differences and 10,000-resample percentile intervals are in summary.json, with deterministic bootstrap seed17. Intervals describe this small fixed diagnostic only. Close scores or an interval spanning zero do not establish equivalent quality. The sample is manually selected and nonrepresentative.','',
 f"Observed total elapsed time: {status.get('elapsed_s',0):.1f}s. Preflight includes model load, composition, reference checks and all five methods; its completed results are reused. Prefill timers include cache copies and position logging. Generation timers include branch cloning and autoregressive decode. Independent chunk prefill, native reference and numerical checks are separate evaluation overhead and do not feed selection. Cold/warm timing order, OS load and fanless sustained operation prevent serving-speedup claims. Native memory samples, RSS high-water measurements, MPS allocated/driver/recommended memory and swap counters are preserved in memory.jsonl and worker-memory.jsonl. These metrics overlap and must not be summed as physical memory.",'']
 if (OUT/'decision.json').exists():lines += ['Preflight decision: `'+json.dumps(json.loads((OUT/'decision.json').read_text()))+'`','']
 failures=[f for f in ['worker-status.json','supervisor-error.json','stop-request.json'] if (OUT/f).exists()]
 lines += ['Status/failure evidence: '+', '.join(failures)+'. Completed checkpoints and partial token IDs on failure are preserved. Truncations and missing results are explicitly listed in summary.json.','',
 'Recommendation: the mechanism can be documented as implemented and checked within its stated numerical scope. This diagnostic alone is insufficient to declare a completed general mechanism-and-quality reproduction: official evidence annotations and retriever provenance are absent, the sample is small and manually controlled, and the original paper’s workload/model/serving setup is not reproduced. Report the measured quality differences with those limits; do not infer equivalence. If stopped before completion, the status and preflight diagnosis are the immediate additional limitation. No further experiments are queued.','']
 (OUT/'report.md').write_text('\n'.join(lines))
 print('report:',OUT/'report.md')
if __name__=='__main__':main()
