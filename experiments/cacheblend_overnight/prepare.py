"""Manual evidence annotations and a frozen, disjoint diagnostic. No inference."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/cacheblend_overnight'
HERE=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def flatten(x): return [x] if isinstance(x,str) else [v for y in x for v in flatten(y)]

# Human evidence-chain annotations, authorized explicitly by the user.
# Each quoted claim must occur verbatim in the stated cached text window.
ANNOTATIONS={
 'musique:022':[(0,'dimuthu bandara abayakoon ( born september 6, 1971 ) is a sri lankan politician, belonging to the janatha vimukthi peramuna.'),(5,'known as rohana wijeweera, was a sri lankan marxist politician, revolutionary and the founding leader of the janatha vimukthi peramuna.')],
 'musique:027':[(3,'the rabat ajax football ground is the training ground of maltese football team rabat ajax f. c.'),(6,'balzan and rabat ajax in the maltese premier league')],
 'musique:030':[(0,'the united states naval institute ( usni ), based in annapolis, maryland'),(0,'in the shadow of greatness is a 2012 book written by 33 members of the united states naval academy class of 2002 and published by the united states naval institute.')],
 'musique:033':[(0,'snappy tomato pizza is a pizza chain that started in fort mitchell, kentucky'),(4,'fort mitchell is a home rule - class city in kenton county, kentucky')],
 'musique:039':[(8,"the party's founder and longest - serving leader robert menzies"),(7,"lyons'death in 1939 saw robert menzies assume the prime ministership")],
 'wikimqa:025':[(0,'guy arvely dolsin ( born january 15, 1957 in toamasina )'),(3,'altug celikbilek ( turkish pronunciation : [ ˈaɫtuː ˈtʃelicbilec ] ; born 7 september 1996 )')],
 'wikimqa:026':[(0,'jugband blues " is a song by the english psychedelic rock band pink floyd, released on their second album, a saucerful of secrets, in 1968.'),(1,'it was released on december 1, 2014.')],
 'wikimqa:031':[(1,'one of his sons, mark getty, co - founded the visual media company getty images.'),(1,'he was the third of five sons born to j. paul getty ( 1892 – 1976 )')],
 'wikimqa:032':[(9,'" fantasy " is a song by the english singer - songwriter george michael.'),(0,'michael died of heart disease on christmas day in 2016, at his home in goring - on - thames, oxfordshire.')],
 'wikimqa:035':[(1,'above rubies is a 1932 british comedy film'),(1,'is a chinese animated film released june 19, 2009')]
}
EXCLUSIONS={
 'musique':{20:'Juba to Central Equatoria present; shared swamp does not explicitly establish the requested state-border relation.',21:'Spouse is Charles Tupper; Amherst birth sentence concerns an unnamed son of Robert Barry Dickey, not an anchored Charles Tupper birthplace.',23:'Bancroft and the required county-border chain absent from inspected windows.',24:'Greenfield birthplace present; Greenfield to Hancock County link not found.',25:'Ambiguous sibling question; text supports Lorenzo as well as gold Giuliano, but gold accepts only Giuliano.',26:'School to New Richland present; county relation missing.',28:'Film cast present; spouse relation missing.',29:'Station to Park Rapids present; Hubbard County reference concerns Emmaville, not a validated Park Rapids county relation.',31:'Several band members explicitly supported; single gold accepts Erick Sermon only. Excluded for non-unique answer rubric.',32:'Bansaan to Talibon present; needed municipality-island relation not validated.',34:'Required member-birthplace chain not validated.',35:'Required birthplace-district chain not validated.',36:'Required birthplace-county chain not validated.',37:'Ruth Gordon screenplay attribution present; spouse relation not found.',38:'Modiano award present; English Missing Person authorship link not explicitly validated.'},
 'wikimqa':{20:'Father Fredrik Monsen identified; his death place missing.',21:'Performer-to-song and graduation chain not found.',22:'Film-director links present; both director birth dates missing.',23:'Castro birth date present; Niksic birth date missing.',24:'Both director/death-date chains not validated.',27:'Open Mobile serves Puerto Rico; same-country comparison with Primestar lacks an explicit, unambiguous country chain.',28:'Father Rathold identified; grandfather relation missing.',29:'Father Yakshya identified; his birthplace missing.',30:'Fourth-earl subject anchoring not validated within two complete windows; do not infer identity from an unanchored son/pronoun.',33:'Godsend director-nationality chain missing.',34:'Film-director links present; required birth-date comparison not validated.'}}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 tok=AutoTokenizer.from_pretrained(ROOT/'.models/qwen3-1.7b',local_files_only=True,trust_remote_code=False)
 marker='CACHEBLEND_EVIDENCE_MARKER'
 before,after=tok.apply_chat_template([{'role':'user','content':marker}],tokenize=False,add_generation_prompt=True,enable_thinking=False).split(marker)
 enc=lambda s:tok.encode(s,add_special_tokens=False)
 prefix='You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n'
 query='\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:'
 candidates=[]; items=[]; sources={}; schemas={}
 for name in ('musique','wikimqa'):
  path=ROOT/f'experiments/cacheblend/reference/inputs__{name}_s.json'
  data=json.loads(path.read_text()); sources[str(path.relative_to(ROOT))]=sha(path)
  schemas[name]={'records':len(data),'item_keys':sorted(set(k for r in data for k in r)),'context_keys':sorted(set(k for r in data for c in r['ctxs'] for k in c)), 'empty_titles':sum(not c['title'] for r in data for c in r['ctxs']),'official_support_annotations':False,'retrieval_ranking_provenance':'not identifiable from cached schema; supplied order only'}
  for idx in range(20,40):
   key=f'{name}:{idx:03}'; raw=data[idx]
   if key not in ANNOTATIONS:
    reason=EXCLUSIONS[name].get(idx,'Per-workload five-item cap reached; no further manual qualification or model evaluation.')
    candidates.append({'id':key,'included':False,'reason':reason});continue
   annotations=[]
   for ci,quote in ANNOTATIONS[key]:
    text=raw['ctxs'][ci]['text']; start=text.index(quote)
    annotations.append({'source_context_index':ci,'quote':quote,'character_span':[start,start+len(quote)]})
   required=sorted(set(a['source_context_index'] for a in annotations))
   chosen=list(required)
   if len(chosen)==1:
    extras=[c for c in range(10) if c not in chosen]
    # Additional window, not asserted to be irrelevant or free of support.
    chosen.append(extras[0] if key!='musique:030' else 2)
   chosen.sort(); assert len(chosen)==2
   q=raw['question']; q=q if q.endswith('?') else q+'?'; q=q[0].lower()+q[1:]
   chunks=[enc(before+prefix)]+[enc(raw['ctxs'][ci]['title']+'\n\n'+raw['ctxs'][ci]['text']+'\n\n') for ci in chosen]
   fresh=enc(query+q+'\nAnswer:')+enc(after); ids=sum(chunks,[])+fresh
   if len(ids)>1536:
    candidates.append({'id':key,'included':False,'reason':'Two intact windows exceed fixed 1536-token budget. No truncation or replacement.','tokens':len(ids),'annotations':annotations});continue
   item={'id':key,'source_index':idx,'workload':name,'raw':raw,'answers':flatten(raw['answers']),'manual_annotations':annotations,'required_context_indices':required,'retained_context_indices':chosen,'omitted_context_indices':[c for c in range(10) if c not in chosen],'chunks':chunks,'fresh_ids':fresh,'input_ids':ids,'document_length':sum(map(len,chunks)),'decoded_input':tok.decode(ids),'tokens':len(ids),'transformation':'two whole cached text windows, original order; no text/token truncation; reference wording in one native nonthinking Qwen wrapper; question normalized as release'}
   assert ids.count(tok.convert_tokens_to_ids('<|im_start|>'))==2
   assert ids.count(tok.convert_tokens_to_ids('<|im_end|>'))==1
   items.append(item); candidates.append({'id':key,'included':True,'tokens':len(ids),'required':required,'retained':chosen,'annotations':annotations})
 assert items
 preflight=max(items,key=lambda r:(r['tokens'],r['id']))['id']
 protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'label':'MANUALLY ANNOTATED EVIDENCE-CONTROLLED DIAGNOSTIC; not a retrieval benchmark','selection':'fixed pool indices 20..39 of each cached dataset, disjoint from prior pilot; up to five qualified manually annotated items per workload; no model outputs inspected for these IDs; no replacement after token exclusions','configuration':'exactly two intact cached text windows plus instruction prefix, <=1536 assembled prompt tokens; two support windows or one support plus fixed additional window; original window ordering','max_new_tokens':128,'thinking':False,'generation':'deterministic unrestricted greedy, native EOS, seed 17, no penalties','methods':[['full','full',1.0],['naive','naive',0.0],['zero','reference',0.0],['hkvd_16','reference',0.16],['hkvd_32','reference',0.32]],'model_revision':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e','device':'mps','dtype':'float32','attention':'eager','six_hour_limit_seconds':21600,'soft_stop_seconds':21540,'severe_pressure':'three successive 10s samples with kernel critical>=4 or memory_pressure free<=10%, or >256MiB swapouts in rolling60s with warning>=2/free<=20%','preflight_comfort':'max process ru_maxrss <=12GiB; max observed MPS driver <=95% recommended_max_memory; free percentage >=15%; no kernel warning>=2; preflight swapout increase <=128MiB; estimated total <=5.5h','scoring':'existing raw normalized word EM/F1; reference first-line/yes-no parse + normalized EM and Qwen token-overlap F1 without erroneous BOS stripping; exact legacy Qwen [1:] token F1 retained only as bug diagnostic','source_sha256':sources,'implementation_sha256':sha(ROOT/'experiments/cacheblend/core.py'),'preflight_item':preflight,'items':items,'candidates':candidates,'schema_audit':schemas}
 path=HERE/'frozen.json'
 assert not path.exists(),'Freeze already exists; do not overwrite'
 path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'items':len(items),'preflight':preflight,'tokens':[r['tokens'] for r in items],'sha256':sha(path)},indent=2))
if __name__=='__main__':main()
