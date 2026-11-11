"""Fixed scoring, atomic checkpoints and paths; no model execution."""
import collections,hashlib,json,os,re,string
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/cacheblend_overnight'
FROZEN=HERE/'frozen.json'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def append(path,obj):
 with Path(path).open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def norm(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split())
def overlap(p,g):
 if not p or not g:return float(p==g)
 return 2*sum((collections.Counter(p)&collections.Counter(g)).values())/(len(p)+len(g))
def parse(s):
 s=s.lstrip('\n').split('\n')[0]
 if s.startswith(('Yes','yes')):return 'Yes'
 if s.split() and s.split()[0].startswith(('No','no')):return 'No'
 return s

def score(answer,golds,tok):
 parsed=parse(answer);p=norm(parsed);raw=norm(answer)
 pids=tok.encode(p,add_special_tokens=False)
 return {'existing_em':max(float(raw==norm(g)) for g in golds),
  'existing_word_f1':max(overlap(raw.split(),norm(g).split()) for g in golds),
  'reference_parsed_answer':parsed,
  'reference_em':max(float(p==norm(g)) for g in golds),
  'reference_qwen_token_f1':max(overlap(pids,tok.encode(norm(g),add_special_tokens=False)) for g in golds),
  'legacy_drop_first_token_f1_DIAGNOSTIC_ONLY':max(overlap(pids[1:],tok.encode(norm(g),add_special_tokens=False)[1:]) for g in golds)}
def result_path(item,method):return OUT/'results'/f"{item.replace(':','_')}__{method}.json"
