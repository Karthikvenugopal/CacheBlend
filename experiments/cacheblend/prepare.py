"""Freeze records and token assemblies before model inference; never filter by answers."""
import collections
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PIN = '55ad02675939f783a38d579393527d218a7fd581'
CONFIGS = [{'name':'4x128', 'chunks':4, 'tokens':128}, {'name':'8x128','chunks':8,'tokens':128}]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(x):
    if isinstance(x, str):
        return [x]
    return [v for y in x for v in flatten(y)]


def main():
    tok = AutoTokenizer.from_pretrained(ROOT/'.models/qwen3-1.7b', local_files_only=True)
    records = []
    for name in ('musique', 'wikimqa'):
        path = HERE/'reference'/f'inputs__{name}_s.json'
        data = json.loads(path.read_text())
        for index, item in enumerate(data[:20]):
            records.append({'id':f'{name}:{index:03}', 'source_index':index,
                            'source_sha256':sha(path), 'workload':name,
                            'raw':item, 'answers':flatten(item['answers'])})
    runs = []
    instruction = 'Read the passages and answer the question directly in at most five words.\nPassages:\n'
    prefix = tok.apply_chat_template([{'role':'user','content':instruction}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    # Assemble one native chat wrapper. Split its user content at a unique marker;
    # tokenize all segments with add_special_tokens=False. No decode/re-encode path.
    marker = 'CACHEBLEND_CONTENT_MARKER'
    chat = tok.apply_chat_template([{'role':'user','content':marker}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    before, after = chat.split(marker)
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    for record in records:
        ex = record['raw']
        for config in CONFIGS:
            chunks = [enc(before+instruction)]
            evidence = []
            for ci, ctx in enumerate(ex['ctxs'][:config['chunks']]):
                original = enc(ctx['title']+'\n\n'+ctx['text'])
                body = original[:config['tokens']]
                chunk = enc(f'\nPassage {ci+1}:\n')+body+enc('\n\n')
                chunks.append(chunk)
                evidence.append({'source_context_index':ci,'original_tokens':len(original),
                                 'retained_tokens':len(body),'truncated':len(body)<len(original)})
            fresh = enc('\nQuestion: '+ex['question']+'\nAnswer:')+enc(after)
            ids = sum(chunks, [])+fresh
            runs.append({'id':record['id'], 'config':config['name'], 'chunks':chunks,
                         'fresh_ids':fresh,'input_ids':ids, 'document_length':sum(map(len,chunks)),
                         'evidence':evidence,'source_context_count':len(ex['ctxs']),
                         'answers':record['answers'], 'decoded_input':tok.decode(ids)})
    manifest = {'paper':'https://arxiv.org/html/2405.16444v3', 'reference_commit':PIN,
                'model_revision':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
                'selection':'first 20 records each, no exclusions or answer-based filtering',
                'configs':CONFIGS,'records':records,'runs':runs,
                'reference_files':{p.name:sha(p) for p in sorted((HERE/'reference').iterdir())}}
    (HERE/'pilot.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'records':len(records),'runs':len(runs),'tokens_min':min(len(r['input_ids']) for r in runs),
                      'tokens_max':max(len(r['input_ids']) for r in runs),'sha256':sha(HERE/'pilot.json')}))

if __name__ == '__main__': main()
