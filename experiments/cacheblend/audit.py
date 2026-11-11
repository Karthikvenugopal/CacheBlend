"""Audit saved evidence for completeness, identities, mandatory fresh rows and scoring."""
import hashlib
import json
from pathlib import Path
from run import HERE, OUT, quality, save
from transformers import AutoTokenizer


def main():
    manifest=json.loads((HERE/'pilot.json').read_text())
    frozen={(r['id'],r['config']):r for r in manifest['runs']}
    assert len(frozen)==80
    tok=AutoTokenizer.from_pretrained(HERE.parents[1]/'.models/qwen3-1.7b',local_files_only=True)
    for r in frozen.values():
        ids=r['input_ids']
        assert sum(r['chunks'],[])+r['fresh_ids']==ids
        assert ids.count(tok.convert_tokens_to_ids('<|im_start|>'))==2
        assert ids.count(tok.convert_tokens_to_ids('<|im_end|>'))==1
    output={'pilot_sha256':hashlib.sha256((HERE/'pilot.json').read_bytes()).hexdigest(),
            'frozen_records':40,'frozen_pairs':len(frozen),'stages':{}}
    for stage,names in [('baseline',{'full','naive'}),('selection',{'zero','hkvd_16','hkvd_32'})]:
        path=OUT/f'{stage}.jsonl'
        if not path.exists(): continue
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        keys=[(r['id'],r['config']) for r in rows]
        assert len(set(keys))==len(keys) and set(keys)==set(frozen)
        for row in rows:
            r=frozen[(row['id'],row['config'])]
            assert row['input_sha256']==hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest()
            assert set(row['methods'])==names
            assert row['answers']==r['answers']
            d,n=r['document_length'],len(r['input_ids'])
            for method,result in row['methods'].items():
                scores=quality(result['answer'],r['answers'])
                assert all(result[k]==v for k,v in scores.items())
                for layer in result['work']:
                    assert layer['positions']==sorted(set(layer['positions']))
                    assert layer['positions'][-(n-d):]==list(range(d,n))
                if method=='full': assert result['endpoint_check']['passed']
                assert result['qkv_rows']==sum(x['qkv_rows'] for x in result['work'])
                assert result['attention_mlp_rows']==sum(x['attention_mlp_rows'] for x in result['work'])
        output['stages'][stage]={'pairs':len(rows),'methods_per_pair':len(names),'complete':True,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    output['passed']=len(output['stages'])==2
    save(OUT/'audit.json',output)
    print(json.dumps(output,indent=2))

if __name__=='__main__': main()
