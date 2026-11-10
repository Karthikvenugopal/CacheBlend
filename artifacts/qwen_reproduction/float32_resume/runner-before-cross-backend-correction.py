"""Authorized six-case float32 continuation; preserve the original BF16 run verbatim."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/qwen_reproduction/float32_resume'
spec = importlib.util.spec_from_file_location('prepared_reproduction', HERE/'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def hashes():
    return {**r.fingerprint(), 'resume_float32.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'float32-protocol.md': hashlib.sha256((HERE/'float32-protocol.md').read_bytes()).hexdigest()}


def metrics(a, b, correct_id, stale_id):
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    delta = (a-b).abs()
    return {'max_abs': float(delta.max()), 'relative_l2': float((a-b).norm()/a.norm()),
            'relative_linf': float(delta.max()/a.abs().max()),
            'within_tolerance': bool(torch.allclose(a, b, atol=.0002, rtol=.0002)),
            'top_agreement': int(a.argmax()) == int(b.argmax()),
            'top_ids': [int(a.argmax()), int(b.argmax())],
            'margins': [float(a[correct_id]-a[stale_id]), float(b[correct_id]-b[stale_id])],
            'restricted_choice_agreement': bool((a[correct_id]>=a[stale_id]) == (b[correct_id]>=b[stale_id]))}


def passed(m):
    return m['within_tolerance'] and m['top_agreement'] and m['restricted_choice_agreement']


def memory(device):
    if device == 'mps':
        torch.mps.synchronize()
        return {'allocated_bytes': torch.mps.current_allocated_memory(),
                'driver_bytes': torch.mps.driver_allocated_memory()}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['cpu_reference', 'restricted', 'generation'])
    stage = parser.parse_args().stage
    path = OUT/(stage+'.json')
    assert not path.exists(), 'Preserve prior results'
    frozen = json.loads((HERE/'prepared.json').read_text())
    assert frozen['files'] == r.fingerprint()
    current_hashes = hashes()
    cases = json.loads((HERE/'cases.json').read_text())
    assert len(cases) == 6
    prior = None
    reference = None
    if stage != 'cpu_reference':
        reference = json.loads((OUT/'cpu_reference.json').read_text())
        assert reference.get('completed') and reference['files'] == current_hashes
    if stage == 'generation':
        prior = json.loads((OUT/'restricted.json').read_text())
        assert prior.get('completed') and prior['files'] == current_hashes
    device = 'cpu' if stage == 'cpu_reference' else 'mps'
    torch.manual_seed(17)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    report = {'stage': stage, 'device': device, 'dtype': 'float32', 'attention': 'eager',
              'seed': 17, 'atol': .0002, 'rtol': .0002, 'files': current_hashes,
              'author_model_revision': 'unknown', 'candidate_model_revision': r.MODEL_REVISION,
              'thinking': False, 'max_new_tokens': r.MAX_NEW, 'generation_vocabulary': 'unrestricted',
              'records': [{'id': c['id'], 'status': 'not_run'} for c in cases]}
    r.save(path, report)
    start = perf_counter()
    try:
        model_path = ROOT/'.models/qwen3-1.7b'
        manifest = json.loads((model_path/'download-manifest.json').read_text())
        assert manifest['revision'] == r.MODEL_REVISION
        for name, info in manifest['files'].items():
            h = hashlib.sha256()
            with (model_path/name).open('rb') as f:
                for block in iter(lambda: f.read(8*1024*1024), b''):
                    h.update(block)
            assert h.hexdigest() == info['sha256'], name
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                    trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval()
        assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
        model.lm_head.weight = model.model.embed_tokens.weight
        model.to(device)
        report['loaded_memory'] = memory(device)
        for case in cases:
            item = next(x for x in report['records'] if x['id'] == case['id'])
            item['status'] = 'started'
            r.save(path, report)
            ci = tokenizer(case['correct'], add_special_tokens=False)['input_ids'][0]
            si = tokenizer(case['stale'], add_special_tokens=False)['input_ids'][0]
            if stage == 'cpu_reference':
                text = r.render(tokenizer, case['new_body'])
                assert text == case['expected_rendered_prompts']['new']
                _, ids, edit = r.aligned(tokenizer, r.render(tokenizer, case['old_body']), text)
                with torch.inference_mode():
                    inputs = torch.tensor([ids], device=device)
                    full = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
                    prefix = model(input_ids=inputs[:, :-1], use_cache=True, logits_to_keep=1)
                    assert prefix.past_key_values.get_seq_length() == len(ids)-1
                    native = model(input_ids=inputs[:, -1:], past_key_values=prefix.past_key_values,
                                   use_cache=True, logits_to_keep=1)
                check = metrics(full.logits, native.logits, ci, si)
                item.update(new_ids=ids, edit=edit, endpoint=check,
                            full_logits=full.logits.flatten().float().cpu().tolist())
                assert passed(check), check
                del inputs, full, prefix, native
            else:
                meta, branches = r.states(model, tokenizer, case)
                item.update(meta)
                ref = next(x for x in reference['records'] if x['id'] == case['id'])
                assert meta['new_ids'] == ref['new_ids']
                # These logits use the same last-token replay convention as published scoring.
                full_score = r.last_step(model, branches['full'][0], branches['full'][1]).logits
                check = metrics(torch.tensor(ref['full_logits']), full_score, ci, si)
                item['cpu_reference_vs_mps_full_replay'] = check
                del full_score
                assert passed(check), check
                item['outcomes'] = {}
                if stage == 'restricted':
                    err = r.forward(model, meta['erratum_ids']).past_key_values
                    branches['erratum'] = (err, meta['erratum_ids'])
                for name, (cache, ids) in branches.items():
                    score = r.restricted(r.last_step(model, cache, ids).logits, tokenizer,
                                         case['correct'], case['stale'])
                    if stage == 'restricted':
                        score['published_choice'] = case['published'].get(name)
                        score['matches_published'] = (score['choice'] == score['published_choice']
                                                      if score['published_choice'] else None)
                        item['outcomes'][name] = score
                    else:
                        saved = next(x for x in prior['records'] if x['id'] == case['id'])
                        assert meta['old_ids'] == saved['old_ids'] and meta['new_ids'] == saved['new_ids']
                        assert score['choice'] == saved['outcomes'][name]['choice']
                        item['outcomes'][name] = {'restricted': score,
                                                 **r.generate(model, tokenizer, cache, ids)}
                    r.save(path, report)
                item['memory'] = memory(device)
                del branches, cache
            item['status'] = 'completed'
            r.save(path, report)
            print(stage, case['id'], 'saved', flush=True)
        report['completed'] = True
        report['elapsed_seconds_including_loading'] = perf_counter()-start
        r.save(path, report)
    except Exception as exc:
        report.update(completed=False, stopped_reason=type(exc).__name__+': '+str(exc))
        active = next((x for x in report['records'] if x['status'] == 'started'), None)
        if active is not None:
            active.update(status='failed', error=report['stopped_reason'])
        r.save(path, report)
        raise


if __name__ == '__main__':
    main()
